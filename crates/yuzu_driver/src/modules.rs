//! Turning a program's imports into the files it is made of.
//!
//! A module is named by a dotted path and read by a [`ModuleResolver`], so
//! the filesystem is one answer and a test or an embedding supplies its own.
//! Loading walks the imports depth first, which puts every module ahead of
//! whatever imports it and leaves the entry file last, in the order the
//! conversion wants.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use yuzu_ast::AstNode;
use yuzu_ast::ast;
use yuzu_diagnostics::diagnostics::Span;
use yuzu_diagnostics::diagnostics::builder::DiagnosticBuilder;
use yuzu_diagnostics::diagnostics::engine::DiagnosticsEngine;
use yuzu_diagnostics::source_map::{SourceId, SourceMap};
use yuzu_lexer::lexer::{Lexer, Token};
use yuzu_passes::File;

/// A module's source, and the name to show for it in a diagnostic.
pub struct ModuleSource {
    pub name: String,
    pub source: String,
}

/// Where a module's source comes from. `None` means the resolver has no
/// module under that path, which the loader reports against the import that
/// asked for it.
pub trait ModuleResolver {
    fn resolve(&self, path: &str) -> Option<ModuleSource>;
}

/// Modules beside the entry file. A path's segments are directories, so
/// `yuzu.std.math` is `yuzu/std/math.yz` under the base.
pub struct FsResolver {
    pub base: PathBuf,
}

impl ModuleResolver for FsResolver {
    fn resolve(&self, path: &str) -> Option<ModuleSource> {
        let mut file = self.base.clone();
        for segment in path.split('.') {
            file.push(segment);
        }

        file.set_extension("yz");
        let source = std::fs::read_to_string(&file).ok()?;
        Some(ModuleSource {
            name: file.display().to_string(),
            source,
        })
    }
}

/// Modules held in memory, for tests and for embeddings that have no files.
pub struct MapResolver(pub HashMap<String, String>);

impl ModuleResolver for MapResolver {
    fn resolve(&self, path: &str) -> Option<ModuleSource> {
        self.0.get(path).map(|source| ModuleSource {
            name: format!("{path}.yz"),
            source: source.clone(),
        })
    }
}

/// The entry file and every module it reaches, in the order the conversion
/// wants them. `None` once anything has been reported.
///
/// Each file is parsed here rather than by the conversion, because finding
/// what a file imports means parsing it, and parsing twice would report
/// every syntax error twice.
pub fn load(
    entry: SourceId,
    sources: &mut SourceMap,
    diagnostics: &mut DiagnosticsEngine,
    resolver: &dyn ModuleResolver,
) -> Option<Vec<File>> {
    let mut loader = Loader {
        sources,
        diagnostics,
        resolver,
        loaded: HashSet::new(),
        loading: Vec::new(),
        files: Vec::new(),
    };

    let root = loader.parse(entry);
    loader.follow_imports(entry, &root);
    loader.files.push(File::entry(entry, root));
    (!has_errors(loader.diagnostics)).then_some(loader.files)
}

struct Loader<'a> {
    sources: &'a mut SourceMap,
    diagnostics: &'a mut DiagnosticsEngine,
    resolver: &'a dyn ModuleResolver,
    loaded: HashSet<String>,
    /// The modules being loaded, outermost first: the chain to show when one
    /// of them turns out to import something already on it.
    loading: Vec<String>,
    files: Vec<File>,
}

impl Loader<'_> {
    /// One module, and everything it imports before it. A module already
    /// loaded is left alone, so two files importing the same one read it
    /// once.
    fn load_module(&mut self, path: &str, at: &impl AstNode, asked_by: SourceId) {
        if self.loaded.contains(path) {
            return;
        }

        if let Some(from) = self.loading.iter().position(|seen| seen == path) {
            let chain = self.loading[from..]
                .iter()
                .map(String::as_str)
                .chain([path])
                .collect::<Vec<_>>()
                .join(" imports ");
            self.report(at, asked_by, &format!("circular import: {chain}"));
            return;
        }

        let Some(module) = self.resolver.resolve(path) else {
            self.report(at, asked_by, &format!("cannot find module `{path}`"));
            return;
        };

        let source_id = self.sources.add(module.name, module.source);
        let root = self.parse(source_id);
        self.loading.push(path.to_string());
        self.follow_imports(source_id, &root);
        self.loading.pop();

        self.loaded.insert(path.to_string());
        self.files.push(File {
            source_id,
            module: Some(path.to_string()),
            root,
        });
    }

    fn follow_imports(&mut self, source_id: SourceId, root: &ast::Root) {
        for stmt in root.stmts() {
            let path = match &stmt {
                ast::Stmt::ImportStmt(import) => import.path(),
                ast::Stmt::FromImportStmt(import) => import.path(),
                _ => continue,
            };

            let Some(path) = path.map(|path| dotted(&path)) else {
                continue;
            };

            self.load_module(&path, &stmt, source_id);
        }
    }

    fn parse(&mut self, source_id: SourceId) -> ast::Root {
        let tokens: Vec<Token> = Lexer::new(self.sources.text(source_id)).collect();
        let syntax = yuzu_parser::parse(&tokens, self.diagnostics, source_id);
        ast::Root::cast(syntax).expect("a parse always yields a root")
    }

    fn report(&mut self, at: &impl AstNode, source_id: SourceId, message: &str) {
        let span = Span {
            source_id,
            range: at.syntax().text_range(),
        };
        self.diagnostics
            .emit(DiagnosticBuilder::error(span, message.to_string()));
    }
}

/// A path as the program wrote it, for resolving and for reporting.
fn dotted(path: &ast::ModulePath) -> String {
    path.segments()
        .filter_map(|segment| segment.text())
        .collect::<Vec<_>>()
        .join(".")
}

fn has_errors(diagnostics: &DiagnosticsEngine) -> bool {
    diagnostics
        .diagnostics()
        .iter()
        .any(|diagnostic| diagnostic.severity == yuzu_diagnostics::diagnostics::Severity::Error)
}

/// The directory a file's modules are resolved against.
pub fn base_of(file: &Path) -> PathBuf {
    file.parent().unwrap_or(Path::new(".")).to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolver(modules: &[(&str, &str)]) -> MapResolver {
        MapResolver(
            modules
                .iter()
                .map(|(path, source)| (path.to_string(), source.to_string()))
                .collect(),
        )
    }

    /// Loads a program and says which modules came out, in order.
    fn loaded(entry: &str, modules: &[(&str, &str)]) -> Result<Vec<String>, Vec<String>> {
        let mut sources = SourceMap::new();
        let mut diagnostics = DiagnosticsEngine::new();
        let id = sources.add("main.yz".to_string(), entry.to_string());
        match load(id, &mut sources, &mut diagnostics, &resolver(modules)) {
            Some(files) => Ok(files
                .iter()
                .map(|file| file.module.clone().unwrap_or_else(|| "<entry>".to_string()))
                .collect()),
            None => Err(diagnostics
                .diagnostics()
                .iter()
                .map(|diagnostic| diagnostic.message.clone())
                .collect()),
        }
    }

    /// Depth first, so a module comes out ahead of whatever imports it and
    /// the entry file comes out last.
    #[test]
    fn a_module_is_loaded_before_the_file_importing_it() {
        assert_eq!(
            loaded(
                "import helpers\n",
                &[("helpers", "def one() -> int64 { return 1 }\n")]
            ),
            Ok(vec!["helpers".to_string(), "<entry>".to_string()])
        );
    }

    /// Two files importing the same module read it once, and it still comes
    /// out ahead of both.
    #[test]
    fn a_module_reached_twice_is_loaded_once() {
        assert_eq!(
            loaded(
                "import one\nimport two\n",
                &[
                    ("one", "import shared\n"),
                    ("two", "import shared\n"),
                    ("shared", "def one() -> int64 { return 1 }\n"),
                ],
            ),
            Ok(vec![
                "shared".to_string(),
                "one".to_string(),
                "two".to_string(),
                "<entry>".to_string(),
            ])
        );
    }

    /// A path's segments are the modules holding it.
    #[test]
    fn a_path_names_a_module_through_the_ones_holding_it() {
        assert_eq!(
            loaded(
                "from yuzu.std.math import clamp\n",
                &[("yuzu.std.math", "def clamp() -> int64 { return 1 }\n")],
            ),
            Ok(vec!["yuzu.std.math".to_string(), "<entry>".to_string()])
        );
    }

    /// The chain is what makes a cycle readable: which module imported which
    /// to get back to where it started.
    #[test]
    fn a_cycle_reports_the_chain() {
        assert_eq!(
            loaded("import a\n", &[("a", "import b\n"), ("b", "import a\n")],),
            Err(vec!["circular import: a imports b imports a".to_string()])
        );
    }

    #[test]
    fn a_module_no_resolver_has_is_reported() {
        assert_eq!(
            loaded("import nowhere\n", &[]),
            Err(vec!["cannot find module `nowhere`".to_string()])
        );
    }
}
