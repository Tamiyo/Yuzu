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
use yuzu_passes::{File, Lowering};
use yuzu_syntax::{GreenNode, SyntaxNode};

use crate::stdlib::{self, Engine};

/// A submodule a module declares: its name, and whether anyone outside may
/// reach it.
#[derive(Clone)]
pub(crate) struct Submodule {
    name: String,
    public: bool,
}

/// A module's source, and the name to show for it in a diagnostic.
#[derive(Debug)]
pub struct ModuleSource {
    pub name: String,
    pub source: String,
    /// A tree the resolver parsed from `source` before, without errors, so
    /// the loader does not parse it again. With `None` the loader parses.
    pub syntax: Option<GreenNode>,
}

/// Where a module's source comes from. `None` means the resolver has no
/// module under that path, which the loader reports against the import that
/// asked for it.
pub trait ModuleResolver {
    fn resolve(&self, path: &str) -> Option<ModuleSource>;

    /// A library module's source when the caller holds its own copy, as an
    /// editor does for a library file it has open. `None` reads the copy
    /// built into the compiler.
    fn resolve_library(&self, _path: &str) -> Option<ModuleSource> {
        None
    }
}

/// The file that marks a directory as a module and declares what it holds.
pub const MARKER: &str = "mod.yz";

/// Modules beside the entry file. A path's segments are directories, so
/// `yuzu.std.math` is `yuzu/std/math.yz`, or `yuzu/std/math/mod.yz` when it
/// holds submodules of its own.
#[derive(Debug)]
pub struct FsResolver {
    pub base: PathBuf,
}

impl FsResolver {
    /// The files that can hold a module, in the order they are tried.
    #[must_use]
    pub fn candidates(&self, path: &str) -> [PathBuf; 2] {
        let mut directory = self.base.clone();
        for segment in path.split('.') {
            directory.push(segment);
        }

        // A leaf file first: a directory and a file of the same name are
        // two modules under one path, and the file is the one written on
        // purpose.
        [directory.with_extension("yz"), directory.join(MARKER)]
    }
}

impl ModuleResolver for FsResolver {
    fn resolve(&self, path: &str) -> Option<ModuleSource> {
        let file = self
            .candidates(path)
            .into_iter()
            .find(|file| file.is_file())?;
        let source = std::fs::read_to_string(&file).ok()?;
        Some(ModuleSource {
            name: file.display().to_string(),
            source,
            syntax: None,
        })
    }
}

/// Modules held in memory, for tests and for embeddings that have no files.
#[derive(Debug)]
pub struct MapResolver(pub HashMap<String, String>);

impl ModuleResolver for MapResolver {
    fn resolve(&self, path: &str) -> Option<ModuleSource> {
        self.0.get(path).map(|source| ModuleSource {
            name: format!("{path}.yz"),
            source: source.clone(),
            syntax: None,
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
    engine: Engine,
) -> Option<Vec<File>> {
    // The cached library's sources follow its own entry, so they keep their
    // ids only in a map that holds nothing but this program's entry.
    let library = (sources.len() == 1).then(|| stdlib::Library::for_thread(engine));
    let Loaded { files, .. } = load_program(
        EntryFile {
            source: entry,
            syntax: None,
        },
        sources,
        diagnostics,
        resolver,
        engine,
        library.as_deref(),
        None,
    );
    (!has_errors(diagnostics)).then_some(files)
}

/// The file a program starts from, and its tree when a caller parsed its
/// text before without errors.
pub(crate) struct EntryFile {
    pub(crate) source: SourceId,
    pub(crate) syntax: Option<GreenNode>,
}

/// What loading gave.
pub(crate) struct Loaded {
    pub(crate) files: Vec<File>,
    /// Each source's tree, by the source's id.
    pub(crate) trees: Vec<(SourceId, GreenNode)>,
    /// The submodules each module declares, by the module's path.
    pub(crate) submodules: HashMap<String, Vec<Submodule>>,
}

/// Loads the whole program: the library, then the focus module if there is
/// one, then the entry file and what it imports. The library comes from
/// `library` when it is given, and from its files when it is not, which is
/// how the cached library is made.
///
/// `focus` is a module loaded whether or not anything imports it, and
/// lowered in full: the one a check asks about.
///
/// Every file that could be read comes back, even when loading reported:
/// a file with a syntax error still lowers, with a hole where the error is.
pub(crate) fn load_program(
    entry: EntryFile,
    sources: &mut SourceMap,
    diagnostics: &mut DiagnosticsEngine,
    resolver: &dyn ModuleResolver,
    engine: Engine,
    library: Option<&stdlib::Library>,
    focus: Option<&str>,
) -> Loaded {
    let mut loader = Loader {
        sources,
        diagnostics,
        resolver,
        engine,
        loaded: HashSet::new(),
        submodules: HashMap::new(),
        loading: Vec::new(),
        files: Vec::new(),
        trees: Vec::new(),
    };

    let EntryFile {
        source: entry,
        syntax,
    } = entry;
    let root = loader.read_syntax(entry, syntax);
    loader.submodules.insert(String::new(), submodules(&root));
    match library {
        Some(library) => loader.install(library),
        None => loader.load_path(yuzu_passes::PRELUDE, &root, entry, ""),
    }

    if let Some(path) = focus {
        loader.load_module(path, &root, entry);
    }

    // The entry file belongs to no module, so nothing keeps anything from
    // it beyond what `pub` already governs.
    loader.follow_imports(entry, &root, "");
    loader.files.push(File::entry(entry, root));
    if let Some(path) = focus
        && let Some(file) = loader
            .files
            .iter_mut()
            .find(|file| file.module() == Some(path))
    {
        file.set_lowering(Lowering::Eager);
    }
    Loaded {
        files: loader.files,
        submodules: loader.submodules,
        trees: loader.trees,
    }
}

struct Loader<'a> {
    sources: &'a mut SourceMap,
    diagnostics: &'a mut DiagnosticsEngine,
    resolver: &'a dyn ModuleResolver,
    engine: Engine,
    loaded: HashSet<String>,
    /// The submodules each loaded module declares, which is what makes a
    /// path through it resolvable.
    submodules: HashMap<String, Vec<Submodule>>,
    /// The modules being loaded, outermost first: the chain to show when one
    /// of them turns out to import something already on it.
    loading: Vec<String>,
    files: Vec<File>,
    /// Each source's tree, as the files hold it.
    trees: Vec<(SourceId, GreenNode)>,
}

impl Loader<'_> {
    /// Puts a cached load of the library in place: its sources, its files
    /// and what each of its modules declares.
    fn install(&mut self, library: &stdlib::Library) {
        self.sources.extend_from(&library.sources);
        for file in &library.files {
            let path = file.module().expect("a library file is a module");
            self.loaded.insert(path.to_string());
            self.files.push(file.clone());
        }

        for (path, declared) in &library.submodules {
            self.submodules.insert(path.clone(), declared.clone());
        }
    }

    /// A path, one segment at a time. Each module declares the ones it
    /// holds, so reaching `std.math` means loading `std` and asking whether
    /// it declares `math`. A file nobody declares is not part of the program
    /// and cannot be reached by naming it.
    fn load_path(&mut self, path: &str, at: &impl AstNode, asked_by: SourceId, asking: &str) {
        let segments: Vec<&str> = path.split('.').collect();
        let mut reached = String::new();
        for (index, segment) in segments.iter().enumerate() {
            let parent = reached.clone();
            if !reached.is_empty() {
                reached.push('.');
            }

            reached.push_str(segment);
            if !parent.is_empty() {
                match self.declares(&parent, segment, asking) {
                    Reach::Yes => {}
                    Reach::Undeclared => {
                        self.report(
                            at,
                            asked_by,
                            &format!("`{parent}` does not declare a module `{segment}`"),
                        );
                        return;
                    }
                    Reach::Private => {
                        self.report(
                            at,
                            asked_by,
                            &format!("`{parent}` keeps `{segment}` to itself"),
                        );
                        return;
                    }
                }
            }

            // A module partway through loading is only passed over when the
            // path continues through it, which is a module reaching into
            // itself: it recorded what it declares before following its own
            // imports, so the check above already had its answer. Ending on
            // one is a genuine cycle, and `load_module` says so.
            if index + 1 < segments.len() && self.loading.iter().any(|loading| loading == &reached)
            {
                continue;
            }

            self.load_module(&reached, at, asked_by);
            if !self.loaded.contains(&reached) {
                return;
            }
        }
    }

    /// Whether a module declares this submodule, and whether the file asking
    /// may name it. A module keeps a submodule to itself unless it says
    /// `pub`, and the files of that module are the ones it is kept for.
    fn declares(&self, module: &str, name: &str, asking: &str) -> Reach {
        let Some(declared) = self
            .submodules
            .get(module)
            .and_then(|names| names.iter().find(|declared| declared.name == name))
        else {
            return Reach::Undeclared;
        };

        if declared.public || asking == module || asking.starts_with(&format!("{module}.")) {
            Reach::Yes
        } else {
            Reach::Private
        }
    }

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

        let module = if stdlib::reserves(path) {
            self.resolver
                .resolve_library(path)
                .or_else(|| stdlib::resolve(path, self.engine))
        } else {
            self.resolver.resolve(path)
        };

        let Some(module) = module else {
            self.report(at, asked_by, &format!("cannot find module `{path}`"));
            return;
        };

        let source_id = self.sources.add(module.name, module.source);
        let root = self.read_syntax(source_id, module.syntax);
        self.submodules.insert(path.to_string(), submodules(&root));
        self.loading.push(path.to_string());
        self.follow_imports(source_id, &root, path);
        self.loading.pop();

        self.loaded.insert(path.to_string());
        let mut file = File::new(source_id, Some(path.to_string()), root);
        if stdlib::reserves(path) {
            file.set_lowering(Lowering::OnDemand);
        }

        self.files.push(file);
    }

    fn follow_imports(&mut self, source_id: SourceId, root: &ast::Root, asking: &str) {
        for stmt in root.stmts() {
            let path = match &stmt {
                ast::Stmt::ImportStmt(import) => import.path(),
                ast::Stmt::FromImportStmt(import) => import.path(),
                _ => continue,
            };

            let Some(path) = path.map(|path| path.to_dotted()) else {
                continue;
            };

            self.load_path(&path, &stmt, source_id, asking);
        }
    }

    /// A source's tree: the one a resolver parsed before, or a parse of the
    /// text now, whose errors go to the diagnostics.
    fn read_syntax(&mut self, source_id: SourceId, syntax: Option<GreenNode>) -> ast::Root {
        let tree = match syntax {
            Some(green) => {
                debug_assert_eq!(
                    usize::from(green.text_len()),
                    self.sources.text(source_id).len(),
                    "a tree given for a source was parsed from its text"
                );
                SyntaxNode::new_root(green)
            }
            None => {
                yuzu_parser::parse_text(self.sources.text(source_id), self.diagnostics, source_id)
            }
        };
        self.trees.push((source_id, tree.green().into_owned()));
        ast::Root::cast(tree).expect("a parse always yields a root")
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

/// The submodules a file declares.
fn submodules(root: &ast::Root) -> Vec<Submodule> {
    root.stmts()
        .filter_map(|stmt| match stmt {
            ast::Stmt::ModStmt(decl) => Some(Submodule {
                name: decl.name()?.token()?.text().to_owned(),
                public: decl.visibility() == yuzu_ast::Visibility::Public,
            }),
            _ => None,
        })
        .collect()
}

/// Whether a file may name a submodule of another module.
enum Reach {
    Yes,
    Undeclared,
    Private,
}

fn has_errors(diagnostics: &DiagnosticsEngine) -> bool {
    diagnostics
        .diagnostics()
        .iter()
        .any(|diagnostic| diagnostic.severity == yuzu_diagnostics::diagnostics::Severity::Error)
}

/// Where a file sits in its program: the start of one, or a module of one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Location {
    /// A file in a directory without a module marker: a program's start.
    Entry { base: PathBuf },
    /// A file in a directory tree of module markers, and the path that
    /// names it from the top of that tree.
    Module { base: PathBuf, path: String },
}

/// Where a file sits, read from the module markers around it: up through
/// the directories that hold one, the first that does not is where the
/// program's module paths start. The reverse of [`FsResolver::candidates`].
#[must_use]
pub fn locate(file: &Path) -> Location {
    let directory = file.parent().unwrap_or(Path::new(""));
    let is_marker = file.file_name().is_some_and(|name| name == MARKER);
    if !is_marker && !directory.join(MARKER).is_file() {
        return Location::Entry {
            base: directory.to_path_buf(),
        };
    }

    let mut segments = Vec::new();
    if !is_marker && let Some(stem) = file.file_stem() {
        segments.push(stem.to_string_lossy().into_owned());
    }
    let mut base = directory;
    while base.join(MARKER).is_file() {
        let Some(name) = base.file_name() else {
            break;
        };
        segments.push(name.to_string_lossy().into_owned());
        base = base.parent().unwrap_or(Path::new(""));
    }
    segments.reverse();

    Location::Module {
        base: base.to_path_buf(),
        path: segments.join("."),
    }
}

/// The directory a file's modules are resolved against.
#[must_use]
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

    /// Loads a program and says which of its own modules came out, in order.
    fn loaded(entry: &str, modules: &[(&str, &str)]) -> Result<Vec<String>, Vec<String>> {
        loaded_with_library(entry, modules).map(|paths| {
            paths
                .into_iter()
                .filter(|path| !stdlib::reserves(path))
                .collect()
        })
    }

    /// Every module that came out, the library's included.
    fn loaded_with_library(
        entry: &str,
        modules: &[(&str, &str)],
    ) -> Result<Vec<String>, Vec<String>> {
        let mut sources = SourceMap::new();
        let mut diagnostics = DiagnosticsEngine::new();
        let id = sources.add("main.yz".to_string(), entry.to_string());
        match load(
            id,
            &mut sources,
            &mut diagnostics,
            &resolver(modules),
            Engine::DataFusion,
        ) {
            Some(files) => Ok(files
                .iter()
                .map(|file| file.module().unwrap_or("<entry>").to_string())
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
                "from lib.std.math import clamp\n",
                &[
                    ("lib", "pub mod std\n"),
                    ("lib.std", "pub mod math\n"),
                    ("lib.std.math", "pub def clamp() -> int64 { return 1 }\n"),
                ],
            ),
            Ok(vec![
                "lib".to_string(),
                "lib.std".to_string(),
                "lib.std.math".to_string(),
                "<entry>".to_string(),
            ])
        );
    }

    /// The chain is what makes a cycle readable: which module imported which
    /// to get back to where it started.
    /// A package declares what it holds, so a file nobody declared is not
    /// part of the program even though it sits in the directory.
    #[test]
    fn a_file_the_package_never_declared_is_not_reachable() {
        assert_eq!(
            loaded(
                "from std.stray import loose\n",
                &[
                    ("std", "pub mod math\n"),
                    ("std.stray", "pub def loose() -> int64 { return 1 }\n")
                ],
            ),
            Err(vec!["`std` does not declare a module `stray`".to_string()])
        );
    }

    /// A submodule is the package's own unless it says `pub`, which is what
    /// lets a package hold an implementation and offer a surface over it.
    #[test]
    fn a_private_submodule_is_not_reachable_from_outside() {
        assert_eq!(
            loaded(
                "from std.internal import clamp\n",
                &[
                    ("std", "mod internal\n"),
                    ("std.internal", "pub def clamp() -> int64 { return 1 }\n"),
                ],
            ),
            Err(vec!["`std` keeps `internal` to itself".to_string()])
        );
    }

    /// The package itself reaches its own private submodule, which is the
    /// whole point of one.
    #[test]
    fn a_package_reaches_its_own_private_submodule() {
        assert_eq!(
            loaded(
                "from std import clamp\n",
                &[
                    ("std", "mod internal\nfrom std.internal import clamp\n"),
                    ("std.internal", "pub def clamp() -> int64 { return 1 }\n"),
                ],
            ),
            Ok(vec![
                "std.internal".to_string(),
                "std".to_string(),
                "<entry>".to_string(),
            ])
        );
    }

    #[test]
    fn a_cycle_reports_the_chain() {
        assert_eq!(
            loaded("import a\n", &[("a", "import b\n"), ("b", "import a\n")],),
            Err(vec!["circular import: a imports b imports a".to_string()])
        );
    }

    #[test]
    fn a_library_path_is_read_from_the_library() {
        let paths = loaded_with_library("import yuzu.std\n", &[]).expect("the library loads");
        assert!(paths.iter().any(|path| path == "yuzu.std"), "{paths:?}");
    }

    #[test]
    fn the_engine_module_is_written_for_the_engine() {
        let paths = loaded_with_library("from yuzu.engine import ENGINE\n", &[])
            .expect("the library loads");
        assert!(paths.iter().any(|path| path == "yuzu.engine"), "{paths:?}");
    }

    #[test]
    fn a_program_module_cannot_take_a_library_path() {
        assert_eq!(
            loaded(
                "import yuzu.extra\n",
                &[("yuzu.extra", "pub def one() -> int64 { return 1 }\n")],
            ),
            Err(vec!["`yuzu` does not declare a module `extra`".to_string()])
        );
    }

    #[test]
    fn a_module_no_resolver_has_is_reported() {
        assert_eq!(
            loaded("import nowhere\n", &[]),
            Err(vec!["cannot find module `nowhere`".to_string()])
        );
    }

    #[test]
    fn a_file_is_located_by_the_markers_around_it() {
        let root = std::env::temp_dir().join(format!("yuzu-locate-{}", std::process::id()));
        for file in [
            "main.yz",
            "app/mod.yz",
            "app/util.yz",
            "app/sub/mod.yz",
            "app/sub/deep.yz",
        ] {
            let file = root.join(file);
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
            std::fs::write(file, "").unwrap();
        }

        let located: Vec<String> = ["main.yz", "app/util.yz", "app/mod.yz", "app/sub/deep.yz"]
            .iter()
            .map(|file| match locate(&root.join(file)) {
                Location::Entry { base } => format!("{file}: entry at {}", base == root),
                Location::Module { base, path } => {
                    format!("{file}: module {path} at {}", base == root)
                }
            })
            .collect();
        std::fs::remove_dir_all(&root).unwrap();

        assert_eq!(
            located,
            [
                "main.yz: entry at true",
                "app/util.yz: module app.util at true",
                "app/mod.yz: module app at true",
                "app/sub/deep.yz: module app.sub.deep at true",
            ]
        );
    }
}
