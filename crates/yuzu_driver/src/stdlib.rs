//! The standard library: the Yuzu modules under `stdlib/`, built into the
//! compiler so that no files have to be installed beside it.

use std::collections::HashMap;
use std::sync::OnceLock;

use yuzu_ast::{AstNode, ast};
use yuzu_diagnostics::diagnostics::engine::DiagnosticsEngine;
use yuzu_diagnostics::source_map::SourceMap;
use yuzu_lexer::lexer::{Lexer, Token};
use yuzu_syntax::{GreenNode, SyntaxNode};

use crate::modules::ModuleSource;

/// One library file: its module path, the name a diagnostic shows for it,
/// and its text.
struct Embedded {
    path: &'static str,
    name: &'static str,
    source: &'static str,
}

include!(concat!(env!("OUT_DIR"), "/stdlib.rs"));

/// The first segment of every library path.
const ROOT: &str = "yuzu";

/// The module the compiler writes for the engine being compiled for.
const ENGINE_MODULE: &str = "yuzu.engine";

/// The engine a program is compiled for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Engine {
    DataFusion,
}

impl Engine {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "datafusion" => Some(Engine::DataFusion),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Engine::DataFusion => "datafusion",
        }
    }
}

/// Whether a path belongs to the library. A program's own module cannot
/// take such a path, so it cannot stand in for a library module.
pub(crate) fn reserves(path: &str) -> bool {
    path.split('.').next() == Some(ROOT)
}

/// `yuzu.engine` is written for the engine; every other path is a file.
pub(crate) fn resolve(path: &str, engine: Engine) -> Option<ModuleSource> {
    if path == ENGINE_MODULE {
        return Some(ModuleSource {
            name: format!("<{ENGINE_MODULE}>"),
            source: format!("pub let ENGINE = \"{}\"\n", engine.name()),
        });
    }

    MODULES
        .iter()
        .find(|module| module.path == path)
        .map(|module| ModuleSource {
            name: module.name.to_string(),
            source: module.source.to_string(),
        })
}

/// A library file's syntax tree. Each file is parsed once for the process;
/// every compile after the first starts from the same tree.
pub(crate) fn syntax(path: &str) -> Option<ast::Root> {
    let green = trees().get(path)?.clone();
    ast::Root::cast(SyntaxNode::new_root(green))
}

fn trees() -> &'static HashMap<&'static str, GreenNode> {
    static TREES: OnceLock<HashMap<&'static str, GreenNode>> = OnceLock::new();
    TREES.get_or_init(|| {
        MODULES
            .iter()
            .map(|module| {
                let mut sources = SourceMap::new();
                let source_id = sources.add(module.name.to_string(), module.source.to_string());
                let mut diagnostics = DiagnosticsEngine::new();
                let tokens: Vec<Token> = Lexer::new(module.source).collect();
                let syntax = yuzu_parser::parse(&tokens, &mut diagnostics, source_id);
                assert!(
                    diagnostics.diagnostics().is_empty(),
                    "the library file `{}` has syntax errors",
                    module.name
                );
                (module.path, syntax.green().into_owned())
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::{MODULES, syntax};

    #[test]
    fn each_library_file_parses() {
        for module in MODULES {
            assert!(syntax(module.path).is_some(), "{}", module.name);
        }
    }
}
