//! The standard library: the Yuzu modules under `stdlib/`, built into the
//! compiler so that no files have to be installed beside it.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::OnceLock;

use melior::Context;

use yuzu_ast::{AstNode, ast};
use yuzu_diagnostics::diagnostics::engine::DiagnosticsEngine;
use yuzu_diagnostics::source_map::SourceMap;
use yuzu_lexer::lexer::{Lexer, Token};
use yuzu_passes::{BoundLibrary, File};
use yuzu_syntax::{GreenNode, SyntaxNode};

use crate::modules::{self, MapResolver, ModuleSource, Submodule};

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
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
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

/// The library as every compile on the thread starts from it: its files
/// loaded and what each of its modules declares.
pub(crate) struct Library {
    pub(crate) sources: SourceMap,
    pub(crate) files: Vec<File>,
    pub(crate) submodules: HashMap<String, Vec<Submodule>>,
}

thread_local! {
    static LIBRARIES: RefCell<HashMap<Engine, Rc<Library>>> = RefCell::new(HashMap::new());
}

/// The library for an engine, loaded once for the thread. A syntax tree
/// cannot cross threads, so each thread loads its own.
pub(crate) fn library(engine: Engine) -> Rc<Library> {
    LIBRARIES.with(|libraries| {
        libraries
            .borrow_mut()
            .entry(engine)
            .or_insert_with(|| Rc::new(load_library(engine)))
            .clone()
    })
}

/// The library's names for an engine, bound once for the process.
pub(crate) fn bound_library(engine: Engine) -> &'static BoundLibrary<'static> {
    static DATAFUSION: OnceLock<BoundLibrary<'static>> = OnceLock::new();
    let bound = match engine {
        Engine::DataFusion => &DATAFUSION,
    };
    bound.get_or_init(|| bind_library(engine))
}

/// Loads the library's files for an engine.
fn load_library(engine: Engine) -> Library {
    let mut sources = SourceMap::new();
    let mut diagnostics = DiagnosticsEngine::new();
    let entry = sources.add("<library>".to_string(), String::new());
    let (files, mut submodules) = modules::load_with(
        entry,
        &mut sources,
        &mut diagnostics,
        &MapResolver(HashMap::new()),
        engine,
        None,
    )
    .expect("the library loads");
    assert!(
        diagnostics.diagnostics().is_empty(),
        "the library loads without diagnostics"
    );
    submodules.remove("");
    let files: Vec<File> = files
        .into_iter()
        .filter(|file| file.module().is_some())
        .collect();

    Library {
        sources,
        files,
        submodules,
    }
}

/// Binds the library's names for an engine. The names live in a context
/// made for them and never dropped: one for each engine, for the process.
fn bind_library(engine: Engine) -> BoundLibrary<'static> {
    let Library { sources, files, .. } = load_library(engine);
    let mut diagnostics = DiagnosticsEngine::new();
    let context: &'static Context = Box::leak(Box::new(yuzu_mlir::context()));
    let bound = yuzu_passes::bind_library(
        context,
        &sources,
        &files,
        &mut diagnostics,
        &yuzu_types::Builtins,
    );
    assert!(
        diagnostics.diagnostics().is_empty(),
        "the library binds without diagnostics"
    );
    bound
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
    use std::collections::HashMap;

    use melior::ir::operation::OperationLike;
    use yuzu_diagnostics::diagnostics::engine::DiagnosticsEngine;
    use yuzu_diagnostics::source_map::SourceMap;
    use yuzu_passes::Lowering;

    use super::{Engine, MODULES, syntax};
    use crate::modules::{self, MapResolver};

    #[test]
    fn each_library_file_parses() {
        for module in MODULES {
            assert!(syntax(module.path).is_some(), "{}", module.name);
        }
    }

    #[test]
    fn the_whole_library_lowers_without_diagnostics() {
        let imports: String = MODULES
            .iter()
            .map(|module| format!("import {}\n", module.path))
            .collect();
        let mut sources = SourceMap::new();
        let mut diagnostics = DiagnosticsEngine::new();
        let entry = sources.add("main.yz".to_string(), imports);
        let resolver = MapResolver(HashMap::new());
        let mut files = modules::load(
            entry,
            &mut sources,
            &mut diagnostics,
            &resolver,
            Engine::DataFusion,
        )
        .expect("the library loads");
        for file in &mut files {
            file.set_lowering(Lowering::Eager);
        }

        let context = yuzu_mlir::context();
        let module = yuzu_passes::lower_ast_to_yzl(
            &context,
            &sources,
            &files,
            &mut diagnostics,
            &yuzu_types::Builtins,
            None,
        );
        let messages: Vec<&str> = diagnostics
            .diagnostics()
            .iter()
            .map(|diagnostic| diagnostic.message.as_str())
            .collect();
        assert!(messages.is_empty(), "{messages:?}");
        assert!(module.as_operation().verify());
    }
}
