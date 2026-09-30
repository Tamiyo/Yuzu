//! The standard library: the Yuzu modules under `stdlib/`, built into the
//! compiler so that no files have to be installed beside it.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::str::FromStr;
use std::sync::{Arc, OnceLock};

use melior::Context;
use rustc_hash::FxHasher;

use yuzu_diagnostics::{DiagnosticsEngine, SourceId, SourceMap};
use yuzu_passes::{BoundLibrary, File};
use yuzu_syntax::GreenNode;

use crate::modules::{
    self, Loaded, MapResolver, ModuleResolver, ModuleSource, Origin, Submodule, Unreadable,
};

/// One library file: its module path, its path under the library's root,
/// the name a diagnostic shows for it, and its text.
struct Embedded {
    path: &'static str,
    file: &'static str,
    name: &'static str,
    source: &'static str,
}

include!(concat!(env!("OUT_DIR"), "/stdlib.rs"));

/// The first segment of every library path.
const ROOT: &str = "yuzu";

/// The module the compiler writes for the engine being compiled for.
const ENGINE_MODULE: &str = "yuzu.engine";

/// The engine a program is compiled for.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Engine {
    #[default]
    DataFusion,
}

impl Engine {
    /// Every engine, in the order a message lists them.
    pub const ALL: &[Engine] = &[Engine::DataFusion];

    /// The name an option gives the engine.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Engine::DataFusion => "datafusion",
        }
    }
}

impl fmt::Display for Engine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Engine {
    type Err = UnknownEngine;

    fn from_str(name: &str) -> Result<Self, UnknownEngine> {
        Engine::ALL
            .iter()
            .copied()
            .find(|engine| engine.as_str() == name)
            .ok_or_else(|| UnknownEngine {
                name: name.to_owned(),
            })
    }
}

/// A name that is no engine's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnknownEngine {
    name: String,
}

impl fmt::Display for UnknownEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "`{}` is not a supported engine; ", self.name)?;
        match Engine::ALL {
            [only] => write!(f, "the supported engine is `{only}`"),
            all => {
                f.write_str("the supported engines are ")?;
                for (index, engine) in all.iter().enumerate() {
                    if index > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "`{engine}`")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for UnknownEngine {}

/// Whether a path belongs to the library. A program's own module cannot
/// take such a path, so it cannot stand in for a library module.
#[must_use]
pub fn is_library_path(path: &str) -> bool {
    path.split('.').next() == Some(ROOT)
}

/// `yuzu.engine` is written for the engine; every other path is a file.
pub(crate) fn resolve(path: &str, engine: Engine) -> Option<ModuleSource> {
    if path == ENGINE_MODULE {
        return Some(ModuleSource {
            origin: Origin::Named(format!("<{ENGINE_MODULE}>")),
            source: format!("pub let ENGINE = \"{engine}\"\n").into(),
            syntax: None,
        });
    }

    MODULES
        .iter()
        .find(|module| module.path == path)
        .map(|module| ModuleSource {
            origin: Origin::Named(module.name.to_owned()),
            source: Arc::clone(&texts()[path]),
            syntax: trees().get(path).cloned(),
        })
}

/// Writes the library's files under `cache`, read-only, and returns the
/// folder that holds them.
///
/// The folder's name comes from the files' content, so a call reuses a
/// folder that is already there. The files go into a staging folder first,
/// and the staging folder then moves into place. Thus a reader never sees
/// part of the files.
///
/// # Errors
///
/// When a folder or a file cannot be written.
pub fn install(cache: &Path) -> io::Result<PathBuf> {
    let mut hasher = FxHasher::default();
    for module in MODULES {
        module.file.hash(&mut hasher);
        module.source.hash(&mut hasher);
    }
    let root = cache.join(format!("stdlib-{:016x}", hasher.finish()));
    if root.is_dir() {
        return Ok(root);
    }

    let staging = cache.join(format!("stdlib-staging-{}", std::process::id()));
    remove_staging(&staging);
    for module in MODULES {
        let file = staging.join(module.file);
        if let Some(parent) = file.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&file, module.source)?;
        let mut permissions = fs::metadata(&file)?.permissions();
        permissions.set_readonly(true);
        fs::set_permissions(&file, permissions)?;
    }

    match fs::rename(&staging, &root) {
        Ok(()) => Ok(root),
        // Another process put the same content in place first.
        Err(_) if root.is_dir() => {
            remove_staging(&staging);
            Ok(root)
        }
        Err(error) => {
            remove_staging(&staging);
            Err(error)
        }
    }
}

fn remove_staging(staging: &Path) {
    for module in MODULES {
        let file = staging.join(module.file);
        if let Ok(metadata) = fs::metadata(&file) {
            let mut permissions = metadata.permissions();
            #[expect(
                clippy::permissions_set_readonly_false,
                reason = "the file is ours, written a moment ago, and is removed next"
            )]
            permissions.set_readonly(false);
            let _ = fs::set_permissions(&file, permissions);
        }
    }
    let _ = fs::remove_dir_all(staging);
}

/// A library module as the file [`install`] wrote under `root`. `None` for
/// a module that has no file, as `yuzu.engine` has none.
#[must_use]
pub fn resolve_under(root: &Path, path: &str) -> Option<ModuleSource> {
    let module = MODULES.iter().find(|module| module.path == path)?;
    Some(ModuleSource {
        origin: Origin::File(root.join(module.file)),
        source: Arc::clone(&texts()[path]),
        syntax: trees().get(path).cloned(),
    })
}

/// The library as every compile on the thread starts from it: its files
/// loaded and what each of its modules declares.
pub(crate) struct Library {
    pub(crate) sources: SourceMap,
    pub(crate) files: Vec<File>,
    pub(crate) submodules: HashMap<String, Vec<Submodule>>,
    pub(crate) trees: Vec<(SourceId, GreenNode)>,
}

thread_local! {
    static LIBRARIES: RefCell<HashMap<Engine, Rc<Library>>> = RefCell::new(HashMap::new());
    static INSTALLED: RefCell<HashMap<(Engine, PathBuf), Rc<Library>>> = RefCell::new(HashMap::new());
}

/// Reads each library module from the files [`install`] wrote under a
/// folder.
struct Installed<'r>(&'r Path);

impl ModuleResolver for Installed<'_> {
    fn resolve(&self, _path: &str) -> Result<Option<ModuleSource>, Unreadable> {
        Ok(None)
    }

    fn resolve_library(&self, path: &str) -> Result<Option<ModuleSource>, Unreadable> {
        Ok(resolve_under(self.0, path))
    }
}

impl Library {
    /// The library for an engine, loaded once for the thread. A syntax tree
    /// cannot cross threads, so each thread loads its own.
    pub(crate) fn for_thread(engine: Engine) -> Rc<Self> {
        LIBRARIES.with(|libraries| {
            Rc::clone(
                libraries
                    .borrow_mut()
                    .entry(engine)
                    .or_insert_with(|| Rc::new(Self::load(engine, &MapResolver(HashMap::new())))),
            )
        })
    }

    /// The library for an engine as the files [`install`] wrote under
    /// `root`, loaded once for the thread. Its sources name those files,
    /// so a reference into the library has a file to go to.
    pub(crate) fn for_thread_under(engine: Engine, root: &Path) -> Rc<Self> {
        INSTALLED.with(|libraries| {
            Rc::clone(
                libraries
                    .borrow_mut()
                    .entry((engine, root.to_path_buf()))
                    .or_insert_with(|| Rc::new(Self::load(engine, &Installed(root)))),
            )
        })
    }

    fn load(engine: Engine, resolver: &dyn ModuleResolver) -> Self {
        let mut sources = SourceMap::new();
        let mut diagnostics = DiagnosticsEngine::new();
        let entry = sources.add("<library>", "");
        let Loaded {
            files,
            mut submodules,
            trees,
        } = modules::load_program(
            modules::EntryFile {
                source: entry,
                syntax: None,
            },
            &mut sources,
            &mut diagnostics,
            resolver,
            engine,
            None,
            None,
        );
        assert!(
            diagnostics.diagnostics().is_empty(),
            "the library loads without diagnostics: {:#?}",
            diagnostics.diagnostics()
        );
        submodules.remove(modules::ENTRY);
        let files: Vec<File> = files
            .into_iter()
            .filter(|file| file.module().is_some())
            .collect();

        let trees = trees
            .into_iter()
            .filter(|(source, _)| *source != entry)
            .collect();

        Self {
            sources,
            files,
            submodules,
            trees,
        }
    }
}

/// The library's names for an engine, bound once for the process.
pub(crate) fn bound_library(engine: Engine) -> &'static BoundLibrary<'static> {
    static DATAFUSION: OnceLock<BoundLibrary<'static>> = OnceLock::new();
    let bound = match engine {
        Engine::DataFusion => &DATAFUSION,
    };
    bound.get_or_init(|| bind_library(engine))
}

/// Binds the library's names for an engine. The names live in a context
/// made for them and never dropped: one for each engine, for the process.
fn bind_library(engine: Engine) -> BoundLibrary<'static> {
    let Library { sources, files, .. } = Library::load(engine, &MapResolver(HashMap::new()));
    let mut diagnostics = DiagnosticsEngine::new();
    let context: &'static Context = Box::leak(Box::new(yuzu_mlir::context()));
    let bound = yuzu_passes::bind_library(context, &sources, &files, &mut diagnostics);
    assert!(
        diagnostics.diagnostics().is_empty(),
        "the library binds without diagnostics: {:#?}",
        diagnostics.diagnostics()
    );
    bound
}

/// Each library file's text, shared by every source made from it.
fn texts() -> &'static HashMap<&'static str, Arc<str>> {
    static TEXTS: OnceLock<HashMap<&'static str, Arc<str>>> = OnceLock::new();
    TEXTS.get_or_init(|| {
        MODULES
            .iter()
            .map(|module| (module.path, Arc::from(module.source)))
            .collect()
    })
}

/// Each library file's tree, parsed once for the process; every compile
/// after the first starts from the same trees.
fn trees() -> &'static HashMap<&'static str, GreenNode> {
    static TREES: OnceLock<HashMap<&'static str, GreenNode>> = OnceLock::new();
    TREES.get_or_init(|| {
        MODULES
            .iter()
            .map(|module| {
                let mut sources = SourceMap::new();
                let source_id = sources.add(module.name, Arc::clone(&texts()[module.path]));
                let mut diagnostics = DiagnosticsEngine::new();
                let syntax = yuzu_parser::parse_text(module.source, &mut diagnostics, source_id);
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
    use std::fmt::Write;

    use melior::ir::operation::OperationLike;
    use yuzu_diagnostics::{DiagnosticsEngine, SourceMap};
    use yuzu_passes::Lowering;

    use super::{Engine, MODULES, install, resolve_under, trees};
    use crate::modules::{self, MapResolver, Origin};

    #[test]
    fn an_engine_the_compiler_does_not_know_is_named_in_the_error() {
        let unknown = "postgres"
            .parse::<Engine>()
            .expect_err("postgres is not an engine");
        assert_eq!(
            unknown.to_string(),
            "`postgres` is not a supported engine; the supported engine is `datafusion`"
        );
        assert_eq!("datafusion".parse::<Engine>(), Ok(Engine::default()));
    }

    #[test]
    fn the_library_installs_read_only_once() {
        let cache = std::env::temp_dir().join(format!("yuzu-install-{}", std::process::id()));
        let root = install(&cache).expect("the library installs");
        let first = MODULES.first().expect("the library has a file");
        let file = root.join(first.file);
        let written = std::fs::read_to_string(&file).expect("the file is written");
        assert_eq!(written, first.source);
        let permissions = std::fs::metadata(&file)
            .expect("the file is there")
            .permissions();
        assert!(permissions.readonly());
        assert_eq!(install(&cache).expect("the library installs again"), root);

        let module = resolve_under(&root, first.path).expect("a file module resolves");
        assert_eq!(module.origin, Origin::File(file));
        assert!(resolve_under(&root, "yuzu.engine").is_none());

        std::fs::remove_dir_all(&cache).expect("the test's cache is removed");
    }

    #[test]
    fn each_library_file_parses() {
        for module in MODULES {
            assert!(trees().contains_key(module.path), "{}", module.name);
        }
    }

    #[test]
    fn the_whole_library_lowers_without_diagnostics() {
        let imports = MODULES.iter().fold(String::new(), |mut imports, module| {
            writeln!(imports, "import {}", module.path).expect("writing to a String cannot fail");
            imports
        });
        let mut sources = SourceMap::new();
        let mut diagnostics = DiagnosticsEngine::new();
        let entry = sources.add("main.yz", imports);
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
        let module =
            yuzu_passes::lower_ast_to_yzl(&context, &sources, &files, &mut diagnostics, None);
        let messages: Vec<&str> = diagnostics
            .diagnostics()
            .iter()
            .map(|diagnostic| diagnostic.message.as_str())
            .collect();
        assert!(messages.is_empty(), "{messages:?}");
        assert!(module.as_operation().verify());
    }
}
