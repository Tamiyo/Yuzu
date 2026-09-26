//! The standard library: the Yuzu modules under `stdlib/`, built into the
//! compiler so that no files have to be installed beside it.

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
