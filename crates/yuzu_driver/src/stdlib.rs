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

/// Whether a path belongs to the library. A program's own module cannot
/// take such a path, so it cannot stand in for a library module.
pub(crate) fn reserves(path: &str) -> bool {
    path.split('.').next() == Some(ROOT)
}

pub(crate) fn resolve(path: &str) -> Option<ModuleSource> {
    MODULES
        .iter()
        .find(|module| module.path == path)
        .map(|module| ModuleSource {
            name: module.name.to_string(),
            source: module.source.to_string(),
        })
}
