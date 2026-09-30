//! Builds the standard library into the compiler: each `.yz` file under
//! `stdlib/` becomes one entry of a table that `stdlib.rs` includes.

use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets it"));
    let root = manifest
        .join("../../stdlib")
        .canonicalize()
        .expect("the stdlib directory exists");
    println!("cargo:rerun-if-changed={}", root.display());

    let mut files = Vec::new();
    collect(&root, &mut files);
    files.sort();

    let mut table = String::from("const MODULES: &[Embedded] = &[\n");
    for file in &files {
        let relative = file
            .strip_prefix(&root)
            .expect("each file is under the root");
        writeln!(
            table,
            "    Embedded {{ path: {path:?}, file: {file:?}, name: {name:?}, source: include_str!({absolute:?}) }},",
            path = module_path(relative),
            file = file_path(relative),
            name = format!("stdlib/{}", file_path(relative)),
            absolute = file.display().to_string(),
        )
        .expect("writing to a string cannot fail");
    }
    table.push_str("];\n");

    let out = PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets it"));
    fs::write(out.join("stdlib.rs"), table).expect("OUT_DIR is writable");
}

fn collect(directory: &Path, files: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(directory).expect("the stdlib directory is readable");
    for entry in entries {
        let path = entry.expect("a directory entry is readable").path();
        if path.is_dir() {
            collect(&path, files);
        } else if path.extension().is_some_and(|extension| extension == "yz") {
            files.push(path);
        }
    }
}

/// A library file's path under the library's root, with `/` between its
/// parts on every platform.
fn file_path(relative: &Path) -> String {
    segments(relative).join("/")
}

/// The module path of a library file: `yuzu/std/math.yz` is
/// `yuzu.std.math`, and `yuzu/std/mod.yz` is `yuzu.std`.
fn module_path(relative: &Path) -> String {
    let mut segments = segments(&relative.with_extension(""));

    if segments.last().is_some_and(|last| last == "mod") {
        segments.pop();
    }

    segments.join(".")
}

fn segments(path: &Path) -> Vec<String> {
    path.components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect()
}
