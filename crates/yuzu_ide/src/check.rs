//! A file's program checked by the compiler. The editor's documents are read
//! before the files on disk, and an open library file before the copy built
//! into the compiler.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use rustc_hash::FxHashMap;
use text_size::{TextRange, TextSize};
use yuzu_diagnostics::diagnostics::{Diagnostic, Span};
use yuzu_diagnostics::source_map::SourceId;
use yuzu_driver::index::Index;
use yuzu_driver::modules::{FsResolver, MARKER, ModuleResolver, ModuleSource};
use yuzu_driver::{Focus, stdlib};

use crate::HlRange;
use crate::hover::HoverResult;
use crate::inlay_hints::InlayHint;
use crate::navigation::FileRange;
use crate::{hover, inlay_hints, navigation, syntax_highlighting};

/// What checking a file's program found.
#[derive(Debug)]
pub struct Checked(yuzu_driver::Checked);

impl Checked {
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.0.diagnostics
    }

    /// The file a source was read from. `None` for a library module built
    /// into the compiler, and for the entry a module's check makes up.
    pub fn path(&self, source: SourceId) -> Option<&Path> {
        let path = Path::new(self.0.sources.name(source));
        path.is_absolute().then_some(path)
    }

    pub fn text(&self, source: SourceId) -> &str {
        self.0.sources.text(source)
    }

    /// The text of a file as this check read it.
    pub fn file_text(&self, file: &Path) -> Option<&str> {
        Some(self.text(self.source_of(file)?))
    }

    /// Where the name at a position is declared.
    pub fn goto_definition(&self, file: &Path, offset: TextSize) -> Option<FileRange> {
        navigation::goto_definition(self, self.source_of(file)?, offset)
    }

    /// The declaration of the name at a position, and each use of it.
    pub fn references(&self, file: &Path, offset: TextSize) -> Vec<FileRange> {
        self.source_of(file)
            .map(|source| navigation::references(self, source, offset))
            .unwrap_or_default()
    }

    /// The declaration and uses of the name at a position, in its own file.
    pub fn highlight(&self, file: &Path, offset: TextSize) -> Vec<TextRange> {
        self.source_of(file)
            .map(|source| navigation::highlight(self, source, offset))
            .unwrap_or_default()
    }

    pub fn hover(&self, file: &Path, offset: TextSize) -> Option<HoverResult> {
        hover::hover(self, self.source_of(file)?, offset)
    }

    /// Each resolved use in a file, highlighted as its declaration is.
    pub fn highlight_uses(&self, file: &Path) -> Vec<HlRange> {
        self.source_of(file)
            .map(|source| syntax_highlighting::highlight_uses(self, source))
            .unwrap_or_default()
    }

    pub fn inlay_hints(&self, file: &Path) -> Vec<InlayHint> {
        self.source_of(file)
            .map(|source| inlay_hints::inlay_hints(self, source))
            .unwrap_or_default()
    }

    pub(crate) fn index(&self) -> &Index {
        &self.0.index
    }

    /// The type inference gave the syntax at a span, when it gave one.
    pub(crate) fn type_at(&self, at: Span) -> Option<String> {
        self.0
            .index
            .types
            .iter()
            .find(|typed| typed.at.source_id == at.source_id && typed.at.range == at.range)
            .map(|typed| typed.ty.clone())
    }

    fn source_of(&self, file: &Path) -> Option<SourceId> {
        self.0.sources.id(&file.to_string_lossy())
    }
}

/// Where a file sits in its program.
enum Location {
    /// A file in a directory without a module marker: a program's start.
    Entry { base: PathBuf },
    /// A file in a directory tree of module markers, and the path that
    /// names it from the top of that tree.
    Module { base: PathBuf, path: String },
}

/// `documents` holds each open file's path and text.
pub(crate) fn check(file: &Path, text: &str, documents: &FxHashMap<PathBuf, Arc<str>>) -> Checked {
    let overlay = Overlay::new(file, documents);
    let name = file.to_string_lossy();
    let focus = match &overlay.location {
        Location::Entry { .. } => Focus::Entry {
            name: &name,
            source: text,
        },
        Location::Module { path, .. } => Focus::Module(path),
    };
    Checked(yuzu_driver::check(focus, &overlay))
}

/// Up through the directories that hold a module marker: the first one that
/// does not is where the program's module paths start.
fn locate(file: &Path) -> Location {
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

/// Modules read from the editor's documents first and the disk second.
struct Overlay<'d> {
    location: Location,
    files: FsResolver,
    documents: &'d FxHashMap<PathBuf, Arc<str>>,
    /// Open library files, by module path.
    library: FxHashMap<String, (&'d Path, &'d Arc<str>)>,
    /// Where the library's files are, when the file checked is one of them.
    library_files: Option<FsResolver>,
}

impl<'d> Overlay<'d> {
    fn new(file: &Path, documents: &'d FxHashMap<PathBuf, Arc<str>>) -> Self {
        let location = locate(file);
        let base = match &location {
            Location::Entry { base } | Location::Module { base, .. } => base.clone(),
        };
        let library_files = match &location {
            Location::Module { path, .. } if stdlib::reserves(path) => {
                Some(FsResolver { base: base.clone() })
            }
            Location::Module { .. } | Location::Entry { .. } => None,
        };

        let library = documents
            .iter()
            .filter_map(|(path, text)| match locate(path) {
                Location::Module { path: module, .. } if stdlib::reserves(&module) => {
                    Some((module, (path.as_path(), text)))
                }
                Location::Module { .. } | Location::Entry { .. } => None,
            })
            .collect();

        Overlay {
            location,
            files: FsResolver { base },
            documents,
            library,
            library_files,
        }
    }

    /// The first candidate an open document or a file on disk holds.
    fn read(&self, candidates: [PathBuf; 2]) -> Option<ModuleSource> {
        candidates.into_iter().find_map(|file| {
            let source = match self.documents.get(&file) {
                Some(text) => text.to_string(),
                None => std::fs::read_to_string(&file).ok()?,
            };
            Some(ModuleSource {
                name: file.to_string_lossy().into_owned(),
                source,
            })
        })
    }
}

impl ModuleResolver for Overlay<'_> {
    fn resolve(&self, path: &str) -> Option<ModuleSource> {
        self.read(self.files.candidates(path))
    }

    fn resolve_library(&self, path: &str) -> Option<ModuleSource> {
        if let Some((file, text)) = self.library.get(path) {
            return Some(ModuleSource {
                name: file.to_string_lossy().into_owned(),
                source: text.to_string(),
            });
        }
        self.read(self.library_files.as_ref()?.candidates(path))
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use expect_test::{Expect, expect};

    use crate::test_support::Tree;
    use crate::{AnalysisHost, Change, FileId};

    fn check(open: &[(&Path, &str)], file: usize, expected: Expect) {
        let mut change = Change::default();
        for (at, (path, text)) in open.iter().enumerate() {
            let file_id = FileId(u32::try_from(at).unwrap());
            change.set_file(file_id, Some((*text).into()));
            change.set_path(file_id, Some(path.to_path_buf()));
        }
        let mut host = AnalysisHost::default();
        host.apply_change(change);

        let checked = host
            .analysis()
            .check(FileId(u32::try_from(file).unwrap()))
            .unwrap();
        let rendered: Vec<String> = checked
            .diagnostics()
            .iter()
            .map(|diagnostic| {
                let span = diagnostic.labels[0].span;
                let file = checked
                    .path(span.source_id)
                    .and_then(Path::file_name)
                    .map_or("-".into(), |name| name.to_string_lossy());
                format!("{file} {:?} {}", span.range, diagnostic.message)
            })
            .collect();
        expected.assert_eq(&rendered.join("\n"));
    }

    #[test]
    fn an_entry_file_is_checked_with_its_imports() {
        let tree = Tree::new(&[("helpers.yz", "pub def two() -> i64 { return 2 }\n")]);
        let main = tree.0.join("main.yz");
        check(
            &[(&main, "import helpers\nlet x: str = 1\n")],
            0,
            expect![[r#"
                helpers.yz 17..20 unknown type `i64`
                main.yz 15..29 expected `str`, found `int64`"#]],
        );
    }

    #[test]
    fn a_file_under_a_marker_is_checked_as_its_module() {
        let tree = Tree::new(&[
            ("app/mod.yz", "pub mod util\n"),
            ("app/util.yz", "def unused(x: i64) -> int64 { return 1 }\n"),
        ]);
        let util = tree.0.join("app/util.yz");
        check(
            &[(&util, "def unused(x: i64) -> int64 { return 1 }\n")],
            0,
            expect!["util.yz 14..17 unknown type `i64`"],
        );
    }

    #[test]
    fn an_open_document_is_read_before_the_disk() {
        let tree = Tree::new(&[
            ("main.yz", "import helpers\n"),
            ("helpers.yz", "pub def two() -> int64 { return 2 }\n"),
        ]);
        let main = tree.0.join("main.yz");
        let helpers = tree.0.join("helpers.yz");
        check(
            &[
                (&main, "import helpers\n"),
                (&helpers, "pub def two() -> float32 { return 2 }\n"),
            ],
            0,
            expect!["helpers.yz 17..24 unknown type `float32`"],
        );
    }

    #[test]
    fn an_open_library_file_is_read_before_the_built_in_copy() {
        let datafusion = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../stdlib/yuzu/target/datafusion.yz")
            .canonicalize()
            .unwrap();
        let text = std::fs::read_to_string(&datafusion)
            .unwrap()
            .replacen("int64", "i64", 1);
        check(
            &[(&datafusion, &text)],
            0,
            expect!["datafusion.yz 39..42 unknown type `i64`"],
        );
    }
}
