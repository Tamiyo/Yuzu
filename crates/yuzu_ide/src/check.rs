//! A file's program checked by the compiler. The editor's documents are read
//! before the files on disk, and an open library file before the copy built
//! into the compiler.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

use rustc_hash::FxHashMap;
use text_size::{TextRange, TextSize};
use yuzu_diagnostics::diagnostics::{Diagnostic, Span};
use yuzu_diagnostics::source_map::SourceId;
use yuzu_driver::index::Index;
use yuzu_driver::modules::{FsResolver, Location, ModuleResolver, ModuleSource, locate};
use yuzu_driver::{Focus, stdlib};
use yuzu_syntax::{GreenNode, SyntaxNode};

use crate::HlRange;
use crate::analysis::ParsedFile;
use crate::hover::HoverResult;
use crate::inlay_hints::InlayHint;
use crate::names::{self, Resolution};
use crate::navigation::FileRange;
use crate::{hover, inlay_hints, navigation, syntax_highlighting};

/// What checking a file's program found. Its names are resolved once, when
/// the check is made on the checker thread, so a request only looks them up.
#[derive(Debug)]
pub struct Checked {
    inner: yuzu_driver::Checked,
    resolutions: Vec<Resolution>,
    /// Each type the index holds, by the span it belongs to.
    types: FxHashMap<Span, String>,
}

impl Checked {
    fn new(inner: yuzu_driver::Checked) -> Self {
        let resolutions = names::resolutions(&inner);
        let types = inner
            .index
            .types
            .iter()
            .map(|typed| (typed.at, typed.ty.clone()))
            .collect();
        Checked {
            inner,
            resolutions,
            types,
        }
    }

    /// Each diagnostic the check reported, in every file it read.
    #[must_use]
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.inner.diagnostics
    }

    /// The file a source was read from. `None` for a library module built
    /// into the compiler, and for the entry a module's check makes up.
    #[must_use]
    pub fn path(&self, source: SourceId) -> Option<&Path> {
        let path = Path::new(self.inner.sources.name(source));
        path.is_absolute().then_some(path)
    }

    /// A source's text, as the check read it.
    #[must_use]
    pub fn text(&self, source: SourceId) -> &str {
        self.inner.sources.text(source)
    }

    /// The text of a file as this check read it.
    #[must_use]
    pub fn file_text(&self, file: &Path) -> Option<&str> {
        Some(self.text(self.source_of(file)?))
    }

    /// Where the name at a position is declared.
    #[must_use]
    pub fn goto_definition(&self, file: &Path, offset: TextSize) -> Option<FileRange> {
        navigation::goto_definition(self, self.source_of(file)?, offset)
    }

    /// The declaration of the name at a position, and each use of it.
    #[must_use]
    pub fn references(&self, file: &Path, offset: TextSize) -> Vec<FileRange> {
        self.source_of(file)
            .map(|source| navigation::references(self, source, offset))
            .unwrap_or_default()
    }

    /// The declaration and uses of the name at a position, in its own file.
    #[must_use]
    pub fn highlight(&self, file: &Path, offset: TextSize) -> Vec<TextRange> {
        self.source_of(file)
            .map(|source| navigation::highlight(self, source, offset))
            .unwrap_or_default()
    }

    /// What hovering at a position shows.
    #[must_use]
    pub fn hover(&self, file: &Path, offset: TextSize) -> Option<HoverResult> {
        hover::hover(self, self.source_of(file)?, offset)
    }

    /// Each resolved use in a file, highlighted as its declaration is.
    #[must_use]
    pub fn highlight_uses(&self, file: &Path) -> Vec<HlRange> {
        self.source_of(file)
            .map(|source| syntax_highlighting::highlight_uses(self, source))
            .unwrap_or_default()
    }

    /// The type of each `let` in a file that does not write one.
    #[must_use]
    pub fn inlay_hints(&self, file: &Path) -> Vec<InlayHint> {
        self.source_of(file)
            .map(|source| inlay_hints::inlay_hints(self, source))
            .unwrap_or_default()
    }

    pub(crate) fn index(&self) -> &Index {
        &self.inner.index
    }

    pub(crate) fn resolutions(&self) -> &[Resolution] {
        &self.resolutions
    }

    /// The tree a source was lowered from.
    pub(crate) fn syntax(&self, source: SourceId) -> Option<SyntaxNode> {
        names::tree(&self.inner, source)
    }

    /// The type inference gave the syntax at a span, when it gave one.
    pub(crate) fn type_at(&self, at: Span) -> Option<&str> {
        self.types.get(&at).map(String::as_str)
    }

    fn source_of(&self, file: &Path) -> Option<SourceId> {
        self.inner.sources.id(&file.to_string_lossy())
    }
}

/// `documents` holds each open file by its path.
pub(crate) fn check(
    file: &Path,
    parsed: &ParsedFile,
    documents: &FxHashMap<PathBuf, Arc<ParsedFile>>,
    disk: &DiskCache,
) -> Checked {
    let overlay = Overlay::new(file, documents, disk);
    let name = file.to_string_lossy();
    let focus = match &overlay.location {
        Location::Entry { .. } => Focus::Entry {
            name: &name,
            source: parsed.text(),
            syntax: parsed.clean_tree(),
        },
        Location::Module { path, .. } => Focus::Module(path),
    };
    Checked::new(yuzu_driver::check(focus, &overlay))
}

/// Files read from disk, each parsed once for each text it has had. The
/// host lives on the main thread and checks run on the checker's, so the
/// cache is behind a lock.
#[derive(Debug, Default)]
pub(crate) struct DiskCache {
    files: Mutex<FxHashMap<PathBuf, CachedFile>>,
}

#[derive(Debug)]
struct CachedFile {
    text: String,
    syntax: Option<GreenNode>,
}

impl DiskCache {
    /// A file's text, and its tree when that text parses without errors.
    fn read(&self, path: &Path) -> Option<(String, Option<GreenNode>)> {
        let text = std::fs::read_to_string(path).ok()?;
        // A panic while the lock is held leaves each entry whole, so a
        // poisoned cache still holds only sound entries.
        let mut files = self.files.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(cached) = files.get(path)
            && cached.text == text
        {
            return Some((text, cached.syntax.clone()));
        }

        let (tree, errors) = crate::analysis::parse(&text);
        let syntax = errors.is_empty().then(|| tree.green().into_owned());
        files.insert(
            path.to_path_buf(),
            CachedFile {
                text: text.clone(),
                syntax: syntax.clone(),
            },
        );
        Some((text, syntax))
    }
}

/// Modules read from the editor's documents first and the disk second.
struct Overlay<'d> {
    location: Location,
    files: FsResolver,
    documents: &'d FxHashMap<PathBuf, Arc<ParsedFile>>,
    disk: &'d DiskCache,
    /// Open library files, by module path.
    library: FxHashMap<String, (&'d Path, &'d ParsedFile)>,
    /// Where the library's files are, when the file checked is one of them.
    library_files: Option<FsResolver>,
}

impl<'d> Overlay<'d> {
    fn new(
        file: &Path,
        documents: &'d FxHashMap<PathBuf, Arc<ParsedFile>>,
        disk: &'d DiskCache,
    ) -> Self {
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
            .filter_map(|(path, parsed)| match locate(path) {
                Location::Module { path: module, .. } if stdlib::reserves(&module) => {
                    Some((module, (path.as_path(), parsed.as_ref())))
                }
                Location::Module { .. } | Location::Entry { .. } => None,
            })
            .collect();

        Overlay {
            location,
            files: FsResolver { base },
            documents,
            disk,
            library,
            library_files,
        }
    }

    /// The first candidate an open document or a file on disk holds.
    fn read(&self, candidates: [PathBuf; 2]) -> Option<ModuleSource> {
        candidates.into_iter().find_map(|file| {
            let (source, syntax) = match self.documents.get(&file) {
                Some(parsed) => (parsed.text().to_owned(), parsed.clean_tree().cloned()),
                None => self.disk.read(&file)?,
            };
            Some(ModuleSource {
                name: file.to_string_lossy().into_owned(),
                source,
                syntax,
            })
        })
    }
}

impl ModuleResolver for Overlay<'_> {
    fn resolve(&self, path: &str) -> Option<ModuleSource> {
        self.read(self.files.candidates(path))
    }

    fn resolve_library(&self, path: &str) -> Option<ModuleSource> {
        if let Some((file, parsed)) = self.library.get(path) {
            return Some(ModuleSource {
                name: file.to_string_lossy().into_owned(),
                source: parsed.text().to_owned(),
                syntax: parsed.clean_tree().cloned(),
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
            expect![[r"
                helpers.yz 17..20 unknown type `i64`
                main.yz 15..29 expected `str`, found `int64`"]],
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
            expect!["datafusion.yz 74..77 unknown type `i64`"],
        );
    }

    #[test]
    fn a_file_on_disk_is_parsed_again_only_when_its_text_changes() {
        let tree = Tree::new(&[("helpers.yz", "pub def two() -> int64 { return 2 }\n")]);
        let helpers = tree.0.join("helpers.yz");
        let cache = super::DiskCache::default();

        let (_, first) = cache.read(&helpers).unwrap();
        let (_, again) = cache.read(&helpers).unwrap();
        let (first, again) = (first.unwrap(), again.unwrap());
        assert!(
            std::ptr::eq(&raw const *first, &raw const *again),
            "an unchanged file keeps its tree"
        );

        std::fs::write(&helpers, "pub def three() -> int64 { return 3 }\n").unwrap();
        let (_, changed) = cache.read(&helpers).unwrap();
        assert!(
            !std::ptr::eq(&raw const *first, &raw const *changed.unwrap()),
            "a changed file is parsed again"
        );

        std::fs::write(&helpers, "pub def = 1\n").unwrap();
        let (_, broken) = cache.read(&helpers).unwrap();
        assert!(broken.is_none(), "a tree with errors is not handed on");
    }
}
