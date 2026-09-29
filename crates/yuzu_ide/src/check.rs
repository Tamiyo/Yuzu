//! A file's program checked by the compiler. The editor's documents are read
//! before the files on disk, and an open library file before the copy built
//! into the compiler.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

use rustc_hash::FxHashMap;
use text_size::TextRange;
use yuzu_diagnostics::{Diagnostic, SourceId, Span};
use yuzu_driver::index::Index;
use yuzu_driver::modules::{
    FsResolver, Location, ModuleResolver, ModuleSource, Origin, Unreadable,
};
use yuzu_driver::{Focus, stdlib};
use yuzu_syntax::{GreenNode, SyntaxNode};

use crate::analysis::{ParsedFile, SavedFile};
use crate::hover::HoverResult;
use crate::inlay_hints::InlayHint;
use crate::names::{self, Resolution, Trees};
use crate::navigation::{FileRange, References};
use crate::{CallSite, FileId, FilePosition, HlRange, SignatureHelp};
use crate::{hover, inlay_hints, navigation, signature_help, syntax_highlighting};

/// What checking a file's program found. Its names are resolved once, when
/// the check is made on the checker thread, so a request only looks them up.
#[derive(Debug)]
pub struct Checked {
    inner: yuzu_driver::Checked,
    /// The source each open document was read as.
    files: FxHashMap<FileId, SourceId>,
    trees: Trees,
    resolutions: Vec<Resolution>,
    /// Each type the index holds, by the span it belongs to.
    types: FxHashMap<Span, usize>,
}

impl Checked {
    fn new(inner: yuzu_driver::Checked, documents: &[Document<'_>]) -> Self {
        let files = documents
            .iter()
            .filter_map(|document| {
                let source = inner.sources.file_id(&document.saved.path)?;
                Some((document.file_id, source))
            })
            .collect();
        let trees: Trees = inner.syntax.iter().cloned().collect();
        let resolutions = names::resolutions(&inner.index.references, &trees);
        let types = inner
            .index
            .types
            .iter()
            .enumerate()
            .map(|(at, typed)| (typed.at, at))
            .collect();
        Checked {
            inner,
            files,
            trees,
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
        self.inner.sources.path(source)
    }

    /// A source's text, as the check read it.
    #[must_use]
    pub fn text(&self, source: SourceId) -> &str {
        self.inner.sources.text(source)
    }

    /// An open document's text, as the check read it.
    #[must_use]
    pub fn file_text(&self, file_id: FileId) -> Option<&str> {
        Some(self.text(self.source(file_id)?))
    }

    /// The text of a file as this check read it, by its path.
    #[must_use]
    pub fn path_text(&self, path: &Path) -> Option<&str> {
        let source = self.inner.sources.file_id(path)?;
        Some(self.text(source))
    }

    /// Where the name at a position is declared.
    #[must_use]
    pub fn goto_definition(&self, position: FilePosition) -> Option<FileRange> {
        navigation::goto_definition(self, self.source(position.file_id)?, position.offset)
    }

    /// The overloads of the function a call names, as this check resolved
    /// the name at `site.callee` in a file.
    #[must_use]
    pub fn signature_help(&self, file_id: FileId, site: CallSite) -> Option<SignatureHelp> {
        signature_help::signature_help(self, self.source(file_id)?, site)
    }

    /// The declaration of the name at a position, and each use of it.
    #[must_use]
    pub fn references(&self, position: FilePosition) -> References {
        self.source(position.file_id)
            .map(|source| navigation::references(self, source, position.offset))
            .unwrap_or_default()
    }

    /// The declaration and uses of the name at a position, in its own file.
    #[must_use]
    pub fn highlight_related(&self, position: FilePosition) -> Vec<TextRange> {
        self.source(position.file_id)
            .map(|source| navigation::highlight_related(self, source, position.offset))
            .unwrap_or_default()
    }

    /// What hovering at a position shows.
    #[must_use]
    pub fn hover(&self, position: FilePosition) -> Option<HoverResult> {
        hover::hover(self, self.source(position.file_id)?, position.offset)
    }

    /// Each resolved use in a file, highlighted as its declaration is.
    #[must_use]
    pub fn highlight_uses(&self, file_id: FileId) -> Vec<HlRange> {
        self.source(file_id)
            .map(|source| syntax_highlighting::highlight_uses(self, source))
            .unwrap_or_default()
    }

    /// The type of each `let` in `range` of a file that does not write one.
    #[must_use]
    pub fn inlay_hints(&self, file_id: FileId, range: TextRange) -> Vec<InlayHint> {
        self.source(file_id)
            .map(|source| inlay_hints::inlay_hints(self, source, range))
            .unwrap_or_default()
    }

    pub(crate) fn index(&self) -> &Index {
        &self.inner.index
    }

    pub(crate) fn resolutions(&self) -> &[Resolution] {
        &self.resolutions
    }

    /// The name a resolution resolves, as its declaration spells it.
    pub(crate) fn name(&self, resolution: &Resolution) -> &str {
        &self.inner.index.references[resolution.reference].name
    }

    /// The tree a source was lowered from.
    pub(crate) fn syntax(&self, source: SourceId) -> Option<SyntaxNode> {
        self.trees.get(&source).cloned().map(SyntaxNode::new_root)
    }

    /// The type inference gave the syntax at a span, when it gave one.
    pub(crate) fn type_at(&self, at: Span) -> Option<&str> {
        let typed = &self.inner.index.types[*self.types.get(&at)?];
        Some(&typed.ty)
    }

    fn source(&self, file_id: FileId) -> Option<SourceId> {
        self.files.get(&file_id).copied()
    }
}

/// An open document with a path, as a check reads it.
pub(crate) struct Document<'a> {
    pub(crate) file_id: FileId,
    pub(crate) saved: &'a SavedFile,
    pub(crate) parsed: &'a ParsedFile,
}

/// Checks the program `file` belongs to, reading `documents` before the disk.
pub(crate) fn check(
    saved: &SavedFile,
    parsed: &ParsedFile,
    documents: &[Document<'_>],
    disk: &DiskCache,
    library_root: Option<&Path>,
) -> Checked {
    let overlay = Overlay::new(saved, documents, disk, library_root);
    let origin = Origin::File(saved.path.clone());
    let focus = match &saved.location {
        Location::Entry { .. } => Focus::Entry {
            origin: &origin,
            source: parsed.text(),
            syntax: parsed.clean_tree(),
        },
        Location::Module { path, .. } => Focus::Module(path),
    };
    disk.start_check();
    let checked = Checked::new(yuzu_driver::check(focus, &overlay), documents);
    disk.finish_check();
    checked
}

/// How many checks a file on disk stays cached without being read.
const KEEP_UNREAD: u64 = 64;

/// Files read from disk, each parsed once for each text it has had. The
/// host lives on the main thread and checks run on the checker's, so the
/// cache is behind a lock. A file no check has read for [`KEEP_UNREAD`]
/// checks is dropped.
#[derive(Debug, Default)]
pub(crate) struct DiskCache {
    state: Mutex<DiskState>,
}

#[derive(Debug, Default)]
struct DiskState {
    /// How many checks have started.
    checks: u64,
    files: FxHashMap<PathBuf, CachedFile>,
}

#[derive(Debug)]
struct CachedFile {
    text: Arc<str>,
    syntax: Option<GreenNode>,
    /// The check that last read the file.
    read_by: u64,
}

impl DiskCache {
    /// A file's text, and its tree when that text parses without errors.
    fn read(&self, path: &Path) -> std::io::Result<(Arc<str>, Option<GreenNode>)> {
        let text = std::fs::read_to_string(path)?;
        let mut state = self.lock();
        let check = state.checks;
        if let Some(cached) = state.files.get_mut(path)
            && *cached.text == *text
        {
            cached.read_by = check;
            return Ok((Arc::clone(&cached.text), cached.syntax.clone()));
        }

        let (tree, errors) = crate::analysis::parse(&text);
        let syntax = errors.is_empty().then(|| tree.green().into_owned());
        let text: Arc<str> = text.into();
        state.files.insert(
            path.to_path_buf(),
            CachedFile {
                text: Arc::clone(&text),
                syntax: syntax.clone(),
                read_by: check,
            },
        );
        Ok((text, syntax))
    }

    fn start_check(&self) {
        self.lock().checks += 1;
    }

    fn finish_check(&self) {
        let mut state = self.lock();
        let oldest = state.checks.saturating_sub(KEEP_UNREAD);
        state.files.retain(|_, cached| cached.read_by >= oldest);
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, DiskState> {
        // A panic while the lock is held leaves each entry whole, so a
        // poisoned cache still holds only sound entries.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Modules read from the editor's documents first and the disk second.
struct Overlay<'d> {
    files: FsResolver,
    documents: FxHashMap<&'d Path, &'d ParsedFile>,
    disk: &'d DiskCache,
    /// Open library files, by module path.
    library: FxHashMap<&'d str, (&'d Path, &'d ParsedFile)>,
    /// Where the library's files are, when the file checked is one of them.
    library_files: Option<FsResolver>,
    /// Where the built-in library was written, so a reference into it has a
    /// file to go to.
    library_root: Option<&'d Path>,
}

impl<'d> Overlay<'d> {
    fn new(
        saved: &SavedFile,
        documents: &[Document<'d>],
        disk: &'d DiskCache,
        library_root: Option<&'d Path>,
    ) -> Self {
        let base = match &saved.location {
            Location::Entry { base } | Location::Module { base, .. } => base.clone(),
        };
        let library_files = match &saved.location {
            Location::Module { path, .. } if stdlib::is_library_path(path) => {
                Some(FsResolver { base: base.clone() })
            }
            Location::Module { .. } | Location::Entry { .. } => None,
        };

        let library = documents
            .iter()
            .filter_map(|document| match &document.saved.location {
                Location::Module { path, .. } if stdlib::is_library_path(path) => Some((
                    path.as_str(),
                    (document.saved.path.as_path(), document.parsed),
                )),
                Location::Module { .. } | Location::Entry { .. } => None,
            })
            .collect();

        Overlay {
            files: FsResolver { base },
            documents: documents
                .iter()
                .map(|document| (document.saved.path.as_path(), document.parsed))
                .collect(),
            disk,
            library,
            library_files,
            library_root,
        }
    }

    /// The first candidate an open document or a file on disk holds.
    fn read(&self, candidates: [PathBuf; 2]) -> Result<Option<ModuleSource>, Unreadable> {
        for file in candidates {
            let (source, syntax) = if let Some(parsed) = self.documents.get(file.as_path()) {
                (parsed.shared_text(), parsed.clean_tree().cloned())
            } else if file.is_file() {
                match self.disk.read(&file) {
                    Ok(read) => read,
                    Err(error) => return Err(Unreadable::new(file, error)),
                }
            } else {
                continue;
            };
            return Ok(Some(ModuleSource {
                origin: Origin::File(file),
                source,
                syntax,
            }));
        }
        Ok(None)
    }
}

impl ModuleResolver for Overlay<'_> {
    fn resolve(&self, path: &str) -> Result<Option<ModuleSource>, Unreadable> {
        self.read(self.files.candidates(path))
    }

    fn resolve_library(&self, path: &str) -> Result<Option<ModuleSource>, Unreadable> {
        if let Some((file, parsed)) = self.library.get(path) {
            return Ok(Some(ModuleSource {
                origin: Origin::File(file.to_path_buf()),
                source: parsed.shared_text(),
                syntax: parsed.clean_tree().cloned(),
            }));
        }
        match (&self.library_files, self.library_root) {
            (Some(files), _) => self.read(files.candidates(path)),
            (None, Some(root)) => Ok(stdlib::resolve_under(root, path)),
            (None, None) => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use expect_test::{Expect, expect};

    use crate::test_support::Tree;
    use crate::{AnalysisHost, Change, FileId};

    fn check(open: &[(&Path, &str)], expected: &Expect) {
        let mut change = Change::default();
        for (at, (path, text)) in open.iter().enumerate() {
            let file_id = FileId(u32::try_from(at).unwrap());
            change.set_file(file_id, Some((*text).into()));
            change.set_path(file_id, Some(path.to_path_buf()));
        }
        let mut host = AnalysisHost::default();
        host.apply_change(change);

        let checked = host.analysis().check(FileId(0)).unwrap();
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
            &expect![[r"
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
            &expect!["util.yz 14..17 unknown type `i64`"],
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
            &expect!["helpers.yz 17..24 unknown type `float32`"],
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
            &expect!["datafusion.yz 74..77 unknown type `i64`"],
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

    #[test]
    fn a_file_no_check_reads_is_dropped() {
        let tree = Tree::new(&[("helpers.yz", "pub def two() -> int64 { return 2 }\n")]);
        let cache = super::DiskCache::default();
        cache.start_check();
        cache.read(&tree.0.join("helpers.yz")).unwrap();
        cache.finish_check();

        for _ in 0..super::KEEP_UNREAD {
            cache.start_check();
            cache.finish_check();
        }
        assert_eq!(
            cache.lock().files.len(),
            1,
            "a file stays for the kept checks"
        );

        cache.start_check();
        cache.finish_check();
        assert!(cache.lock().files.is_empty(), "then it is dropped");
    }
}
