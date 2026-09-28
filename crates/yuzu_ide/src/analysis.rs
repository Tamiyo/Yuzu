//! The host that takes each change, and the snapshot a request reads.

use std::path::PathBuf;
use std::sync::Arc;

use rustc_hash::FxHashMap;
use text_size::TextRange;
use yuzu_ast::{AstNode, Root};
use yuzu_diagnostics::diagnostics::Diagnostic;
use yuzu_diagnostics::diagnostics::engine::DiagnosticsEngine;
use yuzu_diagnostics::source_map::SourceMap;
use yuzu_driver::modules::{Location, locate};
use yuzu_syntax::{GreenNode, SyntaxNode};

use crate::check::{self, DiskCache, Document};
use crate::{
    Checked, FileId, FilePosition, Fold, HlRange, StructureNode, file_structure, folding_ranges,
    selection_ranges, syntax_highlighting,
};

/// New texts and paths for some files. `None` removes a file's text, or
/// says it has no path, as a document not yet saved has none.
#[derive(Debug, Default)]
pub struct Change {
    files: Vec<(FileId, Option<Arc<str>>)>,
    paths: Vec<(FileId, Option<PathBuf>)>,
}

impl Change {
    /// Sets a file's text, or removes the file with `None`.
    pub fn set_file(&mut self, file_id: FileId, text: Option<Arc<str>>) {
        self.files.push((file_id, text));
    }

    /// Sets where a file is saved: `None` for a document not yet saved.
    pub fn set_path(&mut self, file_id: FileId, path: Option<PathBuf>) {
        self.paths.push((file_id, path));
    }
}

/// The files, their paths, and the cache of files read from disk.
#[derive(Debug, Default)]
pub struct AnalysisHost {
    files: Arc<FxHashMap<FileId, Arc<ParsedFile>>>,
    paths: Arc<FxHashMap<FileId, SavedFile>>,
    disk: Arc<DiskCache>,
}

impl AnalysisHost {
    /// A file is parsed when it changes, so a snapshot never parses.
    pub fn apply_change(&mut self, change: Change) {
        let files = Arc::make_mut(&mut self.files);
        for (file_id, text) in change.files {
            match text {
                Some(text) => files.insert(file_id, Arc::new(ParsedFile::parse(text))),
                None => files.remove(&file_id),
            };
        }

        if change.paths.is_empty() {
            return;
        }
        let paths = Arc::make_mut(&mut self.paths);
        for (file_id, path) in change.paths {
            match path {
                Some(path) => paths.insert(file_id, SavedFile::new(path)),
                None => paths.remove(&file_id),
            };
        }
    }

    /// A snapshot of the files as they are now, for a request to read.
    #[must_use]
    pub fn analysis(&self) -> Analysis {
        Analysis {
            files: Arc::clone(&self.files),
            paths: Arc::clone(&self.paths),
            disk: Arc::clone(&self.disk),
        }
    }
}

/// Each query returns `None` for a file the host does not hold.
#[derive(Debug)]
pub struct Analysis {
    files: Arc<FxHashMap<FileId, Arc<ParsedFile>>>,
    paths: Arc<FxHashMap<FileId, SavedFile>>,
    disk: Arc<DiskCache>,
}

impl Analysis {
    /// Runs the compiler's checks over the program a file belongs to. `None`
    /// for a file with no path, whose imports cannot be found.
    #[must_use]
    pub fn check(&self, file_id: FileId) -> Option<Checked> {
        let saved = self.paths.get(&file_id)?;
        let file = self.file(file_id)?;
        let documents: Vec<Document<'_>> = self
            .paths
            .iter()
            .filter_map(|(&file_id, saved)| {
                Some(Document {
                    file_id,
                    saved,
                    parsed: self.file(file_id)?,
                })
            })
            .collect();
        Some(check::check(saved, file, &documents, &self.disk))
    }

    /// A file's text.
    #[must_use]
    pub fn file_text(&self, file_id: FileId) -> Option<Arc<str>> {
        Some(Arc::clone(&self.file(file_id)?.text))
    }

    /// The errors the parse of a file reported. Every label is in that file.
    #[must_use]
    pub fn syntax_diagnostics(&self, file_id: FileId) -> Option<Arc<[Diagnostic]>> {
        Some(Arc::clone(&self.file(file_id)?.diagnostics))
    }

    /// The highlights a file's syntax tree gives, in text order.
    #[must_use]
    pub fn highlight(&self, file_id: FileId) -> Option<Vec<HlRange>> {
        Some(syntax_highlighting::highlight(
            &self.file(file_id)?.syntax(),
        ))
    }

    /// A file's declarations, as an outline.
    #[must_use]
    pub fn file_structure(&self, file_id: FileId) -> Option<Vec<StructureNode>> {
        Some(file_structure::file_structure(&self.file(file_id)?.root()))
    }

    /// The ranges of a file that an editor can fold.
    #[must_use]
    pub fn folding_ranges(&self, file_id: FileId) -> Option<Vec<Fold>> {
        let file = self.file(file_id)?;
        Some(folding_ranges::folding_ranges(&file.syntax(), file.text()))
    }

    /// The ranges around a position, innermost first.
    #[must_use]
    pub fn selection_ranges(&self, position: FilePosition) -> Option<Vec<TextRange>> {
        let root = self.file(position.file_id)?.syntax();
        Some(selection_ranges::selection_ranges(&root, position.offset))
    }

    fn file(&self, file_id: FileId) -> Option<&ParsedFile> {
        self.files.get(&file_id).map(Arc::as_ref)
    }
}

/// A file's path, and where the path puts it in its program.
#[derive(Clone, Debug)]
pub(crate) struct SavedFile {
    pub(crate) path: PathBuf,
    pub(crate) location: Location,
}

impl SavedFile {
    fn new(path: PathBuf) -> Self {
        let location = locate(&path);
        Self { path, location }
    }
}

#[derive(Debug)]
pub(crate) struct ParsedFile {
    text: Arc<str>,
    green: GreenNode,
    diagnostics: Arc<[Diagnostic]>,
}

impl ParsedFile {
    fn parse(text: Arc<str>) -> Self {
        let (root, diagnostics) = parse(&text);
        Self {
            green: root.green().into_owned(),
            diagnostics: diagnostics.into(),
            text,
        }
    }

    pub(crate) fn text(&self) -> &str {
        &self.text
    }

    /// The text, shared rather than copied.
    pub(crate) fn shared_text(&self) -> Arc<str> {
        Arc::clone(&self.text)
    }

    /// The tree, when its parse reported nothing: only such a tree is handed
    /// on for reuse, so a reused tree hides no error.
    pub(crate) fn clean_tree(&self) -> Option<&GreenNode> {
        self.diagnostics.is_empty().then_some(&self.green)
    }

    fn syntax(&self) -> SyntaxNode {
        SyntaxNode::new_root(self.green.clone())
    }

    fn root(&self) -> Root {
        Root::cast(self.syntax()).expect("a parse always yields a root")
    }
}

/// A text's syntax tree, and the errors its parse reported.
pub(crate) fn parse(text: &str) -> (SyntaxNode, Vec<Diagnostic>) {
    let mut sources = SourceMap::new();
    let source_id = sources.add(String::new(), String::new());
    let mut engine = DiagnosticsEngine::new();
    let root = yuzu_parser::parse_text(text, &mut engine, source_id);
    (root, engine.into_diagnostics())
}
