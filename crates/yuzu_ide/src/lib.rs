//! What an editor asks about Yuzu source, answered in file ids and text
//! ranges. The protocol is `yuzu_lsp`'s concern; nothing here knows LSP.
//!
//! The shape follows rust-analyzer's `ide` crate: an [`AnalysisHost`] takes
//! each [`Change`], and an [`Analysis`] is an immutable snapshot that a
//! request reads.

use std::path::PathBuf;
use std::sync::Arc;

use rustc_hash::FxHashMap;
use text_size::{TextRange, TextSize};
use yuzu_ast::{AstNode, Root};
use yuzu_diagnostics::diagnostics::Diagnostic;
use yuzu_diagnostics::diagnostics::engine::DiagnosticsEngine;
use yuzu_diagnostics::source_map::SourceMap;
use yuzu_lexer::lexer::{Lexer, Token};
use yuzu_syntax::{GreenNode, SyntaxNode};

mod check;
mod file_structure;
mod folding_ranges;
mod hover;
mod inlay_hints;
mod names;
mod navigation;
mod selection_ranges;
mod syntax_highlighting;
#[cfg(test)]
mod test_support;

pub use check::Checked;
pub use file_structure::{StructureNode, StructureNodeKind};
pub use folding_ranges::{Fold, FoldKind};
pub use hover::HoverResult;
pub use inlay_hints::InlayHint;
pub use navigation::FileRange;
pub use syntax_highlighting::{Highlight, HlMod, HlMods, HlRange, HlTag};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FileId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FilePosition {
    pub file_id: FileId,
    pub offset: TextSize,
}

/// New texts and paths for some files. `None` removes a file's text, or
/// says it has no path, as a document not yet saved has none.
#[derive(Debug, Default)]
pub struct Change {
    files: Vec<(FileId, Option<Arc<str>>)>,
    paths: Vec<(FileId, Option<PathBuf>)>,
}

impl Change {
    pub fn set_file(&mut self, file_id: FileId, text: Option<Arc<str>>) {
        self.files.push((file_id, text));
    }

    pub fn set_path(&mut self, file_id: FileId, path: Option<PathBuf>) {
        self.paths.push((file_id, path));
    }
}

#[derive(Debug, Default)]
pub struct AnalysisHost {
    files: Arc<FxHashMap<FileId, Arc<ParsedFile>>>,
    paths: Arc<FxHashMap<FileId, PathBuf>>,
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

        let paths = Arc::make_mut(&mut self.paths);
        for (file_id, path) in change.paths {
            match path {
                Some(path) => paths.insert(file_id, path),
                None => paths.remove(&file_id),
            };
        }
    }

    pub fn analysis(&self) -> Analysis {
        Analysis {
            files: Arc::clone(&self.files),
            paths: Arc::clone(&self.paths),
        }
    }
}

/// Each query returns `None` for a file the host does not hold.
#[derive(Debug)]
pub struct Analysis {
    files: Arc<FxHashMap<FileId, Arc<ParsedFile>>>,
    paths: Arc<FxHashMap<FileId, PathBuf>>,
}

impl Analysis {
    /// Runs the compiler's checks over the program a file belongs to. `None`
    /// for a file with no path, whose imports cannot be found.
    pub fn check(&self, file_id: FileId) -> Option<Checked> {
        let path = self.paths.get(&file_id)?;
        let file = self.file(file_id)?;
        let documents = self
            .paths
            .iter()
            .filter_map(|(id, path)| Some((path.clone(), Arc::clone(&self.file(*id)?.text))))
            .collect();
        Some(check::check(path, &file.text, &documents))
    }

    pub fn file_text(&self, file_id: FileId) -> Option<Arc<str>> {
        Some(Arc::clone(&self.file(file_id)?.text))
    }

    pub fn diagnostics(&self, file_id: FileId) -> Option<Arc<[Diagnostic]>> {
        Some(Arc::clone(&self.file(file_id)?.diagnostics))
    }

    pub fn highlight(&self, file_id: FileId) -> Option<Vec<HlRange>> {
        Some(syntax_highlighting::highlight(
            &self.file(file_id)?.syntax(),
        ))
    }

    pub fn file_structure(&self, file_id: FileId) -> Option<Vec<StructureNode>> {
        Some(file_structure::file_structure(&self.file(file_id)?.root()))
    }

    pub fn folding_ranges(&self, file_id: FileId) -> Option<Vec<Fold>> {
        Some(folding_ranges::folding_ranges(
            &self.file(file_id)?.syntax(),
        ))
    }

    /// The ranges around a position, innermost first.
    pub fn selection_ranges(&self, position: FilePosition) -> Option<Vec<TextRange>> {
        let root = self.file(position.file_id)?.syntax();
        Some(selection_ranges::selection_ranges(&root, position.offset))
    }

    fn file(&self, file_id: FileId) -> Option<&ParsedFile> {
        self.files.get(&file_id).map(Arc::as_ref)
    }
}

#[derive(Debug)]
struct ParsedFile {
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

    fn syntax(&self) -> SyntaxNode {
        SyntaxNode::new_root(self.green.clone())
    }

    fn root(&self) -> Root {
        Root::cast(self.syntax()).expect("a parse always yields a root")
    }
}

/// A text's syntax tree, and the errors its parse reported.
pub(crate) fn parse(text: &str) -> (SyntaxNode, Vec<Diagnostic>) {
    let tokens: Vec<Token> = Lexer::new(text).collect();
    let mut sources = SourceMap::new();
    let source_id = sources.add(String::new(), String::new());
    let mut engine = DiagnosticsEngine::new();
    let root = yuzu_parser::parse(&tokens, &mut engine, source_id);
    (root, engine.into_diagnostics())
}
