//! The open documents, kept in step with the client's edits.

use std::ops::Range;
use std::path::PathBuf;

use lsp_types::notification::PublishDiagnostics;
use lsp_types::{
    DidChangeTextDocumentParams, DidCloseTextDocumentParams, DidOpenTextDocumentParams,
    PublishDiagnosticsParams, TextDocumentContentChangeEvent, Url,
};
use yuzu_ide::{Change, FileId};

use crate::global_state::GlobalState;
use crate::line_index::{LineIndex, PositionEncoding};
use crate::{RunError, from_proto, to_proto};

/// An open document, as the client last described it.
pub(crate) struct Document {
    pub(crate) url: Url,
    /// Where the document is saved. A document not yet saved has no imports
    /// to find, so the compiler cannot check it.
    pub(crate) path: Option<PathBuf>,
    pub(crate) version: i32,
    pub(crate) text: String,
    pub(crate) line_index: LineIndex,
}

impl GlobalState<'_> {
    pub(crate) fn on_did_open(
        &mut self,
        params: DidOpenTextDocumentParams,
    ) -> Result<(), RunError> {
        let document = params.text_document;
        let file_id = self.file_id(&document.uri);
        let line_index = LineIndex::new(&document.text, self.encoding);
        let path = document.uri.to_file_path().ok();
        let mut change = Change::default();
        change.set_path(file_id, path.clone());
        self.host.apply_change(change);
        self.documents.insert(
            file_id,
            Document {
                path,
                url: document.uri,
                version: document.version,
                text: document.text,
                line_index,
            },
        );
        self.update_file(file_id)
    }

    pub(crate) fn on_did_change(
        &mut self,
        params: DidChangeTextDocumentParams,
    ) -> Result<(), RunError> {
        let url = params.text_document.uri;
        let Some(document) = self
            .file_ids
            .get(&url)
            .and_then(|file_id| self.documents.get_mut(file_id))
        else {
            return self.log_error(format!("a change arrived for {url}, which is not open"));
        };

        document.version = params.text_document.version;
        let outside = apply_changes(document, params.content_changes, self.encoding);
        let file_id = self.file_id(&url);
        self.update_file(file_id)?;
        match outside {
            Some(range) => self.log_error(format!(
                "{url}: the change range {range:?} is outside the text"
            )),
            None => Ok(()),
        }
    }

    pub(crate) fn on_did_close(
        &mut self,
        params: DidCloseTextDocumentParams,
    ) -> Result<(), RunError> {
        let url = params.text_document.uri;
        let Some(file_id) = self.file_ids.get(&url).copied() else {
            return Ok(());
        };
        self.documents.remove(&file_id);
        self.checks.remove(&file_id);

        let mut change = Change::default();
        change.set_file(file_id, None);
        change.set_path(file_id, None);
        self.host.apply_change(change);
        self.send_notification::<PublishDiagnostics>(PublishDiagnosticsParams::new(
            url,
            Vec::new(),
            None,
        ))?;
        self.request_check()
    }

    /// Hands a document's text to the analysis, and asks for the program to
    /// be checked. A document with no path cannot be checked, so its syntax
    /// errors are published at once.
    pub(crate) fn update_file(&mut self, file_id: FileId) -> Result<(), RunError> {
        let document = &self.documents[&file_id];
        let mut change = Change::default();
        change.set_file(file_id, Some(document.text.as_str().into()));
        self.host.apply_change(change);

        if document.path.is_none() {
            let diagnostics = self
                .analysis()
                .diagnostics(file_id)
                .expect("the analysis holds the file just set")
                .iter()
                .filter_map(|diagnostic| {
                    to_proto::diagnostic(diagnostic, |_| {
                        Some((&document.url, &document.line_index))
                    })
                })
                .map(|(_, diagnostic)| diagnostic)
                .collect();
            let params = PublishDiagnosticsParams::new(
                document.url.clone(),
                diagnostics,
                Some(document.version),
            );
            self.send_notification::<PublishDiagnostics>(params)?;
        }
        self.request_check()
    }
}

/// Applies each change in order, each against the text the one before left.
/// The first range the text does not hold, when there is one: it stops the
/// rest, which would be relative to an edit that did not happen.
pub(crate) fn apply_changes(
    document: &mut Document,
    changes: Vec<TextDocumentContentChangeEvent>,
    encoding: PositionEncoding,
) -> Option<lsp_types::Range> {
    for change in changes {
        match change.range {
            None => document.text = change.text,
            Some(range) => {
                let Some(edit) = edit_range(document, range) else {
                    return Some(range);
                };
                document.text.replace_range(edit, &change.text);
            }
        }
        document.line_index = LineIndex::new(&document.text, encoding);
    }
    None
}

fn edit_range(document: &Document, range: lsp_types::Range) -> Option<Range<usize>> {
    let range = from_proto::text_range(&document.line_index, range)?;
    let range = Range::<usize>::from(range);
    let text = &document.text;
    (range.end <= text.len()
        && text.is_char_boundary(range.start)
        && text.is_char_boundary(range.end))
    .then_some(range)
}

#[cfg(test)]
mod tests {
    use lsp_types::{Position, Range, TextDocumentContentChangeEvent, Url};

    use super::{Document, apply_changes};
    use crate::line_index::{LineIndex, PositionEncoding};

    fn document(text: &str, encoding: PositionEncoding) -> Document {
        Document {
            url: Url::parse("file:///main.yz").unwrap(),
            path: None,
            version: 1,
            text: text.to_owned(),
            line_index: LineIndex::new(text, encoding),
        }
    }

    fn change(start: (u32, u32), end: (u32, u32), text: &str) -> TextDocumentContentChangeEvent {
        TextDocumentContentChangeEvent {
            range: Some(Range::new(
                Position::new(start.0, start.1),
                Position::new(end.0, end.1),
            )),
            range_length: None,
            text: text.to_owned(),
        }
    }

    #[test]
    fn each_change_applies_to_the_text_the_one_before_left() {
        let mut document = document("let x = 1\n", PositionEncoding::Utf8);
        let outside = apply_changes(
            &mut document,
            vec![
                change((0, 4), (0, 5), "total"),
                change((0, 12), (0, 13), "2"),
            ],
            PositionEncoding::Utf8,
        );
        assert_eq!(outside, None);
        assert_eq!(document.text, "let total = 2\n");
    }

    #[test]
    fn a_range_outside_the_text_stops_the_rest() {
        let mut document = document("let x = 1\n", PositionEncoding::Utf8);
        let outside = apply_changes(
            &mut document,
            vec![change((5, 0), (5, 1), "y"), change((0, 4), (0, 5), "z")],
            PositionEncoding::Utf8,
        );
        assert_eq!(
            outside,
            Some(Range::new(Position::new(5, 0), Position::new(5, 1)))
        );
        assert_eq!(document.text, "let x = 1\n");
    }

    #[test]
    fn a_utf16_position_counts_code_units() {
        let mut document = document("let s = \"日本\" let t = 1\n", PositionEncoding::Utf16);
        let outside = apply_changes(
            &mut document,
            vec![change((0, 17), (0, 18), "u")],
            PositionEncoding::Utf16,
        );
        assert_eq!(outside, None);
        assert_eq!(document.text, "let s = \"日本\" let u = 1\n");
    }
}
