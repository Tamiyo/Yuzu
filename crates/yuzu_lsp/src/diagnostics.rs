//! A check's diagnostics, published to the files they point into.

use lsp_types::notification::PublishDiagnostics;
use lsp_types::request::{InlayHintRefreshRequest, SemanticTokensRefresh};
use lsp_types::{PublishDiagnosticsParams, Url};
use rustc_hash::{FxHashMap, FxHashSet};
use yuzu_diagnostics::source_map::SourceId;
use yuzu_ide::{Checked, FileId};

use crate::capabilities::Refresh;
use crate::checker::{CheckResult, Checker};
use crate::global_state::GlobalState;
use crate::line_index::{LineIndex, PositionEncoding};
use crate::{RunError, to_proto};

impl GlobalState<'_> {
    /// Takes one document's check. A check of a text the document no longer
    /// has is dropped.
    pub(crate) fn on_checked(&mut self, result: CheckResult) -> Result<(), RunError> {
        match result {
            CheckResult::Checked {
                file_id,
                version,
                checked,
            } => self.on_document_checked(file_id, version, *checked),
            CheckResult::Panicked {
                file_id,
                version,
                message,
            } => self.on_check_panicked(file_id, version, &message),
        }
    }

    /// The checker thread went away without a word: it starts again.
    pub(crate) fn on_checker_lost(&mut self) -> Result<(), RunError> {
        self.log_error("the checker stopped; it starts again".to_owned())?;
        self.checker = Checker::spawn()?;
        self.request_check(None)
    }

    /// Publishes what a document's check found in the document itself. A
    /// file that is not open takes its diagnostics from one open document's
    /// check, so a file two checks share is not reported twice.
    fn on_document_checked(
        &mut self,
        file_id: FileId,
        version: i32,
        checked: Checked,
    ) -> Result<(), RunError> {
        let Some(document) = self.documents.get(&file_id) else {
            return Ok(());
        };
        if document.version != version {
            return Ok(());
        }
        self.failed.remove(&file_id);

        let files = CheckedFiles::new(&checked, self.encoding);
        let mut own = Vec::new();
        let mut closed: FxHashMap<Url, Vec<lsp_types::Diagnostic>> = FxHashMap::default();
        for diagnostic in checked.diagnostics() {
            let Some((url, diagnostic)) = to_proto::diagnostic(diagnostic, |id| files.get(id))
            else {
                continue;
            };
            if *url == document.url {
                own.push(diagnostic);
            } else if !self.is_open(url) {
                closed.entry(url.clone()).or_default().push(diagnostic);
            }
        }
        self.send_notification::<PublishDiagnostics>(PublishDiagnosticsParams::new(
            document.url.clone(),
            own,
            Some(version),
        ))?;

        let mut touched: FxHashSet<Url> = closed.keys().cloned().collect();
        if let Some(previous) = self.closed_diagnostics.insert(file_id, closed) {
            touched.extend(previous.into_keys());
        }
        self.publish_closed(touched)?;

        self.checks.insert(file_id, (version, checked));
        if self.inlay_hint_refresh == Refresh::Supported {
            self.send_request::<InlayHintRefreshRequest>(())?;
        }
        if self.semantic_tokens_refresh == Refresh::Supported {
            self.send_request::<SemanticTokensRefresh>(())?;
        }
        Ok(())
    }

    /// The document keeps its syntax errors, and is not checked again until
    /// its text changes. The rest are checked on a new thread.
    fn on_check_panicked(
        &mut self,
        file_id: FileId,
        version: i32,
        message: &str,
    ) -> Result<(), RunError> {
        let url = self
            .documents
            .get(&file_id)
            .map_or_else(String::new, |document| document.url.to_string());
        self.log_error(format!(
            "the compiler panicked while it checked {url}: {message}"
        ))?;
        self.failed.insert(file_id, version);
        if self.documents.contains_key(&file_id) {
            self.publish_syntax_diagnostics(file_id)?;
        }
        self.checker = Checker::spawn()?;
        self.request_check(None)
    }

    /// Publishes each file in `urls` that is not open, with the diagnostics
    /// the open document with the lowest id found in it.
    pub(crate) fn publish_closed(&self, urls: FxHashSet<Url>) -> Result<(), RunError> {
        for url in urls {
            if self.is_open(&url) {
                continue;
            }
            let diagnostics = self
                .closed_diagnostics
                .iter()
                .filter_map(|(file_id, closed)| Some((file_id.0, closed.get(&url)?)))
                .min_by_key(|&(id, _)| id)
                .map(|(_, diagnostics)| diagnostics.clone())
                .unwrap_or_default();
            self.send_notification::<PublishDiagnostics>(PublishDiagnosticsParams::new(
                url,
                diagnostics,
                None,
            ))?;
        }
        Ok(())
    }
}

/// The files a check read, for the diagnostics that point into them.
struct CheckedFiles {
    files: FxHashMap<SourceId, (Url, LineIndex)>,
}

impl CheckedFiles {
    /// The files the diagnostics' labels name; a source with no file on
    /// disk has no URL and is left out.
    fn new(checked: &Checked, encoding: PositionEncoding) -> Self {
        let mut files = FxHashMap::default();
        let labels = checked
            .diagnostics()
            .iter()
            .flat_map(|diagnostic| &diagnostic.labels);
        for label in labels {
            let source = label.span.source_id;
            if files.contains_key(&source) {
                continue;
            }
            let Some(url) = checked
                .path(source)
                .and_then(|path| Url::from_file_path(path).ok())
            else {
                continue;
            };
            files.insert(
                source,
                (url, LineIndex::new(checked.text(source), encoding)),
            );
        }
        CheckedFiles { files }
    }

    fn get(&self, source: SourceId) -> Option<(&Url, &LineIndex)> {
        self.files
            .get(&source)
            .map(|(url, line_index)| (url, line_index))
    }
}
