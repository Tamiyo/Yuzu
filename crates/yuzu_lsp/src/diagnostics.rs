//! A check's diagnostics, published to the files they point into.

use std::sync::Arc;

use lsp_types::notification::PublishDiagnostics;
use lsp_types::request::{InlayHintRefreshRequest, SemanticTokensRefresh};
use lsp_types::{PublishDiagnosticsParams, Url};
use rustc_hash::{FxHashMap, FxHashSet};
use yuzu_diagnostics::source_map::SourceId;
use yuzu_ide::{Checked, FileId};

use crate::capabilities::Refresh;
use crate::checker::CheckResult;
use crate::global_state::GlobalState;
use crate::line_index::{LineIndex, PositionEncoding};
use crate::{RunError, to_proto};

impl GlobalState<'_> {
    /// Publishes a check's diagnostics. A file that is open takes them from
    /// its own check only; a file that is not open takes them from the first
    /// check that reached it, so a file two checks share is not reported
    /// twice.
    pub(crate) fn on_checked(&mut self, result: CheckResult) -> Result<(), RunError> {
        if result.generation != self.generation {
            return Ok(());
        }

        let mut publish: FxHashMap<Url, (Option<i32>, Vec<lsp_types::Diagnostic>)> =
            FxHashMap::default();
        let mut claimed: FxHashMap<Url, FileId> = FxHashMap::default();
        for (file_id, version, checked) in &result.checks {
            let Some(document) = self.documents.get(file_id) else {
                continue;
            };
            publish
                .entry(document.url.clone())
                .or_insert_with(|| (Some(*version), Vec::new()));

            let files = CheckedFiles::new(checked, self.encoding);
            for diagnostic in checked.diagnostics() {
                let Some((url, diagnostic)) = to_proto::diagnostic(diagnostic, |id| files.get(id))
                else {
                    continue;
                };
                let own = *url == document.url;
                let open = self
                    .file_ids
                    .get(url)
                    .is_some_and(|id| self.documents.contains_key(id));
                let first = *claimed.entry(url.clone()).or_insert(*file_id) == *file_id;
                if own || (!open && first) {
                    publish
                        .entry(url.clone())
                        .or_insert_with(|| (None, Vec::new()))
                        .1
                        .push(diagnostic);
                }
            }
        }

        let closed: FxHashSet<Url> = publish
            .iter()
            .filter(|(_, (version, _))| version.is_none())
            .map(|(url, _)| url.clone())
            .collect();
        for url in self.published.difference(&closed) {
            publish
                .entry(url.clone())
                .or_insert_with(|| (None, Vec::new()));
        }
        self.published = closed;

        for (url, (version, diagnostics)) in publish {
            self.send_notification::<PublishDiagnostics>(PublishDiagnosticsParams::new(
                url,
                diagnostics,
                version,
            ))?;
        }

        for (file_id, version, checked) in result.checks {
            if self.documents.contains_key(&file_id) {
                self.checks.insert(file_id, (version, Arc::new(checked)));
            }
        }
        if self.inlay_hints == Refresh::Supported {
            self.send_request::<InlayHintRefreshRequest>(())?;
        }
        if self.semantic_tokens == Refresh::Supported {
            self.send_request::<SemanticTokensRefresh>(())?;
        }
        Ok(())
    }
}

/// The files a check read, for the diagnostics that point into them.
struct CheckedFiles {
    files: Vec<(SourceId, Url, LineIndex)>,
}

impl CheckedFiles {
    /// The files the diagnostics' labels name; a source with no file on
    /// disk has no URL and is left out.
    fn new(checked: &Checked, encoding: PositionEncoding) -> Self {
        let mut files: Vec<(SourceId, Url, LineIndex)> = Vec::new();
        let labels = checked
            .diagnostics()
            .iter()
            .flat_map(|diagnostic| &diagnostic.labels);
        for label in labels {
            let source = label.span.source_id;
            if files.iter().any(|(seen, _, _)| *seen == source) {
                continue;
            }
            let Some(url) = checked
                .path(source)
                .and_then(|path| Url::from_file_path(path).ok())
            else {
                continue;
            };
            files.push((source, url, LineIndex::new(checked.text(source), encoding)));
        }
        CheckedFiles { files }
    }

    fn get(&self, source: SourceId) -> Option<(&Url, &LineIndex)> {
        self.files
            .iter()
            .find(|(id, _, _)| *id == source)
            .map(|(_, url, line_index)| (url, line_index))
    }
}
