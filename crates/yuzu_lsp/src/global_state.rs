//! The server's state: the open documents, the analysis, the checker, and
//! what the client said it supports.

use lsp_server::{Connection, Message, Notification, Request, RequestId};
use lsp_types::notification::LogMessage;
use lsp_types::{LogMessageParams, MessageType, SemanticTokens, Url};
use rustc_hash::FxHashMap;
use serde::Serialize;
use yuzu_ide::{Analysis, AnalysisHost, Checked, FileId};

use crate::RunError;
use crate::capabilities::{Folding, Refresh};
use crate::checker::{CheckRequest, Checker};
use crate::documents::Document;
use crate::line_index::PositionEncoding;
use crate::text_shift::TextShift;

pub(crate) struct GlobalState<'c> {
    pub(crate) connection: &'c Connection,
    pub(crate) host: AnalysisHost,
    pub(crate) documents: FxHashMap<FileId, Document>,
    pub(crate) file_ids: FxHashMap<Url, FileId>,
    pub(crate) encoding: PositionEncoding,
    pub(crate) folding: Folding,
    pub(crate) checker: Checker,
    /// Each open document's last check, and the version of the text it read.
    pub(crate) checks: FxHashMap<FileId, (i32, Checked)>,
    /// What each open document's last check found in files that are not
    /// open, by the file's URL.
    pub(crate) closed_diagnostics: FxHashMap<FileId, FxHashMap<Url, Vec<lsp_types::Diagnostic>>>,
    /// The version of each document whose check panicked. It is not checked
    /// again until its text changes.
    pub(crate) failed: FxHashMap<FileId, i32>,
    /// Each open document whose last check read a file that changed since,
    /// with the number of the last request that asked for it again. Only a
    /// check from that request or a later one is current.
    pub(crate) stale: FxHashMap<FileId, u64>,
    /// The number of the last check request.
    pub(crate) generation: u64,
    /// The semantic tokens last sent for each open document, with the
    /// result id a delta request names them by.
    pub(crate) semantic_tokens: FxHashMap<FileId, SemanticTokens>,
    /// The result id the next semantic tokens get.
    pub(crate) next_result_id: u64,
    pub(crate) inlay_hint_refresh: Refresh,
    pub(crate) semantic_tokens_refresh: Refresh,
    /// The id of the next request the server sends the client.
    pub(crate) next_request: i32,
}

impl GlobalState<'_> {
    pub(crate) fn send_request<R>(&mut self, params: R::Params) -> Result<(), RunError>
    where
        R: lsp_types::request::Request,
        R::Params: Serialize,
    {
        self.next_request += 1;
        let id = RequestId::from(self.next_request);
        self.send(Request::new(id, R::METHOD.to_owned(), params).into())
    }

    pub(crate) fn send_notification<N>(&self, params: N::Params) -> Result<(), RunError>
    where
        N: lsp_types::notification::Notification,
        N::Params: Serialize,
    {
        self.send(Notification::new(N::METHOD.to_owned(), params).into())
    }

    pub(crate) fn send(&self, message: Message) -> Result<(), RunError> {
        self.connection
            .sender
            .send(message)
            // A send fails only when the client is gone. The error holds only
            // the message that was not sent.
            .map_err(|_unsent| RunError::disconnected())
    }

    pub(crate) fn log_error(&self, message: String) -> Result<(), RunError> {
        self.send_notification::<LogMessage>(LogMessageParams {
            typ: MessageType::ERROR,
            message,
        })
    }

    /// Asks the checker for the open documents that `changed` can change:
    /// the document itself, each one whose last check read it, and each one
    /// not checked yet. `None` asks for every open document, as after a
    /// change on disk. `changed` is checked first. A checker that stopped
    /// is started again.
    pub(crate) fn request_check(&mut self, changed: Option<FileId>) -> Result<(), RunError> {
        self.generation += 1;
        let changed_path =
            changed.and_then(|file_id| self.documents.get(&file_id)?.path.as_deref());
        for (&file_id, document) in &self.documents {
            let reads_change = match (changed, changed_path) {
                (None, _) => true,
                (Some(changed), _) if changed == file_id => true,
                (Some(_), Some(path)) => self
                    .checks
                    .get(&file_id)
                    .is_none_or(|(_, checked)| checked.source_of(path).is_some()),
                (Some(_), None) => !self.checks.contains_key(&file_id),
            };
            if reads_change && document.path.is_some() {
                self.stale.insert(file_id, self.generation);
            }
        }
        self.send_stale(changed)
    }

    /// Sends the checker every stale document, `first` before the rest.
    fn send_stale(&mut self, first: Option<FileId>) -> Result<(), RunError> {
        let mut files: Vec<(FileId, i32)> = self
            .stale
            .keys()
            .filter_map(|file_id| Some((*file_id, self.documents.get(file_id)?)))
            .filter(|(file_id, document)| self.failed.get(file_id) != Some(&document.version))
            .map(|(file_id, document)| (file_id, document.version))
            .collect();
        files.sort_by_key(|&(file_id, _)| (Some(file_id) != first, file_id.0));

        let request = CheckRequest {
            analysis: self.analysis(),
            files,
            generation: self.generation,
        };
        if let Err(stopped) = self.checker.request(request) {
            self.log_error(format!(
                "the checker stopped ({stopped:?}); it starts again"
            ))?;
            self.checker = Checker::spawn()?;
            return self.send_stale(first);
        }
        Ok(())
    }

    /// Keeps the tokens sent for a document, under a new result id.
    pub(crate) fn remember_tokens(
        &mut self,
        file_id: FileId,
        mut tokens: SemanticTokens,
    ) -> &SemanticTokens {
        self.next_result_id += 1;
        tokens.result_id = Some(self.next_result_id.to_string());
        self.semantic_tokens.insert(file_id, tokens);
        &self.semantic_tokens[&file_id]
    }

    pub(crate) fn file_id(&mut self, url: &Url) -> FileId {
        let next = FileId(u32::try_from(self.file_ids.len()).expect("fewer than 2^32 files"));
        *self.file_ids.entry(url.clone()).or_insert(next)
    }

    pub(crate) fn analysis(&self) -> Analysis {
        self.host.analysis()
    }

    pub(crate) fn document(&self, url: &Url) -> Option<(FileId, &Document)> {
        let file_id = *self.file_ids.get(url)?;
        Some((file_id, self.documents.get(&file_id)?))
    }

    pub(crate) fn is_open(&self, url: &Url) -> bool {
        self.document(url).is_some()
    }

    /// An open document and its last check, when that check read the current
    /// text. An answer from an older check points at offsets that moved.
    pub(crate) fn fresh_check(&self, url: &Url) -> Option<(FileId, &Document, &Checked)> {
        let (file_id, document) = self.document(url)?;
        let (version, checked) = self.checks.get(&file_id)?;
        (*version == document.version).then_some((file_id, document, checked))
    }

    /// An open document and its last check, of any text.
    pub(crate) fn latest_check(&self, url: &Url) -> Option<(FileId, &Document, &Checked)> {
        let (file_id, document) = self.document(url)?;
        let (_, checked) = self.checks.get(&file_id)?;
        Some((file_id, document, checked))
    }

    /// An open document, its last check, and how the text of that check maps
    /// to the current text.
    pub(crate) fn last_check(&self, url: &Url) -> Option<(FileId, &Document, &Checked, TextShift)> {
        let (file_id, document, checked) = self.latest_check(url)?;
        let shift = TextShift::between(checked.file_text(file_id)?, &document.text);
        Some((file_id, document, checked, shift))
    }
}
