//! The server's state: the open documents, the analysis, the checker, and
//! what the client said it supports.

use lsp_server::{Connection, Message, Notification, Request, RequestId};
use lsp_types::notification::LogMessage;
use lsp_types::{LogMessageParams, MessageType, Url};
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
            // A send fails only when the client is gone; the error holds
            // nothing but the unsent message.
            .map_err(|_unsent| RunError::disconnected())
    }

    pub(crate) fn log_error(&self, message: String) -> Result<(), RunError> {
        self.send_notification::<LogMessage>(LogMessageParams {
            typ: MessageType::ERROR,
            message,
        })
    }

    /// Asks the checker for every open document with a path, `first` before
    /// the rest. A change to one file can change what another reports, so
    /// each is checked again. A checker that stopped is started again.
    pub(crate) fn request_check(&mut self, first: Option<FileId>) -> Result<(), RunError> {
        let mut files: Vec<(FileId, i32)> = self
            .documents
            .iter()
            .filter(|(_, document)| document.path.is_some())
            .filter(|(file_id, document)| self.failed.get(file_id) != Some(&document.version))
            .map(|(&file_id, document)| (file_id, document.version))
            .collect();
        files.sort_by_key(|&(file_id, _)| (Some(file_id) != first, file_id.0));

        let request = CheckRequest {
            analysis: self.analysis(),
            files,
        };
        if let Err(stopped) = self.checker.request(request) {
            self.log_error(format!(
                "the checker stopped ({stopped:?}); it starts again"
            ))?;
            self.checker = Checker::spawn()?;
            return self.request_check(first);
        }
        Ok(())
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

    /// An open document and its last check, while that check read the text
    /// the document has now. An answer from an older check would point at
    /// offsets that have moved.
    pub(crate) fn fresh_check(&self, url: &Url) -> Option<(FileId, &Document, &Checked)> {
        let (file_id, document) = self.document(url)?;
        let (version, checked) = self.checks.get(&file_id)?;
        (*version == document.version).then_some((file_id, document, checked))
    }

    /// An open document, its last check, and how the text that check read
    /// maps onto the document's text now.
    pub(crate) fn last_check(&self, url: &Url) -> Option<(FileId, &Document, &Checked, TextShift)> {
        let (file_id, document) = self.document(url)?;
        let (_, checked) = self.checks.get(&file_id)?;
        let shift = TextShift::between(checked.file_text(file_id)?, &document.text);
        Some((file_id, document, checked, shift))
    }
}
