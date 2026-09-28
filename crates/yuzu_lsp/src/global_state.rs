//! The server's state: the open documents, the analysis, the checker, and
//! what the client said it supports.

use std::path::Path;
use std::sync::Arc;

use lsp_server::{Connection, Message, Notification, Request, RequestId};
use lsp_types::notification::LogMessage;
use lsp_types::{LogMessageParams, MessageType, Url};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::Serialize;
use yuzu_ide::{Analysis, AnalysisHost, Checked, FileId};

use crate::RunError;
use crate::capabilities::{Folding, Refresh};
use crate::checker::{CheckRequest, Checker};
use crate::documents::Document;
use crate::line_index::PositionEncoding;

pub(crate) struct GlobalState<'c> {
    pub(crate) connection: &'c Connection,
    pub(crate) host: AnalysisHost,
    pub(crate) documents: FxHashMap<FileId, Document>,
    pub(crate) file_ids: FxHashMap<Url, FileId>,
    pub(crate) encoding: PositionEncoding,
    pub(crate) folding: Folding,
    pub(crate) checker: Checker,
    /// How many changes the server has seen. A check answers the one it was
    /// asked after, and is dropped once a newer change has come.
    pub(crate) generation: u64,
    /// The files that are not open and got diagnostics from the last check,
    /// so the next check can clear the ones it no longer reports.
    pub(crate) published: FxHashSet<Url>,
    /// Each open document's last check, and the version of the text it read.
    pub(crate) checks: FxHashMap<FileId, (i32, Arc<Checked>)>,
    /// Whether the client asks for inlay hints again when told to.
    pub(crate) inlay_hints: Refresh,
    /// Whether the client asks for semantic tokens again when told to.
    pub(crate) semantic_tokens: Refresh,
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

    /// Asks the checker for every open document with a path. A change to one
    /// file can change what another reports, so each is checked again.
    pub(crate) fn request_check(&mut self) -> Result<(), RunError> {
        self.generation += 1;
        let files = self
            .documents
            .iter()
            .filter(|(_, document)| document.path.is_some())
            .map(|(&file_id, document)| (file_id, document.version))
            .collect();
        let request = CheckRequest {
            generation: self.generation,
            analysis: self.analysis(),
            files,
        };
        if self.checker.request(request) {
            Ok(())
        } else {
            self.log_error("the checker stopped; only syntax errors are shown".to_owned())
        }
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

    pub(crate) fn folding(&self) -> Folding {
        self.folding
    }

    pub(crate) fn encoding(&self) -> PositionEncoding {
        self.encoding
    }

    /// An open document, its path and its last check, while that check read
    /// the text the document has now. An answer from an older check would
    /// point at offsets that have moved.
    pub(crate) fn checked_document(&self, url: &Url) -> Option<(&Document, &Path, &Checked)> {
        let (file_id, document) = self.document(url)?;
        let path = document.path.as_deref()?;
        let (version, checked) = self.checks.get(&file_id)?;
        (*version == document.version).then_some((document, path, checked.as_ref()))
    }
}
