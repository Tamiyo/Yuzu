//! The server's state, and the loop that feeds it messages until the client
//! shuts the server down.

use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crossbeam_channel::select;
use lsp_server::{Connection, ErrorCode, Message, Notification, Request, RequestId, Response};
use lsp_types::notification::{
    DidChangeTextDocument, DidCloseTextDocument, DidOpenTextDocument, LogMessage,
    Notification as _, PublishDiagnostics,
};
use lsp_types::request::{
    DocumentHighlightRequest, DocumentSymbolRequest, FoldingRangeRequest, GotoDefinition,
    HoverRequest, InlayHintRefreshRequest, InlayHintRequest, References, Request as _,
    SelectionRangeRequest, SemanticTokensFullRequest, SemanticTokensRefresh,
};
use lsp_types::{
    DidChangeTextDocumentParams, DidCloseTextDocumentParams, DidOpenTextDocumentParams,
    InitializeParams, InitializeResult, LogMessageParams, MessageType, PublishDiagnosticsParams,
    ServerInfo, TextDocumentContentChangeEvent, Url,
};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::Serialize;
use serde::de::DeserializeOwned;
use yuzu_diagnostics::source_map::SourceId;
use yuzu_ide::{Analysis, AnalysisHost, Change, Checked, FileId};

use crate::capabilities::{self, Folding, Refresh};
use crate::checker::{CheckRequest, CheckResult, Checker};
use crate::line_index::{LineIndex, PositionEncoding};
use crate::{RunError, from_proto, handlers, to_proto};

/// Serves one client over `connection`, from its `initialize` request to its
/// `exit` notification.
///
/// # Errors
///
/// When the handshake fails, when the initialize params do not parse, or
/// when the client goes away without asking the server to shut down.
pub fn run(connection: &Connection) -> Result<(), RunError> {
    let (id, params) = connection.initialize_start().map_err(RunError::protocol)?;
    let params: InitializeParams =
        serde_json::from_value(params).map_err(RunError::initialize_params)?;

    let encoding = capabilities::position_encoding(&params.capabilities);
    let result = InitializeResult {
        capabilities: capabilities::server_capabilities(encoding),
        server_info: Some(ServerInfo {
            name: "yuzu-lsp".to_owned(),
            version: Some(env!("CARGO_PKG_VERSION").to_owned()),
        }),
    };
    let result = serde_json::to_value(result).expect("an initialize result serializes");
    connection
        .initialize_finish(id, result)
        .map_err(RunError::protocol)?;

    let checker = Checker::spawn();
    let mut checks = checker.results().clone();
    let mut state = GlobalState {
        connection,
        host: AnalysisHost::default(),
        documents: FxHashMap::default(),
        file_ids: FxHashMap::default(),
        encoding,
        folding: capabilities::folding(&params.capabilities),
        checker,
        generation: 0,
        published: FxHashSet::default(),
        checks: FxHashMap::default(),
        inlay_hints: capabilities::inlay_hint_refresh(&params.capabilities),
        semantic_tokens: capabilities::semantic_tokens_refresh(&params.capabilities),
        next_request: 0,
    };
    loop {
        select! {
            recv(connection.receiver) -> message => {
                let Ok(message) = message else {
                    return Err(RunError::disconnected());
                };
                match message {
                    Message::Request(request) => {
                        if connection
                            .handle_shutdown(&request)
                            .map_err(RunError::protocol)?
                        {
                            return Ok(());
                        }
                        state.on_request(request)?;
                    }
                    Message::Notification(notification) => {
                        state.on_notification(notification)?;
                    }
                    Message::Response(_) => {}
                }
            }
            recv(checks) -> result => match result {
                Ok(result) => state.on_checked(result)?,
                Err(_) => {
                    checks = crossbeam_channel::never();
                    state.log_error("the checker stopped; only syntax errors are shown".to_owned())?;
                }
            },
        }
    }
}

pub(crate) struct GlobalState<'c> {
    connection: &'c Connection,
    host: AnalysisHost,
    documents: FxHashMap<FileId, Document>,
    file_ids: FxHashMap<Url, FileId>,
    encoding: PositionEncoding,
    folding: Folding,
    checker: Checker,
    /// How many changes the server has seen. A check answers the one it was
    /// asked after, and is dropped once a newer change has come.
    generation: u64,
    /// The files that are not open and got diagnostics from the last check,
    /// so the next check can clear the ones it no longer reports.
    published: FxHashSet<Url>,
    /// Each open document's last check, and the version of the text it read.
    checks: FxHashMap<FileId, (i32, Arc<Checked>)>,
    /// Whether the client asks for inlay hints again when told to.
    inlay_hints: Refresh,
    /// Whether the client asks for semantic tokens again when told to.
    semantic_tokens: Refresh,
    /// The id of the next request the server sends the client.
    next_request: i32,
}

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
    fn on_request(&mut self, request: Request) -> Result<(), RunError> {
        let response = match request.method.as_str() {
            DocumentSymbolRequest::METHOD => {
                self.respond::<DocumentSymbolRequest>(request, handlers::document_symbol)
            }
            FoldingRangeRequest::METHOD => {
                self.respond::<FoldingRangeRequest>(request, handlers::folding_range)
            }
            SelectionRangeRequest::METHOD => {
                self.respond::<SelectionRangeRequest>(request, handlers::selection_range)
            }
            SemanticTokensFullRequest::METHOD => {
                self.respond::<SemanticTokensFullRequest>(request, handlers::semantic_tokens_full)
            }
            GotoDefinition::METHOD => {
                self.respond::<GotoDefinition>(request, handlers::goto_definition)
            }
            References::METHOD => self.respond::<References>(request, handlers::references),
            DocumentHighlightRequest::METHOD => {
                self.respond::<DocumentHighlightRequest>(request, handlers::document_highlight)
            }
            HoverRequest::METHOD => self.respond::<HoverRequest>(request, handlers::hover),
            InlayHintRequest::METHOD => {
                self.respond::<InlayHintRequest>(request, handlers::inlay_hint)
            }
            _ => Response::new_err(
                request.id,
                ErrorCode::MethodNotFound as i32,
                format!("unknown request: {}", request.method),
            ),
        };
        self.send(response.into())
    }

    fn on_notification(&mut self, notification: Notification) -> Result<(), RunError> {
        match notification.method.as_str() {
            DidOpenTextDocument::METHOD => {
                self.notify::<DidOpenTextDocument>(notification, Self::on_did_open)
            }
            DidChangeTextDocument::METHOD => {
                self.notify::<DidChangeTextDocument>(notification, Self::on_did_change)
            }
            DidCloseTextDocument::METHOD => {
                self.notify::<DidCloseTextDocument>(notification, Self::on_did_close)
            }
            _ => Ok(()),
        }
    }

    fn on_did_open(&mut self, params: DidOpenTextDocumentParams) -> Result<(), RunError> {
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

    fn on_did_change(&mut self, params: DidChangeTextDocumentParams) -> Result<(), RunError> {
        let url = params.text_document.uri;
        let Some(document) = self
            .file_ids
            .get(&url)
            .and_then(|file_id| self.documents.get_mut(file_id))
        else {
            return self.log_error(format!("a change arrived for {url}, which is not open"));
        };

        document.version = params.text_document.version;
        let applied = apply_changes(document, params.content_changes, self.encoding);
        let file_id = self.file_id(&url);
        self.update_file(file_id)?;
        match applied {
            Ok(()) => Ok(()),
            Err(message) => self.log_error(format!("{url}: {message}")),
        }
    }

    fn on_did_close(&mut self, params: DidCloseTextDocumentParams) -> Result<(), RunError> {
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

    /// Publishes a check's diagnostics. A file that is open takes them from
    /// its own check only; a file that is not open takes them from the first
    /// check that reached it, so a file two checks share is not reported
    /// twice.
    fn on_checked(&mut self, result: CheckResult) -> Result<(), RunError> {
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

    fn respond<R>(&self, request: Request, handler: fn(&Self, R::Params) -> R::Result) -> Response
    where
        R: lsp_types::request::Request,
        R::Params: DeserializeOwned,
        R::Result: Serialize,
    {
        match serde_json::from_value(request.params) {
            Ok(params) => Response::new_ok(request.id, handler(self, params)),
            Err(error) => Response::new_err(
                request.id,
                ErrorCode::InvalidParams as i32,
                format!("invalid {} params: {error}", R::METHOD),
            ),
        }
    }

    fn notify<N>(
        &mut self,
        notification: Notification,
        handler: fn(&mut Self, N::Params) -> Result<(), RunError>,
    ) -> Result<(), RunError>
    where
        N: lsp_types::notification::Notification,
        N::Params: DeserializeOwned,
    {
        match serde_json::from_value(notification.params) {
            Ok(params) => handler(self, params),
            Err(error) => self.log_error(format!("invalid {} params: {error}", N::METHOD)),
        }
    }

    /// Hands a document's text to the analysis, and asks for the program to
    /// be checked. A document with no path cannot be checked, so its syntax
    /// errors are published at once.
    fn update_file(&mut self, file_id: FileId) -> Result<(), RunError> {
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

    /// Asks the checker for every open document with a path. A change to one
    /// file can change what another reports, so each is checked again.
    fn request_check(&mut self) -> Result<(), RunError> {
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

    fn log_error(&self, message: String) -> Result<(), RunError> {
        self.send_notification::<LogMessage>(LogMessageParams {
            typ: MessageType::ERROR,
            message,
        })
    }

    fn send_request<R>(&mut self, params: R::Params) -> Result<(), RunError>
    where
        R: lsp_types::request::Request,
        R::Params: Serialize,
    {
        self.next_request += 1;
        let id = RequestId::from(self.next_request);
        self.send(Request::new(id, R::METHOD.to_owned(), params).into())
    }

    fn send_notification<N>(&self, params: N::Params) -> Result<(), RunError>
    where
        N: lsp_types::notification::Notification,
        N::Params: Serialize,
    {
        self.send(Notification::new(N::METHOD.to_owned(), params).into())
    }

    fn send(&self, message: Message) -> Result<(), RunError> {
        self.connection
            .sender
            .send(message)
            .map_err(|_| RunError::disconnected())
    }

    fn file_id(&mut self, url: &Url) -> FileId {
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

/// Applies each change in order, each against the text the one before left.
/// A range the text does not hold stops the rest, which would be relative to
/// an edit that did not happen.
fn apply_changes(
    document: &mut Document,
    changes: Vec<TextDocumentContentChangeEvent>,
    encoding: PositionEncoding,
) -> Result<(), String> {
    for change in changes {
        match change.range {
            None => document.text = change.text,
            Some(range) => {
                let range = edit_range(document, range)
                    .ok_or_else(|| format!("the change range {range:?} is outside the text"))?;
                document.text.replace_range(range, &change.text);
            }
        }
        document.line_index = LineIndex::new(&document.text, encoding);
    }
    Ok(())
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
