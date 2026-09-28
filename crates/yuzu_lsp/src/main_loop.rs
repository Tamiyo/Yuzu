//! The loop that feeds the server messages until the client shuts it down,
//! and the dispatch of each request and notification to its handler.

use std::panic::{self, AssertUnwindSafe};

use crossbeam_channel::{RecvError, select};
use lsp_server::{Connection, ErrorCode, Message, Notification, Request, Response};
use lsp_types::notification::{
    DidChangeTextDocument, DidChangeWatchedFiles, DidCloseTextDocument, DidOpenTextDocument,
    Notification as _,
};
use lsp_types::request::{
    DocumentHighlightRequest, DocumentSymbolRequest, FoldingRangeRequest, GotoDefinition,
    HoverRequest, InlayHintRequest, References, RegisterCapability, Request as _,
    SelectionRangeRequest, SemanticTokensFullRequest,
};
use lsp_types::{InitializeParams, InitializeResult, RegistrationParams, ServerInfo};
use rustc_hash::FxHashMap;
use serde::Serialize;
use serde::de::DeserializeOwned;
use yuzu_ide::AnalysisHost;

use crate::capabilities;
use crate::checker::{CheckResult, Checker, panic_message};
use crate::global_state::GlobalState;
use crate::{RunError, handlers};

/// Serves one client over `connection`, from its `initialize` request to its
/// `exit` notification.
///
/// # Errors
///
/// When the handshake fails, when the initialize params do not parse, when
/// the checker thread cannot start, or when the client goes away without
/// asking the server to shut down.
///
/// # Panics
///
/// Panics if the initialize result does not serialize, which it always does.
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

    let mut state = GlobalState {
        connection,
        host: AnalysisHost::default(),
        documents: FxHashMap::default(),
        file_ids: FxHashMap::default(),
        encoding,
        folding: capabilities::folding(&params.capabilities),
        checker: Checker::spawn()?,
        checks: FxHashMap::default(),
        closed_diagnostics: FxHashMap::default(),
        failed: FxHashMap::default(),
        inlay_hint_refresh: capabilities::inlay_hint_refresh(&params.capabilities),
        semantic_tokens_refresh: capabilities::semantic_tokens_refresh(&params.capabilities),
        next_request: 0,
    };
    if capabilities::watches_files(&params.capabilities) {
        state.send_request::<RegisterCapability>(RegistrationParams {
            registrations: vec![capabilities::watched_files_registration()],
        })?;
    }
    loop {
        let checks = state.checker.results().clone();
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
            recv(checks) -> result => state.on_check_result(result)?,
        }
    }
}

impl GlobalState<'_> {
    fn on_check_result(&mut self, result: Result<CheckResult, RecvError>) -> Result<(), RunError> {
        match result {
            Ok(result) => self.on_checked(result),
            Err(RecvError) => self.on_checker_lost(),
        }
    }

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
            DidChangeWatchedFiles::METHOD => self
                .notify::<DidChangeWatchedFiles>(notification, Self::on_did_change_watched_files),
            _ => Ok(()),
        }
    }

    /// Answers a request. A handler that panics answers with an error, and
    /// the server goes on.
    fn respond<R>(&self, request: Request, handler: fn(&Self, &R::Params) -> R::Result) -> Response
    where
        R: lsp_types::request::Request,
        R::Params: DeserializeOwned,
        R::Result: Serialize,
    {
        let params: R::Params = match serde_json::from_value(request.params) {
            Ok(params) => params,
            Err(error) => {
                return Response::new_err(
                    request.id,
                    ErrorCode::InvalidParams as i32,
                    format!("invalid {} params: {error}", R::METHOD),
                );
            }
        };
        match panic::catch_unwind(AssertUnwindSafe(|| handler(self, &params))) {
            Ok(result) => Response::new_ok(request.id, result),
            Err(payload) => Response::new_err(
                request.id,
                ErrorCode::InternalError as i32,
                format!(
                    "{} panicked: {}",
                    R::METHOD,
                    panic_message(payload.as_ref())
                ),
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
}
