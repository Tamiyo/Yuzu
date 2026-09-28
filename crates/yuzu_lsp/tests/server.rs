use std::thread::{self, JoinHandle};
use std::time::Duration;

use expect_test::{Expect, expect};
use lsp_server::{Connection, Message, Notification, Request, RequestId, Response};
use lsp_types::notification::{
    DidChangeTextDocument, DidOpenTextDocument, Exit, Initialized, PublishDiagnostics,
};
use lsp_types::request::{
    DocumentSymbolRequest, GotoDefinition, HoverRequest, Initialize, InlayHintRequest,
    SemanticTokensFullRequest, Shutdown,
};
use lsp_types::{
    ClientCapabilities, DidChangeTextDocumentParams, DidOpenTextDocumentParams,
    DocumentSymbolParams, DocumentSymbolResponse, GeneralClientCapabilities, InitializeParams,
    PartialResultParams, Position, PositionEncodingKind, PublishDiagnosticsParams, Range,
    SemanticTokensParams, SemanticTokensResult, TextDocumentContentChangeEvent,
    TextDocumentIdentifier, TextDocumentItem, Url, VersionedTextDocumentIdentifier,
    WorkDoneProgressParams,
};
use serde::Serialize;
use serde::de::DeserializeOwned;

const TIMEOUT: Duration = Duration::from_secs(10);

struct Client {
    connection: Connection,
    server: JoinHandle<Result<(), yuzu_lsp::RunError>>,
    next_id: i32,
}

impl Client {
    fn start(capabilities: ClientCapabilities) -> Self {
        let (connection, server) = Connection::memory();
        let server = thread::spawn(move || yuzu_lsp::run(&server));
        let mut client = Client {
            connection,
            server,
            next_id: 0,
        };

        client.request::<Initialize>(InitializeParams {
            capabilities,
            ..InitializeParams::default()
        });
        client.notify::<Initialized>(lsp_types::InitializedParams {});
        client
    }

    fn open(&self, text: &str) {
        self.open_at(url(), text);
    }

    fn open_at(&self, url: Url, text: &str) {
        self.notify::<DidOpenTextDocument>(DidOpenTextDocumentParams {
            text_document: TextDocumentItem::new(url, "yuzu".to_owned(), 1, text.to_owned()),
        });
    }

    fn diagnostics_for(&self, url: &Url) -> PublishDiagnosticsParams {
        loop {
            let params = self.notification::<PublishDiagnostics>();
            if params.uri == *url {
                return params;
            }
        }
    }

    fn request<R>(&mut self, params: R::Params) -> R::Result
    where
        R: lsp_types::request::Request,
        R::Params: Serialize,
        R::Result: DeserializeOwned,
    {
        let response = self.raw_request(R::METHOD, params);
        let result = response.response_result.expect("the request succeeds");
        serde_json::from_value(result).expect("the result parses")
    }

    fn raw_request(&mut self, method: &str, params: impl Serialize) -> Response {
        self.next_id += 1;
        let id = RequestId::from(self.next_id);
        let request = Request::new(id.clone(), method.to_owned(), params);
        self.connection.sender.send(request.into()).unwrap();
        loop {
            match self.receive() {
                Message::Response(response) if response.id == id => return response,
                Message::Response(_) | Message::Notification(_) | Message::Request(_) => {}
            }
        }
    }

    fn notify<N>(&self, params: N::Params)
    where
        N: lsp_types::notification::Notification,
        N::Params: Serialize,
    {
        let notification = Notification::new(N::METHOD.to_owned(), params);
        self.connection.sender.send(notification.into()).unwrap();
    }

    fn notification<N>(&self) -> N::Params
    where
        N: lsp_types::notification::Notification,
        N::Params: DeserializeOwned,
    {
        loop {
            if let Message::Notification(notification) = self.receive()
                && notification.method == N::METHOD
            {
                return serde_json::from_value(notification.params).unwrap();
            }
        }
    }

    fn receive(&self) -> Message {
        self.connection
            .receiver
            .recv_timeout(TIMEOUT)
            .expect("the server answers in time")
    }

    fn shutdown(mut self) {
        self.request::<Shutdown>(());
        self.notify::<Exit>(());
        self.server
            .join()
            .expect("the server thread does not panic")
            .expect("the server exits cleanly");
    }
}

fn url() -> Url {
    Url::parse("file:///work/main.yz").unwrap()
}

fn document() -> TextDocumentIdentifier {
    TextDocumentIdentifier::new(url())
}

fn check_diagnostics(params: &PublishDiagnosticsParams, expected: &Expect) {
    let rendered: Vec<String> = params
        .diagnostics
        .iter()
        .map(|diagnostic| {
            let Range { start, end } = diagnostic.range;
            format!(
                "{}:{}-{}:{} {}",
                start.line, start.character, end.line, end.character, diagnostic.message
            )
        })
        .collect();
    expected.assert_eq(&rendered.join("\n"));
}

fn symbols(client: &mut Client) -> Vec<lsp_types::DocumentSymbol> {
    let response = client.request::<DocumentSymbolRequest>(DocumentSymbolParams {
        text_document: document(),
        work_done_progress_params: WorkDoneProgressParams::default(),
        partial_result_params: PartialResultParams::default(),
    });
    match response {
        Some(DocumentSymbolResponse::Nested(symbols)) => symbols,
        Some(DocumentSymbolResponse::Flat(_)) | None => panic!("the server nests its symbols"),
    }
}

#[test]
fn an_open_file_publishes_its_parse_errors() {
    let client = Client::start(ClientCapabilities::default());
    client.open("let x = 1\nlet = 2\n");

    let params = client.notification::<PublishDiagnostics>();
    assert_eq!(params.version, Some(1));
    check_diagnostics(
        &params,
        &expect!["1:4-1:5 expected one of mut, identifier, found ="],
    );
    client.shutdown();
}

#[test]
fn a_change_is_applied_where_the_client_says() {
    let mut client = Client::start(ClientCapabilities::default());
    client.open("let x = 1\n");
    client.notification::<PublishDiagnostics>();

    client.notify::<DidChangeTextDocument>(DidChangeTextDocumentParams {
        text_document: VersionedTextDocumentIdentifier::new(url(), 2),
        content_changes: vec![TextDocumentContentChangeEvent {
            range: Some(Range::new(Position::new(0, 4), Position::new(0, 5))),
            range_length: None,
            text: "total".to_owned(),
        }],
    });
    let params = client.notification::<PublishDiagnostics>();
    assert_eq!(params.version, Some(2));
    check_diagnostics(&params, &expect![""]);

    let names: Vec<String> = symbols(&mut client)
        .into_iter()
        .map(|symbol| symbol.name)
        .collect();
    assert_eq!(names, ["total"]);
    client.shutdown();
}

#[test]
fn document_symbols_nest_fields_under_their_struct() {
    let mut client = Client::start(ClientCapabilities::default());
    client.open(
        "struct Point {\n    x: float64,\n    y: float64,\n}\ndef f() -> int64 { return 1 }\n",
    );

    let rendered: Vec<String> = symbols(&mut client)
        .into_iter()
        .map(|symbol| {
            let children: Vec<String> = symbol
                .children
                .unwrap_or_default()
                .into_iter()
                .map(|child| child.name)
                .collect();
            format!(
                "{:?} {} [{}]",
                symbol.kind,
                symbol.name,
                children.join(", ")
            )
        })
        .collect();
    expect![[r"
        Struct Point [x, y]
        Function f []"]]
    .assert_eq(&rendered.join("\n"));
    client.shutdown();
}

#[test]
fn semantic_tokens_are_sent_relative_to_the_one_before() {
    let mut client = Client::start(ClientCapabilities::default());
    client.open("def f(x: int64) {\n    return x\n}\n");

    let result = client.request::<SemanticTokensFullRequest>(SemanticTokensParams {
        text_document: document(),
        work_done_progress_params: WorkDoneProgressParams::default(),
        partial_result_params: PartialResultParams::default(),
    });
    let Some(SemanticTokensResult::Tokens(tokens)) = result else {
        panic!("the server sends the tokens whole");
    };
    let rendered: Vec<String> = tokens
        .data
        .iter()
        .map(|token| {
            format!(
                "+{} +{} len {} type {} mods {:b}",
                token.delta_line,
                token.delta_start,
                token.length,
                token.token_type,
                token.token_modifiers_bitset
            )
        })
        .collect();
    expect![[r"
        +0 +0 len 3 type 0 mods 0
        +0 +4 len 1 type 9 mods 1
        +0 +2 len 1 type 10 mods 1
        +0 +3 len 5 type 12 mods 0
        +1 +4 len 6 type 0 mods 0"]]
    .assert_eq(&rendered.join("\n"));
    client.shutdown();
}

#[test]
fn positions_count_utf16_unless_the_client_takes_utf8() {
    let text = "let s = \"日本\" let t = 1\n";
    let utf8 = ClientCapabilities {
        general: Some(GeneralClientCapabilities {
            position_encodings: Some(vec![PositionEncodingKind::UTF8]),
            ..GeneralClientCapabilities::default()
        }),
        ..ClientCapabilities::default()
    };

    let mut starts = Vec::new();
    for capabilities in [ClientCapabilities::default(), utf8] {
        let mut client = Client::start(capabilities);
        client.open(text);
        let t = symbols(&mut client)
            .into_iter()
            .find(|symbol| symbol.name == "t")
            .expect("`t` is declared");
        starts.push(t.selection_range.start.character);
        client.shutdown();
    }
    assert_eq!(starts, [17, 21]);
}

#[test]
fn an_unknown_request_is_answered_with_an_error() {
    let mut client = Client::start(ClientCapabilities::default());
    let response = client.raw_request("yuzu/unknown", ());
    let error = response.response_result.expect_err("the request fails");
    assert_eq!(error.code, lsp_server::ErrorCode::MethodNotFound as i32);
    client.shutdown();
}

#[test]
fn a_semantic_error_is_published() {
    let client = Client::start(ClientCapabilities::default());
    client.open("def f(x: i64) -> int64 { return 1 }\n");

    let params = client.notification::<PublishDiagnostics>();
    check_diagnostics(&params, &expect!["0:9-0:12 unknown type `i64`"]);
    client.shutdown();
}

#[test]
fn an_error_in_an_imported_file_is_published_for_that_file() {
    let root = std::env::temp_dir().join(format!("yuzu-lsp-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("helpers.yz"),
        "pub def two() -> i64 { return 2 }\n",
    )
    .unwrap();

    let client = Client::start(ClientCapabilities::default());
    client.open_at(
        Url::from_file_path(root.join("main.yz")).unwrap(),
        "import helpers\n",
    );
    let helpers = Url::from_file_path(root.join("helpers.yz")).unwrap();
    let params = client.diagnostics_for(&helpers);
    assert_eq!(params.version, None);
    check_diagnostics(&params, &expect!["0:17-0:20 unknown type `i64`"]);
    client.shutdown();
    std::fs::remove_dir_all(&root).unwrap();
}

const CHECKED: &str = "def double(x: int64) -> int64 {\n    let y = x * 2\n    return y\n}\n";

fn position_params(line: u32, character: u32) -> lsp_types::TextDocumentPositionParams {
    lsp_types::TextDocumentPositionParams {
        text_document: document(),
        position: Position::new(line, character),
    }
}

#[test]
fn a_use_goes_to_its_declaration() {
    let mut client = Client::start(ClientCapabilities::default());
    client.open(CHECKED);
    client.notification::<PublishDiagnostics>();

    let response = client.request::<GotoDefinition>(lsp_types::GotoDefinitionParams {
        text_document_position_params: position_params(2, 11),
        work_done_progress_params: WorkDoneProgressParams::default(),
        partial_result_params: PartialResultParams::default(),
    });
    let Some(lsp_types::GotoDefinitionResponse::Scalar(location)) = response else {
        panic!("the server answers with one location");
    };
    assert_eq!(location.uri, url());
    assert_eq!(
        location.range,
        Range::new(Position::new(1, 8), Position::new(1, 9))
    );
    client.shutdown();
}

#[test]
fn a_hover_shows_the_declaration() {
    let mut client = Client::start(ClientCapabilities::default());
    client.open(CHECKED);
    client.notification::<PublishDiagnostics>();

    let hover = client
        .request::<HoverRequest>(lsp_types::HoverParams {
            text_document_position_params: position_params(2, 11),
            work_done_progress_params: WorkDoneProgressParams::default(),
        })
        .expect("the server hovers a name");
    let lsp_types::HoverContents::Markup(markup) = hover.contents else {
        panic!("the hover is markdown");
    };
    expect![[r"
        ```yuzu
        let y: int64
        ```"]]
    .assert_eq(&markup.value);
    client.shutdown();
}

#[test]
fn a_let_gets_its_type_as_a_hint() {
    let mut client = Client::start(ClientCapabilities::default());
    client.open(CHECKED);
    client.notification::<PublishDiagnostics>();

    let hints = client
        .request::<InlayHintRequest>(lsp_types::InlayHintParams {
            text_document: document(),
            range: Range::new(Position::new(0, 0), Position::new(4, 0)),
            work_done_progress_params: WorkDoneProgressParams::default(),
        })
        .expect("the server gives hints");
    let rendered: Vec<String> = hints
        .iter()
        .map(|hint| {
            let lsp_types::InlayHintLabel::String(label) = &hint.label else {
                panic!("a hint is a plain label");
            };
            format!("{}:{} {label}", hint.position.line, hint.position.character)
        })
        .collect();
    expect!["1:9 : int64"].assert_eq(&rendered.join("\n"));
    client.shutdown();
}
