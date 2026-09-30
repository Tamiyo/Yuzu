use std::thread::{self, JoinHandle};
use std::time::Duration;

use expect_test::{Expect, expect};
use lsp_server::{Connection, Message, Notification, Request, RequestId, Response};
use lsp_types::notification::{
    DidChangeTextDocument, DidOpenTextDocument, Exit, Initialized, PublishDiagnostics,
};
use lsp_types::request::{
    DocumentSymbolRequest, GotoDefinition, HoverRequest, Initialize, InlayHintRequest,
    RegisterCapability, Request as _, SelectionRangeRequest, SemanticTokensFullDeltaRequest,
    SemanticTokensFullRequest, Shutdown,
};
use lsp_types::{
    ClientCapabilities, DidChangeTextDocumentParams, DidOpenTextDocumentParams,
    DocumentSymbolParams, DocumentSymbolResponse, GeneralClientCapabilities, InitializeParams,
    PartialResultParams, Position, PositionEncodingKind, PublishDiagnosticsParams, Range,
    SemanticTokensDeltaParams, SemanticTokensFullDeltaResult, SemanticTokensParams,
    SemanticTokensResult, TextDocumentContentChangeEvent, TextDocumentIdentifier, TextDocumentItem,
    Url, VersionedTextDocumentIdentifier, WorkDoneProgressParams,
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

struct TempDir(std::path::PathBuf);

impl TempDir {
    fn new(files: &[(&str, &str)]) -> Self {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "yuzu-lsp-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        for (name, text) in files {
            std::fs::write(root.join(name), text).unwrap();
        }
        TempDir(root)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
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
        &expect!["1:4-1:5 expected one of `mut`, identifier, found `=`"],
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
    client.open("def f(x: int64) -> int64 {\n    return x\n}\n");

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
        +0 +10 len 5 type 12 mods 0
        +1 +4 len 6 type 0 mods 0"]]
    .assert_eq(&rendered.join("\n"));
    client.shutdown();
}

#[test]
fn a_delta_sends_only_the_tokens_that_changed() {
    let mut client = Client::start(ClientCapabilities::default());
    client.open("def f(x: int64) -> int64 {\n    return x\n}\n");
    let result = client.request::<SemanticTokensFullRequest>(SemanticTokensParams {
        text_document: document(),
        work_done_progress_params: WorkDoneProgressParams::default(),
        partial_result_params: PartialResultParams::default(),
    });
    let Some(SemanticTokensResult::Tokens(tokens)) = result else {
        panic!("the server sends the tokens whole");
    };

    client.notify::<DidChangeTextDocument>(DidChangeTextDocumentParams {
        text_document: VersionedTextDocumentIdentifier::new(url(), 2),
        content_changes: vec![TextDocumentContentChangeEvent {
            range: Some(Range::new(Position::new(3, 0), Position::new(3, 0))),
            range_length: None,
            text: "let y = 1\n".to_owned(),
        }],
    });
    let result = client.request::<SemanticTokensFullDeltaRequest>(SemanticTokensDeltaParams {
        text_document: document(),
        previous_result_id: tokens.result_id.expect("the tokens have a result id"),
        work_done_progress_params: WorkDoneProgressParams::default(),
        partial_result_params: PartialResultParams::default(),
    });
    let Some(SemanticTokensFullDeltaResult::TokensDelta(delta)) = result else {
        panic!("the server sends a delta");
    };
    let rendered: Vec<String> = delta
        .edits
        .iter()
        .map(|edit| {
            format!(
                "at {} delete {} insert {}",
                edit.start,
                edit.delete_count,
                edit.data.as_ref().map_or(0, Vec::len)
            )
        })
        .collect();
    expect!["at 30 delete 0 insert 3"].assert_eq(&rendered.join("\n"));
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
    let dir = TempDir::new(&[("helpers.yz", "pub def two() -> i64 { return 2 }\n")]);
    let root = &dir.0;

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
}

#[test]
fn a_change_checks_again_the_files_that_import_it() {
    const HELPERS: &str = "pub def two() -> int64 { return 2 }\n";
    let dir = TempDir::new(&[("helpers.yz", HELPERS)]);
    let main = Url::from_file_path(dir.0.join("main.yz")).unwrap();
    let helpers = Url::from_file_path(dir.0.join("helpers.yz")).unwrap();

    let client = Client::start(ClientCapabilities::default());
    client.open_at(main.clone(), "from helpers import two\nlet x = two()\n");
    client.open_at(helpers.clone(), HELPERS);
    client.notify::<DidChangeTextDocument>(DidChangeTextDocumentParams {
        text_document: VersionedTextDocumentIdentifier::new(helpers, 2),
        content_changes: vec![TextDocumentContentChangeEvent {
            range: Some(Range::new(Position::new(0, 0), Position::new(0, 4))),
            range_length: None,
            text: String::new(),
        }],
    });

    let params = loop {
        let params = client.diagnostics_for(&main);
        if !params.diagnostics.is_empty() {
            break params;
        }
    };
    check_diagnostics(
        &params,
        &expect![[r"
        0:20-0:23 `two` is not public; `helpers` keeps it to itself
        1:8-1:13 unresolved identifier `two`"]],
    );
    client.shutdown();
}

#[test]
fn a_position_past_the_end_of_its_line_is_its_end() {
    let mut client = Client::start(ClientCapabilities::default());
    client.open("let a = 1\nlet b = 2");
    let ranges = client
        .request::<SelectionRangeRequest>(lsp_types::SelectionRangeParams {
            text_document: document(),
            positions: vec![Position::new(0, 99), Position::new(1, 99)],
            work_done_progress_params: WorkDoneProgressParams::default(),
            partial_result_params: PartialResultParams::default(),
        })
        .expect("the server answers for each position");
    let ends: Vec<Position> = ranges.iter().map(|range| range.range.end).collect();
    assert_eq!(ends, [Position::new(0, 9), Position::new(1, 9)]);
    client.shutdown();
}

#[test]
fn the_server_asks_to_watch_yuzu_files() {
    let capabilities = ClientCapabilities {
        workspace: Some(lsp_types::WorkspaceClientCapabilities {
            did_change_watched_files: Some(lsp_types::DidChangeWatchedFilesClientCapabilities {
                dynamic_registration: Some(true),
                relative_pattern_support: None,
            }),
            ..lsp_types::WorkspaceClientCapabilities::default()
        }),
        ..ClientCapabilities::default()
    };
    let client = Client::start(capabilities);
    let registration = loop {
        if let Message::Request(request) = client.receive()
            && request.method == RegisterCapability::METHOD
        {
            break request;
        }
    };
    let params: lsp_types::RegistrationParams =
        serde_json::from_value(registration.params).unwrap();
    expect![[r#"[{"id":"yuzu-watched-files","method":"workspace/didChangeWatchedFiles","registerOptions":{"watchers":[{"globPattern":"**/*.yz"}]}}]"#]]
        .assert_eq(&serde_json::to_string(&params.registrations).unwrap());
    client.shutdown();
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

#[test]
fn a_rename_edits_the_declaration_and_its_uses() {
    let mut client = Client::start(ClientCapabilities::default());
    client.open(CHECKED);
    client.notification::<PublishDiagnostics>();

    let edit = client
        .request::<lsp_types::request::Rename>(lsp_types::RenameParams {
            text_document_position: position_params(2, 11),
            new_name: "z".to_owned(),
            work_done_progress_params: WorkDoneProgressParams::default(),
        })
        .expect("the server renames a local");
    let changes = edit.changes.expect("the edit names its files");
    let mut ranges: Vec<Range> = changes[&url()].iter().map(|edit| edit.range).collect();
    ranges.sort_by_key(|range| (range.start.line, range.start.character));
    assert_eq!(
        ranges,
        [
            Range::new(Position::new(1, 8), Position::new(1, 9)),
            Range::new(Position::new(2, 11), Position::new(2, 12)),
        ]
    );
    client.shutdown();
}

#[test]
fn a_rename_the_server_refuses_says_why() {
    let mut client = Client::start(ClientCapabilities::default());
    client.open(CHECKED);
    client.notification::<PublishDiagnostics>();

    let response = client.raw_request(
        "textDocument/rename",
        lsp_types::RenameParams {
            text_document_position: position_params(2, 11),
            new_name: "let".to_owned(),
            work_done_progress_params: WorkDoneProgressParams::default(),
        },
    );
    let error = response
        .response_result
        .expect_err("the server refuses the rename");
    assert_eq!(error.message, "`let` is not a name");
    client.shutdown();
}

#[test]
fn a_body_completes_its_locals() {
    let mut client = Client::start(ClientCapabilities::default());
    client.open(CHECKED);
    client.notification::<PublishDiagnostics>();

    let response = client.request::<lsp_types::request::Completion>(lsp_types::CompletionParams {
        text_document_position: position_params(2, 11),
        work_done_progress_params: WorkDoneProgressParams::default(),
        partial_result_params: PartialResultParams::default(),
        context: None,
    });
    let Some(lsp_types::CompletionResponse::Array(items)) = response else {
        panic!("the server answers with a list");
    };
    let labels: Vec<&str> = items.iter().map(|item| item.label.as_str()).collect();
    assert!(labels.contains(&"y") && labels.contains(&"x"), "{labels:?}");
    client.shutdown();
}

#[test]
fn signature_help_reads_a_call_the_check_has_not_seen() {
    let mut client = Client::start(ClientCapabilities::default());
    let text =
        "def f(x: int64) -> int64 { return x }\ntable t = { a: int64 }\nfrom t |> select f as v\n";
    client.open(text);
    client.notification::<PublishDiagnostics>();

    // `(` is typed after `f`; the check read `f` alone.
    client.notify::<DidChangeTextDocument>(DidChangeTextDocumentParams {
        text_document: VersionedTextDocumentIdentifier::new(url(), 2),
        content_changes: vec![TextDocumentContentChangeEvent {
            range: Some(Range::new(Position::new(2, 18), Position::new(2, 18))),
            range_length: None,
            text: "(".to_owned(),
        }],
    });
    let help = client
        .request::<lsp_types::request::SignatureHelpRequest>(lsp_types::SignatureHelpParams {
            context: None,
            text_document_position_params: position_params(2, 19),
            work_done_progress_params: WorkDoneProgressParams::default(),
        })
        .expect("the server finds the call");
    assert_eq!(help.signatures[0].label, "def f(x: int64) -> int64");
    client.shutdown();
}
