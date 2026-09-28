//! Times the server as a client sees it, through the protocol: a keystroke
//! followed by a semantic-token request, a hover, and the time from an edit
//! to its diagnostics. Run it in release mode:
//! `cargo run --release -p yuzu_lsp --example bench`.

use std::thread;
use std::time::{Duration, Instant};

use lsp_server::{Connection, Message, Notification, Request, RequestId};
use lsp_types::notification::{
    DidChangeTextDocument, DidOpenTextDocument, Exit, Initialized, Notification as _,
    PublishDiagnostics,
};
use lsp_types::request::{
    HoverRequest, Initialize, Request as _, SemanticTokensFullRequest, Shutdown,
};
use lsp_types::{
    ClientCapabilities, DidChangeTextDocumentParams, DidOpenTextDocumentParams, HoverParams,
    InitializeParams, Position, Range, SemanticTokensParams, TextDocumentContentChangeEvent,
    TextDocumentIdentifier, TextDocumentItem, TextDocumentPositionParams, Url,
    VersionedTextDocumentIdentifier,
};
use serde::Serialize;

struct Client {
    connection: Connection,
    next_id: i32,
}

impl Client {
    fn request(&mut self, method: &str, params: impl Serialize) {
        self.next_id += 1;
        let id = RequestId::from(self.next_id);
        let request = Request::new(id.clone(), method.to_owned(), params);
        self.connection.sender.send(request.into()).unwrap();
        loop {
            match self.connection.receiver.recv().unwrap() {
                Message::Response(response) if response.id == id => return,
                _ => {}
            }
        }
    }

    fn notify(&self, method: &str, params: impl Serialize) {
        let notification = Notification::new(method.to_owned(), params);
        self.connection.sender.send(notification.into()).unwrap();
    }

    fn wait_for_diagnostics(&self, version: i32) {
        loop {
            if let Message::Notification(notification) = self.connection.receiver.recv().unwrap()
                && notification.method == PublishDiagnostics::METHOD
                && notification.params["version"] == version
            {
                return;
            }
        }
    }
}

fn program(functions: usize) -> String {
    let mut text = String::from("table t = { a: int64, b: str }\nlet cap = 10\n");
    for i in 0..functions {
        text.push_str(&format!(
            "def f{i}(x: int64) -> int64 {{\n    let y = x * {i} + cap\n    return y\n}}\n\
             from t |> where a > {i} |> select f{i}(a) as v{i}\n\n"
        ));
    }
    text
}

fn summary(mut samples: Vec<Duration>) -> String {
    samples.sort();
    let median = samples[samples.len() / 2];
    let p95 = samples[samples.len() * 95 / 100];
    format!("median {median:>10.1?}   p95 {p95:>10.1?}")
}

fn main() {
    let dir = std::env::temp_dir().join(format!("yuzu-lsp-bench-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let url = Url::from_file_path(dir.join("main.yz")).unwrap();
    let document = || TextDocumentIdentifier::new(url.clone());

    for functions in [10, 100, 500] {
        let text = program(functions);
        let (connection, server) = Connection::memory();
        let server = thread::spawn(move || yuzu_lsp::run(&server));
        let mut client = Client {
            connection,
            next_id: 0,
        };
        client.request(
            Initialize::METHOD,
            InitializeParams {
                capabilities: ClientCapabilities::default(),
                ..InitializeParams::default()
            },
        );
        client.notify(Initialized::METHOD, lsp_types::InitializedParams {});
        client.notify(
            DidOpenTextDocument::METHOD,
            DidOpenTextDocumentParams {
                text_document: TextDocumentItem::new(url.clone(), "yuzu".into(), 1, text.clone()),
            },
        );
        client.wait_for_diagnostics(1);

        let mut version = 1;
        let mut keystrokes = Vec::new();
        for i in 0..50 {
            version += 1;
            let start = Instant::now();
            client.notify(
                DidChangeTextDocument::METHOD,
                DidChangeTextDocumentParams {
                    text_document: VersionedTextDocumentIdentifier::new(url.clone(), version),
                    content_changes: vec![TextDocumentContentChangeEvent {
                        range: Some(Range::new(Position::new(1, 10), Position::new(1, 12))),
                        range_length: None,
                        text: format!("{}", 10 + i % 2),
                    }],
                },
            );
            client.request(
                SemanticTokensFullRequest::METHOD,
                SemanticTokensParams {
                    text_document: document(),
                    work_done_progress_params: Default::default(),
                    partial_result_params: Default::default(),
                },
            );
            keystrokes.push(start.elapsed());
        }
        client.wait_for_diagnostics(version);

        let mut hovers = Vec::new();
        for _ in 0..50 {
            let start = Instant::now();
            client.request(
                HoverRequest::METHOD,
                HoverParams {
                    text_document_position_params: TextDocumentPositionParams {
                        text_document: document(),
                        position: Position::new(4, 11),
                    },
                    work_done_progress_params: Default::default(),
                },
            );
            hovers.push(start.elapsed());
        }

        let mut checks = Vec::new();
        for i in 0..10 {
            version += 1;
            let start = Instant::now();
            client.notify(
                DidChangeTextDocument::METHOD,
                DidChangeTextDocumentParams {
                    text_document: VersionedTextDocumentIdentifier::new(url.clone(), version),
                    content_changes: vec![TextDocumentContentChangeEvent {
                        range: Some(Range::new(Position::new(1, 10), Position::new(1, 12))),
                        range_length: None,
                        text: format!("{}", 10 + i % 2),
                    }],
                },
            );
            client.wait_for_diagnostics(version);
            checks.push(start.elapsed());
        }

        println!("{functions} functions, {} lines", text.lines().count());
        println!("  keystroke + semantic tokens    {}", summary(keystrokes));
        println!("  hover                          {}", summary(hovers));
        println!("  edit to diagnostics (200 ms debounce included)");
        println!("                                 {}", summary(checks));

        client.request(Shutdown::METHOD, ());
        client.notify(Exit::METHOD, ());
        server.join().unwrap().unwrap();
    }
    let _ = std::fs::remove_dir_all(&dir);
}
