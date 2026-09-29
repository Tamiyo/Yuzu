//! What the server offers, and what it reads from what the client offers.

use lsp_types::notification::{DidChangeWatchedFiles, Notification as _};
use lsp_types::{
    ClientCapabilities, DidChangeWatchedFilesRegistrationOptions, FileSystemWatcher,
    FoldingRangeProviderCapability, GlobPattern, HoverProviderCapability, OneOf,
    PositionEncodingKind, Registration, RenameOptions, SelectionRangeProviderCapability,
    SemanticTokensFullOptions, SemanticTokensLegend, SemanticTokensOptions,
    SemanticTokensServerCapabilities, ServerCapabilities, SignatureHelpOptions,
    TextDocumentSyncCapability, TextDocumentSyncKind, WorkDoneProgressOptions,
};

use crate::line_index::PositionEncoding;
use crate::semantic_tokens;

pub(crate) fn server_capabilities(encoding: PositionEncoding) -> ServerCapabilities {
    ServerCapabilities {
        position_encoding: Some(position_encoding_kind(encoding)),
        text_document_sync: Some(TextDocumentSyncCapability::Kind(
            TextDocumentSyncKind::INCREMENTAL,
        )),
        document_symbol_provider: Some(OneOf::Left(true)),
        definition_provider: Some(OneOf::Left(true)),
        references_provider: Some(OneOf::Left(true)),
        document_highlight_provider: Some(OneOf::Left(true)),
        hover_provider: Some(HoverProviderCapability::Simple(true)),
        rename_provider: Some(OneOf::Right(RenameOptions {
            prepare_provider: Some(true),
            work_done_progress_options: WorkDoneProgressOptions::default(),
        })),
        signature_help_provider: Some(SignatureHelpOptions {
            trigger_characters: Some(vec!["(".to_owned(), ",".to_owned()]),
            retrigger_characters: None,
            work_done_progress_options: WorkDoneProgressOptions::default(),
        }),
        inlay_hint_provider: Some(OneOf::Left(true)),
        folding_range_provider: Some(FoldingRangeProviderCapability::Simple(true)),
        selection_range_provider: Some(SelectionRangeProviderCapability::Simple(true)),
        semantic_tokens_provider: Some(SemanticTokensServerCapabilities::SemanticTokensOptions(
            SemanticTokensOptions {
                legend: SemanticTokensLegend {
                    token_types: semantic_tokens::TYPES.to_vec(),
                    token_modifiers: semantic_tokens::MODIFIERS.to_vec(),
                },
                full: Some(SemanticTokensFullOptions::Bool(true)),
                range: None,
                ..SemanticTokensOptions::default()
            },
        )),
        ..ServerCapabilities::default()
    }
}

/// UTF-8 when the client can take it, since offsets are bytes already.
/// Otherwise UTF-16, which every client must support.
pub(crate) fn position_encoding(client: &ClientCapabilities) -> PositionEncoding {
    let offered = client
        .general
        .as_ref()
        .and_then(|general| general.position_encodings.as_deref())
        .unwrap_or_default();

    if offered.contains(&PositionEncodingKind::UTF8) {
        PositionEncoding::Utf8
    } else {
        PositionEncoding::Utf16
    }
}

/// How much of a line the client can fold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Folding {
    /// Whole lines only, as VS Code folds.
    Lines,
    Characters,
}

pub(crate) fn folding(client: &ClientCapabilities) -> Folding {
    let lines_only = client
        .text_document
        .as_ref()
        .and_then(|document| document.folding_range.as_ref())
        .and_then(|folding| folding.line_folding_only);

    match lines_only {
        Some(true) => Folding::Lines,
        Some(false) | None => Folding::Characters,
    }
}

/// Whether the client asks for something again when the server says it
/// changed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Refresh {
    Supported,
    Unsupported,
}

pub(crate) fn semantic_tokens_refresh(client: &ClientCapabilities) -> Refresh {
    refresh(
        client
            .workspace
            .as_ref()
            .and_then(|workspace| workspace.semantic_tokens.as_ref())
            .and_then(|semantic_tokens| semantic_tokens.refresh_support),
    )
}

pub(crate) fn inlay_hint_refresh(client: &ClientCapabilities) -> Refresh {
    refresh(
        client
            .workspace
            .as_ref()
            .and_then(|workspace| workspace.inlay_hint.as_ref())
            .and_then(|inlay_hint| inlay_hint.refresh_support),
    )
}

fn refresh(supported: Option<bool>) -> Refresh {
    match supported {
        Some(true) => Refresh::Supported,
        Some(false) | None => Refresh::Unsupported,
    }
}

/// Whether the client lets the server ask it to watch files.
pub(crate) fn watches_files(client: &ClientCapabilities) -> bool {
    client
        .workspace
        .as_ref()
        .and_then(|workspace| workspace.did_change_watched_files.as_ref())
        .and_then(|watched| watched.dynamic_registration)
        .unwrap_or(false)
}

/// The registration that asks the client to report changes to Yuzu files.
pub(crate) fn watched_files_registration() -> Registration {
    let options = DidChangeWatchedFilesRegistrationOptions {
        watchers: vec![FileSystemWatcher {
            glob_pattern: GlobPattern::String("**/*.yz".to_owned()),
            kind: None,
        }],
    };
    Registration {
        id: "yuzu-watched-files".to_owned(),
        method: DidChangeWatchedFiles::METHOD.to_owned(),
        register_options: Some(
            serde_json::to_value(options).expect("registration options serialize"),
        ),
    }
}

fn position_encoding_kind(encoding: PositionEncoding) -> PositionEncodingKind {
    match encoding {
        PositionEncoding::Utf8 => PositionEncodingKind::UTF8,
        PositionEncoding::Utf16 => PositionEncodingKind::UTF16,
    }
}
