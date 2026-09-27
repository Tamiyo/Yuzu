//! What the server offers, and what it reads from what the client offers.

use lsp_types::{
    ClientCapabilities, FoldingRangeProviderCapability, HoverProviderCapability, OneOf,
    PositionEncodingKind, SelectionRangeProviderCapability, SemanticTokensFullOptions,
    SemanticTokensLegend, SemanticTokensOptions, SemanticTokensServerCapabilities,
    ServerCapabilities, TextDocumentSyncCapability, TextDocumentSyncKind,
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

pub(crate) fn inlay_hint_refresh(client: &ClientCapabilities) -> Refresh {
    let supported = client
        .workspace
        .as_ref()
        .and_then(|workspace| workspace.inlay_hint.as_ref())
        .and_then(|inlay_hint| inlay_hint.refresh_support);

    match supported {
        Some(true) => Refresh::Supported,
        Some(false) | None => Refresh::Unsupported,
    }
}

fn position_encoding_kind(encoding: PositionEncoding) -> PositionEncodingKind {
    match encoding {
        PositionEncoding::Utf8 => PositionEncodingKind::UTF8,
        PositionEncoding::Utf16 => PositionEncodingKind::UTF16,
    }
}
