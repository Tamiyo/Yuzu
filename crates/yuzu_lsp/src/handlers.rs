//! One function for each request. A request about a document the client has
//! not opened gets `None`, which the protocol sends as `null`.

use lsp_types::{
    DocumentHighlight, DocumentHighlightParams, DocumentSymbolParams, DocumentSymbolResponse,
    FoldingRange, FoldingRangeParams, GotoDefinitionParams, GotoDefinitionResponse, Hover,
    HoverContents, HoverParams, InlayHint, InlayHintKind, InlayHintLabel, InlayHintParams,
    Location, MarkupContent, MarkupKind, ReferenceParams, SelectionRange, SelectionRangeParams,
    SemanticTokensParams, SemanticTokensResult,
};
use rustc_hash::FxHashSet;
use text_size::TextRange;
use yuzu_ide::FilePosition;

use crate::global_state::GlobalState;
use crate::{from_proto, to_proto};

pub(crate) fn document_symbol(
    state: &GlobalState,
    params: &DocumentSymbolParams,
) -> Option<DocumentSymbolResponse> {
    let (file_id, document) = state.document(&params.text_document.uri)?;
    let nodes = state.analysis().file_structure(file_id)?;
    let symbols = to_proto::document_symbols(&document.line_index, nodes);
    Some(DocumentSymbolResponse::Nested(symbols))
}

pub(crate) fn folding_range(
    state: &GlobalState,
    params: &FoldingRangeParams,
) -> Option<Vec<FoldingRange>> {
    let (file_id, document) = state.document(&params.text_document.uri)?;
    let folds = state.analysis().folding_ranges(file_id)?;
    let ranges = folds
        .into_iter()
        .map(|fold| {
            to_proto::folding_range(&document.text, &document.line_index, state.folding(), fold)
        })
        .collect();
    Some(ranges)
}

pub(crate) fn selection_range(
    state: &GlobalState,
    params: &SelectionRangeParams,
) -> Option<Vec<SelectionRange>> {
    let (file_id, document) = state.document(&params.text_document.uri)?;
    let analysis = state.analysis();
    params
        .positions
        .iter()
        .map(|&position| {
            let offset = from_proto::offset(&document.line_index, position)?;
            let ranges = analysis.selection_ranges(FilePosition { file_id, offset })?;
            Some(to_proto::selection_range(&document.line_index, &ranges))
        })
        .collect()
}

pub(crate) fn semantic_tokens_full(
    state: &GlobalState,
    params: &SemanticTokensParams,
) -> Option<SemanticTokensResult> {
    let url = &params.text_document.uri;
    let (file_id, document) = state.document(url)?;
    let mut highlights = state.analysis().highlight(file_id)?;
    if let Some((_, path, checked)) = state.checked_document(url) {
        let uses = checked.highlight_uses(path);
        let resolved: FxHashSet<TextRange> = uses.iter().map(|used| used.range).collect();
        highlights.retain(|syntax| !resolved.contains(&syntax.range));
        highlights.extend(uses);
        highlights.sort_by_key(|highlight| highlight.range.start());
    }
    let tokens = to_proto::semantic_tokens(&document.text, &document.line_index, &highlights);
    Some(SemanticTokensResult::Tokens(tokens))
}

pub(crate) fn goto_definition(
    state: &GlobalState,
    params: &GotoDefinitionParams,
) -> Option<GotoDefinitionResponse> {
    let position = &params.text_document_position_params;
    let (document, path, checked) = state.checked_document(&position.text_document.uri)?;
    let offset = from_proto::offset(&document.line_index, position.position)?;
    let target = checked.goto_definition(path, offset)?;
    let location = to_proto::location(checked, &target, state.encoding())?;
    Some(GotoDefinitionResponse::Scalar(location))
}

pub(crate) fn references(state: &GlobalState, params: &ReferenceParams) -> Option<Vec<Location>> {
    let position = &params.text_document_position;
    let (document, path, checked) = state.checked_document(&position.text_document.uri)?;
    let offset = from_proto::offset(&document.line_index, position.position)?;
    let skip = usize::from(!params.context.include_declaration);
    let locations = checked
        .references(path, offset)
        .iter()
        .skip(skip)
        .filter_map(|found| to_proto::location(checked, found, state.encoding()))
        .collect();
    Some(locations)
}

pub(crate) fn document_highlight(
    state: &GlobalState,
    params: &DocumentHighlightParams,
) -> Option<Vec<DocumentHighlight>> {
    let position = &params.text_document_position_params;
    let (document, path, checked) = state.checked_document(&position.text_document.uri)?;
    let offset = from_proto::offset(&document.line_index, position.position)?;
    let highlights = checked
        .highlight(path, offset)
        .into_iter()
        .map(|range| DocumentHighlight {
            range: to_proto::range(&document.line_index, range),
            kind: None,
        })
        .collect();
    Some(highlights)
}

pub(crate) fn hover(state: &GlobalState, params: &HoverParams) -> Option<Hover> {
    let position = &params.text_document_position_params;
    let (document, path, checked) = state.checked_document(&position.text_document.uri)?;
    let offset = from_proto::offset(&document.line_index, position.position)?;
    let hover = checked.hover(path, offset)?;
    Some(Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: hover.markup,
        }),
        range: Some(to_proto::range(&document.line_index, hover.range)),
    })
}

pub(crate) fn inlay_hint(state: &GlobalState, params: &InlayHintParams) -> Option<Vec<InlayHint>> {
    let (document, path, checked) = state.checked_document(&params.text_document.uri)?;
    let hints = checked
        .inlay_hints(path)
        .into_iter()
        .map(|hint| InlayHint {
            position: to_proto::position(&document.line_index, hint.offset),
            label: InlayHintLabel::String(hint.label),
            kind: Some(InlayHintKind::TYPE),
            text_edits: None,
            tooltip: None,
            padding_left: None,
            padding_right: None,
            data: None,
        })
        .collect();
    Some(hints)
}
