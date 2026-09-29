//! One function for each request. A request about a document the client has
//! not opened gets `None`, which the protocol sends as `null`.

use std::collections::HashMap;

use lsp_types::{
    CompletionItemKind, CompletionParams, CompletionResponse, DocumentHighlight,
    DocumentHighlightParams, DocumentSymbolParams, DocumentSymbolResponse, FoldingRange,
    FoldingRangeParams, GotoDefinitionParams, GotoDefinitionResponse, Hover, HoverContents,
    HoverParams, InlayHint, InlayHintKind, InlayHintLabel, InlayHintParams, Location,
    MarkupContent, MarkupKind, ParameterInformation, ParameterLabel, PrepareRenameResponse,
    ReferenceParams, RenameParams, SelectionRange, SelectionRangeParams, SemanticTokensParams,
    SemanticTokensResult, SignatureHelp, SignatureHelpParams, SignatureInformation,
    TextDocumentPositionParams, TextEdit, Url, WorkspaceEdit,
};
use rustc_hash::FxHashSet;
use text_size::{TextRange, TextSize};
use yuzu_ide::{CallSite, Checked, CompletionKind, FilePosition, HlRange};

use crate::documents::Document;
use crate::global_state::GlobalState;
use crate::text_shift::TextShift;
use crate::to_proto::Locations;
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
            to_proto::folding_range(&document.text, &document.line_index, state.folding, fold)
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

/// The syntax's highlights, with each resolved use highlighted as its
/// declaration. The uses come from the last check, carried over to the
/// text the document has now.
pub(crate) fn semantic_tokens_full(
    state: &GlobalState,
    params: &SemanticTokensParams,
) -> Option<SemanticTokensResult> {
    let url = &params.text_document.uri;
    let (file_id, document) = state.document(url)?;
    let mut highlights = state.analysis().highlight(file_id)?;
    if let Some((file_id, _, checked, shift)) = state.last_check(url) {
        let uses: Vec<HlRange> = checked
            .highlight_uses(file_id)
            .into_iter()
            .filter_map(|used| {
                Some(HlRange {
                    range: shift.map(used.range)?,
                    highlight: used.highlight,
                })
            })
            .collect();
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
    let (_, checked, position) = checked_position(state, &params.text_document_position_params)?;
    let target = checked.goto_definition(position)?;
    let location = Locations::new(checked, state.encoding).location(&target)?;
    Some(GotoDefinitionResponse::Scalar(location))
}

pub(crate) fn references(state: &GlobalState, params: &ReferenceParams) -> Option<Vec<Location>> {
    let (_, checked, position) = checked_position(state, &params.text_document_position)?;
    let references = checked.references(position);
    let declaration = references
        .declaration
        .filter(|_| params.context.include_declaration);
    let mut locations = Locations::new(checked, state.encoding);
    let found = declaration
        .iter()
        .chain(&references.uses)
        .filter_map(|found| locations.location(found))
        .collect();
    Some(found)
}

pub(crate) fn document_highlight(
    state: &GlobalState,
    params: &DocumentHighlightParams,
) -> Option<Vec<DocumentHighlight>> {
    let (document, checked, position) =
        checked_position(state, &params.text_document_position_params)?;
    let highlights = checked
        .highlight_related(position)
        .into_iter()
        .map(|range| DocumentHighlight {
            range: to_proto::range(&document.line_index, range),
            kind: None,
        })
        .collect();
    Some(highlights)
}

pub(crate) fn hover(state: &GlobalState, params: &HoverParams) -> Option<Hover> {
    let (document, checked, position) =
        checked_position(state, &params.text_document_position_params)?;
    let hover = checked.hover(position)?;
    Some(Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: hover.markup,
        }),
        range: Some(to_proto::range(&document.line_index, hover.range)),
    })
}

/// The names that fit at a position. Where the position is, is read from
/// the document's text now; the names come from the last check, which may
/// be older, by where the stage around the position starts.
pub(crate) fn completion(
    state: &GlobalState,
    params: &CompletionParams,
) -> Option<CompletionResponse> {
    let position = &params.text_document_position;
    let (file_id, document, checked, _) = state.last_check(&position.text_document.uri)?;
    let offset = from_proto::offset(&document.line_index, position.position)?;
    let mut site = state
        .analysis()
        .completion_site(FilePosition { file_id, offset })?;
    let back = TextShift::between(&document.text, checked.file_text(file_id)?);
    site.stage = site
        .stage
        .and_then(|stage| back.map(TextRange::empty(stage)))
        .map(TextRange::start);
    let items = checked
        .completions(file_id, &site)
        .into_iter()
        .map(|item| lsp_types::CompletionItem {
            label: item.label,
            kind: Some(completion_kind(item.kind)),
            ..lsp_types::CompletionItem::default()
        })
        .collect();
    Some(CompletionResponse::Array(items))
}

fn completion_kind(kind: CompletionKind) -> CompletionItemKind {
    match kind {
        CompletionKind::Keyword => CompletionItemKind::KEYWORD,
        CompletionKind::Column => CompletionItemKind::FIELD,
        CompletionKind::Local | CompletionKind::Parameter | CompletionKind::Binding => {
            CompletionItemKind::VARIABLE
        }
        CompletionKind::Function => CompletionItemKind::FUNCTION,
        CompletionKind::Module => CompletionItemKind::MODULE,
    }
}

/// The name a rename at a position would change.
pub(crate) fn prepare_rename(
    state: &GlobalState,
    params: &TextDocumentPositionParams,
) -> Result<Option<PrepareRenameResponse>, String> {
    let (document, checked, position) = checked_position(state, params).ok_or(NOT_CHECKED)?;
    let range = checked
        .prepare_rename(position)
        .map_err(|error| error.to_string())?;
    Ok(Some(PrepareRenameResponse::Range(to_proto::range(
        &document.line_index,
        range,
    ))))
}

/// Writes the new name over the declaration and each use that spells it.
pub(crate) fn rename(
    state: &GlobalState,
    params: &RenameParams,
) -> Result<Option<WorkspaceEdit>, String> {
    let (_, checked, position) =
        checked_position(state, &params.text_document_position).ok_or(NOT_CHECKED)?;
    let edits = checked
        .rename(position, &params.new_name)
        .map_err(|error| error.to_string())?;
    let mut locations = Locations::new(checked, state.encoding);
    let mut changes: HashMap<Url, Vec<TextEdit>> = HashMap::new();
    for edit in &edits {
        let location = locations
            .location(edit)
            .ok_or_else(|| format!("`{}` cannot be read", edit.path.display()))?;
        changes.entry(location.uri).or_default().push(TextEdit {
            range: location.range,
            new_text: params.new_name.clone(),
        });
    }
    Ok(Some(WorkspaceEdit {
        changes: Some(changes),
        ..WorkspaceEdit::default()
    }))
}

/// Why a request that changes files is refused before the check catches up.
const NOT_CHECKED: &str = "the file is not checked since its last change; try again";

/// The overloads of the function a call names. The call is found in the
/// text the document has now, and its function's name is carried back to
/// the text the last check read, which resolved it.
pub(crate) fn signature_help(
    state: &GlobalState,
    params: &SignatureHelpParams,
) -> Option<SignatureHelp> {
    let position = &params.text_document_position_params;
    let (file_id, document, checked, _) = state.last_check(&position.text_document.uri)?;
    let offset = from_proto::offset(&document.line_index, position.position)?;
    let site = state.analysis().call_at(FilePosition { file_id, offset })?;
    let back = TextShift::between(&document.text, checked.file_text(file_id)?);
    let site = CallSite {
        callee: back.map(site.callee)?,
        argument: site.argument,
    };
    let help = checked.signature_help(file_id, site)?;
    let signatures = help
        .signatures
        .into_iter()
        .map(|signature| {
            let parameters = signature
                .parameters
                .iter()
                .map(|&range| ParameterInformation {
                    label: ParameterLabel::Simple(signature.label[range].to_owned()),
                    documentation: None,
                })
                .collect();
            SignatureInformation {
                label: signature.label,
                documentation: None,
                parameters: Some(parameters),
                active_parameter: None,
            }
        })
        .collect();
    Some(SignatureHelp {
        signatures,
        active_signature: u32::try_from(help.active_signature).ok(),
        active_parameter: u32::try_from(help.active_parameter).ok(),
    })
}

/// The hints of the last check in the requested range, carried over to the
/// text the document has now.
pub(crate) fn inlay_hint(state: &GlobalState, params: &InlayHintParams) -> Option<Vec<InlayHint>> {
    let (file_id, document, checked, shift) = state.last_check(&params.text_document.uri)?;
    let requested = from_proto::text_range(&document.line_index, params.range)?;
    let checked_text = TextRange::up_to(TextSize::of(checked.file_text(file_id)?));
    let hints = checked
        .inlay_hints(file_id, checked_text)
        .into_iter()
        .filter_map(|hint| {
            let offset = shift.map(TextRange::empty(hint.offset))?.start();
            requested
                .contains_inclusive(offset)
                .then(|| inlay_hint_at(document, offset, hint.label))
        })
        .collect();
    Some(hints)
}

fn inlay_hint_at(document: &Document, offset: TextSize, label: String) -> InlayHint {
    InlayHint {
        position: to_proto::position(&document.line_index, offset),
        label: InlayHintLabel::String(label),
        kind: Some(InlayHintKind::TYPE),
        text_edits: None,
        tooltip: None,
        padding_left: None,
        padding_right: None,
        data: None,
    }
}

/// The document a position is in, its check while that check read the text
/// the document has now, and the position as an offset in it.
fn checked_position<'s>(
    state: &'s GlobalState,
    params: &TextDocumentPositionParams,
) -> Option<(&'s Document, &'s Checked, FilePosition)> {
    let (file_id, document, checked) = state.fresh_check(&params.text_document.uri)?;
    let offset = from_proto::offset(&document.line_index, params.position)?;
    Some((document, checked, FilePosition { file_id, offset }))
}
