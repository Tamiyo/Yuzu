//! One function for each request. A request about a document the client has
//! not opened gets `None`, which the protocol sends as `null`.

use std::collections::HashMap;

use lsp_types::{
    CompletionItemKind, CompletionParams, CompletionResponse, DocumentHighlight,
    DocumentHighlightParams, DocumentSymbolParams, DocumentSymbolResponse, FoldingRange,
    FoldingRangeParams, GotoDefinitionParams, GotoDefinitionResponse, Hover, HoverContents,
    HoverParams, InlayHint, InlayHintKind, InlayHintLabel, InlayHintParams, Location,
    MarkupContent, MarkupKind, ParameterInformation, ParameterLabel, PrepareRenameResponse,
    ReferenceParams, RenameParams, SelectionRange, SelectionRangeParams, SemanticTokens,
    SemanticTokensDelta, SemanticTokensDeltaParams, SemanticTokensFullDeltaResult,
    SemanticTokensParams, SemanticTokensResult, SignatureHelp, SignatureHelpParams,
    SignatureInformation, TextDocumentPositionParams, TextEdit, Url, WorkspaceEdit,
};
use rustc_hash::FxHashSet;
use text_size::{TextRange, TextSize};
use yuzu_ide::{CallSite, Checked, CompletionKind, FileId, FilePosition, HlRange};

use crate::documents::Document;
use crate::global_state::GlobalState;
use crate::text_shift::TextShift;
use crate::to_proto::CheckedFiles;
use crate::{from_proto, semantic_tokens, to_proto};

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

/// A document's semantic tokens, kept for a later delta request.
pub(crate) fn semantic_tokens_full(
    state: &mut GlobalState,
    params: &SemanticTokensParams,
) -> Option<SemanticTokensResult> {
    let url = &params.text_document.uri;
    let tokens = highlight_tokens(state, url)?;
    let (file_id, _) = state.document(url)?;
    let tokens = state.remember_tokens(file_id, tokens);
    Some(SemanticTokensResult::Tokens(tokens.clone()))
}

/// The changes since the tokens the client names, when the server still has
/// them. Otherwise, all the tokens.
pub(crate) fn semantic_tokens_full_delta(
    state: &mut GlobalState,
    params: &SemanticTokensDeltaParams,
) -> Option<SemanticTokensFullDeltaResult> {
    let url = &params.text_document.uri;
    let tokens = highlight_tokens(state, url)?;
    let (file_id, _) = state.document(url)?;
    let edits = state
        .semantic_tokens
        .get(&file_id)
        .filter(|sent| sent.result_id.as_deref() == Some(params.previous_result_id.as_str()))
        .map(|sent| semantic_tokens::diff(&sent.data, &tokens.data));
    let tokens = state.remember_tokens(file_id, tokens);
    Some(match edits {
        Some(edits) => SemanticTokensFullDeltaResult::TokensDelta(SemanticTokensDelta {
            result_id: tokens.result_id.clone(),
            edits,
        }),
        None => SemanticTokensFullDeltaResult::Tokens(tokens.clone()),
    })
}

/// The syntax's highlights, with each resolved use highlighted as its
/// declaration. The uses come from the last check, mapped to the current
/// text.
fn highlight_tokens(state: &GlobalState, url: &Url) -> Option<SemanticTokens> {
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
    Some(to_proto::semantic_tokens(
        &document.text,
        &document.line_index,
        &highlights,
    ))
}

pub(crate) fn goto_definition(
    state: &GlobalState,
    params: &GotoDefinitionParams,
) -> Option<GotoDefinitionResponse> {
    let (_, checked, position) = checked_position(state, &params.text_document_position_params)?;
    let target = checked.goto_definition(position)?;
    let location = CheckedFiles::new(checked, state.encoding).location(&target)?;
    Some(GotoDefinitionResponse::Scalar(location))
}

pub(crate) fn references(state: &GlobalState, params: &ReferenceParams) -> Option<Vec<Location>> {
    let (_, checked, position) = checked_position(state, &params.text_document_position)?;
    let references = checked.references(position);
    let declaration = references
        .declaration
        .filter(|_| params.context.include_declaration);
    let mut locations = CheckedFiles::new(checked, state.encoding);
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

/// The names that fit at a position. The site comes from the current text.
/// The names come from the last check, which can be older. They are found
/// by where the stage around the position starts.
pub(crate) fn completion(
    state: &GlobalState,
    params: &CompletionParams,
) -> Option<CompletionResponse> {
    let position = &params.text_document_position;
    let (file_id, document) = state.document(&position.text_document.uri)?;
    let offset = from_proto::offset(&document.line_index, position.position)?;
    let mut site = state
        .analysis()
        .completion_site(FilePosition { file_id, offset })?;
    // A document not checked yet still has its keywords and locals.
    let items = match state.latest_check(&position.text_document.uri) {
        Some((_, _, checked)) => {
            let back = to_checked(document, checked, file_id)?;
            site.stage = site
                .stage
                .and_then(|stage| back.map(TextRange::empty(stage)))
                .map(TextRange::start);
            checked.completions(file_id, &site)
        }
        None => site.syntax_completions(),
    };
    let items = items
        .into_iter()
        .map(|item| lsp_types::CompletionItem {
            label: item.label,
            kind: Some(completion_kind(item.kind)),
            ..lsp_types::CompletionItem::default()
        })
        .collect();
    Some(CompletionResponse::Array(items))
}

/// How the current text of a document maps to the text its check read.
fn to_checked(document: &Document, checked: &Checked, file_id: FileId) -> Option<TextShift> {
    Some(TextShift::between(
        &document.text,
        checked.file_text(file_id)?,
    ))
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
        CompletionKind::Relation => CompletionItemKind::CLASS,
        CompletionKind::Struct | CompletionKind::Type => CompletionItemKind::STRUCT,
        CompletionKind::Trait => CompletionItemKind::INTERFACE,
    }
}

/// The name a rename at a position would change.
pub(crate) fn prepare_rename(
    state: &GlobalState,
    params: &TextDocumentPositionParams,
) -> Result<Option<PrepareRenameResponse>, String> {
    let (document, checked, position) = fresh_position(state, params)?;
    let range = checked
        .prepare_rename(position)
        .map_err(|error| error.to_string())?;
    Ok(Some(PrepareRenameResponse::Range(to_proto::range(
        &document.line_index,
        range,
    ))))
}

/// Writes the new name over the declaration and its uses.
pub(crate) fn rename(
    state: &GlobalState,
    params: &RenameParams,
) -> Result<Option<WorkspaceEdit>, String> {
    let (_, checked, position) = fresh_position(state, &params.text_document_position)?;
    let edits = checked
        .rename(position, &params.new_name)
        .map_err(|error| error.to_string())?;
    // The edits do not fit an open file that changed after the check.
    for edit in &edits {
        if let Ok(url) = Url::from_file_path(&edit.path)
            && let Some((_, open)) = state.document(&url)
            && checked.path_text(&edit.path) != Some(open.text.as_str())
        {
            return Err(format!(
                "`{}` changed since it was last checked; try again",
                edit.path.display()
            ));
        }
    }
    let mut locations = CheckedFiles::new(checked, state.encoding);
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

/// The document a position is in, its current check, and the position as
/// an offset. The error tells a request that changes files why there is
/// none.
fn fresh_position<'s>(
    state: &'s GlobalState,
    params: &TextDocumentPositionParams,
) -> Result<(&'s Document, &'s Checked, FilePosition), String> {
    let url = &params.text_document.uri;
    if !state.is_open(url) {
        return Err("the file is not open".to_owned());
    }
    let (file_id, document, checked) = state
        .fresh_check(url)
        .ok_or("the file is not checked since its last change; try again")?;
    let offset = from_proto::offset(&document.line_index, params.position)
        .ok_or("the position is past the end of the file")?;
    Ok((document, checked, FilePosition { file_id, offset }))
}

/// The overloads of the function a call names. The call is found in the
/// current text. The function's name is then mapped back to the text of the
/// last check, which resolved it.
pub(crate) fn signature_help(
    state: &GlobalState,
    params: &SignatureHelpParams,
) -> Option<SignatureHelp> {
    let position = &params.text_document_position_params;
    let (file_id, document, checked) = state.latest_check(&position.text_document.uri)?;
    let offset = from_proto::offset(&document.line_index, position.position)?;
    let site = state.analysis().call_at(FilePosition { file_id, offset })?;
    let back = to_checked(document, checked, file_id)?;
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

/// The hints of the last check in the requested range, mapped to the
/// current text.
pub(crate) fn inlay_hint(state: &GlobalState, params: &InlayHintParams) -> Option<Vec<InlayHint>> {
    let (file_id, document, checked, shift) = state.last_check(&params.text_document.uri)?;
    let requested = from_proto::text_range(&document.line_index, params.range)?;
    // The requested range in the checked text. It is all of the text when the
    // range touches the edit.
    let checked_text = checked.file_text(file_id)?;
    let in_checked = to_checked(document, checked, file_id)
        .and_then(|back| back.map(requested))
        .unwrap_or_else(|| TextRange::up_to(TextSize::of(checked_text)));
    let hints = checked
        .inlay_hints(file_id, in_checked)
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

/// The document a position is in, its current check, and the position as
/// an offset.
fn checked_position<'s>(
    state: &'s GlobalState,
    params: &TextDocumentPositionParams,
) -> Option<(&'s Document, &'s Checked, FilePosition)> {
    let (file_id, document, checked) = state.fresh_check(&params.text_document.uri)?;
    let offset = from_proto::offset(&document.line_index, params.position)?;
    Some((document, checked, FilePosition { file_id, offset }))
}
