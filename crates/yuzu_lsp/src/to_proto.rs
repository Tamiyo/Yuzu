//! Analysis answers to protocol types. Every offset here comes from the
//! analysis of the same text the line index was built from.

use line_index::WideEncoding;
use lsp_types::{
    DiagnosticRelatedInformation, DiagnosticSeverity, DocumentSymbol, FoldingRange,
    FoldingRangeKind, Location, NumberOrString, Position, Range, SelectionRange, SemanticTokens,
    SymbolKind, Url,
};
use text_size::{TextRange, TextSize};
use yuzu_diagnostics::diagnostics::{Diagnostic, LabelStyle, Severity};
use yuzu_diagnostics::source_map::SourceId;
use yuzu_ide::{Checked, FileRange, Fold, FoldKind, HlRange, StructureNode, StructureNodeKind};

use crate::capabilities::Folding;
use crate::line_index::{LineIndex, PositionEncoding};
use crate::semantic_tokens::{self, SemanticTokensBuilder};

pub(crate) fn position(line_index: &LineIndex, offset: TextSize) -> Position {
    let line_col = line_index.index.line_col(offset);
    match line_index.encoding {
        PositionEncoding::Utf8 => Position::new(line_col.line, line_col.col),
        PositionEncoding::Utf16 => {
            let wide = line_index
                .index
                .to_wide(WideEncoding::Utf16, line_col)
                .expect("an offset from the analysis lies inside its text");
            Position::new(wide.line, wide.col)
        }
    }
}

pub(crate) fn range(line_index: &LineIndex, range: TextRange) -> Range {
    Range::new(
        position(line_index, range.start()),
        position(line_index, range.end()),
    )
}

/// A range in a file a check read, measured in the text the check read.
pub(crate) fn location(
    checked: &Checked,
    target: &FileRange,
    encoding: PositionEncoding,
) -> Option<Location> {
    let text = checked.file_text(&target.path)?;
    let line_index = LineIndex::new(text, encoding);
    Some(Location::new(
        Url::from_file_path(&target.path).ok()?,
        range(&line_index, target.range),
    ))
}

/// The diagnostic and the file its primary label is in. `file` gives the URL
/// and line index of the file a source was read from. `None` when the
/// primary label lies in no file the client can open, as a
/// library module built into the compiler is not; a secondary label there is
/// left out.
pub(crate) fn diagnostic<'f>(
    diagnostic: &Diagnostic,
    file: impl Fn(SourceId) -> Option<(&'f Url, &'f LineIndex)>,
) -> Option<(&'f Url, lsp_types::Diagnostic)> {
    let primary = diagnostic
        .labels
        .iter()
        .find(|label| matches!(label.style, LabelStyle::Primary))?;
    let (url, line_index) = file(primary.span.source_id)?;

    let mut message = diagnostic.message.clone();
    if !primary.message.is_empty() {
        message.push_str(": ");
        message.push_str(&primary.message);
    }
    for note in &diagnostic.notes {
        message.push_str("\nnote: ");
        message.push_str(note);
    }

    let related: Vec<DiagnosticRelatedInformation> = diagnostic
        .labels
        .iter()
        .filter(|label| matches!(label.style, LabelStyle::Secondary))
        .filter_map(|label| {
            let (url, line_index) = file(label.span.source_id)?;
            Some(DiagnosticRelatedInformation {
                location: Location::new(url.clone(), range(line_index, label.span.range)),
                message: label.message.clone(),
            })
        })
        .collect();

    let diagnostic = lsp_types::Diagnostic {
        range: range(line_index, primary.span.range),
        severity: Some(severity(&diagnostic.severity)),
        code: (!diagnostic.code.is_empty())
            .then(|| NumberOrString::String(diagnostic.code.clone())),
        source: Some("yuzu".to_owned()),
        message,
        related_information: (!related.is_empty()).then_some(related),
        ..lsp_types::Diagnostic::default()
    };
    Some((url, diagnostic))
}

fn severity(severity: &Severity) -> DiagnosticSeverity {
    match severity {
        Severity::Error => DiagnosticSeverity::ERROR,
        Severity::Warning => DiagnosticSeverity::WARNING,
        Severity::Remark => DiagnosticSeverity::INFORMATION,
    }
}

/// The flat list nests by each node's parent index. A parent comes before
/// its children, so walking backwards finishes every child first.
pub(crate) fn document_symbols(
    line_index: &LineIndex,
    nodes: Vec<StructureNode>,
) -> Vec<DocumentSymbol> {
    let mut parents = Vec::with_capacity(nodes.len());
    let mut symbols: Vec<Option<DocumentSymbol>> = Vec::with_capacity(nodes.len());
    for node in nodes {
        parents.push(node.parent);
        symbols.push(Some(document_symbol(line_index, node)));
    }

    let mut roots = Vec::new();
    for at in (0..symbols.len()).rev() {
        let symbol = symbols[at].take().expect("each symbol is placed once");
        match parents[at] {
            Some(parent) => symbols[parent]
                .as_mut()
                .expect("a parent comes before its children")
                .children
                .get_or_insert_with(Vec::new)
                .insert(0, symbol),
            None => roots.push(symbol),
        }
    }
    roots.reverse();
    roots
}

fn document_symbol(line_index: &LineIndex, node: StructureNode) -> DocumentSymbol {
    #[expect(deprecated, reason = "the protocol still requires the field")]
    DocumentSymbol {
        name: node.label,
        detail: node.detail,
        kind: symbol_kind(node.kind),
        tags: None,
        deprecated: None,
        range: range(line_index, node.node_range),
        selection_range: range(line_index, node.navigation_range),
        children: None,
    }
}

fn symbol_kind(kind: StructureNodeKind) -> SymbolKind {
    match kind {
        StructureNodeKind::Module => SymbolKind::MODULE,
        StructureNodeKind::Function => SymbolKind::FUNCTION,
        StructureNodeKind::Struct | StructureNodeKind::Table => SymbolKind::STRUCT,
        StructureNodeKind::Trait => SymbolKind::INTERFACE,
        StructureNodeKind::Impl | StructureNodeKind::Query => SymbolKind::OBJECT,
        StructureNodeKind::Field => SymbolKind::FIELD,
        StructureNodeKind::Constant => SymbolKind::CONSTANT,
    }
}

/// A client that folds whole lines hides the last line too, so a fold that
/// ends before more code on its line stops one line earlier.
pub(crate) fn folding_range(
    text: &str,
    line_index: &LineIndex,
    folding: Folding,
    fold: Fold,
) -> FoldingRange {
    let range = range(line_index, fold.range);
    let kind = match fold.kind {
        FoldKind::Comment => Some(FoldingRangeKind::Comment),
        FoldKind::Imports => Some(FoldingRangeKind::Imports),
        FoldKind::Block | FoldKind::Query => None,
    };

    match folding {
        Folding::Lines => {
            let rest = &text[usize::from(fold.range.end())..];
            let more_on_end_line = rest
                .chars()
                .take_while(|&c| c != '\n')
                .any(|c| !c.is_whitespace());
            let end_line = if more_on_end_line {
                range.end.line.saturating_sub(1)
            } else {
                range.end.line
            };
            FoldingRange {
                start_line: range.start.line,
                start_character: None,
                end_line,
                end_character: None,
                kind,
                collapsed_text: None,
            }
        }
        Folding::Characters => FoldingRange {
            start_line: range.start.line,
            start_character: Some(range.start.character),
            end_line: range.end.line,
            end_character: Some(range.end.character),
            kind,
            collapsed_text: None,
        },
    }
}

/// `ranges` goes from the innermost out; each protocol range names the one
/// around it as its parent.
pub(crate) fn selection_range(line_index: &LineIndex, ranges: &[TextRange]) -> SelectionRange {
    let mut outer: Option<SelectionRange> = None;
    for &inner in ranges.iter().rev() {
        outer = Some(SelectionRange {
            range: range(line_index, inner),
            parent: outer.map(Box::new),
        });
    }
    outer.expect("a position has at least the file around it")
}

/// A token over several lines is sent as one token for each line.
pub(crate) fn semantic_tokens(
    text: &str,
    line_index: &LineIndex,
    highlights: &[HlRange],
) -> SemanticTokens {
    let mut builder = SemanticTokensBuilder::default();
    for highlight in highlights {
        let token_type = semantic_tokens::token_type(highlight.highlight.tag);
        let token_modifiers = semantic_tokens::token_modifiers(highlight.highlight.mods);
        for mut line in line_index.index.lines(highlight.range) {
            if text[line].ends_with('\n') {
                line = TextRange::new(line.start(), line.end() - TextSize::of('\n'));
            }
            if !line.is_empty() {
                builder.push(range(line_index, line), token_type, token_modifiers);
            }
        }
    }
    builder.build()
}
