//! Analysis answers to protocol types. Every offset here comes from the
//! analysis of the same text the line index was built from.

use line_index::WideEncoding;
use lsp_types::{
    DiagnosticRelatedInformation, DiagnosticSeverity, DocumentSymbol, FoldingRange,
    FoldingRangeKind, Location, NumberOrString, Position, Range, SelectionRange, SemanticTokens,
    SymbolKind, Url,
};
use rustc_hash::FxHashMap;
use text_size::{TextRange, TextSize};
use yuzu_diagnostics::{Diagnostic, LabelStyle, Severity, SourceId};
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

/// The files a check read, as the protocol names them. Each file has its
/// URL and a line index of the text the check read. A source with no file
/// on disk has no URL.
pub(crate) struct CheckedFiles<'c> {
    checked: &'c Checked,
    encoding: PositionEncoding,
    files: FxHashMap<SourceId, Option<(Url, LineIndex)>>,
}

impl<'c> CheckedFiles<'c> {
    pub(crate) fn new(checked: &'c Checked, encoding: PositionEncoding) -> Self {
        Self {
            checked,
            encoding,
            files: FxHashMap::default(),
        }
    }

    /// Reads a source's file, when it has one, for [`Self::get`].
    pub(crate) fn load(&mut self, source: SourceId) {
        let (checked, encoding) = (self.checked, self.encoding);
        self.files.entry(source).or_insert_with(|| {
            let url = Url::from_file_path(checked.path(source)?).ok()?;
            Some((url, LineIndex::new(checked.text(source), encoding)))
        });
    }

    /// A source's file, once [`Self::load`] read it.
    pub(crate) fn get(&self, source: SourceId) -> Option<(&Url, &LineIndex)> {
        self.files
            .get(&source)?
            .as_ref()
            .map(|(url, line_index)| (url, line_index))
    }

    pub(crate) fn location(&mut self, target: &FileRange) -> Option<Location> {
        let source = self.checked.source_of(&target.path)?;
        self.load(source);
        let (url, line_index) = self.get(source)?;
        Some(Location::new(url.clone(), range(line_index, target.range)))
    }
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
        severity: Some(severity(diagnostic.severity)),
        code: diagnostic.code.clone().map(NumberOrString::String),
        source: Some("yuzu".to_owned()),
        message,
        related_information: (!related.is_empty()).then_some(related),
        ..lsp_types::Diagnostic::default()
    };
    Some((url, diagnostic))
}

fn severity(severity: Severity) -> DiagnosticSeverity {
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

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use lsp_types::Url;
    use text_size::TextRange;
    use yuzu_diagnostics::{DiagnosticBuilder, SourceMap, Span};
    use yuzu_ide::{Fold, FoldKind, StructureNode, StructureNodeKind};

    use crate::capabilities::Folding;
    use crate::line_index::{LineIndex, PositionEncoding};

    fn range(start: u32, end: u32) -> TextRange {
        TextRange::new(start.into(), end.into())
    }

    #[test]
    fn a_symbol_nests_under_its_parent() {
        let text = "struct P {\n    x: int64,\n}\ndef f() {}\n";
        let node = |parent, label: &str, range, kind| StructureNode {
            parent,
            label: label.to_owned(),
            navigation_range: range,
            node_range: range,
            kind,
            detail: None,
        };
        let nodes = vec![
            node(None, "P", range(0, 27), StructureNodeKind::Struct),
            node(Some(0), "x", range(15, 23), StructureNodeKind::Field),
            node(None, "f", range(28, 38), StructureNodeKind::Function),
        ];
        let line_index = LineIndex::new(text, PositionEncoding::Utf8);
        let rendered: Vec<String> = super::document_symbols(&line_index, nodes)
            .iter()
            .map(|symbol| {
                let children: Vec<&str> = symbol
                    .children
                    .iter()
                    .flatten()
                    .map(|child| child.name.as_str())
                    .collect();
                format!("{} [{}]", symbol.name, children.join(", "))
            })
            .collect();
        expect![[r"
            P [x]
            f []"]]
        .assert_eq(&rendered.join("\n"));
    }

    #[test]
    fn a_whole_line_fold_keeps_code_after_its_close() {
        let text = "def f() {\n    return 1\n} let x = 1\ndef g() {\n    return 2\n}\n";
        let line_index = LineIndex::new(text, PositionEncoding::Utf8);
        let fold = |start, end| Fold {
            range: range(start, end),
            kind: FoldKind::Block,
        };
        let rendered: Vec<String> = [fold(8, 24), fold(43, 59)]
            .into_iter()
            .map(|fold| super::folding_range(text, &line_index, Folding::Lines, fold))
            .map(|folded| format!("{}..{}", folded.start_line, folded.end_line))
            .collect();
        expect![[r"
            0..1
            3..5"]]
        .assert_eq(&rendered.join("\n"));
    }

    #[test]
    fn a_diagnostic_keeps_its_notes_and_the_labels_a_client_can_open() {
        let mut sources = SourceMap::new();
        let main = sources.add("main.yz".to_owned(), "let x: str = 1\n".to_owned());
        let library = sources.add("<library>".to_owned(), String::new());
        let diagnostic = DiagnosticBuilder::error(
            Span {
                source_id: main,
                range: range(13, 14),
            },
            "expected `str`, found `int64`",
        )
        .secondary_label(
            Span {
                source_id: main,
                range: range(7, 10),
            },
            "the annotation",
        )
        .secondary_label(
            Span {
                source_id: library,
                range: range(0, 0),
            },
            "declared here",
        )
        .note("a literal is an `int64`")
        .build();

        let url = Url::parse("file:///main.yz").unwrap();
        let line_index = LineIndex::new(sources.text(main), PositionEncoding::Utf8);
        let (at, converted) = super::diagnostic(&diagnostic, |id| {
            (id == main).then_some((&url, &line_index))
        })
        .expect("the primary label is in a file");
        assert_eq!(*at, url);
        let related: Vec<String> = converted
            .related_information
            .iter()
            .flatten()
            .map(|related| format!("{:?} {}", related.location.range, related.message))
            .collect();
        expect![[r"
            Range { start: Position { line: 0, character: 13 }, end: Position { line: 0, character: 14 } }
            expected `str`, found `int64`
            note: a literal is an `int64`
            Range { start: Position { line: 0, character: 7 }, end: Position { line: 0, character: 10 } } the annotation"]].assert_eq(&format!(
            "{:?}\n{}\n{}",
            converted.range,
            converted.message,
            related.join("\n")
        ));
    }
}
