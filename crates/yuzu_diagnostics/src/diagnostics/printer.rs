use std::fmt::Write;

use crate::diagnostics::{Diagnostic, Label, LabelStyle, Severity};
use crate::source_map::{SourceId, SourceMap};

#[derive(Debug)]
pub struct DiagnosticPrinter<'a> {
    sources: &'a SourceMap,
}

impl<'a> DiagnosticPrinter<'a> {
    #[must_use]
    pub fn new(sources: &'a SourceMap) -> Self {
        Self { sources }
    }

    #[must_use]
    pub fn print(&self, diagnostic: &Diagnostic) -> String {
        let mut out = String::new();

        let severity = match diagnostic.severity {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Remark => "remark",
        };
        out.push_str(severity);
        if !diagnostic.code.is_empty() {
            let _ = write!(out, "[{}]", diagnostic.code);
        }
        let _ = writeln!(out, ": {}", diagnostic.message);

        let gutter = match self.primary_label(diagnostic) {
            Some(primary) => self.print_snippet(&mut out, diagnostic, primary),
            None => 1,
        };

        for note in &diagnostic.notes {
            let _ = writeln!(out, "{:gutter$} = note: {note}", "");
        }

        out
    }

    fn primary_label<'d>(&self, diagnostic: &'d Diagnostic) -> Option<&'d Label> {
        diagnostic
            .labels
            .iter()
            .find(|label| matches!(label.style, LabelStyle::Primary))
            .or_else(|| diagnostic.labels.first())
    }

    fn print_snippet(&self, out: &mut String, diagnostic: &Diagnostic, primary: &Label) -> usize {
        let source = primary.span.source_id;
        let position = self.line_col(primary);
        let gutter = position.line.to_string().len();
        let line = position.line;

        let _ = writeln!(
            out,
            " --> {}:{}:{}",
            self.sources.name(source),
            position.line,
            position.col
        );

        let line_text = self.sources.line_text(source, line);
        let _ = writeln!(out, "{:gutter$} |", "");
        let _ = writeln!(out, "{line} | {line_text}");

        let mut underline = vec![b' '; line_text.len()];
        let mut trailing = "";
        for label in self.labels_on(diagnostic, source, line) {
            let mark = match label.style {
                LabelStyle::Primary => b'^',
                LabelStyle::Secondary => b'-',
            };
            let column = self.column_of(label);
            for offset in 0..self.span_len(label) {
                match underline.get_mut(column + offset) {
                    Some(slot) if *slot != b'^' => *slot = mark,
                    _ => {}
                }
            }
            if matches!(label.style, LabelStyle::Primary) && trailing.is_empty() {
                trailing = &label.message;
            }
        }
        while underline.last() == Some(&b' ') {
            underline.pop();
        }
        let _ = write!(out, "{:gutter$} | {}", "", ascii(underline));
        if !trailing.is_empty() {
            let _ = write!(out, " {trailing}");
        }
        out.push('\n');

        self.print_stacked_labels(out, diagnostic, source, line, gutter);
        gutter
    }

    fn print_stacked_labels(
        &self,
        out: &mut String,
        diagnostic: &Diagnostic,
        source: SourceId,
        line: usize,
        gutter: usize,
    ) {
        let mut stacked: Vec<&Label> = self
            .labels_on(diagnostic, source, line)
            .into_iter()
            .filter(|label| {
                matches!(label.style, LabelStyle::Secondary) && !label.message.is_empty()
            })
            .collect();
        stacked.sort_by_key(|label| label.span.range.start());
        if stacked.is_empty() {
            return;
        }

        let columns: Vec<usize> = stacked.iter().map(|label| self.column_of(label)).collect();

        let mut connectors = vec![b' '; columns[columns.len() - 1] + 1];
        for &column in &columns {
            connectors[column] = b'|';
        }
        let _ = writeln!(out, "{:gutter$} | {}", "", ascii(connectors));

        for k in (0..stacked.len()).rev() {
            let mut row = vec![b' '; columns[k]];
            for &column in &columns[..k] {
                row[column] = b'|';
            }
            row.resize(columns[k] + self.span_len(stacked[k]), b'-');
            let _ = writeln!(
                out,
                "{:gutter$} | {} {}",
                "",
                ascii(row),
                stacked[k].message
            );
        }
    }

    fn labels_on<'d>(
        &self,
        diagnostic: &'d Diagnostic,
        source: SourceId,
        line: usize,
    ) -> Vec<&'d Label> {
        diagnostic
            .labels
            .iter()
            .filter(|label| label.span.source_id == source && self.line_col(label).line == line)
            .collect()
    }

    fn line_col(&self, label: &Label) -> crate::source_map::LineCol {
        let offset = u32::from(label.span.range.start()) as usize;
        self.sources.line_col(label.span.source_id, offset)
    }

    fn column_of(&self, label: &Label) -> usize {
        self.line_col(label).col - 1
    }

    fn span_len(&self, label: &Label) -> usize {
        let start = u32::from(label.span.range.start()) as usize;
        let end = u32::from(label.span.range.end()) as usize;
        end.saturating_sub(start).max(1)
    }
}

fn ascii(bytes: Vec<u8>) -> String {
    String::from_utf8(bytes).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};
    use text_size::{TextRange, TextSize};

    use super::*;
    use crate::diagnostics::{Span, builder::DiagnosticBuilder};

    fn span(source: SourceId, range: std::ops::Range<u32>) -> Span {
        Span {
            source_id: source,
            range: TextRange::new(TextSize::from(range.start), TextSize::from(range.end)),
        }
    }

    fn check(source: &str, build: impl FnOnce(SourceId) -> Diagnostic, expected: &Expect) {
        let mut sources = SourceMap::new();
        let id = sources.add("test.yuzu".to_string(), source.to_string());
        let diagnostic = build(id);
        let printer = DiagnosticPrinter::new(&sources);
        expected.assert_eq(&printer.print(&diagnostic));
    }

    #[test]
    fn single_line_error() {
        check(
            "let x: int64 = true\n",
            |id| {
                DiagnosticBuilder::error(
                    span(id, 15..19),
                    "value of type `Bool` is not assignable to `Int64`",
                )
                .build()
            },
            &expect![[r"
                error: value of type `Bool` is not assignable to `Int64`
                 --> test.yuzu:1:16
                  |
                1 | let x: int64 = true
                  |                ^^^^
            "]],
        );
    }

    #[test]
    fn error_with_code_and_note() {
        check(
            "let x = 1\n",
            |id| {
                DiagnosticBuilder::error(span(id, 8..9), "unexpected token")
                    .code("E0001")
                    .note("expected an expression")
                    .build()
            },
            &expect![[r"
                error[E0001]: unexpected token
                 --> test.yuzu:1:9
                  |
                1 | let x = 1
                  |         ^
                  = note: expected an expression
            "]],
        );
    }

    #[test]
    fn primary_and_secondary_underlines() {
        check(
            "let x = true\n",
            |id| {
                DiagnosticBuilder::error(span(id, 8..12), "type error")
                    .label(span(id, 4..5), "")
                    .build()
            },
            &expect![[r"
                error: type error
                 --> test.yuzu:1:9
                  |
                1 | let x = true
                  |     -   ^^^^
            "]],
        );
    }

    #[test]
    fn stacked_secondary_messages() {
        check(
            "1.0 + 2\n",
            |id| {
                DiagnosticBuilder::error(span(id, 0..7), "mismatched operand types")
                    .label(span(id, 0..3), "this is a `float`")
                    .label(span(id, 6..7), "this is an `int`")
                    .build()
            },
            &expect![[r"
                error: mismatched operand types
                 --> test.yuzu:1:1
                  |
                1 | 1.0 + 2
                  | ^^^^^^^
                  | |     |
                  | |     - this is an `int`
                  | --- this is a `float`
            "]],
        );
    }
}
