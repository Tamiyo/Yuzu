use std::fmt::Write;

use crate::diagnostics::{Diagnostic, Label, LabelStyle, Severity};
use crate::source_map::{SourceId, SourceMap};

/// Renders diagnostics as text, with the source lines they point into.
#[derive(Debug)]
pub struct DiagnosticPrinter<'a> {
    sources: &'a SourceMap,
}

impl<'a> DiagnosticPrinter<'a> {
    /// A printer for diagnostics that point into `sources`.
    #[must_use]
    pub fn new(sources: &'a SourceMap) -> Self {
        Self { sources }
    }

    /// Every diagnostic rendered, one after another.
    #[must_use]
    pub fn render_all(&self, diagnostics: &[Diagnostic]) -> String {
        diagnostics
            .iter()
            .map(|diagnostic| self.render(diagnostic))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// One diagnostic: its message, each line it points at, and its notes.
    #[must_use]
    pub fn render(&self, diagnostic: &Diagnostic) -> String {
        let mut out = String::new();

        let severity = match diagnostic.severity {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Remark => "remark",
        };
        out.push_str(severity);
        if let Some(code) = &diagnostic.code {
            let _ = write!(out, "[{code}]");
        }
        let _ = writeln!(out, ": {}", diagnostic.message);

        let lines = self.snippet_lines(diagnostic);
        let gutter = lines
            .iter()
            .map(|&(_, line)| digits(line))
            .max()
            .unwrap_or(1);
        for (source, line) in lines {
            self.render_snippet(&mut out, diagnostic, source, line, gutter);
        }

        for note in &diagnostic.notes {
            let _ = writeln!(out, "{:gutter$} = note: {note}", "");
        }

        out
    }

    /// Each line a label is on: the primary label's first, then the rest in
    /// source order.
    fn snippet_lines(&self, diagnostic: &Diagnostic) -> Vec<(SourceId, usize)> {
        let mut lines: Vec<(SourceId, usize)> = Vec::new();
        let primary = primary_label(diagnostic.labels.iter());
        let mut rest: Vec<(SourceId, usize)> = diagnostic
            .labels
            .iter()
            .map(|label| (label.span.source_id, self.line_col(label).line))
            .collect();
        rest.sort_unstable();
        if let Some(primary) = primary {
            lines.push((primary.span.source_id, self.line_col(primary).line));
        }
        for line in rest {
            if !lines.contains(&line) {
                lines.push(line);
            }
        }
        lines
    }

    fn render_snippet(
        &self,
        out: &mut String,
        diagnostic: &Diagnostic,
        source: SourceId,
        line: usize,
        gutter: usize,
    ) {
        let labels = self.labels_on(diagnostic, source, line);
        let line_text = self.sources.line_text(source, line);
        let first = primary_label(labels.iter().copied()).expect("a snippet's line holds a label");
        let column = line_text
            .get(..self.byte_column(first))
            .map_or(1, |before| before.chars().count() + 1);

        let _ = writeln!(out, " --> {}:{line}:{column}", self.sources.name(source));
        let _ = writeln!(out, "{:gutter$} |", "");
        let _ = writeln!(out, "{line:>gutter$} | {}", shown(line_text));

        let mut underline = vec![' '; width(line_text)];
        let mut trailing = "";
        for label in &labels {
            let mark = match label.style {
                LabelStyle::Primary => '^',
                LabelStyle::Secondary => '-',
            };
            let (start, len) = self.cells(line_text, label);
            for slot in underline.iter_mut().skip(start).take(len) {
                if *slot != '^' {
                    *slot = mark;
                }
            }
            if underline.len() < start + len {
                underline.resize(start + len, mark);
            }
            if matches!(label.style, LabelStyle::Primary) && trailing.is_empty() {
                trailing = &label.message;
            }
        }
        while underline.last() == Some(&' ') {
            underline.pop();
        }
        let underline: String = underline.into_iter().collect();
        let _ = write!(out, "{:gutter$} | {underline}", "");
        if !trailing.is_empty() {
            let _ = write!(out, " {trailing}");
        }
        out.push('\n');

        self.render_stacked_labels(out, &labels, line_text, gutter);
    }

    fn render_stacked_labels(
        &self,
        out: &mut String,
        labels: &[&Label],
        line_text: &str,
        gutter: usize,
    ) {
        let mut stacked: Vec<&Label> = labels
            .iter()
            .copied()
            .filter(|label| {
                matches!(label.style, LabelStyle::Secondary) && !label.message.is_empty()
            })
            .collect();
        stacked.sort_by_key(|label| label.span.range.start());
        if stacked.is_empty() {
            return;
        }

        let cells: Vec<(usize, usize)> = stacked
            .iter()
            .map(|label| self.cells(line_text, label))
            .collect();

        let mut connectors = vec![' '; cells[cells.len() - 1].0 + 1];
        for &(column, _) in &cells {
            connectors[column] = '|';
        }
        let connectors: String = connectors.into_iter().collect();
        let _ = writeln!(out, "{:gutter$} | {connectors}", "");

        for k in (0..stacked.len()).rev() {
            let (column, len) = cells[k];
            let mut row = vec![' '; column];
            for &(before, _) in &cells[..k] {
                row[before] = '|';
            }
            row.resize(column + len, '-');
            let row: String = row.into_iter().collect();
            let _ = writeln!(out, "{:gutter$} | {row} {}", "", stacked[k].message);
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

    fn byte_column(&self, label: &Label) -> usize {
        self.line_col(label).col - 1
    }

    /// Where a label starts on the printed line and how many cells it
    /// covers there, at least one.
    fn cells(&self, line_text: &str, label: &Label) -> (usize, usize) {
        let start = self.byte_column(label).min(line_text.len());
        let end = (start + span_len(label)).min(line_text.len());
        let before = line_text.get(..start).map_or(start, width);
        let under = line_text.get(start..end).map_or(1, width).max(1);
        (before, under)
    }
}

fn width(text: &str) -> usize {
    text.chars().map(|c| if c == '\t' { 4 } else { 1 }).sum()
}

/// A line as printed, each tab as four spaces so the marks under it line up.
fn shown(line_text: &str) -> String {
    line_text.replace('\t', "    ")
}

fn primary_label<'l>(mut labels: impl Iterator<Item = &'l Label> + Clone) -> Option<&'l Label> {
    labels
        .clone()
        .find(|label| matches!(label.style, LabelStyle::Primary))
        .or_else(|| labels.next())
}

fn digits(line: usize) -> usize {
    line.checked_ilog10().map_or(1, |log| log as usize + 1)
}

fn span_len(label: &Label) -> usize {
    let start = u32::from(label.span.range.start()) as usize;
    let end = u32::from(label.span.range.end()) as usize;
    end.saturating_sub(start).max(1)
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
        expected.assert_eq(&printer.render(&diagnostic));
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
                    .secondary_label(span(id, 4..5), "")
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
                    .secondary_label(span(id, 0..3), "this is a `float`")
                    .secondary_label(span(id, 6..7), "this is an `int`")
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

    #[test]
    fn marks_line_up_after_a_tab_and_a_wide_character() {
        check(
            "\tlet é = x\n",
            |id| DiagnosticBuilder::error(span(id, 10..11), "unknown name").build(),
            &expect![[r"
                error: unknown name
                 --> test.yuzu:1:10
                  |
                1 |     let é = x
                  |             ^
            "]],
        );
    }

    #[test]
    fn a_label_on_another_line_gets_its_own_snippet() {
        check(
            "let x = 1\nlet x = 2\n",
            |id| {
                DiagnosticBuilder::error(span(id, 14..15), "`x` is declared twice")
                    .secondary_label(span(id, 4..5), "first declared here")
                    .build()
            },
            &expect![[r"
                error: `x` is declared twice
                 --> test.yuzu:2:5
                  |
                2 | let x = 2
                  |     ^
                 --> test.yuzu:1:5
                  |
                1 | let x = 1
                  |     -
                  |     |
                  |     - first declared here
            "]],
        );
    }
}
