use crate::diagnostics::{Diagnostic, Label, LabelStyle, Severity, Span};

#[derive(Debug)]
#[must_use = "a diagnostic does nothing until it is emitted"]
pub struct DiagnosticBuilder {
    diagnostic: Diagnostic,
}

impl DiagnosticBuilder {
    pub fn error(span: Span, message: impl Into<String>) -> Self {
        Self::new(Severity::Error, span, message.into())
    }

    pub fn warning(span: Span, message: impl Into<String>) -> Self {
        Self::new(Severity::Warning, span, message.into())
    }

    pub fn remark(span: Span, message: impl Into<String>) -> Self {
        Self::new(Severity::Remark, span, message.into())
    }

    /// An error with no place in the source, such as one about the program
    /// as a whole. It prints as its message alone, the way a compiler
    /// reports a file it could not read.
    pub fn error_without_span(message: impl Into<String>) -> Self {
        Self::without_span(Severity::Error, message)
    }

    /// A diagnostic of any severity with no place in the source.
    pub fn without_span(severity: Severity, message: impl Into<String>) -> Self {
        Self {
            diagnostic: Diagnostic {
                severity,
                code: None,
                message: message.into(),
                labels: Vec::new(),
                notes: Vec::new(),
            },
        }
    }

    fn new(severity: Severity, span: Span, message: String) -> Self {
        Self {
            diagnostic: Diagnostic {
                severity,
                code: None,
                message,
                labels: vec![Label {
                    style: LabelStyle::Primary,
                    span,
                    message: String::new(),
                }],
                notes: Vec::new(),
            },
        }
    }

    pub fn code(mut self, code: impl Into<String>) -> Self {
        self.diagnostic.code = Some(code.into());
        self
    }

    pub fn primary_label(mut self, span: Span, message: impl Into<String>) -> Self {
        self.diagnostic.labels.push(Label {
            style: LabelStyle::Primary,
            span,
            message: message.into(),
        });
        self
    }

    pub fn secondary_label(mut self, span: Span, message: impl Into<String>) -> Self {
        self.diagnostic.labels.push(Label {
            style: LabelStyle::Secondary,
            span,
            message: message.into(),
        });
        self
    }

    pub fn note(mut self, text: impl Into<String>) -> Self {
        self.diagnostic.notes.push(text.into());
        self
    }

    #[must_use]
    pub fn build(self) -> Diagnostic {
        self.diagnostic
    }
}

impl From<DiagnosticBuilder> for Diagnostic {
    fn from(builder: DiagnosticBuilder) -> Self {
        builder.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source_map::SourceMap;
    use text_size::TextRange;

    fn dummy_span() -> Span {
        let mut sources = SourceMap::new();
        let source = sources.add(String::new(), String::new());
        Span {
            source_id: source,
            range: TextRange::default(),
        }
    }

    #[test]
    fn error_seeds_severity_message_and_primary_label() {
        let diagnostic = DiagnosticBuilder::error(dummy_span(), "boom").build();

        assert_eq!(diagnostic.severity, Severity::Error);
        assert_eq!(diagnostic.message, "boom");
        assert!(diagnostic.code.is_none());
        assert!(diagnostic.notes.is_empty());
        assert_eq!(diagnostic.labels.len(), 1);
        assert!(matches!(diagnostic.labels[0].style, LabelStyle::Primary));
    }

    #[test]
    fn an_error_without_a_span_carries_no_label() {
        let diagnostic = DiagnosticBuilder::error_without_span("boom").build();

        assert_eq!(diagnostic.severity, Severity::Error);
        assert_eq!(diagnostic.message, "boom");
        assert!(diagnostic.labels.is_empty());
    }

    #[test]
    fn warning_and_remark_set_severity() {
        let warning = DiagnosticBuilder::warning(dummy_span(), "").build();
        let remark = DiagnosticBuilder::remark(dummy_span(), "").build();

        assert_eq!(warning.severity, Severity::Warning);
        assert_eq!(remark.severity, Severity::Remark);
    }

    #[test]
    fn setters_build_up_the_diagnostic() {
        let diagnostic = DiagnosticBuilder::error(dummy_span(), "msg")
            .code("E0001")
            .primary_label(dummy_span(), "here")
            .secondary_label(dummy_span(), "context")
            .note("first note")
            .note("second note")
            .build();

        assert_eq!(diagnostic.code.as_deref(), Some("E0001"));
        assert_eq!(diagnostic.notes, ["first note", "second note"]);
        assert_eq!(diagnostic.labels.len(), 3);
        assert!(matches!(diagnostic.labels[1].style, LabelStyle::Primary));
        assert!(matches!(diagnostic.labels[2].style, LabelStyle::Secondary));
        assert_eq!(diagnostic.labels[2].message, "context");
    }
}
