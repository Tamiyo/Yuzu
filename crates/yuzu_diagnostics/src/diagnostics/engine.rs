use crate::diagnostics::{Diagnostic, Severity};

#[derive(Debug)]
pub struct DiagnosticsEngine {
    diagnostics: Vec<Diagnostic>,
}

impl Default for DiagnosticsEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl DiagnosticsEngine {
    #[must_use]
    pub fn new() -> Self {
        Self {
            diagnostics: Vec::new(),
        }
    }

    pub fn emit(&mut self, diagnostic: impl Into<Diagnostic>) {
        self.diagnostics.push(diagnostic.into());
    }

    #[must_use]
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|diagnostic| matches!(diagnostic.severity, Severity::Error))
    }

    #[must_use]
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    #[must_use]
    pub fn into_diagnostics(self) -> Vec<Diagnostic> {
        self.diagnostics
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnostics::Span;
    use crate::diagnostics::builder::DiagnosticBuilder;
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
    fn emits_a_built_diagnostic() {
        let mut engine = DiagnosticsEngine::new();
        engine.emit(
            DiagnosticBuilder::error(dummy_span(), "expected expression")
                .code("E0001")
                .label(dummy_span(), "this `+` needs a right-hand side")
                .note("expressions can start with a number, identifier, or `(`"),
        );

        let diagnostic = &engine.diagnostics()[0];
        assert_eq!(diagnostic.severity, Severity::Error);
        assert_eq!(diagnostic.code, "E0001");
        assert_eq!(diagnostic.message, "expected expression");
        assert_eq!(diagnostic.labels.len(), 2);
        assert_eq!(diagnostic.notes.len(), 1);
    }

    #[test]
    fn new_engine_has_no_diagnostics() {
        let engine = DiagnosticsEngine::new();
        assert!(engine.diagnostics().is_empty());
    }

    #[test]
    fn emit_accepts_an_already_built_diagnostic() {
        let mut engine = DiagnosticsEngine::new();
        let diagnostic = DiagnosticBuilder::warning(dummy_span(), "deprecated").build();

        engine.emit(diagnostic);

        assert_eq!(engine.diagnostics().len(), 1);
        assert_eq!(engine.diagnostics()[0].severity, Severity::Warning);
    }

    #[test]
    fn emits_in_order() {
        let mut engine = DiagnosticsEngine::new();
        engine.emit(DiagnosticBuilder::error(dummy_span(), "first"));
        engine.emit(DiagnosticBuilder::remark(dummy_span(), "second"));

        let messages: Vec<_> = engine
            .diagnostics()
            .iter()
            .map(|d| d.message.as_str())
            .collect();
        assert_eq!(messages, ["first", "second"]);
    }
}
