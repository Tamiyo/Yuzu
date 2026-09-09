//! MLIR diagnostics folded into Yuzu's engine. `capture` attaches one
//! handler to the context, so everything MLIR emits inside it — pass errors
//! sent through `emit_error`, verifier failures, parse errors — lands in
//! the engine as span-carrying diagnostics instead of on stderr.

use std::cell::RefCell;
use std::ffi::CString;
use std::rc::Rc;

use melior::Context;
use melior::diagnostic::{Diagnostic, DiagnosticSeverity};
use melior::ir::Location;
use text_size::{TextRange, TextSize};
use yuzu_diagnostics::diagnostics::Span;
use yuzu_diagnostics::diagnostics::builder::DiagnosticBuilder;
use yuzu_diagnostics::diagnostics::engine::DiagnosticsEngine;
use yuzu_diagnostics::source_map::SourceId;

/// Emits an error against a location, into whichever handler is attached.
pub fn emit_error(location: Location, message: &str) {
    let message = CString::new(message).expect("diagnostic messages have no interior nul");
    unsafe { mlir_sys::mlirEmitError(location.to_raw(), message.as_ptr()) }
}

/// Runs `f` with MLIR diagnostics routed into the engine.
pub fn capture<T>(
    context: &Context,
    source_id: SourceId,
    source: &str,
    engine: &mut DiagnosticsEngine,
    f: impl FnOnce() -> T,
) -> T {
    let spans = Spans::new(source_id, source);
    let collected = Rc::new(RefCell::new(Vec::new()));
    let sink = collected.clone();
    let handler = context.attach_diagnostic_handler(move |diagnostic| {
        sink.borrow_mut().push(convert(&spans, &diagnostic));
        true
    });

    let result = f();
    context.detach_diagnostic_handler(handler);
    for diagnostic in collected.take() {
        engine.emit(diagnostic);
    }

    result
}

fn convert(spans: &Spans, diagnostic: &Diagnostic) -> DiagnosticBuilder {
    let span = spans.span(diagnostic.location());
    let message = diagnostic.to_string();
    let mut builder = match diagnostic.severity() {
        DiagnosticSeverity::Error => DiagnosticBuilder::error(span, message),
        DiagnosticSeverity::Warning => DiagnosticBuilder::warning(span, message),
        DiagnosticSeverity::Note | DiagnosticSeverity::Remark => {
            DiagnosticBuilder::remark(span, message)
        }
    };

    for index in 0..diagnostic.note_count() {
        if let Ok(note) = diagnostic.note(index) {
            builder = builder.note(note.to_string());
        }
    }

    builder
}

/// MLIR locations mapped back to the engine's spans: the inverse of the
/// locations a conversion mints on the way in.
struct Spans {
    id: SourceId,
    line_starts: Vec<usize>,
    len: usize,
}

impl Spans {
    fn new(id: SourceId, text: &str) -> Self {
        let mut line_starts = vec![0];
        line_starts.extend(text.match_indices('\n').map(|(at, _)| at + 1));
        Self {
            id,
            line_starts,
            len: text.len(),
        }
    }

    /// The span of a location. Locations print as `loc("name":line:col)`;
    /// anything else — an unknown location, a fused one — falls back to the
    /// start of the source. The textual round trip stands in for the
    /// FileLineCol getters the MLIR C API does not expose yet.
    fn span(&self, location: Location) -> Span {
        let offset = self.offset(&location.to_string()).unwrap_or_default();
        let end = (offset + 1).min(self.len);
        Span {
            source_id: self.id,
            range: TextRange::new(TextSize::new(offset as u32), TextSize::new(end as u32)),
        }
    }

    fn offset(&self, printed: &str) -> Option<usize> {
        let (_, tail) = printed.rsplit_once("\":")?;
        let (line, column) = tail.strip_suffix(')')?.split_once(':')?;
        let line: usize = line.parse().ok()?;
        let column: usize = column.parse().ok()?;
        let start = *self.line_starts.get(line.checked_sub(1)?)?;
        Some(start + column - 1)
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use melior::ir::Location;
    use yuzu_diagnostics::diagnostics::engine::DiagnosticsEngine;
    use yuzu_diagnostics::diagnostics::printer::DiagnosticPrinter;
    use yuzu_diagnostics::source_map::SourceMap;

    use super::{capture, emit_error};

    #[test]
    fn emitted_errors_render_with_spans() {
        let context = crate::context();
        let source = "from t\n";
        let mut sources = SourceMap::new();
        let source_id = sources.add("test.yz".to_string(), source.to_string());
        let mut diagnostics = DiagnosticsEngine::new();

        capture(&context, source_id, source, &mut diagnostics, || {
            let location = Location::new(&context, "test.yz", 1, 6);
            emit_error(location, "unknown relation `t`");
        });

        let printer = DiagnosticPrinter::new(&sources);
        let rendered: Vec<String> = diagnostics
            .diagnostics()
            .iter()
            .map(|diagnostic| printer.print(diagnostic))
            .collect();
        expect![[r#"
        error: unknown relation `t`
         --> test.yz:1:6
          |
        1 | from t
          |      ^
    "#]]
        .assert_eq(&rendered.join("\n"));
    }

    #[test]
    fn verifier_failures_land_in_the_engine() {
        use melior::ir::operation::{OperationBuilder, OperationLike};
        use melior::ir::{BlockLike, Module, Type};

        let context = crate::context();
        let source = "from t\n";
        let mut sources = SourceMap::new();
        let source_id = sources.add("test.yz".to_string(), source.to_string());
        let mut diagnostics = DiagnosticsEngine::new();

        // A yz.add whose result is not a numeric type — the ODS verifier rejects
        // it, and capture turns that into an engine diagnostic.
        let location = Location::new(&context, "test.yz", 1, 1);
        let module = Module::new(location);
        let boolean = Type::parse(&context, "!yz.bool").expect("!yz.bool parses");
        let verified = capture(&context, source_id, source, &mut diagnostics, || {
            let constant = module.body().append_operation(
                OperationBuilder::new("yz.constant_bool", location)
                    .add_attributes(&[(
                        melior::ir::Identifier::new(&context, "value"),
                        melior::ir::Attribute::parse(&context, "true").unwrap(),
                    )])
                    .add_results(&[boolean])
                    .build()
                    .expect("the constant builds"),
            );
            module.body().append_operation(
                OperationBuilder::new("yz.add", location)
                    .add_operands(&[
                        constant.result(0).unwrap().into(),
                        constant.result(0).unwrap().into(),
                    ])
                    .add_results(&[boolean])
                    .build()
                    .expect("the add builds"),
            );
            module.as_operation().verify()
        });

        assert!(!verified, "the mistyped add must not verify");
        let messages: Vec<&str> = diagnostics
            .diagnostics()
            .iter()
            .map(|diagnostic| diagnostic.message.as_str())
            .collect();
        assert!(
            messages.iter().any(|message| message.contains("yz.add")),
            "the verifier error names the op: {messages:?}"
        );
    }
}
