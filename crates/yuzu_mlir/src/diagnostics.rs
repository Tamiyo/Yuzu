//! MLIR diagnostics folded into Yuzu's engine. `capture` attaches one
//! handler to the context, so everything MLIR emits inside it — pass errors
//! sent through `emit_error`, verifier failures, parse errors — lands in
//! the engine as span-carrying diagnostics instead of on stderr.

use std::cell::RefCell;
use std::ffi::CString;
use std::rc::Rc;

use melior::Context;
use melior::diagnostic::{Diagnostic, DiagnosticHandlerId, DiagnosticSeverity};
use melior::ir::Location;
use text_size::{TextRange, TextSize};
use yuzu_diagnostics::diagnostics::Span;
use yuzu_diagnostics::diagnostics::builder::DiagnosticBuilder;
use yuzu_diagnostics::diagnostics::engine::DiagnosticsEngine;
use yuzu_diagnostics::source_map::SourceMap;

/// Emits an error against a location, into whichever handler is attached.
pub fn emit_error(location: Location, message: &str) {
    let message = CString::new(message).expect("diagnostic messages have no interior nul");
    // SAFETY: the location belongs to a live context, and the message is nul-terminated and outlives the call.
    unsafe { mlir_sys::mlirEmitError(location.to_raw(), message.as_ptr()) }
}

/// Runs `f` with MLIR diagnostics routed into the engine.
///
/// A location names the file it came from, so every source the compile read
/// is offered here and each diagnostic lands in the one it belongs to.
pub fn capture<T>(
    context: &Context,
    sources: &SourceMap,
    engine: &mut DiagnosticsEngine,
    f: impl FnOnce() -> T,
) -> T {
    let collected = Rc::new(RefCell::new(Vec::new()));
    let sink = collected.clone();
    let handler = context.attach_diagnostic_handler(move |diagnostic| {
        sink.borrow_mut().push(Reported::of(&diagnostic));
        true
    });

    let result = {
        let _attached = Attached { context, handler };
        f()
    };
    for reported in collected.take() {
        engine.emit(reported.build(sources));
    }

    result
}

/// A handler attached to a context. It is detached on drop, so a panic in
/// `f` does not leave it on a context the thread reuses.
struct Attached<'c> {
    context: &'c Context,
    handler: DiagnosticHandlerId,
}

impl Drop for Attached<'_> {
    fn drop(&mut self) {
        self.context.detach_diagnostic_handler(self.handler);
    }
}

/// What a handler saw. The handler has to outlive this call as far as the
/// type system is concerned, so it cannot borrow the sources a span needs.
/// It writes down what MLIR said, and the span is worked out afterwards.
struct Reported {
    message: String,
    position: Option<Position>,
    severity: DiagnosticSeverity,
    notes: Vec<String>,
}

/// Where a location says its diagnostic belongs. Read inside the handler,
/// because a location borrows the context and the handler outlives the call.
struct Position {
    file: String,
    start_line: usize,
    start_column: usize,
    end_line: usize,
    end_column: usize,
}

impl Position {
    /// `None` for a location naming no place: an unknown one, or a fused one
    /// standing for several.
    fn of(location: &Location) -> Option<Self> {
        location.is_file_line_col_range().then(|| Self {
            file: location
                .file_line_col_range_filename()
                .as_string_ref()
                .as_str()
                .unwrap_or_default()
                .to_string(),
            start_line: location.file_line_col_range_start_line(),
            start_column: location.file_line_col_range_start_column(),
            end_line: location.file_line_col_range_end_line(),
            end_column: location.file_line_col_range_end_column(),
        })
    }
}

impl Reported {
    fn of(diagnostic: &Diagnostic) -> Self {
        Self {
            message: diagnostic.to_string(),
            position: Position::of(&diagnostic.location()),
            severity: diagnostic.severity(),
            notes: (0..diagnostic.note_count())
                .filter_map(|index| diagnostic.note(index).ok())
                .map(|note| note.to_string())
                .collect(),
        }
    }

    fn build(self, sources: &SourceMap) -> DiagnosticBuilder {
        let mut builder = match (span_of(sources, self.position.as_ref()), self.severity) {
            (Some(span), DiagnosticSeverity::Error) => DiagnosticBuilder::error(span, self.message),
            (Some(span), DiagnosticSeverity::Warning) => {
                DiagnosticBuilder::warning(span, self.message)
            }
            (Some(span), DiagnosticSeverity::Note | DiagnosticSeverity::Remark) => {
                DiagnosticBuilder::remark(span, self.message)
            }
            (None, _) => DiagnosticBuilder::error_without_span(self.message),
        };

        for note in self.notes {
            builder = builder.note(note);
        }

        builder
    }
}

/// The span a location names in `sources`: what an op's location says it
/// was lowered from. `None` for a location that is no file range, or that
/// names a file no source was added under.
#[must_use]
pub fn span(sources: &SourceMap, location: Location) -> Option<Span> {
    span_of(sources, Position::of(&location).as_ref())
}

/// The span a position names.
///
/// `None` when it names no place we can point at: no position at all, or a
/// file no source was added under. Such a diagnostic prints as its message
/// alone. Pointing it at some other file would send the reader somewhere
/// the problem is not.
fn span_of(sources: &SourceMap, position: Option<&Position>) -> Option<Span> {
    let position = position?;
    let source_id = sources.id(&position.file)?;
    let start = sources.offset(source_id, position.start_line, position.start_column)?;
    let end = sources.offset(source_id, position.end_line, position.end_column)?;
    Some(Span {
        source_id,
        range: TextRange::new(
            TextSize::new(start as u32),
            TextSize::new(end.max(start) as u32),
        ),
    })
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
        sources.add("test.yz".to_string(), source.to_string());
        let mut diagnostics = DiagnosticsEngine::new();

        capture(&context, &sources, &mut diagnostics, || {
            let location = Location::new(&context, "test.yz", 1, 6);
            emit_error(location, "unknown relation `t`");
        });

        let printer = DiagnosticPrinter::new(&sources);
        let rendered: Vec<String> = diagnostics
            .diagnostics()
            .iter()
            .map(|diagnostic| printer.print(diagnostic))
            .collect();
        expect![[r"
        error: unknown relation `t`
         --> test.yz:1:6
          |
        1 | from t
          |      ^
    "]]
        .assert_eq(&rendered.join("\n"));
    }

    /// A location names its file, so a compile that read more than one
    /// puts each diagnostic in the file it came from. Before this the name
    /// was parsed out and dropped, and every span landed in one source.
    #[test]
    fn each_diagnostic_lands_in_the_file_it_names() {
        let context = crate::context();
        let mut sources = SourceMap::new();
        sources.add(
            "prelude.yz".to_string(),
            "external def pow(a: int64, b: int64) -> int64\n".to_string(),
        );
        sources.add("query.yz".to_string(), "from t\n".to_string());
        let mut diagnostics = DiagnosticsEngine::new();

        capture(&context, &sources, &mut diagnostics, || {
            emit_error(
                Location::new(&context, "query.yz", 1, 6),
                "unknown relation `t`",
            );
            emit_error(
                Location::new(&context, "prelude.yz", 1, 13),
                "`pow` is declared twice",
            );
            // A location naming no place prints as its message alone, rather
            // than pointing at a file the problem is not in.
            emit_error(Location::unknown(&context), "the query could not be built");
        });

        let printer = DiagnosticPrinter::new(&sources);
        let rendered: Vec<String> = diagnostics
            .diagnostics()
            .iter()
            .map(|diagnostic| printer.print(diagnostic))
            .collect();
        expect![[r"
            error: unknown relation `t`
             --> query.yz:1:6
              |
            1 | from t
              |      ^

            error: `pow` is declared twice
             --> prelude.yz:1:13
              |
            1 | external def pow(a: int64, b: int64) -> int64
              |             ^

            error: the query could not be built
        "]]
        .assert_eq(&rendered.join("\n"));
    }

    #[test]
    fn verifier_failures_land_in_the_engine() {
        use melior::ir::operation::{OperationBuilder, OperationLike};
        use melior::ir::{BlockLike, Module, Type};

        let context = crate::context();
        let source = "from t\n";
        let mut sources = SourceMap::new();
        sources.add("test.yz".to_string(), source.to_string());
        let mut diagnostics = DiagnosticsEngine::new();

        // A yz.add whose result is not a numeric type — the ODS verifier rejects
        // it, and capture turns that into an engine diagnostic.
        let location = Location::new(&context, "test.yz", 1, 1);
        let module = Module::new(location);
        let boolean = Type::parse(&context, "!yz.bool").expect("!yz.bool parses");
        let verified = capture(&context, &sources, &mut diagnostics, || {
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
