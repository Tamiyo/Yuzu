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
use yuzu_diagnostics::source_map::{SourceId, SourceMap};

/// Emits an error against a location, into whichever handler is attached.
pub fn emit_error(location: Location, message: &str) {
    let message = CString::new(message).expect("diagnostic messages have no interior nul");
    unsafe { mlir_sys::mlirEmitError(location.to_raw(), message.as_ptr()) }
}

/// Runs `f` with MLIR diagnostics routed into the engine.
///
/// A location names the file it came from, so every source the compile read
/// is offered here and each diagnostic lands in the one it belongs to.
/// `unnamed` is where a location naming nothing goes, which is the file the
/// user asked about.
pub fn capture<T>(
    context: &Context,
    sources: &SourceMap,
    unnamed: SourceId,
    engine: &mut DiagnosticsEngine,
    f: impl FnOnce() -> T,
) -> T {
    let collected = Rc::new(RefCell::new(Vec::new()));
    let sink = collected.clone();
    let handler = context.attach_diagnostic_handler(move |diagnostic| {
        sink.borrow_mut().push(Reported::of(&diagnostic));
        true
    });

    let result = f();
    context.detach_diagnostic_handler(handler);
    for reported in collected.take() {
        engine.emit(reported.build(sources, unnamed));
    }

    result
}

/// What a handler saw. The handler has to outlive this call as far as the
/// type system is concerned, so it cannot borrow the sources a span needs.
/// It writes down what MLIR said, and the span is worked out afterwards.
struct Reported {
    message: String,
    location: String,
    severity: DiagnosticSeverity,
    notes: Vec<String>,
}

impl Reported {
    fn of(diagnostic: &Diagnostic) -> Self {
        Self {
            message: diagnostic.to_string(),
            location: diagnostic.location().to_string(),
            severity: diagnostic.severity(),
            notes: (0..diagnostic.note_count())
                .filter_map(|index| diagnostic.note(index).ok())
                .map(|note| note.to_string())
                .collect(),
        }
    }

    fn build(self, sources: &SourceMap, unnamed: SourceId) -> DiagnosticBuilder {
        let span = span_of(sources, unnamed, &self.location);
        let mut builder = match self.severity {
            DiagnosticSeverity::Error => DiagnosticBuilder::error(span, self.message),
            DiagnosticSeverity::Warning => DiagnosticBuilder::warning(span, self.message),
            DiagnosticSeverity::Note | DiagnosticSeverity::Remark => {
                DiagnosticBuilder::remark(span, self.message)
            }
        };

        for note in self.notes {
            builder = builder.note(note);
        }

        builder
    }
}

/// The span a location names. Locations print as `loc("name":line:col)`,
/// where the name is the one the file was read under. Anything else — an
/// unknown location, a fused one, or a name no source was added under —
/// lands at the start of the file the user asked about, which is where a
/// reader looks first.
fn span_of(sources: &SourceMap, unnamed: SourceId, printed: &str) -> Span {
    let Some((name, line, column)) = position(printed) else {
        return start_of(sources, unnamed);
    };

    let Some(id) = sources.id(name) else {
        return start_of(sources, unnamed);
    };

    match sources.offset(id, line, column) {
        Some(offset) => one_character(sources, id, offset),
        None => start_of(sources, id),
    }
}

/// The name, line and column a printed location carries. The name is taken
/// between the first quote and the last one before the numbers, so a path
/// holding a colon of its own survives the split.
fn position(printed: &str) -> Option<(&str, usize, usize)> {
    let (_, quoted) = printed.split_once('"')?;
    let (name, tail) = quoted.rsplit_once("\":")?;
    let (line, column) = tail.strip_suffix(')')?.split_once(':')?;
    Some((name, line.parse().ok()?, column.parse().ok()?))
}

/// One character from the offset: the printer underlines a span, and a
/// location is a point rather than a range.
fn one_character(sources: &SourceMap, source_id: SourceId, offset: usize) -> Span {
    let end = (offset + 1).min(sources.text(source_id).len());
    Span {
        source_id,
        range: TextRange::new(TextSize::new(offset as u32), TextSize::new(end as u32)),
    }
}

fn start_of(sources: &SourceMap, source_id: SourceId) -> Span {
    one_character(sources, source_id, 0)
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

        capture(&context, &sources, source_id, &mut diagnostics, || {
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

    /// A location names its file, so a compile that read more than one
    /// puts each diagnostic in the file it came from. Before this the name
    /// was parsed out and dropped, and every span landed in one source.
    #[test]
    fn each_diagnostic_lands_in_the_file_it_names() {
        let context = crate::context();
        let mut sources = SourceMap::new();
        sources.add(
            "prelude.yz".to_string(),
            "external fn pow(a: int64, b: int64) -> int64\n".to_string(),
        );
        let query = sources.add("query.yz".to_string(), "from t\n".to_string());
        let mut diagnostics = DiagnosticsEngine::new();

        capture(&context, &sources, query, &mut diagnostics, || {
            emit_error(
                Location::new(&context, "query.yz", 1, 6),
                "unknown relation `t`",
            );
            emit_error(
                Location::new(&context, "prelude.yz", 1, 13),
                "`pow` is declared twice",
            );
            // A location naming no source of ours lands in the file the user
            // asked about rather than nowhere.
            emit_error(Location::unknown(&context), "the query could not be built");
        });

        let printer = DiagnosticPrinter::new(&sources);
        let rendered: Vec<String> = diagnostics
            .diagnostics()
            .iter()
            .map(|diagnostic| printer.print(diagnostic))
            .collect();
        expect![[r#"
            error: unknown relation `t`
             --> query.yz:1:6
              |
            1 | from t
              |      ^

            error: `pow` is declared twice
             --> prelude.yz:1:13
              |
            1 | external fn pow(a: int64, b: int64) -> int64
              |             ^

            error: the query could not be built
             --> query.yz:1:1
              |
            1 | from t
              | ^
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
        let verified = capture(&context, &sources, source_id, &mut diagnostics, || {
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
