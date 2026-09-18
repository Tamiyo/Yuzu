//! The conversions and checks that carry a yzl module toward yzr.

mod check_aggregates;
mod infer_types;
mod inline_calls;
mod lower_ast_to_yzl;
mod lower_yzl_to_yzr;
mod simplify_yzr;

pub use check_aggregates::check_aggregates;
pub use infer_types::infer_types;
pub use inline_calls::inline_calls;
pub use lower_ast_to_yzl::lower_ast_to_yzl;
pub use lower_yzl_to_yzr::lower_yzl_to_yzr;
pub use simplify_yzr::simplify_yzr;

#[cfg(test)]
pub(crate) mod test_support {
    use expect_test::Expect;
    use melior::Context;
    use melior::ir::Module;
    use yuzu_diagnostics::diagnostics::engine::DiagnosticsEngine;
    use yuzu_diagnostics::diagnostics::printer::DiagnosticPrinter;
    use yuzu_diagnostics::source_map::SourceMap;

    /// Runs the passes over converted source and renders the module they
    /// stamped, or the diagnostics when the source did not get that far.
    pub(crate) fn check(
        source: &str,
        passes: impl FnOnce(&Context, &Module<'_>),
        expected: Expect,
    ) {
        let (module, rendered) = run(source, passes);
        let output = if rendered.is_empty() {
            module
        } else {
            rendered
        };

        expected.assert_eq(&output);
    }

    /// Renders what the lowering produced, or the diagnostics that stopped
    /// it — the pass returns a new module rather than stamping this one.
    pub(crate) fn check_lowered(source: &str, expected: Expect) {
        let context = yuzu_mlir::context();
        let mut sources = SourceMap::new();
        let source_id = sources.add("test.yz".to_string(), source.to_string());
        let mut diagnostics = DiagnosticsEngine::new();
        let module = crate::lower_ast_to_yzl(
            &context,
            &sources,
            source_id,
            &mut diagnostics,
            &yuzu_types::Builtins,
        )
        .expect("the source converts");

        let lowered = yuzu_mlir::diagnostics::capture(
            &context,
            &sources,
            source_id,
            &mut diagnostics,
            || {
                crate::infer_types(&context, &module);
                crate::lower_yzl_to_yzr(&context, &module)
            },
        );

        let printer = DiagnosticPrinter::new(&sources);
        let rendered: Vec<String> = diagnostics
            .diagnostics()
            .iter()
            .map(|diagnostic| printer.print(diagnostic))
            .collect();
        let output = if rendered.is_empty() {
            lowered.as_operation().to_string()
        } else {
            rendered.join("\n")
        };

        expected.assert_eq(&output);
    }

    /// Renders the lowered query after simplification — what the emitter
    /// would actually be handed.
    pub(crate) fn check_simplified(source: &str, expected: Expect) {
        let context = yuzu_mlir::context();
        let mut sources = SourceMap::new();
        let source_id = sources.add("test.yz".to_string(), source.to_string());
        let mut diagnostics = DiagnosticsEngine::new();
        let module = crate::lower_ast_to_yzl(
            &context,
            &sources,
            source_id,
            &mut diagnostics,
            &yuzu_types::Builtins,
        )
        .expect("the source converts");

        let lowered = yuzu_mlir::diagnostics::capture(
            &context,
            &sources,
            source_id,
            &mut diagnostics,
            || {
                crate::infer_types(&context, &module);
                crate::inline_calls(&context, &module);
                let mut lowered = crate::lower_yzl_to_yzr(&context, &module);
                crate::simplify_yzr(&context, &mut lowered);
                lowered
            },
        );

        let printer = DiagnosticPrinter::new(&sources);
        let rendered: Vec<String> = diagnostics
            .diagnostics()
            .iter()
            .map(|diagnostic| printer.print(diagnostic))
            .collect();
        let output = if rendered.is_empty() {
            lowered.as_operation().to_string()
        } else {
            rendered.join("\n")
        };

        expected.assert_eq(&output);
    }

    /// Runs the passes and renders only what they reported — for the checks
    /// that stamp nothing.
    pub(crate) fn check_diagnostics(
        source: &str,
        passes: impl FnOnce(&Context, &Module<'_>),
        expected: Expect,
    ) {
        let (_, rendered) = run(source, passes);
        let output = if rendered.is_empty() {
            String::from("no diagnostics")
        } else {
            rendered
        };

        expected.assert_eq(&output);
    }

    /// Converts the source, runs the passes under diagnostic capture, and
    /// hands back the module's text and the rendered diagnostics.
    fn run(source: &str, passes: impl FnOnce(&Context, &Module<'_>)) -> (String, String) {
        let context = yuzu_mlir::context();
        let mut sources = SourceMap::new();
        let source_id = sources.add("test.yz".to_string(), source.to_string());
        let mut diagnostics = DiagnosticsEngine::new();
        let module = crate::lower_ast_to_yzl(
            &context,
            &sources,
            source_id,
            &mut diagnostics,
            &yuzu_types::Builtins,
        )
        .expect("the source converts");

        yuzu_mlir::diagnostics::capture(&context, &sources, source_id, &mut diagnostics, || {
            passes(&context, &module);
        });

        let printer = DiagnosticPrinter::new(&sources);
        let rendered: Vec<String> = diagnostics
            .diagnostics()
            .iter()
            .map(|diagnostic| printer.print(diagnostic))
            .collect();

        (module.as_operation().to_string(), rendered.join("\n"))
    }
}
