use expect_test::Expect;
use melior::Context;
use melior::ir::Module;
use melior::ir::operation::OperationLike;
use yuzu_ast::AstNode;
use yuzu_ast::ast;
use yuzu_diagnostics::diagnostics::engine::DiagnosticsEngine;
use yuzu_diagnostics::diagnostics::printer::DiagnosticPrinter;
use yuzu_diagnostics::source_map::{SourceId, SourceMap};

use crate::{File, Lowering};

/// The aggregates the library's prelude brings into every file, declared
/// as the engine's own, since a test compiles without the library.
const PRELUDE: &str = include_str!("../tests/prelude.yz");

/// The files of a program as `(name, module, source)`, the entry last.
pub(crate) type Program<'s> = [(&'s str, Option<&'s str>, &'s str)];

pub(crate) struct Lowered<'c> {
    pub(crate) module: Module<'c>,
    pub(crate) sources: SourceMap,
    pub(crate) diagnostics: DiagnosticsEngine,
}

pub(crate) fn parsed(
    sources: &SourceMap,
    source_id: SourceId,
    diagnostics: &mut DiagnosticsEngine,
) -> ast::Root {
    let syntax = yuzu_parser::parse_text(sources.text(source_id), diagnostics, source_id);
    ast::Root::cast(syntax).expect("a parse always yields a root")
}

pub(crate) fn lower<'c>(context: &'c Context, program: &Program) -> Lowered<'c> {
    let mut sources = SourceMap::new();
    let mut diagnostics = DiagnosticsEngine::new();
    let prelude_id = sources.add("<prelude>".to_string(), PRELUDE.to_string());
    let mut prelude = File::new(
        prelude_id,
        Some(crate::PRELUDE.to_string()),
        parsed(&sources, prelude_id, &mut diagnostics),
    );
    prelude.set_lowering(Lowering::OnDemand);
    let files: Vec<File> = std::iter::once(prelude)
        .chain(program.iter().map(|&(name, module, source)| {
            let source_id = sources.add(name.to_string(), source.to_string());
            let root = parsed(&sources, source_id, &mut diagnostics);
            File::new(source_id, module.map(str::to_string), root)
        }))
        .collect();
    let module = crate::lower_ast_to_yzl(context, &sources, &files, &mut diagnostics, None);

    Lowered {
        module,
        sources,
        diagnostics,
    }
}

pub(crate) fn rendered(sources: &SourceMap, diagnostics: &DiagnosticsEngine) -> String {
    let printer = DiagnosticPrinter::new(sources);
    diagnostics
        .diagnostics()
        .iter()
        .map(|diagnostic| printer.print(diagnostic))
        .collect::<Vec<_>>()
        .join("\n")
}

pub(crate) fn lowered_program(program: &Program) -> String {
    let context = yuzu_mlir::context();
    let lowered = lower(&context, program);
    let reported = rendered(&lowered.sources, &lowered.diagnostics);
    assert!(reported.is_empty(), "lowering reported:\n{reported}");
    assert!(
        lowered.module.as_operation().verify(),
        "the lowered module verifies"
    );

    lowered.module.as_operation().to_string()
}

pub(crate) fn lowered(source: &str) -> String {
    lowered_program(&[("test.yz", None, source)])
}

pub(crate) fn reported_program(program: &Program) -> String {
    let context = yuzu_mlir::context();
    let lowered = lower(&context, program);
    rendered(&lowered.sources, &lowered.diagnostics)
}

pub(crate) fn reported(source: &str) -> String {
    reported_program(&[("test.yz", None, source)])
}

pub(crate) fn check(
    source: &str,
    passes: impl for<'c> FnOnce(&'c Context, &mut Module<'c>) -> String,
    expected: &Expect,
) {
    let context = yuzu_mlir::context();
    let Lowered {
        mut module,
        sources,
        mut diagnostics,
        ..
    } = lower(&context, &[("test.yz", None, source)]);
    let output = yuzu_mlir::diagnostics::capture(&context, &sources, &mut diagnostics, || {
        passes(&context, &mut module)
    });
    let reported = rendered(&sources, &diagnostics);
    expected.assert_eq(if reported.is_empty() {
        &output
    } else {
        &reported
    });
}

pub(crate) fn check_yzr(source: &str, expected: &Expect) {
    check(
        source,
        |context, module| {
            crate::promote_locals(context, module);
            crate::infer_types(context, module);
            crate::lower_yzl_to_yzr(context, module);
            module.as_operation().to_string()
        },
        expected,
    );
}

pub(crate) fn check_simplified(source: &str, expected: &Expect) {
    check(
        source,
        |context, module| {
            crate::promote_locals(context, module);
            crate::infer_types(context, module);
            crate::inline_calls(context, module);
            crate::remove_dead_symbols(context, module);
            crate::lower_yzl_to_yzr(context, module);
            crate::simplify_yzr(context, module);
            module.as_operation().to_string()
        },
        expected,
    );
}
