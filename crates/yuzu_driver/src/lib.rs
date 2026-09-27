pub mod modules;
pub mod stdlib;

use yuzu_diagnostics::{
    diagnostics::{Severity, engine::DiagnosticsEngine, printer::DiagnosticPrinter},
    source_map::SourceMap,
};

#[derive(Default)]
pub struct CompileOptions {
    /// Print the module as the frontend lowered it to yzl.
    pub debug_yzl: bool,
    /// Print the module once it is lowered to yzr and simplified.
    pub debug_yzr: bool,
    pub debug_substrait: bool,
    pub target: Option<String>,
}

/// Compiles a file, printing what each stage asked for, and says whether
/// the program compiled.
pub fn compile(
    name: &str,
    source: &str,
    options: &CompileOptions,
    resolver: &dyn modules::ModuleResolver,
) -> std::process::ExitCode {
    let mut diagnostics = DiagnosticsEngine::new();
    let mut sources = SourceMap::new();
    let source_id = sources.add(name.to_string(), source.to_string());

    let plan = plan_through_mlir(&mut sources, source_id, &mut diagnostics, options, resolver);
    print_diagnostics(&diagnostics, &sources);
    match plan {
        Some(plan) => {
            if options.debug_substrait {
                println!("=== substrait ===");
                println!("{}", yuzu_substrait::to_json(&plan));
            }

            std::process::ExitCode::SUCCESS
        }
        None => std::process::ExitCode::FAILURE,
    }
}

/// Compiles a file to a plan, as protobuf bytes.
pub fn compile_to_substrait(
    name: &str,
    source: &str,
    options: &CompileOptions,
    resolver: &dyn modules::ModuleResolver,
) -> Result<Vec<u8>, String> {
    let mut diagnostics = DiagnosticsEngine::new();
    let mut sources = SourceMap::new();
    let source_id = sources.add(name.to_string(), source.to_string());

    match plan_through_mlir(&mut sources, source_id, &mut diagnostics, options, resolver) {
        Some(plan) => Ok(yuzu_substrait::to_protobuf(&plan)),
        None => Err(render_diagnostics(&diagnostics, &sources)),
    }
}

/// Source to plan, through every MLIR pass in order. `None` once anything
/// has reported: a pass reads what the one before it settled, so running on
/// after an error would report the same mistake again in other words.
///
/// The query is one of the sources rather than a text of its own, so its
/// name, its text and the id a diagnostic carries cannot disagree.
fn plan_through_mlir(
    sources: &mut SourceMap,
    source_id: yuzu_diagnostics::source_map::SourceId,
    diagnostics: &mut DiagnosticsEngine,
    options: &CompileOptions,
    resolver: &dyn modules::ModuleResolver,
) -> Option<yuzu_substrait::Plan> {
    use melior::ir::operation::OperationLike;

    let engine = read_engine(options, diagnostics, source_id)?;
    let files = modules::load(source_id, sources, diagnostics, resolver, engine)?;
    with_context(|context| {
        let mut module = yuzu_passes::lower_ast_to_yzl(
            context,
            sources,
            &files,
            diagnostics,
            &yuzu_types::Builtins,
            Some(stdlib::bound_library(engine)),
        );

        if options.debug_yzl {
            println!("=== yzl ===");
            print!("{}", module.as_operation());
        }

        let verified = yuzu_mlir::diagnostics::capture(context, sources, diagnostics, || {
            module.as_operation().verify()
        });
        if !verified || has_errors(diagnostics) {
            return None;
        }

        yuzu_mlir::diagnostics::capture(context, sources, diagnostics, || {
            yuzu_passes::check_mutability(&module);
            yuzu_passes::promote_locals(context, &mut module);
            yuzu_passes::infer_types(context, &mut module, &yuzu_types::Builtins);
            yuzu_passes::check_aggregates(&module);
        });
        if has_errors(diagnostics) {
            return None;
        }

        // Expansion runs after the aggregate rules, which read an `agg fn` body
        // while it is still a body, and before the lowering, which has no way to
        // carry a function across.
        yuzu_mlir::diagnostics::capture(context, sources, diagnostics, || {
            yuzu_passes::inline_calls(context, &mut module);
            yuzu_passes::remove_dead_symbols(context, &mut module);
        });
        if has_errors(diagnostics) {
            return None;
        }

        yuzu_mlir::diagnostics::capture(context, sources, diagnostics, || {
            yuzu_passes::lower_yzl_to_yzr(context, &mut module);
            yuzu_passes::simplify_yzr(context, &mut module);
            if options.debug_yzr {
                println!("=== yzr ===");
                print!("{}", module.as_operation());
            }

            yuzu_substrait::translate(context, &module)
        })
        .filter(|_| !has_errors(diagnostics))
    })
}

/// How many compiles one thread's context serves. A context keeps every
/// attribute it has uniqued until it is dropped, so a long-lived process
/// replaces it now and then.
const CONTEXT_COMPILES: usize = 1000;

/// The context this thread compiles in.
struct Reused {
    context: melior::Context,
    compiles: usize,
}

thread_local! {
    static CONTEXT: std::cell::RefCell<Reused> = std::cell::RefCell::new(Reused {
        context: yuzu_mlir::context(),
        compiles: 0,
    });
}

/// Runs a compile in this thread's context. Making a context registers every
/// op of the dialects, which was a sixth of a compile.
fn with_context<T>(compile: impl FnOnce(&melior::Context) -> T) -> T {
    CONTEXT.with(|reused| {
        let mut reused = reused.borrow_mut();
        if reused.compiles == CONTEXT_COMPILES {
            *reused = Reused {
                context: yuzu_mlir::context(),
                compiles: 0,
            };
        }

        reused.compiles += 1;
        compile(&reused.context)
    })
}

/// The engine `--target` names; DataFusion when it names none.
fn read_engine(
    options: &CompileOptions,
    diagnostics: &mut DiagnosticsEngine,
    source_id: yuzu_diagnostics::source_map::SourceId,
) -> Option<stdlib::Engine> {
    let Some(name) = &options.target else {
        return Some(stdlib::Engine::DataFusion);
    };

    let engine = stdlib::Engine::from_name(name);
    if engine.is_none() {
        let span = yuzu_diagnostics::diagnostics::Span {
            source_id,
            range: Default::default(),
        };
        diagnostics.emit(
            yuzu_diagnostics::diagnostics::builder::DiagnosticBuilder::error(
                span,
                format!("`{name}` is not a supported engine; the supported engine is `datafusion`"),
            ),
        );
    }

    engine
}

fn render_diagnostics(diagnostics: &DiagnosticsEngine, sources: &SourceMap) -> String {
    let printer = DiagnosticPrinter::new(sources);
    diagnostics
        .diagnostics()
        .iter()
        .map(|diagnostic| printer.print(diagnostic))
        .collect::<Vec<_>>()
        .join("\n")
}

fn has_errors(diagnostics: &DiagnosticsEngine) -> bool {
    diagnostics
        .diagnostics()
        .iter()
        .any(|diagnostic| matches!(diagnostic.severity, Severity::Error))
}

fn print_diagnostics(diagnostics: &DiagnosticsEngine, sources: &SourceMap) {
    let printer = DiagnosticPrinter::new(sources);
    for diagnostic in diagnostics.diagnostics() {
        eprintln!("{}", printer.print(diagnostic));
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{CompileOptions, compile_to_substrait, modules};

    #[test]
    fn an_engine_the_compiler_does_not_know_is_reported() {
        let options = CompileOptions {
            target: Some("postgres".to_string()),
            ..CompileOptions::default()
        };
        let resolver = modules::MapResolver(HashMap::new());
        let error = compile_to_substrait("test.yz", "", &options, &resolver)
            .expect_err("postgres is not an engine");
        assert!(
            error.contains(
                "`postgres` is not a supported engine; the supported engine is `datafusion`"
            ),
            "{error}"
        );
    }
}
