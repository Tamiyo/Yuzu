mod check;
pub mod index;
pub mod modules;
pub mod stdlib;

pub use check::{Checked, Focus, check};

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
    let engine = read_engine(options, diagnostics, source_id)?;
    let files = modules::load(source_id, sources, diagnostics, resolver, engine)?;
    in_thread_context(|context| {
        let mut module = lower_and_check(
            context,
            sources,
            &files,
            diagnostics,
            Some(stdlib::bound_library(engine)),
            options,
            None,
        )
        .filter(|_| !has_errors(diagnostics))?;

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
            yuzu_passes::legalize_operators(context, &mut module);
            if options.debug_yzr {
                println!("=== yzr ===");
                print!("{}", module.as_operation());
            }

            yuzu_substrait::translate(context, &module)
        })
        .filter(|_| !has_errors(diagnostics))
    })
}

/// The frontend and the checks after it, through `check_aggregates`: what
/// a compile and an editor's check share. Each check runs even after an
/// error, and passes over what the error left behind; `None` only when the
/// module does not verify, which no pass can read.
///
/// `library` is the library's names bound ahead of time; without it, the
/// lowering binds them from the library files among `files`. `index` reads
/// the module after lowering and after inference, when a caller wants it.
pub(crate) fn lower_and_check<'c>(
    context: &'c melior::Context,
    sources: &SourceMap,
    files: &[yuzu_passes::File],
    diagnostics: &mut DiagnosticsEngine,
    library: Option<&'c yuzu_passes::BoundLibrary<'c>>,
    options: &CompileOptions,
    mut index: Option<&mut index::IndexReader<'_>>,
) -> Option<melior::ir::Module<'c>> {
    use melior::ir::operation::OperationLike;

    let mut module = yuzu_passes::lower_ast_to_yzl(
        context,
        sources,
        files,
        diagnostics,
        &yuzu_types::Builtins,
        library,
    );

    if options.debug_yzl {
        println!("=== yzl ===");
        print!("{}", module.as_operation());
    }

    let verified = yuzu_mlir::diagnostics::capture(context, sources, diagnostics, || {
        module.as_operation().verify()
    });
    if !verified {
        return None;
    }
    if let Some(reader) = index.as_deref_mut() {
        reader.read_lowered(&module);
    }

    yuzu_mlir::diagnostics::capture(context, sources, diagnostics, || {
        yuzu_passes::check_mutability(&module);
        yuzu_passes::promote_locals(context, &mut module);
        yuzu_passes::infer_types(context, &mut module, &yuzu_types::Builtins);
    });
    if let Some(reader) = index {
        reader.read_inferred(context, &module);
    }

    yuzu_mlir::diagnostics::capture(context, sources, diagnostics, || {
        yuzu_passes::check_aggregates(&module);
    });
    Some(module)
}

/// How many compiles one thread's context serves. A context keeps every
/// attribute it has uniqued until it is dropped, so a long-lived process
/// replaces it now and then.
const CONTEXT_COMPILES: usize = 1000;

/// The context this thread compiles in, and how many compiles it has served.
struct ThreadContext {
    context: melior::Context,
    compiles: usize,
}

thread_local! {
    static CONTEXT: std::cell::RefCell<ThreadContext> = std::cell::RefCell::new(ThreadContext {
        context: yuzu_mlir::context(),
        compiles: 0,
    });
}

/// Runs a compile in this thread's context. Making a context registers every
/// op of the dialects, which was a sixth of a compile.
fn in_thread_context<T>(compile: impl FnOnce(&melior::Context) -> T) -> T {
    CONTEXT.with(|thread| {
        let mut thread = thread.borrow_mut();
        if thread.compiles == CONTEXT_COMPILES {
            *thread = ThreadContext {
                context: yuzu_mlir::context(),
                compiles: 0,
            };
        }

        thread.compiles += 1;
        compile(&thread.context)
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
