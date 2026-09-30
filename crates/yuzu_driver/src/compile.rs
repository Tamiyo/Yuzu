//! A compile: source to Substrait plan, through every MLIR pass in order.

use std::fmt;

use melior::ir::operation::OperationLike;
use yuzu_diagnostics::{Diagnostic, DiagnosticPrinter, DiagnosticsEngine, SourceId, SourceMap};
use yuzu_substrait::Plan;

use crate::index::IndexReader;
use crate::modules::{self, ModuleResolver, Origin};
use crate::stdlib::{self, Engine};

/// What a compile is for, and which intermediate modules it keeps as text.
#[derive(Clone, Copy, Debug, Default)]
pub struct CompileOptions {
    /// The engine the plan is for.
    pub engine: Engine,
    /// Keep the module as the frontend lowered it to yzl.
    pub dump_yzl: bool,
    /// Keep the module once it is lowered to yzr and simplified.
    pub dump_yzr: bool,
}

/// What a compile produced: the plan when the program compiled, and all it
/// reported on the way.
#[derive(Debug)]
pub struct Compilation {
    /// The plan, when the program compiled.
    pub plan: Option<Plan>,
    /// All the compile reported.
    pub diagnostics: Vec<Diagnostic>,
    /// The sources the diagnostics point into.
    pub sources: SourceMap,
    /// The module as the frontend lowered it, when the options asked.
    pub yzl: Option<String>,
    /// The module lowered to yzr, when the options asked and it got there.
    pub yzr: Option<String>,
}

impl Compilation {
    /// The plan, or the error that says why there is none.
    ///
    /// # Errors
    ///
    /// When the program did not compile.
    pub fn into_plan(self) -> Result<Plan, CompileError> {
        match self.plan {
            Some(plan) => Ok(plan),
            None => Err(CompileError {
                sources: self.sources,
                diagnostics: self.diagnostics,
            }),
        }
    }
}

/// A program that did not compile, and the diagnostics that say why.
#[derive(Debug)]
pub struct CompileError {
    sources: SourceMap,
    diagnostics: Vec<Diagnostic>,
}

impl CompileError {
    /// What the compile reported.
    #[must_use]
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    /// The sources the diagnostics point into.
    #[must_use]
    pub fn sources(&self) -> &SourceMap {
        &self.sources
    }
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&DiagnosticPrinter::new(&self.sources).render_all(&self.diagnostics))
    }
}

impl std::error::Error for CompileError {}

/// Compiles the entry `source`, which came from `origin`.
pub fn compile(
    origin: &Origin,
    source: &str,
    options: &CompileOptions,
    resolver: &dyn ModuleResolver,
) -> Compilation {
    let mut diagnostics = DiagnosticsEngine::new();
    let mut sources = SourceMap::new();
    let source_id = origin.add_to(&mut sources, source.into());
    let mut dumps = Dumps {
        keep_yzl: options.dump_yzl,
        keep_yzr: options.dump_yzr,
        yzl: None,
        yzr: None,
    };

    let plan = plan_through_mlir(
        &mut sources,
        source_id,
        &mut diagnostics,
        options.engine,
        resolver,
        &mut dumps,
    );
    Compilation {
        plan,
        diagnostics: diagnostics.into_diagnostics(),
        sources,
        yzl: dumps.yzl,
        yzr: dumps.yzr,
    }
}

/// The modules a compile keeps as text: those the options asked for, once
/// the compile reaches them.
struct Dumps {
    keep_yzl: bool,
    keep_yzr: bool,
    yzl: Option<String>,
    yzr: Option<String>,
}

/// Source to plan, through every MLIR pass in order. `None` once an error
/// is reported: a pass reads what the one before it settled, so running on
/// after an error would report the same mistake again in other words.
///
/// The query is one of the sources rather than a text of its own, so its
/// name, its text and the id a diagnostic carries cannot disagree.
fn plan_through_mlir(
    sources: &mut SourceMap,
    source_id: SourceId,
    diagnostics: &mut DiagnosticsEngine,
    engine: Engine,
    resolver: &dyn ModuleResolver,
    dumps: &mut Dumps,
) -> Option<Plan> {
    let files = modules::load(source_id, sources, diagnostics, resolver, engine)?;
    in_thread_context(|context| {
        let mut module = lower_and_check(
            context,
            sources,
            &files,
            diagnostics,
            Some(stdlib::bound_library(engine)),
            dumps.keep_yzl.then_some(&mut dumps.yzl),
            None,
        )
        .filter(|_| !diagnostics.has_errors())?;

        // Expansion runs after the aggregate rules, which read an `agg fn` body
        // while it is still a body, and before the lowering, which has no way to
        // carry a function across.
        yuzu_mlir::diagnostics::capture(context, sources, diagnostics, || {
            yuzu_passes::inline_calls(context, &mut module);
            yuzu_passes::remove_dead_symbols(context, &mut module);
        });
        if diagnostics.has_errors() {
            return None;
        }

        yuzu_mlir::diagnostics::capture(context, sources, diagnostics, || {
            yuzu_passes::lower_yzl_to_yzr(context, &mut module);
            yuzu_passes::simplify_yzr(context, &mut module);
            yuzu_passes::legalize_operators(context, &mut module);
            if dumps.keep_yzr {
                dumps.yzr = Some(module.as_operation().to_string());
            }
        });
        if diagnostics.has_errors() {
            return None;
        }

        yuzu_mlir::diagnostics::capture(context, sources, diagnostics, || {
            yuzu_substrait::translate(context, &module)
        })
        .filter(|_| !diagnostics.has_errors())
    })
}

/// The frontend and the checks after it, through `check_aggregates`: what
/// a compile and an editor's check share. Each check runs even after an
/// error, and passes over what the error left behind; `None` only when the
/// module does not verify, which no pass can read.
///
/// `library` is the library's names bound ahead of time; without it, the
/// lowering binds them from the library files among `files`. `yzl` takes the
/// lowered module as text, and `index` reads the module after lowering and
/// after inference, when a caller wants them.
pub(crate) fn lower_and_check<'c>(
    context: &'c melior::Context,
    sources: &SourceMap,
    files: &[yuzu_passes::File],
    diagnostics: &mut DiagnosticsEngine,
    library: Option<&'c yuzu_passes::BoundLibrary<'c>>,
    yzl: Option<&mut Option<String>>,
    mut index: Option<&mut IndexReader<'_>>,
) -> Option<melior::ir::Module<'c>> {
    let mut module = match index.as_deref_mut() {
        Some(reader) => yuzu_passes::lower_ast_to_yzl_with_listener(
            context,
            sources,
            files,
            diagnostics,
            library,
            reader,
        ),
        None => yuzu_passes::lower_ast_to_yzl(context, sources, files, diagnostics, library),
    };

    if let Some(yzl) = yzl {
        *yzl = Some(module.as_operation().to_string());
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
        yuzu_passes::infer_types(context, &mut module);
        yuzu_passes::promote_locals(context, &mut module);
    });
    if let Some(reader) = index {
        reader.read_inferred(&module);
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
/// op of the dialects, and that is slow.
pub(crate) fn in_thread_context<T>(compile: impl FnOnce(&melior::Context) -> T) -> T {
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fmt::Write;

    use super::{CompileOptions, compile};
    use crate::modules::{MapResolver, Origin};

    #[test]
    fn a_failed_compile_keeps_its_diagnostics_as_data() {
        let resolver = MapResolver(HashMap::new());
        let error = compile(
            &Origin::Named("test.yz".to_owned()),
            "let x: i64 = 1\n",
            &CompileOptions::default(),
            &resolver,
        )
        .into_plan()
        .expect_err("i64 is not a type");
        assert_eq!(error.diagnostics().len(), 1);
        assert!(error.to_string().contains("unknown type `i64`"), "{error}");
    }

    #[test]
    fn the_dumps_are_kept_only_when_asked_for() {
        let resolver = MapResolver(HashMap::new());
        let source = "struct Row { a: int64 }\ntable t = Row\nfrom t\n";
        let quiet = compile(
            &Origin::Named("test.yz".to_owned()),
            source,
            &CompileOptions::default(),
            &resolver,
        );
        assert!(quiet.plan.is_some() && quiet.yzl.is_none() && quiet.yzr.is_none());

        let options = CompileOptions {
            dump_yzl: true,
            dump_yzr: true,
            ..CompileOptions::default()
        };
        let dumped = compile(
            &Origin::Named("test.yz".to_owned()),
            source,
            &options,
            &resolver,
        );
        assert!(dumped.yzl.is_some_and(|yzl| yzl.contains("yzl.table")));
        assert!(dumped.yzr.is_some_and(|yzr| yzr.contains("yzr.output")));
    }

    #[test]
    fn a_call_tree_of_many_expansions_compiles() {
        let mut source = String::from("struct Row { a: int64 }\ntable t = Row\n");
        source.push_str("def f0(x: int64) -> int64 { return x }\n");
        for level in 1..=10 {
            let below = level - 1;
            writeln!(
                source,
                "def f{level}(x: int64) -> int64 {{ return f{below}(x) + f{below}(x) }}"
            )
            .expect("writing to a String cannot fail");
        }
        source.push_str("from t |> select f10(a) as v\n");
        let resolver = MapResolver(HashMap::new());
        let compilation = compile(
            &Origin::Named("test.yz".to_owned()),
            &source,
            &CompileOptions::default(),
            &resolver,
        );
        assert!(
            compilation.diagnostics.is_empty(),
            "{:?}",
            compilation.diagnostics
        );
        assert!(compilation.plan.is_some());
    }
}
