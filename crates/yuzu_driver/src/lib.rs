use std::time::Instant;

use yuzu_anf::AnfCtx;
use yuzu_ast::ast::{AstNode, Root};
use yuzu_core::adt::StringInterner;
use yuzu_diagnostics::{
    diagnostics::{Severity, engine::DiagnosticsEngine, printer::DiagnosticPrinter},
    source_map::SourceMap,
};
use yuzu_hir::HirCtx;
use yuzu_lexer::lexer::{Lexer, Token};
use yuzu_plan::RelGraphConverter;
use yuzu_types::TypeCtx;

#[derive(Default)]
pub struct CompileOptions {
    pub debug_tokens: bool,
    pub debug_ast: bool,
    pub debug_hir: bool,
    pub debug_anf: bool,
    pub debug_reduce: bool,
    pub debug_plan: bool,
    pub debug_substrait: bool,
    pub time_phases: bool,
    pub target: Option<String>,
}

pub fn compile(name: &str, source: &str, options: &CompileOptions) {
    let mut diagnostics = DiagnosticsEngine::new();
    let mut sources = SourceMap::new();
    let source_id = sources.add(name.to_string(), source.to_string());
    let mut phases = Phases::new();

    let start = Instant::now();
    let tokens: Vec<Token> = Lexer::new(source).collect();
    phases.record("lex", start);
    if options.debug_tokens {
        println!("=== tokens ===");
        for token in &tokens {
            println!("{:?}@{:?}", token.kind, token.range);
        }
    }

    let start = Instant::now();
    let syntax = yuzu_parser::parse(&tokens, &mut diagnostics, source_id);
    phases.record("parse", start);
    drop(tokens);
    if options.debug_ast {
        println!("=== syntax ===\n{syntax:#?}");
    }

    let Some(root) = Root::cast(syntax) else {
        print_diagnostics(&diagnostics, &sources);
        return;
    };

    let mut hir = HirCtx::new();
    let mut interner = StringInterner::new();
    let start = Instant::now();
    let (root, source_map) =
        yuzu_hir::lower(root, &mut hir, &mut interner, &mut diagnostics, source_id);
    phases.record("hir", start);
    if options.debug_hir {
        println!("=== hir ===");
        print!("{}", yuzu_hir::dump(&hir, &interner, &root));
    }

    let mut types = TypeCtx::new();
    let start = Instant::now();
    let inference = yuzu_hir::infer(
        &root,
        &hir,
        &yuzu_types::Builtins,
        &mut interner,
        &mut types,
        &mut diagnostics,
        &source_map,
    );
    phases.record("infer", start);

    let backend = options.debug_anf
        || options.debug_reduce
        || options.debug_plan
        || options.debug_substrait
        || options.time_phases;
    if backend && !has_errors(&diagnostics) {
        let mut anf = AnfCtx::new();
        let start = Instant::now();
        let (anf_root, mut anf_source_map) = yuzu_anf::lower(
            &root,
            &hir,
            &inference,
            &types,
            &mut anf,
            &mut interner,
            &mut diagnostics,
            &source_map,
        );
        phases.record("anf", start);
        drop(root);
        drop(hir);
        drop(source_map);
        drop(inference);
        if options.debug_anf {
            println!("=== anf ===");
            print!("{}", yuzu_anf::dump(&anf, &interner, &anf_root));
        }
        if options.debug_reduce
            || options.debug_plan
            || options.debug_substrait
            || options.time_phases
        {
            let start = Instant::now();
            let reduced = yuzu_anf::reduce(&anf_root, &mut anf, &mut interner, &mut anf_source_map);
            phases.record("reduce", start);
            if options.debug_reduce {
                println!("=== reduced ===");
                print!("{}", yuzu_anf::dump(&anf, &interner, &reduced));
            }
            let start = Instant::now();
            let query = find_query_span(&reduced, &anf, &anf_source_map);
            let graph = query.as_ref().and_then(|query| {
                build_plan(
                    query,
                    &anf,
                    &mut types,
                    &interner,
                    &anf_source_map,
                    &mut diagnostics,
                )
            });
            phases.record("plan", start);
            drop(anf_source_map);
            drop(anf);
            if let (Some(graph), Some(query)) = (graph, query) {
                if let Some(target) = parse_target(options, &mut diagnostics, source_id) {
                    yuzu_plan::validate(&graph, &target, &mut diagnostics, query.span);
                }
                if options.debug_plan {
                    println!("=== plan ===");
                    print!("{}", yuzu_plan::dump(&graph, &interner));
                }
                let start = Instant::now();
                let plan =
                    yuzu_substrait::emit(&graph, &types, &interner, &mut diagnostics, query.span);
                phases.record("substrait", start);
                if options.debug_substrait
                    && let Some(plan) = plan
                {
                    println!("=== substrait ===");
                    println!("{}", yuzu_substrait::to_json(&plan));
                }
            }
        }
    }

    if options.time_phases {
        phases.print();
    }
    print_diagnostics(&diagnostics, &sources);
}

/// Compiles through the MLIR pipeline — as far as it goes today: the AST →
/// yzl conversion, printed. The checking and lowering conversions extend this
/// path until it reaches Substrait and the old pipeline retires.
pub fn compile_mlir(name: &str, source: &str) -> std::process::ExitCode {
    use melior::ir::operation::OperationLike;

    let context = yuzu_mlir::context();
    let mut sources = SourceMap::new();
    let source_id = sources.add(name.to_string(), source.to_string());
    let mut diagnostics = DiagnosticsEngine::new();
    let Some(module) =
        yuzu_lang::convert_source(&context, name, source, source_id, &mut diagnostics)
    else {
        print_diagnostics(&diagnostics, &sources);
        return std::process::ExitCode::FAILURE;
    };

    let verified =
        yuzu_mlir::diagnostics::capture(&context, source_id, source, &mut diagnostics, || {
            let verified = module.as_operation().verify();
            if verified {
                yuzu_passes::resolve_names(&context, &module, &yuzu_types::Builtins);
            }
            verified
        });
    if verified && !has_errors(&diagnostics) {
        yuzu_mlir::diagnostics::capture(&context, source_id, source, &mut diagnostics, || {
            yuzu_passes::infer_types(&context, &module);
            yuzu_passes::check_aggregates(&module, &yuzu_types::Builtins);
        });
    }
    print_diagnostics(&diagnostics, &sources);
    if !verified || has_errors(&diagnostics) {
        return std::process::ExitCode::FAILURE;
    }
    print!("{}", module.as_operation());
    std::process::ExitCode::SUCCESS
}

/// Wall-clock time spent in each compile phase.
struct Phases {
    entries: Vec<(&'static str, std::time::Duration)>,
}

impl Phases {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    fn record(&mut self, phase: &'static str, start: Instant) {
        self.entries.push((phase, start.elapsed()));
    }

    fn print(&self) {
        println!("=== timing ===");
        for (phase, duration) in &self.entries {
            println!("{phase:<10} {duration:>10.1?}");
        }
        let total: std::time::Duration = self.entries.iter().map(|(_, d)| *d).sum();
        println!("{:<10} {total:>10.1?}", "total");
    }
}

pub fn compile_to_substrait(
    name: &str,
    source: &str,
    options: &CompileOptions,
) -> Result<Vec<u8>, String> {
    let mut diagnostics = DiagnosticsEngine::new();
    let mut sources = SourceMap::new();
    let source_id = sources.add(name.to_string(), source.to_string());

    let tokens: Vec<Token> = Lexer::new(source).collect();
    if options.debug_tokens {
        println!("=== tokens ===");
        for token in &tokens {
            println!("{:?}@{:?}", token.kind, token.range);
        }
    }

    let syntax = yuzu_parser::parse(&tokens, &mut diagnostics, source_id);
    if options.debug_ast {
        println!("=== syntax ===\n{syntax:#?}");
    }

    let Some(root) = Root::cast(syntax) else {
        return Err(render_diagnostics(&diagnostics, &sources));
    };

    let mut hir = HirCtx::new();
    let mut interner = StringInterner::new();
    let (root, source_map) =
        yuzu_hir::lower(root, &mut hir, &mut interner, &mut diagnostics, source_id);
    if options.debug_hir {
        println!("=== hir ===");
        print!("{}", yuzu_hir::dump(&hir, &interner, &root));
    }

    let mut types = TypeCtx::new();
    let inference = yuzu_hir::infer(
        &root,
        &hir,
        &yuzu_types::Builtins,
        &mut interner,
        &mut types,
        &mut diagnostics,
        &source_map,
    );
    if has_errors(&diagnostics) {
        return Err(render_diagnostics(&diagnostics, &sources));
    }

    let mut anf = AnfCtx::new();
    let (anf_root, mut anf_source_map) = yuzu_anf::lower(
        &root,
        &hir,
        &inference,
        &types,
        &mut anf,
        &mut interner,
        &mut diagnostics,
        &source_map,
    );
    drop(root);
    drop(hir);
    drop(source_map);
    drop(inference);
    if options.debug_anf {
        println!("=== anf ===");
        print!("{}", yuzu_anf::dump(&anf, &interner, &anf_root));
    }

    let reduced = yuzu_anf::reduce(&anf_root, &mut anf, &mut interner, &mut anf_source_map);
    if options.debug_reduce {
        println!("=== reduced ===");
        print!("{}", yuzu_anf::dump(&anf, &interner, &reduced));
    }

    let query = find_query_span(&reduced, &anf, &anf_source_map);
    let graph = query.as_ref().and_then(|query| {
        build_plan(
            query,
            &anf,
            &mut types,
            &interner,
            &anf_source_map,
            &mut diagnostics,
        )
    });
    drop(anf_source_map);
    drop(anf);
    if options.debug_plan
        && let Some(graph) = &graph
    {
        println!("=== plan ===");
        print!("{}", yuzu_plan::dump(graph, &interner));
    }
    if let (Some(graph), Some(query), Some(target)) = (
        &graph,
        &query,
        parse_target(options, &mut diagnostics, source_id),
    ) {
        yuzu_plan::validate(graph, &target, &mut diagnostics, query.span);
    }
    if has_errors(&diagnostics) {
        return Err(render_diagnostics(&diagnostics, &sources));
    }
    let plan = if let (Some(graph), Some(query)) = (&graph, &query) {
        yuzu_substrait::emit(graph, &types, &interner, &mut diagnostics, query.span)
    } else {
        None
    };
    if has_errors(&diagnostics) {
        return Err(render_diagnostics(&diagnostics, &sources));
    }
    let Some(plan) = plan else {
        return Err("the program has no query".to_string());
    };
    if options.debug_substrait {
        println!("=== substrait ===");
        println!("{}", yuzu_substrait::to_json(&plan));
    }

    Ok(yuzu_substrait::to_protobuf(&plan))
}

/// The program's query with its location, resolved once so later phases can
/// outlive the ANF it was found in.
struct Query {
    rel: yuzu_anf::RelId,
    stmt: yuzu_anf::StmtId,
    span: yuzu_diagnostics::diagnostics::Span,
}

fn find_query_span(
    reduced: &yuzu_anf::Root,
    anf: &AnfCtx,
    anf_source_map: &yuzu_anf::AnfSourceMap,
) -> Option<Query> {
    let (rel, stmt) = yuzu_anf::find_query(reduced, anf)?;
    let span = anf_source_map
        .stmt(stmt)
        .expect("the query is in the source map");
    Some(Query { rel, stmt, span })
}

/// Converts the query into the plan graph — `None` if it contains something a
/// plan cannot express.
fn build_plan(
    query: &Query,
    anf: &AnfCtx,
    types: &mut TypeCtx,
    interner: &StringInterner,
    anf_source_map: &yuzu_anf::AnfSourceMap,
    diagnostics: &mut DiagnosticsEngine,
) -> Option<yuzu_plan::RelGraph> {
    let mut converter = yuzu_plan::AnfToRelGraphConverter::new(
        anf,
        types,
        interner,
        anf_source_map,
        diagnostics,
        query.stmt,
    );
    converter.convert(query.rel)
}

fn parse_target(
    options: &CompileOptions,
    diagnostics: &mut DiagnosticsEngine,
    source_id: yuzu_diagnostics::source_map::SourceId,
) -> Option<yuzu_plan::Target> {
    let Some(text) = &options.target else {
        return Some(yuzu_plan::Target {
            dialect: yuzu_plan::Dialect::DataFusion,
            version: None,
        });
    };
    match text.parse() {
        Ok(target) => Some(target),
        Err(message) => {
            let span = yuzu_diagnostics::diagnostics::Span {
                source_id,
                range: Default::default(),
            };
            diagnostics.emit(
                yuzu_diagnostics::diagnostics::builder::DiagnosticBuilder::error(span, message),
            );
            None
        }
    }
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
