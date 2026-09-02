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
            let graph = build_plan(
                &reduced,
                &anf,
                &mut types,
                &interner,
                &anf_source_map,
                &mut diagnostics,
            );
            phases.record("plan", start);
            if let Some(graph) = graph {
                if let Some(target) = parse_target(options, &mut diagnostics, source_id) {
                    validate_plan(
                        &graph,
                        &target,
                        &reduced,
                        &anf,
                        &anf_source_map,
                        &mut diagnostics,
                    );
                }
                if options.debug_plan {
                    println!("=== plan ===");
                    print!("{}", yuzu_plan::dump(&graph, &interner));
                }
                let start = Instant::now();
                let plan = emit_plan(
                    &graph,
                    &reduced,
                    &anf,
                    &types,
                    &interner,
                    &anf_source_map,
                    &mut diagnostics,
                );
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
    if options.debug_anf {
        println!("=== anf ===");
        print!("{}", yuzu_anf::dump(&anf, &interner, &anf_root));
    }

    let reduced = yuzu_anf::reduce(&anf_root, &mut anf, &mut interner, &mut anf_source_map);
    if options.debug_reduce {
        println!("=== reduced ===");
        print!("{}", yuzu_anf::dump(&anf, &interner, &reduced));
    }

    let graph = build_plan(
        &reduced,
        &anf,
        &mut types,
        &interner,
        &anf_source_map,
        &mut diagnostics,
    );
    if options.debug_plan
        && let Some(graph) = &graph
    {
        println!("=== plan ===");
        print!("{}", yuzu_plan::dump(graph, &interner));
    }
    if let (Some(graph), Some(target)) =
        (&graph, parse_target(options, &mut diagnostics, source_id))
    {
        validate_plan(
            graph,
            &target,
            &reduced,
            &anf,
            &anf_source_map,
            &mut diagnostics,
        );
    }
    if has_errors(&diagnostics) {
        return Err(render_diagnostics(&diagnostics, &sources));
    }
    let plan = graph.as_ref().and_then(|graph| {
        emit_plan(
            graph,
            &reduced,
            &anf,
            &types,
            &interner,
            &anf_source_map,
            &mut diagnostics,
        )
    });
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

/// Converts the reduced program's query into the plan graph — `None` if there
/// is no query, or if it contains something a plan cannot express.
#[allow(clippy::too_many_arguments)]
fn build_plan(
    reduced: &yuzu_anf::Root,
    anf: &AnfCtx,
    types: &mut TypeCtx,
    interner: &StringInterner,
    anf_source_map: &yuzu_anf::AnfSourceMap,
    diagnostics: &mut DiagnosticsEngine,
) -> Option<yuzu_plan::RelGraph> {
    let (query, query_stmt) = yuzu_anf::find_query(reduced, anf)?;
    let mut converter = yuzu_plan::AnfToRelGraphConverter::new(
        anf,
        types,
        interner,
        anf_source_map,
        diagnostics,
        query_stmt,
    );
    converter.convert(query)
}

#[allow(clippy::too_many_arguments)]
fn emit_plan(
    graph: &yuzu_plan::RelGraph,
    reduced: &yuzu_anf::Root,
    anf: &AnfCtx,
    types: &TypeCtx,
    interner: &StringInterner,
    anf_source_map: &yuzu_anf::AnfSourceMap,
    diagnostics: &mut DiagnosticsEngine,
) -> Option<yuzu_substrait::Plan> {
    let (_, query_stmt) = yuzu_anf::find_query(reduced, anf)?;
    let query_span = anf_source_map
        .stmt(query_stmt)
        .expect("the query is in the source map");
    yuzu_substrait::emit(graph, types, interner, diagnostics, query_span)
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

fn validate_plan(
    graph: &yuzu_plan::RelGraph,
    target: &yuzu_plan::Target,
    reduced: &yuzu_anf::Root,
    anf: &AnfCtx,
    anf_source_map: &yuzu_anf::AnfSourceMap,
    diagnostics: &mut DiagnosticsEngine,
) {
    let Some((_, query_stmt)) = yuzu_anf::find_query(reduced, anf) else {
        return;
    };
    let query_span = anf_source_map
        .stmt(query_stmt)
        .expect("the query is in the source map");
    yuzu_plan::validate(graph, target, diagnostics, query_span);
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
