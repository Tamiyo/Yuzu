use substrait::proto::{Plan, PlanRel, RelRoot, plan_rel};
use substrait::version;
use yuzu_core::adt::StringInterner;
use yuzu_diagnostics::diagnostics::{Span, builder::DiagnosticBuilder, engine::DiagnosticsEngine};
use yuzu_plan::RelGraph;
use yuzu_types::TypeCtx;

use crate::emitter::types::row_columns;
use crate::extensions::Extensions;

mod expr;
mod rel;
mod types;

/// Emits a plan for the graph's query — `None` if the graph contains something
/// a plan cannot express, reported as a diagnostic at the query's position.
pub fn emit(
    graph: &RelGraph,
    types: &TypeCtx,
    interner: &StringInterner,
    diagnostics: &mut DiagnosticsEngine,
    query_span: Span,
) -> Option<Plan> {
    let root = graph.root()?;

    let mut emitter = GraphEmitter {
        graph,
        types,
        interner,
        diagnostics,
        query_span,
        extensions: Extensions::default(),
    };

    // Build the relation first, so function registration populates the tables.
    let relation = emitter.emit_rel(root).ok()?;

    // Output column names come from the query's row type, so aliased,
    // bare-ident, and generated names all carry through.
    let names = row_columns(types, graph.plan().rel(root).ty())
        .iter()
        .map(|column| interner.text(column.name).to_string())
        .collect();

    Some(Plan {
        version: Some(version::version_with_producer("yuzu")),
        extension_urns: emitter.extensions.urns(),
        extensions: emitter.extensions.declarations(),
        relations: vec![PlanRel {
            rel_type: Some(plan_rel::RelType::Root(RelRoot {
                input: Some(relation),
                names,
            })),
        }],
        ..Default::default()
    })
}

struct GraphEmitter<'e> {
    graph: &'e RelGraph,
    types: &'e TypeCtx,
    interner: &'e StringInterner,
    diagnostics: &'e mut DiagnosticsEngine,
    query_span: Span,
    extensions: Extensions,
}

/// The plan graph contained something Substrait cannot express; a diagnostic
/// has already been reported.
struct Unsupported;

impl GraphEmitter<'_> {
    fn unsupported_query(&mut self, message: impl Into<String>) -> Unsupported {
        let diagnostic = DiagnosticBuilder::error(self.query_span, message)
            .note("expressions must be evaluatable at compile time");
        self.diagnostics.emit(diagnostic);
        Unsupported
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use expect_test::Expect;
    use yuzu_ast::ast::{AstNode, Root as AstRoot};
    use yuzu_core::adt::StringInterner;
    use yuzu_diagnostics::diagnostics::engine::DiagnosticsEngine;
    use yuzu_diagnostics::source_map::SourceMap;
    use yuzu_hir::HirCtx;
    use yuzu_lexer::lexer::{Lexer, Token};
    use yuzu_plan::RelGraphConverter;
    use yuzu_types::TypeCtx;

    use crate::to_json;

    pub(crate) const TABLE: &str = "struct Row { a: int32, b: int32 }\ntable t = Row\n";

    fn compile(input: &str) -> (Option<String>, Vec<String>) {
        let mut interner = StringInterner::new();
        let mut diagnostics = DiagnosticsEngine::new();
        let mut sources = SourceMap::new();
        let source_id = sources.add("test".to_string(), input.to_string());

        let tokens: Vec<Token> = Lexer::new(input).collect();
        let syntax = yuzu_parser::parse(&tokens, &mut diagnostics, source_id);
        let ast_root = AstRoot::cast(syntax).expect("root node");

        let mut hir = HirCtx::new();
        let (hir_root, hir_source_map) = yuzu_hir::lower(
            ast_root,
            &mut hir,
            &mut interner,
            &mut diagnostics,
            source_id,
        );

        let mut types = TypeCtx::new();
        let inference = yuzu_hir::infer(
            &hir_root,
            &hir,
            &yuzu_types::Builtins,
            &mut interner,
            &mut types,
            &mut diagnostics,
            &hir_source_map,
        );

        let messages: Vec<&str> = diagnostics
            .diagnostics()
            .iter()
            .map(|d| d.message.as_str())
            .collect();
        assert!(
            messages.is_empty(),
            "program should type-check cleanly, got: {messages:?}"
        );

        let mut anf = yuzu_anf::AnfCtx::new();
        let (anf_root, mut anf_source_map) = yuzu_anf::lower(
            &hir_root,
            &hir,
            &inference,
            &types,
            &mut anf,
            &mut interner,
            &mut diagnostics,
            &hir_source_map,
        );
        let reduced = yuzu_anf::reduce(&anf_root, &mut anf, &mut interner, &mut anf_source_map);

        let plan = yuzu_anf::find_query(&reduced, &anf).and_then(|(query, query_stmt)| {
            let graph = {
                let mut converter = yuzu_plan::AnfToRelGraphConverter::new(
                    &anf,
                    &mut types,
                    &interner,
                    &anf_source_map,
                    &mut diagnostics,
                    query_stmt,
                );
                converter.convert(query)
            }?;
            let query_span = anf_source_map
                .stmt(query_stmt)
                .expect("the query is in the source map");
            super::emit(&graph, &types, &interner, &mut diagnostics, query_span)
        });

        let messages: Vec<String> = diagnostics
            .diagnostics()
            .iter()
            .map(|d| d.message.clone())
            .collect();
        (plan.map(|plan| to_json(&plan)), messages)
    }

    pub(crate) fn plan(input: &str) -> Option<String> {
        let (plan, messages) = compile(input);
        assert!(messages.is_empty(), "emission reported: {messages:?}");
        plan
    }

    pub(crate) fn check(input: &str, expected: Expect) {
        expected.assert_eq(&plan(input).expect("program should have a query"));
    }

    pub(crate) fn check_error(input: &str, expected: Expect) {
        let (plan, messages) = compile(input);
        assert!(plan.is_none(), "an unsupported program should have no plan");
        expected.assert_eq(&messages.join("\n"));
    }
}

#[cfg(test)]
mod tests {
    use crate::emitter::test_support::{TABLE, plan};

    #[test]
    fn program_without_query_has_no_plan() {
        assert!(plan(&format!("{TABLE}let x = 1")).is_none());
    }
}
