//! LowerAst: the AST → yzl conversion. Names resolve as they are emitted;
//! unresolved types come out as `!yzl.var` for inference, sugar intact.

use std::collections::HashMap;

use melior::Context;
use melior::ir::attribute::StringAttribute;
use melior::ir::operation::OperationLike;
use melior::ir::{BlockLike, BlockRef, Location, Module, Type, Value, ValueLike};
use yuzu_ast::{AstNode, ast};
use yuzu_diagnostics::diagnostics::Span;
use yuzu_diagnostics::diagnostics::builder::DiagnosticBuilder;
use yuzu_diagnostics::diagnostics::engine::DiagnosticsEngine;
use yuzu_diagnostics::source_map::SourceId;
use yuzu_lexer::lexer::{Lexer, Token};
use yuzu_mlir::ext::BlockExt;
use yuzu_mlir::ext::OperationExt;
use yuzu_mlir::ods::yzl;
use yuzu_mlir::types;
use yuzu_types::FunctionRegistry;

use crate::lower_ast_to_yzl::resolve::Resolver;

/// Parses the source and converts it to a yzl module. Everything the
/// conversion cannot carry — parse errors, missing pieces, unsupported
/// constructs — lands in the engine as an error, and a missing piece
/// converts to a `yzl.missing` value, the way HIR lowered `Expr::Missing`.
/// Returns `None` when the source has no root.
pub fn lower_ast_to_yzl<'c>(
    context: &'c Context,
    name: &str,
    source: &str,
    source_id: SourceId,
    diagnostics: &mut DiagnosticsEngine,
    registry: &dyn FunctionRegistry,
) -> Option<Module<'c>> {
    let tokens: Vec<Token> = Lexer::new(source).collect();
    let syntax = yuzu_parser::parse(&tokens, diagnostics, source_id);
    let root = ast::Root::cast(syntax)?;
    let mut converter = AstToYzl::new(context, name, source, source_id, diagnostics, registry);
    Some(converter.convert(&root))
}

struct AstToYzl<'c, 'd> {
    context: &'c Context,
    name: String,
    source_id: SourceId,
    diagnostics: &'d mut DiagnosticsEngine,
    line_starts: Vec<usize>,
    resolver: Resolver<'c>,
    registry: &'d dyn FunctionRegistry,
}

type Locals<'c, 'a> = HashMap<&'c str, Value<'c, 'a>>;

impl<'c, 'd> AstToYzl<'c, 'd> {
    fn new(
        context: &'c Context,
        name: &str,
        source: &str,
        source_id: SourceId,
        diagnostics: &'d mut DiagnosticsEngine,
        registry: &'d dyn FunctionRegistry,
    ) -> Self {
        let mut line_starts = vec![0];
        line_starts.extend(source.match_indices('\n').map(|(at, _)| at + 1));
        Self {
            context,
            name: name.to_string(),
            source_id,
            diagnostics,
            line_starts,
            resolver: Resolver::new(),
            registry,
        }
    }

    /// The context uniques attribute strings for as long as it lives, so
    /// every name is held as the `&'c str` its attribute hands back.
    fn intern(&self, name: &str) -> &'c str {
        StringAttribute::new(self.context, name).value()
    }

    fn ident(&self, ident: Option<ast::Ident>) -> Option<&'c str> {
        ident
            .and_then(|ident| ident.text())
            .map(|text| self.intern(&text))
    }

    fn error(&mut self, node: &impl AstNode, message: &str) {
        self.error_at(node.syntax().text_range(), message);
    }

    fn error_at(&mut self, range: text_size::TextRange, message: &str) {
        let span = Span {
            source_id: self.source_id,
            range,
        };

        self.diagnostics
            .emit(DiagnosticBuilder::error(span, message));
    }

    /// Reports and stands a `yzl.missing` value in for the hole.
    fn missing<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        node: &impl AstNode,
        message: &str,
        ty: Type<'c>,
    ) -> Value<'c, 'a> {
        self.error(node, message);
        self.hole(block, node.syntax().text_range(), ty)
    }

    fn hole<'a>(
        &self,
        block: BlockRef<'c, 'a>,
        range: text_size::TextRange,
        ty: Type<'c>,
    ) -> Value<'c, 'a> {
        let loc = self.location_at(range.start().into());
        block
            .append_operation(yzl::missing(self.context, ty, loc).into())
            .first_result()
    }

    fn location(&self, node: &impl AstNode) -> Location<'c> {
        self.location_at(node.syntax().text_range().start().into())
    }

    fn location_at(&self, offset: usize) -> Location<'c> {
        let line = self.line_starts.partition_point(|&start| start <= offset);
        let column = offset - self.line_starts[line - 1] + 1;
        Location::new(self.context, &self.name, line, column)
    }

    fn convert(&mut self, root: &ast::Root) -> Module<'c> {
        let module = Module::new(Location::new(self.context, &self.name, 1, 1));
        let top = module.body();
        self.hoist(root);
        for stmt in root.stmts() {
            self.convert_stmt(top, &stmt);
        }

        self.convert_output(top);
        module
    }

    /// The program's result is its trailing query: the last top-level value
    /// of type `!yzl.query` anchors the module's `yzl.output`.
    fn convert_output<'a>(&mut self, top: BlockRef<'c, 'a>) {
        let query = top
            .operations()
            .filter_map(|op| {
                let value = op.try_first_result()?;
                (value.r#type() == types::query(self.context)).then(|| (value, op.location()))
            })
            .last();
        if let Some((value, loc)) = query {
            top.append_operation(yzl::output(self.context, value, loc).into());
        }
    }
}

mod expr;
mod rel;
mod resolve;
mod stmt;

#[cfg(test)]
pub(crate) mod test_support {
    use melior::ir::Module;
    use melior::ir::operation::OperationLike;
    use yuzu_diagnostics::diagnostics::engine::DiagnosticsEngine;
    use yuzu_diagnostics::source_map::SourceMap;

    /// Converts a source file, handing back everything a test needs to
    /// render what came out.
    pub(crate) fn convert<'c>(
        context: &'c melior::Context,
        name: &str,
        source: &str,
    ) -> (Module<'c>, SourceMap, DiagnosticsEngine) {
        let mut sources = SourceMap::new();
        let source_id = sources.add(name.to_string(), source.to_string());
        let mut diagnostics = DiagnosticsEngine::new();
        let module = super::lower_ast_to_yzl(
            context,
            name,
            source,
            source_id,
            &mut diagnostics,
            &yuzu_types::Builtins,
        )
        .expect("the source converts");

        (module, sources, diagnostics)
    }

    /// The module a clean conversion produces, as text.
    pub(crate) fn converted(source: &str) -> String {
        let context = yuzu_mlir::context();
        let (module, _, diagnostics) = convert(&context, "test.yz", source);
        assert!(
            diagnostics.diagnostics().is_empty(),
            "conversion reported: {:?}",
            diagnostics
                .diagnostics()
                .iter()
                .map(|diagnostic| diagnostic.message.clone())
                .collect::<Vec<_>>()
        );
        assert!(
            module.as_operation().verify(),
            "the converted module verifies"
        );

        module.as_operation().to_string()
    }
}

#[cfg(test)]
mod tests {
    use melior::ir::operation::OperationLike;

    /// Every query the existing end-to-end suites compile must convert cleanly:
    /// no unsupported constructs, and a module that verifies.
    #[test]
    fn the_correctness_corpus_converts() {
        let corpus = concat!(env!("CARGO_MANIFEST_DIR"), "/../../python/tests");
        let mut sources = vec![std::fs::read_to_string(format!("{corpus}/support.py")).unwrap()];
        for entry in std::fs::read_dir(format!("{corpus}/correctness")).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|extension| extension == "py") {
                sources.push(std::fs::read_to_string(path).unwrap());
            }
        }

        // The harness compiles `schema + query`; the schema is the one
        // triple-quoted chunk of support.py that declares the tables.
        let schema = sources[0]
            .split(r#"""""#)
            .find(|chunk| chunk.contains("struct Employee"))
            .expect("support.py declares the schema")
            .to_string();

        let context = yuzu_mlir::context();
        let mut queries = 0;
        let mut failures = Vec::new();
        for source in &sources {
            let parts: Vec<&str> = source.split(r#"""""#).collect();
            for (index, &chunk) in parts.iter().enumerate() {
                // Odd chunks are the contents of triple-quoted strings; the
                // ones holding Yuzu source pipe from a relation or declare —
                // prose docstrings mentioning `|>` do neither.
                if index % 2 == 0
                    || !((chunk.contains("|>") && chunk.contains("from"))
                        || chunk.contains("struct "))
                {
                    continue;
                }

                queries += 1;
                // The harness wraps a query it expects to fail in
                // `error_of(...)`, in the code that follows the string, and
                // such a query converting cleanly would be the failure here —
                // except when the rejection is the target's: that is decided
                // at emission, and conversion is right to let it through.
                let expects_error = parts.get(index + 1).is_some_and(|after| {
                    after.contains("error_of(") && !after.contains("not supported by the")
                });
                let program = if chunk.contains("struct ") {
                    chunk.to_string()
                } else {
                    format!("{schema}{chunk}")
                };
                let chunk = program.as_str();
                let mut sources = yuzu_diagnostics::source_map::SourceMap::new();
                let source_id = sources.add("corpus.yz".to_string(), chunk.to_string());
                let mut diagnostics =
                    yuzu_diagnostics::diagnostics::engine::DiagnosticsEngine::new();
                match super::lower_ast_to_yzl(
                    &context,
                    "corpus.yz",
                    chunk,
                    source_id,
                    &mut diagnostics,
                    &yuzu_types::Builtins,
                ) {
                    Some(module)
                        if diagnostics.diagnostics().is_empty()
                            && module.as_operation().verify()
                            && !expects_error => {}
                    Some(_) if expects_error && !diagnostics.diagnostics().is_empty() => {}
                    Some(_) if expects_error => failures.push(format!(
                        "{chunk}\n  -> converted, but the harness expects an error"
                    )),
                    Some(_) => {
                        let messages: Vec<&str> = diagnostics
                            .diagnostics()
                            .iter()
                            .map(|diagnostic| diagnostic.message.as_str())
                            .collect();
                        failures.push(format!("{chunk}\n  -> {messages:?}"))
                    }
                    None => failures.push(format!("{chunk}\n  -> no root")),
                }
            }
        }

        assert!(
            queries > 30,
            "the corpus extraction found only {queries} queries"
        );
        assert!(
            failures.is_empty(),
            "{} of {queries} corpus queries failed to convert:\n{}",
            failures.len(),
            failures.join("\n---\n")
        );
    }
}
