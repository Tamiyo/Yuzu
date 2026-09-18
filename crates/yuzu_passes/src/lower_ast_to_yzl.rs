//! LowerAst: the AST → yzl conversion. Names resolve as they are emitted;
//! unresolved types come out as `!yzl.var` for inference, sugar intact.

use melior::Context;
use melior::ir::attribute::StringAttribute;
use melior::ir::operation::OperationLike;
use melior::ir::{BlockLike, BlockRef, Location, Module, Type, Value, ValueLike};
use yuzu_ast::{AstNode, ast};
use yuzu_diagnostics::diagnostics::Span;
use yuzu_diagnostics::diagnostics::builder::DiagnosticBuilder;
use yuzu_diagnostics::diagnostics::engine::DiagnosticsEngine;
use yuzu_diagnostics::source_map::{SourceId, SourceMap};
use yuzu_lexer::lexer::{Lexer, Token};
use yuzu_mlir::ext::BlockExt;
use yuzu_mlir::ext::OperationExt;
use yuzu_mlir::ods::yzl;
use yuzu_mlir::types;
use yuzu_types::FunctionRegistry;

use crate::lower_ast_to_yzl::symbols::SymbolTable;

/// Parses the source and converts it to a yzl module. Everything the
/// conversion cannot carry — parse errors, missing pieces, unsupported
/// constructs — lands in the engine as an error, and a missing piece
/// converts to a `yzl.missing` value, the way HIR lowered `Expr::Missing`.
/// Returns `None` when the source has no root.
/// One file of the program: which source it is, and the module path holding
/// its declarations. The entry file is held under no module, so the names it
/// declares keep the symbols they were written with.
#[derive(Clone, Copy)]
pub struct File<'a> {
    pub source_id: SourceId,
    pub module: Option<&'a str>,
}

impl File<'_> {
    /// The file the user asked about, which is the one holding the query.
    pub fn entry(source_id: SourceId) -> Self {
        Self {
            source_id,
            module: None,
        }
    }
}

pub fn lower_ast_to_yzl<'c>(
    context: &'c Context,
    sources: &SourceMap,
    files: &[File<'_>],
    diagnostics: &mut DiagnosticsEngine,
    registry: &dyn FunctionRegistry,
) -> Option<Module<'c>> {
    let first = files.first()?;
    let mut converter = AstToYzl::new(context, sources, first.source_id, diagnostics, registry);
    converter.convert(files)
}

struct AstToYzl<'c, 'd> {
    context: &'c Context,
    /// Every file the compile read, and which of them is being converted.
    /// The name a location carries, the text being lexed and the identifier
    /// a diagnostic gets all come from here, so none of them can disagree.
    sources: &'d SourceMap,
    source_id: SourceId,
    /// The module path holding what the file being converted declares, or
    /// none for the entry file. It is what qualifies every symbol, so one
    /// module's `Row` and another's are two declarations in one module.
    module: Option<String>,
    diagnostics: &'d mut DiagnosticsEngine,
    symbols: SymbolTable<'c>,
    registry: &'d dyn FunctionRegistry,
    /// How many `let`s have taken a name something else already held. One
    /// name is one symbol in the module, so a rebinding needs its own.
    rebound: usize,
}

/// The values a function body's `let`s bound, in the order they bound them.
/// The symbol table says which slot a name resolves to; the values can only
/// live here, since each borrows the block being built.
type Locals<'c, 'a> = Vec<Value<'c, 'a>>;

impl<'c, 'd> AstToYzl<'c, 'd> {
    fn new(
        context: &'c Context,
        sources: &'d SourceMap,
        source_id: SourceId,
        diagnostics: &'d mut DiagnosticsEngine,
        registry: &'d dyn FunctionRegistry,
    ) -> Self {
        Self {
            context,
            sources,
            source_id,
            module: None,
            diagnostics,
            symbols: SymbolTable::new(),
            registry,
            rebound: 0,
        }
    }

    /// The symbol a declared name is held under. A module qualifies what it
    /// declares, so two modules may each declare `Row` and the module they
    /// lower into holds two.
    fn symbol_for(&self, name: &'c str) -> &'c str {
        match &self.module {
            Some(module) => self.intern(&format!("{module}.{name}")),
            None => name,
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

    fn span(&self, range: text_size::TextRange) -> Span {
        Span {
            source_id: self.source_id,
            range,
        }
    }

    fn error(&mut self, node: &impl AstNode, message: &str) {
        self.error_at(node.syntax().text_range(), message);
    }

    fn error_at(&mut self, range: text_size::TextRange, message: &str) {
        let span = self.span(range);
        self.diagnostics
            .emit(DiagnosticBuilder::error(span, message));
    }

    /// Reports with a note under the snippet — for what the reader would
    /// otherwise have to go and look up: the columns actually in the row,
    /// the declaration a name already has.
    fn error_at_noting(&mut self, range: text_size::TextRange, message: &str, note: String) {
        let span = self.span(range);
        self.diagnostics
            .emit(DiagnosticBuilder::error(span, message).note(note));
    }

    /// Reports a column reference the row could not answer, noting what the
    /// row does carry.
    fn unresolved_column(&mut self, node: &impl AstNode, message: &str) {
        let range = node.syntax().text_range();
        match self.row_note() {
            Some(note) => self.error_at_noting(range, message, note),
            None => self.error_at(range, message),
        }
    }

    /// The columns a reader could have written here, as a note. `None` when
    /// no relation is in scope, so there is no row to list.
    fn row_note(&self) -> Option<String> {
        const SHOWN: usize = 8;

        let row = self.symbols.current_row()?;
        if row.len() == 0 {
            return Some("this relation carries no columns".to_string());
        }

        let mut names: Vec<String> = row
            .references()
            .take(SHOWN)
            .map(|reference| format!("`{reference}`"))
            .collect();
        if row.len() > SHOWN {
            names.push(format!("and {} more", row.len() - SHOWN));
        }

        Some(format!("the row carries {}", names.join(", ")))
    }

    /// Where a range starts, as the printer would show it.
    fn position(&self, range: text_size::TextRange) -> String {
        let (line, column) = self.line_col(range.start().into());
        format!("{}:{line}:{column}", self.name())
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
        let (line, column) = self.line_col(offset);
        Location::new(self.context, self.name(), line, column)
    }

    fn line_col(&self, offset: usize) -> (usize, usize) {
        let at = self.sources.line_col(self.source_id, offset);
        (at.line, at.col)
    }

    /// The name of the file being converted, which is what a location says.
    fn name(&self) -> &str {
        self.sources.name(self.source_id)
    }

    /// Every file into one module, in the order the caller resolved them,
    /// with the entry last. A module is a scope rather than a nesting: what
    /// a file declares is qualified by its path and lowered alongside
    /// everything else, so nothing downstream has a module to walk into.
    fn convert(&mut self, files: &[File<'_>]) -> Option<Module<'c>> {
        let entry = files.last()?;
        let module = Module::new(Location::new(
            self.context,
            self.sources.name(entry.source_id),
            1,
            1,
        ));
        let top = module.body();
        for file in files {
            self.convert_file(top, *file);
        }

        self.convert_output(top);
        Some(module)
    }

    /// One file, under a scope of its own. Nothing crosses between files
    /// until imports bind a name, so each starts from an empty module scope.
    fn convert_file<'a>(&mut self, top: BlockRef<'c, 'a>, file: File<'_>) {
        self.source_id = file.source_id;
        self.module = file.module.map(|module| module.to_string());
        self.symbols = SymbolTable::new();

        let tokens: Vec<Token> = Lexer::new(self.sources.text(file.source_id)).collect();
        let syntax = yuzu_parser::parse(&tokens, self.diagnostics, file.source_id);
        let Some(root) = ast::Root::cast(syntax) else {
            return;
        };

        self.hoist(&root);
        for stmt in root.stmts() {
            self.convert_stmt(top, &stmt);
        }
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
mod stmt;
mod symbols;

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
            &sources,
            &[super::File::entry(source_id)],
            &mut diagnostics,
            &yuzu_types::Builtins,
        )
        .expect("the source converts");

        (module, sources, diagnostics)
    }

    /// Converts a program of several files, the entry last, and renders the
    /// one module they became.
    pub(crate) fn converted_program(files: &[(&str, Option<&str>, &str)]) -> String {
        let context = yuzu_mlir::context();
        let mut sources = SourceMap::new();
        let mut diagnostics = DiagnosticsEngine::new();
        let ids: Vec<super::File<'_>> = files
            .iter()
            .map(|(name, module, source)| super::File {
                source_id: sources.add(name.to_string(), source.to_string()),
                module: *module,
            })
            .collect();

        let module = super::lower_ast_to_yzl(
            &context,
            &sources,
            &ids,
            &mut diagnostics,
            &yuzu_types::Builtins,
        )
        .expect("the program converts");
        assert!(
            diagnostics.diagnostics().is_empty(),
            "conversion reported: {:?}",
            diagnostics
                .diagnostics()
                .iter()
                .map(|d| d.message.as_str())
                .collect::<Vec<_>>()
        );

        module.as_operation().to_string()
    }

    /// What a program of several files reported, rendered with its snippet.
    pub(crate) fn reported_program(files: &[(&str, Option<&str>, &str)]) -> String {
        use yuzu_diagnostics::diagnostics::printer::DiagnosticPrinter;

        let context = yuzu_mlir::context();
        let mut sources = SourceMap::new();
        let mut diagnostics = DiagnosticsEngine::new();
        let ids: Vec<super::File<'_>> = files
            .iter()
            .map(|(name, module, source)| super::File {
                source_id: sources.add(name.to_string(), source.to_string()),
                module: *module,
            })
            .collect();

        super::lower_ast_to_yzl(
            &context,
            &sources,
            &ids,
            &mut diagnostics,
            &yuzu_types::Builtins,
        );

        let printer = DiagnosticPrinter::new(&sources);
        diagnostics
            .diagnostics()
            .iter()
            .map(|diagnostic| printer.print(diagnostic))
            .collect::<Vec<String>>()
            .join("\n")
    }

    /// Everything a conversion reported, rendered with its snippet.
    pub(crate) fn reported(context: &melior::Context, source: &str) -> String {
        use yuzu_diagnostics::diagnostics::printer::DiagnosticPrinter;

        let (_, sources, diagnostics) = convert(context, "test.yz", source);
        let printer = DiagnosticPrinter::new(&sources);
        diagnostics
            .diagnostics()
            .iter()
            .map(|diagnostic| printer.print(diagnostic))
            .collect::<Vec<String>>()
            .join("\n")
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
    use expect_test::expect;
    use melior::ir::operation::OperationLike;

    use crate::lower_ast_to_yzl::test_support::{converted_program, reported_program};

    /// Several files become one module, and a module's declarations are held
    /// under its path. Two modules each declaring `Row` is two declarations,
    /// not a collision, which is what qualification buys.
    #[test]
    fn a_program_of_several_files_becomes_one_module() {
        expect![[r#"
            module {
              yzl.struct @helpers.Row ["a"] : [!yz.int64]
              yzl.fn @helpers.double params ["x"] (!yz.int64) -> !yz.int64 {
              ^bb0(%arg0: !yzl.var):
                %2 = yz.constant_int 2
                %3 = yz.mul %arg0, %2 : !yzl.var, !yz.int64 -> !yzl.var
                yzl.return %3 : !yzl.var
              }
              yzl.struct @Row ["b"] : [!yz.int64]
              yzl.table @t of @Row
              %0 = yzl.from @t
              %1 = yzl.select %0 as ["v"] {
              ^bb0(%arg0: !yzl.var):
                yzl.yield %arg0 : !yzl.var
              }
              yzl.output %1
            }
        "#]]
        .assert_eq(&converted_program(&[
            (
                "helpers.yz",
                Some("helpers"),
                "struct Row { a: int64 }\ndef double(x: int64) -> int64 { return x * 2 }\n",
            ),
            (
                "main.yz",
                None,
                "struct Row { b: int64 }\ntable t = Row\n\nfrom t |> select b as v\n",
            ),
        ]));
    }

    /// The program's query is the entry file's. A module holding one would
    /// leave two, and which ran would be an accident of resolution order.
    #[test]
    fn a_module_cannot_hold_a_query() {
        expect![[r#"
            error: a module cannot hold a query
             --> helpers.yz:4:1
              |
            4 | from t |> select a as v
              | ^^^^^^^^^^^^^^^^^^^^^^^
        "#]]
        .assert_eq(&reported_program(&[
            (
                "helpers.yz",
                Some("helpers"),
                "struct Row { a: int64 }\ntable t = Row\n\nfrom t |> select a as v\n",
            ),
            (
                "main.yz",
                None,
                "struct M { b: int64 }\ntable m = M\n\nfrom m |> select b as w\n",
            ),
        ]));
    }

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
                    &sources,
                    &[super::File::entry(source_id)],
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
