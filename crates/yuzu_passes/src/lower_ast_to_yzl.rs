//! LowerAst: the AST → yzl conversion. Names resolve as they are emitted;
//! unresolved types come out as `!yzl.var` for inference, sugar intact.

use std::collections::{HashMap, HashSet};

use melior::Context;
use melior::ir::attribute::StringAttribute;
use melior::ir::operation::OperationLike;
use melior::ir::{BlockLike, BlockRef, Location, Module, Type, Value, ValueLike};
use yuzu_ast::{AstNode, ast};
use yuzu_diagnostics::diagnostics::Span;
use yuzu_diagnostics::diagnostics::builder::DiagnosticBuilder;
use yuzu_diagnostics::diagnostics::engine::DiagnosticsEngine;
use yuzu_diagnostics::source_map::{SourceId, SourceMap};
use yuzu_mlir::ext::BlockExt;
use yuzu_mlir::ext::OperationExt;
use yuzu_mlir::ods::yzl;
use yuzu_mlir::types;
use yuzu_types::FunctionRegistry;

use crate::lower_ast_to_yzl::symbols::{Binding, SymbolTable};

/// Parses the source and converts it to a yzl module. Everything the
/// conversion cannot carry — parse errors, missing pieces, unsupported
/// constructs — lands in the engine as an error, and a missing piece
/// converts to a `yzl.missing` value, the way HIR lowered `Expr::Missing`.
/// Returns `None` when the source has no root.
/// The name a statement declares, when it declares one.
fn declared_name(stmt: &ast::Stmt) -> Option<String> {
    let name = match stmt {
        ast::Stmt::StructStmt(decl) => decl.name(),
        ast::Stmt::TableStmt(decl) => decl.name(),
        ast::Stmt::FuncStmt(decl) => decl.name(),
        ast::Stmt::TraitStmt(decl) => decl.name(),
        ast::Stmt::LetStmt(decl) => decl.name(),
        ast::Stmt::ImplStmt(_)
        | ast::Stmt::ModStmt(_)
        | ast::Stmt::ImportStmt(_)
        | ast::Stmt::FromImportStmt(_)
        | ast::Stmt::BlockStmt(_)
        | ast::Stmt::AssignStmt(_)
        | ast::Stmt::ReturnStmt(_)
        | ast::Stmt::ExprStmt(_) => None,
    }?;

    name.text()
}

/// Every identifier a statement mentions, which over-approximates what it
/// refers to.
fn identifiers(stmt: &ast::Stmt) -> Vec<String> {
    stmt.syntax()
        .descendants_with_tokens()
        .filter_map(|element| element.into_token())
        .filter(|token| token.kind() == yuzu_syntax::SyntaxKind::Identifier)
        .map(|token| token.text().to_string())
        .collect()
}

/// One file of the program: which source it is, the module path holding its
/// declarations, and what it parsed to. The entry file is held under no
/// module, so the names it declares keep the symbols they were written with.
///
/// The root comes in rather than being parsed here, because whoever resolved
/// the imports had to parse the file to find them, and parsing twice would
/// report every syntax error twice.
#[derive(Clone)]
pub struct File {
    pub source_id: SourceId,
    pub module: Option<String>,
    pub root: ast::Root,
}

impl File {
    /// The file the user asked about, which is the one holding the query.
    pub fn entry(source_id: SourceId, root: ast::Root) -> Self {
        Self {
            source_id,
            module: None,
            root,
        }
    }
}

pub fn lower_ast_to_yzl<'c>(
    context: &'c Context,
    sources: &SourceMap,
    files: &[File],
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
    /// What each module converted so far declared, by its path. A file is
    /// converted after everything it imports, so whatever it asks for is
    /// already in here.
    exports: HashMap<String, HashMap<&'c str, Binding<'c>>>,
    /// Where each declared symbol was written, and the scope its own
    /// references resolve against. A module's declaration is built only if
    /// the program reaches it.
    declarations: HashMap<&'c str, (String, ast::Stmt)>,
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
            exports: HashMap::new(),
            declarations: HashMap::new(),
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

    fn error_at_noting(&mut self, range: text_size::TextRange, message: &str, note: String) {
        let span = self.span(range);
        self.diagnostics
            .emit(DiagnosticBuilder::error(span, message).note(note));
    }

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

    fn position(&self, range: text_size::TextRange) -> String {
        let (line, column) = self.line_col(range.start().into());
        format!("{}:{line}:{column}", self.name())
    }

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
    fn convert(&mut self, files: &[File]) -> Option<Module<'c>> {
        let entry = files.last()?;
        let module = Module::new(Location::new(
            self.context,
            self.sources.name(entry.source_id),
            1,
            1,
        ));

        // Names first, for every file: a reference may point forward, and
        // deciding what to convert means resolving references before any
        // body is built.
        for file in files {
            self.enter(file);
            self.bind_imports(&file.root);
            self.hoist(&file.root);
            self.record_declarations(&file.root);
            self.close(file);
        }

        let reached = self.reached(files);
        let top = module.body();
        for file in files {
            self.enter(file);
            self.restore(file);
            for stmt in file.root.stmts() {
                // The entry file is the program, so all of it converts. A
                // module is a library, and only what the program reaches is
                // worth building or type checking.
                if file.module.is_none() || self.is_reached(&reached, &stmt) {
                    self.convert_stmt(top, &stmt);
                }
            }
        }

        self.convert_output(top);
        Some(module)
    }

    /// Points the conversion at one file: what a location names, what a line
    /// number counts against, and what qualifies a declared symbol.
    fn enter(&mut self, file: &File) {
        self.source_id = file.source_id;
        self.module = file.module.clone();
        self.symbols = SymbolTable::new();
    }

    /// Keeps what the file declared, which is what a file importing it reads
    /// and what its own bodies resolve against on the second pass.
    fn close(&mut self, file: &File) {
        self.exports.insert(Self::key(file), self.symbols.exports());
    }

    fn restore(&mut self, file: &File) {
        if let Some(scope) = self.exports.get(&Self::key(file)) {
            self.symbols.restore(scope.clone());
        }
    }

    /// The entry file has no module path, so it is held under one no module
    /// can take.
    fn key(file: &File) -> String {
        file.module.clone().unwrap_or_default()
    }

    /// Records where each declared symbol was written, so a body can be
    /// built later for the ones the program turns out to reach.
    fn record_declarations(&mut self, root: &ast::Root) {
        let scope = Self::key_of(&self.module);
        for stmt in root.stmts() {
            let Some(name) = declared_name(&stmt) else {
                continue;
            };

            let Some(binding) = self.symbols.binding(&name) else {
                continue;
            };

            if let Some(symbol) = binding.kind.symbol() {
                self.declarations.insert(symbol, (scope.clone(), stmt));
            }
        }
    }

    fn key_of(module: &Option<String>) -> String {
        module.clone().unwrap_or_default()
    }

    /// The symbols the program reaches, from the entry file outward.
    ///
    /// A name is looked up in the scope of the file that wrote it, and every
    /// identifier in a declaration counts as a reference. That says yes too
    /// often — a parameter sharing a function's name reads as a use of it —
    /// and never too seldom, so nothing the program needs is left unbuilt.
    fn reached(&mut self, files: &[File]) -> HashSet<&'c str> {
        let mut reached = HashSet::new();
        let mut pending: Vec<(String, ast::Stmt)> = files
            .iter()
            .filter(|file| file.module.is_none())
            .flat_map(|file| file.root.stmts().map(|stmt| (String::new(), stmt)))
            .collect();

        while let Some((scope, stmt)) = pending.pop() {
            for name in identifiers(&stmt) {
                let Some(symbol) = self
                    .exports
                    .get(&scope)
                    .and_then(|bindings| bindings.get(name.as_str()))
                    .and_then(|binding| binding.kind.symbol())
                else {
                    continue;
                };

                if !reached.insert(symbol) {
                    continue;
                }

                if let Some((scope, declaration)) = self.declarations.get(symbol) {
                    pending.push((scope.clone(), declaration.clone()));
                }
            }
        }

        reached
    }

    /// Whether the program reaches what this statement declares.
    fn is_reached(&self, reached: &HashSet<&'c str>, stmt: &ast::Stmt) -> bool {
        let Some(name) = declared_name(stmt) else {
            // An import or a module declaration holds no body of its own.
            return true;
        };

        self.symbols
            .binding(&name)
            .and_then(|binding| binding.kind.symbol())
            .is_some_and(|symbol| reached.contains(symbol))
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

    /// The root a source parses to, with parse errors going to the engine.
    pub(crate) fn parsed(
        sources: &SourceMap,
        source_id: yuzu_diagnostics::source_map::SourceId,
        diagnostics: &mut DiagnosticsEngine,
    ) -> yuzu_ast::ast::Root {
        use yuzu_lexer::lexer::{Lexer, Token};

        let tokens: Vec<Token> = Lexer::new(sources.text(source_id)).collect();
        let syntax = yuzu_parser::parse(&tokens, diagnostics, source_id);
        use yuzu_ast::AstNode;
        yuzu_ast::ast::Root::cast(syntax).expect("a source has a root")
    }

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
        let root = parsed(&sources, source_id, &mut diagnostics);
        let module = super::lower_ast_to_yzl(
            context,
            &sources,
            &[super::File::entry(source_id, root)],
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
        let ids: Vec<super::File> = files
            .iter()
            .map(|(name, module, source)| {
                let source_id = sources.add(name.to_string(), source.to_string());
                super::File {
                    source_id,
                    module: module.map(str::to_string),
                    root: parsed(&sources, source_id, &mut diagnostics),
                }
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
        let ids: Vec<super::File> = files
            .iter()
            .map(|(name, module, source)| {
                let source_id = sources.add(name.to_string(), source.to_string());
                super::File {
                    source_id,
                    module: module.map(str::to_string),
                    root: parsed(&sources, source_id, &mut diagnostics),
                }
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
              yzl.table @h of @helpers.Row
              yzl.struct @Row ["b"] : [!yz.int64]
              yzl.table @t of @Row
              %0 = yzl.from @t
              %1 = yzl.select %0 as ["v"] {
              ^bb0(%arg0: !yzl.var):
                %2 = yzl.call @helpers.double(%arg0) : (!yzl.var) -> !yzl.var {callee_kind = "fn"}
                yzl.yield %2 : !yzl.var
              }
              yzl.output %1
            }
        "#]].assert_eq(&converted_program(&[
            (
                "helpers.yz",
                Some("helpers"),
                "pub struct Row { a: int64 }\npub def double(x: int64) -> int64 { return x * 2 }\n",
            ),
            (
                "main.yz",
                None,
                "from helpers import double, Row as Shape\n\ntable h = Shape\nstruct Row { b: int64 }\ntable t = Row\n\nfrom t |> select double(b) as v\n",
            ),
        ]));
    }

    /// A module is a library, so only what the program reaches is built. The
    /// entry file is the program, so all of it is.
    #[test]
    fn a_module_builds_only_what_the_program_reaches() {
        let module = converted_program(&[
            (
                "helpers.yz",
                Some("helpers"),
                "pub def used(x: int64) -> int64 { return x * 2 }\npub def unused(x: int64) -> int64 { return x + 99 }\n",
            ),
            (
                "main.yz",
                None,
                "from helpers import used\n\nstruct Row { a: int64 }\ntable t = Row\n\nfrom t |> select used(a) as v\n",
            ),
        ]);
        assert!(
            module.contains("@helpers.used") && !module.contains("@helpers.unused"),
            "the reached one is built and the other is not:\n{module}"
        );
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
                let root = crate::lower_ast_to_yzl::test_support::parsed(
                    &sources,
                    source_id,
                    &mut diagnostics,
                );
                match super::lower_ast_to_yzl(
                    &context,
                    &sources,
                    &[super::File::entry(source_id, root)],
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
