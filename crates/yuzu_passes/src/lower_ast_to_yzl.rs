//! The AST → yzl conversion. Names resolve as they are emitted; unresolved
//! types come out as `!yzl.var` for inference, sugar intact.

use std::collections::HashMap;

use melior::Context;
use melior::ir::attribute::StringAttribute;
use melior::ir::{BlockLike, BlockRef, Location, Module, Type, Value};
use text_size::TextRange;
use yuzu_ast::{AstNode, ast};
use yuzu_diagnostics::diagnostics::Span;
use yuzu_diagnostics::diagnostics::builder::DiagnosticBuilder;
use yuzu_diagnostics::diagnostics::engine::DiagnosticsEngine;
use yuzu_diagnostics::source_map::{SourceId, SourceMap};
use yuzu_mlir::ext::OperationExt;
use yuzu_mlir::ods::yzl;
use yuzu_types::FunctionRegistry;

use crate::lower_ast_to_yzl::symbols::{Binding, SymbolTable};

mod expr;
mod program;
mod rel;
mod stmt;
mod symbols;

pub use program::File;

pub fn lower_ast_to_yzl<'c>(
    context: &'c Context,
    sources: &SourceMap,
    files: &[File],
    diagnostics: &mut DiagnosticsEngine,
    registry: &dyn FunctionRegistry,
) -> Option<Module<'c>> {
    let entry = files.last()?;
    let mut lowerer = AstToYzl {
        context,
        sources,
        source_id: entry.source_id,
        module: None,
        diagnostics,
        symbols: SymbolTable::new(),
        exports: HashMap::new(),
        declarations: HashMap::new(),
        registry,
        rebound: 0,
    };

    Some(lowerer.lower(files, entry))
}

struct AstToYzl<'c, 'd> {
    context: &'c Context,
    sources: &'d SourceMap,
    /// The file being lowered: what a location names and a diagnostic
    /// lands in.
    source_id: SourceId,
    /// The module path qualifying what the file being lowered declares,
    /// or none for the entry file.
    module: Option<&'c str>,
    diagnostics: &'d mut DiagnosticsEngine,
    symbols: SymbolTable<'c>,
    /// What each module declared, by its path. A file is lowered after
    /// everything it imports, so whatever it asks for is already here.
    exports: HashMap<Option<&'c str>, HashMap<&'c str, Binding<'c>>>,
    /// Where each declared symbol was written, and the module its own
    /// references resolve in.
    declarations: HashMap<&'c str, (Option<&'c str>, ast::Stmt)>,
    registry: &'d dyn FunctionRegistry,
    /// How many `let`s have taken a name something else already held.
    rebound: usize,
}

/// The values a function body's `let`s bound, in the order they bound them.
/// The symbol table says which slot a name resolves to; the values can only
/// live here, since each borrows the block being built.
type Locals<'c, 'a> = Vec<Value<'c, 'a>>;

impl<'c, 'd> AstToYzl<'c, 'd> {
    /// The symbol a declared name is held under: qualified by the module, so
    /// two modules may each declare `Row`.
    fn symbol_for(&self, name: &'c str) -> &'c str {
        match self.module {
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

    fn diagnostic(&self, range: TextRange, message: &str) -> DiagnosticBuilder {
        let span = Span {
            source_id: self.source_id,
            range,
        };
        DiagnosticBuilder::error(span, message)
    }

    fn report(&mut self, node: &impl AstNode, message: &str) {
        self.report_at(node.syntax().text_range(), message);
    }

    fn report_at(&mut self, range: TextRange, message: &str) {
        let diagnostic = self.diagnostic(range, message);
        self.diagnostics.emit(diagnostic);
    }

    /// A column reference that did not land, with the row it was resolved
    /// against as the note.
    fn unresolved_column(&mut self, node: &impl AstNode, message: &str) {
        let mut diagnostic = self.diagnostic(node.syntax().text_range(), message);
        if let Some(note) = self.row_note() {
            diagnostic = diagnostic.note(note);
        }

        self.diagnostics.emit(diagnostic);
    }

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

    fn position(&self, range: TextRange) -> String {
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
        self.report(node, message);
        self.hole(block, node.syntax().text_range(), ty)
    }

    fn hole<'a>(&self, block: BlockRef<'c, 'a>, range: TextRange, ty: Type<'c>) -> Value<'c, 'a> {
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

    fn name(&self) -> &str {
        self.sources.name(self.source_id)
    }
}
