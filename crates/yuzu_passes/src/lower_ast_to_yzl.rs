//! Names resolve as they are emitted, and a type the source does not write
//! comes out as `!yzl.var` for inference.

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

/// `files` is in the order the imports were resolved, the entry file last.
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
    source_id: SourceId,
    /// The path of the file being lowered; `None` for the entry file.
    module: Option<&'c str>,
    diagnostics: &'d mut DiagnosticsEngine,
    symbols: SymbolTable<'c>,
    /// What each module declared, by path.
    exports: HashMap<Option<&'c str>, HashMap<&'c str, Binding<'c>>>,
    /// Where each symbol was declared, and the module it was declared in.
    declarations: HashMap<&'c str, (Option<&'c str>, ast::Stmt)>,
    registry: &'d dyn FunctionRegistry,
    /// How many `let`s have taken a name something else already held.
    rebound: usize,
}

/// The values a function body's `let`s bound, by slot. They live apart from
/// the symbol table because each borrows the block being built.
type Locals<'c, 'a> = Vec<Value<'c, 'a>>;

impl<'c, 'd> AstToYzl<'c, 'd> {
    fn symbol_for(&self, name: &'c str) -> &'c str {
        match self.module {
            Some(module) => self.intern(&format!("{module}.{name}")),
            None => name,
        }
    }

    /// An attribute string lives as long as the context, so a name is held
    /// as one.
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
        let loc = self.location_at(range);
        block
            .append_operation(yzl::missing(self.context, ty, loc).into())
            .first_result()
    }

    fn location(&self, node: &impl AstNode) -> Location<'c> {
        self.location_at(node.syntax().text_range())
    }

    /// A location carries the whole range, so a pass reporting against the
    /// op underlines what the program wrote rather than one character of it.
    fn location_at(&self, range: TextRange) -> Location<'c> {
        let (start_line, start_column) = self.line_col(range.start().into());
        let (end_line, end_column) = self.line_col(range.end().into());
        Location::file_line_col_range(
            self.context,
            self.name(),
            start_line,
            start_column,
            end_line,
            end_column,
        )
    }

    fn line_col(&self, offset: usize) -> (usize, usize) {
        let at = self.sources.line_col(self.source_id, offset);
        (at.line, at.col)
    }

    fn name(&self) -> &str {
        self.sources.name(self.source_id)
    }
}
