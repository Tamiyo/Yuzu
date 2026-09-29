//! Names resolve as they are emitted, and a type the source does not write
//! comes out as `!yzl.unresolved` for inference.

use melior::Context;
use melior::ir::attribute::StringAttribute;
use melior::ir::{BlockLike, BlockRef, Location, Module, Type, Value};
use text_size::TextRange;
use yuzu_ast::{AstNode, ast};
use yuzu_diagnostics::diagnostics::Span;
use yuzu_diagnostics::diagnostics::builder::DiagnosticBuilder;
use yuzu_diagnostics::diagnostics::engine::DiagnosticsEngine;
use yuzu_diagnostics::source_map::{SourceId, SourceMap};
use yuzu_mlir::ir::location::LocationExt;
use yuzu_mlir::ir::operation::OperationExt;
use yuzu_mlir::ods::yzl;

use crate::lower_ast_to_yzl::symbols::{Row, SymbolTable};

mod expr;
mod program;
mod rel;
mod stmt;
mod symbols;

pub use program::{File, Lowering};
pub use symbols::{BoundLibrary, PRELUDE};

/// `files` is in the order the imports were resolved, the entry file last,
/// and holds at least that one.
///
/// What it could not lower it reports, and stands a `yzl.missing` hole in
/// the place, so the module it returns may hold holes. Whether the program
/// compiled is the engine's answer, not this one's.
///
/// # Panics
///
/// If `files` is empty. A program without an entry file is a caller bug,
/// not a program that failed to compile.
pub fn lower_ast_to_yzl<'c>(
    context: &'c Context,
    sources: &SourceMap,
    files: &[File],
    diagnostics: &mut DiagnosticsEngine,
    library: Option<&'c BoundLibrary<'c>>,
) -> Module<'c> {
    let entry = files.last().expect("a program has an entry file");
    let mut lowerer = AstToYzl::new(context, library, sources, entry, diagnostics);
    lowerer.lower(files, entry)
}

/// Binds the names `files` declare, in the order the imports were resolved,
/// without lowering anything. A compile given the result skips binding
/// those files again.
///
/// # Panics
///
/// If `files` is empty.
pub fn bind_library<'c>(
    context: &'c Context,
    sources: &SourceMap,
    files: &[File],
    diagnostics: &mut DiagnosticsEngine,
) -> BoundLibrary<'c> {
    let first = files.first().expect("a library has a file");
    let mut lowerer = AstToYzl::new(context, None, sources, first, diagnostics);
    lowerer.bind_names(files);
    lowerer.symbols.into_library()
}

struct AstToYzl<'c, 'd> {
    // What the whole run is given.
    context: &'c Context,
    symbols: SymbolTable<'c>,
    sources: &'d SourceMap,
    source_id: SourceId,
    /// The name of the file being lowered, made once for its locations.
    file: StringAttribute<'c>,
    diagnostics: &'d mut DiagnosticsEngine,
}

/// The values a function body's `let`s bound, by slot. They live apart from
/// the symbol table because each borrows the block being built.
type Locals<'c, 'a> = Vec<Value<'c, 'a>>;

impl<'c, 'd> AstToYzl<'c, 'd> {
    /// Starts at `file`; the walk moves from file to file.
    fn new(
        context: &'c Context,
        library: Option<&'c BoundLibrary<'c>>,
        sources: &'d SourceMap,
        file: &File,
        diagnostics: &'d mut DiagnosticsEngine,
    ) -> Self {
        Self {
            context,
            symbols: SymbolTable::new(context, library),
            sources,
            source_id: file.source_id,
            file: StringAttribute::new(context, sources.name(file.source_id)),
            diagnostics,
        }
    }
}

impl<'c> AstToYzl<'c, '_> {
    fn in_type_params<R>(&mut self, names: Vec<&'c str>, f: impl FnOnce(&mut Self) -> R) -> R {
        self.symbols.open_type_params(names);
        let result = f(self);
        self.symbols.close();
        result
    }

    fn in_block<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
        self.symbols.open_block();
        let result = f(self);
        self.symbols.close();
        result
    }

    /// Also hands back the row the relation has when `f` returns.
    fn in_relation<R>(&mut self, row: Row<'c>, f: impl FnOnce(&mut Self) -> R) -> (R, Row<'c>) {
        self.symbols.open_relation(row);
        let result = f(self);
        (result, self.symbols.close_relation())
    }

    fn diagnostic_at(&self, range: TextRange, message: &str) -> DiagnosticBuilder {
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
        let diagnostic = self.diagnostic_at(range, message);
        self.diagnostics.emit(diagnostic);
    }

    /// `None`, after a report, when `int64` cannot hold the literal.
    fn read_int64(&mut self, int: &ast::IntLiteral) -> Option<i64> {
        let value = int.value().and_then(|value| i64::try_from(value).ok());
        if value.is_none() {
            self.report(int, "integer literal is out of range for `int64`");
        }

        value
    }

    fn report_unresolved(&mut self, node: &impl AstNode, message: &str) {
        let mut diagnostic = self.diagnostic_at(node.syntax().text_range(), message);
        if let Some(note) = self.row_note() {
            diagnostic = diagnostic.note(note);
        }

        self.diagnostics.emit(diagnostic);
    }

    fn row_note(&self) -> Option<String> {
        const SHOWN: usize = 8;

        let row = self.symbols.current_row()?;
        if row.is_empty() {
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

    fn text_at_range(&self, range: TextRange) -> String {
        let (line, column) = self.line_col(range.start().into());
        format!("{}:{line}:{column}", self.name())
    }

    fn error_hole<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        node: &impl AstNode,
        message: &str,
        ty: Type<'c>,
    ) -> Value<'c, 'a> {
        self.report(node, message);
        self.emit_hole(block, node.syntax().text_range(), ty)
    }

    /// A piece of syntax the tree lacks. Only a syntax error leaves one, and
    /// the parser reported it, so the lowering adds nothing.
    fn reported_by_parser(&self, what: &str) {
        debug_assert!(
            self.has_error_in_file(),
            "{what}, which only a syntax error leaves"
        );
    }

    /// A hole for a piece of syntax the tree lacks.
    fn parser_hole<'a>(
        &self,
        block: BlockRef<'c, 'a>,
        node: &impl AstNode,
        what: &str,
        ty: Type<'c>,
    ) -> Value<'c, 'a> {
        self.reported_by_parser(what);
        self.emit_hole(block, node.syntax().text_range(), ty)
    }

    fn has_error_in_file(&self) -> bool {
        self.diagnostics.diagnostics().iter().any(|diagnostic| {
            diagnostic
                .labels
                .iter()
                .any(|label| label.span.source_id == self.source_id)
        })
    }

    fn emit_hole<'a>(
        &self,
        block: BlockRef<'c, 'a>,
        range: TextRange,
        ty: Type<'c>,
    ) -> Value<'c, 'a> {
        let loc = self.location_at(range);
        let op = yzl::missing(self.context, ty, loc).into();
        block.append_operation(op).first_result()
    }

    fn location(&self, node: &impl AstNode) -> Location<'c> {
        self.location_at(node.syntax().text_range())
    }

    /// A location carries the whole range, so a pass reporting against the
    /// op underlines what the program wrote rather than one character of it.
    fn location_at(&self, range: TextRange) -> Location<'c> {
        let (start_line, start_column) = self.line_col(range.start().into());
        let (end_line, end_column) = self.line_col(range.end().into());
        Location::file_range(self.file, start_line, start_column, end_line, end_column)
    }

    fn line_col(&self, offset: usize) -> (usize, usize) {
        let at = self.sources.line_col(self.source_id, offset);
        (at.line, at.col)
    }

    /// A name the source wrote, interned for the rest of the pass.
    fn read_ident(&self, ident: Option<ast::Ident>) -> Option<&'c str> {
        let token = ident?.token()?;
        Some(self.symbols.intern(token.text()))
    }

    fn name(&self) -> &str {
        self.sources.name(self.source_id)
    }
}
