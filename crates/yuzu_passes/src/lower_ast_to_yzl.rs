//! Names resolve as they are emitted, and a type the source does not write
//! comes out as `!yzl.unresolved` for inference.

use melior::Context;
use melior::ir::attribute::StringAttribute;
use melior::ir::{BlockLike, BlockRef, Location, Module, Type, Value};
use rustc_hash::FxHashMap;
use text_size::TextRange;
use yuzu_ast::ast::{self, AstNode, Visibility};
use yuzu_diagnostics::{DiagnosticBuilder, DiagnosticsEngine, SourceId, SourceMap, Span};
use yuzu_mlir::ir::location::LocationExt;
use yuzu_mlir::ir::operation::OperationExt;
use yuzu_mlir::ods::yzl;

use crate::lower_ast_to_yzl::symbols::{
    DeclarationKind, Field, ModulePath, Row, SymbolTable, Target,
};

mod expr;
mod program;
mod rel;
mod stmt;
mod symbols;

pub use program::{DeclarationLowering, File};
pub use symbols::{BoundLibrary, PRELUDE};

/// A name the program wrote, and what it names.
#[derive(Clone, Copy, Debug)]
pub struct NameUse<'a> {
    /// Where the name is written.
    pub used: Span,
    /// The name as written. An alias spells it differently from the
    /// declaration it names.
    pub spelling: &'a str,
    /// What the name names.
    pub target: NameTarget<'a>,
    /// The `x as y` import item that renamed the declaration in this file,
    /// when the name goes through one.
    pub alias: Option<Span>,
}

/// What a name names.
#[derive(Clone, Copy, Debug)]
pub enum NameTarget<'a> {
    /// A declaration: all of its syntax, and the name it declares.
    Declaration { at: Span, name: &'a str },
    /// A module: the file that holds it, and its path.
    Module { file: SourceId, path: &'a str },
}

/// What a name a file can use declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NameKind {
    Struct,
    Relation,
    Function,
    Trait,
    /// A module-level `let`.
    Binding,
    Module,
}

/// A listener that the lowering tells each name it resolves.
///
/// An editor sees through it the names the IR does not keep: an import, an
/// alias, a trait in a bound. The lowering also tells it the names in scope,
/// for completion.
pub trait NameListener {
    /// A name the program wrote, and what it names.
    fn on_name(&mut self, name: NameUse<'_>);

    /// A declaration the program wrote, with all of its syntax and the name
    /// it declares. The lowering also reports a declaration that no name uses.
    fn on_declaration(&mut self, _at: Span, _name: &str) {}

    /// The columns the expressions of the stage at `stage` can read.
    fn on_row(&mut self, _stage: Span, _columns: &mut dyn Iterator<Item = &str>) {}

    /// The names the top level of the file `file` can use. `module` is the
    /// file's module path. The entry file has none.
    fn on_file(
        &mut self,
        _file: SourceId,
        _module: Option<&str>,
        _names: &mut dyn Iterator<Item = ScopeEntry<'_>>,
    ) {
    }
}

/// A name a file's top level can use.
#[derive(Clone, Copy, Debug)]
pub struct ScopeEntry<'a> {
    pub name: &'a str,
    pub kind: NameKind,
    /// How far the name reaches from this file. A name the file sees from
    /// the prelude is private here.
    pub visibility: Visibility,
    /// For a module, the file that holds it.
    pub module_file: Option<SourceId>,
}

struct Silent;

impl NameListener for Silent {
    fn on_name(&mut self, _name: NameUse<'_>) {}
}

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
    lower_ast_to_yzl_with_listener(context, sources, files, diagnostics, library, &mut Silent)
}

/// [`lower_ast_to_yzl`], telling `listener` each name it resolves, the
/// columns each stage can read and the names each file can use.
///
/// # Panics
///
/// If `files` is empty.
pub fn lower_ast_to_yzl_with_listener<'c>(
    context: &'c Context,
    sources: &SourceMap,
    files: &[File],
    diagnostics: &mut DiagnosticsEngine,
    library: Option<&'c BoundLibrary<'c>>,
    listener: &mut dyn NameListener,
) -> Module<'c> {
    let entry = files.last().expect("a program has an entry file");
    let mut lowerer = AstToYzl::new(
        context,
        library,
        sources,
        files,
        entry,
        diagnostics,
        listener,
    );
    let module = lowerer.lower(files, entry);
    lowerer.record_files(files);
    module
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
    let mut silent = Silent;
    let mut lowerer = AstToYzl::new(
        context,
        None,
        sources,
        files,
        first,
        diagnostics,
        &mut silent,
    );
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
    listener: &'d mut dyn NameListener,
    /// The file each module was read from, for a name to point into.
    module_files: FxHashMap<ModulePath<'c>, SourceId>,
}

/// The values a function body's `let`s bound, by slot. They live apart from
/// the symbol table because each borrows the block being built.
type Locals<'c, 'a> = Vec<Value<'c, 'a>>;

/// A name as the program wrote it: its interned text, and where the file
/// being lowered writes it.
#[derive(Clone, Copy, Debug)]
struct Name<'c> {
    text: &'c str,
    range: TextRange,
}

/// What a declaration is, as a completion shows it. The scope follows an
/// import to what it names, so an import has no kind.
fn name_kind(kind: DeclarationKind) -> Option<NameKind> {
    Some(match kind {
        DeclarationKind::Struct => NameKind::Struct,
        DeclarationKind::Relation => NameKind::Relation,
        DeclarationKind::Function => NameKind::Function,
        DeclarationKind::Trait => NameKind::Trait,
        DeclarationKind::Binding => NameKind::Binding,
        DeclarationKind::Module => NameKind::Module,
        DeclarationKind::Import => return None,
    })
}

impl<'c, 'd> AstToYzl<'c, 'd> {
    /// Starts at `file`. The walk then moves from file to file.
    fn new(
        context: &'c Context,
        library: Option<&'c BoundLibrary<'c>>,
        sources: &'d SourceMap,
        files: &[File],
        file: &File,
        diagnostics: &'d mut DiagnosticsEngine,
        listener: &'d mut dyn NameListener,
    ) -> Self {
        let symbols = SymbolTable::new(context, library);
        let module_files = files
            .iter()
            .map(|file| (file.module_path(&symbols), file.source_id))
            .collect();
        Self {
            context,
            symbols,
            sources,
            source_id: file.source_id,
            file: StringAttribute::new(context, sources.name(file.source_id)),
            diagnostics,
            listener,
            module_files,
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

    /// Tells the listener the names each file's top level can use.
    fn record_files(&mut self, files: &[File]) {
        for file in files {
            let module = file.module_path(&self.symbols);
            let visible = self.symbols.visible_in(module);
            let module_files = &self.module_files;
            let mut names = visible.iter().filter_map(|visible| {
                Some(ScopeEntry {
                    name: visible.name,
                    kind: name_kind(visible.kind)?,
                    visibility: visible.visibility,
                    module_file: visible
                        .module
                        .and_then(|path| module_files.get(&ModulePath::from_path(path)).copied()),
                })
            });
            self.listener
                .on_file(file.source_id, file.module.as_deref(), &mut names);
        }
    }

    /// Tells the listener that `name` names a declaration, through the
    /// file's own import of it when there is one.
    fn record(&mut self, name: Name<'_>, target: Target<'c>) {
        let alias = self.symbols.import_alias(name.text);
        self.record_through(name, target, alias);
    }

    /// Tells the listener that `name` names a declaration, through the
    /// import at `alias` that renamed it. A declaration in a module that
    /// this run did not read has no file. A bound library is an example.
    fn record_through(&mut self, name: Name<'_>, target: Target<'c>, alias: Option<TextRange>) {
        let Some(&source_id) = self.module_files.get(&target.at.module) else {
            return;
        };
        self.listener.on_name(NameUse {
            used: self.span(name.range),
            spelling: name.text,
            alias: alias.map(|alias| self.span(alias)),
            target: NameTarget::Declaration {
                at: Span {
                    source_id,
                    range: target.range,
                },
                name: target.at.name,
            },
        });
    }

    /// Tells the listener that `name` names what `declared` declares,
    /// under the same name: a column.
    fn record_declared(&mut self, name: Name<'_>, declared: Span) {
        self.listener.on_name(NameUse {
            used: self.span(name.range),
            spelling: name.text,
            target: NameTarget::Declaration {
                at: declared,
                name: name.text,
            },
            alias: None,
        });
    }

    /// Tells the listener that `name` names a local of this file.
    fn record_local(&mut self, name: Name<'_>, declared: TextRange) {
        self.record_declared(name, self.span(declared));
    }

    /// Tells the listener that this file declares `name` at `declared`.
    fn declare(&mut self, name: &str, declared: TextRange) {
        self.listener.on_declaration(self.span(declared), name);
    }

    fn record_module(&mut self, name: Name<'_>, path: &'c str) {
        let Some(&source_id) = self.module_files.get(&ModulePath::from_path(path)) else {
            return;
        };
        self.listener.on_name(NameUse {
            used: self.span(name.range),
            spelling: name.text,
            target: NameTarget::Module {
                file: source_id,
                path,
            },
            alias: None,
        });
    }

    fn span(&self, range: TextRange) -> Span {
        Span {
            source_id: self.source_id,
            range,
        }
    }

    fn diagnostic_at(&self, range: TextRange, message: &str) -> DiagnosticBuilder {
        DiagnosticBuilder::error(self.span(range), message)
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
            self.report_out_of_range(int);
        }

        value
    }

    fn report_out_of_range(&mut self, int: &ast::IntLiteral) {
        self.report(int, "integer literal is out of range for `int64`");
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

    fn hole_and_report<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        node: &impl AstNode,
        message: &str,
        ty: Type<'c>,
    ) -> Value<'c, 'a> {
        self.report(node, message);
        self.emit_hole(block, node.syntax().text_range(), ty)
    }

    /// Checks, in a debug build, that the file has an error. Only a syntax
    /// error leaves a piece of syntax out of the tree. The parser reports
    /// it, so the lowering reports nothing.
    fn assert_syntax_error(&self, what: &str) {
        debug_assert!(
            self.has_error_in_file(),
            "{what}, which only a syntax error leaves"
        );
    }

    /// A hole for a piece of syntax the tree lacks.
    fn hole_and_assert<'a>(
        &self,
        block: BlockRef<'c, 'a>,
        node: &impl AstNode,
        what: &str,
        ty: Type<'c>,
    ) -> Value<'c, 'a> {
        self.assert_syntax_error(what);
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

    /// A name the source wrote, with where it wrote it.
    fn read_ident(&self, ident: Option<ast::Ident>) -> Option<Name<'c>> {
        self.name_of(&ident?)
    }

    /// The text of a name the source wrote, interned for the rest of the
    /// pass.
    fn read_ident_as_string(&self, ident: Option<ast::Ident>) -> Option<&'c str> {
        self.read_ident(ident).map(|ident| ident.text)
    }

    /// A field that the item at `range` declares under `name`.
    fn field_at(&mut self, name: &'c str, range: TextRange) -> Field<'c> {
        self.declare(name, range);
        Field {
            name,
            declared: Some(self.span(range)),
        }
    }

    /// [`Self::read_name`] for a name already in hand.
    fn name_of(&self, ident: &ast::Ident) -> Option<Name<'c>> {
        let token = ident.token()?;
        Some(Name {
            text: self.symbols.intern(token.text()),
            range: ident.syntax().text_range(),
        })
    }

    fn name(&self) -> &str {
        self.sources.name(self.source_id)
    }
}
