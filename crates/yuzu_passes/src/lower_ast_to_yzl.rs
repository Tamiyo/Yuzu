//! Names resolve as they are emitted, and a type the source does not write
//! comes out as `!yzl.unresolved` for inference.

use melior::Context;
use melior::ir::attribute::StringAttribute;
use melior::ir::{BlockLike, BlockRef, Location, Module, Type, Value};
use rustc_hash::FxHashMap;
use text_size::TextRange;
use yuzu_ast::ast::{self, AstNode};
use yuzu_diagnostics::{DiagnosticBuilder, DiagnosticsEngine, SourceId, SourceMap, Span};
use yuzu_mlir::ir::location::LocationExt;
use yuzu_mlir::ir::operation::OperationExt;
use yuzu_mlir::ods::yzl;

use crate::lower_ast_to_yzl::symbols::{DeclarationKind, ModulePath, Row, SymbolTable, Target};

mod expr;
mod program;
mod rel;
mod stmt;
mod symbols;

pub use program::{File, Lowering};
pub use symbols::{BoundLibrary, PRELUDE};

/// A name the program wrote, and what it names.
#[derive(Clone, Copy, Debug)]
pub struct NameUse<'a> {
    /// Where the name is written.
    pub used: Span,
    /// The name as written. An alias spells it differently from the
    /// declaration it names.
    pub spelling: &'a str,
    pub target: NameTarget<'a>,
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

/// Told each name the lowering resolves, so an editor sees the names the
/// IR does not keep: an import, an alias, a trait in a bound. It is also
/// told the names in scope, for completion.
pub trait NameListener {
    fn on_name(&mut self, name: NameUse<'_>);

    /// The columns the expressions of the stage at `stage` can read.
    fn on_row(&mut self, _stage: Span, _columns: &mut dyn Iterator<Item = &str>) {}

    /// The names the top level of the file `file` can use.
    fn on_file(&mut self, _file: SourceId, _names: &mut dyn Iterator<Item = (&str, NameKind)>) {}
}

/// A listener for a lowering that nothing watches.
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

/// [`lower_ast_to_yzl`], telling `listener` each name it resolves.
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
    lowerer.report_files(files);
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

/// What a declaration is, as a completion shows it. An import is followed to
/// what it names, so it is none of these.
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
    /// Starts at `file`; the walk moves from file to file.
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
            .map(|file| {
                let module = match file.module.as_deref() {
                    Some(path) => ModulePath::from_path(symbols.intern(path)),
                    None => ModulePath::entry(),
                };
                (module, file.source_id)
            })
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
    fn report_files(&mut self, files: &[File]) {
        for file in files {
            let module = match file.module.as_deref() {
                Some(path) => ModulePath::from_path(self.symbols.intern(path)),
                None => ModulePath::entry(),
            };
            let visible = self.symbols.visible_in(module);
            let mut names = visible
                .iter()
                .filter_map(|&(name, kind)| Some((name, name_kind(kind)?)));
            self.listener.on_file(file.source_id, &mut names);
        }
    }

    /// Tells the listener that the name at `used` in this file names a
    /// declaration. A declaration in a module this run did not read, as a
    /// bound library's, has no file to point into.
    fn record(&mut self, used: TextRange, spelling: &str, target: Target<'c>) {
        let Some(&source_id) = self.module_files.get(&target.at.module) else {
            return;
        };
        self.listener.on_name(NameUse {
            used: self.span(used),
            spelling,
            target: NameTarget::Declaration {
                at: Span {
                    source_id,
                    range: target.range,
                },
                name: target.at.name,
            },
        });
    }

    /// Tells the listener that the name at `used` names what `declared`
    /// declares, under the same name: a column.
    fn record_declared(&mut self, used: TextRange, name: &str, declared: Span) {
        self.listener.on_name(NameUse {
            used: self.span(used),
            spelling: name,
            target: NameTarget::Declaration { at: declared, name },
        });
    }

    /// Tells the listener that the name at `used` names a local of this file.
    fn record_local(&mut self, used: TextRange, name: &str, declared: TextRange) {
        self.listener.on_name(NameUse {
            used: self.span(used),
            spelling: name,
            target: NameTarget::Declaration {
                at: self.span(declared),
                name,
            },
        });
    }

    /// Tells the listener that the name at `used` names a module.
    fn record_module(&mut self, used: TextRange, spelling: &str, path: &'c str) {
        let Some(&source_id) = self.module_files.get(&ModulePath::from_path(path)) else {
            return;
        };
        self.listener.on_name(NameUse {
            used: self.span(used),
            spelling,
            target: NameTarget::Module {
                file: source_id,
                path,
            },
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

    /// Checks, in a debug build, that a syntax error was reported. Only one
    /// leaves a piece of syntax out of the tree, and the parser reports it,
    /// so the lowering reports nothing.
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

    /// A name the source wrote, interned for the rest of the pass.
    fn read_ident(&self, ident: Option<ast::Ident>) -> Option<&'c str> {
        let token = ident?.token()?;
        Some(self.symbols.intern(token.text()))
    }

    fn name(&self) -> &str {
        self.sources.name(self.source_id)
    }
}
