//! What a name means where it is written: every declaration in the
//! program, the scopes open inside a file, and the lookups the walk asks.
//!
//! Every name is interned in the MLIR context when it is read, so the
//! table holds `&'c str` and does not own a name.

use std::fmt;

use melior::Context;
use melior::ir::attribute::StringAttribute;
use rustc_hash::FxHashMap;
use text_size::TextRange;
use yuzu_ast::ast::Visibility;
use yuzu_diagnostics::Span;
use yuzu_mlir::attributes::CalleeSource;
use yuzu_mlir::ir::attribute::string;

use crate::operators::Operator;

/// The module every file sees without importing it.
pub const PRELUDE: &str = "yuzu.prelude";

/// A name as a lookup asks for it: what the source wrote, with the
/// qualifier if it wrote one.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Reference<'a> {
    pub(super) qualifier: Option<&'a str>,
    pub(super) name: &'a str,
}

impl<'a> Reference<'a> {
    pub(super) fn unqualified(name: &'a str) -> Self {
        Self {
            qualifier: None,
            name,
        }
    }
}

impl fmt::Display for Reference<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.qualifier {
            Some(qualifier) => write!(f, "{qualifier}.{}", self.name),
            None => f.write_str(self.name),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Column<'c> {
    qualifier: Option<&'c str>,
    name: &'c str,
    declared: Option<Span>,
}

/// A column's name, and the syntax that named it: a struct's field, or a
/// stage's item. A column nothing names, as `column0`, has none.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Field<'c> {
    pub(super) name: &'c str,
    pub(super) declared: Option<Span>,
}

impl<'c> Column<'c> {
    fn matches(&self, reference: Reference<'_>) -> bool {
        self.name == reference.name
            && reference
                .qualifier
                .is_none_or(|qualifier| self.qualifier == Some(qualifier))
    }

    fn reference(&self) -> Reference<'c> {
        Reference {
            qualifier: self.qualifier,
            name: self.name,
        }
    }
}

/// Two columns may share a name, since a join concatenates both sides, so a
/// column is addressed by position.
#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub(super) struct Row<'c> {
    columns: Vec<Column<'c>>,
    /// The first position of each name, so a lookup does not scan the row.
    first: FxHashMap<&'c str, usize>,
    /// For each position, the next position with the same name.
    next: Vec<Option<usize>>,
    schema: Schema,
}

/// Whether a row's columns are known. A relation an error left behind has
/// columns nobody knows, so a name read in it is not reported again.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
enum Schema {
    #[default]
    Known,
    Lost,
}

impl<'c> Row<'c> {
    /// The row of a relation an error left behind.
    pub(super) fn lost() -> Self {
        Self {
            schema: Schema::Lost,
            ..Self::default()
        }
    }

    fn from_columns(columns: Vec<Column<'c>>, schema: Schema) -> Self {
        let mut row = Self {
            columns,
            first: FxHashMap::default(),
            next: Vec::new(),
            schema,
        };
        row.index();
        row
    }

    /// Rebuilds the positions of each name after the columns change.
    fn index(&mut self) {
        self.first.clear();
        self.first.reserve(self.columns.len());
        self.next.clear();
        self.next.resize(self.columns.len(), None);
        for (at, column) in self.columns.iter().enumerate().rev() {
            if let Some(later) = self.first.insert(column.name, at) {
                self.next[at] = Some(later);
            }
        }
    }

    pub(super) fn len(&self) -> usize {
        self.columns.len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    pub(super) fn names(&self) -> impl Iterator<Item = &'c str> + use<'_, 'c> {
        self.columns.iter().map(|column| column.name)
    }

    /// The syntax that named the column at `index`.
    pub(super) fn declared(&self, index: usize) -> Option<Span> {
        self.columns[index].declared
    }

    pub(super) fn references(&self) -> impl Iterator<Item = Reference<'c>> + use<'_, 'c> {
        self.columns.iter().map(Column::reference)
    }

    pub(super) fn column(&self, reference: Reference<'_>) -> ColumnLookup {
        let same_name =
            std::iter::successors(self.first.get(reference.name).copied(), |&at| self.next[at]);
        let mut matches = same_name.filter(|&at| self.columns[at].matches(reference));

        match (matches.next(), matches.next(), self.schema) {
            (Some(index), None, _) => ColumnLookup::Unique(index),
            (Some(_), Some(_), _) => ColumnLookup::Ambiguous,
            (None, _, Schema::Known) => ColumnLookup::Absent,
            (None, _, Schema::Lost) => ColumnLookup::Lost,
        }
    }

    pub(super) fn has_column(&self, reference: Reference<'_>) -> bool {
        !matches!(self.column(reference), ColumnLookup::Absent)
    }

    pub(super) fn qualify(&mut self, alias: &'c str) {
        for column in &mut self.columns {
            column.qualifier = Some(alias);
        }
    }

    /// Renames a column, and hands back its old name.
    pub(super) fn rename(&mut self, index: usize, field: Field<'c>) -> &'c str {
        let column = &mut self.columns[index];
        column.declared = field.declared;
        let old = std::mem::replace(&mut column.name, field.name);
        self.index();
        old
    }

    /// Removes a column, and hands back its name.
    pub(super) fn remove(&mut self, index: usize) -> &'c str {
        let removed = self.columns.remove(index).name;
        self.index();
        removed
    }

    pub(super) fn append(&mut self, other: Row<'c>) {
        self.columns.extend(other.columns);
        if other.schema == Schema::Lost {
            self.schema = Schema::Lost;
        }
        self.index();
    }
}

impl<'c> From<Vec<Field<'c>>> for Row<'c> {
    fn from(fields: Vec<Field<'c>>) -> Self {
        let columns = fields
            .into_iter()
            .map(|field| Column {
                qualifier: None,
                name: field.name,
                declared: field.declared,
            })
            .collect();
        Self::from_columns(columns, Schema::Known)
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) struct Binding<'c> {
    pub(super) kind: BindingKind<'c>,
    pub(super) text_range: TextRange,
    pub(super) visibility: Visibility,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) enum BindingKind<'c> {
    Struct {
        fields: Vec<Field<'c>>,
    },
    /// A table, or a `let` bound to a query: what `from` and `join` name.
    Relation {
        row: Row<'c>,
    },
    /// Every overload has the same kind, and no two take the same number
    /// of arguments.
    Func {
        kind: FunctionKind,
        overloads: Vec<Overload>,
    },
    /// The methods are not names of their own: dispatch is not written yet.
    Trait {
        methods: Vec<Method<'c>>,
    },
    /// A `let` bound to a value; the inliner expands it where it is used.
    Let,
    /// A module this file named; what it declares is not in this scope.
    Module {
        path: &'c str,
    },
    /// A `let` the hoist has seen and the second walk has not reached. Its
    /// row is known only once its body is converted.
    Pending,
    /// A declaration in another file. A lookup follows it to where it was
    /// written.
    Import {
        from: Declared<'c>,
    },
}

impl Binding<'_> {
    /// Whether the declaration at a range is the one bound here, or one of
    /// its overloads.
    pub(super) fn is_declared_at(&self, text_range: TextRange) -> bool {
        match &self.kind {
            BindingKind::Func { overloads, .. } => overloads
                .iter()
                .any(|overload| overload.text_range == text_range),
            BindingKind::Struct { .. }
            | BindingKind::Relation { .. }
            | BindingKind::Trait { .. }
            | BindingKind::Let
            | BindingKind::Module { .. }
            | BindingKind::Pending
            | BindingKind::Import { .. } => self.text_range == text_range,
        }
    }
}

impl BindingKind<'_> {
    /// What the binding declares, without its data.
    pub(super) fn declaration_kind(&self) -> DeclarationKind {
        match self {
            BindingKind::Struct { .. } => DeclarationKind::Struct,
            BindingKind::Relation { .. } => DeclarationKind::Relation,
            BindingKind::Func { .. } => DeclarationKind::Function,
            BindingKind::Trait { .. } => DeclarationKind::Trait,
            BindingKind::Let | BindingKind::Pending => DeclarationKind::Binding,
            BindingKind::Module { .. } => DeclarationKind::Module,
            BindingKind::Import { .. } => DeclarationKind::Import,
        }
    }

    /// The word a diagnostic uses for it.
    pub(super) fn name(&self) -> &'static str {
        self.declaration_kind().name()
    }
}

impl fmt::Display for BindingKind<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// What a declaration is, for a diagnostic.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum DeclarationKind {
    Struct,
    Relation,
    Function,
    Trait,
    Binding,
    Module,
    Import,
}

impl DeclarationKind {
    pub(super) fn name(self) -> &'static str {
        match self {
            DeclarationKind::Struct => "struct",
            DeclarationKind::Relation => "relation",
            DeclarationKind::Function => "function",
            DeclarationKind::Trait => "trait",
            DeclarationKind::Binding => "binding",
            DeclarationKind::Module => "module",
            DeclarationKind::Import => "import",
        }
    }
}

impl fmt::Display for DeclarationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) struct Local<'c> {
    pub(super) name: &'c str,
    pub(super) slot: usize,
    /// The `let` or the parameter that declares it.
    pub(super) declared: TextRange,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum FunctionKind {
    Scalar,
    Aggregate,
}

impl FunctionKind {
    /// What a diagnostic calls a function of this kind.
    pub(super) fn name(self) -> &'static str {
        match self {
            FunctionKind::Scalar => "scalar function",
            FunctionKind::Aggregate => "aggregate function",
        }
    }
}

/// One declaration of a function name, told apart by its parameter count.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Overload {
    pub(super) source: CalleeSource,
    pub(super) arity: usize,
    pub(super) text_range: TextRange,
    pub(super) visibility: Visibility,
}

/// A method a trait declares. Methods may overload by parameter count.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Method<'c> {
    pub(super) name: &'c str,
    pub(super) arity: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Callable<'c> {
    pub(super) symbol: &'c str,
    pub(super) source: CalleeSource,
    pub(super) kind: FunctionKind,
    pub(super) target: Target<'c>,
}

impl<'c> Callable<'c> {
    pub(super) fn constant(symbol: &'c str, target: Target<'c>) -> Self {
        Self {
            symbol,
            source: CalleeSource::Const,
            kind: FunctionKind::Scalar,
            target,
        }
    }
}

/// The declaration a name resolved to, and the range of its syntax in the
/// file that declares it: for a function, the one overload the name picked.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Target<'c> {
    pub(super) at: Declared<'c>,
    pub(super) range: TextRange,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) enum Lookup<'c> {
    Column {
        index: usize,
        declared: Option<Span>,
    },
    Local {
        slot: usize,
        declared: TextRange,
    },
    Let(&'c str, Target<'c>),
    Ambiguous,
    NarrowedAway,
    NotAValue(DeclarationKind),
    /// A `let` further down the file: a name the file declares, not yet
    /// bound where it is read.
    NotYet,
    /// A name read in a relation an error left behind: not reported again.
    Lost,
    Unknown,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum ColumnLookup {
    Unique(usize),
    Ambiguous,
    Absent,
    /// The row an error left behind: any column may be in it.
    Lost,
}

enum Scope<'c> {
    /// The outermost block of a function body holds its parameters.
    Block { locals: Vec<Local<'c>> },
    /// The type parameters of the function being read. Names in the type
    /// namespace, so value lookups pass through it.
    TypeParams { names: Vec<&'c str> },
    /// `narrowed` is the names earlier stages stopped carrying, which get a
    /// diagnostic of their own.
    Relation {
        row: Row<'c>,
        narrowed: Vec<&'c str>,
    },
}

/// Which module a declaration lives in. The entry file has no path, since
/// nothing can import from it.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, Debug)]
pub(super) struct ModulePath<'c>(Option<&'c str>);

impl<'c> ModulePath<'c> {
    pub(super) fn entry() -> Self {
        Self(None)
    }

    pub(super) fn from_path(path: &'c str) -> Self {
        Self(Some(path))
    }

    pub(super) fn is_entry(&self) -> bool {
        self.0.is_none()
    }

    pub(super) fn declares(&self, name: &'c str) -> Declared<'c> {
        Declared {
            module: *self,
            name,
        }
    }
}

/// Where a declaration was written: which file, and the name written there.
/// A module declares a name once, so the pair is unique.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) struct Declared<'c> {
    pub(super) module: ModulePath<'c>,
    pub(super) name: &'c str,
}

/// What each module declares, by the name the source wrote.
type Modules<'c> = FxHashMap<ModulePath<'c>, FxHashMap<&'c str, Binding<'c>>>;

/// The names the library's files declare, bound once and read by every
/// compile that loads the library.
#[derive(Default, Debug)]
pub struct BoundLibrary<'l> {
    modules: Modules<'l>,
}

pub(super) struct SymbolTable<'c> {
    context: &'c Context,
    modules: Modules<'c>,
    library: Option<&'c BoundLibrary<'c>>,
    module: ModulePath<'c>,
    scopes: Vec<Scope<'c>>,
    /// The declarations a reference has named since the lowering last
    /// asked, so the library's can be lowered on demand.
    used: Vec<Use<'c>>,
}

/// A declaration a reference named, with the number of arguments when the
/// reference is a call, which picks one overload of a function.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Use<'c> {
    pub(super) at: Declared<'c>,
    pub(super) arity: Option<usize>,
}

impl<'c> SymbolTable<'c> {
    pub(super) fn new(context: &'c Context, library: Option<&'c BoundLibrary<'c>>) -> Self {
        Self {
            context,
            modules: FxHashMap::default(),
            library,
            module: ModulePath::entry(),
            scopes: Vec::new(),
            used: Vec::new(),
        }
    }

    /// A name as the context holds it. This is the one place the pass
    /// interns: a name read from the syntax, a module path, and a symbol
    /// the module qualifies all come through here.
    pub(super) fn intern(&self, text: &str) -> &'c str {
        StringAttribute::new(self.context, text).value()
    }

    /// A formatted name as the context holds it, formatted on the stack.
    pub(super) fn intern_fmt(&self, args: fmt::Arguments<'_>) -> &'c str {
        string::intern_fmt(self.context, args)
    }

    /// The name a declaration's op is built under. MLIR has one namespace
    /// for the whole program, so the module qualifies it. A qualified name
    /// is interned on each call.
    pub(super) fn symbol(&self, at: Declared<'c>) -> &'c str {
        match at.module.0 {
            Some(path) => self.intern_fmt(format_args!("{path}.{}", at.name)),
            None => at.name,
        }
    }

    /// The symbol of one declaration under a name that may be overloaded.
    /// MLIR has one symbol per name, so an overloaded one adds its
    /// parameter count; [`crate::written_name`] removes it again.
    pub(super) fn overload_symbol(
        &self,
        base: &'c str,
        arity: usize,
        is_overloaded: bool,
    ) -> &'c str {
        if is_overloaded {
            self.intern_fmt(format_args!("{base}.{arity}"))
        } else {
            base
        }
    }

    /// The symbol a function declared at a place is built under, for the
    /// overload with `arity` parameters.
    pub(super) fn function_symbol(&self, at: Declared<'c>, arity: usize) -> &'c str {
        let is_overloaded = matches!(
            self.find_in(at).map(|(_, binding)| &binding.kind),
            Some(BindingKind::Func { overloads, .. }) if overloads.len() > 1
        );
        self.overload_symbol(self.symbol(at), arity, is_overloaded)
    }

    /// The symbol a reference names, recorded as a use.
    fn refer(&mut self, at: Declared<'c>) -> &'c str {
        self.used.push(Use { at, arity: None });
        self.symbol(at)
    }

    /// The symbol a call names, recorded as a use of the overload that takes
    /// `arity` arguments.
    fn refer_call(&mut self, at: Declared<'c>, arity: usize) -> &'c str {
        self.used.push(Use {
            at,
            arity: Some(arity),
        });
        self.symbol(at)
    }

    /// Records a use of the library function that implements an operator,
    /// so the lowering brings it into the module. A program built without
    /// the library has none to use.
    pub(super) fn refer_operator(&mut self, operator: &Operator) {
        let at = Declared {
            module: ModulePath::from_path(self.intern(operator.module)),
            name: self.intern(operator.name),
        };
        if let Some((at, _)) = self.find_in(at) {
            self.used.push(Use { at, arity: None });
        }
    }

    /// The declarations references have named since the last call.
    pub(super) fn take_used(&mut self) -> Vec<Use<'c>> {
        std::mem::take(&mut self.used)
    }

    /// The symbol this file declares a name under.
    pub(super) fn symbol_here(&self, name: &'c str) -> &'c str {
        self.symbol(self.module.declares(name))
    }

    pub(super) fn module(&self) -> ModulePath<'c> {
        self.module
    }

    /// Makes `module` the current one, with no scope open.
    pub(super) fn enter_module(&mut self, module: ModulePath<'c>) {
        self.modules.entry(module).or_default();
        self.scopes.clear();
        self.module = module;
    }

    /// Whether a scope is still open: a block, a function's type
    /// parameters, or a relation.
    pub(super) fn has_open_scope(&self) -> bool {
        !self.scopes.is_empty()
    }

    pub(super) fn contains_module(&self, module: ModulePath<'c>) -> bool {
        self.modules.contains_key(&module)
            || self
                .library
                .is_some_and(|library| library.modules.contains_key(&module))
    }

    /// Whether the bound library already holds a module's names.
    pub(super) fn is_library_module(&self, module: ModulePath<'c>) -> bool {
        self.library
            .is_some_and(|library| library.modules.contains_key(&module))
    }

    /// The names bound so far, as a library later compiles read.
    pub(super) fn into_library(self) -> BoundLibrary<'c> {
        BoundLibrary {
            modules: self.modules,
        }
    }

    /// A name a module declares, with the name as the table holds it. This
    /// compile's own binding comes first: lowering a library `let` binds it
    /// again, over the pending one the library holds.
    fn declared_in(&self, module: ModulePath<'c>, name: &str) -> Option<(&'c str, &Binding<'c>)> {
        self.modules
            .get(&module)
            .and_then(|own| own.get_key_value(name))
            .or_else(|| self.library?.modules.get(&module)?.get_key_value(name))
            .map(|(&name, binding)| (name, binding))
    }

    pub(super) fn bind(&mut self, name: &'c str, binding: Binding<'c>) {
        self.modules
            .entry(self.module)
            .or_default()
            .insert(name, binding);
    }

    /// Adds an overload to a function this module declares. The name is
    /// public once any overload of it is.
    pub(super) fn add_overload(&mut self, name: &str, overload: Overload) {
        let binding = self
            .modules
            .get_mut(&self.module)
            .and_then(|declared| declared.get_mut(name))
            .unwrap_or_else(|| panic!("`{name}` is bound before an overload is added to it"));

        let BindingKind::Func { overloads, .. } = &mut binding.kind else {
            panic!("`{name}` is a {}, not a function", binding.kind.name());
        };

        overloads.push(overload);
        if overload.visibility == Visibility::Public {
            binding.visibility = Visibility::Public;
        }
    }

    /// What a module holds under a name, an import left as it is.
    pub(super) fn declared_binding(
        &self,
        module: ModulePath<'c>,
        name: &str,
    ) -> Option<&Binding<'c>> {
        self.declared_in(module, name).map(|(_, binding)| binding)
    }

    /// What this file holds under a name, an import left as it is.
    pub(super) fn binding(&self, name: &str) -> Option<&Binding<'c>> {
        self.declared_binding(self.module, name)
    }

    /// What a name in this file stands for, and where it was declared,
    /// following an import to the file that wrote it.
    /// A name the file does not declare or import is looked up among the
    /// prelude's public names.
    pub(super) fn find(&self, name: &str) -> Option<(Declared<'c>, &Binding<'c>)> {
        self.find_declared(self.module, name)
            .or_else(|| self.find_in_prelude(name))
    }

    fn find_in_prelude(&self, name: &str) -> Option<(Declared<'c>, &Binding<'c>)> {
        let prelude = ModulePath::from_path(PRELUDE);
        let (_, binding) = self.declared_in(prelude, name)?;
        if binding.visibility != Visibility::Public {
            return None;
        }

        self.find_declared(prelude, name)
    }

    /// A name a module declares, with the name as the table holds it.
    pub(super) fn find_declared(
        &self,
        module: ModulePath<'c>,
        name: &str,
    ) -> Option<(Declared<'c>, &Binding<'c>)> {
        let (name, _) = self.declared_in(module, name)?;
        self.find_in(module.declares(name))
    }

    /// Where a declaration was written, following imports. Each link points
    /// at a file read earlier, so a chain cannot come back around.
    pub(super) fn find_in(&self, at: Declared<'c>) -> Option<(Declared<'c>, &Binding<'c>)> {
        let (_, binding) = self.declared_in(at.module, at.name)?;
        match &binding.kind {
            BindingKind::Import { from } => self.find_in(*from),
            BindingKind::Struct { .. }
            | BindingKind::Relation { .. }
            | BindingKind::Func { .. }
            | BindingKind::Trait { .. }
            | BindingKind::Let
            | BindingKind::Module { .. }
            | BindingKind::Pending => Some((at, binding)),
        }
    }

    pub(super) fn is_method(&self, name: &str) -> bool {
        let library = self
            .library
            .and_then(|library| library.modules.get(&self.module));
        self.modules
            .get(&self.module)
            .into_iter()
            .chain(library)
            .flat_map(|declarations| declarations.keys())
            .any(|declared| {
                matches!(
                    self.find(declared).map(|(_, binding)| &binding.kind),
                    Some(BindingKind::Trait { methods })
                        if methods.iter().any(|method| method.name == name)
                )
            })
    }

    pub(super) fn kind(&self, name: &str) -> Option<&BindingKind<'c>> {
        self.find(name).map(|(_, binding)| &binding.kind)
    }

    pub(super) fn struct_symbol(&mut self, name: &str) -> Option<(&'c str, Target<'c>)> {
        let (at, binding) = self.find(name)?;
        if !matches!(binding.kind, BindingKind::Struct { .. }) {
            return None;
        }

        let target = Target {
            at,
            range: binding.text_range,
        };
        Some((self.refer(at), target))
    }

    pub(super) fn trait_symbol(&mut self, name: &str) -> Option<(&'c str, Target<'c>)> {
        let (at, binding) = self.find(name)?;
        if !matches!(binding.kind, BindingKind::Trait { .. }) {
            return None;
        }

        let target = Target {
            at,
            range: binding.text_range,
        };
        Some((self.refer(at), target))
    }

    pub(super) fn module_of(&self, name: &str) -> Option<&'c str> {
        match self.kind(name)? {
            BindingKind::Module { path } => Some(*path),
            BindingKind::Struct { .. }
            | BindingKind::Relation { .. }
            | BindingKind::Func { .. }
            | BindingKind::Trait { .. }
            | BindingKind::Let
            | BindingKind::Pending
            | BindingKind::Import { .. } => None,
        }
    }

    pub(super) fn relation(
        &mut self,
        name: &str,
        alias: Option<&'c str>,
    ) -> Option<(&'c str, Row<'c>, Target<'c>)> {
        let (at, binding) = self.find(name)?;
        let BindingKind::Relation { row } = &binding.kind else {
            return None;
        };

        let mut row = row.clone();
        if let Some(alias) = alias {
            row.qualify(alias);
        }

        let target = Target {
            at,
            range: binding.text_range,
        };
        Some((self.refer(at), row, target))
    }

    /// What a call names with `given` arguments: a `let`, or the overload
    /// of a function that takes that many.
    pub(super) fn callable(&mut self, name: &str, given: usize) -> Option<Callable<'c>> {
        let (at, binding) = self.find(name)?;
        if matches!(binding.kind, BindingKind::Let) {
            let target = Target {
                at,
                range: binding.text_range,
            };
            return Some(Callable::constant(self.refer(at), target));
        }

        self.callable_in(at, given)
    }

    /// The overload of the function declared at a place that takes `given`
    /// arguments. A private overload is seen only in its own module.
    pub(super) fn callable_in(&mut self, at: Declared<'c>, given: usize) -> Option<Callable<'c>> {
        let (at, binding) = self.find_in(at)?;
        let BindingKind::Func { kind, overloads } = &binding.kind else {
            return None;
        };

        let kind = *kind;
        let is_overloaded = overloads.len() > 1;
        let overload = *self
            .visible(at, overloads)
            .find(|overload| overload.arity == given)?;

        let target = Target {
            at,
            range: overload.text_range,
        };
        let symbol = self.refer_call(at, given);
        Some(Callable {
            symbol: self.overload_symbol(symbol, given, is_overloaded),
            source: overload.source,
            kind,
            target,
        })
    }

    /// The declaration a name in this file names, whatever it is: for a
    /// function, its first overload this module can call.
    pub(super) fn target_of(&self, name: &str) -> Option<Target<'c>> {
        let (at, _) = self.find(name)?;
        self.target_in(at)
    }

    /// The declaration at a place, followed through imports: for a
    /// function, its first overload this module can call.
    pub(super) fn target_in(&self, at: Declared<'c>) -> Option<Target<'c>> {
        let (at, binding) = self.find_in(at)?;
        let range = match &binding.kind {
            BindingKind::Func { overloads, .. } => self.visible(at, overloads).next()?.text_range,
            BindingKind::Struct { .. }
            | BindingKind::Relation { .. }
            | BindingKind::Trait { .. }
            | BindingKind::Let
            | BindingKind::Module { .. }
            | BindingKind::Pending
            | BindingKind::Import { .. } => binding.text_range,
        };
        Some(Target { at, range })
    }

    /// The argument counts a function's visible overloads take, least
    /// first, when a name is a function.
    pub(super) fn arities(&self, name: &str) -> Option<Vec<usize>> {
        let (at, _) = self.find(name)?;
        self.arities_in(at)
    }

    /// The argument counts the visible overloads of the function declared
    /// at a place take, least first.
    pub(super) fn arities_in(&self, at: Declared<'c>) -> Option<Vec<usize>> {
        let (at, binding) = self.find_in(at)?;
        let BindingKind::Func { overloads, .. } = &binding.kind else {
            return None;
        };

        let mut arities: Vec<usize> = self
            .visible(at, overloads)
            .map(|overload| overload.arity)
            .collect();
        arities.sort_unstable();
        Some(arities)
    }

    /// The overloads of a function declared at a place that this module
    /// can call.
    fn visible<'o>(
        &self,
        at: Declared<'c>,
        overloads: &'o [Overload],
    ) -> impl Iterator<Item = &'o Overload> + use<'o, 'c> {
        let is_home = at.module == self.module;
        overloads
            .iter()
            .filter(move |overload| is_home || overload.visibility == Visibility::Public)
    }

    // --- scopes ---

    pub(super) fn open_type_params(&mut self, names: Vec<&'c str>) {
        self.scopes.push(Scope::TypeParams { names });
    }

    /// A type parameter is not a value, so this walks past isolated scopes.
    pub(super) fn is_type_param(&self, name: &str) -> bool {
        self.scopes
            .iter()
            .rev()
            .any(|scope| matches!(scope, Scope::TypeParams { names } if names.contains(&name)))
    }

    /// A function body is open, so a statement is local to it.
    pub(super) fn is_in_body(&self) -> bool {
        self.scopes
            .iter()
            .any(|scope| matches!(scope, Scope::Block { .. }))
    }

    pub(super) fn open_block(&mut self) {
        self.scopes.push(Scope::Block { locals: Vec::new() });
    }

    /// Binding a name twice shadows it.
    pub(super) fn bind_local(&mut self, name: &'c str, slot: usize, declared: TextRange) {
        let Some(Scope::Block { locals }) = self.scopes.last_mut() else {
            panic!("a local is being bound outside a block")
        };

        locals.push(Local {
            name,
            slot,
            declared,
        });
    }

    pub(super) fn open_relation(&mut self, row: Row<'c>) {
        self.scopes.push(Scope::Relation {
            row,
            narrowed: Vec::new(),
        });
    }

    pub(super) fn close(&mut self) {
        self.scopes.pop();
    }

    /// Closes the relation a query opened, and hands back its row.
    pub(super) fn close_relation(&mut self) -> Row<'c> {
        match self.scopes.pop() {
            Some(Scope::Relation { row, .. }) => row,
            _ => panic!("a relation is closed outside a relation"),
        }
    }

    pub(super) fn row(&self) -> &Row<'c> {
        self.current_row()
            .expect("a stage is being lowered outside a relation")
    }

    pub(super) fn current_row(&self) -> Option<&Row<'c>> {
        match self.scopes.last() {
            Some(Scope::Relation { row, .. }) => Some(row),
            _ => None,
        }
    }

    // --- lookups ---

    /// The walk stops at the first isolated scope: a function body and a
    /// stage's region are both `IsolatedFromAbove`. The module's declarations
    /// are symbols, not values, so they answer from any depth.
    pub(super) fn lookup(&mut self, reference: Reference<'_>) -> Lookup<'c> {
        for scope in self.scopes.iter().rev() {
            match scope {
                Scope::TypeParams { .. } => {}
                Scope::Relation { row, narrowed } => {
                    match row.column(reference) {
                        ColumnLookup::Unique(index) => {
                            return Lookup::Column {
                                index,
                                declared: row.declared(index),
                            };
                        }
                        ColumnLookup::Ambiguous => return Lookup::Ambiguous,
                        ColumnLookup::Lost => return Lookup::Lost,
                        ColumnLookup::Absent if narrowed.contains(&reference.name) => {
                            return Lookup::NarrowedAway;
                        }
                        ColumnLookup::Absent => {}
                    }

                    break;
                }
                Scope::Block { locals } => {
                    if reference.qualifier.is_none()
                        && let Some(local) = locals
                            .iter()
                            .rev()
                            .find(|local| local.name == reference.name)
                    {
                        return Lookup::Local {
                            slot: local.slot,
                            declared: local.declared,
                        };
                    }
                }
            }
        }

        self.module_lookup(reference)
    }

    fn module_lookup(&mut self, reference: Reference<'_>) -> Lookup<'c> {
        if reference.qualifier.is_some() {
            return Lookup::Unknown;
        }

        let Some((at, binding)) = self.find(reference.name) else {
            return Lookup::Unknown;
        };

        match &binding.kind {
            BindingKind::Let => {}
            BindingKind::Pending => return Lookup::NotYet,
            kind @ (BindingKind::Struct { .. }
            | BindingKind::Relation { .. }
            | BindingKind::Func { .. }
            | BindingKind::Trait { .. }
            | BindingKind::Module { .. }
            | BindingKind::Import { .. }) => return Lookup::NotAValue(kind.declaration_kind()),
        }

        let target = Target {
            at,
            range: binding.text_range,
        };
        Lookup::Let(self.refer(at), target)
    }

    pub(super) fn column(&self, reference: Reference<'_>) -> ColumnLookup {
        self.row().column(reference)
    }

    // --- what each stage does to the row ---

    fn replace_row(&mut self, next: Row<'c>) {
        let (row, narrowed) = self.relation_mut();
        for name in row.names() {
            if !next.has_column(Reference::unqualified(name)) && !narrowed.contains(&name) {
                narrowed.push(name);
            }
        }

        *row = next;
    }

    fn relation_mut(&mut self) -> (&mut Row<'c>, &mut Vec<&'c str>) {
        match self.scopes.last_mut() {
            Some(Scope::Relation { row, narrowed }) => (row, narrowed),
            _ => panic!("a stage is being lowered outside a relation"),
        }
    }

    fn row_mut(&mut self) -> &mut Row<'c> {
        self.relation_mut().0
    }

    /// Records a name the row no longer carries.
    fn narrow(&mut self, name: &'c str) {
        let (row, narrowed) = self.relation_mut();
        if !row.has_column(Reference::unqualified(name)) && !narrowed.contains(&name) {
            narrowed.push(name);
        }
    }

    pub(super) fn alias(&mut self, alias: &'c str) {
        self.row_mut().qualify(alias);
    }

    pub(super) fn replace(&mut self, fields: Vec<Field<'c>>) {
        self.replace_row(Row::from(fields));
    }

    /// Adds columns; every column already there stays, so none is narrowed
    /// away.
    pub(super) fn extend(&mut self, fields: Vec<Field<'c>>) {
        self.row_mut().append(Row::from(fields));
    }

    pub(super) fn remove(&mut self, index: usize) {
        let name = self.row_mut().remove(index);
        self.narrow(name);
    }

    pub(super) fn rename(&mut self, renames: &[(usize, Field<'c>)]) {
        let old: Vec<&'c str> = renames
            .iter()
            .map(|&(index, to)| self.row_mut().rename(index, to))
            .collect();
        for name in old {
            self.narrow(name);
        }
    }

    pub(super) fn concat(&mut self, rhs: Row<'c>) {
        self.row_mut().append(rhs);
    }
}

#[cfg(test)]
mod tests {
    use text_size::TextRange;
    use yuzu_ast::ast::Visibility;
    use yuzu_mlir::attributes::CalleeSource;

    use super::{
        Binding, BindingKind, ColumnLookup, DeclarationKind, Field, FunctionKind, Lookup, Method,
        ModulePath, Overload, Reference, Row, SymbolTable,
    };

    fn named(names: &[&'static str]) -> Vec<Field<'static>> {
        names
            .iter()
            .map(|&name| Field {
                name,
                declared: None,
            })
            .collect()
    }

    fn table() -> SymbolTable<'static> {
        SymbolTable::new(Box::leak(Box::new(yuzu_mlir::context())), None)
    }

    fn symbols() -> SymbolTable<'static> {
        let mut symbols = table();
        symbols.bind(
            "Row",
            Binding {
                kind: BindingKind::Struct {
                    fields: named(&["id", "dept_id"]),
                },
                text_range: TextRange::default(),
                visibility: Visibility::Private,
            },
        );
        bind_relation(&mut symbols, "t", &["id", "dept_id"]);
        symbols
    }

    fn bind_relation(symbols: &mut SymbolTable<'static>, name: &'static str, row: &[&'static str]) {
        symbols.bind(
            name,
            Binding {
                kind: BindingKind::Relation {
                    row: Row::from(named(row)),
                },
                text_range: TextRange::default(),
                visibility: Visibility::Private,
            },
        );
    }

    fn bind_let(symbols: &mut SymbolTable<'static>, name: &'static str) {
        symbols.bind(
            name,
            Binding {
                kind: BindingKind::Let,
                text_range: TextRange::default(),
                visibility: Visibility::Private,
            },
        );
    }

    fn bind_fn(symbols: &mut SymbolTable<'static>, name: &'static str, arities: &[usize]) {
        let overload = |arity| Overload {
            source: CalleeSource::Fn,
            arity,
            text_range: TextRange::default(),
            visibility: Visibility::Private,
        };
        symbols.bind(
            name,
            Binding {
                kind: BindingKind::Func {
                    kind: FunctionKind::Scalar,
                    overloads: arities.iter().copied().map(overload).collect(),
                },
                text_range: TextRange::default(),
                visibility: Visibility::Private,
            },
        );
    }

    fn open_relation(symbols: &mut SymbolTable<'static>, relation: &'static str) {
        let (_, row, _) = symbols
            .relation(relation, None)
            .expect("the relation is declared");
        symbols.open_relation(row);
    }

    fn bare(name: &str) -> Reference<'_> {
        Reference::unqualified(name)
    }

    fn qualified<'c>(qualifier: &'c str, name: &'c str) -> Reference<'c> {
        Reference {
            qualifier: Some(qualifier),
            name,
        }
    }

    #[test]
    fn a_column_resolves_by_position() {
        let mut symbols = symbols();
        open_relation(&mut symbols, "t");

        assert_eq!(
            symbols.lookup(bare("dept_id")),
            Lookup::Column {
                index: 1,
                declared: None
            }
        );
        assert_eq!(symbols.lookup(bare("nope")), Lookup::Unknown);
    }

    #[test]
    fn a_qualifier_picks_between_same_named_columns() {
        let mut symbols = symbols();
        bind_relation(&mut symbols, "depts", &["id"]);
        open_relation(&mut symbols, "t");
        symbols.alias("a");
        let (_, rhs, _) = symbols
            .relation("depts", Some("d"))
            .expect("depts is bound");
        symbols.concat(rhs);

        assert_eq!(symbols.lookup(bare("id")), Lookup::Ambiguous);
        assert_eq!(
            symbols.lookup(qualified("a", "id")),
            Lookup::Column {
                index: 0,
                declared: None
            }
        );
        assert_eq!(
            symbols.lookup(qualified("d", "id")),
            Lookup::Column {
                index: 2,
                declared: None
            }
        );
    }

    #[test]
    fn a_narrowed_column_is_not_an_unknown_name() {
        let mut symbols = symbols();
        open_relation(&mut symbols, "t");
        symbols.remove(0);

        assert_eq!(symbols.lookup(bare("id")), Lookup::NarrowedAway);
        assert_eq!(
            symbols.lookup(bare("dept_id")),
            Lookup::Column {
                index: 0,
                declared: None
            }
        );
    }

    #[test]
    fn the_innermost_scope_holding_a_name_decides_it() {
        let mut symbols = symbols();
        bind_let(&mut symbols, "cap");
        bind_let(&mut symbols, "id");

        assert!(matches!(symbols.lookup(bare("cap")), Lookup::Let("cap", _)));
        symbols.open_block();
        symbols.bind_local("cap", 0, TextRange::default());
        assert_eq!(
            symbols.lookup(bare("cap")),
            Lookup::Local {
                slot: 0,
                declared: TextRange::default()
            }
        );
        symbols.open_block();
        symbols.bind_local("cap", 7, TextRange::default());
        assert_eq!(
            symbols.lookup(bare("cap")),
            Lookup::Local {
                slot: 7,
                declared: TextRange::default()
            }
        );
        symbols.close();
        assert_eq!(
            symbols.lookup(bare("cap")),
            Lookup::Local {
                slot: 0,
                declared: TextRange::default()
            }
        );
        symbols.close();
        assert!(matches!(symbols.lookup(bare("cap")), Lookup::Let("cap", _)));
        open_relation(&mut symbols, "t");
        assert_eq!(
            symbols.lookup(bare("id")),
            Lookup::Column {
                index: 0,
                declared: None
            }
        );
    }

    #[test]
    fn an_isolated_scope_reaches_the_module_and_nothing_between() {
        let mut symbols = symbols();
        bind_let(&mut symbols, "cap");
        symbols.open_block();
        symbols.bind_local("x", 0, TextRange::default());
        symbols.open_block();
        symbols.bind_local("local", 1, TextRange::default());
        open_relation(&mut symbols, "t");

        assert_eq!(
            symbols.lookup(bare("id")),
            Lookup::Column {
                index: 0,
                declared: None
            }
        );
        assert_eq!(symbols.lookup(bare("x")), Lookup::Unknown);
        assert_eq!(symbols.lookup(bare("local")), Lookup::Unknown);
        assert!(matches!(symbols.lookup(bare("cap")), Lookup::Let("cap", _)));
    }

    #[test]
    fn a_declaration_that_is_not_a_value_says_so() {
        let mut symbols = symbols();

        assert_eq!(
            symbols.lookup(bare("t")),
            Lookup::NotAValue(DeclarationKind::Relation)
        );
        assert_eq!(
            symbols.lookup(bare("Row")),
            Lookup::NotAValue(DeclarationKind::Struct)
        );
    }

    #[test]
    fn a_symbol_is_qualified_by_the_module_that_declared_it() {
        let mut symbols = table();
        symbols.enter_module(ModulePath::from_path("helpers"));
        symbols.bind(
            "Row",
            Binding {
                kind: BindingKind::Struct {
                    fields: named(&["a"]),
                },
                text_range: TextRange::default(),
                visibility: Visibility::Private,
            },
        );
        symbols.bind(
            "Show",
            Binding {
                kind: BindingKind::Trait {
                    methods: vec![Method {
                        name: "show",
                        arity: 1,
                    }],
                },
                text_range: TextRange::default(),
                visibility: Visibility::Private,
            },
        );
        bind_fn(&mut symbols, "f", &[0]);

        assert_eq!(
            symbols.struct_symbol("Row").map(|(symbol, _)| symbol),
            Some("helpers.Row")
        );
        assert_eq!(
            symbols.trait_symbol("Show").map(|(symbol, _)| symbol),
            Some("helpers.Show")
        );
        assert_eq!(
            symbols.callable("f", 0).map(|callable| callable.symbol),
            Some("helpers.f")
        );
        assert_eq!(symbols.trait_symbol("Row"), None);
        assert_eq!(symbols.struct_symbol("Show"), None);
    }

    #[test]
    fn callables() {
        let mut symbols = symbols();
        bind_let(&mut symbols, "cap");
        bind_fn(&mut symbols, "f", &[2]);
        symbols.bind(
            "Zero",
            Binding {
                kind: BindingKind::Trait {
                    methods: vec![Method {
                        name: "zero",
                        arity: 0,
                    }],
                },
                text_range: TextRange::default(),
                visibility: Visibility::Private,
            },
        );

        assert_eq!(
            symbols.callable("cap", 0).map(|callable| callable.source),
            Some(CalleeSource::Const)
        );
        let callable = symbols.callable("f", 2).expect("`f` takes two");
        assert_eq!(
            (callable.symbol, callable.source, callable.kind),
            ("f", CalleeSource::Fn, FunctionKind::Scalar)
        );
        assert_eq!(symbols.callable("f", 1), None);
        assert_eq!(symbols.arities("f"), Some(vec![2]));

        assert_eq!(symbols.callable("zero", 0), None);
        assert!(symbols.is_method("zero"));
        assert_eq!(symbols.callable("Zero", 0), None);
        assert!(!symbols.is_method("Zero"));
    }

    #[test]
    fn a_call_picks_the_overload_that_takes_its_arguments() {
        let mut symbols = table();
        symbols.enter_module(ModulePath::from_path("stats"));
        bind_fn(&mut symbols, "spread", &[2, 1]);

        let symbol = |symbols: &mut SymbolTable<'static>, given| {
            symbols
                .callable("spread", given)
                .map(|callable| callable.symbol)
        };
        assert_eq!(symbol(&mut symbols, 1), Some("stats.spread.1"));
        assert_eq!(symbol(&mut symbols, 2), Some("stats.spread.2"));
        assert_eq!(symbol(&mut symbols, 0), None);
        assert_eq!(symbols.arities("spread"), Some(vec![1, 2]));
        assert_eq!(crate::written_name("stats.spread.2"), "spread");
    }

    #[test]
    fn replacing_the_row_narrows_the_names_it_drops() {
        let mut symbols = symbols();
        open_relation(&mut symbols, "t");
        assert_eq!(symbols.column(bare("dept_id")), ColumnLookup::Unique(1));
        symbols.replace(named(&["dept_id", "n"]));

        assert_eq!(
            symbols.lookup(bare("dept_id")),
            Lookup::Column {
                index: 0,
                declared: None
            }
        );
        assert_eq!(
            symbols.lookup(bare("n")),
            Lookup::Column {
                index: 1,
                declared: None
            }
        );
        assert_eq!(symbols.lookup(bare("id")), Lookup::NarrowedAway);
    }

    #[test]
    fn join_concatenates_both_rows() {
        let mut symbols = symbols();
        bind_relation(&mut symbols, "depts", &["dept_id"]);
        open_relation(&mut symbols, "t");
        let (_, rhs, _) = symbols.relation("depts", None).expect("depts is bound");
        symbols.concat(rhs);

        assert_eq!(
            symbols.row().names().collect::<Vec<_>>(),
            ["id", "dept_id", "dept_id"]
        );
        assert_eq!(symbols.column(bare("dept_id")), ColumnLookup::Ambiguous);
    }
}
