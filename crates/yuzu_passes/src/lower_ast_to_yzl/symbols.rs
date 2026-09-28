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
use yuzu_ast::Visibility;
use yuzu_mlir::attributes::CalleeSource;
use yuzu_types::FunctionRegistry;

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
            columns: Vec::new(),
            schema: Schema::Lost,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.columns.len()
    }

    pub(super) fn names(&self) -> impl Iterator<Item = &'c str> + use<'_, 'c> {
        self.columns.iter().map(|column| column.name)
    }

    pub(super) fn references(&self) -> impl Iterator<Item = Reference<'c>> + use<'_, 'c> {
        self.columns.iter().map(Column::reference)
    }

    pub(super) fn column(&self, reference: Reference<'_>) -> ColumnLookup {
        let mut matches = self
            .columns
            .iter()
            .enumerate()
            .filter(|(_, column)| column.matches(reference));

        match (matches.next(), matches.next(), self.schema) {
            (Some((index, _)), None, _) => ColumnLookup::Unique(index),
            (Some(_), Some(_), _) => ColumnLookup::Ambiguous,
            (None, _, Schema::Known) => ColumnLookup::Absent,
            (None, _, Schema::Lost) => ColumnLookup::Lost,
        }
    }

    pub(super) fn has(&self, reference: Reference<'_>) -> bool {
        !matches!(self.column(reference), ColumnLookup::Absent)
    }

    pub(super) fn qualify(&mut self, alias: &'c str) {
        for column in &mut self.columns {
            column.qualifier = Some(alias);
        }
    }

    pub(super) fn rename(&mut self, index: usize, name: &'c str) {
        self.columns[index].name = name;
    }

    pub(super) fn remove(&mut self, index: usize) {
        self.columns.remove(index);
    }

    pub(super) fn append(&mut self, other: Row<'c>) {
        self.columns.extend(other.columns);
        if other.schema == Schema::Lost {
            self.schema = Schema::Lost;
        }
    }
}

impl<'c> From<Vec<&'c str>> for Row<'c> {
    fn from(names: Vec<&'c str>) -> Self {
        Self {
            columns: names
                .into_iter()
                .map(|name| Column {
                    qualifier: None,
                    name,
                })
                .collect(),
            schema: Schema::Known,
        }
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
        fields: Vec<&'c str>,
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
            _ => self.text_range == text_range,
        }
    }
}

impl BindingKind<'_> {
    /// The word a diagnostic uses for it.
    pub(super) fn name(&self) -> &'static str {
        match self {
            BindingKind::Struct { .. } => "struct",
            BindingKind::Relation { .. } => "relation",
            BindingKind::Func { .. } => "function",
            BindingKind::Trait { .. } => "trait",
            BindingKind::Let | BindingKind::Pending => "binding",
            BindingKind::Module { .. } => "module",
            BindingKind::Import { .. } => {
                unreachable!("a lookup follows an import before anything asks its name")
            }
        }
    }
}

impl fmt::Display for BindingKind<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) struct Local<'c> {
    pub(super) name: &'c str,
    pub(super) slot: usize,
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
}

impl<'c> Callable<'c> {
    pub(super) fn constant(symbol: &'c str) -> Self {
        Self {
            symbol,
            source: CalleeSource::Const,
            kind: FunctionKind::Scalar,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) enum Lookup<'c> {
    Column(usize),
    Local(usize),
    Let(&'c str),
    Ambiguous,
    NarrowedAway,
    NotAValue(&'static str),
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
    used: Vec<Declared<'c>>,
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

    /// The name a declaration's op is built under. MLIR has one namespace
    /// for the whole program, so the module qualifies it.
    pub(super) fn symbol(&self, at: Declared<'c>) -> &'c str {
        match at.module.0 {
            Some(path) => self.intern(&format!("{path}.{}", at.name)),
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
            self.intern(&format!("{base}.{arity}"))
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
        self.used.push(at);
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
            self.used.push(at);
        }
    }

    /// The declarations references have named since the last call.
    pub(super) fn take_used(&mut self) -> Vec<Declared<'c>> {
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

    /// What this file holds under a name, an import left as it is.
    pub(super) fn binding(&self, name: &str) -> Option<&Binding<'c>> {
        self.declared_in(self.module, name)
            .map(|(_, binding)| binding)
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
            _ => Some((at, binding)),
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

    pub(super) fn struct_symbol(&mut self, name: &str) -> Option<&'c str> {
        let (at, binding) = self.find(name)?;
        if !matches!(binding.kind, BindingKind::Struct { .. }) {
            return None;
        }

        Some(self.refer(at))
    }

    pub(super) fn trait_symbol(&mut self, name: &str) -> Option<&'c str> {
        let (at, binding) = self.find(name)?;
        if !matches!(binding.kind, BindingKind::Trait { .. }) {
            return None;
        }

        Some(self.refer(at))
    }

    pub(super) fn module_of(&self, name: &str) -> Option<&'c str> {
        match self.kind(name)? {
            BindingKind::Module { path } => Some(*path),
            _ => None,
        }
    }

    pub(super) fn relation(
        &mut self,
        name: &str,
        alias: Option<&'c str>,
    ) -> Option<(&'c str, Row<'c>)> {
        let (at, binding) = self.find(name)?;
        let BindingKind::Relation { row } = &binding.kind else {
            return None;
        };

        let mut row = row.clone();
        if let Some(alias) = alias {
            row.qualify(alias);
        }

        Some((self.refer(at), row))
    }

    /// What a call names with `given` arguments: a `let`, the overload of
    /// a function that takes that many, or a builtin.
    pub(super) fn callable(
        &mut self,
        name: &str,
        given: usize,
        registry: &dyn FunctionRegistry,
    ) -> Option<Callable<'c>> {
        let Some((at, binding)) = self.find(name) else {
            return builtin(name, given, registry);
        };

        if matches!(binding.kind, BindingKind::Let) {
            return Some(Callable::constant(self.refer(at)));
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

        let symbol = self.refer(at);
        Some(Callable {
            symbol: self.overload_symbol(symbol, given, is_overloaded),
            source: overload.source,
            kind,
        })
    }

    /// The argument counts a function's visible overloads take, least
    /// first, when a name is a function.
    pub(super) fn arities(
        &self,
        name: &str,
        registry: &dyn FunctionRegistry,
    ) -> Option<Vec<usize>> {
        match self.find(name) {
            Some((at, _)) => self.arities_in(at),
            None => registry
                .entries()
                .iter()
                .find(|entry| entry.name == name)
                .map(|entry| (entry.min_args..=entry.max_args).collect()),
        }
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

    pub(super) fn enter_type_params(&mut self, names: Vec<&'c str>) {
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
    pub(super) fn in_body(&self) -> bool {
        self.scopes
            .iter()
            .any(|scope| matches!(scope, Scope::Block { .. }))
    }

    pub(super) fn enter_block(&mut self) {
        self.scopes.push(Scope::Block { locals: Vec::new() });
    }

    /// Binding a name twice shadows it.
    pub(super) fn bind_local(&mut self, name: &'c str, slot: usize) {
        let Some(Scope::Block { locals }) = self.scopes.last_mut() else {
            panic!("a local is being bound outside a block")
        };

        locals.push(Local { name, slot });
    }

    pub(super) fn enter_relation(&mut self, row: Row<'c>) {
        self.scopes.push(Scope::Relation {
            row,
            narrowed: Vec::new(),
        });
    }

    pub(super) fn leave(&mut self) {
        self.scopes.pop();
    }

    /// Leaves the relation a query opened, and hands back its row.
    pub(super) fn leave_relation(&mut self) -> Row<'c> {
        match self.scopes.pop() {
            Some(Scope::Relation { row, .. }) => row,
            _ => panic!("a query is left outside a relation"),
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
                        ColumnLookup::Unique(index) => return Lookup::Column(index),
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
                        return Lookup::Local(local.slot);
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
            kind => return Lookup::NotAValue(kind.name()),
        }

        Lookup::Let(self.refer(at))
    }

    pub(super) fn column(&self, reference: Reference<'_>) -> ColumnLookup {
        self.row().column(reference)
    }

    // --- what each stage does to the row ---

    fn replace_row(&mut self, next: Row<'c>) {
        let Some(Scope::Relation { row, narrowed }) = self.scopes.last_mut() else {
            panic!("a stage is being lowered outside a relation")
        };

        for name in row.names() {
            if !next.has(Reference::unqualified(name)) && !narrowed.contains(&name) {
                narrowed.push(name);
            }
        }

        *row = next;
    }

    fn row_mut(&mut self) -> &mut Row<'c> {
        match self.scopes.last_mut() {
            Some(Scope::Relation { row, .. }) => row,
            _ => panic!("a stage is being lowered outside a relation"),
        }
    }

    pub(super) fn alias(&mut self, alias: &'c str) {
        self.row_mut().qualify(alias);
    }

    pub(super) fn replace(&mut self, names: Vec<&'c str>) {
        self.replace_row(Row::from(names));
    }

    /// Adds columns; every column already there stays, so none is narrowed
    /// away.
    pub(super) fn extend(&mut self, names: Vec<&'c str>) {
        self.row_mut().append(Row::from(names));
    }

    pub(super) fn remove(&mut self, index: usize) {
        let mut next = self.row().clone();
        next.remove(index);
        self.replace_row(next);
    }

    pub(super) fn rename(&mut self, renames: &[(usize, &'c str)]) {
        let mut next = self.row().clone();
        for (index, to) in renames {
            next.rename(*index, to);
        }

        self.replace_row(next);
    }

    pub(super) fn concat(&mut self, rhs: Row<'c>) {
        self.row_mut().append(rhs);
    }
}

fn builtin<'c>(name: &str, given: usize, registry: &dyn FunctionRegistry) -> Option<Callable<'c>> {
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.name == name && (entry.min_args..=entry.max_args).contains(&given))?;
    Some(Callable {
        symbol: entry.name,
        source: CalleeSource::Builtin,
        kind: match entry.func {
            yuzu_types::BuiltinFunc::Scalar(_) => FunctionKind::Scalar,
            yuzu_types::BuiltinFunc::Aggregate(_) => FunctionKind::Aggregate,
        },
    })
}

#[cfg(test)]
mod tests {
    use text_size::TextRange;
    use yuzu_ast::Visibility;
    use yuzu_mlir::attributes::CalleeSource;

    use super::{
        Binding, BindingKind, Callable, ColumnLookup, FunctionKind, Lookup, Method, ModulePath,
        Overload, Reference, Row, SymbolTable,
    };

    fn table() -> SymbolTable<'static> {
        SymbolTable::new(Box::leak(Box::new(yuzu_mlir::context())), None)
    }

    fn symbols() -> SymbolTable<'static> {
        let mut symbols = table();
        symbols.bind(
            "Row",
            Binding {
                kind: BindingKind::Struct {
                    fields: vec!["id", "dept_id"],
                },
                text_range: TextRange::default(),
                visibility: Visibility::Private,
            },
        );
        bind_relation(&mut symbols, "t", vec!["id", "dept_id"]);
        symbols
    }

    fn bind_relation(
        symbols: &mut SymbolTable<'static>,
        name: &'static str,
        row: Vec<&'static str>,
    ) {
        symbols.bind(
            name,
            Binding {
                kind: BindingKind::Relation {
                    row: Row::from(row),
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

    fn enter_relation(symbols: &mut SymbolTable<'static>, relation: &'static str) {
        let (_, row) = symbols
            .relation(relation, None)
            .expect("the relation is declared");
        symbols.enter_relation(row);
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
        enter_relation(&mut symbols, "t");

        assert_eq!(symbols.lookup(bare("dept_id")), Lookup::Column(1));
        assert_eq!(symbols.lookup(bare("nope")), Lookup::Unknown);
    }

    #[test]
    fn a_qualifier_picks_between_same_named_columns() {
        let mut symbols = symbols();
        bind_relation(&mut symbols, "depts", vec!["id"]);
        enter_relation(&mut symbols, "t");
        symbols.alias("a");
        let (_, rhs) = symbols
            .relation("depts", Some("d"))
            .expect("depts is bound");
        symbols.concat(rhs);

        assert_eq!(symbols.lookup(bare("id")), Lookup::Ambiguous);
        assert_eq!(symbols.lookup(qualified("a", "id")), Lookup::Column(0));
        assert_eq!(symbols.lookup(qualified("d", "id")), Lookup::Column(2));
    }

    #[test]
    fn a_narrowed_column_is_not_an_unknown_name() {
        let mut symbols = symbols();
        enter_relation(&mut symbols, "t");
        symbols.remove(0);

        assert_eq!(symbols.lookup(bare("id")), Lookup::NarrowedAway);
        assert_eq!(symbols.lookup(bare("dept_id")), Lookup::Column(0));
    }

    #[test]
    fn the_innermost_scope_holding_a_name_decides_it() {
        let mut symbols = symbols();
        bind_let(&mut symbols, "cap");
        bind_let(&mut symbols, "id");

        assert_eq!(symbols.lookup(bare("cap")), Lookup::Let("cap"));
        symbols.enter_block();
        symbols.bind_local("cap", 0);
        assert_eq!(symbols.lookup(bare("cap")), Lookup::Local(0));
        symbols.enter_block();
        symbols.bind_local("cap", 7);
        assert_eq!(symbols.lookup(bare("cap")), Lookup::Local(7));
        symbols.leave();
        assert_eq!(symbols.lookup(bare("cap")), Lookup::Local(0));
        symbols.leave();
        assert_eq!(symbols.lookup(bare("cap")), Lookup::Let("cap"));
        enter_relation(&mut symbols, "t");
        assert_eq!(symbols.lookup(bare("id")), Lookup::Column(0));
    }

    #[test]
    fn an_isolated_scope_reaches_the_module_and_nothing_between() {
        let mut symbols = symbols();
        bind_let(&mut symbols, "cap");
        symbols.enter_block();
        symbols.bind_local("x", 0);
        symbols.enter_block();
        symbols.bind_local("local", 1);
        enter_relation(&mut symbols, "t");

        assert_eq!(symbols.lookup(bare("id")), Lookup::Column(0));
        assert_eq!(symbols.lookup(bare("x")), Lookup::Unknown);
        assert_eq!(symbols.lookup(bare("local")), Lookup::Unknown);
        assert_eq!(symbols.lookup(bare("cap")), Lookup::Let("cap"));
    }

    #[test]
    fn a_declaration_that_is_not_a_value_says_so() {
        let mut symbols = symbols();

        assert_eq!(symbols.lookup(bare("t")), Lookup::NotAValue("relation"));
        assert_eq!(symbols.lookup(bare("Row")), Lookup::NotAValue("struct"));
    }

    #[test]
    fn a_symbol_is_qualified_by_the_module_that_declared_it() {
        let mut symbols = table();
        symbols.enter_module(ModulePath::from_path("helpers"));
        symbols.bind(
            "Row",
            Binding {
                kind: BindingKind::Struct { fields: vec!["a"] },
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

        assert_eq!(symbols.struct_symbol("Row"), Some("helpers.Row"));
        assert_eq!(symbols.trait_symbol("Show"), Some("helpers.Show"));
        assert_eq!(
            symbols
                .callable("f", 0, &yuzu_types::Builtins)
                .map(|callable| callable.symbol),
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

        let registry = &yuzu_types::Builtins;
        assert_eq!(
            symbols
                .callable("cap", 0, registry)
                .map(|callable| callable.source),
            Some(CalleeSource::Const)
        );
        assert_eq!(
            symbols.callable("f", 2, registry),
            Some(Callable {
                symbol: "f",
                source: CalleeSource::Fn,
                kind: FunctionKind::Scalar,
            })
        );
        assert_eq!(symbols.callable("f", 1, registry), None);
        assert_eq!(symbols.arities("f", registry), Some(vec![2]));
        assert_eq!(
            symbols
                .callable("count", 1, registry)
                .map(|callable| callable.source),
            Some(CalleeSource::Builtin)
        );

        assert_eq!(symbols.callable("zero", 0, registry), None);
        assert!(symbols.is_method("zero"));
        assert_eq!(symbols.callable("Zero", 0, registry), None);
        assert!(!symbols.is_method("Zero"));
    }

    #[test]
    fn a_call_picks_the_overload_that_takes_its_arguments() {
        let mut symbols = table();
        symbols.enter_module(ModulePath::from_path("stats"));
        bind_fn(&mut symbols, "spread", &[2, 1]);
        let registry = &yuzu_types::Builtins;

        let symbol = |symbols: &mut SymbolTable<'static>, given| {
            symbols
                .callable("spread", given, registry)
                .map(|callable| callable.symbol)
        };
        assert_eq!(symbol(&mut symbols, 1), Some("stats.spread.1"));
        assert_eq!(symbol(&mut symbols, 2), Some("stats.spread.2"));
        assert_eq!(symbol(&mut symbols, 0), None);
        assert_eq!(symbols.arities("spread", registry), Some(vec![1, 2]));
        assert_eq!(crate::written_name("stats.spread.2"), "spread");
    }

    #[test]
    fn replacing_the_row_narrows_the_names_it_drops() {
        let mut symbols = symbols();
        enter_relation(&mut symbols, "t");
        assert_eq!(symbols.column(bare("dept_id")), ColumnLookup::Unique(1));
        symbols.replace(vec!["dept_id", "n"]);

        assert_eq!(symbols.lookup(bare("dept_id")), Lookup::Column(0));
        assert_eq!(symbols.lookup(bare("n")), Lookup::Column(1));
        assert_eq!(symbols.lookup(bare("id")), Lookup::NarrowedAway);
    }

    #[test]
    fn join_concatenates_both_rows() {
        let mut symbols = symbols();
        bind_relation(&mut symbols, "depts", vec!["dept_id"]);
        enter_relation(&mut symbols, "t");
        let (_, rhs) = symbols.relation("depts", None).expect("depts is bound");
        symbols.concat(rhs);

        assert_eq!(
            symbols.row().names().collect::<Vec<_>>(),
            ["id", "dept_id", "dept_id"]
        );
        assert_eq!(symbols.column(bare("dept_id")), ColumnLookup::Ambiguous);
    }
}
