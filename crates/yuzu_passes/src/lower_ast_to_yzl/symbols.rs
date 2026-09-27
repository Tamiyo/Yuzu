//! What a name means where it is written: every declaration in the
//! program, the scopes open inside a file, and the lookups the walk asks.
//!
//! A name is held as the source wrote it. Only a symbol, the name an op is
//! built under, is interned, since that is the only name MLIR ever sees.

use std::fmt;

use melior::Context;
use melior::ir::attribute::StringAttribute;
use rustc_hash::FxHashMap;
use text_size::TextRange;
use yuzu_ast::Visibility;
use yuzu_mlir::attributes::CalleeSource;
use yuzu_types::FunctionRegistry;

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
}

impl<'c> Row<'c> {
    pub(super) fn new() -> Self {
        Self::default()
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

        match (matches.next(), matches.next()) {
            (Some((index, _)), None) => ColumnLookup::Unique(index),
            (Some(_), Some(_)) => ColumnLookup::Ambiguous,
            (None, _) => ColumnLookup::Absent,
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
    Func {
        source: CalleeSource,
        kind: FunctionKind,
        arity: usize,
    },
    /// The methods are not names of their own: dispatch is not written yet.
    Trait {
        methods: Vec<&'c str>,
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

impl BindingKind<'_> {
    /// The word a diagnostic uses for it.
    pub(super) fn name(&self) -> &'static str {
        match self {
            BindingKind::Struct { .. } => "struct",
            BindingKind::Relation { .. } => "relation",
            BindingKind::Func { .. } => "function",
            BindingKind::Trait { .. } => "trait",
            BindingKind::Let => "binding",
            BindingKind::Module { .. } => "module",
            BindingKind::Pending => "binding",
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

#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) struct Callable<'c> {
    pub(super) symbol: &'c str,
    pub(super) source: CalleeSource,
    pub(super) kind: FunctionKind,
    pub(super) min_args: usize,
    pub(super) max_args: usize,
}

impl<'c> Callable<'c> {
    pub(super) fn constant(symbol: &'c str) -> Self {
        Self {
            symbol,
            source: CalleeSource::Const,
            kind: FunctionKind::Scalar,
            min_args: 0,
            max_args: 0,
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
    Unknown,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum ColumnLookup {
    Unique(usize),
    Ambiguous,
    Absent,
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

pub(super) struct SymbolTable<'c> {
    context: &'c Context,
    /// What each module declares, by the name the source wrote.
    modules: FxHashMap<ModulePath<'c>, FxHashMap<&'c str, Binding<'c>>>,
    module: ModulePath<'c>,
    scopes: Vec<Scope<'c>>,
    /// The declarations a reference has named since the lowering last
    /// asked, so the library's can be lowered on demand.
    used: Vec<Declared<'c>>,
}

impl<'c> SymbolTable<'c> {
    pub(super) fn new(context: &'c Context) -> Self {
        Self {
            context,
            modules: FxHashMap::default(),
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

    /// The symbol a reference names, recorded as a use.
    fn refer(&mut self, at: Declared<'c>) -> &'c str {
        self.used.push(at);
        self.symbol(at)
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

    /// An import names a declaration in a file read earlier, so what it
    /// names is known by the time this file is walked, and the binding
    /// itself stands in the import's place from here on.
    pub(super) fn set_module(&mut self, module: ModulePath<'c>) {
        self.modules.entry(module).or_default();
        self.scopes.clear();
        self.module = module;
    }

    pub(super) fn contains_module(&self, module: ModulePath<'c>) -> bool {
        self.modules.contains_key(&module)
    }

    pub(super) fn bind(&mut self, name: &'c str, binding: Binding<'c>) {
        self.modules
            .entry(self.module)
            .or_default()
            .insert(name, binding);
    }

    /// What this file holds under a name, an import left as it is.
    pub(super) fn binding(&self, name: &str) -> Option<&Binding<'c>> {
        self.modules.get(&self.module)?.get(name)
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
        let binding = self.modules.get(&prelude)?.get(name)?;
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
        let (&name, _) = self.modules.get(&module)?.get_key_value(name)?;
        self.find_in(module.declares(name))
    }

    /// Where a declaration was written, following imports. Each link points
    /// at a file read earlier, so a chain cannot come back around.
    pub(super) fn find_in(&self, at: Declared<'c>) -> Option<(Declared<'c>, &Binding<'c>)> {
        let binding = self.modules.get(&at.module)?.get(at.name)?;
        match &binding.kind {
            BindingKind::Import { from } => self.find_in(*from),
            _ => Some((at, binding)),
        }
    }

    pub(super) fn is_method(&self, name: &str) -> bool {
        self.modules.get(&self.module).is_some_and(|declarations| {
            declarations.keys().any(|declared| {
                matches!(
                    self.find(declared).map(|(_, binding)| &binding.kind),
                    Some(BindingKind::Trait { methods }) if methods.contains(&name)
                )
            })
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

    pub(super) fn callable(
        &mut self,
        name: &str,
        registry: &dyn FunctionRegistry,
    ) -> Option<Callable<'c>> {
        if let Some((at, binding)) = self.find(name)
            && matches!(binding.kind, BindingKind::Let)
        {
            return Some(Callable::constant(self.refer(at)));
        }

        self.operator(name, registry)
    }

    /// The function declared at a place in another module, as a call names it.
    pub(super) fn callable_in(&mut self, at: Declared<'c>) -> Option<Callable<'c>> {
        let (at, binding) = self.find_in(at)?;
        let BindingKind::Func {
            source,
            kind,
            arity,
        } = binding.kind
        else {
            return None;
        };

        Some(Callable {
            symbol: self.refer(at),
            source,
            kind,
            min_args: arity,
            max_args: arity,
        })
    }

    /// Like `callable`, except a `let` sharing the name does not stand in.
    pub(super) fn operator(
        &mut self,
        name: &str,
        registry: &dyn FunctionRegistry,
    ) -> Option<Callable<'c>> {
        if let Some((at, binding)) = self.find(name)
            && let BindingKind::Func {
                source,
                kind,
                arity,
            } = binding.kind
        {
            return Some(Callable {
                symbol: self.refer(at),
                source,
                kind,
                min_args: arity,
                max_args: arity,
            });
        }

        self.builtin(name, registry)
    }

    fn builtin(&self, name: &str, registry: &dyn FunctionRegistry) -> Option<Callable<'c>> {
        let entry = registry.entries().iter().find(|entry| entry.name == name)?;
        Some(Callable {
            symbol: entry.name,
            source: CalleeSource::Builtin,
            kind: match entry.func {
                yuzu_types::BuiltinFunc::Scalar(_) => FunctionKind::Scalar,
                yuzu_types::BuiltinFunc::Aggregate(_) => FunctionKind::Aggregate,
            },
            min_args: entry.min_args,
            max_args: entry.max_args,
        })
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

    pub(super) fn alias(&mut self, alias: &'c str) {
        let Some(Scope::Relation { row, .. }) = self.scopes.last_mut() else {
            panic!("a stage is being lowered outside a relation")
        };

        row.qualify(alias);
    }

    pub(super) fn replace(&mut self, names: Vec<&'c str>) {
        self.replace_row(Row::from(names));
    }

    pub(super) fn extend(&mut self, names: Vec<&'c str>) {
        let mut next = self.row().clone();
        next.append(Row::from(names));
        self.replace_row(next);
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
        let mut next = self.row().clone();
        next.append(rhs);
        self.replace_row(next);
    }
}

#[cfg(test)]
mod tests {
    use text_size::TextRange;
    use yuzu_ast::Visibility;
    use yuzu_mlir::attributes::CalleeSource;

    use super::{
        Binding, BindingKind, Callable, ColumnLookup, FunctionKind, Lookup, ModulePath, Reference,
        Row, SymbolTable,
    };

    fn table() -> SymbolTable<'static> {
        SymbolTable::new(Box::leak(Box::new(yuzu_mlir::context())))
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
        symbols.set_module(ModulePath::from_path("helpers"));
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
                    methods: vec!["show"],
                },
                text_range: TextRange::default(),
                visibility: Visibility::Private,
            },
        );
        symbols.bind(
            "f",
            Binding {
                kind: BindingKind::Func {
                    source: CalleeSource::Fn,
                    kind: FunctionKind::Scalar,
                    arity: 0,
                },
                text_range: TextRange::default(),
                visibility: Visibility::Private,
            },
        );

        assert_eq!(symbols.struct_symbol("Row"), Some("helpers.Row"));
        assert_eq!(symbols.trait_symbol("Show"), Some("helpers.Show"));
        assert_eq!(
            symbols
                .callable("f", &yuzu_types::Builtins)
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
        symbols.bind(
            "f",
            Binding {
                kind: BindingKind::Func {
                    source: CalleeSource::Fn,
                    kind: FunctionKind::Scalar,
                    arity: 2,
                },
                text_range: TextRange::default(),
                visibility: Visibility::Private,
            },
        );
        symbols.bind(
            "Zero",
            Binding {
                kind: BindingKind::Trait {
                    methods: vec!["zero"],
                },
                text_range: TextRange::default(),
                visibility: Visibility::Private,
            },
        );

        let registry = &yuzu_types::Builtins;
        assert_eq!(
            symbols
                .callable("cap", registry)
                .map(|callable| callable.source),
            Some(CalleeSource::Const)
        );
        assert_eq!(
            symbols.callable("f", registry),
            Some(Callable {
                symbol: "f",
                source: CalleeSource::Fn,
                kind: FunctionKind::Scalar,
                min_args: 2,
                max_args: 2,
            })
        );
        assert_eq!(
            symbols
                .callable("sum", registry)
                .map(|callable| callable.source),
            Some(CalleeSource::Builtin)
        );

        assert_eq!(symbols.callable("zero", registry), None);
        assert!(symbols.is_method("zero"));
        assert_eq!(symbols.callable("Zero", registry), None);
        assert!(!symbols.is_method("Zero"));
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
