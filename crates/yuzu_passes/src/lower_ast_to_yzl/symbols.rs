//! What a name means where it is written: every declaration in the
//! program, the scopes open inside a file, and the lookups the walk asks.

use std::collections::{HashMap, HashSet};
use std::fmt;

use text_size::TextRange;
use yuzu_ast::Visibility;
use yuzu_ast::ast::Mutability;
use yuzu_mlir::attributes::CalleeSource;
use yuzu_types::FunctionRegistry;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Reference<'c> {
    pub(super) qualifier: Option<&'c str>,
    pub(super) name: &'c str,
}

impl<'c> Reference<'c> {
    pub(super) fn bare(name: &'c str) -> Self {
        Self {
            qualifier: None,
            name,
        }
    }

    fn matches(self, reference: Reference<'_>) -> bool {
        self.name == reference.name
            && reference
                .qualifier
                .is_none_or(|qualifier| self.qualifier == Some(qualifier))
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

/// Two columns may share a name, since a join concatenates both sides, so a
/// column is addressed by position.
#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub(super) struct Row<'c> {
    columns: Vec<Reference<'c>>,
}

impl<'c> Row<'c> {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn len(&self) -> usize {
        self.columns.len()
    }

    pub(super) fn names(&self) -> impl Iterator<Item = &'c str> + '_ {
        self.columns.iter().map(|column| column.name)
    }

    pub(super) fn references(&self) -> impl Iterator<Item = Reference<'c>> + '_ {
        self.columns.iter().copied()
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
            columns: names.into_iter().map(Reference::bare).collect(),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) struct Binding<'c> {
    pub(super) kind: BindingKind<'c>,
    pub(super) declared: TextRange,
    pub(super) visibility: Visibility,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) enum BindingKind<'c> {
    Struct {
        fields: Vec<&'c str>,
        symbol: &'c str,
    },
    /// A table, or a `let` bound to a query: what `from` and `join` name.
    Relation {
        row: Row<'c>,
        symbol: &'c str,
    },
    Func(Callable<'c>),
    /// The methods are not names of their own: dispatch is not written yet.
    Trait {
        methods: Vec<&'c str>,
        symbol: &'c str,
    },
    /// A `let` bound to a value; the inliner expands it where it is used.
    Let {
        symbol: &'c str,
    },
    /// A module this file named; what it declares is not in this scope.
    Module {
        path: &'c str,
    },
}

impl<'c> BindingKind<'c> {
    pub(super) fn what(&self) -> &'static str {
        match self {
            BindingKind::Struct { .. } => "struct",
            BindingKind::Relation { .. } => "relation",
            BindingKind::Func(_) => "function",
            BindingKind::Trait { .. } => "trait",
            BindingKind::Let { .. } => "binding",
            BindingKind::Module { .. } => "module",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Local<'c> {
    pub(super) name: &'c str,
    pub(super) slot: usize,
    pub(super) mutability: Mutability,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum FunctionKind {
    Scalar,
    Aggregate,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Callable<'c> {
    pub(super) symbol: &'c str,
    pub(super) source: CalleeSource,
    pub(super) kind: FunctionKind,
    pub(super) min_args: usize,
    pub(super) max_args: usize,
}

impl<'c> Callable<'c> {
    pub(super) fn let_binding(symbol: &'c str) -> Self {
        Self {
            symbol,
            source: CalleeSource::Let,
            kind: FunctionKind::Scalar,
            min_args: 0,
            max_args: 0,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Lookup<'c> {
    Column(usize),
    Param(usize),
    Local { slot: usize, mutability: Mutability },
    Let(&'c str),
    Ambiguous,
    NarrowedAway,
    NotAValue(&'static str),
    Unknown,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum ColumnLookup {
    Unique(usize),
    Ambiguous,
    Absent,
}

enum Scope<'c> {
    Function {
        params: Vec<&'c str>,
    },
    Block {
        locals: Vec<Local<'c>>,
    },
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

    pub(super) fn of(path: &'c str) -> Self {
        Self(Some(path))
    }

    pub(super) fn is_entry(self) -> bool {
        self.0.is_none()
    }

    pub(super) fn qualify(self, name: &str) -> Option<String> {
        self.0.map(|path| format!("{path}.{name}"))
    }

    pub(super) fn declares(self, name: &'c str) -> Declared<'c> {
        Declared { module: self, name }
    }
}

/// Where a declaration was written: which file, and the name written there.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) struct Declared<'c> {
    pub(super) module: ModulePath<'c>,
    pub(super) name: &'c str,
}

pub(super) struct SymbolTable<'c> {
    /// Every name the program declares, for the whole run.
    declarations: HashMap<Declared<'c>, Binding<'c>>,
    modules: HashSet<ModulePath<'c>>,
    module: ModulePath<'c>,
    scopes: Vec<Scope<'c>>,
}

impl<'c> SymbolTable<'c> {
    pub(super) fn new() -> Self {
        Self {
            declarations: HashMap::new(),
            modules: HashSet::new(),
            module: ModulePath::entry(),
            scopes: Vec::new(),
        }
    }

    // --- modules ---

    pub(super) fn module(&self) -> ModulePath<'c> {
        self.module
    }

    pub(super) fn set_module(&mut self, module: ModulePath<'c>) {
        self.module = module;
        self.modules.insert(module);
        self.scopes.clear();
    }

    pub(super) fn contains_module(&self, module: ModulePath<'c>) -> bool {
        self.modules.contains(&module)
    }

    // --- what the program declares ---

    pub(super) fn bind(&mut self, name: &'c str, binding: Binding<'c>) {
        self.declarations
            .insert(self.module.declares(name), binding);
    }

    pub(super) fn binding(&self, name: &'c str) -> Option<&Binding<'c>> {
        self.declared(self.module.declares(name))
    }

    pub(super) fn declared(&self, at: Declared<'c>) -> Option<&Binding<'c>> {
        self.declarations.get(&at)
    }

    pub(super) fn is_method(&self, name: &str) -> bool {
        self.declarations.iter().any(|(at, binding)| {
            at.module == self.module
                && matches!(&binding.kind, BindingKind::Trait { methods, .. } if methods.contains(&name))
        })
    }

    pub(super) fn struct_symbol(&self, name: &'c str) -> Option<&'c str> {
        match self.kind(name)? {
            BindingKind::Struct { symbol, .. } => Some(symbol),
            _ => None,
        }
    }

    pub(super) fn module_of(&self, name: &'c str) -> Option<&'c str> {
        match self.kind(name)? {
            BindingKind::Module { path } => Some(path),
            _ => None,
        }
    }

    pub(super) fn trait_symbol(&self, name: &'c str) -> Option<&'c str> {
        match self.kind(name)? {
            BindingKind::Trait { symbol, .. } => Some(symbol),
            _ => None,
        }
    }

    pub(super) fn kind(&self, name: &'c str) -> Option<&BindingKind<'c>> {
        self.binding(name).map(|binding| &binding.kind)
    }

    pub(super) fn relation(
        &self,
        name: &'c str,
        alias: Option<&'c str>,
    ) -> Option<(&'c str, Row<'c>)> {
        let BindingKind::Relation { row, symbol } = self.kind(name)? else {
            return None;
        };

        let mut row = row.clone();
        if let Some(alias) = alias {
            row.qualify(alias);
        }

        Some((symbol, row))
    }

    pub(super) fn callable(
        &self,
        name: &'c str,
        registry: &dyn FunctionRegistry,
    ) -> Option<Callable<'c>> {
        match self.kind(name) {
            Some(BindingKind::Func(callable)) => Some(*callable),
            Some(BindingKind::Let { symbol }) => Some(Callable::let_binding(symbol)),
            _ => self.builtin(name, registry),
        }
    }

    /// Like `callable`, except a `let` sharing the name does not stand in.
    pub(super) fn operator(
        &self,
        name: &'c str,
        registry: &dyn FunctionRegistry,
    ) -> Option<Callable<'c>> {
        match self.kind(name) {
            Some(BindingKind::Func(callable)) => Some(*callable),
            _ => self.builtin(name, registry),
        }
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

    pub(super) fn enter_function(&mut self, params: Vec<&'c str>) {
        self.scopes.push(Scope::Function { params });
    }

    pub(super) fn enter_block(&mut self) {
        self.scopes.push(Scope::Block { locals: Vec::new() });
    }

    /// Binding a name twice shadows it: a second `let` and an assignment
    /// both do.
    pub(super) fn bind_local(&mut self, name: &'c str, slot: usize, mutability: Mutability) {
        let Some(Scope::Block { locals }) = self.scopes.last_mut() else {
            panic!("a local is being bound outside a block")
        };

        locals.push(Local {
            name,
            slot,
            mutability,
        });
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
    pub(super) fn lookup(&self, reference: Reference<'c>) -> Lookup<'c> {
        for scope in self.scopes.iter().rev() {
            match scope {
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
                        return Lookup::Local {
                            slot: local.slot,
                            mutability: local.mutability,
                        };
                    }
                }
                Scope::Function { params } => {
                    if reference.qualifier.is_none()
                        && let Some(index) =
                            params.iter().position(|&param| param == reference.name)
                    {
                        return Lookup::Param(index);
                    }

                    break;
                }
            }
        }

        self.module_lookup(reference)
    }

    fn module_lookup(&self, reference: Reference<'c>) -> Lookup<'c> {
        if reference.qualifier.is_some() {
            return Lookup::Unknown;
        }

        match self.kind(reference.name) {
            Some(BindingKind::Let { symbol }) => Lookup::Let(symbol),
            Some(kind) => Lookup::NotAValue(kind.what()),
            None => Lookup::Unknown,
        }
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
            if !next.has(Reference::bare(name)) && !narrowed.contains(&name) {
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
        for &(index, to) in renames {
            next.rename(index, to);
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
    use yuzu_ast::ast::Mutability;
    use yuzu_mlir::attributes::CalleeSource;

    use super::{
        Binding, BindingKind, Callable, ColumnLookup, FunctionKind, Lookup, Reference, Row,
        SymbolTable,
    };

    fn symbols() -> SymbolTable<'static> {
        let mut symbols = SymbolTable::new();
        symbols.bind(
            "Row",
            Binding {
                kind: BindingKind::Struct {
                    fields: vec!["id", "dept_id"],
                    symbol: "Row",
                },
                declared: TextRange::default(),
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
                    symbol: name,
                },
                declared: TextRange::default(),
                visibility: Visibility::Private,
            },
        );
    }

    fn bind_let(symbols: &mut SymbolTable<'static>, name: &'static str) {
        symbols.bind(
            name,
            Binding {
                kind: BindingKind::Let { symbol: name },
                declared: TextRange::default(),
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
        Reference::bare(name)
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
        symbols.enter_function(vec!["cap"]);
        assert_eq!(symbols.lookup(bare("cap")), Lookup::Param(0));
        symbols.enter_block();
        symbols.bind_local("cap", 7, Mutability::Immutable);
        assert_eq!(
            symbols.lookup(bare("cap")),
            Lookup::Local {
                slot: 7,
                mutability: Mutability::Immutable
            }
        );
        symbols.leave();
        assert_eq!(symbols.lookup(bare("cap")), Lookup::Param(0));
        symbols.leave();
        assert_eq!(symbols.lookup(bare("cap")), Lookup::Let("cap"));
        enter_relation(&mut symbols, "t");
        assert_eq!(symbols.lookup(bare("id")), Lookup::Column(0));
    }

    #[test]
    fn an_isolated_scope_reaches_the_module_and_nothing_between() {
        let mut symbols = symbols();
        bind_let(&mut symbols, "cap");
        symbols.enter_function(vec!["x"]);
        symbols.enter_block();
        symbols.bind_local("local", 0, Mutability::Immutable);
        enter_relation(&mut symbols, "t");

        assert_eq!(symbols.lookup(bare("id")), Lookup::Column(0));
        assert_eq!(symbols.lookup(bare("x")), Lookup::Unknown);
        assert_eq!(symbols.lookup(bare("local")), Lookup::Unknown);
        assert_eq!(symbols.lookup(bare("cap")), Lookup::Let("cap"));
    }

    #[test]
    fn a_declaration_that_is_not_a_value_says_so() {
        let symbols = symbols();

        assert_eq!(symbols.lookup(bare("t")), Lookup::NotAValue("relation"));
        assert_eq!(symbols.lookup(bare("Row")), Lookup::NotAValue("struct"));
    }

    #[test]
    fn every_declaration_carries_its_symbol() {
        let mut symbols = SymbolTable::new();
        symbols.bind(
            "Row",
            Binding {
                kind: BindingKind::Struct {
                    fields: vec!["a"],
                    symbol: "helpers.Row",
                },
                declared: TextRange::default(),
                visibility: Visibility::Private,
            },
        );
        symbols.bind(
            "Show",
            Binding {
                kind: BindingKind::Trait {
                    methods: vec!["show"],
                    symbol: "helpers.Show",
                },
                declared: TextRange::default(),
                visibility: Visibility::Private,
            },
        );
        symbols.bind(
            "f",
            Binding {
                kind: BindingKind::Func(Callable {
                    symbol: "helpers.f",
                    source: CalleeSource::Fn,
                    kind: FunctionKind::Scalar,
                    min_args: 0,
                    max_args: 0,
                }),
                declared: TextRange::default(),
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
                kind: BindingKind::Func(Callable {
                    symbol: "f",
                    source: CalleeSource::Fn,
                    kind: FunctionKind::Scalar,
                    min_args: 2,
                    max_args: 2,
                }),
                declared: TextRange::default(),
                visibility: Visibility::Private,
            },
        );
        symbols.bind(
            "Zero",
            Binding {
                kind: BindingKind::Trait {
                    methods: vec!["zero"],
                    symbol: "Zero",
                },
                declared: TextRange::default(),
                visibility: Visibility::Private,
            },
        );

        let registry = &yuzu_types::Builtins;
        assert_eq!(
            symbols
                .callable("cap", registry)
                .map(|callable| callable.source),
            Some(CalleeSource::Let)
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
