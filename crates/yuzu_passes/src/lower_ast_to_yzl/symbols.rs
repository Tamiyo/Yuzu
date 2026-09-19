//! The names a program can use, and what each means where it is used.
//!
//! The module's declarations, and a stack of the scopes inside them: a
//! function's parameters, a block's `let`s, the row a relation carries. A
//! lookup walks the stack from the top and falls back to the declarations,
//! so the innermost scope holding a name decides what it means. Answers are
//! bindings and positions, never IR — a column is the index of a block
//! argument — so nothing here depends on the builder.

use std::collections::HashMap;
use std::fmt;
use std::mem;

use text_size::TextRange;
use yuzu_ast::Visibility;
use yuzu_ast::ast::Mutability;
use yuzu_mlir::attributes::CalleeKind;
use yuzu_types::FunctionRegistry;

/// A column as the program refers to it: a bare name, or one qualified by
/// the alias of the side it came from.
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

    /// Whether a reference names this column: the name must match, and a
    /// qualified reference must name the side this column came from.
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

/// The columns a relation carries, in order. Two of them may share a name —
/// a join concatenates both sides — so a column is addressed by position,
/// and a name matching more than one is ambiguous until a qualifier picks.
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

    /// Names every column through one alias: what `as` does to the row, and
    /// what `join … as d` does to the side it names.
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

/// What a module-level name was declared as, where, and how far it reaches.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) struct Binding<'c> {
    pub(super) kind: Kind<'c>,
    pub(super) declared: TextRange,
    pub(super) visibility: Visibility,
}

/// The symbol a declaration carries is what the module holds it under: the
/// written name, qualified by the module, unless a later `let` took the name.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) enum Kind<'c> {
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
    /// A trait and the methods it declares. The methods are not names of
    /// their own — choosing an implementation is dispatch, which nothing
    /// does yet — so they hang off the trait rather than sit in the scope.
    Trait {
        methods: Vec<&'c str>,
        symbol: &'c str,
    },
    /// A `let` bound to a value: a callable of no arguments, which the
    /// inliner expands wherever the name is used.
    Let {
        symbol: &'c str,
    },
    /// A module this file named, by the path it was loaded under. What it
    /// declares is not in this scope: a qualified reference asks the module
    /// for it, one name at a time.
    Module {
        path: &'c str,
    },
}

impl<'c> Kind<'c> {
    /// The symbol the module holds this declaration under. A module is not
    /// one: it names a file rather than something the module declares.
    pub(super) fn symbol(&self) -> Option<&'c str> {
        match self {
            Kind::Struct { symbol, .. }
            | Kind::Relation { symbol, .. }
            | Kind::Trait { symbol, .. }
            | Kind::Let { symbol } => Some(symbol),
            Kind::Func(callable) => Some(callable.symbol),
            Kind::Module { .. } => None,
        }
    }

    /// What the name is, for a diagnostic about using it as something else.
    pub(super) fn what(&self) -> &'static str {
        match self {
            Kind::Struct { .. } => "struct",
            Kind::Relation { .. } => "relation",
            Kind::Func(_) => "function",
            Kind::Trait { .. } => "trait",
            Kind::Let { .. } => "binding",
            Kind::Module { .. } => "module",
        }
    }
}

/// A `let` in a function body: the name it binds, the slot the traversal put
/// its value in, and whether an assignment may write it again.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Local<'c> {
    pub(super) name: &'c str,
    pub(super) slot: usize,
    pub(super) mutability: Mutability,
}

/// A callee's kind, and the argument counts it takes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Callable<'c> {
    pub(super) symbol: &'c str,
    pub(super) kind: CalleeKind,
    pub(super) min_args: usize,
    pub(super) max_args: usize,
    /// Whether the callee aggregates, which is not the same question as
    /// what kind of callee it is: a builtin, an `agg fn` and an
    /// `external agg fn` are three kinds and all three aggregate.
    pub(super) is_agg: bool,
}

impl<'c> Callable<'c> {
    /// A `let` is a callable of no arguments, expanded wherever the name is
    /// used.
    pub(super) fn let_binding(symbol: &'c str) -> Self {
        Self {
            symbol,
            kind: CalleeKind::Let,
            min_args: 0,
            max_args: 0,
            is_agg: false,
        }
    }
}

/// What a name in expression position means.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Lookup<'c> {
    /// A column of the enclosing relation's row, by position.
    Column(usize),
    /// A parameter of the enclosing function, by position.
    Param(usize),
    /// A `let` in the enclosing function body, by the slot the traversal put
    /// its value in.
    Local {
        slot: usize,
        mutability: Mutability,
    },
    Let(&'c str),
    Ambiguous,
    /// The relation carried the column until a stage stopped carrying it.
    NarrowedAway,
    /// Declared, but not as something a value can be read from.
    NotAValue(&'static str),
    Unknown,
}

/// Which column of a row a stage item names.
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
    /// A lexical block inside a function body, and the `let`s it binds.
    Block {
        locals: Vec<Local<'c>>,
    },
    /// A relation and the row it carries. Its stages replace that row as
    /// they run, and `narrowed` is the names they stopped carrying: a column
    /// the program lost track of deserves a different diagnostic from a name
    /// it never declared.
    Relation {
        row: Row<'c>,
        narrowed: Vec<&'c str>,
    },
}

pub(super) struct SymbolTable<'c> {
    /// What the file declares, imports included: the symbols every lookup
    /// falls back to and every declaration accessor reads.
    module: HashMap<&'c str, Binding<'c>>,
    scopes: Vec<Scope<'c>>,
}

impl<'c> SymbolTable<'c> {
    pub(super) fn new() -> Self {
        Self {
            module: HashMap::new(),
            scopes: Vec::new(),
        }
    }

    // --- the module's declarations ---

    pub(super) fn bind(&mut self, name: &'c str, binding: Binding<'c>) {
        self.module.insert(name, binding);
    }

    pub(super) fn binding(&self, name: &str) -> Option<&Binding<'c>> {
        self.module.get(name)
    }

    /// Puts back the declarations taken earlier, so a second pass over the
    /// same file resolves against the names it already bound.
    pub(super) fn restore(&mut self, bindings: HashMap<&'c str, Binding<'c>>) {
        self.module = bindings;
    }

    /// Everything the file being lowered declared, which is what another
    /// file importing it may ask for.
    pub(super) fn take_exports(&mut self) -> HashMap<&'c str, Binding<'c>> {
        mem::take(&mut self.module)
    }

    /// Whether some trait declares this method.
    pub(super) fn is_method(&self, name: &str) -> bool {
        self.module.values().any(|binding| {
            matches!(&binding.kind, Kind::Trait { methods, .. } if methods.contains(&name))
        })
    }

    pub(super) fn struct_symbol(&self, name: &str) -> Option<&'c str> {
        match self.kind(name)? {
            Kind::Struct { symbol, .. } => Some(symbol),
            _ => None,
        }
    }

    pub(super) fn module_of(&self, name: &str) -> Option<&'c str> {
        match self.kind(name)? {
            Kind::Module { path } => Some(path),
            _ => None,
        }
    }

    pub(super) fn trait_symbol(&self, name: &str) -> Option<&'c str> {
        match self.kind(name)? {
            Kind::Trait { symbol, .. } => Some(symbol),
            _ => None,
        }
    }

    pub(super) fn kind(&self, name: &str) -> Option<&Kind<'c>> {
        self.binding(name).map(|binding| &binding.kind)
    }

    /// A relation's symbol and its row, seen through the alias it is named
    /// by.
    pub(super) fn relation(
        &self,
        name: &str,
        alias: Option<&'c str>,
    ) -> Option<(&'c str, Row<'c>)> {
        let Kind::Relation { row, symbol } = self.kind(name)? else {
            return None;
        };

        let mut row = row.clone();
        if let Some(alias) = alias {
            row.qualify(alias);
        }

        Some((symbol, row))
    }

    /// A callee by name, from the module's declarations or the registry.
    pub(super) fn callable(
        &self,
        name: &str,
        registry: &dyn FunctionRegistry,
    ) -> Option<Callable<'c>> {
        match self.kind(name) {
            Some(Kind::Func(callable)) => Some(*callable),
            Some(Kind::Let { symbol }) => Some(Callable::let_binding(symbol)),
            _ => self.builtin(name, registry),
        }
    }

    /// The function an operator stands for. An operator is sugar for a call
    /// to a name the registry offers, so a function declared under that name
    /// stands for it too; a value or a type sharing the name does not.
    pub(super) fn operator(
        &self,
        name: &str,
        registry: &dyn FunctionRegistry,
    ) -> Option<Callable<'c>> {
        match self.kind(name) {
            Some(Kind::Func(callable)) => Some(*callable),
            _ => self.builtin(name, registry),
        }
    }

    fn builtin(&self, name: &str, registry: &dyn FunctionRegistry) -> Option<Callable<'c>> {
        let entry = registry.entries().iter().find(|entry| entry.name == name)?;
        Some(Callable {
            symbol: entry.name,
            kind: CalleeKind::Builtin,
            min_args: entry.min_args,
            max_args: entry.max_args,
            is_agg: matches!(entry.func, yuzu_types::BuiltinFunc::Aggregate(_)),
        })
    }

    // --- scopes ---

    pub(super) fn enter_function(&mut self, params: Vec<&'c str>) {
        self.scopes.push(Scope::Function { params });
    }

    pub(super) fn enter_block(&mut self) {
        self.scopes.push(Scope::Block { locals: Vec::new() });
    }

    /// Binds a `let` in the innermost block. Binding a name twice shadows
    /// it, which is what a second `let` and an assignment both do.
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

    /// What a name means here. The walk stops at the first isolated scope:
    /// a function body and a stage's region are both `IsolatedFromAbove`, so
    /// no value bound outside one is in reach from inside it. The module's
    /// declarations answer from any depth — those are symbols, not values —
    /// so the walk falls back to them however it ends.
    pub(super) fn lookup(&self, reference: Reference<'_>) -> Lookup<'c> {
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

    fn module_lookup(&self, reference: Reference<'_>) -> Lookup<'c> {
        if reference.qualifier.is_some() {
            return Lookup::Unknown;
        }

        match self.kind(reference.name) {
            Some(Kind::Let { symbol }) => Lookup::Let(symbol),
            Some(kind) => Lookup::NotAValue(kind.what()),
            None => Lookup::Unknown,
        }
    }

    /// The column of the current row a stage item names.
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

    /// What `select` and `aggregate` do: the row becomes what they name.
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
    use yuzu_mlir::attributes::CalleeKind;

    use super::{Binding, Callable, ColumnLookup, Kind, Lookup, Reference, Row, SymbolTable};

    /// A module declaring `struct Row` and a table `t` over it.
    fn symbols() -> SymbolTable<'static> {
        let mut symbols = SymbolTable::new();
        symbols.bind(
            "Row",
            Binding {
                kind: Kind::Struct {
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
                kind: Kind::Relation {
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
                kind: Kind::Let { symbol: name },
                declared: TextRange::default(),
                visibility: Visibility::Private,
            },
        );
    }

    fn enter_relation(symbols: &mut SymbolTable<'static>, relation: &str) {
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

    /// A qualifier is spent choosing between two columns of one name; once
    /// it has, only the position remains.
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

    /// A stage's region and a function body are both `IsolatedFromAbove`, so
    /// a lookup inside one reaches the module's declarations and nothing
    /// between.
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
                kind: Kind::Struct {
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
                kind: Kind::Trait {
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
                kind: Kind::Func(Callable {
                    symbol: "helpers.f",
                    kind: CalleeKind::Fn,
                    min_args: 0,
                    max_args: 0,
                    is_agg: false,
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
                kind: Kind::Func(Callable {
                    symbol: "f",
                    kind: CalleeKind::Fn,
                    min_args: 2,
                    max_args: 2,
                    is_agg: false,
                }),
                declared: TextRange::default(),
                visibility: Visibility::Private,
            },
        );
        symbols.bind(
            "Zero",
            Binding {
                kind: Kind::Trait {
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
                .map(|callable| callable.kind),
            Some(CalleeKind::Let)
        );
        assert_eq!(
            symbols.callable("f", registry),
            Some(Callable {
                symbol: "f",
                kind: CalleeKind::Fn,
                min_args: 2,
                max_args: 2,
                is_agg: false
            })
        );
        assert_eq!(
            symbols
                .callable("sum", registry)
                .map(|callable| callable.kind),
            Some(CalleeKind::Builtin)
        );

        // A trait's methods are the trait's, not the module's.
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
