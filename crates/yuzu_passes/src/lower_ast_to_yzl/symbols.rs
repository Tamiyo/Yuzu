//! The names a program can use, and what each means where it is used.
//!
//! One scope stack: the module's declarations at the bottom, then a
//! function's parameters, then the row a relation carries. A lookup walks it
//! from the top, so the innermost scope holding a name decides what it means.
//! Answers are bindings and positions, never IR — a column is the index of a
//! block argument — so nothing here depends on the builder. Names are
//! `&'c str`, interned in the MLIR context.
//!
//! A function body's `let`s are the one scope that stays in the traversal:
//! they bind SSA values, whose lifetime is the block being built.

use std::collections::HashMap;
use std::fmt;

use text_size::TextRange;
use yuzu_ast::Visibility;
use yuzu_mlir::attributes::CalleeKind;
use yuzu_types::FunctionRegistry;

/// A column reference as the program wrote it: a bare name, or one qualified
/// by the alias of the side it came from. The two halves stay apart, so
/// resolving one never takes a string apart to find them.
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
}

impl fmt::Display for Reference<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.qualifier {
            Some(qualifier) => write!(f, "{qualifier}.{}", self.name),
            None => f.write_str(self.name),
        }
    }
}

/// A column of a relation's row: its name, and the alias qualifying it once
/// an `as` or a join has named its side.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Column<'c> {
    qualifier: Option<&'c str>,
    name: &'c str,
}

impl Column<'_> {
    /// Whether a reference names this column: the name must match, and a
    /// qualified reference must name the side this column came from.
    fn matches(&self, reference: Reference<'_>) -> bool {
        self.name == reference.name
            && reference
                .qualifier
                .is_none_or(|qualifier| self.qualifier == Some(qualifier))
    }
}

/// The columns a relation carries, in order. Two of them may share a name —
/// a join concatenates both sides — so a column is addressed by position,
/// and a name matching more than one is ambiguous until a qualifier picks.
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

    /// The names the row carries, in order.
    pub(super) fn names(&self) -> impl Iterator<Item = &'c str> + '_ {
        self.columns.iter().map(|column| column.name)
    }

    /// Every column as a reader could refer to it, qualifier and all.
    pub(super) fn references(&self) -> impl Iterator<Item = Reference<'c>> + '_ {
        self.columns.iter().map(|column| Reference {
            qualifier: column.qualifier,
            name: column.name,
        })
    }

    /// Which column a reference names.
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

    /// Whether any column answers to the reference, ambiguously or not.
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

    fn qualified(mut self, alias: Option<&'c str>) -> Self {
        if let Some(alias) = alias {
            self.qualify(alias);
        }

        self
    }

    pub(super) fn rename(&mut self, index: usize, name: &'c str) {
        self.columns[index].name = name;
    }

    pub(super) fn remove(&mut self, index: usize) {
        self.columns.remove(index);
    }

    /// Appends another row's columns: what `extend` adds, and what a join
    /// concatenates onto the side it starts from.
    pub(super) fn append(&mut self, other: Row<'c>) {
        self.columns.extend(other.columns);
    }
}

impl<'c> From<Vec<&'c str>> for Row<'c> {
    /// The row a list of names describes, before an alias qualifies it.
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

/// What a module-level name was declared as, where it was declared, and how
/// far the name reaches.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) struct Binding<'c> {
    pub(super) kind: Kind<'c>,
    pub(super) declared: TextRange,
    pub(super) visibility: Visibility,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) enum Kind<'c> {
    Struct {
        fields: Vec<&'c str>,
        symbol: &'c str,
    },
    /// A table, or a `let` bound to a query: what `from` and `join` name.
    /// The symbol is what the module holds it under, which is the written
    /// name unless a later `let` took that name for something else.
    Relation {
        row: Row<'c>,
        symbol: &'c str,
    },
    Func(Callable<'c>),
    /// A trait, and the methods it declares. The methods are not names of
    /// their own — choosing an implementation is dispatch, which nothing
    /// does yet — so they hang off the trait rather than sit in the scope.
    Trait {
        methods: Vec<&'c str>,
        symbol: &'c str,
    },
    /// A `let` bound to a value: a callable of no arguments, which the
    /// inliner expands wherever the name is used. The symbol is what the
    /// module holds it under, which is the written name unless a later
    /// `let` took that name for something else.
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

impl Kind<'_> {
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
    pub(super) mutable: bool,
}

/// A callee's kind, and the argument counts it takes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Callable<'c> {
    /// What the module holds the callee under, which is the written name
    /// unless a declaration took that name for something else.
    pub(super) symbol: &'c str,
    pub(super) kind: CalleeKind,
    pub(super) min_args: usize,
    pub(super) max_args: usize,
    /// Whether the callee aggregates, which is not the same question as
    /// what kind of callee it is: a builtin, an `agg fn` and an
    /// `external agg fn` are three kinds and all three aggregate.
    pub(super) agg: bool,
}

/// What a name in expression position means.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Lookup<'c> {
    /// A column of the enclosing relation's row, by position.
    Column(usize),
    /// A parameter of the enclosing function, by position.
    Param(usize),
    /// A `let` in the enclosing function body: the slot the traversal put
    /// its value in, and whether an assignment may write it again.
    Local {
        slot: usize,
        mutable: bool,
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
    Module(HashMap<&'c str, Binding<'c>>),
    Function {
        params: Vec<&'c str>,
    },
    /// A lexical block inside a function body, and the `let`s it binds. A
    /// local stands for an SSA value, which only the traversal can hold, so
    /// what is kept here is the slot the traversal put it in.
    Block {
        locals: Vec<Local<'c>>,
    },
    /// A relation and the row it carries. Its stages replace that row as
    /// they run, and `narrowed` is the names they stopped carrying: a name
    /// the relation once had is a column the program lost track of, not a
    /// name it never declared, and the two deserve different diagnostics.
    Relation {
        row: Row<'c>,
        narrowed: Vec<&'c str>,
    },
}

pub(super) struct SymbolTable<'c> {
    scopes: Vec<Scope<'c>>,
}

impl<'c> SymbolTable<'c> {
    pub(super) fn new() -> Self {
        Self {
            scopes: vec![Scope::Module(HashMap::new())],
        }
    }

    // --- the module scope ---

    pub(super) fn bind(&mut self, name: &'c str, binding: Binding<'c>) {
        match self.scopes.first_mut() {
            Some(Scope::Module(bindings)) => bindings.insert(name, binding),
            Some(Scope::Function { .. } | Scope::Block { .. } | Scope::Relation { .. }) | None => {
                unreachable!("the module scope is the bottom of the stack")
            }
        };
    }

    pub(super) fn binding(&self, name: &str) -> Option<&Binding<'c>> {
        self.module().get(name)
    }

    /// Everything the file being converted declared, which is what another
    /// file importing it may ask for.
    pub(super) fn exports(&self) -> HashMap<&'c str, Binding<'c>> {
        self.module().clone()
    }

    fn module(&self) -> &HashMap<&'c str, Binding<'c>> {
        match self.scopes.first() {
            Some(Scope::Module(bindings)) => bindings,
            Some(Scope::Function { .. } | Scope::Block { .. } | Scope::Relation { .. }) | None => {
                unreachable!("the module scope is the bottom of the stack")
            }
        }
    }

    /// Whether some trait declares this method.
    pub(super) fn is_method(&self, name: &str) -> bool {
        self.module().values().any(|binding| match &binding.kind {
            Kind::Trait { methods, .. } => methods.contains(&name),
            Kind::Struct { .. }
            | Kind::Relation { .. }
            | Kind::Func(_)
            | Kind::Let { .. }
            | Kind::Module { .. } => false,
        })
    }

    /// The symbol a struct name was declared under, which is what a type
    /// referring to it names.
    pub(super) fn struct_symbol(&self, name: &str) -> Option<&'c str> {
        match self.kind(name) {
            Some(Kind::Struct { symbol, .. }) => Some(symbol),
            Some(
                Kind::Relation { .. }
                | Kind::Func(_)
                | Kind::Trait { .. }
                | Kind::Let { .. }
                | Kind::Module { .. },
            )
            | None => None,
        }
    }

    /// The module a name stands for, by the path it was loaded under.
    pub(super) fn module_of(&self, name: &str) -> Option<&'c str> {
        match self.kind(name) {
            Some(Kind::Module { path }) => Some(path),
            Some(
                Kind::Struct { .. }
                | Kind::Relation { .. }
                | Kind::Func(_)
                | Kind::Trait { .. }
                | Kind::Let { .. },
            )
            | None => None,
        }
    }

    /// The symbol a trait name was declared under.
    pub(super) fn trait_symbol(&self, name: &str) -> Option<&'c str> {
        match self.kind(name) {
            Some(Kind::Trait { symbol, .. }) => Some(symbol),
            Some(
                Kind::Struct { .. }
                | Kind::Relation { .. }
                | Kind::Func(_)
                | Kind::Let { .. }
                | Kind::Module { .. },
            )
            | None => None,
        }
    }

    pub(super) fn kind(&self, name: &str) -> Option<&Kind<'c>> {
        self.binding(name).map(|binding| &binding.kind)
    }

    /// A relation's symbol and its row, seen through the alias it is named
    /// by. The symbol is what a `yzl.from` names, which is not the written
    /// name once a `let` has been rebound.
    pub(super) fn relation(
        &self,
        name: &str,
        alias: Option<&'c str>,
    ) -> Option<(&'c str, Row<'c>)> {
        let Some(Kind::Relation { row, symbol }) = self.kind(name) else {
            return None;
        };

        Some((symbol, row.clone().qualified(alias)))
    }

    /// A callee by name, from the module's declarations or the registry.
    pub(super) fn callable(
        &self,
        name: &str,
        registry: &dyn FunctionRegistry,
    ) -> Option<Callable<'c>> {
        match self.kind(name) {
            Some(Kind::Func(callable)) => return Some(*callable),
            Some(Kind::Let { symbol }) => {
                return Some(Callable {
                    symbol,
                    kind: CalleeKind::Let,
                    min_args: 0,
                    max_args: 0,
                    agg: false,
                });
            }
            Some(
                Kind::Struct { .. }
                | Kind::Relation { .. }
                | Kind::Trait { .. }
                | Kind::Module { .. },
            )
            | None => {}
        }

        self.builtin(name, registry)
    }

    /// The function an operator stands for. An operator is sugar for a call
    /// to a name the registry already offers, so it resolves the same way —
    /// except that only a function can stand for one. A value or a type
    /// sharing the name is a different thing, and the operator goes on
    /// meaning what the language says it means.
    pub(super) fn operator(
        &self,
        name: &str,
        registry: &dyn FunctionRegistry,
    ) -> Option<Callable<'c>> {
        match self.kind(name) {
            Some(Kind::Func(callable)) => Some(*callable),
            Some(
                Kind::Let { .. }
                | Kind::Struct { .. }
                | Kind::Relation { .. }
                | Kind::Trait { .. }
                | Kind::Module { .. },
            )
            | None => self.builtin(name, registry),
        }
    }

    fn builtin(&self, name: &str, registry: &dyn FunctionRegistry) -> Option<Callable<'c>> {
        let entry = registry.entries().iter().find(|entry| entry.name == name)?;
        Some(Callable {
            symbol: entry.name,
            kind: CalleeKind::Builtin,
            min_args: entry.min_args,
            max_args: entry.max_args,
            agg: matches!(entry.func, yuzu_types::BuiltinFunc::Aggregate(_)),
        })
    }

    // --- scopes ---

    pub(super) fn enter_function(&mut self, params: Vec<&'c str>) {
        self.scopes.push(Scope::Function { params });
    }

    pub(super) fn enter_block(&mut self) {
        self.scopes.push(Scope::Block { locals: Vec::new() });
    }

    /// Binds a `let` in the innermost block to the slot the traversal put
    /// its value in. Binding a name twice shadows it, which is what a
    /// second `let` and an assignment both do.
    pub(super) fn bind_local(&mut self, name: &'c str, slot: usize, mutable: bool) {
        let Some(Scope::Block { locals }) = self.scopes.last_mut() else {
            panic!("a local is being bound outside a block")
        };

        locals.push(Local {
            name,
            slot,
            mutable,
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

    /// The row the enclosing relation carries, for building block arguments.
    pub(super) fn row(&self) -> &Row<'c> {
        self.current_row()
            .expect("a stage is being converted outside a relation")
    }

    /// The row in scope, when a relation is being converted at all.
    pub(super) fn current_row(&self) -> Option<&Row<'c>> {
        match self.scopes.last() {
            Some(Scope::Relation { row, .. }) => Some(row),
            Some(Scope::Module(_) | Scope::Function { .. } | Scope::Block { .. }) | None => None,
        }
    }

    // --- lookups ---

    /// What a name means here. The walk stops at the first isolated scope:
    /// a function body and a stage's region are both `IsolatedFromAbove`, so
    /// no value bound outside one is in reach from inside it. The module's
    /// declarations answer from anywhere — those are symbols, not values.
    pub(super) fn lookup(&self, reference: Reference<'_>) -> Lookup<'c> {
        for scope in self.scopes.iter().rev() {
            match scope {
                Scope::Relation { row, narrowed } => {
                    match row.column(reference) {
                        ColumnLookup::Unique(index) => return Lookup::Column(index),
                        ColumnLookup::Ambiguous => return Lookup::Ambiguous,
                        // A name the relation once carried is the column the
                        // program meant, not what an outer scope calls it.
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
                            mutable: local.mutable,
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
                Scope::Module(_) => break,
            }
        }

        self.module_lookup(reference)
    }

    fn module_lookup(&self, reference: Reference<'_>) -> Lookup<'c> {
        if reference.qualifier.is_some() {
            return Lookup::Unknown;
        }

        match self.binding(reference.name) {
            Some(Binding {
                kind: Kind::Let { symbol },
                ..
            }) => Lookup::Let(symbol),
            Some(binding) => Lookup::NotAValue(binding.kind.what()),
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
            panic!("a stage is being converted outside a relation")
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
            panic!("a stage is being converted outside a relation")
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

    /// Stages narrow the row in written order, and a column the relation
    /// stopped carrying is a different mistake from a name it never had.
    #[test]
    fn a_narrowed_column_is_not_an_unknown_name() {
        let mut symbols = symbols();
        enter_relation(&mut symbols, "t");
        symbols.remove(0);

        assert_eq!(symbols.lookup(bare("id")), Lookup::NarrowedAway);
        assert_eq!(symbols.lookup(bare("dept_id")), Lookup::Column(0));
    }

    /// The module scope is the bottom of the stack, so its names are
    /// visible from every scope above and shadowed by each of them.
    #[test]
    fn the_innermost_scope_holding_a_name_decides_it() {
        let mut symbols = symbols();
        symbols.bind(
            "cap",
            Binding {
                kind: Kind::Let { symbol: "cap" },
                declared: TextRange::default(),
                visibility: Visibility::Private,
            },
        );
        symbols.bind(
            "id",
            Binding {
                kind: Kind::Let { symbol: "id" },
                declared: TextRange::default(),
                visibility: Visibility::Private,
            },
        );

        assert_eq!(symbols.lookup(bare("cap")), Lookup::Let("cap"));
        symbols.enter_function(vec!["cap"]);
        assert_eq!(symbols.lookup(bare("cap")), Lookup::Param(0));
        symbols.enter_block();
        symbols.bind_local("cap", 7, false);
        assert_eq!(
            symbols.lookup(bare("cap")),
            Lookup::Local {
                slot: 7,
                mutable: false
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
    /// a lookup inside one reaches the module's symbols and nothing in
    /// between: no SSA value bound outside it is in reach.
    #[test]
    fn an_isolated_scope_reaches_the_module_and_nothing_between() {
        let mut symbols = symbols();
        symbols.bind(
            "cap",
            Binding {
                kind: Kind::Let { symbol: "cap" },
                declared: TextRange::default(),
                visibility: Visibility::Private,
            },
        );
        symbols.enter_function(vec!["x"]);
        symbols.enter_block();
        symbols.bind_local("local", 0, false);
        enter_relation(&mut symbols, "t");

        assert_eq!(symbols.lookup(bare("id")), Lookup::Column(0));
        assert_eq!(symbols.lookup(bare("x")), Lookup::Unknown);
        assert_eq!(symbols.lookup(bare("local")), Lookup::Unknown);
        assert_eq!(symbols.lookup(bare("cap")), Lookup::Let("cap"));
    }

    /// One namespace, so a name declared as anything but a `let` is known
    /// and simply cannot be read from.
    #[test]
    fn a_declaration_that_is_not_a_value_says_so() {
        let symbols = symbols();

        assert_eq!(symbols.lookup(bare("t")), Lookup::NotAValue("relation"));
        assert_eq!(symbols.lookup(bare("Row")), Lookup::NotAValue("struct"));
    }

    /// Every declaration says what the module holds it under, not only the
    /// two that needed it first. The symbol is the written name while that
    /// name is free, and diverges as soon as something else takes it.
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
                    agg: false,
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

        // A name declared as one thing is not another kind's symbol.
        assert_eq!(symbols.trait_symbol("Row"), None);
        assert_eq!(symbols.struct_symbol("Show"), None);
    }

    #[test]
    fn callables() {
        let mut symbols = symbols();
        symbols.bind(
            "cap",
            Binding {
                kind: Kind::Let { symbol: "cap" },
                declared: TextRange::default(),
                visibility: Visibility::Private,
            },
        );
        symbols.bind(
            "f",
            Binding {
                kind: Kind::Func(Callable {
                    symbol: "f",
                    kind: CalleeKind::Fn,
                    min_args: 2,
                    max_args: 2,
                    agg: false,
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
                agg: false
            })
        );
        assert_eq!(
            symbols
                .callable("sum", registry)
                .map(|callable| callable.kind),
            Some(CalleeKind::Builtin)
        );

        // A trait's methods are the trait's, not the module's: `zero` is
        // known enough to report, and `Zero` is not callable at all.
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
