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
        self.columns.iter().any(|column| column.matches(reference))
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

/// What a module-level name was declared as, and where it was declared.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) struct Binding<'c> {
    pub(super) kind: Kind<'c>,
    pub(super) declared: TextRange,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) enum Kind<'c> {
    Struct {
        fields: Vec<&'c str>,
    },
    /// A table, or a `let` bound to a query: what `from` and `join` name.
    Relation {
        row: Row<'c>,
    },
    Func(Callable),
    /// A trait, and the methods it declares. The methods are not names of
    /// their own — choosing an implementation is dispatch, which nothing
    /// does yet — so they hang off the trait rather than sit in the scope.
    Trait {
        methods: Vec<&'c str>,
    },
    /// A `let` bound to a value: a callable of no arguments, which the
    /// inliner expands wherever the name is used.
    Let,
}

impl Kind<'_> {
    /// What the name is, for a diagnostic about using it as something else.
    pub(super) fn what(&self) -> &'static str {
        match self {
            Kind::Struct { .. } => "struct",
            Kind::Relation { .. } => "relation",
            Kind::Func(_) => "function",
            Kind::Trait { .. } => "trait",
            Kind::Let => "binding",
        }
    }
}

/// A callee's kind, and the argument counts it takes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Callable {
    pub(super) kind: CalleeKind,
    pub(super) min_args: usize,
    pub(super) max_args: usize,
}

/// What a name in expression position means.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Lookup<'c> {
    /// A column of the enclosing relation's row, by position.
    Column(usize),
    /// A parameter of the enclosing function, by position.
    Param(usize),
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

    pub(super) fn bind(&mut self, name: &'c str, kind: Kind<'c>, declared: TextRange) {
        match self.scopes.first_mut() {
            Some(Scope::Module(bindings)) => bindings.insert(name, Binding { kind, declared }),
            Some(Scope::Function { .. } | Scope::Relation { .. }) | None => {
                unreachable!("the module scope is the bottom of the stack")
            }
        };
    }

    pub(super) fn binding(&self, name: &str) -> Option<&Binding<'c>> {
        self.module().get(name)
    }

    fn module(&self) -> &HashMap<&'c str, Binding<'c>> {
        match self.scopes.first() {
            Some(Scope::Module(bindings)) => bindings,
            Some(Scope::Function { .. } | Scope::Relation { .. }) | None => {
                unreachable!("the module scope is the bottom of the stack")
            }
        }
    }

    /// Whether some trait declares this method.
    pub(super) fn is_method(&self, name: &str) -> bool {
        self.module().values().any(|binding| match &binding.kind {
            Kind::Trait { methods } => methods.contains(&name),
            Kind::Struct { .. } | Kind::Relation { .. } | Kind::Func(_) | Kind::Let => false,
        })
    }

    /// Whether a name denotes a type an `impl` or an annotation can name.
    pub(super) fn is_type_name(&self, name: &str) -> bool {
        matches!(name, "int64" | "float64" | "bool" | "str")
            || matches!(self.kind(name), Some(Kind::Struct { .. }))
    }

    pub(super) fn kind(&self, name: &str) -> Option<&Kind<'c>> {
        self.binding(name).map(|binding| &binding.kind)
    }

    /// A relation's row, seen through the alias it is named by.
    pub(super) fn relation(&self, name: &str, alias: Option<&'c str>) -> Option<Row<'c>> {
        let Some(Kind::Relation { row }) = self.kind(name) else {
            return None;
        };

        Some(row.clone().qualified(alias))
    }

    /// A callee by name, from the module's declarations or the registry.
    pub(super) fn callable(&self, name: &str, registry: &dyn FunctionRegistry) -> Option<Callable> {
        match self.kind(name) {
            Some(Kind::Func(callable)) => return Some(*callable),
            Some(Kind::Let) => {
                return Some(Callable {
                    kind: CalleeKind::Let,
                    min_args: 0,
                    max_args: 0,
                });
            }
            Some(Kind::Struct { .. } | Kind::Relation { .. } | Kind::Trait { .. }) | None => {}
        }

        let entry = registry.entries().iter().find(|entry| entry.name == name)?;
        Some(Callable {
            kind: CalleeKind::Builtin,
            min_args: entry.min_args,
            max_args: entry.max_args,
        })
    }

    // --- scopes ---

    pub(super) fn enter_function(&mut self, params: Vec<&'c str>) {
        self.scopes.push(Scope::Function { params });
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
        match self.scopes.last() {
            Some(Scope::Relation { row, .. }) => row,
            Some(Scope::Module(_) | Scope::Function { .. }) | None => {
                panic!("a stage is being converted outside a relation")
            }
        }
    }

    // --- lookups ---

    pub(super) fn lookup(&self, reference: Reference<'_>) -> Lookup<'c> {
        for scope in self.scopes.iter().rev() {
            match scope {
                Scope::Relation { row, narrowed } => match row.column(reference) {
                    ColumnLookup::Unique(index) => return Lookup::Column(index),
                    ColumnLookup::Ambiguous => return Lookup::Ambiguous,
                    // A name the relation once carried is the column the
                    // program meant, not whatever an outer scope calls it.
                    ColumnLookup::Absent if narrowed.contains(&reference.name) => {
                        return Lookup::NarrowedAway;
                    }
                    ColumnLookup::Absent => {}
                },
                Scope::Function { params } => {
                    if reference.qualifier.is_none()
                        && let Some(index) =
                            params.iter().position(|&param| param == reference.name)
                    {
                        return Lookup::Param(index);
                    }
                }
                Scope::Module(bindings) => {
                    if reference.qualifier.is_some() {
                        return Lookup::Unknown;
                    }

                    return match bindings.get_key_value(reference.name) {
                        Some((
                            &name,
                            Binding {
                                kind: Kind::Let, ..
                            },
                        )) => Lookup::Let(name),
                        Some((_, binding)) => Lookup::NotAValue(binding.kind.what()),
                        None => Lookup::Unknown,
                    };
                }
            }
        }

        unreachable!("the module scope is the bottom of the stack")
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

    pub(super) fn select(&mut self, names: Vec<&'c str>) {
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
    use yuzu_mlir::attributes::CalleeKind;

    use super::{Callable, ColumnLookup, Kind, Lookup, Reference, Row, SymbolTable};

    /// A module declaring `struct Row` and a table `t` over it.
    fn symbols() -> SymbolTable<'static> {
        let mut symbols = SymbolTable::new();
        symbols.bind(
            "Row",
            Kind::Struct {
                fields: vec!["id", "dept_id"],
            },
            TextRange::default(),
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
            Kind::Relation {
                row: Row::from(row),
            },
            TextRange::default(),
        );
    }

    fn enter_relation(symbols: &mut SymbolTable<'static>, relation: &str) {
        let row = symbols
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
        let rhs = symbols
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
        symbols.bind("cap", Kind::Let, TextRange::default());
        symbols.bind("id", Kind::Let, TextRange::default());

        assert_eq!(symbols.lookup(bare("cap")), Lookup::Let("cap"));
        symbols.enter_function(vec!["cap"]);
        assert_eq!(symbols.lookup(bare("cap")), Lookup::Param(0));
        enter_relation(&mut symbols, "t");
        assert_eq!(symbols.lookup(bare("id")), Lookup::Column(0));
        assert_eq!(symbols.lookup(bare("cap")), Lookup::Param(0));
        symbols.leave();
        symbols.leave();
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

    #[test]
    fn callables() {
        let mut symbols = symbols();
        symbols.bind("cap", Kind::Let, TextRange::default());
        symbols.bind(
            "f",
            Kind::Func(Callable {
                kind: CalleeKind::Fn,
                min_args: 2,
                max_args: 2,
            }),
            TextRange::default(),
        );
        symbols.bind(
            "Zero",
            Kind::Trait {
                methods: vec!["zero"],
            },
            TextRange::default(),
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
                kind: CalleeKind::Fn,
                min_args: 2,
                max_args: 2
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
    fn select_replaces_the_row() {
        let mut symbols = symbols();
        enter_relation(&mut symbols, "t");
        assert_eq!(symbols.column(bare("dept_id")), ColumnLookup::Unique(1));
        symbols.select(vec!["dept_id", "n"]);

        assert_eq!(symbols.lookup(bare("dept_id")), Lookup::Column(0));
        assert_eq!(symbols.lookup(bare("n")), Lookup::Column(1));
        assert_eq!(symbols.lookup(bare("id")), Lookup::NarrowedAway);
    }

    #[test]
    fn join_concatenates_both_rows() {
        let mut symbols = symbols();
        bind_relation(&mut symbols, "depts", vec!["dept_id"]);
        enter_relation(&mut symbols, "t");
        let rhs = symbols.relation("depts", None).expect("depts is bound");
        symbols.concat(rhs);

        assert_eq!(
            symbols.row().names().collect::<Vec<_>>(),
            ["id", "dept_id", "dept_id"]
        );
        assert_eq!(symbols.column(bare("dept_id")), ColumnLookup::Ambiguous);
    }
}
