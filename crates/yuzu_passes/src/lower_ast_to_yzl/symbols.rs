//! The names a program can use, and what each means where it is used.
//!
//! One scope stack: the module's declarations at the bottom, then a
//! function's parameters, then a query stage's row. A lookup walks it from
//! the top, so the innermost scope holding a name decides what it means.
//! Answers are bindings and positions, never IR — a column is the index of a
//! block argument — so nothing here depends on the builder. Names are
//! `&'c str`, interned in the MLIR context.
//!
//! A function body's `let`s are the one scope that stays in the traversal:
//! they bind SSA values, whose lifetime is the block being built.

use std::collections::HashMap;

use yuzu_mlir::attributes::CalleeKind;
use yuzu_types::FunctionRegistry;

/// A column the query carries at some stage: its name, and the alias
/// qualifying it once an `as` or a join has named its side.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Column<'c> {
    pub(super) qualifier: Option<&'c str>,
    pub(super) name: &'c str,
}

impl Column<'_> {
    fn matches(&self, reference: &str) -> bool {
        match reference.split_once('.') {
            Some((qualifier, name)) => self.qualifier == Some(qualifier) && self.name == name,
            None => self.name == reference,
        }
    }
}

pub(super) type Row<'c> = Vec<Column<'c>>;

/// The row a list of names describes, before an alias qualifies it.
pub(super) fn unqualified<'c>(names: Vec<&'c str>) -> Row<'c> {
    names
        .into_iter()
        .map(|name| Column {
            qualifier: None,
            name,
        })
        .collect()
}

pub(super) fn has_column(row: &[Column<'_>], name: &str) -> bool {
    row.iter().any(|column| column.matches(name))
}

/// What a module-level name was declared as.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) enum Binding<'c> {
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

impl Binding<'_> {
    /// What the name is, for a diagnostic about using it as something else.
    pub(super) fn what(&self) -> &'static str {
        match self {
            Binding::Struct { .. } => "struct",
            Binding::Relation { .. } => "relation",
            Binding::Func(_) => "function",
            Binding::Trait { .. } => "trait",
            Binding::Let => "binding",
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
    /// A column of the enclosing stage's row, by position.
    Column(usize),
    /// A parameter of the enclosing function, by position.
    Param(usize),
    Let(&'c str),
    Ambiguous,
    /// An earlier stage carried the column and narrowed it away.
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
    Function { params: Vec<&'c str> },
    Stage { row: Row<'c>, shed: Vec<&'c str> },
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
            Some(Scope::Function { .. } | Scope::Stage { .. }) | None => {
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
            Some(Scope::Function { .. } | Scope::Stage { .. }) | None => {
                unreachable!("the module scope is the bottom of the stack")
            }
        }
    }

    /// Whether some trait declares this method.
    pub(super) fn is_method(&self, name: &str) -> bool {
        self.module().values().any(|binding| match binding {
            Binding::Trait { methods } => methods.contains(&name),
            Binding::Struct { .. } | Binding::Relation { .. } | Binding::Func(_) | Binding::Let => {
                false
            }
        })
    }

    /// Whether a name denotes a type an `impl` or an annotation can name.
    pub(super) fn is_type_name(&self, name: &str) -> bool {
        matches!(name, "int64" | "float64" | "bool" | "str")
            || matches!(self.binding(name), Some(Binding::Struct { .. }))
    }

    /// A relation's row, qualified by the alias it is named through.
    pub(super) fn relation(&self, name: &str, alias: Option<&'c str>) -> Option<Row<'c>> {
        let Some(Binding::Relation { row }) = self.binding(name) else {
            return None;
        };

        Some(
            row.iter()
                .map(|&column| Column {
                    qualifier: alias.or(column.qualifier),
                    name: column.name,
                })
                .collect(),
        )
    }

    /// A callee by name, from the module's declarations or the registry.
    pub(super) fn callable(&self, name: &str, registry: &dyn FunctionRegistry) -> Option<Callable> {
        match self.binding(name) {
            Some(Binding::Func(callable)) => return Some(*callable),
            Some(Binding::Let) => {
                return Some(Callable {
                    kind: CalleeKind::Let,
                    min_args: 0,
                    max_args: 0,
                });
            }
            Some(Binding::Struct { .. } | Binding::Relation { .. } | Binding::Trait { .. })
            | None => {}
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

    pub(super) fn enter_query(&mut self, row: Row<'c>) {
        self.scopes.push(Scope::Stage {
            row,
            shed: Vec::new(),
        });
    }

    pub(super) fn leave(&mut self) {
        self.scopes.pop();
    }

    /// The row the enclosing stage sees, for building block arguments.
    pub(super) fn row(&self) -> &Row<'c> {
        match self.scopes.last() {
            Some(Scope::Stage { row, .. }) => row,
            Some(Scope::Module(_) | Scope::Function { .. }) | None => {
                panic!("a stage is being resolved outside a query")
            }
        }
    }

    // --- lookups ---

    pub(super) fn lookup(&self, reference: &str) -> Lookup<'c> {
        for scope in self.scopes.iter().rev() {
            match scope {
                Scope::Stage { row, shed } => match column_of(row, reference) {
                    ColumnLookup::Unique(index) => return Lookup::Column(index),
                    ColumnLookup::Ambiguous => return Lookup::Ambiguous,
                    ColumnLookup::Absent => {
                        // A name the row carried and shed is the column the
                        // program meant, not whatever an outer scope calls it.
                        let column = reference
                            .split_once('.')
                            .map_or(reference, |(_, name)| name);
                        if shed.contains(&column) {
                            return Lookup::NarrowedAway;
                        }
                    }
                },
                Scope::Function { params } => {
                    if let Some(index) = params.iter().position(|&param| param == reference) {
                        return Lookup::Param(index);
                    }
                }
                Scope::Module(bindings) => {
                    return match bindings.get_key_value(reference) {
                        Some((&name, Binding::Let)) => Lookup::Let(name),
                        Some((_, binding)) => Lookup::NotAValue(binding.what()),
                        None => Lookup::Unknown,
                    };
                }
            }
        }

        unreachable!("the module scope is the bottom of the stack")
    }

    /// The column of the current row a stage item names.
    pub(super) fn column(&self, reference: &str) -> ColumnLookup {
        column_of(self.row(), reference)
    }

    // --- stage transitions ---
    //
    // Each replaces the current row with the one the stage produces. Names
    // the stage stops carrying are remembered as shed, so a later reference
    // to one can be told apart from a name that never existed.

    fn replace_row(&mut self, next: Row<'c>) {
        let Some(Scope::Stage { row, shed }) = self.scopes.last_mut() else {
            panic!("a stage is being resolved outside a query")
        };

        for column in row.iter() {
            if !next.iter().any(|kept| kept.name == column.name) && !shed.contains(&column.name) {
                shed.push(column.name);
            }
        }

        *row = next;
    }

    pub(super) fn alias(&mut self, alias: &'c str) {
        let Some(Scope::Stage { row, .. }) = self.scopes.last_mut() else {
            panic!("a stage is being resolved outside a query")
        };

        for column in row.iter_mut() {
            column.qualifier = Some(alias);
        }
    }

    pub(super) fn select(&mut self, names: Vec<&'c str>) {
        self.replace_row(unqualified(names));
    }

    pub(super) fn extend(&mut self, names: Vec<&'c str>) {
        let mut next = self.row().clone();
        next.extend(unqualified(names));
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
            next[index].name = to;
        }

        self.replace_row(next);
    }

    pub(super) fn concat(&mut self, rhs: Row<'c>) {
        let mut next = self.row().clone();
        next.extend(rhs);
        self.replace_row(next);
    }
}

fn column_of(row: &[Column<'_>], reference: &str) -> ColumnLookup {
    let mut matches = row
        .iter()
        .enumerate()
        .filter(|(_, column)| column.matches(reference));
    match (matches.next(), matches.next()) {
        (Some((index, _)), None) => ColumnLookup::Unique(index),
        (Some(_), Some(_)) => ColumnLookup::Ambiguous,
        (None, _) => ColumnLookup::Absent,
    }
}

#[cfg(test)]
mod tests {
    use yuzu_mlir::attributes::CalleeKind;

    use super::{Binding, Callable, Column, ColumnLookup, Lookup, SymbolTable, unqualified};

    /// A module declaring `struct Row` and a table `t` over it.
    fn symbols() -> SymbolTable<'static> {
        let mut symbols = SymbolTable::new();
        symbols.bind(
            "Row",
            Binding::Struct {
                fields: vec!["id", "dept_id"],
            },
        );
        symbols.bind(
            "t",
            Binding::Relation {
                row: unqualified(vec!["id", "dept_id"]),
            },
        );
        symbols
    }

    fn enter_query(symbols: &mut SymbolTable<'static>, relation: &str) {
        let row = symbols
            .relation(relation, None)
            .expect("the relation is declared");
        symbols.enter_query(row);
    }

    #[test]
    fn a_column_resolves_by_position() {
        let mut symbols = symbols();
        enter_query(&mut symbols, "t");

        assert_eq!(symbols.lookup("dept_id"), Lookup::Column(1));
        assert_eq!(symbols.lookup("nope"), Lookup::Unknown);
    }

    /// A qualifier is spent choosing between two columns of one name; once
    /// it has, only the position remains.
    #[test]
    fn a_qualifier_picks_between_same_named_columns() {
        let mut symbols = symbols();
        symbols.bind(
            "depts",
            Binding::Relation {
                row: unqualified(vec!["id"]),
            },
        );
        enter_query(&mut symbols, "t");
        symbols.alias("a");
        let rhs = symbols
            .relation("depts", Some("d"))
            .expect("depts is bound");
        symbols.concat(rhs);

        assert_eq!(symbols.lookup("id"), Lookup::Ambiguous);
        assert_eq!(symbols.lookup("a.id"), Lookup::Column(0));
        assert_eq!(symbols.lookup("d.id"), Lookup::Column(2));
    }

    /// Stages narrow the row in written order, and a name a stage shed is a
    /// different mistake from one that never existed.
    #[test]
    fn a_shed_column_is_narrowed_away() {
        let mut symbols = symbols();
        enter_query(&mut symbols, "t");
        symbols.remove(0);

        assert_eq!(symbols.lookup("id"), Lookup::NarrowedAway);
        assert_eq!(symbols.lookup("dept_id"), Lookup::Column(0));
    }

    /// The module scope is the bottom of the stack, so its names are
    /// visible from every scope above and shadowed by each of them.
    #[test]
    fn the_innermost_scope_holding_a_name_decides_it() {
        let mut symbols = symbols();
        symbols.bind("cap", Binding::Let);
        symbols.bind("id", Binding::Let);

        assert_eq!(symbols.lookup("cap"), Lookup::Let("cap"));
        symbols.enter_function(vec!["cap"]);
        assert_eq!(symbols.lookup("cap"), Lookup::Param(0));
        enter_query(&mut symbols, "t");
        assert_eq!(symbols.lookup("id"), Lookup::Column(0));
        assert_eq!(symbols.lookup("cap"), Lookup::Param(0));
        symbols.leave();
        symbols.leave();
        assert_eq!(symbols.lookup("cap"), Lookup::Let("cap"));
    }

    /// One namespace, so a name declared as anything but a `let` is known
    /// and simply cannot be read from.
    #[test]
    fn a_declaration_that_is_not_a_value_says_so() {
        let symbols = symbols();

        assert_eq!(symbols.lookup("t"), Lookup::NotAValue("relation"));
        assert_eq!(symbols.lookup("Row"), Lookup::NotAValue("struct"));
    }

    #[test]
    fn callables() {
        let mut symbols = symbols();
        symbols.bind("cap", Binding::Let);
        symbols.bind(
            "f",
            Binding::Func(Callable {
                kind: CalleeKind::Fn,
                min_args: 2,
                max_args: 2,
            }),
        );
        symbols.bind(
            "Zero",
            Binding::Trait {
                methods: vec!["zero"],
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
        enter_query(&mut symbols, "t");
        assert_eq!(symbols.column("dept_id"), ColumnLookup::Unique(1));
        symbols.select(vec!["dept_id", "n"]);

        assert_eq!(symbols.lookup("dept_id"), Lookup::Column(0));
        assert_eq!(symbols.lookup("n"), Lookup::Column(1));
        assert_eq!(symbols.lookup("id"), Lookup::NarrowedAway);
    }

    #[test]
    fn join_concatenates_both_rows() {
        let mut symbols = symbols();
        symbols.bind(
            "depts",
            Binding::Relation {
                row: unqualified(vec!["dept_id"]),
            },
        );
        enter_query(&mut symbols, "t");
        let rhs = symbols.relation("depts", None).expect("depts is bound");
        symbols.concat(rhs);

        assert_eq!(
            symbols.row(),
            &vec![
                Column {
                    qualifier: None,
                    name: "id"
                },
                Column {
                    qualifier: None,
                    name: "dept_id"
                },
                Column {
                    qualifier: None,
                    name: "dept_id"
                },
            ]
        );
        assert_eq!(symbols.column("dept_id"), ColumnLookup::Ambiguous);
    }
}
