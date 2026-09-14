//! The names in scope while the AST is converted: module declarations at the
//! bottom, a function's parameters or a query stage's row above. Answers are
//! positions and symbols, never IR — a column is the index of a block
//! argument, a module-level `let` is a symbol to call — so the resolver holds
//! nothing tied to a builder. Names are `&'c str` interned in the MLIR
//! context.

use std::collections::{HashMap, HashSet};

use yuzu_mlir::attributes::CalleeKind;
use yuzu_types::FunctionRegistry;

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

pub(super) fn has_column(row: &[Column<'_>], name: &str) -> bool {
    row.iter().any(|column| column.matches(name))
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Lookup<'c> {
    Column(usize),
    Param(usize),
    Let(&'c str),
    Ambiguous,
    /// An earlier stage carried the column and narrowed it away.
    NarrowedAway,
    Unknown,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum ColumnLookup {
    Unique(usize),
    Ambiguous,
    Absent,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Callable {
    pub(super) kind: CalleeKind,
    pub(super) min_args: usize,
    pub(super) max_args: usize,
}

enum Scope<'c> {
    Function { params: Vec<&'c str> },
    Stage { row: Row<'c>, shed: Vec<&'c str> },
}

pub(super) struct Resolver<'c> {
    scopes: Vec<Scope<'c>>,
    structs: HashMap<&'c str, Vec<&'c str>>,
    relations: HashMap<&'c str, Row<'c>>,
    callables: HashMap<&'c str, Callable>,
    traits: HashSet<&'c str>,
    methods: HashSet<&'c str>,
    lets: HashSet<&'c str>,
}

impl<'c> Resolver<'c> {
    pub(super) fn new() -> Self {
        Self {
            scopes: Vec::new(),
            structs: HashMap::new(),
            relations: HashMap::new(),
            callables: HashMap::new(),
            traits: HashSet::new(),
            methods: HashSet::new(),
            lets: HashSet::new(),
        }
    }

    pub(super) fn is_declared(&self, name: &str) -> bool {
        self.structs.contains_key(name)
            || self.relations.contains_key(name)
            || self.callables.contains_key(name)
            || self.traits.contains(name)
            || self.lets.contains(name)
    }

    pub(super) fn declare_struct(&mut self, name: &'c str, fields: Vec<&'c str>) {
        self.structs.insert(name, fields);
    }

    pub(super) fn declare_table(&mut self, name: &'c str, row: &str) -> Option<()> {
        let fields = self.structs.get(row)?.clone();
        self.relations.insert(name, unqualified(fields));
        Some(())
    }

    pub(super) fn declare_inline_table(&mut self, name: &'c str, fields: Vec<&'c str>) {
        self.relations.insert(name, unqualified(fields.clone()));
        self.structs.insert(name, fields);
    }

    pub(super) fn declare_trait(&mut self, name: &'c str, methods: Vec<&'c str>) {
        self.traits.insert(name);
        self.methods.extend(methods);
    }

    pub(super) fn declare_function(&mut self, name: &'c str, arity: usize, kind: CalleeKind) {
        self.callables.insert(
            name,
            Callable {
                kind,
                min_args: arity,
                max_args: arity,
            },
        );
    }

    pub(super) fn declare_query_let(&mut self, name: &'c str, row: Row<'c>) {
        self.relations.insert(name, row);
    }

    pub(super) fn declare_scalar_let(&mut self, name: &'c str) {
        self.lets.insert(name);
    }

    pub(super) fn has_trait(&self, name: &str) -> bool {
        self.traits.contains(name)
    }

    pub(super) fn is_method(&self, name: &str) -> bool {
        self.methods.contains(name)
    }

    pub(super) fn is_type_name(&self, name: &str) -> bool {
        matches!(name, "int64" | "float64" | "bool" | "str") || self.structs.contains_key(name)
    }

    pub(super) fn enter_function(&mut self, params: Vec<&'c str>) {
        self.scopes.push(Scope::Function { params });
    }

    /// Enters a query over the relation's row. An unknown relation enters an
    /// empty row instead, so the stages after it are still checked.
    pub(super) fn enter_query(&mut self, relation: &str) -> Option<()> {
        let row = self.relations.get(relation).cloned();
        self.scopes.push(Scope::Stage {
            row: row.clone().unwrap_or_default(),
            shed: Vec::new(),
        });
        row.map(|_| ())
    }

    pub(super) fn leave(&mut self) {
        self.scopes.pop();
    }

    pub(super) fn row(&self) -> &Row<'c> {
        match self.scopes.last() {
            Some(Scope::Stage { row, .. }) => row,
            Some(Scope::Function { .. }) | None => {
                panic!("a stage is being resolved outside a query")
            }
        }
    }

    pub(super) fn lookup(&self, reference: &str) -> Lookup<'c> {
        match self.scopes.last() {
            Some(Scope::Stage { row, shed }) => match self.column_of(row, reference) {
                ColumnLookup::Unique(index) => return Lookup::Column(index),
                ColumnLookup::Ambiguous => return Lookup::Ambiguous,
                ColumnLookup::Absent => {
                    let column = reference.split_once('.').map_or(reference, |(_, n)| n);
                    if shed.contains(&column) {
                        return Lookup::NarrowedAway;
                    }
                }
            },
            Some(Scope::Function { params }) => {
                if let Some(index) = params.iter().position(|&param| param == reference) {
                    return Lookup::Param(index);
                }
            }
            None => {}
        }

        match self.lets.get(reference) {
            Some(&name) => Lookup::Let(name),
            None => Lookup::Unknown,
        }
    }

    pub(super) fn callable(&self, name: &str, registry: &dyn FunctionRegistry) -> Option<Callable> {
        if let Some(callable) = self.callables.get(name) {
            return Some(*callable);
        }

        if self.lets.contains(name) {
            return Some(Callable {
                kind: CalleeKind::Let,
                min_args: 0,
                max_args: 0,
            });
        }

        let entry = registry.entries().iter().find(|entry| entry.name == name)?;
        Some(Callable {
            kind: CalleeKind::Builtin,
            min_args: entry.min_args,
            max_args: entry.max_args,
        })
    }

    /// The column of the current row a stage item names.
    pub(super) fn column(&self, reference: &str) -> ColumnLookup {
        self.column_of(self.row(), reference)
    }

    fn column_of(&self, row: &Row<'c>, reference: &str) -> ColumnLookup {
        let mut matches = row.iter().enumerate().filter(|(_, c)| c.matches(reference));
        match (matches.next(), matches.next()) {
            (Some((index, _)), None) => ColumnLookup::Unique(index),
            (Some(_), Some(_)) => ColumnLookup::Ambiguous,
            (None, _) => ColumnLookup::Absent,
        }
    }

    /// A relation's row as a join's right side, qualified by its alias.
    pub(super) fn relation(&self, name: &str, alias: Option<&'c str>) -> Option<Row<'c>> {
        let row = self.relations.get(name)?;
        Some(
            row.iter()
                .map(|&column| Column {
                    qualifier: alias.or(column.qualifier),
                    name: column.name,
                })
                .collect(),
        )
    }

    // Stage transitions replace the current row. Names the stage stops
    // carrying are remembered as shed, so a later reference to one can be
    // told apart from a name that never existed.

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

fn unqualified<'c>(names: Vec<&'c str>) -> Row<'c> {
    names
        .into_iter()
        .map(|name| Column {
            qualifier: None,
            name,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use yuzu_mlir::attributes::CalleeKind;

    use super::{Callable, Column, ColumnLookup, Lookup, Resolver};

    fn resolver() -> Resolver<'static> {
        let mut resolver = Resolver::new();
        resolver.declare_struct("Row", vec!["id", "dept_id"]);
        resolver.declare_table("t", "Row").expect("Row is declared");
        resolver
    }

    #[test]
    fn column_lookup() {
        let mut resolver = resolver();
        resolver.enter_query("t").expect("t is a relation");

        assert_eq!(resolver.lookup("dept_id"), Lookup::Column(1));
        assert_eq!(resolver.lookup("nope"), Lookup::Unknown);
    }

    #[test]
    fn qualifier_picks_between_same_named_columns() {
        let mut resolver = resolver();
        resolver.declare_struct("Dept", vec!["id"]);
        resolver
            .declare_table("depts", "Dept")
            .expect("Dept is declared");
        resolver.enter_query("t").expect("t is a relation");
        resolver.alias("a");
        let rhs = resolver
            .relation("depts", Some("d"))
            .expect("depts is a relation");
        resolver.concat(rhs);

        assert_eq!(resolver.lookup("id"), Lookup::Ambiguous);
        assert_eq!(resolver.lookup("a.id"), Lookup::Column(0));
        assert_eq!(resolver.lookup("d.id"), Lookup::Column(2));
    }

    #[test]
    fn shed_column_is_narrowed_away() {
        let mut resolver = resolver();
        resolver.enter_query("t").expect("t is a relation");
        resolver.remove(0);

        assert_eq!(resolver.lookup("id"), Lookup::NarrowedAway);
        assert_eq!(resolver.lookup("dept_id"), Lookup::Column(0));
    }

    #[test]
    fn scalar_let_is_visible_from_every_scope() {
        let mut resolver = resolver();
        resolver.declare_scalar_let("cap");

        assert_eq!(resolver.lookup("cap"), Lookup::Let("cap"));
        resolver.enter_function(vec!["x"]);
        assert_eq!(resolver.lookup("x"), Lookup::Param(0));
        assert_eq!(resolver.lookup("cap"), Lookup::Let("cap"));
        resolver.leave();
        resolver.enter_query("t").expect("t is a relation");
        assert_eq!(resolver.lookup("cap"), Lookup::Let("cap"));
    }

    #[test]
    fn column_shadows_a_module_let() {
        let mut resolver = resolver();
        resolver.declare_scalar_let("id");
        resolver.enter_query("t").expect("t is a relation");

        assert_eq!(resolver.lookup("id"), Lookup::Column(0));
    }

    #[test]
    fn unknown_relation_enters_an_empty_row() {
        let mut resolver = resolver();
        assert_eq!(resolver.enter_query("nope"), None);
        assert!(resolver.row().is_empty());
        assert_eq!(resolver.lookup("id"), Lookup::Unknown);
    }

    #[test]
    fn callables() {
        let mut resolver = resolver();
        resolver.declare_scalar_let("cap");
        resolver.declare_function("f", 2, CalleeKind::Fn);
        resolver.declare_trait("Zero", vec!["zero"]);

        let registry = &yuzu_types::Builtins;
        assert_eq!(
            resolver.callable("cap", registry),
            Some(Callable {
                kind: CalleeKind::Let,
                min_args: 0,
                max_args: 0
            })
        );
        assert_eq!(
            resolver.callable("f", registry),
            Some(Callable {
                kind: CalleeKind::Fn,
                min_args: 2,
                max_args: 2
            })
        );
        assert_eq!(
            resolver.callable("sum", registry).map(|c| c.kind),
            Some(CalleeKind::Builtin)
        );
        assert_eq!(resolver.callable("zero", registry), None);
        assert!(resolver.is_method("zero"));
    }

    #[test]
    fn select_replaces_the_row() {
        let mut resolver = resolver();
        resolver.enter_query("t").expect("t is a relation");
        assert_eq!(resolver.column("dept_id"), ColumnLookup::Unique(1));
        resolver.select(vec!["dept_id", "n"]);

        assert_eq!(resolver.lookup("dept_id"), Lookup::Column(0));
        assert_eq!(resolver.lookup("n"), Lookup::Column(1));
        assert_eq!(resolver.lookup("id"), Lookup::NarrowedAway);
    }

    #[test]
    fn join_concatenates_both_rows() {
        let mut resolver = resolver();
        resolver.declare_struct("Dept", vec!["dept_id"]);
        resolver
            .declare_table("depts", "Dept")
            .expect("Dept is declared");
        resolver.enter_query("t").expect("t is a relation");
        let rhs = resolver
            .relation("depts", None)
            .expect("depts is a relation");
        resolver.concat(rhs);

        assert_eq!(
            resolver.row(),
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
        assert_eq!(resolver.column("dept_id"), ColumnLookup::Ambiguous);
    }
}
