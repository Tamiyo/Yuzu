//! What a name means where it is used. The resolver owns every scope the
//! emitter walks — module declarations at the bottom, a function's
//! parameters or a stage's row above — and answers lookups with positions and
//! symbols, never with IR: a column is an index the traversal turns into a
//! block argument, a module-level `let` is a symbol it turns into a call.
//!
//! Holding positions rather than values is what keeps this out of the
//! traversal. Block arguments of different regions have different lifetimes,
//! and a resolver that stored them would depend on the builder; one that
//! stores where a name lives depends on nothing.
//!
//! The row is the one piece of state a query threads from stage to stage.
//! Each stage's transition — what `select` names, what `join` concatenates,
//! what `drop` sheds — is resolution, so it is decided here and the traversal
//! is handed the indices the op needs as attributes.

use std::collections::{HashMap, HashSet};

use yuzu_mlir::attributes::CalleeKind;

/// A column the query carries at some stage: its name, and the alias
/// qualifying it once an `as` or a join has named its side.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) struct Column {
    pub(super) qualifier: Option<String>,
    pub(super) name: String,
}

impl Column {
    fn matches(&self, reference: &str) -> bool {
        match reference.split_once('.') {
            Some((qualifier, name)) => {
                self.qualifier.as_deref() == Some(qualifier) && self.name == name
            }
            None => self.name == reference,
        }
    }
}

pub(super) type Row = Vec<Column>;

/// What a name in expression position resolved to.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) enum Lookup {
    /// A column of the row the enclosing stage sees, by position.
    Column(usize),
    /// A parameter of the enclosing function, by position.
    Param(usize),
    /// A module-level `let`: the value lives in its body, and a use is a
    /// call for expansion to inline.
    Let(String),
    /// Two columns answer to the name; a qualifier would pick one.
    Ambiguous,
    /// An earlier stage carried the column and narrowed it away — a
    /// different mistake from a name that never existed.
    NarrowedAway,
    Unknown,
}

/// What a callee resolved to, and the argument count it accepts.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Callable {
    pub(super) kind: CalleeKind,
    pub(super) min_args: usize,
    pub(super) max_args: usize,
}

/// Why a callee did not resolve.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) enum CallError {
    /// A trait declares it, so no module-level name reaches one: choosing
    /// an implementation is dispatch, which nothing does yet.
    TraitMethod,
    Unknown,
}

/// A scope the traversal is inside. Only what can be named by position
/// lives here; module declarations sit on the resolver itself.
enum Scope {
    Function { params: Vec<String> },
    Stage { row: Row, shed: Vec<String> },
}

pub(super) struct Resolver {
    scopes: Vec<Scope>,
    structs: HashMap<String, Vec<String>>,
    /// Tables and query-valued lets: what `from` and `join` may name.
    relations: HashMap<String, Row>,
    callables: HashMap<String, Callable>,
    traits: HashSet<String>,
    /// Methods the traits declare — known, so a call to one can say so.
    methods: HashSet<String>,
    /// Scalar-valued lets: what an expression may name.
    lets: HashSet<String>,
}

impl Resolver {
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

    // --- hoisting: every declaration registers before anything resolves ---

    /// Whether the name is already a declaration of any kind; the caller
    /// reports the duplicate with the span it has.
    pub(super) fn is_declared(&self, name: &str) -> bool {
        self.structs.contains_key(name)
            || self.relations.contains_key(name)
            || self.callables.contains_key(name)
            || self.traits.contains(name)
            || self.lets.contains(name)
    }

    pub(super) fn declare_struct(&mut self, name: &str, fields: Vec<String>) {
        self.structs.insert(name.to_string(), fields);
    }

    /// A table over a declared struct; `None` when no such struct exists.
    pub(super) fn declare_table(&mut self, name: &str, row: &str) -> Option<()> {
        let fields = self.structs.get(row)?.clone();
        self.relations.insert(name.to_string(), unqualified(fields));
        Some(())
    }

    pub(super) fn declare_trait(&mut self, name: &str, methods: impl IntoIterator<Item = String>) {
        self.traits.insert(name.to_string());
        self.methods.extend(methods);
    }

    pub(super) fn declare_function(&mut self, name: &str, arity: usize, kind: CalleeKind) {
        self.callables.insert(
            name.to_string(),
            Callable {
                kind,
                min_args: arity,
                max_args: arity,
            },
        );
    }

    /// A `let` whose body is a query: what `from` may name, once its row is
    /// known — which is after the body is emitted, so the row comes later.
    pub(super) fn declare_query_let(&mut self, name: &str, row: Row) {
        self.relations.insert(name.to_string(), row);
    }

    /// A `let` whose body is a value: what an expression may name.
    pub(super) fn declare_scalar_let(&mut self, name: &str) {
        self.lets.insert(name.to_string());
    }

    pub(super) fn has_trait(&self, name: &str) -> bool {
        self.traits.contains(name)
    }

    /// Whether a name denotes a type an `impl` can target.
    pub(super) fn is_type_name(&self, name: &str) -> bool {
        matches!(name, "int64" | "float64" | "bool" | "str") || self.structs.contains_key(name)
    }

    // --- scopes ---

    pub(super) fn enter_function(&mut self, params: Vec<String>) {
        self.scopes.push(Scope::Function { params });
    }

    /// A query begins with the row its source relation carries.
    pub(super) fn enter_query(&mut self, relation: &str) -> Option<()> {
        let row = self.relations.get(relation)?.clone();
        self.scopes.push(Scope::Stage {
            row,
            shed: Vec::new(),
        });
        Some(())
    }

    /// A query over a relation nobody declared: the error is reported where
    /// the name is, and conversion carries on against an empty row so the
    /// rest of the pipeline is still checked.
    pub(super) fn enter_unknown_query(&mut self) {
        self.scopes.push(Scope::Stage {
            row: Row::new(),
            shed: Vec::new(),
        });
    }

    pub(super) fn leave(&mut self) {
        self.scopes.pop();
    }

    /// Whether the innermost scope is a stage — which decides whether an
    /// unknown name is reported as a column or as a name.
    pub(super) fn in_query(&self) -> bool {
        matches!(self.scopes.last(), Some(Scope::Stage { .. }))
    }

    /// The row the enclosing stage sees, for building block arguments.
    pub(super) fn row(&self) -> &Row {
        match self.scopes.last() {
            Some(Scope::Stage { row, .. }) => row,
            Some(Scope::Function { .. }) | None => {
                panic!("a stage is being resolved outside a query")
            }
        }
    }

    // --- lookups ---

    pub(super) fn lookup(&self, reference: &str) -> Lookup {
        match self.scopes.last() {
            Some(Scope::Stage { row, shed }) => self.lookup_column(row, shed, reference),
            Some(Scope::Function { params }) => match params.iter().position(|p| p == reference) {
                Some(index) => Lookup::Param(index),
                None => self.lookup_module(reference),
            },
            None => self.lookup_module(reference),
        }
    }

    fn lookup_column(&self, row: &Row, shed: &[String], reference: &str) -> Lookup {
        let mut matches = row.iter().enumerate().filter(|(_, c)| c.matches(reference));
        match (matches.next(), matches.next()) {
            (Some((index, _)), None) => Lookup::Column(index),
            (Some(_), Some(_)) => Lookup::Ambiguous,
            (None, _) => {
                let column = reference.split_once('.').map_or(reference, |(_, n)| n);
                if shed.iter().any(|s| s == column) {
                    return Lookup::NarrowedAway;
                }

                self.lookup_module(reference)
            }
        }
    }

    fn lookup_module(&self, reference: &str) -> Lookup {
        if self.lets.contains(reference) {
            Lookup::Let(reference.to_string())
        } else {
            Lookup::Unknown
        }
    }

    /// A callee by name, from the module's functions and lets or the
    /// registry the caller supplies.
    pub(super) fn callable(
        &self,
        name: &str,
        registry: &dyn yuzu_types::FunctionRegistry,
    ) -> Result<Callable, CallError> {
        if let Some(callable) = self.callables.get(name) {
            return Ok(*callable);
        }

        if self.lets.contains(name) {
            return Ok(Callable {
                kind: CalleeKind::Let,
                min_args: 0,
                max_args: 0,
            });
        }

        if let Some(entry) = registry.entries().iter().find(|entry| entry.name == name) {
            return Ok(Callable {
                kind: CalleeKind::Builtin,
                min_args: entry.min_args,
                max_args: entry.max_args,
            });
        }

        if self.methods.contains(name) {
            return Err(CallError::TraitMethod);
        }

        Err(CallError::Unknown)
    }

    // --- stage transitions ---
    //
    // Each replaces the current row with the one the stage produces, and
    // hands back whatever the op needs to carry as an attribute. Names the
    // stage stops carrying join the shed list, so a later reference to one
    // can say so.

    fn stage(&mut self) -> (&mut Row, &mut Vec<String>) {
        match self.scopes.last_mut() {
            Some(Scope::Stage { row, shed }) => (row, shed),
            Some(Scope::Function { .. }) | None => {
                panic!("a stage is being resolved outside a query")
            }
        }
    }

    fn replace_row(&mut self, next: Row) {
        let (row, shed) = self.stage();
        for column in row.iter() {
            if !next.iter().any(|kept| kept.name == column.name) && !shed.contains(&column.name) {
                shed.push(column.name.clone());
            }
        }

        *row = next;
    }

    pub(super) fn alias(&mut self, alias: &str) {
        let (row, _) = self.stage();
        for column in row.iter_mut() {
            column.qualifier = Some(alias.to_string());
        }
    }

    pub(super) fn select(&mut self, names: Vec<String>) {
        self.replace_row(unqualified(names));
    }

    pub(super) fn extend(&mut self, names: Vec<String>) {
        let mut next = self.row().clone();
        next.extend(unqualified(names));
        self.replace_row(next);
    }

    /// The indices the named columns occupy; `Err` names the first that is
    /// not in the row.
    pub(super) fn set(&mut self, names: &[String]) -> Result<Vec<usize>, String> {
        let row = self.row();
        names
            .iter()
            .map(|name| {
                row.iter()
                    .position(|c| c.matches(name))
                    .ok_or_else(|| name.clone())
            })
            .collect()
    }

    /// Resolution removes the first column each name matches, so dropping
    /// one name twice drops two columns.
    pub(super) fn drop(&mut self, names: &[String]) -> Result<(), String> {
        let mut next = self.row().clone();
        for name in names {
            match next.iter().position(|c| c.matches(name)) {
                Some(index) => {
                    next.remove(index);
                }
                None => return Err(name.clone()),
            }
        }

        self.replace_row(next);
        Ok(())
    }

    /// The index each renamed column occupies; `Err` names the first that is
    /// not in the row.
    pub(super) fn rename(&mut self, from: &[String], to: &[String]) -> Result<Vec<usize>, String> {
        let mut next = self.row().clone();
        let mut indices = Vec::with_capacity(from.len());
        for (from, to) in from.iter().zip(to) {
            match next.iter().position(|c| c.matches(from)) {
                Some(index) => {
                    indices.push(index);
                    next[index].name = to.clone();
                }
                None => return Err(from.clone()),
            }
        }

        self.replace_row(next);
        Ok(indices)
    }

    /// The row a grouping produces is the keys, in order, then the measures.
    /// `Err` is the first key not in the row.
    pub(super) fn aggregate(
        &mut self,
        group_by: &[String],
        names: Vec<String>,
    ) -> Result<Vec<usize>, Lookup> {
        let row = self.row();
        let mut keys = Vec::with_capacity(group_by.len());
        let mut next = Row::with_capacity(group_by.len() + names.len());
        for name in group_by {
            let mut matches = row.iter().enumerate().filter(|(_, c)| c.matches(name));
            match (matches.next(), matches.next()) {
                (Some((index, column)), None) => {
                    keys.push(index);
                    next.push(column.clone());
                }
                (Some(_), Some(_)) => return Err(Lookup::Ambiguous),
                (None, _) => return Err(Lookup::Unknown),
            }
        }

        next.extend(unqualified(names));
        self.replace_row(next);
        Ok(keys)
    }

    /// Both sides carry through, the right qualified by its alias when it
    /// has one. `Err` is the first `using` column absent from a side.
    pub(super) fn join(
        &mut self,
        relation: &str,
        alias: Option<&str>,
        using: &[String],
    ) -> Result<(), JoinError> {
        let Some(rhs) = self.relations.get(relation).cloned() else {
            return Err(JoinError::UnknownRelation);
        };

        let rhs: Row = rhs
            .into_iter()
            .map(|column| Column {
                qualifier: alias.map(str::to_string).or(column.qualifier),
                name: column.name,
            })
            .collect();

        let lhs = self.row();
        for name in using {
            for side in [lhs, &rhs] {
                if !side.iter().any(|c| c.matches(name)) {
                    return Err(JoinError::UsingColumn(name.clone()));
                }
            }
        }

        let mut next = lhs.clone();
        next.extend(rhs);
        self.replace_row(next);
        Ok(())
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) enum JoinError {
    UnknownRelation,
    UsingColumn(String),
}

fn unqualified(names: Vec<String>) -> Row {
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
    use super::{CallError, Column, JoinError, Lookup, Resolver};
    use yuzu_mlir::attributes::CalleeKind;

    fn resolver_with_table() -> Resolver {
        let mut resolver = Resolver::new();
        resolver.declare_struct("Row", vec!["id".into(), "dept_id".into()]);
        resolver.declare_table("t", "Row").expect("Row is declared");
        resolver
    }

    #[test]
    fn a_column_resolves_by_position_in_the_row() {
        let mut resolver = resolver_with_table();
        resolver.enter_query("t").expect("t is a relation");

        assert_eq!(resolver.lookup("dept_id"), Lookup::Column(1));
        assert_eq!(resolver.lookup("nope"), Lookup::Unknown);
    }

    /// A qualifier is spent choosing between two columns of one name; once
    /// it has, only the position remains.
    #[test]
    fn a_qualifier_picks_between_same_named_columns() {
        let mut resolver = resolver_with_table();
        resolver.declare_struct("Dept", vec!["id".into()]);
        resolver
            .declare_table("depts", "Dept")
            .expect("Dept is declared");
        resolver.enter_query("t").expect("t is a relation");
        resolver.alias("a");
        resolver.join("depts", Some("d"), &[]).expect("depts joins");

        assert_eq!(resolver.lookup("id"), Lookup::Ambiguous);
        assert_eq!(resolver.lookup("a.id"), Lookup::Column(0));
        assert_eq!(resolver.lookup("d.id"), Lookup::Column(2));
    }

    /// Stages narrow the row in written order; a name a stage shed is
    /// reported as narrowed away, not as never having existed.
    #[test]
    fn a_shed_column_is_narrowed_away_not_unknown() {
        let mut resolver = resolver_with_table();
        resolver.enter_query("t").expect("t is a relation");
        resolver.drop(&["id".into()]).expect("id is in the row");

        assert_eq!(resolver.lookup("id"), Lookup::NarrowedAway);
        assert_eq!(resolver.lookup("dept_id"), Lookup::Column(0));
    }

    /// Module scope sits under every other: a scalar `let` is visible from a
    /// stage, a function body, or nowhere in particular.
    #[test]
    fn a_scalar_let_is_visible_from_every_scope() {
        let mut resolver = resolver_with_table();
        resolver.declare_scalar_let("cap");

        assert_eq!(resolver.lookup("cap"), Lookup::Let("cap".into()));
        resolver.enter_function(vec!["x".into()]);
        assert_eq!(resolver.lookup("x"), Lookup::Param(0));
        assert_eq!(resolver.lookup("cap"), Lookup::Let("cap".into()));
        resolver.leave();
        resolver.enter_query("t").expect("t is a relation");
        assert_eq!(resolver.lookup("cap"), Lookup::Let("cap".into()));
    }

    /// Innermost wins: a column shadows a module-level let of the same name.
    #[test]
    fn a_column_shadows_a_module_let() {
        let mut resolver = resolver_with_table();
        resolver.declare_scalar_let("id");
        resolver.enter_query("t").expect("t is a relation");

        assert_eq!(resolver.lookup("id"), Lookup::Column(0));
    }

    #[test]
    fn a_let_is_callable_with_no_arguments() {
        let mut resolver = resolver_with_table();
        resolver.declare_scalar_let("cap");
        resolver.declare_function("f", 2, CalleeKind::Fn);
        resolver.declare_trait("Zero", ["zero".into()]);

        let cap = resolver
            .callable("cap", &yuzu_types::Builtins)
            .expect("cap is callable");
        assert_eq!(
            (cap.kind, cap.min_args, cap.max_args),
            (CalleeKind::Let, 0, 0)
        );
        let f = resolver
            .callable("f", &yuzu_types::Builtins)
            .expect("f is callable");
        assert_eq!((f.kind, f.min_args), (CalleeKind::Fn, 2));
        assert_eq!(
            resolver.callable("zero", &yuzu_types::Builtins),
            Err(CallError::TraitMethod)
        );
        assert_eq!(
            resolver.callable("nope", &yuzu_types::Builtins),
            Err(CallError::Unknown)
        );
    }

    #[test]
    fn a_grouping_puts_keys_before_measures() {
        let mut resolver = resolver_with_table();
        resolver.enter_query("t").expect("t is a relation");
        let keys = resolver
            .aggregate(&["dept_id".into()], vec!["n".into()])
            .expect("dept_id is in the row");

        assert_eq!(keys, vec![1]);
        assert_eq!(resolver.lookup("dept_id"), Lookup::Column(0));
        assert_eq!(resolver.lookup("n"), Lookup::Column(1));
        assert_eq!(resolver.lookup("id"), Lookup::NarrowedAway);
    }

    #[test]
    fn using_needs_the_column_on_both_sides() {
        let mut resolver = resolver_with_table();
        resolver.declare_struct("Dept", vec!["dept_id".into()]);
        resolver
            .declare_table("depts", "Dept")
            .expect("Dept is declared");
        resolver.enter_query("t").expect("t is a relation");

        assert_eq!(
            resolver.join("depts", None, &["id".into()]),
            Err(JoinError::UsingColumn("id".into()))
        );
        assert_eq!(resolver.join("depts", None, &["dept_id".into()]), Ok(()));
        assert_eq!(
            resolver.row(),
            &vec![
                Column {
                    qualifier: None,
                    name: "id".into()
                },
                Column {
                    qualifier: None,
                    name: "dept_id".into()
                },
                Column {
                    qualifier: None,
                    name: "dept_id".into()
                },
            ]
        );
    }
}
