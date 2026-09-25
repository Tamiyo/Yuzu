//! What a name means where it is written: every declaration in the
//! program, the scopes open inside a file, and the lookups the walk asks.
//!
//! A name is held as the source wrote it. Only a symbol, the name an op is
//! built under, is interned, since that is the only name MLIR ever sees.

use std::collections::HashMap;
use std::fmt;

use text_size::TextRange;
use yuzu_ast::Visibility;
use yuzu_mlir::attributes::CalleeSource;
use yuzu_types::FunctionRegistry;

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

#[derive(Clone, PartialEq, Eq, Debug)]
struct Column {
    qualifier: Option<String>,
    name: String,
}

impl Column {
    fn matches(&self, reference: Reference<'_>) -> bool {
        self.name == reference.name
            && reference
                .qualifier
                .is_none_or(|qualifier| self.qualifier.as_deref() == Some(qualifier))
    }

    fn reference(&self) -> Reference<'_> {
        Reference {
            qualifier: self.qualifier.as_deref(),
            name: &self.name,
        }
    }
}

/// Two columns may share a name, since a join concatenates both sides, so a
/// column is addressed by position.
#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub(super) struct Row {
    columns: Vec<Column>,
}

impl Row {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn len(&self) -> usize {
        self.columns.len()
    }

    pub(super) fn names(&self) -> impl Iterator<Item = &str> {
        self.columns.iter().map(|column| column.name.as_str())
    }

    pub(super) fn references(&self) -> impl Iterator<Item = Reference<'_>> {
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

    pub(super) fn qualify(&mut self, alias: &str) {
        for column in &mut self.columns {
            column.qualifier = Some(alias.to_string());
        }
    }

    pub(super) fn rename(&mut self, index: usize, name: &str) {
        self.columns[index].name = name.to_string();
    }

    pub(super) fn remove(&mut self, index: usize) {
        self.columns.remove(index);
    }

    pub(super) fn append(&mut self, other: Row) {
        self.columns.extend(other.columns);
    }
}

impl From<Vec<String>> for Row {
    fn from(names: Vec<String>) -> Self {
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

impl From<Vec<&str>> for Row {
    fn from(names: Vec<&str>) -> Self {
        Self::from(names.into_iter().map(String::from).collect::<Vec<_>>())
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) struct Binding {
    pub(super) kind: BindingKind,
    pub(super) text_range: TextRange,
    pub(super) visibility: Visibility,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) enum BindingKind {
    Struct {
        fields: Vec<String>,
    },
    /// A table, or a `let` bound to a query: what `from` and `join` name.
    Relation {
        row: Row,
    },
    Func {
        source: CalleeSource,
        kind: FunctionKind,
        arity: usize,
    },
    /// The methods are not names of their own: dispatch is not written yet.
    Trait {
        methods: Vec<String>,
    },
    /// A `let` bound to a value; the inliner expands it where it is used.
    Let,
    /// A module this file named; what it declares is not in this scope.
    Module {
        path: String,
    },
    /// A `let` the hoist has seen and the second walk has not reached. Its
    /// row is known only once its body is converted.
    Pending,
    /// A declaration in another file. A lookup follows it to where it was
    /// written.
    Import {
        from: Declared,
    },
}

impl BindingKind {
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

impl fmt::Display for BindingKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) struct Local {
    pub(super) name: String,
    pub(super) slot: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum FunctionKind {
    Scalar,
    Aggregate,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) struct Callable {
    pub(super) symbol: String,
    pub(super) source: CalleeSource,
    pub(super) kind: FunctionKind,
    pub(super) min_args: usize,
    pub(super) max_args: usize,
}

impl Callable {
    pub(super) fn constant(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            source: CalleeSource::Const,
            kind: FunctionKind::Scalar,
            min_args: 0,
            max_args: 0,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) enum Lookup {
    Column(usize),
    Local(usize),
    Let(String),
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

enum Scope {
    /// The outermost block of a function body holds its parameters.
    Block { locals: Vec<Local> },
    /// The type parameters of the function being read. Names in the type
    /// namespace, so value lookups pass through it.
    TypeParams { names: Vec<String> },
    /// `narrowed` is the names earlier stages stopped carrying, which get a
    /// diagnostic of their own.
    Relation { row: Row, narrowed: Vec<String> },
}

/// Which module a declaration lives in. The entry file has no path, since
/// nothing can import from it.
#[derive(Clone, Default, PartialEq, Eq, Hash, Debug)]
pub(super) struct ModulePath(Option<String>);

impl ModulePath {
    pub(super) fn entry() -> Self {
        Self(None)
    }

    pub(super) fn from_path(path: &str) -> Self {
        Self(Some(path.to_string()))
    }

    pub(super) fn is_entry(&self) -> bool {
        self.0.is_none()
    }

    pub(super) fn qualify(&self, name: &str) -> Option<String> {
        self.0.as_ref().map(|path| format!("{path}.{name}"))
    }

    pub(super) fn declares(&self, name: &str) -> Declared {
        Declared {
            module: self.clone(),
            name: name.to_string(),
        }
    }
}

/// Where a declaration was written: which file, and the name written there.
/// A module declares a name once, so the pair is unique.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(super) struct Declared {
    pub(super) module: ModulePath,
    pub(super) name: String,
}

impl Declared {
    /// The name its op is built under. MLIR has one namespace for the whole
    /// program, so the module qualifies it.
    pub(super) fn symbol(&self) -> String {
        self.module
            .qualify(&self.name)
            .unwrap_or_else(|| self.name.clone())
    }
}

pub(super) struct SymbolTable {
    /// What each module declares, by the name the source wrote.
    modules: HashMap<ModulePath, HashMap<String, Binding>>,
    module: ModulePath,
    scopes: Vec<Scope>,
}

impl SymbolTable {
    pub(super) fn new() -> Self {
        Self {
            modules: HashMap::new(),
            module: ModulePath::entry(),
            scopes: Vec::new(),
        }
    }

    pub(super) fn module(&self) -> &ModulePath {
        &self.module
    }

    /// An import names a declaration in a file read earlier, so what it
    /// names is known by the time this file is walked, and the binding
    /// itself stands in the import's place from here on.
    pub(super) fn set_module(&mut self, module: ModulePath) {
        self.modules.entry(module.clone()).or_default();
        self.scopes.clear();
        self.module = module;
    }

    pub(super) fn contains_module(&self, module: &ModulePath) -> bool {
        self.modules.contains_key(module)
    }

    pub(super) fn bind(&mut self, name: &str, binding: Binding) {
        self.modules
            .entry(self.module.clone())
            .or_default()
            .insert(name.to_string(), binding);
    }

    /// What this file holds under a name, an import left as it is.
    pub(super) fn binding(&self, name: &str) -> Option<&Binding> {
        self.modules.get(&self.module)?.get(name)
    }

    /// What a name in this file stands for, and where it was declared,
    /// following an import to the file that wrote it.
    pub(super) fn find(&self, name: &str) -> Option<(Declared, &Binding)> {
        self.find_in(self.module.declares(name))
    }

    /// Where a declaration was written, following imports. Each link points
    /// at a file read earlier, so a chain cannot come back around.
    pub(super) fn find_in(&self, at: Declared) -> Option<(Declared, &Binding)> {
        let binding = self.modules.get(&at.module)?.get(&at.name)?;
        match &binding.kind {
            BindingKind::Import { from } => self.find_in(from.clone()),
            _ => Some((at, binding)),
        }
    }

    pub(super) fn is_method(&self, name: &str) -> bool {
        self.modules.get(&self.module).is_some_and(|declarations| {
            declarations.keys().any(|declared| {
                matches!(
                    self.find(declared).map(|(_, binding)| &binding.kind),
                    Some(BindingKind::Trait { methods }) if methods.iter().any(|method| method == name)
                )
            })
        })
    }

    pub(super) fn kind(&self, name: &str) -> Option<&BindingKind> {
        self.find(name).map(|(_, binding)| &binding.kind)
    }

    pub(super) fn struct_symbol(&self, name: &str) -> Option<String> {
        match self.find(name)? {
            (
                at,
                Binding {
                    kind: BindingKind::Struct { .. },
                    ..
                },
            ) => Some(at.symbol()),
            _ => None,
        }
    }

    pub(super) fn trait_symbol(&self, name: &str) -> Option<String> {
        match self.find(name)? {
            (
                at,
                Binding {
                    kind: BindingKind::Trait { .. },
                    ..
                },
            ) => Some(at.symbol()),
            _ => None,
        }
    }

    pub(super) fn module_of(&self, name: &str) -> Option<&str> {
        match self.kind(name)? {
            BindingKind::Module { path } => Some(path),
            _ => None,
        }
    }

    pub(super) fn relation(&self, name: &str, alias: Option<&str>) -> Option<(String, Row)> {
        let (
            at,
            Binding {
                kind: BindingKind::Relation { row },
                ..
            },
        ) = self.find(name)?
        else {
            return None;
        };

        let mut row = row.clone();
        if let Some(alias) = alias {
            row.qualify(alias);
        }

        Some((at.symbol(), row))
    }

    pub(super) fn callable(&self, name: &str, registry: &dyn FunctionRegistry) -> Option<Callable> {
        match self.find(name) {
            Some((
                at,
                Binding {
                    kind: BindingKind::Let,
                    ..
                },
            )) => Some(Callable::constant(&at.symbol())),
            _ => self.operator(name, registry),
        }
    }

    /// The function declared at a place in another module, as a call names it.
    pub(super) fn callable_in(&self, at: &Declared) -> Option<Callable> {
        match self.find_in(at.clone())? {
            (
                at,
                Binding {
                    kind:
                        BindingKind::Func {
                            source,
                            kind,
                            arity,
                        },
                    ..
                },
            ) => Some(Callable {
                symbol: at.symbol(),
                source: *source,
                kind: *kind,
                min_args: *arity,
                max_args: *arity,
            }),
            _ => None,
        }
    }

    /// Like `callable`, except a `let` sharing the name does not stand in.
    pub(super) fn operator(&self, name: &str, registry: &dyn FunctionRegistry) -> Option<Callable> {
        match self.find(name) {
            Some((
                at,
                Binding {
                    kind:
                        BindingKind::Func {
                            source,
                            kind,
                            arity,
                        },
                    ..
                },
            )) => Some(Callable {
                symbol: at.symbol(),
                source: *source,
                kind: *kind,
                min_args: *arity,
                max_args: *arity,
            }),
            _ => self.builtin(name, registry),
        }
    }

    fn builtin(&self, name: &str, registry: &dyn FunctionRegistry) -> Option<Callable> {
        let entry = registry.entries().iter().find(|entry| entry.name == name)?;
        Some(Callable {
            symbol: entry.name.to_string(),
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

    pub(super) fn enter_type_params(&mut self, names: Vec<String>) {
        self.scopes.push(Scope::TypeParams { names });
    }

    /// A type parameter is not a value, so this walks past isolated scopes.
    pub(super) fn is_type_param(&self, name: &str) -> bool {
        self.scopes.iter().rev().any(|scope| {
            matches!(scope, Scope::TypeParams { names } if names.iter().any(|param| param == name))
        })
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
    pub(super) fn bind_local(&mut self, name: &str, slot: usize) {
        let Some(Scope::Block { locals }) = self.scopes.last_mut() else {
            panic!("a local is being bound outside a block")
        };

        locals.push(Local {
            name: name.to_string(),
            slot,
        });
    }

    pub(super) fn enter_relation(&mut self, row: Row) {
        self.scopes.push(Scope::Relation {
            row,
            narrowed: Vec::new(),
        });
    }

    pub(super) fn leave(&mut self) {
        self.scopes.pop();
    }

    pub(super) fn row(&self) -> &Row {
        self.current_row()
            .expect("a stage is being lowered outside a relation")
    }

    pub(super) fn current_row(&self) -> Option<&Row> {
        match self.scopes.last() {
            Some(Scope::Relation { row, .. }) => Some(row),
            _ => None,
        }
    }

    // --- lookups ---

    /// The walk stops at the first isolated scope: a function body and a
    /// stage's region are both `IsolatedFromAbove`. The module's declarations
    /// are symbols, not values, so they answer from any depth.
    pub(super) fn lookup(&self, reference: Reference<'_>) -> Lookup {
        for scope in self.scopes.iter().rev() {
            match scope {
                Scope::TypeParams { .. } => {}
                Scope::Relation { row, narrowed } => {
                    match row.column(reference) {
                        ColumnLookup::Unique(index) => return Lookup::Column(index),
                        ColumnLookup::Ambiguous => return Lookup::Ambiguous,
                        ColumnLookup::Absent
                            if narrowed.iter().any(|name| name == reference.name) =>
                        {
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

    fn module_lookup(&self, reference: Reference<'_>) -> Lookup {
        if reference.qualifier.is_some() {
            return Lookup::Unknown;
        }

        match self.find(reference.name) {
            Some((
                at,
                Binding {
                    kind: BindingKind::Let,
                    ..
                },
            )) => Lookup::Let(at.symbol()),
            Some((
                _,
                Binding {
                    kind: BindingKind::Pending,
                    ..
                },
            )) => Lookup::NotYet,
            Some((_, binding)) => Lookup::NotAValue(binding.kind.name()),
            None => Lookup::Unknown,
        }
    }

    pub(super) fn column(&self, reference: Reference<'_>) -> ColumnLookup {
        self.row().column(reference)
    }

    // --- what each stage does to the row ---

    fn replace_row(&mut self, next: Row) {
        let Some(Scope::Relation { row, narrowed }) = self.scopes.last_mut() else {
            panic!("a stage is being lowered outside a relation")
        };

        for name in row.names() {
            if !next.has(Reference::unqualified(name))
                && !narrowed.iter().any(|narrowed| narrowed == name)
            {
                narrowed.push(name.to_string());
            }
        }

        *row = next;
    }

    pub(super) fn alias(&mut self, alias: &str) {
        let Some(Scope::Relation { row, .. }) = self.scopes.last_mut() else {
            panic!("a stage is being lowered outside a relation")
        };

        row.qualify(alias);
    }

    pub(super) fn replace(&mut self, names: Vec<String>) {
        self.replace_row(Row::from(names));
    }

    pub(super) fn extend(&mut self, names: Vec<String>) {
        let mut next = self.row().clone();
        next.append(Row::from(names));
        self.replace_row(next);
    }

    pub(super) fn remove(&mut self, index: usize) {
        let mut next = self.row().clone();
        next.remove(index);
        self.replace_row(next);
    }

    pub(super) fn rename(&mut self, renames: &[(usize, String)]) {
        let mut next = self.row().clone();
        for (index, to) in renames {
            next.rename(*index, to);
        }

        self.replace_row(next);
    }

    pub(super) fn concat(&mut self, rhs: Row) {
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

    fn strings(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| name.to_string()).collect()
    }

    fn symbols() -> SymbolTable {
        let mut symbols = SymbolTable::new();
        symbols.bind(
            "Row",
            Binding {
                kind: BindingKind::Struct {
                    fields: strings(&["id", "dept_id"]),
                },
                text_range: TextRange::default(),
                visibility: Visibility::Private,
            },
        );
        bind_relation(&mut symbols, "t", vec!["id", "dept_id"]);
        symbols
    }

    fn bind_relation(symbols: &mut SymbolTable, name: &'static str, row: Vec<&'static str>) {
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

    fn bind_let(symbols: &mut SymbolTable, name: &'static str) {
        symbols.bind(
            name,
            Binding {
                kind: BindingKind::Let,
                text_range: TextRange::default(),
                visibility: Visibility::Private,
            },
        );
    }

    fn enter_relation(symbols: &mut SymbolTable, relation: &'static str) {
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

        assert_eq!(symbols.lookup(bare("cap")), Lookup::Let("cap".to_string()));
        symbols.enter_block();
        symbols.bind_local("cap", 0);
        assert_eq!(symbols.lookup(bare("cap")), Lookup::Local(0));
        symbols.enter_block();
        symbols.bind_local("cap", 7);
        assert_eq!(symbols.lookup(bare("cap")), Lookup::Local(7));
        symbols.leave();
        assert_eq!(symbols.lookup(bare("cap")), Lookup::Local(0));
        symbols.leave();
        assert_eq!(symbols.lookup(bare("cap")), Lookup::Let("cap".to_string()));
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
        assert_eq!(symbols.lookup(bare("cap")), Lookup::Let("cap".to_string()));
    }

    #[test]
    fn a_declaration_that_is_not_a_value_says_so() {
        let symbols = symbols();

        assert_eq!(symbols.lookup(bare("t")), Lookup::NotAValue("relation"));
        assert_eq!(symbols.lookup(bare("Row")), Lookup::NotAValue("struct"));
    }

    #[test]
    fn a_symbol_is_qualified_by_the_module_that_declared_it() {
        let mut symbols = SymbolTable::new();
        symbols.set_module(ModulePath::from_path("helpers"));
        symbols.bind(
            "Row",
            Binding {
                kind: BindingKind::Struct {
                    fields: strings(&["a"]),
                },
                text_range: TextRange::default(),
                visibility: Visibility::Private,
            },
        );
        symbols.bind(
            "Show",
            Binding {
                kind: BindingKind::Trait {
                    methods: strings(&["show"]),
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

        assert_eq!(symbols.struct_symbol("Row").as_deref(), Some("helpers.Row"));
        assert_eq!(
            symbols.trait_symbol("Show").as_deref(),
            Some("helpers.Show")
        );
        assert_eq!(
            symbols
                .callable("f", &yuzu_types::Builtins)
                .map(|callable| callable.symbol),
            Some("helpers.f".to_string())
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
                    methods: strings(&["zero"]),
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
                symbol: "f".to_string(),
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
        symbols.replace(strings(&["dept_id", "n"]));

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
