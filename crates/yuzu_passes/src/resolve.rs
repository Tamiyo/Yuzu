//! ResolveNames: binds every name in a yzl module — relations, columns,
//! parameters, callees — and records each answer as an attribute for the
//! conversions downstream. Column resolution needs only the *names* flowing
//! through the query, which the syntax fully determines; the types stay
//! unresolved for InferTypes.

use std::collections::{HashMap, HashSet};

use melior::Context;
use melior::ir::attribute::StringAttribute;
use melior::ir::operation::{OperationLike, OperationMutLike, OperationRefMut};
use melior::ir::{BlockRef, Module, RegionLike};
use yuzu_mlir::ext::{ArrayAttributeExt, BlockExt, OperationMutExt, RegionExt};
use yuzu_mlir::ops::yzl::YzlOperationRef;
use yuzu_mlir::value_id;
use yuzu_types::Registry;

/// A column the query carries at some stage: its name, and the alias
/// qualifying it when an `alias` stage or a join has named its side.
#[derive(Clone, PartialEq)]
struct Column {
    qualifier: Option<String>,
    name: String,
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

type Schema = Vec<Column>;

/// What a callable name resolved to, recorded as the op's `callee_kind`.
struct Callable {
    kind: &'static str,
    min_args: usize,
    max_args: usize,
}

/// The columns a region's names may refer to.
enum Ambient<'a> {
    Columns(&'a Schema),
    Params(&'a [String]),
    None,
}

struct Resolver<'c, 'a> {
    context: &'c Context,
    registry: &'a dyn Registry,
    structs: HashMap<String, Schema>,
    relations: HashMap<String, Schema>,
    callables: HashMap<String, Callable>,
    /// The declared traits, which bounds and impls refer to.
    traits: HashSet<String>,
    schemas: HashMap<usize, Schema>,
}

/// Expects a verified module: required ODS attributes are read through
/// typed accessors that panic when absent. Diagnostics go through MLIR —
/// run this inside `yuzu_mlir::diagnostics::capture` to collect them.
pub fn resolve_names<'c>(context: &'c Context, module: &Module<'c>, registry: &dyn Registry) {
    let mut resolver = Resolver {
        context,
        registry,
        structs: HashMap::new(),
        relations: HashMap::new(),
        callables: HashMap::new(),
        traits: HashSet::new(),
        schemas: HashMap::new(),
    };

    resolver.declare(module.body());
    resolver.resolve_block(module.body(), &Ambient::None);
}

impl<'c> Resolver<'c, '_> {
    /// First walk: every declaration registers before anything resolves, so
    /// order between declarations does not matter.
    fn declare(&mut self, block: BlockRef<'c, '_>) {
        for op in block.operations() {
            match YzlOperationRef::of(&op) {
                Some(YzlOperationRef::Struct(item)) => {
                    let name = item.sym_name().value().to_string();
                    let schema = unqualified(item.names().strings());
                    self.declare_named(&op, "struct", &name);
                    self.structs.insert(name, schema);
                }
                Some(YzlOperationRef::Table(table)) => {
                    let name = table.sym_name().value().to_string();
                    let row = table.row().value();
                    if let Some(schema) = self.structs.get(row).cloned() {
                        self.declare_named(&op, "relation", &name);
                        self.relations.insert(name, schema);
                    } else {
                        self.error(&op, format!("unknown struct `{row}`"));
                    }
                }
                Some(YzlOperationRef::Trait(item)) => {
                    let name = item.sym_name().value().to_string();
                    self.declare_named(&op, "trait", &name);
                    self.traits.insert(name);
                }
                Some(YzlOperationRef::Fn(function)) => {
                    let name = function.sym_name().value().to_string();
                    let params = function.params().strings().len();
                    let kind = if function.external() {
                        "external"
                    } else if function.agg() {
                        "agg_fn"
                    } else {
                        "fn"
                    };

                    self.declare_named(&op, "function", &name);
                    self.callables.insert(
                        name,
                        Callable {
                            kind,
                            min_args: params,
                            max_args: params,
                        },
                    );
                }
                _ => {}
            }
        }
    }

    fn declare_named<'a>(&mut self, op: &impl OperationLike<'c, 'a>, what: &str, name: &str)
    where
        'c: 'a,
    {
        if self.structs.contains_key(name)
            || self.relations.contains_key(name)
            || self.callables.contains_key(name)
            || self.traits.contains(name)
        {
            self.error(op, format!("the {what} `{name}` is already defined"));
        }
    }

    fn resolve_block(&mut self, block: BlockRef<'c, '_>, ambient: &Ambient) {
        for mut op in block.operations_mut() {
            self.resolve_op(&mut op, ambient);
        }
    }

    fn resolve_op(&mut self, op: &mut OperationRefMut<'c, '_>, ambient: &Ambient) {
        match YzlOperationRef::of(op) {
            Some(YzlOperationRef::Name(name)) => {
                let reference = name.name().value();
                self.resolve_name(op, reference, ambient);
            }
            Some(YzlOperationRef::Call(call)) => {
                let callee = call.callee().value();
                self.resolve_call(op, callee, ambient);
            }
            Some(YzlOperationRef::Fn(function)) => {
                let params = function.params().strings();
                let generics = function
                    .type_params()
                    .map(|names| names.strings())
                    .unwrap_or_default();
                self.resolve_bounds(op, &function, &generics);
                self.resolve_regions(op, &Ambient::Params(&params));
            }
            Some(YzlOperationRef::Trait(_)) => {
                self.resolve_regions(op, &Ambient::None);
            }
            Some(YzlOperationRef::Impl(item)) => {
                let (trait_name, target) = (item.r#trait().value(), item.target().value());
                if !self.traits.contains(trait_name) {
                    self.error(op, format!("unknown trait `{trait_name}`"));
                }

                if !self.is_type_name(target) {
                    self.error(op, format!("unknown type `{target}`"));
                }

                self.resolve_regions(op, &Ambient::None);
            }
            Some(YzlOperationRef::Let(binding)) => {
                self.resolve_regions(op, &Ambient::None);
                let name = binding.sym_name().value().to_string();
                let schema = self.yielded_schema(op);
                self.declare_named(op, "binding", &name);
                self.relations.insert(name, schema);
            }
            Some(YzlOperationRef::From(from)) => {
                let source = from.source().value();
                let schema = match self.relations.get(source) {
                    Some(schema) => schema.clone(),
                    None => {
                        self.error(op, format!("unknown relation `{source}`"));
                        Schema::new()
                    }
                };

                self.record_schema(op, schema);
            }
            Some(YzlOperationRef::Alias(stage)) => {
                let alias = stage.alias().value();
                let schema = self
                    .input_schema(op)
                    .into_iter()
                    .map(|column| Column {
                        qualifier: Some(alias.to_string()),
                        ..column
                    })
                    .collect();
                self.record_schema(op, schema);
            }
            Some(
                YzlOperationRef::Where(_)
                | YzlOperationRef::Distinct(_)
                | YzlOperationRef::Limit(_),
            ) => {
                let schema = self.input_schema(op);
                self.resolve_regions(op, &Ambient::Columns(&schema));
                self.record_schema(op, schema);
            }
            Some(YzlOperationRef::Select(stage)) => {
                let names = stage.names().strings();
                let input = self.input_schema(op);
                self.resolve_regions(op, &Ambient::Columns(&input));
                self.record_schema(op, unqualified(names));
            }
            Some(YzlOperationRef::Extend(stage)) => {
                let names = stage.names().strings();
                let mut schema = self.input_schema(op);
                self.resolve_regions(op, &Ambient::Columns(&schema));
                schema.extend(unqualified(names));
                self.record_schema(op, schema);
            }
            Some(YzlOperationRef::Set(stage)) => {
                let names = stage.names().strings();
                let schema = self.input_schema(op);
                let mut columns = Vec::new();
                for name in names {
                    match schema.iter().position(|column| column.matches(&name)) {
                        Some(index) => columns.push(index),
                        None => self.error(op, format!("unknown column `{name}`")),
                    }
                }

                op.set_index_array_attribute(self.context, "set_cols", &columns);
                self.resolve_regions(op, &Ambient::Columns(&schema));
                self.record_schema(op, schema);
            }
            Some(YzlOperationRef::Drop(stage)) => {
                let columns = stage.columns().strings();
                let mut schema = self.input_schema(op);
                for name in columns {
                    match schema.iter().position(|column| column.matches(&name)) {
                        Some(index) => {
                            schema.remove(index);
                        }
                        None => self.error(op, format!("unknown column `{name}`")),
                    }
                }

                self.record_schema(op, schema);
            }
            Some(YzlOperationRef::Rename(stage)) => {
                let (from, to) = (stage.from().strings(), stage.to().strings());
                let mut schema = self.input_schema(op);
                for (from, to) in from.iter().zip(to) {
                    match schema.iter().position(|column| column.matches(from)) {
                        Some(index) => schema[index].name = to,
                        None => self.error(op, format!("unknown column `{from}`")),
                    }
                }

                self.record_schema(op, schema);
            }
            Some(YzlOperationRef::Aggregate(stage)) => {
                let group_by = stage.group_by().strings();
                let names = stage.names().strings();
                let input = self.input_schema(op);
                let mut keys = Vec::new();
                let mut schema = Schema::new();
                for name in group_by {
                    match self.find_column(op, &input, &name) {
                        Some(index) => {
                            keys.push(index);
                            schema.push(input[index].clone());
                        }
                        None => schema.push(Column {
                            qualifier: None,
                            name,
                        }),
                    }
                }

                op.set_index_array_attribute(self.context, "key_cols", &keys);
                schema.extend(unqualified(names));
                self.resolve_regions(op, &Ambient::Columns(&input));
                self.record_schema(op, schema);
            }
            Some(YzlOperationRef::Join(stage)) => {
                let relation = stage.rhs().value();
                let alias = stage.rhs_alias().map(|alias| alias.value());
                let using_columns = stage
                    .using_columns()
                    .map(|columns| columns.strings())
                    .unwrap_or_default();
                let mut schema = self.input_schema(op);
                let mut rhs = match self.relations.get(relation) {
                    Some(schema) => schema.clone(),
                    None => {
                        self.error(op, format!("unknown relation `{relation}`"));
                        Schema::new()
                    }
                };

                if let Some(alias) = alias {
                    for column in &mut rhs {
                        column.qualifier = Some(alias.to_string());
                    }
                }

                for name in using_columns {
                    for side in [&schema, &rhs] {
                        if !side.iter().any(|column| column.matches(&name)) {
                            self.error(op, format!("unknown column `{name}`"));
                        }
                    }
                }

                schema.extend(rhs);
                self.resolve_regions(op, &Ambient::Columns(&schema));
                self.record_schema(op, schema);
            }
            Some(
                YzlOperationRef::Output(_)
                | YzlOperationRef::Yield(_)
                | YzlOperationRef::Return(_)
                | YzlOperationRef::List(_)
                | YzlOperationRef::Struct(_)
                | YzlOperationRef::Table(_)
                | YzlOperationRef::Missing(_),
            ) => {}
            None => self.resolve_regions(op, ambient),
        }
    }

    /// A bound names a type parameter of the function it sits on, and a
    /// trait that has been declared.
    fn resolve_bounds(
        &mut self,
        op: &OperationRefMut<'c, '_>,
        function: &yuzu_mlir::ops::yzl::FnOperationRef<'c, '_>,
        generics: &[String],
    ) {
        let subjects = function
            .bound_params()
            .map(|names| names.strings())
            .unwrap_or_default();
        let traits = function
            .bound_traits()
            .map(|names| names.symbols())
            .unwrap_or_default();

        for (subject, bound) in subjects.iter().zip(traits) {
            if !generics.iter().any(|param| param == subject) {
                self.error(op, format!("unknown type parameter `{subject}`"));
            }

            if !self.traits.contains(&bound) {
                self.error(op, format!("unknown trait `{bound}`"));
            }
        }
    }

    /// Whether a name denotes a type an `impl` can target.
    fn is_type_name(&self, name: &str) -> bool {
        matches!(name, "int64" | "float64" | "bool" | "str") || self.structs.contains_key(name)
    }

    fn resolve_name(
        &mut self,
        op: &mut OperationRefMut<'c, '_>,
        reference: &str,
        ambient: &Ambient,
    ) {
        match ambient {
            Ambient::Columns(schema) => {
                if let Some(index) = self.find_column(op, schema, reference) {
                    op.set_index_attribute(self.context, "col", index);
                }
            }
            Ambient::Params(params) => match params.iter().position(|param| param == reference) {
                Some(index) => op.set_index_attribute(self.context, "param", index),
                None => self.error(op, format!("unknown name `{reference}`")),
            },

            Ambient::None => self.error(op, format!("unknown name `{reference}`")),
        }
    }

    fn resolve_call(&mut self, op: &mut OperationRefMut<'c, '_>, callee: &str, ambient: &Ambient) {
        self.resolve_regions(op, ambient);
        let arguments = op.operand_count();
        let (kind, min, max) = match self.callables.get(callee) {
            Some(callable) => (callable.kind, callable.min_args, callable.max_args),
            None => match self
                .registry
                .entries()
                .iter()
                .find(|entry| entry.name == callee)
            {
                Some(entry) => ("builtin", entry.min_args, entry.max_args),
                None => {
                    self.error(op, format!("unknown function `{callee}`"));
                    return;
                }
            },
        };

        if arguments < min || arguments > max {
            let expected = if min == max {
                format!("{min}")
            } else {
                format!("{min} to {max}")
            };

            self.error(
                op,
                format!("`{callee}` expects {expected} arguments, got {arguments}"),
            );
        }

        op.set_attribute(
            "callee_kind",
            StringAttribute::new(self.context, kind).into(),
        );
    }

    fn find_column<'a>(
        &mut self,
        op: &impl OperationLike<'c, 'a>,
        schema: &Schema,
        reference: &str,
    ) -> Option<usize>
    where
        'c: 'a,
    {
        let mut matches = schema
            .iter()
            .enumerate()
            .filter(|(_, column)| column.matches(reference));
        match (matches.next(), matches.next()) {
            (Some((index, _)), None) => Some(index),
            (Some(_), Some(_)) => {
                self.error(op, format!("`{reference}` is ambiguous"));
                None
            }
            (None, _) => {
                self.error(op, format!("unknown column `{reference}`"));
                None
            }
        }
    }

    fn resolve_regions<'a>(&mut self, op: &impl OperationLike<'c, 'a>, ambient: &Ambient)
    where
        'c: 'a,
    {
        for region in op.regions() {
            for block in region.blocks() {
                self.resolve_block(block, ambient);
            }
        }
    }

    /// The schema flowing out of the op a region's yield hands back — the
    /// shape a `let`-bound query exposes to `from`.
    fn yielded_schema<'a>(&mut self, op: &impl OperationLike<'c, 'a>) -> Schema
    where
        'c: 'a,
    {
        op.regions()
            .next()
            .and_then(|region| region.first_block())
            .and_then(|block| block.last_operation())
            .and_then(|last| last.operand(0).ok())
            .and_then(|value| self.schemas.get(&value_id(value)).cloned())
            .unwrap_or_default()
    }

    fn input_schema<'a>(&mut self, op: &impl OperationLike<'c, 'a>) -> Schema
    where
        'c: 'a,
    {
        op.operand(0)
            .ok()
            .and_then(|input| self.schemas.get(&value_id(input)).cloned())
            .unwrap_or_default()
    }

    fn record_schema<'a>(&mut self, op: &impl OperationLike<'c, 'a>, schema: Schema)
    where
        'c: 'a,
    {
        if let Ok(result) = op.result(0) {
            self.schemas.insert(value_id(result.into()), schema);
        }
    }

    fn error<'a>(&mut self, op: &impl OperationLike<'c, 'a>, message: String)
    where
        'c: 'a,
    {
        yuzu_mlir::diagnostics::emit_error(op.location(), &message);
    }
}

fn unqualified(names: Vec<String>) -> Schema {
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
    use expect_test::{Expect, expect};

    use crate::resolve_names;
    use crate::test_support;

    fn check(source: &str, expected: Expect) {
        test_support::check(
            source,
            |context, module| resolve_names(context, module, &yuzu_types::Builtins),
            expected,
        );
    }

    #[test]
    fn resolves_columns_params_and_calls() {
        check(
            r#"
struct Row { a: int64, b: int64 }
table t = Row

fn f(x: int64) -> int64 { return x * 3 }

from t
|> where a > 10
|> extend f(a) + b as e
|> aggregate sum(e) as s group by b
    "#,
            expect![[r#"
            module {
              yzl.struct @Row ["a", "b"] : [!yz.int64, !yz.int64]
              yzl.table @t of @Row
              yzl.fn @f params ["x"] (!yz.int64) -> !yz.int64 {
                %4 = yzl.name "x" : !yzl.var {param = 0 : i64}
                %5 = yz.constant_int 3
                %6 = yz.mul %4, %5 : !yzl.var, !yz.int64 -> !yzl.var
                yzl.return %6 : !yzl.var
              }
              %0 = yzl.from @t
              %1 = yzl.where %0 {
                %4 = yzl.name "a" : !yzl.var {col = 0 : i64}
                %5 = yz.constant_int 10
                %6 = yz.cmp "gt", %4, %5 : !yzl.var, !yz.int64 -> !yzl.var
                yzl.yield %6 : !yzl.var
              }
              %2 = yzl.extend %1 as ["e"] {
                %4 = yzl.name "a" : !yzl.var {col = 0 : i64}
                %5 = yzl.call @f(%4) : (!yzl.var) -> !yzl.var {callee_kind = "fn"}
                %6 = yzl.name "b" : !yzl.var {col = 1 : i64}
                %7 = yz.add %5, %6 : !yzl.var, !yzl.var -> !yzl.var
                yzl.yield %7 : !yzl.var
              }
              %3 = yzl.aggregate %2 group_by ["b"] as ["s"] {
                %4 = yzl.name "e" : !yzl.var {col = 2 : i64}
                %5 = yzl.call @sum(%4) : (!yzl.var) -> !yzl.var {callee_kind = "builtin"}
                yzl.yield %5 : !yzl.var
              } {key_cols = [1]}
              yzl.output %3
            }
            "#]],
        );
    }

    #[test]
    fn resolves_aliases_joins_and_bindings() {
        check(
            r#"
struct Row { id: int64, dept_id: int64 }
table t = Row
struct Dept { id: int64 }
table depts = Dept

let base = from t |> where id > 0

from base
|> inner join depts as d on dept_id == d.id
|> select d.id as out
    "#,
            expect![[r#"
            module {
              yzl.struct @Row ["id", "dept_id"] : [!yz.int64, !yz.int64]
              yzl.table @t of @Row
              yzl.struct @Dept ["id"] : [!yz.int64]
              yzl.table @depts of @Dept
              yzl.let @base {
                %3 = yzl.from @t
                %4 = yzl.where %3 {
                  %5 = yzl.name "id" : !yzl.var {col = 0 : i64}
                  %6 = yz.constant_int 0
                  %7 = yz.cmp "gt", %5, %6 : !yzl.var, !yz.int64 -> !yzl.var
                  yzl.yield %7 : !yzl.var
                }
                yzl.yield %4 : !yzl.query
              }
              %0 = yzl.from @base
              %1 = yzl.join "inner", %0, @depts as "d" {
                %3 = yzl.name "dept_id" : !yzl.var {col = 1 : i64}
                %4 = yzl.name "d.id" : !yzl.var {col = 2 : i64}
                %5 = yz.cmp "eq", %3, %4 : !yzl.var, !yzl.var -> !yzl.var
                yzl.yield %5 : !yzl.var
              }
              %2 = yzl.select %1 as ["out"] {
                %3 = yzl.name "d.id" : !yzl.var {col = 2 : i64}
                yzl.yield %3 : !yzl.var
              }
              yzl.output %2
            }
            "#]],
        );
    }

    #[test]
    fn reports_unknown_names() {
        check(
            r#"
struct Row { a: int64 }
table t = Row

fn f(x: int64) -> int64 { return y }

from missing
|> where nope > 1
|> set ghost = 2
|> extend wrong(a) as e
    "#,
            expect![[r#"
            error: unknown name `y`
             --> test.yz:5:34
              |
            5 | fn f(x: int64) -> int64 { return y }
              |                                  ^

            error: unknown relation `missing`
             --> test.yz:7:1
              |
            7 | from missing
              | ^

            error: unknown column `nope`
             --> test.yz:8:10
              |
            8 | |> where nope > 1
              |          ^

            error: unknown column `ghost`
             --> test.yz:7:1
              |
            7 | from missing
              | ^

            error: unknown column `a`
             --> test.yz:10:17
               |
            10 | |> extend wrong(a) as e
               |                 ^

            error: unknown function `wrong`
             --> test.yz:10:11
               |
            10 | |> extend wrong(a) as e
               |           ^
            "#]],
        );
    }

    #[test]
    fn reports_arity_and_duplicates() {
        check(
            r#"
struct Row { a: int64 }
table t = Row
table t = Row

fn f(x: int64) -> int64 { return x }

from t
|> extend f(a, a) as two, sum() as none
    "#,
            expect![[r#"
            error: the relation `t` is already defined
             --> test.yz:4:1
              |
            4 | table t = Row
              | ^

            error: `f` expects 1 arguments, got 2
             --> test.yz:9:11
              |
            9 | |> extend f(a, a) as two, sum() as none
              |           ^

            error: `sum` expects 1 arguments, got 0
             --> test.yz:9:27
              |
            9 | |> extend f(a, a) as two, sum() as none
              |                           ^
            "#]],
        );
    }

    #[test]
    fn reports_unknown_traits_and_bounds() {
        check(
            r#"
trait Add {
    fn add(x: Self, y: Self) -> Self
}

impl Ord for int64 {
    fn cmp(x: int64, y: int64) -> bool { return x > y }
}

impl Add for Ghost {
    fn add(x: int64, y: int64) -> int64 { return x }
}

fn id[T](x: T) -> T where U: Add { return x }

fn other[T](x: T) -> T where T: Missing { return x }
"#,
            expect![[r#"
                error: unknown trait `Ord`
                 --> test.yz:6:1
                  |
                6 | impl Ord for int64 {
                  | ^

                error: unknown type `Ghost`
                 --> test.yz:10:1
                   |
                10 | impl Add for Ghost {
                   | ^

                error: unknown type parameter `U`
                 --> test.yz:14:1
                   |
                14 | fn id[T](x: T) -> T where U: Add { return x }
                   | ^

                error: unknown trait `Missing`
                 --> test.yz:16:1
                   |
                16 | fn other[T](x: T) -> T where T: Missing { return x }
                   | ^
            "#]],
        );
    }

    #[test]
    fn reports_an_ambiguous_join_column() {
        check(
            r#"
struct Row { id: int64 }
table l = Row
table r = Row

from l
|> inner join r on id == 1
    "#,
            expect![[r#"
            error: `id` is ambiguous
             --> test.yz:7:20
              |
            7 | |> inner join r on id == 1
              |                    ^
            "#]],
        );
    }
}
