use melior::ir::{
    Attribute, Block, BlockLike, BlockRef, Location, Region, RegionLike, Type, Value,
    attribute::{ArrayAttribute, FlatSymbolRefAttribute, IntegerAttribute, StringAttribute},
};
use yuzu_ast::{AstNode, ast};
use yuzu_mlir::ods::yzl;

use crate::lower_ast_to_yzl::{AstToYzl, Locals, ident_text};
use melior::ir::r#type::IntegerType;
use yuzu_mlir::attributes::JoinKind;
use yuzu_mlir::ext::{OperationExt, OperationMutExt};

use crate::lower_ast_to_yzl::resolve::{JoinError, Lookup};
use yuzu_mlir::types;

impl<'c, 'd> AstToYzl<'c, 'd> {
    pub(super) fn convert_rel<'a>(&self, block: BlockRef<'c, 'a>, rel: &ast::Rel) -> Value<'c, 'a> {
        let loc = self.location(rel);
        match rel {
            ast::Rel::FromExpr(from) => {
                let Some(source) = ident_text(from.relation()) else {
                    return self.missing(
                        block,
                        from,
                        "`from` is missing its relation",
                        types::query(self.context),
                    );
                };

                let value = self.from(block, from, &source, loc);
                match ident_text(from.alias()) {
                    Some(alias) => self.alias(block, value, &alias, loc),
                    None => value,
                }
            }
            ast::Rel::WhereExpr(stage) => {
                let input = self.convert_input(block, stage, "`where`", stage.input());
                let region = Region::new();
                let body = self.stage_block(&region, loc);
                let predicate = match stage.predicate() {
                    Some(expr) => self.convert_expr(body, &Locals::new(), &expr),
                    None => self.missing(
                        body,
                        stage,
                        "`where` is missing its predicate",
                        types::var(self.context),
                    ),
                };

                body.append_operation(yzl::r#yield(self.context, &[predicate], loc).into());
                block
                    .append_operation(
                        yzl::r#where(self.context, types::query(self.context), input, region, loc)
                            .into(),
                    )
                    .first_result()
            }
            ast::Rel::SelectExpr(stage) => {
                let input = self.convert_input(block, stage, "`select`", stage.input());
                let items = stage
                    .items()
                    .map(|item| (item.alias(), item.expr(), item.syntax().text_range()))
                    .collect();
                let (names, region) = self.convert_items(items, "select item", loc);
                self.resolver.borrow_mut().select(strings(&names));
                block
                    .append_operation(
                        yzl::select(
                            self.context,
                            types::query(self.context),
                            input,
                            region,
                            ArrayAttribute::new(self.context, &names),
                            loc,
                        )
                        .into(),
                    )
                    .first_result()
            }
            ast::Rel::ExtendExpr(stage) => {
                let input = self.convert_input(block, stage, "`extend`", stage.input());
                let items = stage
                    .items()
                    .map(|item| (item.alias(), item.expr(), item.syntax().text_range()))
                    .collect();
                let (names, region) = self.convert_items(items, "extend item", loc);
                self.resolver.borrow_mut().extend(strings(&names));
                block
                    .append_operation(
                        yzl::extend(
                            self.context,
                            types::query(self.context),
                            input,
                            region,
                            ArrayAttribute::new(self.context, &names),
                            loc,
                        )
                        .into(),
                    )
                    .first_result()
            }
            ast::Rel::AggregateExpr(stage) => {
                let input = self.convert_input(block, stage, "`aggregate`", stage.input());
                let mut group_by = Vec::new();
                for item in stage.group_by().into_iter().flat_map(|group| group.items()) {
                    match ident_text(item.column()) {
                        Some(name) => group_by.push(name),
                        None => self.error(&item, "group by item is missing its column"),
                    }
                }

                let items = stage
                    .items()
                    .map(|item| (item.alias(), item.expr(), item.syntax().text_range()))
                    .collect();
                let (names, region) = self.convert_items(items, "aggregate item", loc);
                // The keys are the grouping's answer, decided against the row
                // the measures were just resolved against.
                let keys = match self
                    .resolver
                    .borrow_mut()
                    .aggregate(&group_by, strings(&names))
                {
                    Ok(keys) => keys,
                    Err(Lookup::Ambiguous) => {
                        self.error(stage, "a group by column is ambiguous");
                        Vec::new()
                    }
                    Err(_) => {
                        self.error(stage, "unknown group by column");
                        Vec::new()
                    }
                };
                let group_attrs: Vec<Attribute> = group_by
                    .iter()
                    .map(|name| StringAttribute::new(self.context, name).into())
                    .collect();
                let mut op: melior::ir::Operation<'c> = yzl::aggregate(
                    self.context,
                    types::query(self.context),
                    input,
                    region,
                    ArrayAttribute::new(self.context, &group_attrs),
                    ArrayAttribute::new(self.context, &names),
                    loc,
                )
                .into();
                op.set_index_array_attribute(self.context, "key_cols", &keys);
                block.append_operation(op).first_result()
            }
            ast::Rel::LimitExpr(stage) => {
                let input = self.convert_input(block, stage, "`limit`", stage.input());
                let count = match stage.count() {
                    Some(ast::Expr::Literal(ast::Literal::IntLiteral(int))) => {
                        int.value().unwrap_or_default() as i64
                    }
                    Some(other) => {
                        self.error(&other, "`limit` takes an integer literal");
                        0
                    }
                    None => {
                        self.error(stage, "`limit` is missing its count");
                        0
                    }
                };

                block
                    .append_operation(
                        yzl::limit(
                            self.context,
                            types::query(self.context),
                            input,
                            IntegerAttribute::new(IntegerType::new(self.context, 64).into(), count),
                            loc,
                        )
                        .into(),
                    )
                    .first_result()
            }
            ast::Rel::RenameExpr(stage) => {
                let input = self.convert_input(block, stage, "`rename`", stage.input());
                let mut from = Vec::new();
                let mut to = Vec::new();
                for item in stage.items() {
                    let (Some(old), Some(new)) = (ident_text(item.from()), ident_text(item.to()))
                    else {
                        self.error(&item, "rename item is missing a column name");
                        continue;
                    };

                    // `b.id as other` keeps its qualifier: the resolver matches
                    // qualified columns, so dropping it would rename whichever
                    // column happened to come first.
                    let old = match ident_text(item.qualifier()) {
                        Some(qualifier) => format!("{qualifier}.{old}"),
                        None => old,
                    };
                    from.push(old);
                    to.push(new);
                }

                let indices = match self.resolver.borrow_mut().rename(&from, &to) {
                    Ok(indices) => indices,
                    Err(name) => {
                        self.error(stage, &format!("unknown column `{name}`"));
                        Vec::new()
                    }
                };
                let attrs = |names: &[String]| -> Vec<Attribute<'c>> {
                    names
                        .iter()
                        .map(|name| StringAttribute::new(self.context, name).into())
                        .collect()
                };
                let mut op: melior::ir::Operation<'c> = yzl::rename(
                    self.context,
                    types::query(self.context),
                    input,
                    ArrayAttribute::new(self.context, &attrs(&from)),
                    ArrayAttribute::new(self.context, &attrs(&to)),
                    loc,
                )
                .into();
                op.set_index_array_attribute(self.context, "rename_cols", &indices);
                block.append_operation(op).first_result()
            }
            ast::Rel::AliasExpr(stage) => {
                let input = self.convert_input(block, stage, "`alias`", stage.input());
                let Some(alias) = ident_text(stage.alias()) else {
                    self.error(stage, "`alias` is missing its name");
                    return input;
                };

                self.alias(block, input, &alias, loc)
            }
            ast::Rel::JoinExpr(stage) => {
                let lhs = self.convert_input(block, stage, "`join`", stage.input());
                let kind = match stage.kind() {
                    Some(ast::JoinKind::Left) => JoinKind::Left,
                    Some(ast::JoinKind::Right) => JoinKind::Right,
                    Some(ast::JoinKind::Full) => JoinKind::Full,
                    _ => JoinKind::Inner,
                };
                let Some(rhs) = ident_text(stage.relation()) else {
                    return self.missing(
                        block,
                        stage,
                        "`join` is missing its relation",
                        types::query(self.context),
                    );
                };

                let alias = ident_text(stage.alias());
                let using: Vec<String> = stage
                    .using()
                    .into_iter()
                    .flat_map(|using| using.columns())
                    .filter_map(|column| column.text())
                    .collect();
                // Both sides carry through, and the `on` region sees exactly
                // that concatenation — so the row moves before the condition
                // is resolved against it.
                match self
                    .resolver
                    .borrow_mut()
                    .join(&rhs, alias.as_deref(), &using)
                {
                    Ok(()) => {}
                    Err(JoinError::UnknownRelation) => {
                        self.error(stage, &format!("unknown relation `{rhs}`"));
                    }
                    Err(JoinError::UsingColumn(name)) => {
                        self.error(stage, &format!("unknown column `{name}`"));
                    }
                }

                let on = Region::new();
                if let Some(condition) = stage.on().and_then(|on| on.condition()) {
                    let body = self.stage_block(&on, loc);
                    let value = self.convert_expr(body, &Locals::new(), &condition);
                    body.append_operation(yzl::r#yield(self.context, &[value], loc).into());
                }

                let mut builder = yzl::JoinOperationBuilder::new(self.context, loc)
                    .result(types::query(self.context))
                    .lhs(lhs)
                    .kind(StringAttribute::new(self.context, kind.as_str()))
                    .rhs(FlatSymbolRefAttribute::new(self.context, &rhs))
                    .on(on);
                if let Some(alias) = alias {
                    builder = builder.rhs_alias(StringAttribute::new(self.context, &alias));
                }
                if !using.is_empty() {
                    let columns: Vec<Attribute> = using
                        .iter()
                        .map(|name| StringAttribute::new(self.context, name).into())
                        .collect();
                    builder = builder.using_columns(ArrayAttribute::new(self.context, &columns));
                }

                block
                    .append_operation(builder.build().into())
                    .first_result()
            }
            ast::Rel::SetExpr(stage) => {
                let input = self.convert_input(block, stage, "`set`", stage.input());
                let items = stage
                    .items()
                    .map(|item| (item.column(), item.value(), item.syntax().text_range()))
                    .collect();
                let (names, region) = self.convert_items(items, "set item", loc);
                let columns = match self.resolver.borrow_mut().set(&strings(&names)) {
                    Ok(columns) => columns,
                    Err(name) => {
                        self.error(stage, &format!("unknown column `{name}`"));
                        Vec::new()
                    }
                };
                let mut op: melior::ir::Operation<'c> = yzl::set(
                    self.context,
                    types::query(self.context),
                    input,
                    region,
                    ArrayAttribute::new(self.context, &names),
                    loc,
                )
                .into();
                op.set_index_array_attribute(self.context, "set_cols", &columns);
                block.append_operation(op).first_result()
            }
            ast::Rel::DistinctExpr(stage) => {
                let input = self.convert_input(block, stage, "`distinct`", stage.input());
                block
                    .append_operation(
                        yzl::distinct(self.context, types::query(self.context), input, loc).into(),
                    )
                    .first_result()
            }
            ast::Rel::DropExpr(stage) => {
                let input = self.convert_input(block, stage, "`drop`", stage.input());
                let names: Vec<String> =
                    stage.columns().filter_map(|column| column.text()).collect();
                if let Err(name) = self.resolver.borrow_mut().drop(&names) {
                    self.error(stage, &format!("unknown column `{name}`"));
                }

                let columns: Vec<Attribute> = names
                    .iter()
                    .map(|name| StringAttribute::new(self.context, name).into())
                    .collect();
                block
                    .append_operation(
                        yzl::drop(
                            self.context,
                            types::query(self.context),
                            input,
                            ArrayAttribute::new(self.context, &columns),
                            loc,
                        )
                        .into(),
                    )
                    .first_result()
            }
        }
    }

    /// A query begins at its relation: the resolver takes the relation's row
    /// as the one every stage after resolves against. An unknown relation is
    /// reported here and the query carries on against an empty row, so the
    /// rest of it is still checked.
    fn from<'a>(
        &self,
        block: BlockRef<'c, 'a>,
        node: &impl AstNode,
        source: &str,
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        if self.resolver.borrow_mut().enter_query(source).is_none() {
            self.error(node, &format!("unknown relation `{source}`"));
            self.resolver.borrow_mut().enter_unknown_query();
        }

        block
            .append_operation(
                yzl::from(
                    self.context,
                    types::query(self.context),
                    FlatSymbolRefAttribute::new(self.context, source),
                    loc,
                )
                .into(),
            )
            .first_result()
    }

    /// `as` qualifies the row's columns: a pure name effect on the row the
    /// resolver carries, and an op so the lowering can see it happened.
    fn alias<'a>(
        &self,
        block: BlockRef<'c, 'a>,
        input: Value<'c, 'a>,
        alias: &str,
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        self.resolver.borrow_mut().alias(alias);
        block
            .append_operation(
                yzl::alias(
                    self.context,
                    types::query(self.context),
                    input,
                    StringAttribute::new(self.context, alias),
                    loc,
                )
                .into(),
            )
            .first_result()
    }

    /// A stage's region takes the row's columns as block arguments, typed
    /// by inference later. Building the block from the resolver's row is
    /// what ties a column's position there to its position here.
    fn stage_block<'r>(&self, region: &'r Region<'c>, loc: Location<'c>) -> BlockRef<'c, 'r> {
        let width = self.resolver.borrow().row().len();
        let arguments: Vec<(Type<'c>, Location<'c>)> = (0..width)
            .map(|_| (types::var(self.context), loc))
            .collect();
        region.append_block(Block::new(&arguments))
    }

    /// A stage's input is another stage, or a bare name referring to a bound
    /// relation — which `from` covers until resolution decides what it was.
    fn convert_input<'a>(
        &self,
        block: BlockRef<'c, 'a>,
        stage: &impl AstNode,
        what: &str,
        input: Option<ast::Expr>,
    ) -> Value<'c, 'a> {
        match input {
            Some(ast::Expr::Rel(rel)) => self.convert_rel(block, &rel),
            Some(ast::Expr::IdentExpr(ident)) => {
                let Some(name) = ident_text(ident.name()) else {
                    return self.missing(
                        block,
                        &ident,
                        &format!("{what} is missing its input relation"),
                        types::query(self.context),
                    );
                };

                let loc = self.location(&ident);
                self.from(block, &ident, &name, loc)
            }
            Some(other) => self.missing(
                block,
                &other,
                "this stage input is not supported yet",
                types::query(self.context),
            ),
            None => self.missing(
                block,
                stage,
                &format!("{what} is missing its input relation"),
                types::query(self.context),
            ),
        }
    }

    fn convert_items(
        &self,
        items: Vec<(Option<ast::Ident>, Option<ast::Expr>, text_size::TextRange)>,
        what: &str,
        loc: Location<'c>,
    ) -> (Vec<Attribute<'c>>, Region<'c>) {
        let region = Region::new();
        let body = self.stage_block(&region, loc);
        let mut names = Vec::with_capacity(items.len());
        let mut values = Vec::with_capacity(items.len());
        for (index, (alias, expr, range)) in items.into_iter().enumerate() {
            let name = ident_text(alias)
                .or_else(|| match &expr {
                    Some(ast::Expr::IdentExpr(ident)) => ident_text(ident.name()),
                    _ => None,
                })
                .unwrap_or_else(|| format!("column{index}"));
            names.push(StringAttribute::new(self.context, &name).into());
            let value = match &expr {
                Some(expr) => self.convert_expr(body, &Locals::new(), expr),
                None => {
                    self.error_at(range, &format!("{what} is missing its expression"));
                    self.hole(body, range, types::var(self.context))
                }
            };

            values.push(value);
        }

        body.append_operation(yzl::r#yield(self.context, &values, loc).into());
        (names, region)
    }
}

/// The names an attribute list carries, for the resolver.
fn strings(names: &[Attribute<'_>]) -> Vec<String> {
    names
        .iter()
        .map(|name| {
            melior::ir::attribute::StringAttribute::try_from(*name)
                .expect("a name attribute is a string")
                .value()
                .to_string()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::lower_ast_to_yzl::test_support::converted;

    #[test]
    fn converts_the_canonical_pipeline() {
        expect![[r#"
            module {
              yzl.struct @Row ["a", "b"] : [!yz.int64, !yz.int64]
              yzl.table @t of @Row
              yzl.fn @f params ["x"] (!yz.int64) -> !yz.int64 {
              ^bb0(%arg0: !yzl.var):
                yzl.return %arg0 : !yzl.var
              }
              %0 = yzl.from @t
              %1 = yzl.where %0 {
              ^bb0(%arg0: !yzl.var, %arg1: !yzl.var):
                %5 = yz.constant_int 10
                %6 = yz.cmp "gt", %arg0, %5 : !yzl.var, !yz.int64 -> !yzl.var
                yzl.yield %6 : !yzl.var
              }
              %2 = yzl.extend %1 as ["e"] {
              ^bb0(%arg0: !yzl.var, %arg1: !yzl.var):
                %5 = yzl.call @f(%arg0) : (!yzl.var) -> !yzl.var {callee_kind = "fn"}
                %6 = yz.add %5, %arg1 : !yzl.var, !yzl.var -> !yzl.var
                yzl.yield %6 : !yzl.var
              }
              %3 = yzl.aggregate %2 group_by ["b"] as ["s"] {
              ^bb0(%arg0: !yzl.var, %arg1: !yzl.var, %arg2: !yzl.var):
                %5 = yzl.call @sum(%arg2) : (!yzl.var) -> !yzl.var {callee_kind = "builtin"}
                yzl.yield %5 : !yzl.var
              } {key_cols = [1]}
              %4 = yzl.limit %3, 10
              yzl.output %4
            }
        "#]]
        .assert_eq(&converted(
            r#"
struct Row { a: int64, b: int64 }
table t = Row
fn f(x: int64) -> int64 { return x }

from t
|> where a > 10
|> extend f(a) + b as e
|> aggregate sum(e) as s group by b
|> limit 10
"#,
        ));
    }

    #[test]
    fn converts_joins_sets_and_membership() {
        expect![[r#"
            module {
              yzl.struct @Employee ["id", "dept_id", "level", "rating"] : [!yz.str, !yz.int64, !yz.int64, !yz.float64]
              yzl.table @employees of @Employee
              yzl.struct @Department ["id", "dept_id"] : [!yz.str, !yz.int64]
              yzl.table @departments of @Department
              %0 = yzl.from @employees
              %1 = yzl.alias %0 as "e"
              %2 = yzl.join "inner", %1, @departments as "d" {
              ^bb0(%arg0: !yzl.var, %arg1: !yzl.var, %arg2: !yzl.var, %arg3: !yzl.var, %arg4: !yzl.var, %arg5: !yzl.var):
                %7 = yz.cmp "eq", %arg1, %arg4 : !yzl.var, !yzl.var -> !yzl.var
                yzl.yield %7 : !yzl.var
              }
              %3 = yzl.set %2 as ["level"] {
              ^bb0(%arg0: !yzl.var, %arg1: !yzl.var, %arg2: !yzl.var, %arg3: !yzl.var, %arg4: !yzl.var, %arg5: !yzl.var):
                %7 = yz.constant_int 1
                %8 = yz.add %arg2, %7 : !yzl.var, !yz.int64 -> !yzl.var
                yzl.yield %8 : !yzl.var
              } {set_cols = [2]}
              %4 = yzl.where %3 {
              ^bb0(%arg0: !yzl.var, %arg1: !yzl.var, %arg2: !yzl.var, %arg3: !yzl.var, %arg4: !yzl.var, %arg5: !yzl.var):
                %7 = yz.constant_int 1
                %8 = yz.constant_int 3
                %9 = yzl.list[%7, %8] : (!yz.int64, !yz.int64) -> !yzl.var
                %10 = yzl.call @in(%arg2, %9) : (!yzl.var, !yzl.var) -> !yzl.var {callee_kind = "builtin"}
                yzl.yield %10 : !yzl.var
              }
              %5 = yzl.drop %4 ["rating"]
              %6 = yzl.distinct %5
              yzl.output %6
            }
        "#]]
        .assert_eq(&converted(
            r#"
struct Employee { id: str, dept_id: int64, level: int64, rating: float64 }
table employees = Employee
struct Department { id: str, dept_id: int64 }
table departments = Department

from employees as e
|> inner join departments as d on e.dept_id == d.id
|> set level = level + 1
|> where level in [1, 3]
|> drop rating
|> distinct
"#,
        ));
    }

    #[test]
    fn converts_sugar_and_bindings() {
        expect![[r#"
            module {
              yzl.struct @Row ["a", "active"] : [!yz.int64, !yz.bool]
              yzl.table @t of @Row
              yzl.let @base {
                %4 = yzl.from @t
                %5 = yzl.where %4 {
                ^bb0(%arg0: !yzl.var, %arg1: !yzl.var):
                  yzl.yield %arg1 : !yzl.var
                }
                yzl.yield %5 : !yzl.query
              }
              %0 = yzl.from @base
              %1 = yzl.rename %0 from ["a"] to ["renamed"] {rename_cols = [0]}
              %2 = yzl.alias %1 as "q"
              %3 = yzl.select %2 as ["out"] {
              ^bb0(%arg0: !yzl.var, %arg1: !yzl.var):
                yzl.yield %arg0 : !yzl.var
              }
              yzl.output %3
            }
        "#]]
        .assert_eq(&converted(
            r#"
struct Row { a: int64, active: bool }
table t = Row

let base = from t |> where active

from base
|> rename a as renamed
|> as q
|> select q.renamed as out
"#,
        ));
    }
}
