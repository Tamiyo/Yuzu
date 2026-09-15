use melior::ir::{
    Attribute, Block, BlockLike, BlockRef, Location, Region, RegionLike, Type, Value,
    attribute::{ArrayAttribute, FlatSymbolRefAttribute, IntegerAttribute, StringAttribute},
};
use yuzu_ast::{AstNode, ast};
use yuzu_mlir::ods::yzl;

use crate::lower_ast_to_yzl::symbols::{ColumnLookup, Reference, Row};
use crate::lower_ast_to_yzl::{AstToYzl, Locals};
use melior::ir::r#type::IntegerType;
use yuzu_mlir::attributes::JoinKind;
use yuzu_mlir::ext::{OperationExt, OperationMutExt};
use yuzu_mlir::types;

impl<'c, 'd> AstToYzl<'c, 'd> {
    pub(super) fn convert_rel<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        rel: &ast::Rel,
    ) -> Value<'c, 'a> {
        let loc = self.location(rel);
        match rel {
            ast::Rel::FromExpr(from) => self.convert_from(block, from, loc),
            ast::Rel::WhereExpr(stage) => self.convert_where(block, stage, loc),
            ast::Rel::SelectExpr(stage) => self.convert_select(block, stage, loc),
            ast::Rel::ExtendExpr(stage) => self.convert_extend(block, stage, loc),
            ast::Rel::AggregateExpr(stage) => self.convert_aggregate(block, stage, loc),
            ast::Rel::LimitExpr(stage) => self.convert_limit(block, stage, loc),
            ast::Rel::RenameExpr(stage) => self.convert_rename(block, stage, loc),
            ast::Rel::AliasExpr(stage) => self.convert_alias(block, stage, loc),
            ast::Rel::JoinExpr(stage) => self.convert_join(block, stage, loc),
            ast::Rel::SetExpr(stage) => self.convert_set(block, stage, loc),
            ast::Rel::DistinctExpr(stage) => self.convert_distinct(block, stage, loc),
            ast::Rel::DropExpr(stage) => self.convert_drop(block, stage, loc),
        }
    }

    fn convert_from<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        from: &ast::FromExpr,
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        let Some(source) = self.ident(from.relation()) else {
            return self.missing(
                block,
                from,
                "`from` is missing its relation",
                types::query(self.context),
            );
        };

        let value = self.scan(block, from, source, loc);
        match self.ident(from.alias()) {
            Some(alias) => self.qualify(block, value, alias, loc),
            None => value,
        }
    }

    fn convert_where<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        stage: &ast::WhereExpr,
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
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
                yzl::r#where(self.context, types::query(self.context), input, region, loc).into(),
            )
            .first_result()
    }

    fn convert_select<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        stage: &ast::SelectExpr,
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        let input = self.convert_input(block, stage, "`select`", stage.input());
        let items = stage
            .items()
            .map(|item| (item.alias(), item.expr(), item.syntax().text_range()))
            .collect();
        let (names, region) = self.convert_items(items, "select item", loc);
        let columns = self.string_attrs(&names);
        self.symbols.replace(names);
        block
            .append_operation(
                yzl::select(
                    self.context,
                    types::query(self.context),
                    input,
                    region,
                    ArrayAttribute::new(self.context, &columns),
                    loc,
                )
                .into(),
            )
            .first_result()
    }

    fn convert_extend<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        stage: &ast::ExtendExpr,
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        let input = self.convert_input(block, stage, "`extend`", stage.input());
        let items = stage
            .items()
            .map(|item| (item.alias(), item.expr(), item.syntax().text_range()))
            .collect();
        let (names, region) = self.convert_items(items, "extend item", loc);
        let columns = self.string_attrs(&names);
        self.symbols.extend(names);
        block
            .append_operation(
                yzl::extend(
                    self.context,
                    types::query(self.context),
                    input,
                    region,
                    ArrayAttribute::new(self.context, &columns),
                    loc,
                )
                .into(),
            )
            .first_result()
    }

    fn convert_aggregate<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        stage: &ast::AggregateExpr,
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        let input = self.convert_input(block, stage, "`aggregate`", stage.input());
        let mut keys = Vec::new();
        let mut key_names: Vec<&'c str> = Vec::new();
        for item in stage.group_by().into_iter().flat_map(|group| group.items()) {
            let Some(column) = self.ident(item.column()) else {
                self.error(&item, "group by key is missing its column");
                continue;
            };

            let reference = Reference {
                qualifier: self.ident(item.qualifier()),
                name: column,
            };
            if let Some(index) = self.column(&item, "group key", reference) {
                keys.push(index);
                key_names.push(self.ident(item.alias()).unwrap_or(column));
            }
        }

        let items = stage
            .items()
            .map(|item| (item.alias(), item.expr(), item.syntax().text_range()))
            .collect();
        let (names, region) = self.convert_items(items, "aggregate item", loc);
        let group_by = self.string_attrs(&key_names);
        let measures = self.string_attrs(&names);
        // A grouping's row is its keys, in order, then its measures.
        key_names.extend(names);
        self.symbols.replace(key_names);

        let mut op: melior::ir::Operation<'c> = yzl::aggregate(
            self.context,
            types::query(self.context),
            input,
            region,
            ArrayAttribute::new(self.context, &group_by),
            ArrayAttribute::new(self.context, &measures),
            loc,
        )
        .into();
        op.set_index_array_attribute(self.context, "key_cols", &keys);
        block.append_operation(op).first_result()
    }

    fn convert_limit<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        stage: &ast::LimitExpr,
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        let input = self.convert_input(block, stage, "`limit`", stage.input());
        let count = match stage.count() {
            Some(count) => self.int_literal(&count),
            None => {
                self.error(stage, "`limit` is missing its row count");
                0
            }
        };
        let offset = stage.offset().map(|offset| self.int_literal(&offset));

        let i64 = IntegerType::new(self.context, 64).into();
        let mut builder = yzl::LimitOperationBuilder::new(self.context, loc)
            .result(types::query(self.context))
            .input(input)
            .count(IntegerAttribute::new(i64, count));
        if let Some(offset) = offset {
            builder = builder.offset(IntegerAttribute::new(i64, offset));
        }

        block
            .append_operation(builder.build().into())
            .first_result()
    }

    fn convert_rename<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        stage: &ast::RenameExpr,
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        let input = self.convert_input(block, stage, "`rename`", stage.input());
        let mut from: Vec<&'c str> = Vec::new();
        let mut to: Vec<&'c str> = Vec::new();
        let mut renames = Vec::new();
        for item in stage.items() {
            let (Some(old), Some(new)) = (self.ident(item.from()), self.ident(item.to())) else {
                self.error(&item, "rename item is missing a column name");
                continue;
            };

            let reference = Reference {
                qualifier: self.ident(item.qualifier()),
                name: old,
            };
            if let Some(index) = self.column(&item, "column", reference) {
                renames.push((index, new));
                from.push(self.intern(&reference.to_string()));
                to.push(new);
            }
        }

        self.symbols.rename(&renames);
        let indices: Vec<usize> = renames.iter().map(|&(index, _)| index).collect();
        let mut op: melior::ir::Operation<'c> = yzl::rename(
            self.context,
            types::query(self.context),
            input,
            ArrayAttribute::new(self.context, &self.string_attrs(&from)),
            ArrayAttribute::new(self.context, &self.string_attrs(&to)),
            loc,
        )
        .into();
        op.set_index_array_attribute(self.context, "rename_cols", &indices);
        block.append_operation(op).first_result()
    }

    fn convert_alias<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        stage: &ast::AliasExpr,
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        let input = self.convert_input(block, stage, "`as`", stage.input());
        let Some(alias) = self.ident(stage.alias()) else {
            self.error(stage, "`as` is missing its alias");
            return input;
        };

        self.qualify(block, input, alias, loc)
    }

    fn convert_join<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        stage: &ast::JoinExpr,
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        let lhs = self.convert_input(block, stage, "`join`", stage.input());
        let kind = match stage.kind() {
            Some(ast::JoinKind::Left) => JoinKind::Left,
            Some(ast::JoinKind::Right) => JoinKind::Right,
            Some(ast::JoinKind::Full) => JoinKind::Full,
            Some(ast::JoinKind::Inner) | None => JoinKind::Inner,
        };
        let Some(relation) = self.ident(stage.relation()) else {
            return self.missing(
                block,
                stage,
                "`join` is missing its relation",
                types::query(self.context),
            );
        };

        let alias = self.ident(stage.alias());
        let rhs = match self.symbols.relation(relation, alias) {
            Some(rhs) => rhs,
            None => {
                self.error(stage, &format!("`{relation}` is not a relation"));
                Row::new()
            }
        };

        let mut using: Vec<&'c str> = Vec::new();
        if let Some(clause) = stage.using() {
            using = clause
                .columns()
                .filter_map(|column| self.ident(Some(column)))
                .collect();
            if using.is_empty() {
                self.error(&clause, "`using` needs at least one column");
            }

            for &column in &using {
                let column = Reference::bare(column);
                if !self.symbols.row().has(column) || !rhs.has(column) {
                    self.error(
                        &clause,
                        &format!("column {column} not present in both relations"),
                    );
                }
            }
        }

        // The condition sees both rows, so the row moves first.
        self.symbols.concat(rhs);
        let on = Region::new();
        if stage.using().is_none() {
            match stage.on() {
                None => self.error(stage, "`join` is missing its `on` or `using` clause"),
                Some(clause) => match clause.condition() {
                    None => self.error(&clause, "`on` is missing its condition"),
                    Some(condition) => {
                        let body = self.stage_block(&on, loc);
                        let value = self.convert_expr(body, &Locals::new(), &condition);
                        body.append_operation(yzl::r#yield(self.context, &[value], loc).into());
                    }
                },
            }
        }

        let mut builder = yzl::JoinOperationBuilder::new(self.context, loc)
            .result(types::query(self.context))
            .lhs(lhs)
            .kind(StringAttribute::new(self.context, kind.as_str()))
            .rhs(FlatSymbolRefAttribute::new(self.context, relation))
            .on(on);
        if let Some(alias) = alias {
            builder = builder.rhs_alias(StringAttribute::new(self.context, alias));
        }
        if !using.is_empty() {
            let columns = self.string_attrs(&using);
            builder = builder.using_columns(ArrayAttribute::new(self.context, &columns));
        }

        block
            .append_operation(builder.build().into())
            .first_result()
    }

    fn convert_set<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        stage: &ast::SetExpr,
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        let input = self.convert_input(block, stage, "`set`", stage.input());
        let mut columns = Vec::new();
        for item in stage.items() {
            let Some(name) = self.ident(item.column()) else {
                self.error(&item, "set item is incomplete");
                continue;
            };

            if let Some(index) = self.column(&item, "column", Reference::bare(name)) {
                columns.push(index);
            }
        }

        let items = stage
            .items()
            .map(|item| (item.column(), item.value(), item.syntax().text_range()))
            .collect();
        let (names, region) = self.convert_items(items, "set item", loc);
        let names = self.string_attrs(&names);
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

    fn convert_distinct<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        stage: &ast::DistinctExpr,
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        let input = self.convert_input(block, stage, "`distinct`", stage.input());
        block
            .append_operation(
                yzl::distinct(self.context, types::query(self.context), input, loc).into(),
            )
            .first_result()
    }

    fn convert_drop<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        stage: &ast::DropExpr,
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        let input = self.convert_input(block, stage, "`drop`", stage.input());
        let mut names: Vec<&'c str> = Vec::new();
        for column in stage.columns() {
            let Some(name) = column.text().map(|text| self.intern(&text)) else {
                continue;
            };

            if let Some(index) = self.column(&column, "column", Reference::bare(name)) {
                self.symbols.remove(index);
            }

            names.push(name);
        }

        let columns = self.string_attrs(&names);
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

    fn int_literal(&mut self, expr: &ast::Expr) -> i64 {
        match expr {
            ast::Expr::Literal(ast::Literal::IntLiteral(int)) => {
                int.value().unwrap_or_default() as i64
            }
            other => {
                self.error(other, "`limit` takes an integer literal");
                0
            }
        }
    }

    fn scan<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        node: &impl AstNode,
        source: &str,
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        // A query nobody can name a row for is still converted, against an
        // empty row, so the stages after it are checked too.
        let row = match self.symbols.relation(source, None) {
            Some(row) => row,
            None => {
                self.error(node, &format!("`{source}` is not a relation"));
                Row::new()
            }
        };

        self.symbols.enter_relation(row);
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

    fn qualify<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        input: Value<'c, 'a>,
        alias: &'c str,
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        self.symbols.alias(alias);
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

    /// A stage's region takes the row's columns as block arguments, typed by
    /// inference later.
    fn stage_block<'r>(&self, region: &'r Region<'c>, loc: Location<'c>) -> BlockRef<'c, 'r> {
        let width = self.symbols.row().len();
        let arguments: Vec<(Type<'c>, Location<'c>)> = (0..width)
            .map(|_| (types::var(self.context), loc))
            .collect();
        region.append_block(Block::new(&arguments))
    }

    fn convert_input<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        stage: &impl AstNode,
        what: &str,
        input: Option<ast::Expr>,
    ) -> Value<'c, 'a> {
        match input {
            Some(ast::Expr::Rel(rel)) => self.convert_rel(block, &rel),
            Some(ast::Expr::IdentExpr(ident)) => {
                let Some(name) = self.ident(ident.name()) else {
                    return self.missing(
                        block,
                        &ident,
                        &format!("{what} is missing its input relation"),
                        types::query(self.context),
                    );
                };

                let loc = self.location(&ident);
                self.scan(block, &ident, name, loc)
            }
            Some(other) => self.missing(
                block,
                &other,
                "expected a relation as the pipe input",
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

    /// The column a stage item names, reported against the item when the row
    /// does not have exactly one of them.
    fn column(
        &mut self,
        node: &impl AstNode,
        what: &str,
        reference: Reference<'c>,
    ) -> Option<usize> {
        let message = match self.symbols.column(reference) {
            ColumnLookup::Unique(index) => return Some(index),
            ColumnLookup::Ambiguous => {
                format!("{what} `{reference}` is ambiguous; qualify it with a relation alias")
            }
            ColumnLookup::Absent => match reference.qualifier {
                Some(qualifier) => format!("`{qualifier}` has no column `{}`", reference.name),
                None => format!("{what} `{reference}` is not in this row"),
            },
        };

        self.unresolved_column(node, &message);
        None
    }

    fn convert_items(
        &mut self,
        items: Vec<(Option<ast::Ident>, Option<ast::Expr>, text_size::TextRange)>,
        what: &str,
        loc: Location<'c>,
    ) -> (Vec<&'c str>, Region<'c>) {
        let region = Region::new();
        let body = self.stage_block(&region, loc);
        let mut names = Vec::with_capacity(items.len());
        let mut values = Vec::with_capacity(items.len());
        for (index, (alias, expr, range)) in items.into_iter().enumerate() {
            let name = self
                .ident(alias)
                .or_else(|| match &expr {
                    Some(ast::Expr::IdentExpr(ident)) => self.ident(ident.name()),
                    _ => None,
                })
                .unwrap_or_else(|| self.intern(&format!("column{index}")));
            names.push(name);
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

    fn string_attrs(&self, names: &[&'c str]) -> Vec<Attribute<'c>> {
        names
            .iter()
            .map(|&name| StringAttribute::new(self.context, name).into())
            .collect()
    }
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
                %5 = yzl.call @sum(%arg2) : (!yzl.var) -> !yzl.var {agg, callee_kind = "builtin"}
                yzl.yield %5 : !yzl.var
              } {key_cols = [1]}
              %4 = yzl.limit %3, 10 offset 2
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
|> limit 10 offset 2
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
