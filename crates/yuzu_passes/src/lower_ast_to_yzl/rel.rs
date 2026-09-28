//! Queries: each stage becomes one `yzl` relational op, with a region for
//! the expressions it evaluates per row.

use melior::ir::attribute::{
    ArrayAttribute, FlatSymbolRefAttribute, IntegerAttribute, StringAttribute,
};
use melior::ir::r#type::IntegerType;
use melior::ir::{
    Block, BlockLike, BlockRef, Location, Operation, Region, RegionLike, Type, Value,
};
use text_size::TextRange;
use yuzu_ast::{AstNode, ast};
use yuzu_mlir::attributes::JoinKind;
use yuzu_mlir::ir::attribute::array::ArrayAttributeExt;
use yuzu_mlir::ir::operation::{OperationExt, OperationMutExt};
use yuzu_mlir::ods::yzl;
use yuzu_mlir::types::{QueryType, UnresolvedType};

use crate::lower_ast_to_yzl::symbols::{ColumnLookup, Reference, Row};
use crate::lower_ast_to_yzl::{AstToYzl, Locals};

/// A stage item: its alias, its expression, and where it was written.
type Item<'c> = (Option<&'c str>, Option<ast::Expr>, TextRange);

impl<'c> AstToYzl<'c, '_> {
    /// A pipeline: its source opens the relation every stage reads, and the
    /// relation closes after the last stage.
    pub(super) fn convert_query<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        pipeline: &ast::Pipeline,
    ) -> (Value<'c, 'a>, Row<'c>) {
        let (mut value, row) = if let Some(from) = pipeline.source() {
            self.convert_from(block, &from)
        } else {
            let hole = self.report_and_hole(
                block,
                pipeline,
                "a query is missing its `from`",
                QueryType::get(self.context),
            );
            (hole, Row::lost())
        };

        self.symbols.enter_relation(row);
        for stage in pipeline.stages() {
            value = self.convert_stage(block, value, &stage);
        }

        (value, self.symbols.leave_relation())
    }

    fn convert_stage<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        input: Value<'c, 'a>,
        stage: &ast::Stage,
    ) -> Value<'c, 'a> {
        match stage {
            ast::Stage::WhereStage(r#where) => self.convert_where(block, input, r#where),
            ast::Stage::SelectStage(select) => self.convert_select(block, input, select),
            ast::Stage::ExtendStage(extend) => self.convert_extend(block, input, extend),
            ast::Stage::AggregateStage(agg) => self.convert_aggregate(block, input, agg),
            ast::Stage::LimitStage(limit) => self.convert_limit(block, input, limit),
            ast::Stage::RenameStage(rename) => self.convert_rename(block, input, rename),
            ast::Stage::AliasStage(alias) => self.convert_alias(block, input, alias),
            ast::Stage::JoinStage(join) => self.convert_join(block, input, join),
            ast::Stage::SetStage(set) => self.convert_set(block, input, set),
            ast::Stage::DistinctStage(distinct) => self.convert_distinct(block, input, distinct),
            ast::Stage::DropStage(drop) => self.convert_drop(block, input, drop),
        }
    }

    /// The relation a pipeline reads, and the row it starts from. An unknown
    /// relation starts from a lost row, so the stages after it are still
    /// checked, and a column they name is not reported against it.
    fn convert_from<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        from: &ast::FromSource,
    ) -> (Value<'c, 'a>, Row<'c>) {
        let loc = self.location(from);
        let Some(source) = self.read_ident(from.relation()) else {
            let hole = self.parser_hole(
                block,
                from,
                "`from` is missing its relation",
                QueryType::get(self.context),
            );
            return (hole, Row::lost());
        };

        let Some((symbol, mut row)) = self.symbols.relation(source, None) else {
            let hole = self.report_and_hole(
                block,
                from,
                &format!("`{source}` is not a relation"),
                QueryType::get(self.context),
            );
            return (hole, Row::lost());
        };

        let mut value = block
            .append_operation(
                yzl::from(
                    self.context,
                    QueryType::get(self.context),
                    FlatSymbolRefAttribute::new(self.context, symbol),
                    loc,
                )
                .into(),
            )
            .first_result();
        if let Some(alias) = self.read_ident(from.alias()) {
            row.qualify(alias);
            value = self.emit_alias(block, value, alias, loc);
        }

        (value, row)
    }

    fn convert_where<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        input: Value<'c, 'a>,
        r#where: &ast::WhereStage,
    ) -> Value<'c, 'a> {
        let loc = self.location(r#where);
        let region = Region::new();
        let body = self.stage_block(&region, loc);
        let predicate = match r#where.predicate() {
            Some(expr) => self.convert_expr(body, &Locals::new(), &expr),
            None => self.parser_hole(
                body,
                r#where,
                "`where` is missing its predicate",
                UnresolvedType::get(self.context),
            ),
        };

        body.append_operation(yzl::r#yield(self.context, &[predicate], loc).into());

        block
            .append_operation(
                yzl::r#where(
                    self.context,
                    QueryType::get(self.context),
                    input,
                    region,
                    loc,
                )
                .into(),
            )
            .first_result()
    }

    fn convert_select<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        input: Value<'c, 'a>,
        select: &ast::SelectStage,
    ) -> Value<'c, 'a> {
        let loc = self.location(select);
        let items = select
            .items()
            .map(|item| {
                (
                    self.read_ident(item.alias()),
                    item.expr(),
                    item.syntax().text_range(),
                )
            })
            .collect::<Vec<_>>();
        let (names, region) = self.convert_items(&items, "select item", loc);
        let columns = ArrayAttribute::from_strings(self.context, &names);
        self.symbols.replace(names);
        block
            .append_operation(
                yzl::select(
                    self.context,
                    QueryType::get(self.context),
                    input,
                    region,
                    columns,
                    loc,
                )
                .into(),
            )
            .first_result()
    }

    fn convert_extend<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        input: Value<'c, 'a>,
        extend: &ast::ExtendStage,
    ) -> Value<'c, 'a> {
        let loc = self.location(extend);
        let items = extend
            .items()
            .map(|item| {
                (
                    self.read_ident(item.alias()),
                    item.expr(),
                    item.syntax().text_range(),
                )
            })
            .collect::<Vec<_>>();
        let (names, region) = self.convert_items(&items, "extend item", loc);
        let columns = ArrayAttribute::from_strings(self.context, &names);
        self.symbols.extend(names);
        block
            .append_operation(
                yzl::extend(
                    self.context,
                    QueryType::get(self.context),
                    input,
                    region,
                    columns,
                    loc,
                )
                .into(),
            )
            .first_result()
    }

    fn convert_aggregate<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        input: Value<'c, 'a>,
        agg: &ast::AggregateStage,
    ) -> Value<'c, 'a> {
        let loc = self.location(agg);
        let mut keys = Vec::new();
        let mut key_names: Vec<&'c str> = Vec::new();
        for item in agg.group_by().into_iter().flat_map(|group| group.items()) {
            let Some(column) = self.read_ident(item.column()) else {
                self.reported_by_parser("group by key is missing its column");
                continue;
            };

            let qualifier = self.read_ident(item.qualifier());

            let reference = Reference {
                qualifier,
                name: column,
            };
            if let Some(index) = self.column(&item, "group key", reference) {
                keys.push(index);
                key_names.push(self.read_ident(item.alias()).unwrap_or(column));
            }
        }

        let items = agg
            .items()
            .map(|item| {
                (
                    self.read_ident(item.alias()),
                    item.expr(),
                    item.syntax().text_range(),
                )
            })
            .collect::<Vec<_>>();
        let (names, region) = self.convert_items(&items, "aggregate item", loc);
        let group_by = ArrayAttribute::from_strings(self.context, &key_names);
        let measures = ArrayAttribute::from_strings(self.context, &names);
        key_names.extend(names);
        self.symbols.replace(key_names);

        let mut op: Operation<'c> = yzl::aggregate(
            self.context,
            QueryType::get(self.context),
            input,
            region,
            group_by,
            measures,
            loc,
        )
        .into();
        op.set_index_array_attribute(self.context, "key_cols", &keys);
        block.append_operation(op).first_result()
    }

    fn convert_limit<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        input: Value<'c, 'a>,
        limit: &ast::LimitStage,
    ) -> Value<'c, 'a> {
        let loc = self.location(limit);
        let count = if let Some(count) = limit.count() {
            self.int_literal(&count)
        } else {
            self.reported_by_parser("`limit` is missing its row count");
            0
        };
        let offset = limit.offset().map(|offset| self.int_literal(&offset));

        let i64 = IntegerType::new(self.context, 64).into();
        let mut builder = yzl::LimitOperationBuilder::new(self.context, loc)
            .result(QueryType::get(self.context))
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
        input: Value<'c, 'a>,
        rename: &ast::RenameStage,
    ) -> Value<'c, 'a> {
        let loc = self.location(rename);
        let mut from: Vec<String> = Vec::new();
        let mut to: Vec<&'c str> = Vec::new();
        let mut renames = Vec::new();
        for item in rename.items() {
            let (Some(old), Some(new)) = (
                self.read_ident(item.column()),
                self.read_ident(item.alias()),
            ) else {
                self.report(&item, "rename item is missing a column name");
                continue;
            };

            let qualifier = self.read_ident(item.qualifier());

            let reference = Reference {
                qualifier,
                name: old,
            };
            if let Some(index) = self.column(&item, "column", reference) {
                renames.push((index, new));
                from.push(reference.to_string());
                to.push(new);
            }
        }

        self.symbols.rename(&renames);
        let indices: Vec<usize> = renames.iter().map(|&(index, _)| index).collect();
        let mut op: Operation<'c> = yzl::rename(
            self.context,
            QueryType::get(self.context),
            input,
            ArrayAttribute::from_strings(self.context, &from),
            ArrayAttribute::from_strings(self.context, &to),
            loc,
        )
        .into();
        op.set_index_array_attribute(self.context, "rename_cols", &indices);
        block.append_operation(op).first_result()
    }

    fn convert_alias<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        input: Value<'c, 'a>,
        alias: &ast::AliasStage,
    ) -> Value<'c, 'a> {
        let loc = self.location(alias);
        let Some(alias) = self.read_ident(alias.alias()) else {
            self.reported_by_parser("`as` is missing its alias");
            return input;
        };

        self.symbols.alias(alias);
        self.emit_alias(block, input, alias, loc)
    }

    fn convert_join<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        input: Value<'c, 'a>,
        join: &ast::JoinStage,
    ) -> Value<'c, 'a> {
        let loc = self.location(join);
        let kind = match join.kind() {
            ast::JoinKind::Inner => JoinKind::Inner,
            ast::JoinKind::Left => JoinKind::Left,
            ast::JoinKind::Right => JoinKind::Right,
            ast::JoinKind::Full => JoinKind::Full,
        };
        let Some(relation) = self.read_ident(join.relation()) else {
            return self.parser_hole(
                block,
                join,
                "`join` is missing its relation",
                QueryType::get(self.context),
            );
        };

        let alias = self.read_ident(join.alias());
        let Some((rhs_symbol, rhs)) = self.symbols.relation(relation, alias) else {
            return self.report_and_hole(
                block,
                join,
                &format!("`{relation}` is not a relation"),
                QueryType::get(self.context),
            );
        };

        let mut using: Vec<&'c str> = Vec::new();
        if let Some(clause) = join.using() {
            using = clause
                .columns()
                .filter_map(|column| self.read_ident(Some(column)))
                .collect();
            if using.is_empty() {
                self.report(&clause, "`using` needs at least one column");
            }

            for column in &using {
                let column = Reference::unqualified(column);
                if !self.symbols.row().has(column) || !rhs.has(column) {
                    self.report(
                        &clause,
                        &format!("column {column} not present in both relations"),
                    );
                }
            }
        }

        // The condition sees both rows, so the row moves first.
        self.symbols.concat(rhs);
        let on = Region::new();
        if join.using().is_none() {
            match join.on() {
                None => self.reported_by_parser("`join` is missing its `on` or `using` clause"),
                Some(clause) => match clause.condition() {
                    None => self.reported_by_parser("`on` is missing its condition"),
                    Some(condition) => {
                        let body = self.stage_block(&on, loc);
                        let value = self.convert_expr(body, &Locals::new(), &condition);
                        body.append_operation(yzl::r#yield(self.context, &[value], loc).into());
                    }
                },
            }
        }

        let mut builder = yzl::JoinOperationBuilder::new(self.context, loc)
            .result(QueryType::get(self.context))
            .lhs(input)
            .kind(StringAttribute::new(self.context, kind.as_str()))
            .rhs(FlatSymbolRefAttribute::new(self.context, rhs_symbol))
            .on(on);

        if let Some(alias) = alias {
            builder = builder.rhs_alias(StringAttribute::new(self.context, alias));
        }
        if !using.is_empty() {
            let columns = ArrayAttribute::from_strings(self.context, &using);
            builder = builder.using_columns(columns);
        }

        block
            .append_operation(builder.build().into())
            .first_result()
    }

    fn convert_set<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        input: Value<'c, 'a>,
        set: &ast::SetStage,
    ) -> Value<'c, 'a> {
        let loc = self.location(set);
        let items: Vec<ast::SetItem> = set.items().collect();
        let mut columns = Vec::new();
        for item in &items {
            let Some(name) = self.read_ident(item.column()) else {
                self.report(item, "set item is incomplete");
                continue;
            };

            if let Some(index) = self.column(item, "column", Reference::unqualified(name)) {
                columns.push(index);
            }
        }

        let items: Vec<_> = items
            .iter()
            .map(|item| {
                (
                    self.read_ident(item.column()),
                    item.value(),
                    item.syntax().text_range(),
                )
            })
            .collect();

        let (names, region) = self.convert_items(&items, "set item", loc);
        let names = ArrayAttribute::from_strings(self.context, &names);

        let mut op: Operation<'c> = yzl::set(
            self.context,
            QueryType::get(self.context),
            input,
            region,
            names,
            loc,
        )
        .into();

        op.set_index_array_attribute(self.context, "set_cols", &columns);
        block.append_operation(op).first_result()
    }

    fn convert_distinct<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        input: Value<'c, 'a>,
        distinct: &ast::DistinctStage,
    ) -> Value<'c, 'a> {
        let loc = self.location(distinct);
        block
            .append_operation(
                yzl::distinct(self.context, QueryType::get(self.context), input, loc).into(),
            )
            .first_result()
    }

    fn convert_drop<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        input: Value<'c, 'a>,
        drop: &ast::DropStage,
    ) -> Value<'c, 'a> {
        let loc = self.location(drop);
        let mut names: Vec<&'c str> = Vec::new();
        for column in drop.columns() {
            let Some(name) = self.read_ident(Some(column.clone())) else {
                continue;
            };

            if let Some(index) = self.column(&column, "column", Reference::unqualified(name)) {
                self.symbols.remove(index);
            }

            names.push(name);
        }

        let columns = ArrayAttribute::from_strings(self.context, &names);
        block
            .append_operation(
                yzl::drop(
                    self.context,
                    QueryType::get(self.context),
                    input,
                    columns,
                    loc,
                )
                .into(),
            )
            .first_result()
    }

    fn int_literal(&mut self, expr: &ast::Expr) -> i64 {
        match expr {
            ast::Expr::Literal(ast::Literal::IntLiteral(int)) => {
                self.int64_value(int).unwrap_or_default()
            }
            other => {
                self.report(other, "`limit` takes an integer literal");
                0
            }
        }
    }

    fn emit_alias<'a>(
        &self,
        block: BlockRef<'c, 'a>,
        input: Value<'c, 'a>,
        alias: &'c str,
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        block
            .append_operation(
                yzl::alias(
                    self.context,
                    QueryType::get(self.context),
                    input,
                    StringAttribute::new(self.context, alias),
                    loc,
                )
                .into(),
            )
            .first_result()
    }

    /// The row's columns are the block arguments, typed by inference later.
    fn stage_block<'r>(&self, region: &'r Region<'c>, loc: Location<'c>) -> BlockRef<'c, 'r> {
        let width = self.symbols.row().len();
        let arguments: Vec<(Type<'c>, Location<'c>)> = (0..width)
            .map(|_| (UnresolvedType::get(self.context), loc))
            .collect();
        region.append_block(Block::new(&arguments))
    }

    fn column(
        &mut self,
        node: &impl AstNode,
        what: &str,
        reference: Reference<'_>,
    ) -> Option<usize> {
        let message = match self.symbols.column(reference) {
            ColumnLookup::Unique(index) => return Some(index),
            ColumnLookup::Lost => return None,
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
        items: &[Item<'c>],
        what: &str,
        loc: Location<'c>,
    ) -> (Vec<&'c str>, Region<'c>) {
        let region = Region::new();
        let body = self.stage_block(&region, loc);
        let mut names = Vec::new();
        let mut values = Vec::new();
        for (index, (alias, expr, range)) in items.iter().enumerate() {
            let name = alias
                .or_else(|| match expr {
                    Some(ast::Expr::IdentExpr(ident)) => self.read_ident(ident.name()),
                    _ => None,
                })
                .unwrap_or_else(|| self.symbols.intern(&format!("column{index}")));

            names.push(name);

            let value = if let Some(expr) = expr {
                self.convert_expr(body, &Locals::new(), expr)
            } else {
                self.reported_by_parser(&format!("{what} is missing its expression"));
                self.emit_hole(body, *range, UnresolvedType::get(self.context))
            };

            values.push(value);
        }

        body.append_operation(yzl::r#yield(self.context, &values, loc).into());
        (names, region)
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::test_support::{lowered, reported};

    #[test]
    fn a_limit_past_int64_is_reported() {
        expect![[r"
            error: integer literal is out of range for `int64`
             --> test.yz:6:10
              |
            6 | |> limit 9223372036854775808
              |          ^^^^^^^^^^^^^^^^^^^
        "]]
        .assert_eq(&reported(
            r"
struct Row { a: int64 }
table t = Row

from t
|> limit 9223372036854775808
",
        ));
    }

    #[test]
    fn converts_the_canonical_pipeline() {
        expect![[r#"
            module {
              yzl.struct @Row ["a", "b"] : [!yz.int64, !yz.int64] {sym_visibility = "private"}
              yzl.table @t of @Row {sym_visibility = "private"}
              yzl.fn @f params ["x"] (!yz.int64) -> !yz.int64 {
              ^bb0(%arg0: !yzl.unresolved):
                %5 = yzl.local "x" param
                yzl.store %5, %arg0 : !yzl.unresolved
                %6 = yzl.load %5 : !yzl.unresolved
                yzl.return %6 : !yzl.unresolved
              } {sym_visibility = "private"}
              %0 = yzl.from @t
              %1 = yzl.where %0 {
              ^bb0(%arg0: !yzl.unresolved, %arg1: !yzl.unresolved):
                %5 = yz.constant_int 10
                %6 = yz.cmp "gt", %arg0, %5 : !yzl.unresolved, !yz.int64 -> !yzl.unresolved
                yzl.yield %6 : !yzl.unresolved
              }
              %2 = yzl.extend %1 as ["e"] {
              ^bb0(%arg0: !yzl.unresolved, %arg1: !yzl.unresolved):
                %5 = yzl.call @f(%arg0) : (!yzl.unresolved) -> !yzl.unresolved {callee_source = "fn"}
                %6 = yz.add %5, %arg1 : !yzl.unresolved, !yzl.unresolved -> !yzl.unresolved
                yzl.yield %6 : !yzl.unresolved
              }
              %3 = yzl.aggregate %2 group_by ["b"] as ["s"] {
              ^bb0(%arg0: !yzl.unresolved, %arg1: !yzl.unresolved, %arg2: !yzl.unresolved):
                %5 = yzl.call @yuzu.prelude.sum(%arg2) : (!yzl.unresolved) -> !yzl.unresolved {callee_source = "external", is_agg}
                yzl.yield %5 : !yzl.unresolved
              } {key_cols = [1]}
              %4 = yzl.limit %3, 10 offset 2
              yzl.fn @yuzu.prelude.sum generics ["T"] params ["x"] (!yzl.param<"T">) -> !yzl.param<"T"> agg external "sum" {
              }
              yzl.output %4
            }
        "#]]
        .assert_eq(&lowered(
            r"
struct Row { a: int64, b: int64 }
table t = Row
def f(x: int64) -> int64 { return x }

from t
|> where a > 10
|> extend f(a) + b as e
|> aggregate sum(e) as s group by b
|> limit 10 offset 2
",
        ));
    }

    #[test]
    fn converts_joins_sets_and_membership() {
        expect![[r#"
            module {
              yzl.struct @Employee ["id", "dept_id", "level", "rating"] : [!yz.str, !yz.int64, !yz.int64, !yz.float64] {sym_visibility = "private"}
              yzl.table @employees of @Employee {sym_visibility = "private"}
              yzl.struct @Department ["id", "dept_id"] : [!yz.str, !yz.int64] {sym_visibility = "private"}
              yzl.table @departments of @Department {sym_visibility = "private"}
              %0 = yzl.from @employees
              %1 = yzl.alias %0 as "e"
              %2 = yzl.join "inner", %1, @departments as "d" {
              ^bb0(%arg0: !yzl.unresolved, %arg1: !yzl.unresolved, %arg2: !yzl.unresolved, %arg3: !yzl.unresolved, %arg4: !yzl.unresolved, %arg5: !yzl.unresolved):
                %7 = yz.cmp "eq", %arg1, %arg4 : !yzl.unresolved, !yzl.unresolved -> !yzl.unresolved
                yzl.yield %7 : !yzl.unresolved
              }
              %3 = yzl.set %2 as ["level"] {
              ^bb0(%arg0: !yzl.unresolved, %arg1: !yzl.unresolved, %arg2: !yzl.unresolved, %arg3: !yzl.unresolved, %arg4: !yzl.unresolved, %arg5: !yzl.unresolved):
                %7 = yz.constant_int 1
                %8 = yz.add %arg2, %7 : !yzl.unresolved, !yz.int64 -> !yzl.unresolved
                yzl.yield %8 : !yzl.unresolved
              } {set_cols = [2]}
              %4 = yzl.where %3 {
              ^bb0(%arg0: !yzl.unresolved, %arg1: !yzl.unresolved, %arg2: !yzl.unresolved, %arg3: !yzl.unresolved, %arg4: !yzl.unresolved, %arg5: !yzl.unresolved):
                %7 = yz.constant_int 1
                %8 = yz.constant_int 3
                %9 = yzl.list[%7, %8] : (!yz.int64, !yz.int64) -> !yzl.unresolved
                %10 = yz.in %arg2, %9 : !yzl.unresolved, !yzl.unresolved -> !yzl.unresolved
                yzl.yield %10 : !yzl.unresolved
              }
              %5 = yzl.drop %4 ["rating"]
              %6 = yzl.distinct %5
              yzl.output %6
            }
        "#]]
        .assert_eq(&lowered(
            r"
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
",
        ));
    }

    #[test]
    fn converts_sugar_and_bindings() {
        expect![[r#"
            module {
              yzl.struct @Row ["a", "active"] : [!yz.int64, !yz.bool] {sym_visibility = "private"}
              yzl.table @t of @Row {sym_visibility = "private"}
              yzl.const @base {
                %4 = yzl.from @t
                %5 = yzl.where %4 {
                ^bb0(%arg0: !yzl.unresolved, %arg1: !yzl.unresolved):
                  yzl.yield %arg1 : !yzl.unresolved
                }
                yzl.yield %5 : !yzl.query
              } {sym_visibility = "private"}
              %0 = yzl.from @base
              %1 = yzl.rename %0 from ["a"] to ["renamed"] {rename_cols = [0]}
              %2 = yzl.alias %1 as "q"
              %3 = yzl.select %2 as ["out"] {
              ^bb0(%arg0: !yzl.unresolved, %arg1: !yzl.unresolved):
                yzl.yield %arg0 : !yzl.unresolved
              }
              yzl.output %3
            }
        "#]]
        .assert_eq(&lowered(
            r"
struct Row { a: int64, active: bool }
table t = Row

let base = from t |> where active

from base
|> rename a as renamed
|> as q
|> select q.renamed as out
",
        ));
    }
}
