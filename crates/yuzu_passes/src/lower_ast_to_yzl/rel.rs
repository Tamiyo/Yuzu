//! Queries: each stage becomes one `yzl` relational op, with a region for
//! the expressions it evaluates per row.

use melior::ir::attribute::{
    ArrayAttribute, FlatSymbolRefAttribute, IntegerAttribute, StringAttribute,
};
use melior::ir::r#type::IntegerType;
use melior::ir::{Block, BlockLike, BlockRef, Location, Region, RegionLike, Value};
use text_size::TextRange;
use yuzu_ast::ast::{self, AstNode};
use yuzu_mlir::attributes::JoinKind;
use yuzu_mlir::ir::attribute::array::ArrayAttributeExt;
use yuzu_mlir::ir::operation::OperationExt;
use yuzu_mlir::ods::yzl;
use yuzu_mlir::types::{QueryType, UnresolvedType};

use crate::lower_ast_to_yzl::symbols::{ColumnLookup, Field, Reference, Row};
use crate::lower_ast_to_yzl::{AstToYzl, Locals};

/// A stage item as the source wrote it.
struct Item<'c> {
    alias: Option<&'c str>,
    expr: Option<ast::Expr>,
    range: TextRange,
}

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
            let hole = self.hole_and_report(
                block,
                pipeline,
                "a query is missing its `from`",
                QueryType::new(self.context).into(),
            );
            (hole, Row::lost())
        };

        let ((), row) = self.in_relation(row, |this| {
            for stage in pipeline.stages() {
                value = this.convert_stage(block, value, &stage);
            }
        });
        (value, row)
    }

    fn convert_stage<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        input: Value<'c, 'a>,
        stage: &ast::Stage,
    ) -> Value<'c, 'a> {
        match stage {
            ast::Stage::WhereStage(where_) => self.convert_where(block, input, where_),
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
            let hole = self.hole_and_assert(
                block,
                from,
                "`from` is missing its relation",
                QueryType::new(self.context).into(),
            );
            return (hole, Row::lost());
        };

        let Some((symbol, mut row, target)) = self.symbols.relation(source, None) else {
            let hole = self.hole_and_report(
                block,
                from,
                &format!("`{source}` is not a relation"),
                QueryType::new(self.context).into(),
            );
            return (hole, Row::lost());
        };
        if let Some(relation) = from.relation() {
            self.record(relation.syntax().text_range(), source, target);
        }

        let mut value = block
            .append_operation(
                yzl::from(
                    self.context,
                    QueryType::new(self.context).into(),
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
        where_: &ast::WhereStage,
    ) -> Value<'c, 'a> {
        let loc = self.location(where_);
        let region = Region::new();
        let body = self.stage_block(&region, loc);
        let predicate = match where_.predicate() {
            Some(expr) => self.convert_expr(body, &Locals::new(), &expr),
            None => self.hole_and_assert(
                body,
                where_,
                "`where` is missing its predicate",
                UnresolvedType::new(self.context).into(),
            ),
        };

        body.append_operation(yzl::r#yield(self.context, &[predicate], loc).into());

        block
            .append_operation(
                yzl::r#where(
                    self.context,
                    QueryType::new(self.context).into(),
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
            .map(|item| Item {
                alias: self.read_ident(item.alias()),
                expr: item.expr(),
                range: item.syntax().text_range(),
            })
            .collect::<Vec<_>>();
        let (fields, region) = self.convert_items(&items, "select item", loc);
        let columns = ArrayAttribute::from_strings(self.context, field_names(&fields));
        self.symbols.replace(fields);
        block
            .append_operation(
                yzl::select(
                    self.context,
                    QueryType::new(self.context).into(),
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
            .map(|item| Item {
                alias: self.read_ident(item.alias()),
                expr: item.expr(),
                range: item.syntax().text_range(),
            })
            .collect::<Vec<_>>();
        let (fields, region) = self.convert_items(&items, "extend item", loc);
        let columns = ArrayAttribute::from_strings(self.context, field_names(&fields));
        self.symbols.extend(fields);
        block
            .append_operation(
                yzl::extend(
                    self.context,
                    QueryType::new(self.context).into(),
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
        let mut keys: Vec<usize> = Vec::new();
        let mut key_fields: Vec<Field<'c>> = Vec::new();
        for item in agg.group_by().into_iter().flat_map(|group| group.items()) {
            let Some(written) = item.column() else {
                self.assert_syntax_error("group by key is missing its column");
                continue;
            };
            let Some(column) = self.read_ident(Some(written.clone())) else {
                continue;
            };

            let qualifier = self.read_ident(item.qualifier());

            let reference = Reference {
                qualifier,
                name: column,
            };
            let used = written.syntax().text_range();
            if let Some(index) = self.resolve_column(&item, used, "group key", reference) {
                keys.push(index);
                key_fields.push(match self.read_ident(item.alias()) {
                    Some(alias) => Field {
                        name: alias,
                        declared: Some(self.span(item.syntax().text_range())),
                    },
                    None => Field {
                        name: column,
                        declared: self.symbols.row().declared(index),
                    },
                });
            }
        }

        let items = agg
            .items()
            .map(|item| Item {
                alias: self.read_ident(item.alias()),
                expr: item.expr(),
                range: item.syntax().text_range(),
            })
            .collect::<Vec<_>>();
        let (fields, region) = self.convert_items(&items, "aggregate item", loc);
        let group_by = ArrayAttribute::from_strings(self.context, field_names(&key_fields));
        let measures = ArrayAttribute::from_strings(self.context, field_names(&fields));
        key_fields.extend(fields);
        self.symbols.replace(key_fields);

        let op = yzl::AggregateOperationBuilder::new(self.context, loc)
            .result(QueryType::new(self.context).into())
            .input(input)
            .body(region)
            .group_by(group_by)
            .names(measures)
            .key_cols(ArrayAttribute::from_indices(self.context, keys))
            .build();
        block.append_operation(op.into()).first_result()
    }

    fn convert_limit<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        input: Value<'c, 'a>,
        limit: &ast::LimitStage,
    ) -> Value<'c, 'a> {
        let loc = self.location(limit);
        let count = if let Some(count) = limit.count() {
            self.read_limit_count(&count)
        } else {
            self.assert_syntax_error("`limit` is missing its row count");
            0
        };
        let offset = limit.offset().map(|offset| self.read_limit_count(&offset));

        let int64 = IntegerType::new(self.context, 64).into();
        let mut builder = yzl::LimitOperationBuilder::new(self.context, loc)
            .result(QueryType::new(self.context).into())
            .input(input)
            .count(IntegerAttribute::new(int64, count));
        if let Some(offset) = offset {
            builder = builder.offset(IntegerAttribute::new(int64, offset));
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
                self.assert_syntax_error("rename item is missing a column name");
                continue;
            };

            let qualifier = self.read_ident(item.qualifier());

            let reference = Reference {
                qualifier,
                name: old,
            };
            let used = item.column().map_or_else(
                || item.syntax().text_range(),
                |column| column.syntax().text_range(),
            );
            if let Some(index) = self.resolve_column(&item, used, "column", reference) {
                let field = Field {
                    name: new,
                    declared: Some(self.span(item.syntax().text_range())),
                };
                renames.push((index, field));
                from.push(reference.to_string());
                to.push(new);
            }
        }

        self.symbols.rename(&renames);
        let indices = renames.iter().map(|&(index, _)| index);
        let op = yzl::RenameOperationBuilder::new(self.context, loc)
            .result(QueryType::new(self.context).into())
            .input(input)
            .from(ArrayAttribute::from_strings(self.context, &from))
            .to(ArrayAttribute::from_strings(self.context, &to))
            .rename_cols(ArrayAttribute::from_indices(self.context, indices))
            .build();
        block.append_operation(op.into()).first_result()
    }

    fn convert_alias<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        input: Value<'c, 'a>,
        alias: &ast::AliasStage,
    ) -> Value<'c, 'a> {
        let loc = self.location(alias);
        let Some(alias) = self.read_ident(alias.alias()) else {
            self.assert_syntax_error("`as` is missing its alias");
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
            return self.hole_and_assert(
                block,
                join,
                "`join` is missing its relation",
                QueryType::new(self.context).into(),
            );
        };

        let alias = self.read_ident(join.alias());
        let Some((rhs_symbol, rhs, target)) = self.symbols.relation(relation, alias) else {
            return self.hole_and_report(
                block,
                join,
                &format!("`{relation}` is not a relation"),
                QueryType::new(self.context).into(),
            );
        };
        if let Some(written) = join.relation() {
            self.record(written.syntax().text_range(), relation, target);
        }

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
                if !self.symbols.row().has_column(column) || !rhs.has_column(column) {
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
                None => self.assert_syntax_error("`join` is missing its `on` or `using` clause"),
                Some(clause) => match clause.condition() {
                    None => self.assert_syntax_error("`on` is missing its condition"),
                    Some(condition) => {
                        let body = self.stage_block(&on, loc);
                        let value = self.convert_expr(body, &Locals::new(), &condition);
                        body.append_operation(yzl::r#yield(self.context, &[value], loc).into());
                    }
                },
            }
        }

        let mut builder = yzl::JoinOperationBuilder::new(self.context, loc)
            .result(QueryType::new(self.context).into())
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
        // An item whose column does not resolve is left out of both lists,
        // so each column stays in step with its value.
        let mut columns: Vec<usize> = Vec::new();
        let mut items = Vec::new();
        for item in set.items() {
            let Some(written) = item.column() else {
                self.assert_syntax_error("set item is missing its column");
                continue;
            };
            let Some(name) = self.read_ident(Some(written.clone())) else {
                continue;
            };

            let used = written.syntax().text_range();
            let Some(index) =
                self.resolve_column(&item, used, "column", Reference::unqualified(name))
            else {
                continue;
            };

            columns.push(index);
            items.push(Item {
                alias: Some(name),
                expr: item.value(),
                range: item.syntax().text_range(),
            });
        }

        let (fields, region) = self.convert_items(&items, "set item", loc);
        let op = yzl::SetOperationBuilder::new(self.context, loc)
            .result(QueryType::new(self.context).into())
            .input(input)
            .body(region)
            .names(ArrayAttribute::from_strings(
                self.context,
                field_names(&fields),
            ))
            .set_cols(ArrayAttribute::from_indices(self.context, columns))
            .build();
        block.append_operation(op.into()).first_result()
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
                yzl::distinct(
                    self.context,
                    QueryType::new(self.context).into(),
                    input,
                    loc,
                )
                .into(),
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
                self.assert_syntax_error("`drop` is missing a column name");
                continue;
            };

            let used = column.syntax().text_range();
            if let Some(index) =
                self.resolve_column(&column, used, "column", Reference::unqualified(name))
            {
                self.symbols.remove(index);
                names.push(name);
            }
        }

        let columns = ArrayAttribute::from_strings(self.context, &names);
        block
            .append_operation(
                yzl::drop(
                    self.context,
                    QueryType::new(self.context).into(),
                    input,
                    columns,
                    loc,
                )
                .into(),
            )
            .first_result()
    }

    fn convert_items(
        &mut self,
        items: &[Item<'c>],
        what: &str,
        loc: Location<'c>,
    ) -> (Vec<Field<'c>>, Region<'c>) {
        let region = Region::new();
        let body = self.stage_block(&region, loc);
        let mut fields = Vec::new();
        let mut values = Vec::new();
        for (index, Item { alias, expr, range }) in items.iter().enumerate() {
            // An item that reads a column as it is keeps the column's name,
            // and where the column was named.
            let read = match expr {
                Some(ast::Expr::IdentExpr(ident)) => {
                    self.read_ident(ident.name()).map(Reference::unqualified)
                }
                Some(ast::Expr::FieldAccessExpr(access)) => {
                    let qualifier = match access.base() {
                        Some(ast::Expr::IdentExpr(base)) => self.read_ident(base.name()),
                        _ => None,
                    };
                    self.read_ident(access.field())
                        .map(|name| Reference { qualifier, name })
                }
                _ => None,
            };
            let field = match (alias, read) {
                (Some(alias), _) => Field {
                    name: alias,
                    declared: Some(self.span(*range)),
                },
                (None, Some(read)) => Field {
                    name: read.name,
                    declared: match self.symbols.column(read) {
                        ColumnLookup::Unique(at) => self.symbols.row().declared(at),
                        ColumnLookup::Ambiguous | ColumnLookup::Absent | ColumnLookup::Lost => None,
                    },
                },
                (None, None) => Field {
                    name: self.symbols.intern_fmt(format_args!("column{index}")),
                    declared: None,
                },
            };

            fields.push(field);

            let value = if let Some(expr) = expr {
                self.convert_expr(body, &Locals::new(), expr)
            } else {
                self.assert_syntax_error(&format!("{what} is missing its expression"));
                self.emit_hole(body, *range, UnresolvedType::new(self.context).into())
            };

            values.push(value);
        }

        body.append_operation(yzl::r#yield(self.context, &values, loc).into());
        (fields, region)
    }

    fn read_limit_count(&mut self, expr: &ast::Expr) -> i64 {
        match expr {
            ast::Expr::Literal(ast::Literal::IntLiteral(int)) => {
                self.read_int64(int).unwrap_or_default()
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
                    QueryType::new(self.context).into(),
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
        let arguments = vec![(UnresolvedType::new(self.context).into(), loc); width];
        region.append_block(Block::new(&arguments))
    }

    /// The column `reference` names, which `used` writes.
    fn resolve_column(
        &mut self,
        node: &impl AstNode,
        used: TextRange,
        what: &str,
        reference: Reference<'_>,
    ) -> Option<usize> {
        let message = match self.symbols.column(reference) {
            ColumnLookup::Unique(index) => {
                if let Some(declared) = self.symbols.row().declared(index) {
                    self.record_declared(used, reference.name, declared);
                }
                return Some(index);
            }
            ColumnLookup::Lost => return None,
            ColumnLookup::Ambiguous => {
                format!("{what} `{reference}` is ambiguous; qualify it with a relation alias")
            }
            ColumnLookup::Absent => match reference.qualifier {
                Some(qualifier) => format!("`{qualifier}` has no column `{}`", reference.name),
                None => format!("{what} `{reference}` is not in this row"),
            },
        };

        self.report_unresolved(node, &message);
        None
    }
}

fn field_names<'f, 'c>(fields: &'f [Field<'c>]) -> impl Iterator<Item = &'c str> + use<'f, 'c> {
    fields.iter().map(|field| field.name)
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::test_support::{check, check_yzr, lowered, reported};

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
                %5 = yzl.local "x" param : !yzl.ref<!yz.int64>
                yzl.store %5, %arg0 : !yzl.ref<!yz.int64>, !yzl.unresolved
                %6 = yzl.load %5 : !yzl.ref<!yz.int64> -> !yzl.unresolved
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
    fn a_set_item_that_does_not_resolve_leaves_the_others_in_step() {
        check(
            "struct Row { a: int64, b: str }\ntable t = Row\nfrom t |> set zz = 1, b = \"x\"\n",
            |context, module| {
                crate::infer_types(context, module);
                crate::promote_locals(context, module);
                String::new()
            },
            &expect![[r#"
                error: column `zz` is not in this row
                 --> test.yz:3:15
                  |
                3 | from t |> set zz = 1, b = "x"
                  |               ^^^^^^
                  = note: the row carries `a`, `b`
            "#]],
        );
    }

    #[test]
    fn a_dropped_column_that_does_not_resolve_is_reported_once() {
        check_yzr(
            "struct Row { a: int64 }\ntable t = Row\nfrom t |> drop zz\n",
            &expect![[r"
                error: column `zz` is not in this row
                 --> test.yz:3:16
                  |
                3 | from t |> drop zz
                  |                ^^
                  = note: the row carries `a`
            "]],
        );
    }

    #[test]
    fn an_item_takes_the_name_of_the_field_it_reads() {
        expect![[r#"
            module {
              yzl.struct @Row ["a"] : [!yz.int64] {sym_visibility = "private"}
              yzl.table @t of @Row {sym_visibility = "private"}
              %0 = yzl.from @t
              %1 = yzl.alias %0 as "e"
              %2 = yzl.select %1 as ["a"] {
              ^bb0(%arg0: !yzl.unresolved):
                yzl.yield %arg0 : !yzl.unresolved
              }
              %3 = yzl.where %2 {
              ^bb0(%arg0: !yzl.unresolved):
                %4 = yz.constant_int 1
                %5 = yz.cmp "gt", %arg0, %4 : !yzl.unresolved, !yz.int64 -> !yzl.unresolved
                yzl.yield %5 : !yzl.unresolved
              }
              yzl.output %3
            }
        "#]]
        .assert_eq(&lowered(
            "struct Row { a: int64 }\ntable t = Row\nfrom t as e |> select e.a |> where a > 1\n",
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
                %7 = yz.constant_list [1, 3] : <!yz.int64>
                %8 = yz.in %arg2, %7 : !yzl.unresolved, !yz.list<!yz.int64> -> !yzl.unresolved
                yzl.yield %8 : !yzl.unresolved
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
