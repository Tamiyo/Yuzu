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

impl<'c, 'd> AstToYzl<'c, 'd> {
    pub(super) fn convert_query<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        rel: &ast::Rel,
    ) -> (Value<'c, 'a>, Row<'c>) {
        let value = self.convert_rel(block, rel);
        let row = self.symbols.row().clone();
        self.symbols.leave();
        (value, row)
    }

    pub(super) fn convert_rel<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        rel: &ast::Rel,
    ) -> Value<'c, 'a> {
        let at = stage_range(rel);
        match rel {
            ast::Rel::FromExpr(from) => self.convert_from(block, from, at),
            ast::Rel::WhereExpr(stage) => self.convert_where(block, stage, at),
            ast::Rel::SelectExpr(stage) => self.convert_select(block, stage, at),
            ast::Rel::ExtendExpr(stage) => self.convert_extend(block, stage, at),
            ast::Rel::AggregateExpr(stage) => self.convert_aggregate(block, stage, at),
            ast::Rel::LimitExpr(stage) => self.convert_limit(block, stage, at),
            ast::Rel::RenameExpr(stage) => self.convert_rename(block, stage, at),
            ast::Rel::AliasExpr(stage) => self.convert_alias(block, stage, at),
            ast::Rel::JoinExpr(stage) => self.convert_join(block, stage, at),
            ast::Rel::SetExpr(stage) => self.convert_set(block, stage, at),
            ast::Rel::DistinctExpr(stage) => self.convert_distinct(block, stage, at),
            ast::Rel::DropExpr(stage) => self.convert_drop(block, stage, at),
        }
    }

    fn convert_from<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        from: &ast::FromExpr,
        at: TextRange,
    ) -> Value<'c, 'a> {
        let loc = self.location_at(at);
        let Some(source) = self.read_ident(from.relation()) else {
            return self.report_and_hole(
                block,
                from,
                "`from` is missing its relation",
                QueryType::get(self.context),
            );
        };

        let value = self.scan(block, from, source, loc);
        match self.read_ident(from.alias()) {
            Some(alias) => self.qualify(block, value, alias, loc),
            None => value,
        }
    }

    fn convert_where<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        r#where: &ast::WhereExpr,
        at: TextRange,
    ) -> Value<'c, 'a> {
        let loc = self.location_at(at);
        let input = self.convert_input(block, r#where, "`where`", r#where.input());
        let region = Region::new();
        let body = self.stage_block(&region, loc);
        let predicate = match r#where.predicate() {
            Some(expr) => self.convert_expr(body, &Locals::new(), &expr),
            None => self.report_and_hole(
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
        select: &ast::SelectExpr,
        at: TextRange,
    ) -> Value<'c, 'a> {
        let loc = self.location_at(at);
        let input = self.convert_input(block, select, "`select`", select.input());
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
        let (names, region) = self.convert_items(items.into_iter(), "select item", loc);
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
        extend: &ast::ExtendExpr,
        at: TextRange,
    ) -> Value<'c, 'a> {
        let loc = self.location_at(at);
        let input = self.convert_input(block, extend, "`extend`", extend.input());
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
        let (names, region) = self.convert_items(items.into_iter(), "extend item", loc);
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
        agg: &ast::AggregateExpr,
        at: TextRange,
    ) -> Value<'c, 'a> {
        let loc = self.location_at(at);
        let input = self.convert_input(block, agg, "`aggregate`", agg.input());
        let mut keys = Vec::new();
        let mut key_names: Vec<&'c str> = Vec::new();
        for item in agg.group_by().into_iter().flat_map(|group| group.items()) {
            let Some(column) = self.read_ident(item.column()) else {
                self.report(&item, "group by key is missing its column");
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
        let (names, region) = self.convert_items(items.into_iter(), "aggregate item", loc);
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
        limit: &ast::LimitExpr,
        at: TextRange,
    ) -> Value<'c, 'a> {
        let loc = self.location_at(at);
        let input = self.convert_input(block, limit, "`limit`", limit.input());
        let count = match limit.count() {
            Some(count) => self.int_literal(&count),
            None => {
                self.report_at(at, "`limit` is missing its row count");
                0
            }
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
        rename: &ast::RenameExpr,
        at: TextRange,
    ) -> Value<'c, 'a> {
        let loc = self.location_at(at);
        let input = self.convert_input(block, rename, "`rename`", rename.input());
        let mut from: Vec<String> = Vec::new();
        let mut to: Vec<&'c str> = Vec::new();
        let mut renames = Vec::new();
        for item in rename.items() {
            let (Some(old), Some(new)) = (self.read_ident(item.from()), self.read_ident(item.to()))
            else {
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
        alias: &ast::AliasExpr,
        at: TextRange,
    ) -> Value<'c, 'a> {
        let loc = self.location_at(at);
        let input = self.convert_input(block, alias, "`as`", alias.input());
        let Some(alias) = self.read_ident(alias.alias()) else {
            self.report_at(at, "`as` is missing its alias");
            return input;
        };

        self.qualify(block, input, alias, loc)
    }

    fn convert_join<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        join: &ast::JoinExpr,
        at: TextRange,
    ) -> Value<'c, 'a> {
        let loc = self.location_at(at);
        let lhs = self.convert_input(block, join, "`join`", join.input());
        let kind = match join.kind() {
            Some(ast::JoinKind::Left) => JoinKind::Left,
            Some(ast::JoinKind::Right) => JoinKind::Right,
            Some(ast::JoinKind::Full) => JoinKind::Full,
            Some(ast::JoinKind::Inner) | None => JoinKind::Inner,
        };
        let Some(relation) = self.read_ident(join.relation()) else {
            return self.report_and_hole(
                block,
                join,
                "`join` is missing its relation",
                QueryType::get(self.context),
            );
        };

        let alias = self.read_ident(join.alias());
        let Some((rhs_symbol, rhs)) = self.symbols.relation(relation, alias) else {
            self.report_at(at, &format!("`{relation}` is not a relation"));
            return self.emit_hole(block, at, QueryType::get(self.context));
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
                None => self.report_at(at, "`join` is missing its `on` or `using` clause"),
                Some(clause) => match clause.condition() {
                    None => self.report(&clause, "`on` is missing its condition"),
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
            .lhs(lhs)
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
        set: &ast::SetExpr,
        at: TextRange,
    ) -> Value<'c, 'a> {
        let loc = self.location_at(at);
        let input = self.convert_input(block, set, "`set`", set.input());
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

        let (names, region) = self.convert_items(items.into_iter(), "set item", loc);
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
        distinct: &ast::DistinctExpr,
        at: TextRange,
    ) -> Value<'c, 'a> {
        let loc = self.location_at(at);
        let input = self.convert_input(block, distinct, "`distinct`", distinct.input());
        block
            .append_operation(
                yzl::distinct(self.context, QueryType::get(self.context), input, loc).into(),
            )
            .first_result()
    }

    fn convert_drop<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        drop: &ast::DropExpr,
        at: TextRange,
    ) -> Value<'c, 'a> {
        let loc = self.location_at(at);
        let input = self.convert_input(block, drop, "`drop`", drop.input());
        let mut names: Vec<String> = Vec::new();
        for column in drop.columns() {
            let Some(name) = column.clone().text() else {
                continue;
            };

            if let Some(index) = self.column(&column, "column", Reference::unqualified(&name)) {
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
                int.value().unwrap_or_default() as i64
            }
            other => {
                self.report(other, "`limit` takes an integer literal");
                0
            }
        }
    }

    /// An unknown relation scans an empty row, so the stages after it are
    /// still checked.
    fn scan<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        node: &impl AstNode,
        source: &str,
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        // The stages after this one still need a scope to resolve in, so an
        // unknown relation opens an empty one and stands a hole.
        let Some((symbol, row)) = self.symbols.relation(source, None) else {
            self.symbols.enter_relation(Row::new());
            return self.report_and_hole(
                block,
                node,
                &format!("`{source}` is not a relation"),
                QueryType::get(self.context),
            );
        };

        self.symbols.enter_relation(row);
        block
            .append_operation(
                yzl::from(
                    self.context,
                    QueryType::get(self.context),
                    FlatSymbolRefAttribute::new(self.context, symbol),
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

    fn convert_input<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        node: &impl AstNode,
        what: &str,
        input: Option<ast::Expr>,
    ) -> Value<'c, 'a> {
        match input {
            Some(ast::Expr::Rel(rel)) => self.convert_rel(block, &rel),
            Some(ast::Expr::IdentExpr(ident)) => {
                let Some(name) = self.read_ident(ident.name()) else {
                    return self.report_and_hole(
                        block,
                        &ident,
                        &format!("{what} is missing its input relation"),
                        QueryType::get(self.context),
                    );
                };

                let loc = self.location(&ident);
                self.scan(block, &ident, name, loc)
            }
            Some(other) => self.report_and_hole(
                block,
                &other,
                "expected a relation as the pipe input",
                QueryType::get(self.context),
            ),
            None => self.report_and_hole(
                block,
                node,
                &format!("{what} is missing its input relation"),
                QueryType::get(self.context),
            ),
        }
    }

    fn column(
        &mut self,
        node: &impl AstNode,
        what: &str,
        reference: Reference<'_>,
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
        items: impl Iterator<Item = Item<'c>>,
        what: &str,
        loc: Location<'c>,
    ) -> (Vec<&'c str>, Region<'c>) {
        let region = Region::new();
        let body = self.stage_block(&region, loc);
        let mut names = Vec::new();
        let mut values = Vec::new();
        for (index, (alias, expr, range)) in items.enumerate() {
            let name = alias
                .or_else(|| match &expr {
                    Some(ast::Expr::IdentExpr(ident)) => self.read_ident(ident.name()),
                    _ => None,
                })
                .unwrap_or_else(|| self.symbols.intern(&format!("column{index}")));

            names.push(name);

            let value = match &expr {
                Some(expr) => self.convert_expr(body, &Locals::new(), expr),
                None => {
                    self.report_at(range, &format!("{what} is missing its expression"));
                    self.emit_hole(body, range, UnresolvedType::get(self.context))
                }
            };

            values.push(value);
        }

        body.append_operation(yzl::r#yield(self.context, &values, loc).into());
        (names, region)
    }
}

fn stage_input(rel: &ast::Rel) -> Option<ast::Expr> {
    match rel {
        ast::Rel::FromExpr(_) => None,
        ast::Rel::WhereExpr(stage) => stage.input(),
        ast::Rel::SelectExpr(stage) => stage.input(),
        ast::Rel::ExtendExpr(stage) => stage.input(),
        ast::Rel::AggregateExpr(stage) => stage.input(),
        ast::Rel::LimitExpr(stage) => stage.input(),
        ast::Rel::RenameExpr(stage) => stage.input(),
        ast::Rel::AliasExpr(stage) => stage.input(),
        ast::Rel::JoinExpr(stage) => stage.input(),
        ast::Rel::SetExpr(stage) => stage.input(),
        ast::Rel::DistinctExpr(stage) => stage.input(),
        ast::Rel::DropExpr(stage) => stage.input(),
    }
}

/// A stage's node covers everything piped into it, so a complaint about the
/// stage points at the first token after its input instead.
fn stage_range(rel: &ast::Rel) -> TextRange {
    let node = rel.syntax();
    let Some(input) = stage_input(rel) else {
        return node.text_range();
    };

    // The input is the stage node's first child and `|>` is one of its own
    // tokens, so the direct children are enough; walking every token under
    // the node would walk the whole pipeline before it again.
    let after = input.syntax().text_range().end();
    let start = node
        .children_with_tokens()
        .filter_map(|element| element.into_token())
        .find(|token| token.text_range().start() >= after && !token.kind().is_trivia())
        .map_or(after, |token| token.text_range().start());

    TextRange::new(start, node.text_range().end())
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::test_support::lowered;

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
                %5 = yzl.call @sum(%arg2) : (!yzl.unresolved) -> !yzl.unresolved {callee_source = "builtin", is_agg}
                yzl.yield %5 : !yzl.unresolved
              } {key_cols = [1]}
              %4 = yzl.limit %3, 10 offset 2
              yzl.output %4
            }
        "#]]
        .assert_eq(&lowered(
            r#"
struct Row { a: int64, b: int64 }
table t = Row
def f(x: int64) -> int64 { return x }

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
                %10 = yzl.call @in(%arg2, %9) : (!yzl.unresolved, !yzl.unresolved) -> !yzl.unresolved {callee_source = "builtin"}
                yzl.yield %10 : !yzl.unresolved
              }
              %5 = yzl.drop %4 ["rating"]
              %6 = yzl.distinct %5
              yzl.output %6
            }
        "#]]
        .assert_eq(&lowered(
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
