use melior::ir::{
    Attribute, Block, BlockLike, BlockRef, Location, Region, RegionLike, Value,
    attribute::{ArrayAttribute, FlatSymbolRefAttribute, IntegerAttribute, StringAttribute},
};
use yuzu_ast::{AstNode, ast};
use yuzu_mlir::ods::yzl;

use crate::lower_ast_to_yzl::{AstToYzl, Locals, ident_text};
use melior::ir::r#type::IntegerType;
use yuzu_mlir::attributes::JoinKind;
use yuzu_mlir::ext::OperationExt;
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

                let value = block
                    .append_operation(
                        yzl::from(
                            self.context,
                            types::query(self.context),
                            FlatSymbolRefAttribute::new(self.context, &source),
                            loc,
                        )
                        .into(),
                    )
                    .first_result();
                match ident_text(from.alias()) {
                    Some(alias) => block
                        .append_operation(
                            yzl::alias(
                                self.context,
                                types::query(self.context),
                                value,
                                StringAttribute::new(self.context, &alias),
                                loc,
                            )
                            .into(),
                        )
                        .first_result(),
                    None => value,
                }
            }
            ast::Rel::WhereExpr(stage) => {
                let input = self.convert_input(block, stage, "`where`", stage.input());
                let region = Region::new();
                let body = region.append_block(Block::new(&[]));
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
                block
                    .append_operation(
                        yzl::select(
                            self.context,
                            types::query(self.context),
                            input,
                            region,
                            names,
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
                block
                    .append_operation(
                        yzl::extend(
                            self.context,
                            types::query(self.context),
                            input,
                            region,
                            names,
                            loc,
                        )
                        .into(),
                    )
                    .first_result()
            }
            ast::Rel::AggregateExpr(stage) => {
                let input = self.convert_input(block, stage, "`aggregate`", stage.input());
                let mut group_by: Vec<Attribute> = Vec::new();
                for item in stage.group_by().into_iter().flat_map(|group| group.items()) {
                    match ident_text(item.column()) {
                        Some(name) => {
                            group_by.push(StringAttribute::new(self.context, &name).into());
                        }
                        None => self.error(&item, "group by item is missing its column"),
                    }
                }

                let items = stage
                    .items()
                    .map(|item| (item.alias(), item.expr(), item.syntax().text_range()))
                    .collect();
                let (names, region) = self.convert_items(items, "aggregate item", loc);
                block
                    .append_operation(
                        yzl::aggregate(
                            self.context,
                            types::query(self.context),
                            input,
                            region,
                            ArrayAttribute::new(self.context, &group_by),
                            names,
                            loc,
                        )
                        .into(),
                    )
                    .first_result()
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

                    // `b.id as other` has to carry its qualifier through:
                    // resolution matches the reference against qualified
                    // columns, so dropping it renames whichever column
                    // happened to come first.
                    let old = match ident_text(item.qualifier()) {
                        Some(qualifier) => format!("{qualifier}.{old}"),
                        None => old,
                    };

                    from.push(StringAttribute::new(self.context, &old).into());
                    to.push(StringAttribute::new(self.context, &new).into());
                }

                block
                    .append_operation(
                        yzl::rename(
                            self.context,
                            types::query(self.context),
                            input,
                            ArrayAttribute::new(self.context, &from),
                            ArrayAttribute::new(self.context, &to),
                            loc,
                        )
                        .into(),
                    )
                    .first_result()
            }
            ast::Rel::AliasExpr(stage) => {
                let input = self.convert_input(block, stage, "`alias`", stage.input());
                let Some(alias) = ident_text(stage.alias()) else {
                    self.error(stage, "`alias` is missing its name");
                    return input;
                };

                block
                    .append_operation(
                        yzl::alias(
                            self.context,
                            types::query(self.context),
                            input,
                            StringAttribute::new(self.context, &alias),
                            loc,
                        )
                        .into(),
                    )
                    .first_result()
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

                let on = Region::new();
                if let Some(condition) = stage.on().and_then(|on| on.condition()) {
                    let body = on.append_block(Block::new(&[]));
                    let value = self.convert_expr(body, &Locals::new(), &condition);
                    body.append_operation(yzl::r#yield(self.context, &[value], loc).into());
                }

                let mut builder = yzl::JoinOperationBuilder::new(self.context, loc)
                    .result(types::query(self.context))
                    .lhs(lhs)
                    .kind(StringAttribute::new(self.context, kind.as_str()))
                    .rhs(FlatSymbolRefAttribute::new(self.context, &rhs))
                    .on(on);
                if let Some(alias) = ident_text(stage.alias()) {
                    builder = builder.rhs_alias(StringAttribute::new(self.context, &alias));
                }

                if let Some(using) = stage.using() {
                    let columns: Vec<Attribute> = using
                        .columns()
                        .filter_map(|column| column.text())
                        .map(|name| StringAttribute::new(self.context, &name).into())
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
                block
                    .append_operation(
                        yzl::set(
                            self.context,
                            types::query(self.context),
                            input,
                            region,
                            names,
                            loc,
                        )
                        .into(),
                    )
                    .first_result()
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
                let columns: Vec<Attribute> = stage
                    .columns()
                    .filter_map(|column| column.text())
                    .map(|name| StringAttribute::new(self.context, &name).into())
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
                block
                    .append_operation(
                        yzl::from(
                            self.context,
                            types::query(self.context),
                            FlatSymbolRefAttribute::new(self.context, &name),
                            loc,
                        )
                        .into(),
                    )
                    .first_result()
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
    ) -> (ArrayAttribute<'c>, Region<'c>) {
        let region = Region::new();
        let body = region.append_block(Block::new(&[]));
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
        (ArrayAttribute::new(self.context, &names), region)
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
              %0 = yzl.from @t
              %1 = yzl.where %0 {
                %5 = yzl.name "a" : !yzl.var
                %6 = yz.constant_int 10
                %7 = yz.cmp "gt", %5, %6 : !yzl.var, !yz.int64 -> !yzl.var
                yzl.yield %7 : !yzl.var
              }
              %2 = yzl.extend %1 as ["e"] {
                %5 = yzl.name "a" : !yzl.var
                %6 = yzl.call @f(%5) : (!yzl.var) -> !yzl.var
                %7 = yzl.name "b" : !yzl.var
                %8 = yz.add %6, %7 : !yzl.var, !yzl.var -> !yzl.var
                yzl.yield %8 : !yzl.var
              }
              %3 = yzl.aggregate %2 group_by ["b"] as ["s"] {
                %5 = yzl.name "e" : !yzl.var
                %6 = yzl.call @sum(%5) : (!yzl.var) -> !yzl.var
                yzl.yield %6 : !yzl.var
              }
              %4 = yzl.limit %3, 10
              yzl.output %4
            }
        "#]]
        .assert_eq(&converted(
            r#"
struct Row { a: int64, b: int64 }
table t = Row

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
              %0 = yzl.from @employees
              %1 = yzl.join "inner", %0, @departments as "d" {
                %6 = yzl.name "dept_id" : !yzl.var
                %7 = yzl.name "d.id" : !yzl.var
                %8 = yz.cmp "eq", %6, %7 : !yzl.var, !yzl.var -> !yzl.var
                yzl.yield %8 : !yzl.var
              }
              %2 = yzl.set %1 as ["level"] {
                %6 = yzl.name "level" : !yzl.var
                %7 = yz.constant_int 1
                %8 = yz.add %6, %7 : !yzl.var, !yz.int64 -> !yzl.var
                yzl.yield %8 : !yzl.var
              }
              %3 = yzl.where %2 {
                %6 = yzl.name "level" : !yzl.var
                %7 = yz.constant_int 1
                %8 = yz.constant_int 3
                %9 = yzl.list[%7, %8] : (!yz.int64, !yz.int64) -> !yzl.var
                %10 = yzl.call @in(%6, %9) : (!yzl.var, !yzl.var) -> !yzl.var
                yzl.yield %10 : !yzl.var
              }
              %4 = yzl.drop %3 ["rating"]
              %5 = yzl.distinct %4
              yzl.output %5
            }
        "#]]
        .assert_eq(&converted(
            r#"
from employees
|> inner join departments as d on dept_id == d.id
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
              yzl.let @base {
                %4 = yzl.from @t
                %5 = yzl.where %4 {
                  %6 = yzl.name "active" : !yzl.var
                  yzl.yield %6 : !yzl.var
                }
                yzl.yield %5 : !yzl.query
              }
              %0 = yzl.from @base
              %1 = yzl.rename %0 from ["a"] to ["renamed"]
              %2 = yzl.alias %1 as "q"
              %3 = yzl.select %2 as ["out"] {
                %4 = yzl.name "q.renamed" : !yzl.var
                yzl.yield %4 : !yzl.var
              }
              yzl.output %3
            }
        "#]]
        .assert_eq(&converted(
            r#"
let base = from t |> where active

from base
|> rename a as renamed
|> as q
|> select q.renamed as out
"#,
        ));
    }
}
