//! The stages: each becomes the `yzr` op that means the same thing, and
//! the ones yzr has no op for become the ops it does have. A stage the
//! lowering does not carry is reported rather than skipped.
use melior::ir::attribute::{DenseI64ArrayAttribute, StringAttribute};
use melior::ir::operation::{OperationLike, OperationRef};
use melior::ir::{Block, BlockLike, BlockRef, Location, Region, RegionLike, Type, Value};
use yuzu_mlir::SymbolTable;
use yuzu_mlir::ext::{ArrayAttributeExt, BlockExt, OperationCast, OperationExt, ValueExt};
use yuzu_mlir::ods::{yz, yzr};
use yuzu_mlir::ops::yzl as yzl_ops;
use yuzu_mlir::ops::yzl::YzlOp;
use yuzu_mlir::types;

use crate::lower_yzl_to_yzr::{Schema, Yielded, YzlToYzr, op_name, struct_fields};

impl<'c, 'a> YzlToYzr<'c, 'a, '_> {
    pub(super) fn lower_op(
        &mut self,
        op: OperationRef<'c, '_>,
        target: BlockRef<'c, 'a>,
        source: &SymbolTable<'c, '_>,
        symbols: &mut SymbolTable<'c, '_>,
    ) {
        match op.as_yzl() {
            Some(YzlOp::Struct(item)) => self.lower_struct(symbols, &item),
            Some(YzlOp::From(from)) => self.lower_from(op, target, source, symbols, &from),
            Some(YzlOp::Let(binding)) => self.lower_let(op, target, source, symbols, &binding),
            Some(YzlOp::Where(stage)) => self.lower_where(op, target, &stage),
            Some(YzlOp::Select(stage)) => self.lower_select(op, target, symbols, &stage),
            Some(YzlOp::Extend(stage)) => self.lower_extend(op, target, symbols, &stage),
            Some(YzlOp::Aggregate(stage)) => self.lower_aggregate(op, target, symbols, &stage),
            Some(YzlOp::Join(stage)) => self.lower_join(op, target, source, symbols, &stage),
            Some(YzlOp::Limit(stage)) => self.lower_limit(op, target, &stage),
            Some(YzlOp::Alias(_)) => self.lower_alias(op),
            Some(YzlOp::Distinct(_)) => self.lower_distinct(op, target, symbols),
            Some(YzlOp::Drop(stage)) => self.lower_drop(op, target, symbols, &stage),
            Some(YzlOp::Set(stage)) => self.lower_set(op, target, symbols, &stage),
            Some(YzlOp::Rename(stage)) => self.lower_rename(op, target, symbols, &stage),
            Some(YzlOp::Output(_)) => self.lower_output(op, target),
            // A table declaration says nothing yzr needs: `yzr.table` names
            // the relation and carries its row as the result type.
            Some(YzlOp::Table(_)) => {}
            // Expansion removes these once every call is gone, so one
            // reaching here means expansion did not finish — the reason is
            // already reported, and this says which declaration outlived it.
            Some(YzlOp::Fn(_) | YzlOp::Trait(_) | YzlOp::Impl(_)) => self.error(
                op,
                format!("`{}` was not expanded before lowering", op_name(op)),
            ),
            Some(YzlOp::Missing(_)) => self.error(op, "this part of the query is missing"),
            // Declarations yzr does not need, and the terminators a region
            // owns rather than the module.
            Some(YzlOp::Call(_) | YzlOp::List(_) | YzlOp::Yield(_) | YzlOp::Return(_)) | None => {}
        }
    }

    fn lower_output(&mut self, op: OperationRef<'c, '_>, target: BlockRef<'c, 'a>) {
        let Some((query, _)) = self.input_stage(op) else {
            return;
        };

        target.append_operation(yzr::output(self.context, query, op.location()).into());
    }

    fn lower_struct(
        &mut self,
        symbols: &mut SymbolTable<'c, '_>,
        item: &yzl_ops::StructOp<'c, '_>,
    ) {
        let fields = struct_fields(item);
        self.declare_struct(item.sym_name().value(), &fields, symbols);
    }

    fn lower_from(
        &mut self,
        op: OperationRef<'c, '_>,
        target: BlockRef<'c, 'a>,
        source: &SymbolTable<'c, '_>,
        symbols: &mut SymbolTable<'c, '_>,
        from: &yzl_ops::FromOp<'c, '_>,
    ) {
        let relation = from.source().value();
        let Some((rows, schema)) =
            self.relation_input(relation, source, target, symbols, op.location())
        else {
            self.error(op, format!("`{relation}` has no row shape to scan"));
            return;
        };

        self.record_stage(op, rows, schema);
    }

    fn lower_let(
        &mut self,
        op: OperationRef<'c, '_>,
        target: BlockRef<'c, 'a>,
        source: &SymbolTable<'c, '_>,
        symbols: &mut SymbolTable<'c, '_>,
        binding: &yzl_ops::LetOp<'c, '_>,
    ) {
        let Some(block) = binding.body().first_block() else {
            self.error(op, "`let` has no body to bind");
            return;
        };

        for inner in block.operations() {
            self.lower_op(inner, target, source, symbols);
        }

        let bound = block
            .last_operation()
            .and_then(|yielded| yielded.try_first_operand())
            .and_then(|value| self.stages.get(&value.id()).cloned());

        match bound {
            Some(rows) => {
                self.bindings.insert(binding.sym_name().value(), rows);
            }
            None => self.error(op, "only a query can be bound by `let`"),
        }
    }

    fn lower_where(
        &mut self,
        op: OperationRef<'c, '_>,
        target: BlockRef<'c, 'a>,
        stage: &yzl_ops::WhereOp<'c, '_>,
    ) {
        let Some((input, schema)) = self.input_stage(op) else {
            return;
        };

        let (region, _) = self.lower_region(stage.body(), &schema, op.location(), Yielded::Body);
        let filtered =
            target.append_operation(yzr::filter(self.context, input, region, op.location()).into());

        self.record_stage(op, filtered.first_result(), schema);
    }

    fn lower_select(
        &mut self,
        op: OperationRef<'c, '_>,
        target: BlockRef<'c, 'a>,
        symbols: &mut SymbolTable<'c, '_>,
        stage: &yzl_ops::SelectOp<'c, '_>,
    ) {
        let Some((input, schema)) = self.input_stage(op) else {
            return;
        };

        let (region, yielded) =
            self.lower_region(stage.body(), &schema, op.location(), Yielded::Body);
        let produced = self.named_row(stage.names().strings(), yielded);
        let row = self.row_type(&produced, symbols);
        let projected = target
            .append_operation(yzr::project(self.context, row, input, region, op.location()).into());

        self.record_stage(op, projected.first_result(), produced);
    }

    fn lower_extend(
        &mut self,
        op: OperationRef<'c, '_>,
        target: BlockRef<'c, 'a>,
        symbols: &mut SymbolTable<'c, '_>,
        stage: &yzl_ops::ExtendOp<'c, '_>,
    ) {
        let Some((input, mut schema)) = self.input_stage(op) else {
            return;
        };

        let (region, yielded) =
            self.lower_region(stage.body(), &schema, op.location(), Yielded::Body);
        schema.extend(self.named_row(stage.names().strings(), yielded));
        let row = self.row_type(&schema, symbols);
        let extended = target
            .append_operation(yzr::extend(self.context, row, input, region, op.location()).into());

        self.record_stage(op, extended.first_result(), schema);
    }

    fn lower_aggregate(
        &mut self,
        op: OperationRef<'c, '_>,
        target: BlockRef<'c, 'a>,
        symbols: &mut SymbolTable<'c, '_>,
        stage: &yzl_ops::AggregateOp<'c, '_>,
    ) {
        let Some((input, schema)) = self.input_stage(op) else {
            return;
        };

        let keys = crate::infer_types::indices(stage.key_cols());
        let (region, yielded) =
            self.lower_region(stage.body(), &schema, op.location(), Yielded::Body);
        let mut produced: Schema<'c> = keys
            .iter()
            .zip(stage.group_by().strings())
            .filter_map(|(&index, name)| schema.get(index).map(|&(_, ty)| (name, ty)))
            .collect();
        produced.extend(self.named_row(stage.names().strings(), yielded));

        let row = self.row_type(&produced, symbols);
        let indices: Vec<i64> = keys.iter().map(|&index| index as i64).collect();
        let grouped = target.append_operation(
            yzr::aggregate(
                self.context,
                row,
                input,
                region,
                DenseI64ArrayAttribute::new(self.context, &indices).into(),
                op.location(),
            )
            .into(),
        );

        self.record_stage(op, grouped.first_result(), produced);
    }

    fn lower_join(
        &mut self,
        op: OperationRef<'c, '_>,
        target: BlockRef<'c, 'a>,
        source: &SymbolTable<'c, '_>,
        symbols: &mut SymbolTable<'c, '_>,
        stage: &yzl_ops::JoinOp<'c, '_>,
    ) {
        let Some((lhs, mut schema)) = self.input_stage(op) else {
            return;
        };

        // The one stage that has to conjure an input: yzl names the
        // right side, yzr joins two relations.
        let relation = stage.rhs().value();
        let Some((rows, right)) =
            self.relation_input(relation, source, target, symbols, op.location())
        else {
            self.error(op, format!("`{relation}` has no row shape to scan"));
            return;
        };

        // Both sides carry through, and the `on` region's names were
        // resolved against exactly this concatenation — a qualifier
        // only ever chose a column, so it is spent by now.
        let left_width = schema.len();
        schema.extend(right.iter().copied());

        let region = match stage.using_columns() {
            Some(columns) => self.join_keys(op, &columns.strings(), left_width, &schema),
            None => {
                self.lower_region(stage.on(), &schema, op.location(), Yielded::Body)
                    .0
            }
        };

        let row = self.row_type(&schema, symbols);
        let joined = target.append_operation(
            yzr::join(
                self.context,
                row,
                lhs,
                rows,
                region,
                StringAttribute::new(self.context, stage.kind().as_str()),
                op.location(),
            )
            .into(),
        );

        self.record_stage(op, joined.first_result(), schema);
    }

    fn lower_limit(
        &mut self,
        op: OperationRef<'c, '_>,
        target: BlockRef<'c, 'a>,
        stage: &yzl_ops::LimitOp<'c, '_>,
    ) {
        let Some((input, schema)) = self.input_stage(op) else {
            return;
        };

        let mut builder = yzr::LimitOperationBuilder::new(self.context, op.location())
            .input(input)
            .count(stage.count());
        if let Some(offset) = stage.offset() {
            builder = builder.offset(offset);
        }

        let limited = target.append_operation(builder.build().into());
        self.record_stage(op, limited.first_result(), schema);
    }

    fn lower_alias(&mut self, op: OperationRef<'c, '_>) {
        let Some((input, schema)) = self.input_stage(op) else {
            return;
        };

        self.record_stage(op, input, schema);
    }

    fn lower_distinct(
        &mut self,
        op: OperationRef<'c, '_>,
        target: BlockRef<'c, 'a>,
        symbols: &mut SymbolTable<'c, '_>,
    ) {
        let Some((input, schema)) = self.input_stage(op) else {
            return;
        };

        let keys: Vec<i64> = (0..schema.len() as i64).collect();
        let region = self.column_region(&schema, &[], op.location());
        let row = self.row_type(&schema, symbols);
        let grouped = target.append_operation(
            yzr::aggregate(
                self.context,
                row,
                input,
                region,
                DenseI64ArrayAttribute::new(self.context, &keys).into(),
                op.location(),
            )
            .into(),
        );

        self.record_stage(op, grouped.first_result(), schema);
    }

    fn lower_drop(
        &mut self,
        op: OperationRef<'c, '_>,
        target: BlockRef<'c, 'a>,
        symbols: &mut SymbolTable<'c, '_>,
        stage: &yzl_ops::DropOp<'c, '_>,
    ) {
        let Some((input, schema)) = self.input_stage(op) else {
            return;
        };

        let Some(kept) = self.kept_columns(op, &stage.columns().strings(), &schema) else {
            return;
        };

        let region = self.column_region(&schema, &kept, op.location());
        let produced: Schema<'c> = kept.iter().map(|&index| schema[index]).collect();
        let row = self.row_type(&produced, symbols);
        let projected = target
            .append_operation(yzr::project(self.context, row, input, region, op.location()).into());

        self.record_stage(op, projected.first_result(), produced);
    }

    fn lower_set(
        &mut self,
        op: OperationRef<'c, '_>,
        target: BlockRef<'c, 'a>,
        symbols: &mut SymbolTable<'c, '_>,
        stage: &yzl_ops::SetOp<'c, '_>,
    ) {
        let Some((input, mut schema)) = self.input_stage(op) else {
            return;
        };

        // The body computes replacements, not a new row: every column
        // it does not name carries through in place.
        let columns = crate::infer_types::indices(stage.set_cols());
        let (region, yielded) =
            self.lower_region(stage.body(), &schema, op.location(), Yielded::Row(&columns));

        for (column, ty) in schema.iter_mut().zip(&yielded) {
            column.1 = *ty;
        }

        let row = self.row_type(&schema, symbols);
        let projected = target
            .append_operation(yzr::project(self.context, row, input, region, op.location()).into());

        self.record_stage(op, projected.first_result(), schema);
    }

    fn lower_rename(
        &mut self,
        op: OperationRef<'c, '_>,
        target: BlockRef<'c, 'a>,
        symbols: &mut SymbolTable<'c, '_>,
        stage: &yzl_ops::RenameOp<'c, '_>,
    ) {
        let Some((input, mut schema)) = self.input_stage(op) else {
            return;
        };

        let columns = crate::infer_types::indices(stage.rename_cols());
        for (&index, name) in columns.iter().zip(stage.to().strings()) {
            match schema.get_mut(index) {
                Some(column) => column.0 = name,
                None => {
                    self.error(op, format!("column {index} is not in the row"));
                    return;
                }
            }
        }

        let all: Vec<usize> = (0..schema.len()).collect();
        let region = self.column_region(&schema, &all, op.location());
        let row = self.row_type(&schema, symbols);
        let projected = target
            .append_operation(yzr::project(self.context, row, input, region, op.location()).into());

        self.record_stage(op, projected.first_result(), schema);
    }

    /// `using [a, b]` is sugar: yzr has only an on-region, so the columns
    /// become the equality the join was asking for.
    fn join_keys(
        &mut self,
        op: OperationRef<'c, '_>,
        columns: &[&str],
        left_width: usize,
        schema: &Schema<'c>,
    ) -> Region<'c> {
        let location = op.location();
        let region = Region::new();
        let arguments: Vec<(Type<'c>, Location<'c>)> = schema
            .iter()
            .map(|(_, column)| (*column, location))
            .collect();
        let body = region.append_block(Block::new(&arguments));

        let mut condition: Option<Value<'c, '_>> = None;
        for column in columns {
            let left = schema[..left_width]
                .iter()
                .position(|(name, _)| name == column);
            let right = schema[left_width..]
                .iter()
                .position(|(name, _)| name == column)
                .map(|index| index + left_width);
            let (Some(left), Some(right)) = (left, right) else {
                self.error(op, format!("`{column}` is not present in both relations"));
                continue;
            };

            let equal = body.append_operation(
                yz::cmp(
                    self.context,
                    types::boolean(self.context),
                    body.argument(left)
                        .expect("the left column is in range")
                        .into(),
                    body.argument(right)
                        .expect("the right column is in range")
                        .into(),
                    StringAttribute::new(self.context, "eq"),
                    location,
                )
                .into(),
            );

            condition = Some(match condition {
                Some(previous) => body
                    .append_operation(
                        yz::and(
                            self.context,
                            types::boolean(self.context),
                            previous,
                            equal.first_result(),
                            location,
                        )
                        .into(),
                    )
                    .first_result(),
                None => equal.first_result(),
            });
        }

        let yielded: Vec<Value<'c, '_>> = condition.into_iter().collect();
        body.append_operation(yzr::r#yield(self.context, &yielded, location).into());

        region
    }

    /// The columns that survive a `drop`, in order. Resolution removes the
    /// first column each name matches, so dropping one name twice drops two
    /// columns — the lowering has to agree with it exactly.
    fn kept_columns(
        &self,
        op: OperationRef<'c, '_>,
        columns: &[&str],
        schema: &Schema<'c>,
    ) -> Option<Vec<usize>> {
        let mut dropped: Vec<usize> = Vec::new();
        for column in columns {
            let found = schema
                .iter()
                .enumerate()
                .find(|(index, (name, _))| name == column && !dropped.contains(index))
                .map(|(index, _)| index);

            match found {
                Some(index) => dropped.push(index),
                None => {
                    self.error(op, format!("`{column}` is not in the row"));
                    return None;
                }
            }
        }

        Some(
            (0..schema.len())
                .filter(|index| !dropped.contains(index))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::test_support::check_lowered;

    /// The stages that name columns declare the row they produce, and a
    /// shape nobody declared is interned under a name of its own.
    #[test]
    fn named_stages_declare_the_row_they_produce() {
        check_lowered(
            r#"
struct Row { a: int64, b: int64 }
table t = Row

from t
|> extend a + b as e
|> select a as x, e as y
|> limit 5 offset 1
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a", "b"] : [!yz.int64, !yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  yz.struct @row ["a", "b", "e"] : [!yz.int64, !yz.int64, !yz.int64]
                  %1 = yzr.extend %0 {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64):
                    %4 = yz.add %arg0, %arg1 : !yz.int64, !yz.int64 -> !yz.int64
                    yzr.yield %4 : !yz.int64
                  } : !yz.struct<@Row> -> !yz.struct<@row>
                  yz.struct @row_0 ["x", "y"] : [!yz.int64, !yz.int64]
                  %2 = yzr.project %1 {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64, %arg2: !yz.int64):
                    yzr.yield %arg0, %arg2 : !yz.int64, !yz.int64
                  } : !yz.struct<@row> -> !yz.struct<@row_0>
                  %3 = yzr.limit %2, 5 offset 1 : !yz.struct<@row_0>
                  yzr.output %3 : !yz.struct<@row_0>
                }
            "#]],
        );
    }

    /// The one stage with two inputs: the named right side becomes a scan,
    /// and the `on` region sees both rows' columns as one block.
    #[test]
    fn join_materialises_its_right_side() {
        check_lowered(
            r#"
struct Row { id: int64, dept_id: int64 }
table t = Row
struct Dept { key: int64, name: str }
table depts = Dept

from t
|> left join depts as d on dept_id == d.key
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["id", "dept_id"] : [!yz.int64, !yz.int64]
                  yz.struct @Dept ["key", "name"] : [!yz.int64, !yz.str]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  %1 = yzr.table @depts : !yz.struct<@Dept>
                  yz.struct @row ["id", "dept_id", "key", "name"] : [!yz.int64, !yz.int64, !yz.int64, !yz.str]
                  %2 = yzr.join "left", %0, %1 {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64, %arg2: !yz.int64, %arg3: !yz.str):
                    %3 = yz.cmp "eq", %arg1, %arg2 : !yz.int64, !yz.int64 -> !yz.bool
                    yzr.yield %3 : !yz.bool
                  } : !yz.struct<@Row>, !yz.struct<@Dept> -> !yz.struct<@row>
                  yzr.output %2 : !yz.struct<@row>
                }
            "#]],
        );
    }

    /// `using` is sugar for the equality it asks for; both sides' columns
    /// carry through, as they do for `on`.
    #[test]
    fn using_becomes_the_equality_it_means() {
        check_lowered(
            r#"
struct Row { id: int64, tag: str, part: int64 }
table t = Row
struct Other { id: int64, part: int64, extra: int64 }
table u = Other

from t
|> inner join u using (id, part)
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["id", "tag", "part"] : [!yz.int64, !yz.str, !yz.int64]
                  yz.struct @Other ["id", "part", "extra"] : [!yz.int64, !yz.int64, !yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  %1 = yzr.table @u : !yz.struct<@Other>
                  yz.struct @row ["id", "tag", "part", "id", "part", "extra"] : [!yz.int64, !yz.str, !yz.int64, !yz.int64, !yz.int64, !yz.int64]
                  %2 = yzr.join "inner", %0, %1 {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.str, %arg2: !yz.int64, %arg3: !yz.int64, %arg4: !yz.int64, %arg5: !yz.int64):
                    %3 = yz.cmp "eq", %arg0, %arg3 : !yz.int64, !yz.int64 -> !yz.bool
                    %4 = yz.cmp "eq", %arg2, %arg4 : !yz.int64, !yz.int64 -> !yz.bool
                    %5 = yz.and %3, %4 : !yz.bool, !yz.bool -> !yz.bool
                    yzr.yield %5 : !yz.bool
                  } : !yz.struct<@Row>, !yz.struct<@Other> -> !yz.struct<@row>
                  yzr.output %2 : !yz.struct<@row>
                }
            "#]],
        );
    }

    /// `alias` only qualifies names, and resolution has already used them
    /// to choose columns: nothing is left for yzr to represent.
    #[test]
    fn alias_leaves_no_trace() {
        check_lowered(
            r#"
struct Row { a: int64, b: int64 }
table t = Row

from t
|> as r
|> where r.a > 1
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a", "b"] : [!yz.int64, !yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  %1 = yzr.filter %0 : !yz.struct<@Row> {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64):
                    %2 = yz.constant_int 1
                    %3 = yz.cmp "gt", %arg0, %2 : !yz.int64, !yz.int64 -> !yz.bool
                    yzr.yield %3 : !yz.bool
                  }
                  yzr.output %1 : !yz.struct<@Row>
                }
            "#]],
        );
    }

    /// yzr has no distinct: it is a group keyed on every column, measuring
    /// nothing.
    #[test]
    fn distinct_groups_on_every_column() {
        check_lowered(
            r#"
struct Row { a: int64, b: str }
table t = Row

from t
|> distinct
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a", "b"] : [!yz.int64, !yz.str]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  %1 = yzr.aggregate %0 keys [0, 1] {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.str):
                    yzr.yield
                  } : !yz.struct<@Row> -> !yz.struct<@Row>
                  yzr.output %1 : !yz.struct<@Row>
                }
            "#]],
        );
    }

    /// `drop` names what leaves; the projection yields what stays.
    #[test]
    fn drop_projects_the_columns_that_stay() {
        check_lowered(
            r#"
struct Row { a: int64, b: str, c: int64 }
table t = Row

from t
|> drop b
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a", "b", "c"] : [!yz.int64, !yz.str, !yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  yz.struct @row ["a", "c"] : [!yz.int64, !yz.int64]
                  %1 = yzr.project %0 {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.str, %arg2: !yz.int64):
                    yzr.yield %arg0, %arg2 : !yz.int64, !yz.int64
                  } : !yz.struct<@Row> -> !yz.struct<@row>
                  yzr.output %1 : !yz.struct<@row>
                }
            "#]],
        );
    }

    /// `set` replaces columns in place, so the projection has to yield the
    /// columns it did not name as well.
    #[test]
    fn set_yields_the_untouched_columns_too() {
        check_lowered(
            r#"
struct Row { a: int64, b: int64, c: int64 }
table t = Row

from t
|> set b = a + 1
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a", "b", "c"] : [!yz.int64, !yz.int64, !yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  %1 = yzr.project %0 {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64, %arg2: !yz.int64):
                    %2 = yz.constant_int 1
                    %3 = yz.add %arg0, %2 : !yz.int64, !yz.int64 -> !yz.int64
                    yzr.yield %arg0, %3, %arg2 : !yz.int64, !yz.int64, !yz.int64
                  } : !yz.struct<@Row> -> !yz.struct<@Row>
                  yzr.output %1 : !yz.struct<@Row>
                }
            "#]],
        );
    }

    /// A binding is a name for rows that already exist: two uses share the
    /// one scan rather than each producing their own.
    #[test]
    fn a_binding_is_reused_not_rescanned() {
        check_lowered(
            r#"
struct Row { a: int64, b: int64 }
table t = Row

let big = from t |> where a > 10

from big
|> inner join big using (a)
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a", "b"] : [!yz.int64, !yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  %1 = yzr.filter %0 : !yz.struct<@Row> {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64):
                    %3 = yz.constant_int 10
                    %4 = yz.cmp "gt", %arg0, %3 : !yz.int64, !yz.int64 -> !yz.bool
                    yzr.yield %4 : !yz.bool
                  }
                  yz.struct @row ["a", "b", "a", "b"] : [!yz.int64, !yz.int64, !yz.int64, !yz.int64]
                  %2 = yzr.join "inner", %1, %1 {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64, %arg2: !yz.int64, %arg3: !yz.int64):
                    %3 = yz.cmp "eq", %arg0, %arg2 : !yz.int64, !yz.int64 -> !yz.bool
                    yzr.yield %3 : !yz.bool
                  } : !yz.struct<@Row>, !yz.struct<@Row> -> !yz.struct<@row>
                  yzr.output %2 : !yz.struct<@row>
                }
            "#]],
        );
    }

    /// Binding something that is not a query is not carried yet, and a gap
    /// is an error rather than a silently dropped binding.
    #[test]
    fn reports_a_binding_that_is_not_a_query() {
        check_lowered(
            r#"
struct Row { a: int64 }
table t = Row

let n = 1 + 2

from t
"#,
            expect![[r#"
                error: only a query can be bound by `let`
                 --> test.yz:5:1
                  |
                5 | let n = 1 + 2
                  | ^
            "#]],
        );
    }

    /// A rename moves names, not values, but yzr rows are typed by their
    /// struct — so the new names need a projection to live on.
    #[test]
    fn rename_projects_under_the_new_names() {
        check_lowered(
            r#"
struct Row { a: int64, b: str }
table t = Row

from t
|> rename a as x, b as y
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a", "b"] : [!yz.int64, !yz.str]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  yz.struct @row ["x", "y"] : [!yz.int64, !yz.str]
                  %1 = yzr.project %0 {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.str):
                    yzr.yield %arg0, %arg1 : !yz.int64, !yz.str
                  } : !yz.struct<@Row> -> !yz.struct<@row>
                  yzr.output %1 : !yz.struct<@row>
                }
            "#]],
        );
    }

    /// The stamp resolution left says which column each name applies to, so
    /// a qualified rename after a join renames one side rather than guessing.
    #[test]
    fn rename_follows_the_stamped_column() {
        check_lowered(
            r#"
struct Row { id: int64 }
table l = Row
struct Other { id: int64 }
table r = Other

from l
|> as a
|> inner join r as b on a.id == b.id
|> rename b.id as other
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["id"] : [!yz.int64]
                  yz.struct @Other ["id"] : [!yz.int64]
                  %0 = yzr.table @l : !yz.struct<@Row>
                  %1 = yzr.table @r : !yz.struct<@Row>
                  yz.struct @row ["id", "id"] : [!yz.int64, !yz.int64]
                  %2 = yzr.join "inner", %0, %1 {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64):
                    %4 = yz.cmp "eq", %arg0, %arg1 : !yz.int64, !yz.int64 -> !yz.bool
                    yzr.yield %4 : !yz.bool
                  } : !yz.struct<@Row>, !yz.struct<@Row> -> !yz.struct<@row>
                  yz.struct @row_0 ["id", "other"] : [!yz.int64, !yz.int64]
                  %3 = yzr.project %2 {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64):
                    yzr.yield %arg0, %arg1 : !yz.int64, !yz.int64
                  } : !yz.struct<@row> -> !yz.struct<@row_0>
                  yzr.output %3 : !yz.struct<@row_0>
                }
            "#]],
        );
    }

    /// A stage the lowering does not carry yet is an error, not a silent gap.
    #[test]
    fn reports_a_stage_that_is_not_lowered() {
        check_lowered(
            r#"
struct Row { a: int64 }
table t = Row

from t
|> rename a as b
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a"] : [!yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  yz.struct @row ["b"] : [!yz.int64]
                  %1 = yzr.project %0 {
                  ^bb0(%arg0: !yz.int64):
                    yzr.yield %arg0 : !yz.int64
                  } : !yz.struct<@Row> -> !yz.struct<@row>
                  yzr.output %1 : !yz.struct<@row>
                }
            "#]],
        );
    }
}
