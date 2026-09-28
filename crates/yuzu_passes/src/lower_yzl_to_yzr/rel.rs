use melior::IrRewriter;
use melior::ir::attribute::{DenseI64ArrayAttribute, StringAttribute, TypeAttribute};
use melior::ir::operation::{OperationLike, OperationRef};
use melior::ir::r#type::FunctionType;
use melior::ir::{Block, BlockLike, Region, RegionLike, Value};
use yuzu_mlir::SymbolTable;
use yuzu_mlir::ir::attribute::array::ArrayAttributeExt;
use yuzu_mlir::ir::block::BlockExt;
use yuzu_mlir::ir::operation::{OperationCast, OperationExt};
use yuzu_mlir::ir::value::ValueExt;
use yuzu_mlir::ods::{yz, yzr};
use yuzu_mlir::ops::yzl::{
    AggregateOp, ConstOp, DropOp, ExtendOp, FnOp, FromOp, JoinOp, LimitOp, RenameOp, SelectOp,
    SetOp, StructOp, WhereOp, YzlOp,
};
use yuzu_mlir::types::BoolType;

use crate::lower_yzl_to_yzr::row::struct_declaration;
use crate::lower_yzl_to_yzr::{Row, Yielded, YzlToYzr, op_name, struct_fields};
use crate::operators::Operator;

impl<'c, 'a> YzlToYzr<'c, 'a> {
    pub(super) fn convert_op(
        &mut self,
        op: OperationRef<'c, '_>,
        symbols: &mut SymbolTable<'c, 'a>,
    ) {
        match op.as_yzl() {
            Some(YzlOp::Struct(item)) => self.convert_struct(op, symbols, &item),
            Some(YzlOp::From(from)) => self.convert_from(op, symbols, &from),
            Some(YzlOp::Const(binding)) => self.convert_let(op, symbols, &binding),
            Some(YzlOp::Where(stage)) => self.convert_where(op, &stage),
            Some(YzlOp::Select(stage)) => self.convert_select(op, symbols, &stage),
            Some(YzlOp::Extend(stage)) => self.convert_extend(op, symbols, &stage),
            Some(YzlOp::Aggregate(stage)) => self.convert_aggregate(op, symbols, &stage),
            Some(YzlOp::Join(stage)) => self.convert_join(op, symbols, &stage),
            Some(YzlOp::Limit(stage)) => self.convert_limit(op, &stage),
            Some(YzlOp::Alias(_)) => self.convert_alias(op),
            Some(YzlOp::Distinct(_)) => self.convert_distinct(op, symbols),
            Some(YzlOp::Drop(stage)) => self.convert_drop(op, symbols, &stage),
            Some(YzlOp::Set(stage)) => self.convert_set(op, symbols, &stage),
            Some(YzlOp::Rename(stage)) => self.convert_rename(op, symbols, &stage),
            Some(YzlOp::Output(_)) => self.convert_output(op),
            // `yzr.table` carries the row as its type, so the declaration is
            // not needed.
            Some(YzlOp::Table(_)) => {}
            // `record_externals` read its name, and a call to it becomes
            // `yz.extern_call`.
            Some(YzlOp::Fn(function)) if function.is_external() => {}
            Some(YzlOp::Fn(function))
                if Operator::implemented_by(function.sym_name().value()).is_some() =>
            {
                self.convert_implementation(op, symbols, &function);
            }
            Some(YzlOp::Fn(_) | YzlOp::Trait(_) | YzlOp::Impl(_)) => self.report(
                op,
                &format!("`{}` was not expanded before lowering", op_name(op)),
            ),
            Some(YzlOp::Local(_) | YzlOp::Load(_) | YzlOp::Store(_)) => self.report(
                op,
                &format!(
                    "`{}` was not promoted to a value before lowering",
                    op_name(op)
                ),
            ),
            Some(YzlOp::Missing(_)) => self.report(op, "this part of the query is missing"),
            Some(YzlOp::Call(_) | YzlOp::List(_) | YzlOp::Yield(_) | YzlOp::Return(_)) | None => {}
        }
    }

    /// The yz struct stands where the yzl one did, under the same name: the
    /// new one is placed first, and named once the old one is erased.
    fn convert_struct(
        &mut self,
        op: OperationRef<'c, '_>,
        symbols: &mut SymbolTable<'c, 'a>,
        item: &StructOp<'c, '_>,
    ) {
        let name = item.sym_name().value();
        let fields = struct_fields(item);
        let placed = self.insert(struct_declaration(
            self.context,
            name,
            &fields,
            op.location(),
        ));
        self.anchor = placed;
        // SAFETY: `op` and `item` are not used after this, and the anchor
        // no longer points at `op`.
        unsafe { symbols.erase(op) };
        symbols.insert_placed(placed);
    }

    /// An operator's implementation becomes a `yz.func`, generic as it was
    /// declared, under the same name. `legalize_operators` copies its body
    /// in place of each operator a fold did not remove.
    fn convert_implementation(
        &mut self,
        op: OperationRef<'c, '_>,
        symbols: &mut SymbolTable<'c, 'a>,
        function: &FnOp<'c, '_>,
    ) {
        let location = op.location();
        let Ok(signature) = FunctionType::try_from(function.signature().value()) else {
            self.report(op, "an operator's implementation has no function type");
            return;
        };

        let parameters: Vec<_> = (0..signature.input_count())
            .map(|index| {
                let ty = signature.input(index).expect("the input index is in range");
                (ty, location)
            })
            .collect();
        let region = Region::new();
        let body = region.append_block(Block::new(&parameters));
        if let Some(block) = function.body().first_block() {
            let arguments: Vec<Value<'c, '_>> = body.arguments().map(Into::into).collect();
            IrRewriter::new(self.context)
                .as_rewriter_base()
                .merge_blocks(block, body, &arguments);
            let returned = self.convert_moved(body);
            body.append_operation(yz::r#return(self.context, &returned, location).into());
        }

        let placed = self.insert(
            yz::func(
                self.context,
                region,
                function.sym_name(),
                TypeAttribute::new(signature.into()),
                location,
            )
            .into(),
        );
        self.anchor = placed;
        // SAFETY: `op` and `function` are not used after this, and the
        // anchor no longer points at `op`.
        unsafe { symbols.erase(op) };
        symbols.insert_placed(placed);
    }

    fn convert_from(
        &mut self,
        op: OperationRef<'c, '_>,
        symbols: &mut SymbolTable<'c, 'a>,
        from: &FromOp<'c, '_>,
    ) {
        let relation = from.source().value();
        if let Some((rows, row)) = self.relation_input(relation, symbols, op.location()) {
            self.record_stage(op, rows, row);
        }
    }

    fn convert_let(
        &mut self,
        op: OperationRef<'c, '_>,
        symbols: &mut SymbolTable<'c, 'a>,
        binding: &ConstOp<'c, '_>,
    ) {
        let Some(block) = binding.body().first_block() else {
            self.report(op, "`let` has no body to bind");
            return;
        };

        for inner in block.operations() {
            self.convert_op(inner, symbols);
        }

        let bound = block
            .last_operation()
            .and_then(|yielded| yielded.try_first_operand())
            .and_then(|value| self.stages.get(&value.id()).cloned());

        match bound {
            Some(rows) => {
                self.bindings.insert(binding.sym_name().value(), rows);
            }
            None => self.report(op, "only a query can be bound by `let`"),
        }
    }

    fn convert_where(&mut self, op: OperationRef<'c, '_>, stage: &WhereOp<'c, '_>) {
        let Some((input, row)) = self.input_stage(op) else {
            return;
        };

        let (region, _) = self.convert_region(stage.body(), &row, op.location(), Yielded::Body);
        let filtered = self.insert(yzr::filter(self.context, input, region, op.location()).into());

        self.record_stage(op, filtered.first_result(), row);
    }

    fn convert_select(
        &mut self,
        op: OperationRef<'c, '_>,
        symbols: &mut SymbolTable<'c, 'a>,
        stage: &SelectOp<'c, '_>,
    ) {
        let Some((input, row)) = self.input_stage(op) else {
            return;
        };

        let (region, yielded) =
            self.convert_region(stage.body(), &row, op.location(), Yielded::Body);
        let produced = stage.names().strings().zip(yielded).collect();
        self.project(op, symbols, input, region, produced);
    }

    fn convert_extend(
        &mut self,
        op: OperationRef<'c, '_>,
        symbols: &mut SymbolTable<'c, 'a>,
        stage: &ExtendOp<'c, '_>,
    ) {
        let Some((input, mut row)) = self.input_stage(op) else {
            return;
        };

        let (region, yielded) =
            self.convert_region(stage.body(), &row, op.location(), Yielded::Body);
        row.extend(stage.names().strings().zip(yielded));
        let ty = self.row_type(&row, symbols, op.location());
        let extended =
            self.insert(yzr::extend(self.context, ty, input, region, op.location()).into());

        self.record_stage(op, extended.first_result(), row);
    }

    fn convert_aggregate(
        &mut self,
        op: OperationRef<'c, '_>,
        symbols: &mut SymbolTable<'c, 'a>,
        stage: &AggregateOp<'c, '_>,
    ) {
        let Some((input, row)) = self.input_stage(op) else {
            return;
        };

        let keys: Vec<usize> = stage
            .key_cols()
            .map(|keys| keys.indices().collect())
            .unwrap_or_default();
        let grouping = self.convert_grouping(stage.body(), &row, &keys, op.location());

        let carried: Row<'c> = keys
            .iter()
            .zip(stage.group_by().strings())
            .filter_map(|(&index, name)| row.get(index).map(|&(_, ty)| (name, ty)))
            .collect();

        // Without a projection the grouping's output is the row; with one it
        // is the keys and the raw measures the projection computes from.
        let names: Vec<&str> = stage.names().strings().collect();
        let mut grouped = carried.clone();
        match &grouping.items {
            Some(_) => grouped.extend(
                grouping
                    .measure_types
                    .iter()
                    .enumerate()
                    .map(|(index, &ty)| (self.intern(&format!("measure{index}")), ty)),
            ),
            None => grouped.extend(
                names
                    .iter()
                    .copied()
                    .zip(grouping.measure_types.iter().copied()),
            ),
        }

        let ty = self.row_type(&grouped, symbols, op.location());
        let indices: Vec<i64> = keys.iter().map(|&index| index as i64).collect();
        let aggregated = self
            .insert(
                yzr::aggregate(
                    self.context,
                    ty,
                    input,
                    grouping.measures,
                    DenseI64ArrayAttribute::new(self.context, &indices),
                    op.location(),
                )
                .into(),
            )
            .first_result();

        let Some((items, types)) = grouping.items else {
            self.record_stage(op, aggregated, grouped);
            return;
        };

        let mut produced = carried;
        produced.extend(names.into_iter().zip(types[keys.len()..].iter().copied()));
        self.project(op, symbols, aggregated, items, produced);
    }

    fn convert_join(
        &mut self,
        op: OperationRef<'c, '_>,
        symbols: &mut SymbolTable<'c, 'a>,
        stage: &JoinOp<'c, '_>,
    ) {
        let Some((lhs, mut row)) = self.input_stage(op) else {
            return;
        };

        // yzl names the right side; yzr joins two relations.
        let relation = stage.rhs().value();
        let Some((rows, right)) = self.relation_input(relation, symbols, op.location()) else {
            return;
        };

        let left_width = row.len();
        row.extend(right.iter().copied());

        let region = match stage.using_columns() {
            Some(columns) => {
                let columns: Vec<&str> = columns.strings().collect();
                self.join_keys(op, &columns, left_width, &row)
            }
            None => {
                self.convert_region(stage.on(), &row, op.location(), Yielded::Body)
                    .0
            }
        };

        let ty = self.row_type(&row, symbols, op.location());
        let joined = self.insert(
            yzr::join(
                self.context,
                ty,
                lhs,
                rows,
                region,
                StringAttribute::new(self.context, stage.kind().as_str()),
                op.location(),
            )
            .into(),
        );

        self.record_stage(op, joined.first_result(), row);
    }

    fn convert_limit(&mut self, op: OperationRef<'c, '_>, stage: &LimitOp<'c, '_>) {
        let Some((input, row)) = self.input_stage(op) else {
            return;
        };

        let mut builder = yzr::LimitOperationBuilder::new(self.context, op.location())
            .input(input)
            .count(stage.count());
        if let Some(offset) = stage.offset() {
            builder = builder.offset(offset);
        }

        let limited = self.insert(builder.build().into());
        self.record_stage(op, limited.first_result(), row);
    }

    /// `alias` only qualifies names, and resolution has already used them.
    fn convert_alias(&mut self, op: OperationRef<'c, '_>) {
        let Some((input, row)) = self.input_stage(op) else {
            return;
        };

        self.record_stage(op, input, row);
    }

    /// yzr has no distinct: a group keyed on every column, measuring nothing.
    fn convert_distinct(&mut self, op: OperationRef<'c, '_>, symbols: &mut SymbolTable<'c, 'a>) {
        let Some((input, row)) = self.input_stage(op) else {
            return;
        };

        let keys: Vec<i64> = (0..row.len() as i64).collect();
        let region = self.column_region(&row, &[], op.location());
        let ty = self.row_type(&row, symbols, op.location());
        let grouped = self.insert(
            yzr::aggregate(
                self.context,
                ty,
                input,
                region,
                DenseI64ArrayAttribute::new(self.context, &keys),
                op.location(),
            )
            .into(),
        );

        self.record_stage(op, grouped.first_result(), row);
    }

    fn convert_drop(
        &mut self,
        op: OperationRef<'c, '_>,
        symbols: &mut SymbolTable<'c, 'a>,
        stage: &DropOp<'c, '_>,
    ) {
        let Some((input, row)) = self.input_stage(op) else {
            return;
        };

        let dropped: Vec<&str> = stage.columns().strings().collect();
        let Some(kept) = self.kept_columns(op, &dropped, &row) else {
            return;
        };

        let region = self.column_region(&row, &kept, op.location());
        let produced = kept.iter().map(|&index| row[index]).collect();
        self.project(op, symbols, input, region, produced);
    }

    fn convert_set(
        &mut self,
        op: OperationRef<'c, '_>,
        symbols: &mut SymbolTable<'c, 'a>,
        stage: &SetOp<'c, '_>,
    ) {
        let Some((input, mut row)) = self.input_stage(op) else {
            return;
        };

        let columns: Vec<usize> = stage
            .set_cols()
            .map(|columns| columns.indices().collect())
            .unwrap_or_default();
        let (region, yielded) = self.convert_region(
            stage.body(),
            &row,
            op.location(),
            Yielded::Substituted(&columns),
        );
        for (column, ty) in row.iter_mut().zip(&yielded) {
            column.1 = *ty;
        }

        self.project(op, symbols, input, region, row);
    }

    /// yzr rows are typed by their struct, so new names need a projection.
    fn convert_rename(
        &mut self,
        op: OperationRef<'c, '_>,
        symbols: &mut SymbolTable<'c, 'a>,
        stage: &RenameOp<'c, '_>,
    ) {
        let Some((input, mut row)) = self.input_stage(op) else {
            return;
        };

        let columns = stage
            .rename_cols()
            .into_iter()
            .flat_map(|columns| columns.indices());
        for (index, name) in columns.zip(stage.to().strings()) {
            match row.get_mut(index) {
                Some(column) => column.0 = name,
                None => {
                    self.report(op, &format!("column {index} is not in the row"));
                    return;
                }
            }
        }

        let all: Vec<usize> = (0..row.len()).collect();
        let region = self.column_region(&row, &all, op.location());
        self.project(op, symbols, input, region, row);
    }

    fn convert_output(&mut self, op: OperationRef<'c, '_>) {
        let Some((query, _)) = self.input_stage(op) else {
            return;
        };

        self.insert(yzr::output(self.context, query, op.location()).into());
    }

    fn project(
        &mut self,
        op: OperationRef<'c, '_>,
        symbols: &mut SymbolTable<'c, 'a>,
        input: Value<'c, 'a>,
        region: Region<'c>,
        produced: Row<'c>,
    ) {
        let ty = self.row_type(&produced, symbols, op.location());
        let projected =
            self.insert(yzr::project(self.context, ty, input, region, op.location()).into());

        self.record_stage(op, projected.first_result(), produced);
    }

    /// yzr has only an `on` region, so `using` becomes the equalities it
    /// means.
    fn join_keys(
        &mut self,
        op: OperationRef<'c, '_>,
        columns: &[&str],
        left_width: usize,
        row: &Row<'c>,
    ) -> Region<'c> {
        let location = op.location();
        let region = Region::new();
        let body = self.row_block(&region, row, location);

        let mut condition: Option<Value<'c, '_>> = None;
        for column in columns {
            let left = row[..left_width]
                .iter()
                .position(|(name, _)| name == column);
            let right = row[left_width..]
                .iter()
                .position(|(name, _)| name == column)
                .map(|index| index + left_width);
            let (Some(left), Some(right)) = (left, right) else {
                self.report(op, &format!("`{column}` is not present in both relations"));
                continue;
            };

            let equal = body.append_operation(
                yz::cmp(
                    self.context,
                    BoolType::get(self.context),
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
                            BoolType::get(self.context),
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

    /// Resolution removes the first column each name matches, so dropping
    /// one name twice drops two columns, and this has to agree exactly.
    fn kept_columns(
        &self,
        op: OperationRef<'c, '_>,
        columns: &[&str],
        row: &Row<'c>,
    ) -> Option<Vec<usize>> {
        let mut dropped: Vec<usize> = Vec::new();
        for column in columns {
            let found = row
                .iter()
                .enumerate()
                .find(|(index, (name, _))| name == column && !dropped.contains(index))
                .map(|(index, _)| index);

            match found {
                Some(index) => dropped.push(index),
                None => {
                    self.report(op, &format!("`{column}` is not in the row"));
                    return None;
                }
            }
        }

        Some(
            (0..row.len())
                .filter(|index| !dropped.contains(index))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::test_support::check_yzr;

    #[test]
    fn named_stages_declare_the_row_they_produce() {
        check_yzr(
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

    #[test]
    fn join_materialises_its_right_side() {
        check_yzr(
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

    #[test]
    fn using_becomes_the_equality_it_means() {
        check_yzr(
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

    #[test]
    fn alias_leaves_no_trace() {
        check_yzr(
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

    #[test]
    fn distinct_groups_on_every_column() {
        check_yzr(
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

    #[test]
    fn drop_projects_the_columns_that_stay() {
        check_yzr(
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

    #[test]
    fn set_yields_the_untouched_columns_too() {
        check_yzr(
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

    #[test]
    fn a_binding_is_reused_not_rescanned() {
        check_yzr(
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

    #[test]
    fn reports_a_binding_that_is_not_a_query() {
        check_yzr(
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
                  | ^^^^^^^^^^^^^
            "#]],
        );
    }

    #[test]
    fn rename_projects_under_the_new_names() {
        check_yzr(
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

    #[test]
    fn rename_follows_the_stamped_column() {
        check_yzr(
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

    #[test]
    fn a_single_rename_projects_too() {
        check_yzr(
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
