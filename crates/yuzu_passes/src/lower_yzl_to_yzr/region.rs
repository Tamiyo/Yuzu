//! Regions, rebuilt so the row's columns are block arguments — which is
//! what turns a column reference into an SSA use.
use std::collections::{HashMap, HashSet};

use melior::ir::operation::{OperationLike, OperationRef, OperationResult};
use melior::ir::{
    Block, BlockLike, BlockRef, Location, Region, RegionLike, Type, Value, ValueLike,
};
use yuzu_mlir::ext::{BlockExt, OperationCast, OperationExt, ValueExt};
use yuzu_mlir::ods::yzr;
use yuzu_mlir::ops::yzl::YzlOp;

use crate::lower_yzl_to_yzr::{Schema, Yielded, YzlToYzr};

/// What a grouping becomes: an aggregation computing its measures over the
/// input row, and — only when an item is more than a bare measure — a
/// projection computing the items over the keys and those measures.
pub(super) struct Grouping<'c> {
    pub(super) measures: Region<'c>,
    pub(super) measure_types: Vec<Type<'c>>,
    pub(super) items: Option<(Region<'c>, Vec<Type<'c>>)>,
}

impl<'c, 'a> YzlToYzr<'c, 'a> {
    /// A stage's region, rebuilt with the input row's columns as block
    /// arguments — which is what makes column reference into SSA use-def.
    pub(super) fn lower_region(
        &mut self,
        source: melior::ir::RegionRef<'c, '_>,
        schema: &Schema<'c>,
        location: Location<'c>,
        yielded: Yielded<'_>,
    ) -> (Region<'c>, Vec<Type<'c>>) {
        let region = Region::new();
        let body = self.row_block(&region, schema, location);

        let mut produced = Vec::new();
        if let Some(block) = source.first_block() {
            // The source's block arguments are the row's columns; so are the
            // ones just built, at the same positions.
            let mut values: HashMap<usize, Value<'c, '_>> = HashMap::new();
            for index in 0..block.argument_count() {
                let source = block
                    .argument(index)
                    .expect("the argument index is in range");
                let target = body
                    .argument(index)
                    .expect("the row was built to the same width");
                values.insert(source.id(), target.into());
            }

            for op in block.operations() {
                self.lower_expression(op, body, &mut values, &mut produced);
            }
        }

        let row = match yielded {
            Yielded::Body => produced,
            Yielded::Row(columns) => Self::substituted_row(body, schema.len(), columns, &produced),
        };

        let types = row.iter().map(|value| value.r#type()).collect();
        body.append_operation(yzr::r#yield(self.context, &row, location).into());

        (region, types)
    }

    /// A `yzl.aggregate` groups and projects at once: each item is an
    /// expression that may compute over its own measures. Substrait's
    /// aggregation cannot — its output is the keys and the measures, full
    /// stop — so an item that is more than a bare measure becomes a
    /// projection after the grouping, which is what every engine does.
    pub(super) fn lower_grouping(
        &mut self,
        source: melior::ir::RegionRef<'c, '_>,
        schema: &Schema<'c>,
        keys: &[usize],
        location: Location<'c>,
    ) -> Grouping<'c> {
        let Some(block) = source.first_block() else {
            return Grouping {
                measures: self.column_region(schema, &[], location),
                measure_types: Vec::new(),
                items: None,
            };
        };

        let measured: Vec<OperationRef<'c, '_>> = block
            .operations()
            .filter(|op| matches!(op.as_yzl(), Some(YzlOp::Call(call)) if call.agg()))
            .collect();
        let arguments: Vec<Value<'c, '_>> = measured.iter().flat_map(|op| op.operands()).collect();
        let feeds_a_measure = rests_on(arguments, &HashSet::new());

        // The aggregation: the measures, and whatever they rest on.
        let region = Region::new();
        let body = self.row_block(&region, schema, location);
        let mut row_values = HashMap::new();
        for index in 0..block.argument_count() {
            let column = block
                .argument(index)
                .expect("the argument index is in range");
            let target = body
                .argument(index)
                .expect("the row was built to the same width");
            row_values.insert(column.id(), target.into());
        }

        // Two items may name the same measure. It is one column of the
        // grouping's output either way — and a target that derives a column
        // name from the measure will not take the same one twice.
        let mut columns: HashMap<usize, usize> = HashMap::new();
        let mut distinct: HashMap<(&'c str, Vec<usize>), usize> = HashMap::new();
        let mut measured: Vec<Value<'c, '_>> = Vec::new();
        let mut discard = Vec::new();
        for op in block.operations() {
            let Some(YzlOp::Call(call)) = op.as_yzl().filter(|_| true) else {
                if op
                    .try_first_result()
                    .is_some_and(|result| feeds_a_measure.contains(&result.id()))
                {
                    self.lower_expression(op, body, &mut row_values, &mut discard);
                }

                continue;
            };

            if !call.agg() {
                if op
                    .try_first_result()
                    .is_some_and(|result| feeds_a_measure.contains(&result.id()))
                {
                    self.lower_expression(op, body, &mut row_values, &mut discard);
                }

                continue;
            }

            // Two measures are the same measure when they apply the same
            // function to the same values.
            let arguments: Option<Vec<usize>> = op
                .operands()
                .map(|operand| row_values.get(&operand.id()).map(|value| value.id()))
                .collect();
            let (Some(arguments), Some(result)) = (arguments, op.try_first_result()) else {
                continue;
            };

            let name = call.callee().value();
            let column = match distinct.get(&(name, arguments.clone())).copied() {
                Some(column) => column,
                None => {
                    self.lower_expression(op, body, &mut row_values, &mut discard);
                    let Some(&value) = row_values.get(&result.id()) else {
                        continue;
                    };

                    measured.push(value);
                    distinct.insert((name, arguments), measured.len() - 1);
                    measured.len() - 1
                }
            };

            columns.insert(result.id(), column);
        }

        let measure_types: Vec<Type<'c>> = measured.iter().map(|value| value.r#type()).collect();
        body.append_operation(yzr::r#yield(self.context, &measured, location).into());

        let items =
            self.lower_grouped_items(block, schema, keys, &columns, &measure_types, location);
        Grouping {
            measures: region,
            measure_types,
            items,
        }
    }

    /// The items, over the keys and the measures — `None` when every item is
    /// a bare measure, which is the grouping's own output already.
    fn lower_grouped_items(
        &mut self,
        block: BlockRef<'c, '_>,
        schema: &Schema<'c>,
        keys: &[usize],
        columns: &HashMap<usize, usize>,
        measures: &[Type<'c>],
        location: Location<'c>,
    ) -> Option<(Region<'c>, Vec<Type<'c>>)> {
        let yielded: Vec<Value<'c, '_>> = block
            .last_operation()
            .filter(|end| matches!(end.as_yzl(), Some(YzlOp::Yield(_))))
            .map(|end| end.operands().collect())
            .unwrap_or_default();

        // Every item a bare measure, each a different one, in order: that
        // is the grouping's own output, and nothing is left to compute.
        let bare = yielded.len() == measures.len()
            && yielded
                .iter()
                .enumerate()
                .all(|(position, value)| columns.get(&value.id()) == Some(&position));
        if bare {
            return None;
        }

        // The row the projection reads is what the grouping produced.
        let mut grouped: Schema<'c> = keys
            .iter()
            .filter_map(|&index| schema.get(index).copied())
            .collect();
        grouped.extend(measures.iter().map(|&ty| ("", ty)));

        let region = Region::new();
        let body = self.row_block(&region, &grouped, location);
        let mut values = HashMap::new();
        for (position, &index) in keys.iter().enumerate() {
            let column = block
                .argument(index)
                .expect("a key is a column of the input row");
            let target = body.argument(position).expect("a key is in the row");
            values.insert(column.id(), target.into());
        }

        for (&id, &column) in columns {
            let target = body
                .argument(keys.len() + column)
                .expect("a measure is in the row");
            values.insert(id, target.into());
        }

        let mut produced = Vec::new();
        let seeded: HashSet<usize> = values.keys().copied().collect();
        let needed = rests_on(yielded.iter().copied(), &seeded);
        for op in block.operations() {
            let wanted = op
                .try_first_result()
                .is_some_and(|result| needed.contains(&result.id()));
            if wanted && !matches!(op.as_yzl(), Some(YzlOp::Call(call)) if call.agg()) {
                self.lower_expression(op, body, &mut values, &mut produced);
            }
        }

        // A projection replaces the row, so the keys carry through it.
        let mut row: Vec<Value<'c, '_>> = (0..keys.len())
            .map(|position| body.argument(position).expect("a key is in the row").into())
            .collect();
        for value in yielded {
            match values.get(&value.id()) {
                Some(&item) => row.push(item),
                None => return None,
            }
        }

        let types = row.iter().map(|value| value.r#type()).collect();
        body.append_operation(yzr::r#yield(self.context, &row, location).into());
        Some((region, types))
    }

    /// A block whose arguments are a row's columns.
    fn row_block<'r>(
        &self,
        region: &'r Region<'c>,
        schema: &Schema<'c>,
        location: Location<'c>,
    ) -> BlockRef<'c, 'r> {
        let arguments: Vec<(Type<'c>, Location<'c>)> = schema
            .iter()
            .map(|(_, column)| (*column, location))
            .collect();
        region.append_block(Block::new(&arguments))
    }

    /// The whole row, with each replaced column taking the value the body
    /// computed for it — what `set` means, against a yzr that only projects.
    fn substituted_row<'b>(
        body: BlockRef<'c, 'b>,
        width: usize,
        columns: &[usize],
        produced: &[Value<'c, 'b>],
    ) -> Vec<Value<'c, 'b>> {
        (0..width)
            .map(|index| {
                columns
                    .iter()
                    .position(|&column| column == index)
                    .and_then(|slot| produced.get(slot).copied())
                    .unwrap_or_else(|| {
                        body.argument(index)
                            .expect("the column is in the row")
                            .into()
                    })
            })
            .collect()
    }

    /// A region yielding the row's own columns, in the order given — what the
    /// stages that only move names become, since yzr projects rows and has no
    /// op for a change of name alone.
    pub(super) fn column_region(
        &self,
        schema: &Schema<'c>,
        columns: &[usize],
        location: Location<'c>,
    ) -> Region<'c> {
        let region = Region::new();
        let arguments: Vec<(Type<'c>, Location<'c>)> = schema
            .iter()
            .map(|(_, column)| (*column, location))
            .collect();
        let body = region.append_block(Block::new(&arguments));

        let yielded: Vec<Value<'c, '_>> = columns
            .iter()
            .map(|&index| {
                body.argument(index)
                    .expect("the column is in the row")
                    .into()
            })
            .collect();

        body.append_operation(yzr::r#yield(self.context, &yielded, location).into());

        region
    }
}

/// The values an expression rests on: every operation result it reaches
/// walking back through the operands, stopping where the caller already has
/// the value in hand. A block argument produces nothing, so the walk ends
/// there on its own.
fn rests_on<'c: 'a, 'a>(
    roots: impl IntoIterator<Item = Value<'c, 'a>>,
    held: &HashSet<usize>,
) -> HashSet<usize> {
    let mut reached = HashSet::new();
    let mut pending: Vec<Value<'c, 'a>> = roots.into_iter().collect();
    while let Some(value) = pending.pop() {
        if held.contains(&value.id()) {
            continue;
        }

        let Ok(result) = OperationResult::try_from(value) else {
            continue;
        };

        if !reached.insert(value.id()) {
            continue;
        }

        pending.extend(result.owner().operands());
    }

    reached
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::test_support::check_lowered;

    /// The stage regions' block arguments are the row's columns, so a column
    /// reference becomes an SSA use.
    #[test]
    fn names_become_block_arguments() {
        check_lowered(
            r#"
struct Row { a: int64, b: int64 }
table t = Row

from t
|> where a > 10
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a", "b"] : [!yz.int64, !yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  %1 = yzr.filter %0 : !yz.struct<@Row> {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64):
                    %2 = yz.constant_int 10
                    %3 = yz.cmp "gt", %arg0, %2 : !yz.int64, !yz.int64 -> !yz.bool
                    yzr.yield %3 : !yz.bool
                  }
                  yzr.output %1 : !yz.struct<@Row>
                }
            "#]],
        );
    }
}
