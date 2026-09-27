use melior::IrRewriter;
use melior::ir::operation::{OperationLike, OperationResult};
use melior::ir::{
    Block, BlockLike, BlockRef, Location, Region, RegionLike, RegionRef, Type, Value, ValueLike,
};
use rustc_hash::{FxHashMap, FxHashSet};
use yuzu_mlir::ir::block::BlockExt;
use yuzu_mlir::ir::operation::{OperationCast, OperationExt};
use yuzu_mlir::ir::value::{ValueExt, ValueId};
use yuzu_mlir::ods::yzr;
use yuzu_mlir::ops::yzl::YzlOp;

use crate::lower_yzl_to_yzr::{Row, Yielded, YzlToYzr};

/// An aggregation over the input row and, when an item is more than a bare
/// measure, a projection over the keys and measures.
pub(super) struct Grouping<'c> {
    pub(super) measures: Region<'c>,
    pub(super) measure_types: Vec<Type<'c>>,
    pub(super) items: Option<(Region<'c>, Vec<Type<'c>>)>,
}

impl<'c, 'a> YzlToYzr<'c, 'a> {
    pub(super) fn convert_region(
        &mut self,
        source: RegionRef<'c, '_>,
        row: &Row<'c>,
        location: Location<'c>,
        yielded: Yielded<'_>,
    ) -> (Region<'c>, Vec<Type<'c>>) {
        let region = Region::new();
        let body = self.row_block(&region, row, location);

        let mut produced = Vec::new();
        if let Some(block) = source.first_block() {
            // The expressions move into the new block, whose arguments stand
            // in for the old ones; only the yzl ops among them are rewritten.
            let arguments: Vec<Value<'c, '_>> = body.arguments().map(Into::into).collect();
            if arguments.len() != block.argument_count() {
                self.report_at(location, "a region's arguments do not match its row");
            } else {
                IrRewriter::new(self.context)
                    .as_rewriter_base()
                    .merge_blocks(block, body, &arguments);
                produced = self.convert_moved(body);
            }
        }

        let results = match yielded {
            Yielded::Body => produced,
            Yielded::Substituted(columns) => {
                Self::substituted_row(body, row.len(), columns, &produced)
            }
        };

        let types = results.iter().map(|value| value.r#type()).collect();
        body.append_operation(yzr::r#yield(self.context, &results, location).into());

        (region, types)
    }

    /// A `yzl.aggregate` item may compute over its own measures. Substrait's
    /// aggregation cannot, so such an item becomes a projection after the
    /// grouping.
    pub(super) fn convert_grouping(
        &mut self,
        source: RegionRef<'c, '_>,
        row: &Row<'c>,
        keys: &[usize],
        location: Location<'c>,
    ) -> Grouping<'c> {
        let Some(block) = source.first_block() else {
            return Grouping {
                measures: self.column_region(row, &[], location),
                measure_types: Vec::new(),
                items: None,
            };
        };

        let measures: Vec<_> = block
            .operations()
            .filter(|op| matches!(op.as_yzl(), Some(YzlOp::Call(call)) if call.is_agg()))
            .collect();
        let feeds_a_measure = rests_on(
            measures.iter().flat_map(|op| op.operands()),
            &FxHashSet::default(),
        );

        let region = Region::new();
        let body = self.row_block(&region, row, location);
        let mut row_values = argument_map(block, body);

        // Two items naming the same measure share one column: a target that
        // names columns after measures will not take the same one twice.
        let mut columns: FxHashMap<ValueId, usize> = FxHashMap::default();
        let mut distinct: FxHashMap<(&'c str, Vec<ValueId>), usize> = FxHashMap::default();
        let mut measured: Vec<Value<'c, '_>> = Vec::new();
        let mut discard = Vec::new();
        for op in block.operations() {
            let call = match op.as_yzl() {
                Some(YzlOp::Call(call)) if call.is_agg() => call,
                _ => {
                    if op
                        .try_first_result()
                        .is_some_and(|result| feeds_a_measure.contains(&result.id()))
                    {
                        self.convert_expression(op, body, &mut row_values, &mut discard);
                    }

                    continue;
                }
            };

            let arguments: Option<Vec<ValueId>> = op
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
                    self.convert_expression(op, body, &mut row_values, &mut discard);
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
            self.convert_grouped_items(block, row, keys, &columns, &measure_types, location);
        Grouping {
            measures: region,
            measure_types,
            items,
        }
    }

    fn convert_grouped_items(
        &mut self,
        block: BlockRef<'c, '_>,
        row: &Row<'c>,
        keys: &[usize],
        columns: &FxHashMap<ValueId, usize>,
        measures: &[Type<'c>],
        location: Location<'c>,
    ) -> Option<(Region<'c>, Vec<Type<'c>>)> {
        let yielded: Vec<Value<'c, '_>> = block
            .last_operation()
            .filter(|end| matches!(end.as_yzl(), Some(YzlOp::Yield(_))))
            .map(|end| end.operands().collect())
            .unwrap_or_default();

        let bare = yielded.len() == measures.len()
            && yielded
                .iter()
                .enumerate()
                .all(|(position, value)| columns.get(&value.id()) == Some(&position));
        if bare {
            return None;
        }

        let mut grouped: Row<'c> = keys
            .iter()
            .filter_map(|&index| row.get(index).copied())
            .collect();
        grouped.extend(measures.iter().map(|&ty| ("", ty)));

        let region = Region::new();
        let body = self.row_block(&region, &grouped, location);
        let mut values = FxHashMap::default();
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
        let seeded: FxHashSet<ValueId> = values.keys().copied().collect();
        let needed = rests_on(yielded.iter().copied(), &seeded);
        for op in block.operations() {
            let wanted = op
                .try_first_result()
                .is_some_and(|result| needed.contains(&result.id()));
            if wanted && !matches!(op.as_yzl(), Some(YzlOp::Call(call)) if call.is_agg()) {
                self.convert_expression(op, body, &mut values, &mut produced);
            }
        }

        // A projection replaces the row, so the keys carry through it.
        let mut results: Vec<Value<'c, '_>> = (0..keys.len())
            .map(|position| body.argument(position).expect("a key is in the row").into())
            .collect();
        for value in yielded {
            match values.get(&value.id()) {
                Some(&item) => results.push(item),
                None => return None,
            }
        }

        let types = results.iter().map(|value| value.r#type()).collect();
        body.append_operation(yzr::r#yield(self.context, &results, location).into());
        Some((region, types))
    }

    pub(super) fn row_block<'r>(
        &self,
        region: &'r Region<'c>,
        row: &Row<'c>,
        location: Location<'c>,
    ) -> BlockRef<'c, 'r> {
        let arguments: Vec<(Type<'c>, Location<'c>)> =
            row.iter().map(|(_, column)| (*column, location)).collect();
        region.append_block(Block::new(&arguments))
    }

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

    pub(super) fn column_region(
        &self,
        row: &Row<'c>,
        columns: &[usize],
        location: Location<'c>,
    ) -> Region<'c> {
        let region = Region::new();
        let body = self.row_block(&region, row, location);
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

fn argument_map<'c, 'b>(
    source: BlockRef<'c, '_>,
    target: BlockRef<'c, 'b>,
) -> FxHashMap<ValueId, Value<'c, 'b>> {
    source
        .arguments()
        .zip(target.arguments())
        .map(|(source, target)| (source.id(), target.into()))
        .collect()
}

/// Every operation result the roots reach through their operands, stopping
/// at values the caller already holds.
fn rests_on<'c: 'a, 'a>(
    roots: impl IntoIterator<Item = Value<'c, 'a>>,
    held: &FxHashSet<ValueId>,
) -> FxHashSet<ValueId> {
    let mut reached = FxHashSet::default();
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

    use crate::test_support::check_yzr;

    #[test]
    fn names_become_block_arguments() {
        check_yzr(
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
