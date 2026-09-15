//! Regions, rebuilt so the row's columns are block arguments — which is
//! what turns a column reference into an SSA use.
use std::collections::HashMap;

use melior::ir::{
    Block, BlockLike, BlockRef, Location, Region, RegionLike, Type, Value, ValueLike,
};
use yuzu_mlir::ext::{BlockExt, ValueExt};
use yuzu_mlir::ods::yzr;

use crate::lower_yzl_to_yzr::{Schema, Yielded, YzlToYzr};

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
        let arguments: Vec<(Type<'c>, Location<'c>)> = schema
            .iter()
            .map(|(_, column)| (*column, location))
            .collect();
        let body = region.append_block(Block::new(&arguments));

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
