use std::collections::HashMap;

use melior::ir::attribute::{FlatSymbolRefAttribute, StringAttribute};
use melior::ir::operation::{OperationBuilder, OperationLike, OperationRef};
use melior::ir::{Attribute, BlockLike, BlockRef, Identifier, Operation, Type, Value, ValueLike};
use yuzu_mlir::ListType;
use yuzu_mlir::attributes::CalleeSource;
use yuzu_mlir::ir::operation::{OperationCast, OperationExt};
use yuzu_mlir::ir::value::{ValueExt, ValueId};
use yuzu_mlir::ods::{yz, yzr};
use yuzu_mlir::ops::yzl::YzlOp;

use crate::lower_yzl_to_yzr::{YzlToYzr, op_name};

impl<'c, 'a> YzlToYzr<'c, 'a> {
    pub(super) fn convert_expression<'b>(
        &mut self,
        op: OperationRef<'c, '_>,
        body: BlockRef<'c, 'b>,
        values: &mut HashMap<ValueId, Value<'c, 'b>>,
        produced: &mut Vec<Value<'c, 'b>>,
    ) {
        match op.as_yzl() {
            Some(YzlOp::Yield(_)) => {
                let Some(operands) = lowered_operands(op, values) else {
                    return;
                };

                produced.extend(operands);
            }
            Some(YzlOp::Call(call)) => {
                let callee = call.callee().value();
                let Some(operands) = lowered_operands(op, values) else {
                    return;
                };

                let ty = op.first_result().r#type();
                let kind = call.callee_source();
                if matches!(kind, Some(CalleeSource::Fn | CalleeSource::Const)) {
                    self.report(op, &format!("`{callee}` was not expanded before lowering"));
                    return;
                }

                let lowered = if call.is_agg() {
                    self.convert_measure(op, callee, &operands, ty)
                } else if kind == Some(CalleeSource::External) {
                    yz::extern_call(
                        self.context,
                        ty,
                        &operands,
                        StringAttribute::new(self.context, callee),
                        op.location(),
                    )
                    .into()
                } else {
                    yz::call(
                        self.context,
                        ty,
                        &operands,
                        FlatSymbolRefAttribute::new(self.context, callee),
                        op.location(),
                    )
                    .into()
                };

                let appended = body.append_operation(lowered);
                values.insert(op.first_result().id(), appended.first_result());
            }
            Some(YzlOp::List(_)) => {
                let Some(operands) = lowered_operands(op, values) else {
                    return;
                };

                let ty = op.first_result().r#type();
                if ListType::from_type(ty).is_none() {
                    self.report(op, "the type of this list could not be inferred");
                    return;
                }

                let appended = body
                    .append_operation(yz::list(self.context, ty, &operands, op.location()).into());
                values.insert(op.first_result().id(), appended.first_result());
            }
            None => {
                let Some(operands) = lowered_operands(op, values) else {
                    return;
                };

                if let Some(rebuilt) = self.rebuild(op, &operands, body) {
                    values.insert(op.first_result().id(), rebuilt);
                }
            }
            Some(YzlOp::Missing(_)) => self.report(op, "this part of the query is missing"),
            Some(
                YzlOp::From(_)
                | YzlOp::Where(_)
                | YzlOp::Select(_)
                | YzlOp::Extend(_)
                | YzlOp::Aggregate(_)
                | YzlOp::Limit(_)
                | YzlOp::Join(_)
                | YzlOp::Rename(_)
                | YzlOp::Alias(_)
                | YzlOp::Distinct(_)
                | YzlOp::Drop(_)
                | YzlOp::Set(_)
                | YzlOp::Output(_)
                | YzlOp::Struct(_)
                | YzlOp::Table(_)
                | YzlOp::Fn(_)
                | YzlOp::Trait(_)
                | YzlOp::Impl(_)
                | YzlOp::Const(_)
                | YzlOp::Return(_),
            ) => self.report(op, &format!("`{}` is not lowered yet", op_name(op))),
            Some(YzlOp::Local(_) | YzlOp::Load(_) | YzlOp::Store(_)) => self.report(
                op,
                &format!(
                    "`{}` was not promoted to a value before lowering",
                    op_name(op)
                ),
            ),
        }
    }

    fn rebuild<'b>(
        &mut self,
        op: OperationRef<'c, '_>,
        operands: &[Value<'c, 'b>],
        body: BlockRef<'c, 'b>,
    ) -> Option<Value<'c, 'b>> {
        let name = op.name();
        let ty = op.try_first_result()?.r#type();
        let attributes: Vec<(Identifier<'c>, Attribute<'c>)> = (0..op.attribute_count())
            .map(|index| {
                op.attribute_at(index)
                    .expect("the attribute index is in range")
            })
            .collect();

        let rebuilt = OperationBuilder::new(
            name.as_string_ref().as_str().expect("op names are utf-8"),
            op.location(),
        )
        .add_operands(operands)
        .add_results(&[ty])
        .add_attributes(&attributes)
        .build()
        .expect("a stamped yz op rebuilds");

        Some(body.append_operation(rebuilt).first_result())
    }

    /// A measure: `count` takes no value, every other aggregate does.
    fn convert_measure(
        &self,
        op: OperationRef<'c, '_>,
        callee: &str,
        operands: &[Value<'c, '_>],
        ty: Type<'c>,
    ) -> Operation<'c> {
        match operands.first() {
            Some(value) => yzr::agg(
                self.context,
                ty,
                *value,
                StringAttribute::new(self.context, callee),
                op.location(),
            )
            .into(),
            None => yzr::count(self.context, ty, op.location()).into(),
        }
    }
}

/// The region is rebuilt from the top and is `IsolatedFromAbove`, so an
/// operand with nothing standing for it means its producer failed and has
/// already reported.
fn lowered_operands<'c, 'b>(
    op: OperationRef<'c, '_>,
    values: &HashMap<ValueId, Value<'c, 'b>>,
) -> Option<Vec<Value<'c, 'b>>> {
    op.operands()
        .map(|operand| values.get(&operand.id()).copied())
        .collect()
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::test_support::check_yzr;

    #[test]
    fn measures_become_aggregate_ops() {
        check_yzr(
            r#"
struct Row { a: int64, b: int64 }
table t = Row

from t
|> aggregate sum(a) as total, count() as n group by b
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a", "b"] : [!yz.int64, !yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  yz.struct @row ["b", "total", "n"] : [!yz.int64, !yz.int64, !yz.int64]
                  %1 = yzr.aggregate %0 keys [1] {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64):
                    %2 = yzr.agg "sum", %arg0 : !yz.int64 -> !yz.int64
                    %3 = yzr.count : !yz.int64
                    yzr.yield %2, %3 : !yz.int64, !yz.int64
                  } : !yz.struct<@Row> -> !yz.struct<@row>
                  yzr.output %1 : !yz.struct<@row>
                }
            "#]],
        );
    }

    #[test]
    fn reports_a_missing_piece() {
        check_yzr(
            r#"
struct Row { a: int64 }
table t = Row

from t
|> where a >
"#,
            expect![[r#"
                error: expected expression, found end of input
                 --> test.yz:6:13
                  |
                6 | |> where a >
                  | 

                error: binary expression is missing its right operand
                 --> test.yz:6:10
                  |
                6 | |> where a >
                  |          ^^^

                error: this part of the query is missing
                 --> test.yz:6:10
                  |
                6 | |> where a >
                  |          ^^^
            "#]],
        );
    }

    #[test]
    fn a_list_lowers_with_its_type() {
        check_yzr(
            r#"
struct Row { a: int64 }
table t = Row

from t
|> where a in [1, 3]
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a"] : [!yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  %1 = yzr.filter %0 : !yz.struct<@Row> {
                  ^bb0(%arg0: !yz.int64):
                    %2 = yz.constant_int 1
                    %3 = yz.constant_int 3
                    %4 = yz.list[%2, %3] : (!yz.int64, !yz.int64) -> !yz.list<!yz.int64>
                    %5 = yz.call @in(%arg0, %4) : (!yz.int64, !yz.list<!yz.int64>) -> !yz.bool
                    yzr.yield %5 : !yz.bool
                  }
                  yzr.output %1 : !yz.struct<@Row>
                }
            "#]],
        );
    }
}
