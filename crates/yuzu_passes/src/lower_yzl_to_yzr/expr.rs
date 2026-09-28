use melior::IrRewriter;
use melior::ir::attribute::{FlatSymbolRefAttribute, StringAttribute};
use melior::ir::operation::{OperationBuilder, OperationLike, OperationRef};
use melior::ir::{Attribute, BlockLike, BlockRef, Identifier, Operation, Type, Value, ValueLike};
use rustc_hash::FxHashMap;
use yuzu_mlir::ListType;
use yuzu_mlir::attributes::CalleeSource;
use yuzu_mlir::ir::block::BlockExt;
use yuzu_mlir::ir::operation::{OperationCast, OperationExt};
use yuzu_mlir::ir::value::{ValueExt, ValueId};
use yuzu_mlir::ods::{yz, yzr};
use yuzu_mlir::ops::yzl::{CallOp, YzlOp};

use crate::lower_yzl_to_yzr::{YzlToYzr, op_name, report};

impl<'c> YzlToYzr<'c, '_> {
    /// Copies one expression into `body`, for a region that cannot take the
    /// source's ops as they are: a grouping splits them between two regions.
    pub(super) fn convert_expression<'b>(
        &mut self,
        op: OperationRef<'c, '_>,
        body: BlockRef<'c, 'b>,
        values: &mut FxHashMap<ValueId, Value<'c, 'b>>,
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
                let Some(operands) = lowered_operands(op, values) else {
                    return;
                };

                if let Some(lowered) = self.convert_call(op, call, &operands) {
                    let appended = body.append_operation(lowered);
                    values.insert(op.first_result().id(), appended.first_result());
                }
            }
            Some(YzlOp::List(_)) => {
                let Some(operands) = lowered_operands(op, values) else {
                    return;
                };

                if let Some(lowered) = self.convert_list(op, &operands) {
                    let appended = body.append_operation(lowered);
                    values.insert(op.first_result().id(), appended.first_result());
                }
            }
            None => {
                let Some(operands) = lowered_operands(op, values) else {
                    return;
                };

                if let Some(rebuilt) = rebuild(op, &operands, body) {
                    values.insert(op.first_result().id(), rebuilt);
                }
            }
            Some(_) => report_unlowered(op),
        }
    }

    /// Lowers the yzl ops among expressions moved into `body`, and returns
    /// what its `yzl.yield` or `yzl.return` gave back. A `yz` op is already what yzr wants,
    /// so it stays where it is.
    pub(super) fn convert_moved<'b>(&mut self, body: BlockRef<'c, 'b>) -> Vec<Value<'c, 'b>> {
        let rewriter = IrRewriter::new(self.context);
        let rewriter = rewriter.as_rewriter_base();
        let mut produced = Vec::new();
        let ops: Vec<OperationRef<'c, 'b>> = body.operations().collect();
        for op in ops {
            let operands: Vec<Value<'c, 'b>> = op.operands().collect();
            let lowered = match op.as_yzl() {
                None => continue,
                Some(YzlOp::Yield(_) | YzlOp::Return(_)) => {
                    produced = operands;
                    rewriter.erase_op(op);
                    continue;
                }
                Some(YzlOp::Call(call)) => self.convert_call(op, call, &operands),
                Some(YzlOp::List(_)) => self.convert_list(op, &operands),
                Some(_) => {
                    report_unlowered(op);
                    None
                }
            };

            let Some(lowered) = lowered else {
                continue;
            };

            rewriter.set_insertion_point_before(op);
            let inserted = rewriter.insert(lowered);
            rewriter.replace_all_op_uses_with_operation(op, inserted);
            rewriter.erase_op(op);
        }

        produced
    }

    fn convert_call(
        &self,
        op: OperationRef<'c, '_>,
        call: CallOp<'c, '_>,
        operands: &[Value<'c, '_>],
    ) -> Option<Operation<'c>> {
        let callee = call.callee().value();
        let ty = op.first_result().r#type();
        let kind = call.callee_source();
        if matches!(kind, Some(CalleeSource::Fn | CalleeSource::Const)) {
            report(op, &format!("`{callee}` was not expanded before lowering"));
            return None;
        }

        // The engine knows an external by its own name, not by the symbol.
        let external = (kind == Some(CalleeSource::External)).then(|| {
            *self
                .externals
                .get(callee)
                .expect("an external call names a top-level external fn")
        });
        Some(if call.is_agg() {
            self.convert_measure(op, external.unwrap_or(callee), operands, ty)
        } else if let Some(name) = external {
            yz::extern_call(
                self.context,
                ty,
                operands,
                StringAttribute::new(self.context, name),
                op.location(),
            )
            .into()
        } else {
            yz::call(
                self.context,
                ty,
                operands,
                FlatSymbolRefAttribute::new(self.context, callee),
                op.location(),
            )
            .into()
        })
    }

    fn convert_list(
        &self,
        op: OperationRef<'c, '_>,
        operands: &[Value<'c, '_>],
    ) -> Option<Operation<'c>> {
        let ty = op.first_result().r#type();
        if ListType::from_type(ty).is_none() {
            report(op, "the type of this list could not be inferred");
            return None;
        }

        Some(yz::list(self.context, ty, operands, op.location()).into())
    }

    pub(super) fn record_externals(&mut self, block: BlockRef<'c, '_>) {
        for op in block.operations() {
            if let Some(YzlOp::Fn(function)) = op.as_yzl()
                && let Some(name) = function.external_name()
            {
                self.externals
                    .insert(function.sym_name().value(), name.value());
            }
        }
    }

    fn convert_measure(
        &self,
        op: OperationRef<'c, '_>,
        callee: &str,
        operands: &[Value<'c, '_>],
        ty: Type<'c>,
    ) -> Operation<'c> {
        yzr::agg(
            self.context,
            ty,
            operands,
            StringAttribute::new(self.context, callee),
            op.location(),
        )
        .into()
    }
}

fn rebuild<'c, 'b>(
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

/// The region is rebuilt from the top and is `IsolatedFromAbove`, so an
/// operand with nothing standing for it means its producer failed and has
/// already reported.
fn lowered_operands<'c, 'b>(
    op: OperationRef<'c, '_>,
    values: &FxHashMap<ValueId, Value<'c, 'b>>,
) -> Option<Vec<Value<'c, 'b>>> {
    op.operands()
        .map(|operand| values.get(&operand.id()).copied())
        .collect()
}

/// An op no expression should still hold when the region is lowered.
fn report_unlowered(op: OperationRef<'_, '_>) {
    match op.as_yzl() {
        Some(YzlOp::Missing(_)) => report(op, "this part of the query is missing"),
        Some(YzlOp::Local(_) | YzlOp::Load(_) | YzlOp::Store(_)) => report(
            op,
            &format!(
                "`{}` was not promoted to a value before lowering",
                op_name(op)
            ),
        ),
        _ => report(op, &format!("`{}` is not lowered yet", op_name(op))),
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::test_support::check_yzr;

    #[test]
    fn measures_become_aggregate_ops() {
        check_yzr(
            r"
struct Row { a: int64, b: int64 }
table t = Row

from t
|> aggregate sum(a) as total, count() as n group by b
",
            &expect![[r#"
                module {
                  yz.struct @Row ["a", "b"] : [!yz.int64, !yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  yz.struct @row ["b", "total", "n"] : [!yz.int64, !yz.int64, !yz.int64]
                  %1 = yzr.aggregate %0 keys [1] {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64):
                    %2 = yzr.agg "sum"(%arg0) : (!yz.int64) -> !yz.int64
                    %3 = yzr.agg "count"() : () -> !yz.int64
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
            r"
struct Row { a: int64 }
table t = Row

from t
|> where a >
",
            &expect![[r"
                error: expected expression, found end of input
                 --> test.yz:6:13
                  |
                6 | |> where a >
                  |             ^

                error: this part of the query is missing
                 --> test.yz:6:10
                  |
                6 | |> where a >
                  |          ^^^
            "]],
        );
    }

    #[test]
    fn a_list_lowers_with_its_type() {
        check_yzr(
            r"
struct Row { a: int64 }
table t = Row

from t
|> where a in [1, 3]
",
            &expect![[r#"
                module {
                  yz.struct @Row ["a"] : [!yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  %1 = yzr.filter %0 : !yz.struct<@Row> {
                  ^bb0(%arg0: !yz.int64):
                    %2 = yz.constant_int 1
                    %3 = yz.constant_int 3
                    %4 = yz.list[%2, %3] : (!yz.int64, !yz.int64) -> !yz.list<!yz.int64>
                    %5 = yz.in %arg0, %4 : !yz.int64, !yz.list<!yz.int64> -> !yz.bool
                    yzr.yield %5 : !yz.bool
                  }
                  yzr.output %1 : !yz.struct<@Row>
                }
            "#]],
        );
    }
}
