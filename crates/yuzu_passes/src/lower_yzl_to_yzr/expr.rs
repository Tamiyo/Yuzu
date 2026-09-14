//! The expressions inside a stage's region. A `yz` op carries over with
//! the type inference gave it; a call becomes a measure, an external
//! call, or a plain one, depending on what resolution decided it names.
use std::collections::HashMap;

use melior::ir::attribute::{FlatSymbolRefAttribute, StringAttribute};
use melior::ir::operation::{OperationBuilder, OperationLike, OperationRef};
use melior::ir::{Attribute, BlockLike, BlockRef, Identifier, Type, Value};
use yuzu_mlir::ListType;
use yuzu_mlir::attributes::CalleeKind;
use yuzu_mlir::ext::{OperationCast, OperationExt, ValueExt};
use yuzu_mlir::ods::{yz, yzr};
use yuzu_mlir::ops::yzl::YzlOp;
use yuzu_types::BuiltinFunc;

use crate::lower_yzl_to_yzr::{YzlToYzr, op_name};

impl<'c, 'a> YzlToYzr<'c, 'a, '_> {
    /// An expression op, rebuilt against the values its operands became.
    pub(super) fn lower_expression<'b>(
        &mut self,
        op: OperationRef<'c, '_>,
        body: BlockRef<'c, 'b>,
        values: &mut HashMap<usize, Value<'c, 'b>>,
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
                let callee = call.callee().value().to_string();
                let Some(operands) = lowered_operands(op, values) else {
                    return;
                };

                let ty = op.ty();
                let kind = call.callee_kind();
                // Expansion removes every call to a function or let; one
                // reaching here means expansion did not finish, which it has
                // already reported.
                if matches!(
                    kind,
                    Some(CalleeKind::Fn | CalleeKind::AggFn | CalleeKind::Let)
                ) {
                    self.error(op, format!("`{callee}` was not expanded before lowering"));
                    return;
                }

                let lowered = if kind == Some(CalleeKind::Builtin) && self.is_aggregate(&callee) {
                    self.lower_measure(op, &callee, &operands, ty, body)
                } else if kind == Some(CalleeKind::External) {
                    yz::extern_call(
                        self.context,
                        ty,
                        &operands,
                        StringAttribute::new(self.context, &callee),
                        op.location(),
                    )
                    .into()
                } else {
                    yz::call(
                        self.context,
                        ty,
                        &operands,
                        FlatSymbolRefAttribute::new(self.context, &callee),
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

                let ty = op.ty();
                if ListType::from_type(ty).is_none() {
                    self.error(op, "the type of this list could not be inferred");
                    return;
                }

                let appended = body
                    .append_operation(yz::list(self.context, ty, &operands, op.location()).into());
                values.insert(op.first_result().id(), appended.first_result());
            }
            // Everything else is a `yz` op, structurally unchanged: the
            // operands it was given, and the type inference stamped on it.
            None => {
                let Some(operands) = lowered_operands(op, values) else {
                    return;
                };

                let rebuilt = self.rebuild(op, &operands, body);
                if let Some(rebuilt) = rebuilt {
                    values.insert(op.first_result().id(), rebuilt);
                }
            }
            // The parse error above it already said what went wrong; this says
            // the query cannot be built from what is left, rather than
            // implying some lowering is still to come.
            Some(YzlOp::Missing(_)) => self.error(op, "this part of the query is missing"),
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
                | YzlOp::Let(_)
                | YzlOp::Return(_),
            ) => self.error(op, format!("`{}` is not lowered yet", op_name(op))),
        }
    }

    /// A `yz` op carried over with its attributes, its mapped operands, and
    /// the concrete type inference gave it.
    fn rebuild<'b>(
        &mut self,
        op: OperationRef<'c, '_>,
        operands: &[Value<'c, 'b>],
        body: BlockRef<'c, 'b>,
    ) -> Option<Value<'c, 'b>> {
        let name = op.name();
        op.try_first_result()?;

        let ty = op.ty();
        let attributes: Vec<(Identifier<'c>, Attribute<'c>)> = (0..op.attribute_count())
            .map(|index| {
                op.attribute_at(index)
                    .expect("the attribute index is in range")
            })
            .filter(|(name, _)| name.as_string_ref().as_str() != Ok("ty"))
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
    fn lower_measure(
        &self,
        op: OperationRef<'c, '_>,
        callee: &str,
        operands: &[Value<'c, '_>],
        ty: Type<'c>,
        _body: BlockRef<'c, '_>,
    ) -> melior::ir::Operation<'c> {
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

    fn is_aggregate(&self, callee: &str) -> bool {
        self.registry
            .entries()
            .iter()
            .any(|entry| entry.name == callee && matches!(entry.func, BuiltinFunc::Aggregate(_)))
    }
}

/// What an op's operands became. The region is rebuilt from the top, so every
/// operand has been lowered by the time its user is reached, and the stage
/// regions are `IsolatedFromAbove` so none can come from outside.
///
/// An operand with nothing to stand for it therefore means its producer
/// failed, and every path that fails to record a value reports first — so
/// this says nothing, it just declines to build the op. Taking the operands
/// that happen to be there would build one of the wrong shape and carry the
/// mistake into the plan.
fn lowered_operands<'c, 'b>(
    op: OperationRef<'c, '_>,
    values: &HashMap<usize, Value<'c, 'b>>,
) -> Option<Vec<Value<'c, 'b>>> {
    op.operands()
        .map(|operand| values.get(&operand.id()).copied())
        .collect()
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::test_support::check_lowered;

    /// Measures become `yzr.agg`, and the keys come from the stamp
    /// resolution left.
    #[test]
    fn measures_become_aggregate_ops() {
        check_lowered(
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

    /// A hole the parser left behind reaches here as a `yzl.missing`. The
    /// parse error above it says what went wrong; this says the query cannot
    /// be built from it, rather than implying a lowering is missing.
    #[test]
    fn reports_a_missing_piece() {
        check_lowered(
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
                  |          ^
            "#]],
        );
    }

    #[test]
    fn a_list_lowers_with_its_type() {
        check_lowered(
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
