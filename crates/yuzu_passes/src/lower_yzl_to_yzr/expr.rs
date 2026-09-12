//! The expressions inside a stage's region. A `yz` op carries over with
//! the type inference gave it; a call becomes a measure, an external
//! call, or a plain one, depending on what resolution decided it names.
use std::collections::HashMap;

use melior::ir::attribute::{FlatSymbolRefAttribute, StringAttribute, TypeAttribute};
use melior::ir::operation::{OperationBuilder, OperationLike, OperationRef};
use melior::ir::{Attribute, BlockLike, BlockRef, Identifier, Type, Value, ValueLike};
use yuzu_mlir::ext::OperationExt;
use yuzu_mlir::ods::{yz, yzr};
use yuzu_mlir::ops::yzl::YzlOperationRef;
use yuzu_mlir::{CalleeKind, value_id};
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
        match YzlOperationRef::of(&op) {
            Some(YzlOperationRef::Name(name)) => {
                let Some(index) = name.col().map(|col| col.value() as usize) else {
                    self.error(op, "a name outside a column context is not lowered yet");
                    return;
                };

                let Ok(column) = body.argument(index) else {
                    self.error(op, format!("column {index} is not in the row"));
                    return;
                };

                values.insert(value_id(op.first_result()), column.into());
            }
            Some(YzlOperationRef::Yield(_)) => {
                produced.extend(self.mapped_operands(op, values));
            }
            Some(YzlOperationRef::Call(call)) => {
                let callee = call.callee().value().to_string();
                let operands = self.mapped_operands(op, values);
                let ty = self.stamped_type(op);
                let kind = call.callee_kind();
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
                values.insert(value_id(op.first_result()), appended.first_result());
            }
            // Everything else is a `yz` op, structurally unchanged: the
            // operands it was given, and the type inference stamped on it.
            None => {
                let operands = self.mapped_operands(op, values);
                let rebuilt = self.rebuild(op, &operands, body);
                if let Some(rebuilt) = rebuilt {
                    values.insert(value_id(op.first_result()), rebuilt);
                }
            }
            // The parse error above it already said what went wrong; this says
            // the query cannot be built from what is left, rather than
            // implying some lowering is still to come.
            Some(YzlOperationRef::Missing(_)) => {
                self.error(op, "this part of the query is missing")
            }
            Some(
                YzlOperationRef::List(_)
                | YzlOperationRef::From(_)
                | YzlOperationRef::Where(_)
                | YzlOperationRef::Select(_)
                | YzlOperationRef::Extend(_)
                | YzlOperationRef::Aggregate(_)
                | YzlOperationRef::Limit(_)
                | YzlOperationRef::Join(_)
                | YzlOperationRef::Rename(_)
                | YzlOperationRef::Alias(_)
                | YzlOperationRef::Distinct(_)
                | YzlOperationRef::Drop(_)
                | YzlOperationRef::Set(_)
                | YzlOperationRef::Output(_)
                | YzlOperationRef::Struct(_)
                | YzlOperationRef::Table(_)
                | YzlOperationRef::Fn(_)
                | YzlOperationRef::Trait(_)
                | YzlOperationRef::Impl(_)
                | YzlOperationRef::Let(_)
                | YzlOperationRef::Return(_),
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
        let existing = op.try_first_result()?;

        let ty = op
            .attribute("ty")
            .ok()
            .and_then(|attribute| TypeAttribute::try_from(attribute).ok())
            .map(|attribute| attribute.value())
            .unwrap_or_else(|| existing.r#type());

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

    fn mapped_operands<'b>(
        &mut self,
        op: OperationRef<'c, '_>,
        values: &HashMap<usize, Value<'c, 'b>>,
    ) -> Vec<Value<'c, 'b>> {
        op.operands()
            .filter_map(|operand| values.get(&value_id(operand)).copied())
            .collect()
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

    /// The type inference stamped, or the one the op already carries.
    fn stamped_type(&self, op: OperationRef<'c, '_>) -> Type<'c> {
        op.attribute("ty")
            .ok()
            .and_then(|attribute| TypeAttribute::try_from(attribute).ok())
            .map(|attribute| attribute.value())
            .unwrap_or_else(|| op.first_result().r#type())
    }
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
}
