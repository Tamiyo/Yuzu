//! The expressions inside a stage's region. A block argument is a column of
//! the row the region sees, by position; everything else is an operation
//! whose operands were translated before it, the region being in order.

use melior::ir::operation::{OperationLike, OperationRef};
use melior::ir::{RegionLike, Value, ValueLike};
use rustc_hash::FxHashMap;
use substrait::proto::{
    Expression, FunctionArgument, Type,
    expression::{RexType, ScalarFunction, SingularOrList, literal::LiteralType},
    function_argument::ArgType,
};
use yuzu_mlir::ir::block::BlockExt;
use yuzu_mlir::ir::operation::{OperationCast, OperationExt};
use yuzu_mlir::ir::value::{ValueExt, ValueId};
use yuzu_mlir::ops::yz::YzOp;
use yuzu_mlir::ops::yzr::YzrOp;

use crate::extensions::{EXTERNAL_URN, Func, function_target};
use crate::proto::{field_index, literal, selection};
use crate::translate::functions;
use crate::translate::types::{emit_type, type_code};
use crate::translate::{Translator, report};

/// A translated region: what each of its values became, and the values its
/// `yzr.yield` named. The yields stay as values because what they mean is
/// the stage's business — a grouping yields measures, not expressions.
pub(crate) struct Region<'c, 'a> {
    /// The stage the region belongs to.
    pub(crate) op: OperationRef<'c, 'a>,
    pub(crate) values: FxHashMap<ValueId, Expression>,
    pub(crate) yielded: Vec<Value<'c, 'a>>,
}

impl<'c, 'a> Translator<'c, 'a, '_> {
    pub(crate) fn translate_region(&mut self, op: OperationRef<'c, 'a>) -> Option<Region<'c, 'a>> {
        let Some(block) = op.regions().next().and_then(|region| region.first_block()) else {
            return Some(Region {
                op,
                values: FxHashMap::default(),
                yielded: Vec::new(),
            });
        };

        let mut values: FxHashMap<ValueId, Expression> = block
            .arguments()
            .enumerate()
            .map(|(index, argument)| (argument.id(), selection(field_index(index))))
            .collect();

        let mut yielded = Vec::new();
        for inner in block.operations() {
            // A measure belongs to the grouping that holds it, and a list
            // only ever stands to the right of a membership test, which
            // reads its elements where they are.
            let skip = matches!(inner.as_yzr(), Some(YzrOp::Agg(_)))
                || matches!(inner.as_yz(), Some(YzOp::List(_)));
            if matches!(inner.as_yzr(), Some(YzrOp::Yield(_))) {
                yielded.extend(inner.operands());
            } else if !skip {
                let expression = self.translate_value(inner, &values)?;
                values.insert(inner.first_result().id(), expression);
            }
        }

        Some(Region {
            op,
            values,
            yielded,
        })
    }

    fn translate_value(
        &mut self,
        op: OperationRef<'c, '_>,
        values: &FxHashMap<ValueId, Expression>,
    ) -> Option<Expression> {
        let Some(value) = op.as_yz() else {
            report(op, "this has no Substrait equivalent");
            return None;
        };
        match value {
            YzOp::ConstantInt(constant) => {
                Some(literal(LiteralType::I64(constant.value().value())))
            }
            YzOp::ConstantFloat(constant) => {
                Some(literal(LiteralType::Fp64(constant.value().value())))
            }
            YzOp::ConstantBool(constant) => {
                Some(literal(LiteralType::Boolean(constant.value().value())))
            }
            YzOp::ConstantStr(constant) => Some(literal(LiteralType::String(
                constant.value().value().to_string(),
            ))),
            YzOp::Add(_) => self.translate_call(op, Func::Add, values),
            YzOp::Sub(_) => self.translate_call(op, Func::Subtract, values),
            YzOp::Mul(_) => self.translate_call(op, Func::Multiply, values),
            YzOp::Div(_) => self.translate_call(op, Func::Divide, values),
            YzOp::Neg(_) => self.translate_call(op, Func::Negate, values),
            YzOp::And(_) => self.translate_call(op, Func::And, values),
            YzOp::Or(_) => self.translate_call(op, Func::Or, values),
            YzOp::Not(_) => self.translate_call(op, Func::Not, values),
            YzOp::Cmp(compare) => {
                self.translate_call(op, functions::of_predicate(compare.predicate()), values)
            }
            YzOp::In(_) => translate_membership(op, values),
            YzOp::Call(call) => {
                let callee = call.callee().value();
                report(op, &format!("`{callee}` has no Substrait mapping yet"));
                None
            }
            YzOp::ExternCall(call) => {
                let callee = call.callee().value();
                self.translate_function(op, EXTERNAL_URN, callee, values)
            }
            // `legalize_operators` puts the library's implementation in its
            // place, and reports when there is none.
            YzOp::Rem(_) | YzOp::Pow(_) | YzOp::Shl(_) | YzOp::Shr(_) => {
                report(
                    op,
                    "an operator reached the translation without an implementation",
                );
                None
            }
            // Declarations and terminators are not values.
            YzOp::Struct(_) | YzOp::Func(_) | YzOp::Return(_) | YzOp::List(_) => {
                report(op, "this is not an expression");
                None
            }
        }
    }

    fn translate_call(
        &mut self,
        op: OperationRef<'c, '_>,
        func: Func,
        values: &FxHashMap<ValueId, Expression>,
    ) -> Option<Expression> {
        let (urn, base) = function_target(func);
        self.translate_function(op, urn, base, values)
    }

    /// A call, with the signature Substrait names its overload by: the
    /// argument type codes, joined.
    fn translate_function(
        &mut self,
        op: OperationRef<'c, '_>,
        urn: &'static str,
        base: &str,
        values: &FxHashMap<ValueId, Expression>,
    ) -> Option<Expression> {
        let operands: Vec<Value<'c, '_>> = op.operands().collect();
        let (anchor, arguments, output) =
            self.translate_arguments(op, urn, base, &operands, values)?;
        Some(Expression {
            rex_type: Some(RexType::ScalarFunction(ScalarFunction {
                function_reference: anchor,
                output_type: Some(output),
                arguments,
                ..Default::default()
            })),
        })
    }
}

impl<'c> Translator<'c, '_, '_> {
    /// A function's arguments, the anchor it is declared under, and its
    /// result type. Substrait names an overload by its argument type codes,
    /// joined.
    pub(crate) fn translate_arguments(
        &mut self,
        op: OperationRef<'c, '_>,
        urn: &'static str,
        base: &str,
        arguments: &[Value<'c, '_>],
        values: &FxHashMap<ValueId, Expression>,
    ) -> Option<(u32, Vec<FunctionArgument>, Type)> {
        let mut signature = Vec::with_capacity(arguments.len());
        let mut emitted = Vec::with_capacity(arguments.len());
        for &argument in arguments {
            let Some(code) = type_code(self.context, argument.r#type()) else {
                report(op, "this argument has no Substrait type");
                return None;
            };

            signature.push(code);
            emitted.push(FunctionArgument {
                arg_type: Some(ArgType::Value(expression_of(op, argument, values)?)),
            });
        }

        let Some(output) = emit_type(self.context, op.first_result().r#type()) else {
            report(op, "this has no Substrait type");
            return None;
        };

        let anchor = self
            .extensions
            .register(urn, format!("{base}:{}", signature.join("_")));
        Some((anchor, emitted, output))
    }
}

/// What a value became, which the operation producing it recorded before
/// this one was reached.
pub(crate) fn expression_of(
    op: OperationRef<'_, '_>,
    value: Value<'_, '_>,
    values: &FxHashMap<ValueId, Expression>,
) -> Option<Expression> {
    if let Some(expression) = values.get(&value.id()) {
        Some(expression.clone())
    } else {
        report(op, "this expression has no Substrait equivalent");
        None
    }
}

/// `x in [a, b]` is a `SingularOrList`, not a call: Substrait spells
/// membership as a value and its options, and the list only holds them.
fn translate_membership(
    op: OperationRef<'_, '_>,
    values: &FxHashMap<ValueId, Expression>,
) -> Option<Expression> {
    let value = op.operand(0).expect("a verified `yz.in` has its value");
    let value = expression_of(op, value, values)?;
    let list = op.operand(1).expect("a verified `yz.in` has its list");
    let producer = Translator::producer(list)
        .filter(|producer| matches!(producer.as_yz(), Some(YzOp::List(_))));
    let Some(producer) = producer else {
        report(op, "`in` takes a list of values on its right");
        return None;
    };

    let options = producer
        .operands()
        .map(|option| expression_of(op, option, values))
        .collect::<Option<Vec<_>>>()?;

    Some(Expression {
        rex_type: Some(RexType::SingularOrList(Box::new(SingularOrList {
            value: Some(Box::new(value)),
            options,
        }))),
    })
}

/// The expressions a region yielded, for a stage that wants values.
pub(crate) fn yielded(
    op: OperationRef<'_, '_>,
    region: &Region<'_, '_>,
) -> Option<Vec<Expression>> {
    region
        .yielded
        .iter()
        .map(|&value| expression_of(op, value, &region.values))
        .collect()
}
