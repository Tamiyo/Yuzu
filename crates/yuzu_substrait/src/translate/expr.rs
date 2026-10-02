//! The expressions inside a stage's region. A block argument is a column of
//! the row the region sees, by position; everything else is an operation
//! whose operands were translated before it, the region being in order.

use melior::ir::attribute::{BoolAttribute, FloatAttribute, IntegerAttribute, StringAttribute};
use melior::ir::operation::{OperationLike, OperationRef};
use melior::ir::{Attribute, RegionLike, Value, ValueLike};
use rustc_hash::FxHashMap;
use substrait::proto::{
    Expression, FunctionArgument, Type,
    expression::{RexType, ScalarFunction, SingularOrList, literal::LiteralType},
    function_argument::ArgType,
};
use yuzu_mlir::ir::attribute::array::ArrayAttributeExt;
use yuzu_mlir::ir::block::BlockExt;
use yuzu_mlir::ir::operation::{OperationCast, OperationExt};
use yuzu_mlir::ir::value::{ValueExt, ValueId};
use yuzu_mlir::ops::yz::YzOp;
use yuzu_mlir::ops::yzr::YzrOp;

use crate::extensions::{COMPARISON_URN, EXTERNAL_URN, standard_urn};
use crate::proto::{field_index, literal, selection};
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
                || matches!(inner.as_yz(), Some(YzOp::List(_) | YzOp::ConstantList(_)));
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
        let Some(yz) = op.as_yz() else {
            report(op, "this has no Substrait equivalent");
            return None;
        };
        match yz {
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
            // A full join `using` a column makes this op, not an operator.
            YzOp::Coalesce(_) => self.translate_function(op, COMPARISON_URN, "coalesce", values),
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
            YzOp::Add(_)
            | YzOp::Sub(_)
            | YzOp::Mul(_)
            | YzOp::Div(_)
            | YzOp::Neg(_)
            | YzOp::Rem(_)
            | YzOp::Pow(_)
            | YzOp::Shl(_)
            | YzOp::Shr(_)
            | YzOp::Cmp(_)
            | YzOp::And(_)
            | YzOp::Or(_)
            | YzOp::Not(_) => {
                report(
                    op,
                    "an operator reached the translation without an implementation",
                );
                None
            }
            // Declarations and terminators are not values.
            YzOp::Struct(_)
            | YzOp::Func(_)
            | YzOp::Return(_)
            | YzOp::List(_)
            | YzOp::ConstantList(_) => {
                report(op, "this is not an expression");
                None
            }
        }
    }

    /// A call of the function named `base` under `urn`.
    fn translate_function(
        &mut self,
        op: OperationRef<'c, '_>,
        urn: &'static str,
        base: &str,
        values: &FxHashMap<ValueId, Expression>,
    ) -> Option<Expression> {
        let operands: Vec<Value<'c, '_>> = op.operands().collect();
        let Application {
            anchor,
            arguments,
            output,
        } = self.translate_arguments(op, urn, base, &operands, values)?;
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

/// A function applied to its arguments, as a call or a measure writes it.
pub(crate) struct Application {
    /// The anchor the function is declared under.
    pub(crate) anchor: u32,
    pub(crate) arguments: Vec<FunctionArgument>,
    pub(crate) output: Type,
}

impl<'c> Translator<'c, '_, '_> {
    /// A function applied to its arguments. Substrait names an overload by
    /// its argument type codes, joined.
    pub(crate) fn translate_arguments(
        &mut self,
        op: OperationRef<'c, '_>,
        urn: &'static str,
        base: &str,
        arguments: &[Value<'c, '_>],
        values: &FxHashMap<ValueId, Expression>,
    ) -> Option<Application> {
        let mut signature = Vec::with_capacity(arguments.len());
        let mut emitted = Vec::with_capacity(arguments.len());
        for &argument in arguments {
            let Some(code) = type_code(argument.r#type()) else {
                report(op, "this argument has no Substrait type");
                return None;
            };

            signature.push(code);
            emitted.push(FunctionArgument {
                arg_type: Some(ArgType::Value(expression_of(op, argument, values)?)),
            });
        }

        let Some(output) = emit_type(op.first_result().r#type()) else {
            report(op, "this has no Substrait type");
            return None;
        };

        // An engine's function that the standard catalogue declares for these
        // types goes under the catalogue's URN, so any engine can find it.
        let urn = if urn == EXTERNAL_URN {
            standard_urn(base, &signature).unwrap_or(EXTERNAL_URN)
        } else {
            urn
        };
        let anchor = self
            .extensions
            .register(urn, format!("{base}:{}", signature.join("_")));
        Some(Application {
            anchor,
            arguments: emitted,
            output,
        })
    }
}

/// What a value became. The operation that produces the value records it
/// before the translation reaches `op`.
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
    let producer = Translator::producer(list);
    let options = match producer.as_ref().and_then(OperationCast::as_yz) {
        Some(YzOp::ConstantList(constants)) => constants
            .values()
            .elements()
            .map(constant_literal)
            .collect(),
        Some(YzOp::List(elements)) => elements
            .operation()
            .operands()
            .map(|option| expression_of(op, option, values))
            .collect::<Option<Vec<_>>>()?,
        _ => {
            report(op, "`in` takes a list of values on its right");
            return None;
        }
    };

    Some(Expression {
        rex_type: Some(RexType::SingularOrList(Box::new(SingularOrList {
            value: Some(Box::new(value)),
            options,
        }))),
    })
}

/// A value of a `yz.constant_list` as a literal. Its verifier holds each
/// value to the constant of the element type.
fn constant_literal(value: Attribute<'_>) -> Expression {
    // A bool is an integer attribute of one bit, so it is asked first.
    if let Ok(boolean) = BoolAttribute::try_from(value) {
        return literal(LiteralType::Boolean(boolean.value()));
    }
    if let Ok(integer) = IntegerAttribute::try_from(value) {
        return literal(LiteralType::I64(integer.value()));
    }
    if let Ok(real) = FloatAttribute::try_from(value) {
        return literal(LiteralType::Fp64(real.value()));
    }
    if let Ok(text) = StringAttribute::try_from(value) {
        return literal(LiteralType::String(text.value().to_owned()));
    }
    unreachable!("a verified `yz.constant_list` holds only constants, not {value}")
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
