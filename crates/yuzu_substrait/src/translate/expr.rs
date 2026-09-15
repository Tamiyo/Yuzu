//! The expressions inside a stage's region. A block argument is a column of
//! the row the region sees, by position; everything else is an operation
//! whose operands were translated before it, the region being in order.

use std::collections::HashMap;

use melior::ir::operation::{OperationLike, OperationRef};
use melior::ir::{BlockLike, RegionLike, Value, ValueLike};
use substrait::proto::{
    Expression, FunctionArgument,
    expression::{
        FieldReference, Literal, ReferenceSegment, RexType, ScalarFunction, SingularOrList,
        field_reference::{ReferenceType, RootReference, RootType},
        literal::LiteralType,
        reference_segment,
    },
    function_argument::ArgType,
};
use yuzu_mlir::ext::{BlockExt, OperationCast, OperationExt, ValueExt};
use yuzu_mlir::ops::yz::YzOp;
use yuzu_mlir::ops::yzr::YzrOp;
use yuzu_types::Func;

use crate::extensions::{EXTERNAL_URN, function_target};
use crate::translate::Translator;
use crate::translate::functions;
use crate::translate::types::{emit_type, type_code};

pub(crate) fn selection(index: i32) -> Expression {
    Expression {
        rex_type: Some(RexType::Selection(Box::new(FieldReference {
            reference_type: Some(ReferenceType::DirectReference(ReferenceSegment {
                reference_type: Some(reference_segment::ReferenceType::StructField(Box::new(
                    reference_segment::StructField {
                        field: index,
                        child: None,
                    },
                ))),
            })),
            root_type: Some(RootType::RootReference(RootReference {})),
        }))),
    }
}

pub(crate) fn literal(value: LiteralType) -> Expression {
    Expression {
        rex_type: Some(RexType::Literal(Literal {
            literal_type: Some(value),
            ..Default::default()
        })),
    }
}

/// A translated region: what each of its values became, and the values its
/// `yzr.yield` named. The yields stay as values because what they mean is
/// the stage's business — a grouping yields measures, not expressions.
pub(crate) struct Region<'c, 'a> {
    pub(crate) values: HashMap<usize, Expression>,
    pub(crate) yielded: Vec<Value<'c, 'a>>,
}

impl<'c, 'a> Translator<'c, 'a, '_> {
    pub(crate) fn translate_region(&mut self, op: OperationRef<'c, 'a>) -> Option<Region<'c, 'a>> {
        let Some(block) = op.regions().next().and_then(|region| region.first_block()) else {
            return Some(Region {
                values: HashMap::new(),
                yielded: Vec::new(),
            });
        };

        let mut values = HashMap::new();
        for index in 0..block.argument_count() {
            let argument = block
                .argument(index)
                .expect("the argument index is in range");
            values.insert(argument.id(), selection(index as i32));
        }

        let mut yielded = Vec::new();
        for inner in block.operations() {
            // A measure belongs to the grouping that holds it, and a list
            // only ever stands to the right of a membership test, which
            // reads its elements where they are.
            let skip = matches!(inner.as_yzr(), Some(YzrOp::Agg(_) | YzrOp::Count(_)))
                || matches!(inner.as_yz(), Some(YzOp::List(_)));
            if matches!(inner.as_yzr(), Some(YzrOp::Yield(_))) {
                yielded.extend(inner.operands());
            } else if !skip {
                let expression = self.translate_value(inner, &values)?;
                values.insert(inner.first_result().id(), expression);
            }
        }

        Some(Region { values, yielded })
    }

    /// The expressions a region yielded, for a stage that wants values.
    pub(crate) fn yielded(
        &self,
        op: OperationRef<'c, '_>,
        region: &Region<'c, 'a>,
    ) -> Option<Vec<Expression>> {
        region
            .yielded
            .iter()
            .map(|&value| self.expression_of(op, value, &region.values))
            .collect()
    }

    /// What a value became, which the operation producing it recorded before
    /// this one was reached.
    pub(crate) fn expression_of(
        &self,
        op: OperationRef<'c, '_>,
        value: Value<'c, '_>,
        values: &HashMap<usize, Expression>,
    ) -> Option<Expression> {
        match values.get(&value.id()) {
            Some(expression) => Some(expression.clone()),
            None => {
                self.unsupported(op, "this expression has no Substrait equivalent");
                None
            }
        }
    }

    fn translate_value(
        &mut self,
        op: OperationRef<'c, '_>,
        values: &HashMap<usize, Expression>,
    ) -> Option<Expression> {
        match op.as_yz()? {
            YzOp::ConstantInt(constant) => {
                Some(literal(LiteralType::I64(constant.value().value())))
            }
            YzOp::ConstantFloat(constant) => {
                Some(literal(LiteralType::Fp64(constant.value().value())))
            }
            // A bare `BoolAttr` has no typed reader of its own.
            YzOp::ConstantBool(constant) => Some(literal(LiteralType::Boolean(
                constant.value().to_string() == "true",
            ))),
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
            YzOp::Call(call) => {
                let callee = call.callee().value();
                let Some(func) = functions::of_builtin(callee) else {
                    self.unsupported(op, format!("`{callee}` has no Substrait mapping yet"));
                    return None;
                };

                match func {
                    Func::In => self.translate_membership(op, values),
                    func => self.translate_call(op, func, values),
                }
            }
            YzOp::ExternCall(call) => {
                let callee = call.callee().value().to_string();
                self.translate_function(op, EXTERNAL_URN, callee, values)
            }
            YzOp::Rem(_) => {
                self.unsupported(op, "`%` is not supported by the datafusion target");
                None
            }
            // Declarations and terminators are not values.
            YzOp::Struct(_) | YzOp::Func(_) | YzOp::Return(_) | YzOp::List(_) => {
                self.unsupported(op, "this is not an expression");
                None
            }
        }
    }

    /// `x in [a, b]` is a `SingularOrList`, not a call: Substrait spells
    /// membership as a value and its options, and the list only holds them.
    fn translate_membership(
        &mut self,
        op: OperationRef<'c, '_>,
        values: &HashMap<usize, Expression>,
    ) -> Option<Expression> {
        let value = self.expression_of(op, op.operand(0).ok()?, values)?;
        let list = op.operand(1).ok()?;
        let Some(producer) = Self::producer(list) else {
            self.unsupported(op, "`in` takes a list of values on its right");
            return None;
        };

        let options = producer
            .operands()
            .map(|option| self.expression_of(op, option, values))
            .collect::<Option<Vec<_>>>()?;

        Some(Expression {
            rex_type: Some(RexType::SingularOrList(Box::new(SingularOrList {
                value: Some(Box::new(value)),
                options,
            }))),
        })
    }

    fn translate_call(
        &mut self,
        op: OperationRef<'c, '_>,
        func: Func,
        values: &HashMap<usize, Expression>,
    ) -> Option<Expression> {
        let Some((urn, base)) = function_target(func) else {
            let symbol = func.symbol();
            self.unsupported(
                op,
                format!("`{symbol}` is not supported by the datafusion target"),
            );
            return None;
        };

        self.translate_function(op, urn, base.to_string(), values)
    }

    /// A call, with the signature Substrait names its overload by: the
    /// argument type codes, joined.
    fn translate_function(
        &mut self,
        op: OperationRef<'c, '_>,
        urn: &'static str,
        base: String,
        values: &HashMap<usize, Expression>,
    ) -> Option<Expression> {
        let mut signature = Vec::new();
        let mut arguments = Vec::new();
        for operand in op.operands() {
            let Some(code) = type_code(self.context, operand.r#type()) else {
                self.unsupported(op, "this argument has no Substrait type");
                return None;
            };

            signature.push(code);
            arguments.push(FunctionArgument {
                arg_type: Some(ArgType::Value(self.expression_of(op, operand, values)?)),
            });
        }

        let Some(output) = emit_type(self.context, op.first_result().r#type()) else {
            self.unsupported(op, "this expression has no Substrait type");
            return None;
        };

        let anchor = self
            .extensions
            .register(urn, format!("{base}:{}", signature.join("_")));
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
