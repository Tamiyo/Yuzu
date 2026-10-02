//! The operators the library implements. Each primitive op an operator
//! lowers to is tied to a `yuzu.std.ops` function. The op folds first, and
//! `legalize_operators` puts that function's body in place of what is left.
//! The body picks the engine's own function.

use melior::ir::operation::OperationRef;
use yuzu_mlir::attributes::CmpPredicate;
use yuzu_mlir::ir::operation::OperationCast;
use yuzu_mlir::ops::yz::YzOp;

/// A primitive op's operator, and the library function that implements it.
pub(crate) struct Operator {
    /// How a program writes it, such as `%`.
    pub(crate) spelling: &'static str,
    /// The library module that declares the implementation.
    pub(crate) module: &'static str,
    /// The implementation's name in that module.
    pub(crate) name: &'static str,
}

/// `+`.
pub(crate) const ADD: Operator = Operator {
    spelling: "+",
    module: "yuzu.std.ops",
    name: "add",
};

/// `-`.
pub(crate) const SUB: Operator = Operator {
    spelling: "-",
    module: "yuzu.std.ops",
    name: "sub",
};

/// `*`.
pub(crate) const MUL: Operator = Operator {
    spelling: "*",
    module: "yuzu.std.ops",
    name: "mul",
};

/// `/`.
pub(crate) const DIV: Operator = Operator {
    spelling: "/",
    module: "yuzu.std.ops",
    name: "div",
};

/// `and`.
pub(crate) const AND: Operator = Operator {
    spelling: "and",
    module: "yuzu.std.ops",
    name: "and",
};

/// `or`.
pub(crate) const OR: Operator = Operator {
    spelling: "or",
    module: "yuzu.std.ops",
    name: "or",
};

/// `%`.
pub(crate) const REM: Operator = Operator {
    spelling: "%",
    module: "yuzu.std.ops",
    name: "modulo",
};

/// `**`.
pub(crate) const POW: Operator = Operator {
    spelling: "**",
    module: "yuzu.std.ops",
    name: "pow",
};

/// `<<`.
pub(crate) const SHL: Operator = Operator {
    spelling: "<<",
    module: "yuzu.std.ops",
    name: "shift_left",
};

/// `>>`.
pub(crate) const SHR: Operator = Operator {
    spelling: ">>",
    module: "yuzu.std.ops",
    name: "shift_right",
};

/// `not`.
pub(crate) const NOT: Operator = Operator {
    spelling: "not",
    module: "yuzu.std.ops",
    name: "not",
};

/// `in`.
pub(crate) const IN: Operator = Operator {
    spelling: "in",
    module: "yuzu.std.ops",
    name: "in",
};

/// Unary `-`.
pub(crate) const NEG: Operator = Operator {
    spelling: "-",
    module: "yuzu.std.ops",
    name: "neg",
};

/// `==`.
pub(crate) const EQ: Operator = Operator {
    spelling: "==",
    module: "yuzu.std.ops",
    name: "eq",
};

/// `!=`.
pub(crate) const NE: Operator = Operator {
    spelling: "!=",
    module: "yuzu.std.ops",
    name: "ne",
};

/// `<`.
pub(crate) const LT: Operator = Operator {
    spelling: "<",
    module: "yuzu.std.ops",
    name: "lt",
};

/// `<=`.
pub(crate) const LE: Operator = Operator {
    spelling: "<=",
    module: "yuzu.std.ops",
    name: "le",
};

/// `>`.
pub(crate) const GT: Operator = Operator {
    spelling: ">",
    module: "yuzu.std.ops",
    name: "gt",
};

/// `>=`.
pub(crate) const GE: Operator = Operator {
    spelling: ">=",
    module: "yuzu.std.ops",
    name: "ge",
};

/// Every operator the library implements. `in` is not one: it becomes a
/// Substrait list expression, not a call.
pub(crate) static OPERATORS: [Operator; 18] = [
    ADD, SUB, MUL, DIV, NEG, REM, POW, SHL, SHR, EQ, NE, LT, LE, GT, GE, AND, OR, NOT,
];

impl Operator {
    /// The operator an op is, when it is one the library implements.
    pub(crate) fn of(op: OperationRef) -> Option<&'static Self> {
        Some(match op.as_yz()? {
            YzOp::Add(_) => &ADD,
            YzOp::Sub(_) => &SUB,
            YzOp::Mul(_) => &MUL,
            YzOp::Div(_) => &DIV,
            YzOp::Neg(_) => &NEG,
            YzOp::Rem(_) => &REM,
            YzOp::Pow(_) => &POW,
            YzOp::Shl(_) => &SHL,
            YzOp::Shr(_) => &SHR,
            YzOp::Cmp(compare) => match compare.predicate() {
                CmpPredicate::Equal => &EQ,
                CmpPredicate::NotEqual => &NE,
                CmpPredicate::Less => &LT,
                CmpPredicate::LessOrEqual => &LE,
                CmpPredicate::Greater => &GT,
                CmpPredicate::GreaterOrEqual => &GE,
            },
            YzOp::And(_) => &AND,
            YzOp::Or(_) => &OR,
            YzOp::Not(_) => &NOT,
            _ => return None,
        })
    }

    /// The operator a symbol implements, when it implements one.
    pub(crate) fn implemented_by(symbol: &str) -> Option<&'static Self> {
        OPERATORS
            .iter()
            .find(|operator| operator.is_implemented_by(symbol))
    }

    /// The symbol of the function that implements the operator.
    pub(crate) fn to_symbol(&self) -> String {
        format!("{}.{}", self.module, self.name)
    }

    fn is_implemented_by(&self, symbol: &str) -> bool {
        symbol
            .strip_prefix(self.module)
            .and_then(|rest| rest.strip_prefix('.'))
            == Some(self.name)
    }
}

#[cfg(test)]
mod tests {
    use super::{OPERATORS, Operator};

    #[test]
    fn an_implementation_is_found_by_its_symbol() {
        let rem = Operator::implemented_by("yuzu.std.ops.modulo").expect("modulo implements `%`");
        assert_eq!(rem.spelling, "%");
        assert!(Operator::implemented_by("yuzu.std.opsmodulo").is_none());
        assert!(Operator::implemented_by("modulo").is_none());
        for operator in &OPERATORS {
            let found = Operator::implemented_by(&operator.to_symbol());
            assert!(found.is_some_and(|found| found.spelling == operator.spelling));
        }
    }
}
