//! The operators the library implements. An engine takes no `yz.rem`, so
//! each such primitive op is tied to the library function that says how the
//! engine computes it: the op folds first, and `legalize_operators` puts
//! that function's body in place of what is left.

use melior::ir::operation::OperationRef;
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

pub(crate) static OPERATORS: [Operator; 4] = [REM, POW, SHL, SHR];

impl Operator {
    /// The operator an op is, when it is one the library implements.
    pub(crate) fn of(op: OperationRef) -> Option<&'static Self> {
        match op.as_yz()? {
            YzOp::Rem(_) => Some(&OPERATORS[0]),
            YzOp::Pow(_) => Some(&OPERATORS[1]),
            YzOp::Shl(_) => Some(&OPERATORS[2]),
            YzOp::Shr(_) => Some(&OPERATORS[3]),
            _ => None,
        }
    }

    /// The operator a symbol implements, when it implements one.
    pub(crate) fn implemented_by(symbol: &str) -> Option<&'static Self> {
        OPERATORS
            .iter()
            .find(|operator| operator.is_implemented_by(symbol))
    }

    /// The symbol of the function that implements the operator: a new
    /// `String` for each call.
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
