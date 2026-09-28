//! The operators the library implements. An engine takes no `yz.rem`, so
//! each such primitive op is tied to the library function that says how the
//! engine computes it: the op folds first, and `legalize_operators` puts
//! that function's body in place of what is left.

use melior::ir::operation::{OperationLike, OperationRef};

/// A primitive op, and the library function that implements it.
pub(crate) struct Operator {
    /// The op's MLIR name, such as `yz.rem`.
    pub(crate) op: &'static str,
    /// How a program writes it, such as `%`.
    pub(crate) spelling: &'static str,
    /// The library module that declares the implementation.
    pub(crate) module: &'static str,
    /// The implementation's name in that module.
    pub(crate) name: &'static str,
}

/// `%`.
pub(crate) const REM: Operator = Operator {
    op: "yz.rem",
    spelling: "%",
    module: "yuzu.std.ops",
    name: "modulo",
};

const OPERATORS: &[Operator] = &[REM];

impl Operator {
    /// The operator an op is, when it is one the library implements.
    pub(crate) fn of(op: OperationRef) -> Option<&'static Self> {
        OPERATORS.iter().find(|operator| is_named(op, operator.op))
    }

    /// The operator a symbol implements, when it implements one.
    pub(crate) fn implemented_by(symbol: &str) -> Option<&'static Self> {
        OPERATORS
            .iter()
            .find(|operator| operator.is_implemented_by(symbol))
    }

    /// The symbol of the function that implements the operator.
    pub(crate) fn symbol(&self) -> String {
        format!("{}.{}", self.module, self.name)
    }

    fn is_implemented_by(&self, symbol: &str) -> bool {
        symbol
            .strip_prefix(self.module)
            .and_then(|rest| rest.strip_prefix('.'))
            == Some(self.name)
    }
}

/// Whether an op has a name, without copying the name out.
pub(crate) fn is_named(op: OperationRef, name: &str) -> bool {
    op.name().as_string_ref().as_str() == Ok(name)
}

#[cfg(test)]
mod tests {
    use super::Operator;

    #[test]
    fn an_implementation_is_found_by_its_symbol() {
        let rem = Operator::implemented_by("yuzu.std.ops.modulo").expect("modulo implements `%`");
        assert_eq!(rem.op, super::REM.op);
        assert!(Operator::implemented_by("yuzu.std.opsmodulo").is_none());
        assert!(Operator::implemented_by("modulo").is_none());
    }
}
