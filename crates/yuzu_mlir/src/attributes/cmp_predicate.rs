use melior::ir::attribute::StringAttribute;
use melior::ir::operation::OperationLike;

use crate::ops::yz::CmpOp;

/// The comparison a `yz.cmp` asks for.
///
/// The C++ folder reads the same six spellings, so a name that disagreed
/// would not fail — it would quietly stop folding. Both sides answering to
/// one list is the point.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CmpPredicate {
    Equal,
    NotEqual,
    Less,
    LessOrEqual,
    Greater,
    GreaterOrEqual,
}

impl CmpPredicate {
    /// The spelling the attribute carries.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Equal => "eq",
            Self::NotEqual => "ne",
            Self::Less => "lt",
            Self::LessOrEqual => "le",
            Self::Greater => "gt",
            Self::GreaterOrEqual => "ge",
        }
    }
}

impl CmpOp<'_, '_> {
    /// Which comparison this asks for. Stands in for the generated accessor.
    ///
    /// # Panics
    ///
    /// Panics if the op has no `predicate` string attribute, or the attribute
    /// holds a spelling that is not a comparison.
    #[must_use]
    pub fn predicate(&self) -> CmpPredicate {
        let attribute = self
            .operation()
            .attribute("predicate")
            .expect("`yz.cmp` has a `predicate` attribute");
        let text = StringAttribute::try_from(attribute)
            .expect("`predicate` on `yz.cmp` is a string attribute");
        match text.value() {
            "eq" => CmpPredicate::Equal,
            "ne" => CmpPredicate::NotEqual,
            "lt" => CmpPredicate::Less,
            "le" => CmpPredicate::LessOrEqual,
            "gt" => CmpPredicate::Greater,
            "ge" => CmpPredicate::GreaterOrEqual,
            other => panic!("`{other}` is not a comparison `yz.cmp` carries"),
        }
    }
}
