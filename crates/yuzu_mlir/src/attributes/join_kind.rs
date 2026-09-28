use melior::ir::attribute::StringAttribute;
use melior::ir::operation::OperationLike;

use crate::ops::{yzl, yzr};

/// Which rows a join keeps. The same four on both sides of the lowering:
/// `yzl.join` names a relation, `yzr.join` takes two, and neither changes
/// what the kind means.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum JoinKind {
    Inner,
    Left,
    Right,
    Full,
}

impl JoinKind {
    /// Every kind, in declaration order.
    pub const ALL: [Self; 4] = [Self::Inner, Self::Left, Self::Right, Self::Full];

    /// The kind a spelling names.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.as_str() == text)
    }

    /// The spelling the attribute carries.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Inner => "inner",
            Self::Left => "left",
            Self::Right => "right",
            Self::Full => "full",
        }
    }

    fn of(operation: &melior::ir::Operation<'_>, op: &str) -> Self {
        let attribute = operation
            .attribute("kind")
            .unwrap_or_else(|_| panic!("`{op}` has a `kind` attribute"));
        let text = StringAttribute::try_from(attribute)
            .unwrap_or_else(|_| panic!("`kind` on `{op}` is a string attribute"));
        let text = text.value();
        Self::parse(text).unwrap_or_else(|| panic!("`{text}` is not a join kind"))
    }
}

impl yzl::JoinOp<'_, '_> {
    /// Which rows this join keeps. Stands in for the generated accessor.
    #[must_use]
    pub fn kind(&self) -> JoinKind {
        JoinKind::of(self.operation(), "yzl.join")
    }
}

impl yzr::JoinOp<'_, '_> {
    /// Which rows this join keeps. Stands in for the generated accessor.
    #[must_use]
    pub fn kind(&self) -> JoinKind {
        JoinKind::of(self.operation(), "yzr.join")
    }
}

#[cfg(test)]
mod tests {
    use super::JoinKind;

    #[test]
    fn every_kind_reads_back_as_itself() {
        for kind in JoinKind::ALL {
            assert_eq!(JoinKind::parse(kind.as_str()), Some(kind));
        }
        assert_eq!(JoinKind::parse("cross"), None);
    }
}
