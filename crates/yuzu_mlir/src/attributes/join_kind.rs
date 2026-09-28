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
        match text.value() {
            "inner" => Self::Inner,
            "left" => Self::Left,
            "right" => Self::Right,
            "full" => Self::Full,
            other => panic!("`{other}` is not a join kind"),
        }
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
