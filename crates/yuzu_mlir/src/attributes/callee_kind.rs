use melior::ir::attribute::StringAttribute;
use melior::ir::operation::OperationLike;

use crate::ops::yzl::CallOperationRef;

/// What a call's name turned out to mean. Resolution decides it and stamps it
/// on `yzl.call`; every pass after reads it to know whether the call survives
/// to the engine, expands away, or is a measure.
///
/// ODS has no enumerated attribute the op generator gives us a Rust type for,
/// so the attribute is a plain string. The spellings live here rather than at
/// each site that tests one, and the enum is what the passes match on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CalleeKind {
    /// A function the program declared. Expansion replaces the call.
    Fn,
    /// An `agg fn`: declared, and aggregating. Expansion replaces it too.
    AggFn,
    /// A function the registry supplies, scalar or aggregate.
    Builtin,
    /// Declared without a body: the engine is promised to have it, and the
    /// call reaches the plan by name.
    External,
}

impl CalleeKind {
    /// The spelling the attribute carries.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fn => "fn",
            Self::AggFn => "agg_fn",
            Self::Builtin => "builtin",
            Self::External => "external",
        }
    }
}

impl CallOperationRef<'_, '_> {
    /// What this call resolved to, or `None` on one that resolution has not
    /// reached. This stands in for the generated accessor — `build.rs` leaves
    /// `callee_kind` out of the view so the attribute reads back as what it
    /// means rather than as the string it is stored in.
    pub fn callee_kind(&self) -> Option<CalleeKind> {
        let attribute = self.operation().attribute("callee_kind").ok()?;
        let text = StringAttribute::try_from(attribute)
            .expect("`callee_kind` on `yzl.call` is a string attribute");
        match text.value() {
            "fn" => Some(CalleeKind::Fn),
            "agg_fn" => Some(CalleeKind::AggFn),
            "builtin" => Some(CalleeKind::Builtin),
            "external" => Some(CalleeKind::External),
            other => panic!("`{other}` is not a callee kind resolution writes"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CalleeKind;

    /// The spelling and the reading are one pair, so a kind that goes onto an
    /// op comes back as itself.
    #[test]
    fn every_kind_spells_itself() {
        for kind in [
            CalleeKind::Fn,
            CalleeKind::AggFn,
            CalleeKind::Builtin,
            CalleeKind::External,
        ] {
            let spelled = kind.as_str();
            let read = match spelled {
                "fn" => CalleeKind::Fn,
                "agg_fn" => CalleeKind::AggFn,
                "builtin" => CalleeKind::Builtin,
                "external" => CalleeKind::External,
                other => panic!("`{other}` is not a kind this reads back"),
            };

            assert_eq!(kind, read);
        }
    }
}
