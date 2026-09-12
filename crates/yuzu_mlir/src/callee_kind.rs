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

    /// What a call resolved to, or `None` on one resolution has not reached.
    pub fn of(call: &CallOperationRef<'_, '_>) -> Option<Self> {
        match call.callee_kind()?.value() {
            "fn" => Some(Self::Fn),
            "agg_fn" => Some(Self::AggFn),
            "builtin" => Some(Self::Builtin),
            "external" => Some(Self::External),
            _ => None,
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
