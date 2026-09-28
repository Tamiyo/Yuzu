use melior::ir::attribute::StringAttribute;
use melior::ir::operation::OperationLike;

use crate::ops::yzl::CallOp;

/// What a call's name turned out to mean. Resolution decides it and stamps it
/// on `yzl.call`; every pass after reads it to know whether the call survives
/// to the engine, expands away, or is a measure.
///
/// ODS has no enumerated attribute the op generator gives us a Rust type for,
/// so the attribute is a plain string. The spellings live here rather than at
/// each site that tests one, and the enum is what the passes match on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CalleeSource {
    /// A function the program declared, scalar or aggregate. Expansion
    /// replaces the call.
    Fn,
    /// Declared without a body: the engine is promised to have it, and the
    /// call reaches the plan by name.
    External,
    /// A file-level constant, `yzl.const`: a body with no parameters,
    /// expanded at each use exactly as a function is. A call is how a use
    /// refers to it, since a stage region cannot reach a value outside itself.
    Const,
}

impl CalleeSource {
    /// Every kind, in declaration order.
    pub const ALL: [Self; 3] = [Self::Fn, Self::External, Self::Const];

    /// The kind a spelling names.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.as_str() == text)
    }

    /// The spelling the attribute carries.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fn => "fn",
            Self::External => "external",
            Self::Const => "const",
        }
    }
}

impl CallOp<'_, '_> {
    /// What the call's name turned out to mean; `None` before resolution stamps it.
    ///
    /// # Panics
    ///
    /// Panics if `callee_source` is not a string attribute, or holds a
    /// spelling resolution does not write.
    #[must_use]
    pub fn callee_source(&self) -> Option<CalleeSource> {
        let attribute = self.operation().attribute("callee_source").ok()?;
        let text = StringAttribute::try_from(attribute)
            .expect("`callee_source` on `yzl.call` is a string attribute");

        let text = text.value();
        let source = CalleeSource::parse(text)
            .unwrap_or_else(|| panic!("`{text}` is not a callee source resolution writes"));
        Some(source)
    }
}

#[cfg(test)]
mod tests {
    use super::CalleeSource;

    #[test]
    fn every_kind_reads_back_as_itself() {
        for kind in CalleeSource::ALL {
            assert_eq!(CalleeSource::parse(kind.as_str()), Some(kind));
        }
        assert_eq!(CalleeSource::parse("macro"), None);
    }
}
