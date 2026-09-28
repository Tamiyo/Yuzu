use substrait::proto::extensions::{
    SimpleExtensionDeclaration, SimpleExtensionUrn,
    simple_extension_declaration::{ExtensionFunction, MappingType},
};
use yuzu_types::{AggFunc, Func};

// Substrait standard extensions (the function families DuckDB consumes).
const ARITHMETIC_URN: &str = "extension:io.substrait:functions_arithmetic";
pub(crate) const COMPARISON_URN: &str = "extension:io.substrait:functions_comparison";
pub(crate) const BOOLEAN_URN: &str = "extension:io.substrait:functions_boolean";
const AGGREGATE_GENERIC_URN: &str = "extension:io.substrait:functions_aggregate_generic";
pub(crate) const EXTERNAL_URN: &str = "extension:io.yuzu:external";

/// Map a plan function to its Substrait extension function. Membership is
/// handled separately (`SingularOrList`); a function with no Substrait
/// equivalent yet (e.g. `**`) returns none.
pub(crate) fn function_target(func: Func) -> Option<(&'static str, &'static str)> {
    let target = match func {
        Func::Add => (ARITHMETIC_URN, "add"),
        Func::Subtract => (ARITHMETIC_URN, "subtract"),
        Func::Multiply => (ARITHMETIC_URN, "multiply"),
        Func::Divide => (ARITHMETIC_URN, "divide"),
        Func::Negate => (ARITHMETIC_URN, "negate"),
        Func::Equal => (COMPARISON_URN, "equal"),
        Func::NotEqual => (COMPARISON_URN, "not_equal"),
        Func::Less => (COMPARISON_URN, "lt"),
        Func::LessEqual => (COMPARISON_URN, "lte"),
        Func::Greater => (COMPARISON_URN, "gt"),
        Func::GreaterEqual => (COMPARISON_URN, "gte"),
        Func::And => (BOOLEAN_URN, "and"),
        Func::Or => (BOOLEAN_URN, "or"),
        Func::Not => (BOOLEAN_URN, "not"),
        Func::In => return None,
    };
    Some(target)
}

/// Map a builtin aggregate to its Substrait extension function: `count`, in
/// the generic aggregate family.
pub(crate) fn aggregate_target(func: AggFunc) -> (&'static str, &'static str) {
    match func {
        AggFunc::Count | AggFunc::CountDistinct => (AGGREGATE_GENERIC_URN, "count"),
    }
}

/// The plan's extension tables, built up as functions are registered.
#[derive(Default)]
pub(crate) struct Extensions {
    urns: Vec<&'static str>,
    functions: Vec<(u32, String)>,
}

impl Extensions {
    /// Intern a `(urn, name)` function; returns its function anchor.
    pub(crate) fn register(&mut self, urn: &'static str, name: String) -> u32 {
        let urn_anchor = if let Some(index) = self.urns.iter().position(|&candidate| candidate == urn) { index as u32 + 1 } else {
            self.urns.push(urn);
            self.urns.len() as u32
        };
        self.functions.push((urn_anchor, name));
        self.functions.len() as u32
    }

    pub(crate) fn urns(&self) -> Vec<SimpleExtensionUrn> {
        self.urns
            .iter()
            .enumerate()
            .map(|(index, &urn)| SimpleExtensionUrn {
                extension_urn_anchor: index as u32 + 1,
                urn: urn.to_string(),
            })
            .collect()
    }

    pub(crate) fn declarations(&self) -> Vec<SimpleExtensionDeclaration> {
        self.functions
            .iter()
            .enumerate()
            .map(|(index, (urn_anchor, name))| SimpleExtensionDeclaration {
                mapping_type: Some(MappingType::ExtensionFunction(ExtensionFunction {
                    extension_urn_reference: *urn_anchor,
                    function_anchor: index as u32 + 1,
                    name: name.clone(),
                })),
            })
            .collect()
    }
}
