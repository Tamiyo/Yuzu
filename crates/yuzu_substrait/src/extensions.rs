use substrait::proto::extensions::{
    SimpleExtensionDeclaration, SimpleExtensionUrn,
    simple_extension_declaration::{ExtensionFunction, MappingType},
};

// Substrait standard extensions (the function families DuckDB consumes).
const ARITHMETIC_URN: &str = "extension:io.substrait:functions_arithmetic";
pub(crate) const COMPARISON_URN: &str = "extension:io.substrait:functions_comparison";
pub(crate) const BOOLEAN_URN: &str = "extension:io.substrait:functions_boolean";
const AGGREGATE_GENERIC_URN: &str = "extension:io.substrait:functions_aggregate_generic";
pub(crate) const EXTERNAL_URN: &str = "extension:io.yuzu:external";

/// A function a primitive yz op computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Func {
    Add,
    Subtract,
    Multiply,
    Divide,
    Negate,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    And,
    Or,
    Not,
}

/// Map a primitive op's function to its Substrait extension function.
/// Membership is not one: it is a `SingularOrList`.
pub(crate) fn function_target(func: Func) -> (&'static str, &'static str) {
    match func {
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
    }
}

/// Substrait's generic `count`, which counts every row or every value, or
/// only distinct ones.
pub(crate) const COUNT: (&str, &str) = (AGGREGATE_GENERIC_URN, "count");

/// The plan's extension tables, built up as functions are registered.
#[derive(Default)]
pub(crate) struct Extensions {
    urns: Vec<&'static str>,
    functions: Vec<(u32, String)>,
}

impl Extensions {
    /// Intern a `(urn, name)` function; returns its function anchor.
    pub(crate) fn register(&mut self, urn: &'static str, name: String) -> u32 {
        let index = if let Some(index) = self.urns.iter().position(|&candidate| candidate == urn) {
            index
        } else {
            self.urns.push(urn);
            self.urns.len() - 1
        };
        self.functions.push((anchor(index), name));
        anchor(self.functions.len() - 1)
    }

    pub(crate) fn urns(&self) -> Vec<SimpleExtensionUrn> {
        self.urns
            .iter()
            .enumerate()
            .map(|(index, &urn)| SimpleExtensionUrn {
                extension_urn_anchor: anchor(index),
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
                    function_anchor: anchor(index),
                    name: name.clone(),
                })),
            })
            .collect()
    }
}

/// The anchor of the extension at `index`: anchors count from 1, since 0
/// means none.
fn anchor(index: usize) -> u32 {
    u32::try_from(index + 1).expect("a plan declares fewer than 2^32 extensions")
}
