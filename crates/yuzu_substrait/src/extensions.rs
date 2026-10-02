use std::collections::hash_map::Entry;

use rustc_hash::FxHashMap;
use substrait::proto::extensions::{
    SimpleExtensionDeclaration, SimpleExtensionUrn,
    simple_extension_declaration::{ExtensionFunction, MappingType},
};

// Substrait standard extensions (the function families DuckDB consumes).
pub(crate) const COMPARISON_URN: &str = "extension:io.substrait:functions_comparison";
const AGGREGATE_GENERIC_URN: &str = "extension:io.substrait:functions_aggregate_generic";
pub(crate) const EXTERNAL_URN: &str = "extension:io.yuzu:external";

/// Substrait's generic `count`, which counts every row or every value, or
/// only distinct ones.
pub(crate) const COUNT: (&str, &str) = (AGGREGATE_GENERIC_URN, "count");

/// The plan's extension tables, built up as functions are registered.
#[derive(Default)]
pub(crate) struct Extensions {
    urns: Vec<&'static str>,
    /// Each function's anchor, by where it is declared and its name.
    functions: FxHashMap<(&'static str, String), u32>,
}

impl Extensions {
    /// Intern a `(urn, name)` function; returns its function anchor.
    pub(crate) fn register(&mut self, urn: &'static str, name: String) -> u32 {
        let next = anchor(self.functions.len());
        match self.functions.entry((urn, name)) {
            Entry::Occupied(declared) => *declared.get(),
            Entry::Vacant(entry) => {
                if !self.urns.contains(&urn) {
                    self.urns.push(urn);
                }
                *entry.insert(next)
            }
        }
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

    /// The functions, in the order they were registered.
    pub(crate) fn declarations(&self) -> Vec<SimpleExtensionDeclaration> {
        let mut functions: Vec<(u32, &'static str, &str)> = self
            .functions
            .iter()
            .map(|((urn, name), &function_anchor)| (function_anchor, *urn, name.as_str()))
            .collect();
        functions.sort_unstable_by_key(|&(function_anchor, ..)| function_anchor);
        functions
            .into_iter()
            .map(|(function_anchor, urn, name)| SimpleExtensionDeclaration {
                mapping_type: Some(MappingType::ExtensionFunction(ExtensionFunction {
                    extension_urn_reference: self.urn_anchor(urn),
                    function_anchor,
                    name: name.to_owned(),
                })),
            })
            .collect()
    }

    fn urn_anchor(&self, urn: &str) -> u32 {
        let index = self
            .urns
            .iter()
            .position(|&declared| declared == urn)
            .expect("a registered function's urn is declared");
        anchor(index)
    }
}

include!(concat!(env!("OUT_DIR"), "/catalogue.rs"));

/// The URN of the standard extension that declares `name` for arguments of
/// these types, when Substrait's catalogue has one. An implementation for
/// exactly these types wins over a generic one.
pub(crate) fn standard_urn(name: &str, args: &[&str]) -> Option<&'static str> {
    let start = CATALOGUE.partition_point(|row| row.0 < name);
    let mut generic = None;
    for &(_, params, variadic, urn) in CATALOGUE[start..].iter().take_while(|row| row.0 == name) {
        match accepts(params, variadic, args) {
            Some(Match::Exact) => return Some(urn),
            Some(Match::Generic) => {
                generic.get_or_insert(urn);
            }
            None => {}
        }
    }
    generic
}

/// How an implementation takes a call's arguments.
#[derive(Clone, Copy)]
enum Match {
    Exact,
    Generic,
}

/// `None` when the implementation cannot take arguments of these types. A
/// variadic implementation repeats its last parameter.
fn accepts(params: &[&str], variadic: bool, args: &[&str]) -> Option<Match> {
    let arity = if variadic {
        args.len() + 1 >= params.len()
    } else {
        args.len() == params.len()
    };
    if !arity {
        return None;
    }

    let mut matched = Match::Exact;
    for (index, arg) in args.iter().enumerate() {
        let param = params.get(index).or(params.last())?;
        if *param == "any" {
            matched = Match::Generic;
        } else if param != arg {
            return None;
        }
    }
    Some(matched)
}

/// The anchor of the extension at `index`: anchors count from 1, since 0
/// means none.
fn anchor(index: usize) -> u32 {
    u32::try_from(index + 1).expect("a plan declares fewer than 2^32 extensions")
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use super::{Extensions, standard_urn};

    #[test]
    fn a_standard_function_is_found_by_its_name_and_argument_types() {
        let cases: &[(&str, &[&str])] = &[
            ("add", &["i64", "i64"]),
            ("add", &["fp64", "fp64"]),
            ("modulus", &["i64", "i64"]),
            ("power", &["fp64", "fp64"]),
            ("lt", &["i64", "i64"]),
            ("equal", &["string", "string"]),
            ("and", &["bool", "bool", "bool"]),
            ("coalesce", &["i64", "i64"]),
            ("sum", &["i64"]),
            ("add", &["string", "string"]),
            ("bitwise_shift_left", &["i64", "i64"]),
        ];
        let found: Vec<String> = cases
            .iter()
            .map(|(name, args)| {
                let urn = standard_urn(name, args).unwrap_or("none");
                format!("{name}({}): {urn}", args.join(", "))
            })
            .collect();
        expect![[r"
            add(i64, i64): extension:io.substrait:functions_arithmetic
            add(fp64, fp64): extension:io.substrait:functions_arithmetic
            modulus(i64, i64): extension:io.substrait:functions_arithmetic
            power(fp64, fp64): extension:io.substrait:functions_arithmetic
            lt(i64, i64): extension:io.substrait:functions_comparison
            equal(string, string): extension:io.substrait:functions_comparison
            and(bool, bool, bool): extension:io.substrait:functions_boolean
            coalesce(i64, i64): extension:io.substrait:functions_comparison
            sum(i64): extension:io.substrait:functions_arithmetic
            add(string, string): none
            bitwise_shift_left(i64, i64): none"]]
        .assert_eq(&found.join("\n"));
    }

    #[test]
    fn a_function_used_twice_is_declared_once() {
        let mut extensions = Extensions::default();
        let urn = super::COMPARISON_URN;
        let first = extensions.register(urn, "coalesce:i64_i64".to_owned());
        let again = extensions.register(urn, "coalesce:i64_i64".to_owned());
        let other = extensions.register(urn, "coalesce:fp64_fp64".to_owned());
        assert_eq!(first, again);
        assert_ne!(first, other);
        assert_eq!(extensions.declarations().len(), 2);
        assert_eq!(extensions.urns().len(), 1);
    }
}
