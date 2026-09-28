use crate::{AggFunc, BuiltinFunc, Func};

/// A function the language offers under a name. Validation reads the metadata
/// here — the builtin's kind and argument-count range — rather than matching
/// on the function itself, so new entries extend the language without touching
/// the checks.
#[derive(Clone, Copy)]
pub struct FunctionRegistryEntry {
    pub name: &'static str,
    pub func: BuiltinFunc,
    pub min_args: usize,
    pub max_args: usize,
}

/// What names exist and how their calls are shaped. Chainable: a custom
/// registry composes with the builtins via [`Registry::chain`], earlier
/// registries winning on a name collision.
pub trait FunctionRegistry {
    fn entries(&self) -> &[FunctionRegistryEntry];

    fn resolve(&self, func: BuiltinFunc) -> Option<&FunctionRegistryEntry> {
        self.entries().iter().find(|entry| entry.func == func)
    }
}

pub struct Builtins;

const BUILTINS: &[FunctionRegistryEntry] = &[
    FunctionRegistryEntry {
        name: "in",
        func: BuiltinFunc::Scalar(Func::In),
        min_args: 2,
        max_args: 2,
    },
    FunctionRegistryEntry {
        name: "count",
        func: BuiltinFunc::Aggregate(AggFunc::Count),
        min_args: 0,
        max_args: 1,
    },
    FunctionRegistryEntry {
        name: "count_distinct",
        func: BuiltinFunc::Aggregate(AggFunc::CountDistinct),
        min_args: 1,
        max_args: 1,
    },
];

impl FunctionRegistry for Builtins {
    fn entries(&self) -> &[FunctionRegistryEntry] {
        BUILTINS
    }
}

/// Registries tried in order; the first entry for a name or function wins.
pub struct Chain {
    entries: Vec<FunctionRegistryEntry>,
}

impl Chain {
    #[must_use]
    pub fn new(registries: Vec<Box<dyn FunctionRegistry>>) -> Self {
        let mut entries: Vec<FunctionRegistryEntry> = Vec::new();
        for registry in &registries {
            for &entry in registry.entries() {
                if !entries.iter().any(|seen| seen.name == entry.name) {
                    entries.push(entry);
                }
            }
        }
        Self { entries }
    }
}

impl FunctionRegistry for Chain {
    fn entries(&self) -> &[FunctionRegistryEntry] {
        &self.entries
    }
}

#[must_use]
pub fn chain(registries: Vec<Box<dyn FunctionRegistry>>) -> Chain {
    Chain::new(registries)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Aliases;

    const ALIASES: &[FunctionRegistryEntry] = &[
        FunctionRegistryEntry {
            name: "tally",
            func: BuiltinFunc::Aggregate(AggFunc::Count),
            min_args: 0,
            max_args: 1,
        },
        FunctionRegistryEntry {
            name: "count",
            func: BuiltinFunc::Aggregate(AggFunc::CountDistinct),
            min_args: 1,
            max_args: 1,
        },
    ];

    impl FunctionRegistry for Aliases {
        fn entries(&self) -> &[FunctionRegistryEntry] {
            ALIASES
        }
    }

    #[test]
    fn chain_keeps_the_first_entry_for_a_name() {
        let chained = chain(vec![Box::new(Aliases), Box::new(Builtins)]);
        let count = chained
            .entries()
            .iter()
            .find(|entry| entry.name == "count")
            .expect("count is registered");
        assert_eq!(count.func, BuiltinFunc::Aggregate(AggFunc::CountDistinct));
    }

    #[test]
    fn chain_adds_new_names() {
        let chained = chain(vec![Box::new(Aliases), Box::new(Builtins)]);
        assert!(chained.entries().iter().any(|entry| entry.name == "tally"));
        assert!(chained.entries().iter().any(|entry| entry.name == "in"));
    }

    #[test]
    fn resolve_finds_the_first_entry_for_a_function() {
        let chained = chain(vec![Box::new(Aliases), Box::new(Builtins)]);
        let count = chained
            .resolve(BuiltinFunc::Aggregate(AggFunc::Count))
            .expect("count is registered");
        assert_eq!(count.name, "tally");
    }
}
