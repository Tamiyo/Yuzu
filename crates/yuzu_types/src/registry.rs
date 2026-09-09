use crate::{AggFunc, BuiltinFunc, Func};

/// A function the language offers under a name. Validation reads the metadata
/// here — the builtin's kind and argument-count range — rather than matching
/// on the function itself, so new entries extend the language without touching
/// the checks.
#[derive(Clone, Copy)]
pub struct Entry {
    pub name: &'static str,
    pub func: BuiltinFunc,
    pub min_args: usize,
    pub max_args: usize,
}

/// What names exist and how their calls are shaped. Chainable: a custom
/// registry composes with the builtins via [`Registry::chain`], earlier
/// registries winning on a name collision.
pub trait Registry {
    fn entries(&self) -> &[Entry];

    fn resolve(&self, func: BuiltinFunc) -> Option<&Entry> {
        self.entries().iter().find(|entry| entry.func == func)
    }
}

pub struct Builtins;

const BUILTINS: &[Entry] = &[
    Entry {
        name: "pow",
        func: BuiltinFunc::Scalar(Func::Power),
        min_args: 2,
        max_args: 2,
    },
    Entry {
        name: "shift_left",
        func: BuiltinFunc::Scalar(Func::ShiftLeft),
        min_args: 2,
        max_args: 2,
    },
    Entry {
        name: "shift_right",
        func: BuiltinFunc::Scalar(Func::ShiftRight),
        min_args: 2,
        max_args: 2,
    },
    Entry {
        name: "in",
        func: BuiltinFunc::Scalar(Func::In),
        min_args: 2,
        max_args: 2,
    },
    Entry {
        name: "count",
        func: BuiltinFunc::Aggregate(AggFunc::Count),
        min_args: 0,
        max_args: 1,
    },
    Entry {
        name: "count_distinct",
        func: BuiltinFunc::Aggregate(AggFunc::CountDistinct),
        min_args: 1,
        max_args: 1,
    },
    Entry {
        name: "sum",
        func: BuiltinFunc::Aggregate(AggFunc::Sum),
        min_args: 1,
        max_args: 1,
    },
    Entry {
        name: "min",
        func: BuiltinFunc::Aggregate(AggFunc::Min),
        min_args: 1,
        max_args: 1,
    },
    Entry {
        name: "max",
        func: BuiltinFunc::Aggregate(AggFunc::Max),
        min_args: 1,
        max_args: 1,
    },
    Entry {
        name: "avg",
        func: BuiltinFunc::Aggregate(AggFunc::Avg),
        min_args: 1,
        max_args: 1,
    },
];

impl Registry for Builtins {
    fn entries(&self) -> &[Entry] {
        BUILTINS
    }
}

/// Registries tried in order; the first entry for a name or function wins.
pub struct Chain {
    entries: Vec<Entry>,
}

impl Chain {
    pub fn new(registries: Vec<Box<dyn Registry>>) -> Self {
        let mut entries: Vec<Entry> = Vec::new();
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

impl Registry for Chain {
    fn entries(&self) -> &[Entry] {
        &self.entries
    }
}

pub fn chain(registries: Vec<Box<dyn Registry>>) -> Chain {
    Chain::new(registries)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Aliases;

    const ALIASES: &[Entry] = &[
        Entry {
            name: "total",
            func: BuiltinFunc::Aggregate(AggFunc::Sum),
            min_args: 1,
            max_args: 1,
        },
        Entry {
            name: "count",
            func: BuiltinFunc::Aggregate(AggFunc::CountDistinct),
            min_args: 1,
            max_args: 1,
        },
    ];

    impl Registry for Aliases {
        fn entries(&self) -> &[Entry] {
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
        assert!(chained.entries().iter().any(|entry| entry.name == "total"));
        assert!(chained.entries().iter().any(|entry| entry.name == "avg"));
    }

    #[test]
    fn resolve_finds_the_first_entry_for_a_function() {
        let chained = chain(vec![Box::new(Aliases), Box::new(Builtins)]);
        let sum = chained
            .resolve(BuiltinFunc::Aggregate(AggFunc::Sum))
            .expect("sum is registered");
        assert_eq!(sum.name, "total");
    }
}
