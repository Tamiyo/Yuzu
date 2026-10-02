//! Which function a measure calls. A scalar function is an external call
//! by the time it reaches the translation, so only an aggregate needs a
//! mapping here.

use substrait::proto::aggregate_function::AggregationInvocation;

use crate::extensions::{COUNT, EXTERNAL_URN};

/// What a measure calls: where Substrait declares the function, the name it
/// is declared under, and whether it sees every value or only distinct ones.
pub(crate) struct Aggregate {
    pub(crate) urn: &'static str,
    pub(crate) base: String,
    pub(crate) invocation: AggregationInvocation,
}

/// A measure's function, by the name the lowering put on it.
///
/// `count` and `count_distinct` are Substrait's generic count. Any other name
/// is one the engine provides, declared under its own name.
pub(crate) fn of_aggregate(name: &str) -> Aggregate {
    let invocation = match name {
        "count" => AggregationInvocation::All,
        // Substrait spells a distinct count as `count` over distinct values.
        "count_distinct" => AggregationInvocation::Distinct,
        external => {
            return Aggregate {
                urn: EXTERNAL_URN,
                base: external.to_string(),
                invocation: AggregationInvocation::All,
            };
        }
    };

    let (urn, base) = COUNT;
    Aggregate {
        urn,
        base: base.to_string(),
        invocation,
    }
}
