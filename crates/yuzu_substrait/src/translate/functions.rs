//! Which function a yzr operation is. A yz op maps onto a [`Func`], and the
//! Substrait tables answer from the [`Func`]. Thus no op needs a mapping of
//! its own.

use substrait::proto::aggregate_function::AggregationInvocation;
use yuzu_mlir::attributes::CmpPredicate;

use crate::extensions::{COUNT, EXTERNAL_URN, Func};

pub(crate) fn of_predicate(predicate: CmpPredicate) -> Func {
    match predicate {
        CmpPredicate::Equal => Func::Equal,
        CmpPredicate::NotEqual => Func::NotEqual,
        CmpPredicate::Less => Func::Less,
        CmpPredicate::LessOrEqual => Func::LessEqual,
        CmpPredicate::Greater => Func::Greater,
        CmpPredicate::GreaterOrEqual => Func::GreaterEqual,
    }
}

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
