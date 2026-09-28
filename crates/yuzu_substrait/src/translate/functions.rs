//! Which function a yzr operation is. `Func` and `AggFunc` are the model's
//! own vocabulary — the identity a plan holds however a dialect spelled it —
//! so a yz op maps onto one of those and the Substrait tables answer from
//! there, rather than growing a second mapping of their own.

use substrait::proto::aggregate_function::AggregationInvocation;
use yuzu_mlir::attributes::CmpPredicate;
use yuzu_types::{AggFunc, Func};

use crate::extensions::{EXTERNAL_URN, aggregate_target};

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

    let (urn, base) = aggregate_target(AggFunc::Count);
    Aggregate {
        urn,
        base: base.to_string(),
        invocation,
    }
}
