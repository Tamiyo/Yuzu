//! Which function a yzr operation is. `Func` and `AggFunc` are the model's
//! own vocabulary — the identity a plan holds however a dialect spelled it —
//! so a yz op maps onto one of those and the Substrait tables answer from
//! there, rather than growing a second mapping of their own.

use yuzu_mlir::attributes::CmpPredicate;
use yuzu_types::{AggFunc, Func};

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

/// The function a call to a registry builtin applies. Membership is one of
/// these, and the caller turns it into a `SingularOrList` rather than a
/// call, the way Substrait spells it.
pub(crate) fn of_builtin(callee: &str) -> Option<Func> {
    Some(match callee {
        "pow" => Func::Power,
        "shift_left" => Func::ShiftLeft,
        "shift_right" => Func::ShiftRight,
        "in" => Func::In,
        _ => return None,
    })
}

pub(crate) fn of_aggregate(callee: &str) -> Option<AggFunc> {
    Some(match callee {
        "count" => AggFunc::Count,
        "count_distinct" => AggFunc::CountDistinct,
        "sum" => AggFunc::Sum,
        "min" => AggFunc::Min,
        "max" => AggFunc::Max,
        "avg" => AggFunc::Avg,
        _ => return None,
    })
}
