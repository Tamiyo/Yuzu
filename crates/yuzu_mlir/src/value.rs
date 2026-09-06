//! Value identity for pass-side maps.

use melior::ir::{Value, ValueLike};

/// A key identifying an SSA value for the lifetime of its context: values
/// wrap uniqued, arena-owned pointers, so the pointer is a stable identity.
pub fn value_id(value: Value) -> usize {
    value.to_raw().ptr as usize
}
