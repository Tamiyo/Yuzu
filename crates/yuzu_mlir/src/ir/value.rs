//! Value identity, for maps a pass keys by value.

use melior::ir::ValueLike;

/// The identity of a value for the lifetime of its context: what the maps a
/// pass keys by value are keyed by. Values wrap uniqued, arena-owned
/// pointers, so the pointer is a stable identity.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ValueId(usize);

pub trait ValueExt<'c>: ValueLike<'c> {
    fn id(&self) -> ValueId {
        ValueId(self.to_raw().ptr as usize)
    }
}

impl<'c, T: ValueLike<'c>> ValueExt<'c> for T {}
