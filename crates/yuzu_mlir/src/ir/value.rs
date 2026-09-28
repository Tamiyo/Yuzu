//! Value identity, for maps a pass keys by value.

use melior::ir::ValueLike;

/// The identity of a value while the op that defines it is alive.
///
/// A value is not uniqued, so a new op can reuse the memory of an erased
/// one: a map keyed by `ValueId` must not outlive an erasure in the IR it
/// reads.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ValueId(usize);

pub trait ValueExt<'c>: ValueLike<'c> {
    fn id(&self) -> ValueId {
        ValueId(self.to_raw().ptr as usize)
    }
}

impl<'c, T: ValueLike<'c>> ValueExt<'c> for T {}
