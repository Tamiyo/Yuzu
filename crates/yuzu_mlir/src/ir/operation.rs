//! Reads, attribute stamps and dialect casts on an operation.

use melior::Context;
use melior::ir::Value;
use melior::ir::attribute::{ArrayAttribute, StringAttribute};
use melior::ir::operation::{OperationLike, OperationMutLike, OperationRef, OperationRefMut};

use crate::ir::attribute::array::ArrayAttributeExt;

/// Reads over an operation's results, operands and attributes.
pub trait OperationExt<'c: 'a, 'a>: OperationLike<'c, 'a> {
    /// The op's single result, as a value. Every op the passes build has
    /// one; a terminator has none and must not be asked.
    fn first_result(&self) -> Value<'c, 'a> {
        self.result(0).expect("the operation has a result").into()
    }

    /// The op's single result, when it has one.
    ///
    /// Ask the count rather than attempting `result(0)` and discarding the
    /// error: that error reports a real contract violation and carries the
    /// printed operation, so using it to mean "no result" is both a misuse
    /// and O(enclosing scope) per op.
    fn try_first_result(&self) -> Option<Value<'c, 'a>> {
        (self.result_count() > 0).then(|| self.first_result())
    }

    /// The op's first operand, when it has one — asked by count, for the
    /// same reason as `try_first_result`.
    fn try_first_operand(&self) -> Option<Value<'c, 'a>> {
        (self.operand_count() > 0).then(|| self.operand(0).expect("the operand index is in range"))
    }

    /// A string attribute, by name; `None` when the op has no such
    /// attribute.
    ///
    /// # Panics
    ///
    /// Panics if the attribute is there but is not a string.
    fn text_attribute(&self, name: &str) -> Option<&'c str> {
        let attribute = self.attribute(name).ok()?;
        let text = StringAttribute::try_from(attribute)
            .unwrap_or_else(|_| panic!("`{name}` is a string attribute"));
        Some(text.value())
    }
}

impl<'c: 'a, 'a, T: OperationLike<'c, 'a>> OperationExt<'c, 'a> for T {}

/// Stamping the answers passes record, as attributes.
pub trait OperationMutExt<'c: 'a, 'a>: OperationMutLike<'c, 'a> {
    /// Marks a symbol private, so symbol DCE may remove it once nothing
    /// refers to it. A symbol is public unless told otherwise.
    fn set_private(&mut self, context: &'c Context) {
        let private = StringAttribute::new(context, "private");
        self.set_attribute("sym_visibility", private.into());
    }

    /// Stamps an array of indices.
    fn set_index_array_attribute(&mut self, context: &'c Context, name: &str, indices: &[usize]) {
        let indices = ArrayAttribute::from_indices(context, indices.iter().copied());
        self.set_attribute(name, indices.into());
    }
}

impl<'c: 'a, 'a, T: OperationMutLike<'c, 'a>> OperationMutExt<'c, 'a> for T {}

/// Views an operation as an op of one of Yuzu's dialects, to match on.
pub trait OperationCast<'c> {
    fn as_yz(&self) -> Option<crate::ops::yz::YzOp<'c, '_>>;
    fn as_yzl(&self) -> Option<crate::ops::yzl::YzlOp<'c, '_>>;
    fn as_yzr(&self) -> Option<crate::ops::yzr::YzrOp<'c, '_>>;
}

/// Written out per handle rather than blanket over `Deref`: an owned
/// `Operation` does not deref to itself, and it is the one a test most often
/// holds.
macro_rules! operation_cast {
    ($handle:ty) => {
        impl<'c> OperationCast<'c> for $handle {
            fn as_yz(&self) -> Option<crate::ops::yz::YzOp<'c, '_>> {
                crate::ops::yz::YzOp::of(self)
            }

            fn as_yzl(&self) -> Option<crate::ops::yzl::YzlOp<'c, '_>> {
                crate::ops::yzl::YzlOp::of(self)
            }

            fn as_yzr(&self) -> Option<crate::ops::yzr::YzrOp<'c, '_>> {
                crate::ops::yzr::YzrOp::of(self)
            }
        }
    };
}

operation_cast!(melior::ir::Operation<'c>);
operation_cast!(OperationRef<'c, '_>);
operation_cast!(OperationRefMut<'c, '_>);
