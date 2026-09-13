//! Extensions over melior's IR traits: the walks and attribute reads every
//! pass repeats, as methods on the handles themselves.

use melior::Context;
use melior::ir::attribute::{
    ArrayAttribute, FlatSymbolRefAttribute, IntegerAttribute, StringAttribute, TypeAttribute,
};
use melior::ir::operation::{OperationLike, OperationMutLike, OperationRef, OperationRefMut};
use melior::ir::r#type::IntegerType;
use melior::ir::{Attribute, BlockLike, BlockRef, RegionLike, Type, Value, ValueLike};

/// Identity for the maps a pass keys by value.
pub trait ValueExt<'c>: ValueLike<'c> {
    /// A key identifying this value for the lifetime of its context. Values
    /// wrap uniqued, arena-owned pointers, so the pointer is a stable
    /// identity — MLIR values are not arena indices we could use instead.
    fn id(&self) -> usize {
        self.to_raw().ptr as usize
    }
}

impl<'c, T: ValueLike<'c>> ValueExt<'c> for T {}

/// Element access for array attributes.
pub trait ArrayAttributeExt<'c> {
    fn elements(&self) -> impl Iterator<Item = Attribute<'c>>;
    fn strings(&self) -> Vec<&'c str>;
    fn symbols(&self) -> Vec<&'c str>;
}

impl<'c> ArrayAttributeExt<'c> for ArrayAttribute<'c> {
    fn elements(&self) -> impl Iterator<Item = Attribute<'c>> {
        let array = *self;
        (0..array.len())
            .map(move |index| array.element(index).expect("the element index is in range"))
    }

    /// Borrowed: attribute strings are context-uniqued, so they outlive any
    /// pass reading them.
    fn strings(&self) -> Vec<&'c str> {
        self.elements()
            .filter_map(|element| StringAttribute::try_from(element).ok())
            .map(|string| string.value())
            .collect()
    }

    fn symbols(&self) -> Vec<&'c str> {
        self.elements()
            .filter_map(|element| FlatSymbolRefAttribute::try_from(element).ok())
            .map(|symbol| symbol.value())
            .collect()
    }
}

/// Iteration over a block's operations.
pub trait BlockExt<'c: 'a, 'a> {
    fn operations(&self) -> impl Iterator<Item = OperationRef<'c, 'a>>;
    fn operations_mut(&self) -> impl Iterator<Item = OperationRefMut<'c, 'a>>;
    fn last_operation(&self) -> Option<OperationRef<'c, 'a>>;
}

impl<'c: 'a, 'a, T: BlockLike<'c, 'a>> BlockExt<'c, 'a> for T {
    fn operations(&self) -> impl Iterator<Item = OperationRef<'c, 'a>> {
        std::iter::successors(self.first_operation(), |op| op.next_in_block())
    }

    fn operations_mut(&self) -> impl Iterator<Item = OperationRefMut<'c, 'a>> {
        std::iter::successors(self.first_operation_mut(), |op| op.next_in_block_mut())
    }

    fn last_operation(&self) -> Option<OperationRef<'c, 'a>> {
        self.operations().last()
    }
}

/// Iteration over a region's blocks.
pub trait RegionExt<'c: 'a, 'a> {
    fn blocks(&self) -> impl Iterator<Item = BlockRef<'c, 'a>>;
}

impl<'c: 'a, 'a, T: RegionLike<'c, 'a>> RegionExt<'c, 'a> for T {
    fn blocks(&self) -> impl Iterator<Item = BlockRef<'c, 'a>> {
        std::iter::successors(self.first_block(), |block| block.next_in_region())
    }
}

/// Typed reads of the attributes ops carry outside their ODS arguments,
/// such as the indices the passes stamp.
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

    /// The type this op produces. Inference stamps what it settled on rather
    /// than rewriting the IR, so the stamp is the better answer wherever it
    /// exists and the result's own type stands in where it does not.
    fn ty(&self) -> Type<'c> {
        self.attribute("ty")
            .ok()
            .and_then(|attribute| TypeAttribute::try_from(attribute).ok())
            .map(|attribute| attribute.value())
            .unwrap_or_else(|| self.first_result().r#type())
    }

    /// A string attribute, by name.
    fn text_attribute(&self, name: &str) -> Option<String> {
        let attribute = self.attribute(name).ok()?;
        StringAttribute::try_from(attribute)
            .ok()
            .map(|string| string.value().to_string())
    }

    /// An index-valued attribute, by name.
    fn index_attribute(&self, name: &str) -> Option<usize> {
        let attribute = self.attribute(name).ok()?;
        IntegerAttribute::try_from(attribute)
            .ok()
            .map(|index| index.value() as usize)
    }

    /// An array of indices, by name.
    fn index_array_attribute(&self, name: &str) -> Vec<usize> {
        let Some(array) = self
            .attribute(name)
            .ok()
            .and_then(|attribute| ArrayAttribute::try_from(attribute).ok())
        else {
            return Vec::new();
        };

        array
            .elements()
            .filter_map(|element| IntegerAttribute::try_from(element).ok())
            .map(|index| index.value() as usize)
            .collect()
    }
}

impl<'c: 'a, 'a, T: OperationLike<'c, 'a>> OperationExt<'c, 'a> for T {}

/// Stamping the answers passes record, as attributes.
pub trait OperationMutExt<'c: 'a, 'a>: OperationMutLike<'c, 'a> {
    /// Stamps an index-valued attribute.
    fn set_index_attribute(&mut self, context: &'c Context, name: &str, index: usize) {
        let i64 = IntegerType::new(context, 64).into();
        self.set_attribute(name, IntegerAttribute::new(i64, index as i64).into());
    }

    /// Stamps an array of indices.
    fn set_index_array_attribute(&mut self, context: &'c Context, name: &str, indices: &[usize]) {
        let i64 = IntegerType::new(context, 64).into();
        let elements: Vec<Attribute> = indices
            .iter()
            .map(|&index| IntegerAttribute::new(i64, index as i64).into())
            .collect();
        self.set_attribute(name, ArrayAttribute::new(context, &elements).into());
    }
}

impl<'c: 'a, 'a, T: OperationMutLike<'c, 'a>> OperationMutExt<'c, 'a> for T {}

/// Reading an operation as one of our dialects. MLIR spells this
/// `dyn_cast<AddOp>(op)`; in Rust the conversion belongs on the handle, and
/// the dialect enum stands in for what a `TypeSwitch` would have matched.
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
