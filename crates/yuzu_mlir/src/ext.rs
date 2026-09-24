//! Extensions over melior's IR traits: the walks and attribute reads every
//! pass repeats, as methods on the handles themselves.

use melior::Context;
use melior::ir::attribute::{
    ArrayAttribute, FlatSymbolRefAttribute, IntegerAttribute, StringAttribute, TypeAttribute,
};
use melior::ir::block::BlockArgument;
use melior::ir::operation::{OperationLike, OperationMutLike, OperationRef, OperationRefMut};
use melior::ir::r#type::IntegerType;
use melior::ir::{Attribute, BlockLike, BlockRef, RegionLike, Type, Value, ValueLike};

/// The identity of a value for the lifetime of its context: what the maps a
/// pass keys by value are keyed by. Values wrap uniqued, arena-owned
/// pointers, so the pointer is a stable identity — MLIR values are not arena
/// indices we could use instead.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ValueId(usize);

pub trait ValueExt<'c>: ValueLike<'c> {
    fn id(&self) -> ValueId {
        ValueId(self.to_raw().ptr as usize)
    }
}

impl<'c, T: ValueLike<'c>> ValueExt<'c> for T {}

pub trait IntegerAttributeExt<'c> {
    fn from_i64(context: &'c Context, value: i64) -> IntegerAttribute<'c> {
        IntegerAttribute::new(IntegerType::new(context, 64).into(), value)
    }
}

impl<'c> IntegerAttributeExt<'c> for IntegerAttribute<'c> {}

/// Array attributes, built from a Rust list and read back as one. Strings
/// read back are borrowed: attribute strings are context-uniqued, so they
/// outlive any pass reading them.
pub trait ArrayAttributeExt<'c> {
    fn from_strings(
        context: &'c Context,
        names: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> ArrayAttribute<'c> {
        let strings: Vec<Attribute<'c>> = names
            .into_iter()
            .map(|name| StringAttribute::new(context, name.as_ref()).into())
            .collect();

        ArrayAttribute::new(context, &strings)
    }

    fn from_symbols(
        context: &'c Context,
        symbols: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> ArrayAttribute<'c> {
        let symbols: Vec<Attribute<'c>> = symbols
            .into_iter()
            .map(|symbol| FlatSymbolRefAttribute::new(context, symbol.as_ref()).into())
            .collect();
        ArrayAttribute::new(context, &symbols)
    }

    fn from_types(
        context: &'c Context,
        types: impl IntoIterator<Item = Type<'c>>,
    ) -> ArrayAttribute<'c> {
        let types: Vec<Attribute<'c>> = types
            .into_iter()
            .map(|ty| TypeAttribute::new(ty).into())
            .collect();
        ArrayAttribute::new(context, &types)
    }

    fn from_indices(
        context: &'c Context,
        indices: impl IntoIterator<Item = usize>,
    ) -> ArrayAttribute<'c> {
        let i64 = IntegerType::new(context, 64).into();
        let indices: Vec<Attribute<'c>> = indices
            .into_iter()
            .map(|index| IntegerAttribute::new(i64, index as i64).into())
            .collect();
        ArrayAttribute::new(context, &indices)
    }

    fn elements(&self) -> impl Iterator<Item = Attribute<'c>>;
    fn strings(&self) -> Vec<&'c str>;
    fn symbols(&self) -> Vec<&'c str>;
    fn types(&self) -> Vec<Type<'c>>;
    fn indices(&self) -> Vec<usize>;
}

impl<'c> ArrayAttributeExt<'c> for ArrayAttribute<'c> {
    fn elements(&self) -> impl Iterator<Item = Attribute<'c>> {
        let array = *self;
        (0..array.len())
            .map(move |index| array.element(index).expect("the element index is in range"))
    }

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

    fn types(&self) -> Vec<Type<'c>> {
        self.elements()
            .filter_map(|element| TypeAttribute::try_from(element).ok())
            .map(|attribute| attribute.value())
            .collect()
    }

    fn indices(&self) -> Vec<usize> {
        self.elements()
            .filter_map(|element| IntegerAttribute::try_from(element).ok())
            .map(|index| index.value() as usize)
            .collect()
    }
}

/// Iteration over a block's arguments and operations.
pub trait BlockExt<'c: 'a, 'a> {
    fn arguments(&self) -> impl Iterator<Item = BlockArgument<'c, 'a>>;
    fn operations(&self) -> impl Iterator<Item = OperationRef<'c, 'a>>;
    fn operations_mut(&self) -> impl Iterator<Item = OperationRefMut<'c, 'a>>;
    fn last_operation(&self) -> Option<OperationRef<'c, 'a>>;
}

impl<'c: 'a, 'a, T: BlockLike<'c, 'a>> BlockExt<'c, 'a> for T {
    fn arguments(&self) -> impl Iterator<Item = BlockArgument<'c, 'a>> {
        (0..self.argument_count()).map(|index| {
            self.argument(index)
                .expect("the argument index is in range")
        })
    }

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

    /// A string attribute, by name.
    fn text_attribute(&self, name: &str) -> Option<String> {
        let attribute = self.attribute(name).ok()?;
        StringAttribute::try_from(attribute)
            .ok()
            .map(|string| string.value().to_string())
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

#[cfg(test)]
mod tests {
    use melior::ir::Type;
    use melior::ir::attribute::ArrayAttribute;

    use super::ArrayAttributeExt;

    #[test]
    fn strings_round_trip() {
        let context = crate::context();
        let array = ArrayAttribute::from_strings(&context, ["a", "b"]);
        assert_eq!(array.strings(), ["a", "b"]);
        assert!(array.symbols().is_empty());
    }

    #[test]
    fn symbols_round_trip() {
        let context = crate::context();
        let array = ArrayAttribute::from_symbols(&context, [String::from("helpers.f")]);
        assert_eq!(array.symbols(), ["helpers.f"]);
        assert!(array.strings().is_empty());
    }

    #[test]
    fn types_round_trip() {
        let context = crate::context();
        let int64 = Type::index(&context);
        let array = ArrayAttribute::from_types(&context, [int64, int64]);
        assert_eq!(array.types(), [int64, int64]);
    }

    #[test]
    fn indices_round_trip() {
        let context = crate::context();
        let array = ArrayAttribute::from_indices(&context, [2, 0, 1]);
        assert_eq!(array.indices(), [2, 0, 1]);
    }
}
