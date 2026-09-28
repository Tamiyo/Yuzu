//! Iteration over a block's arguments and operations.

use melior::ir::BlockLike;
use melior::ir::block::BlockArgument;
use melior::ir::operation::{OperationRef, OperationRefMut};

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
        std::iter::successors(
            self.first_operation(),
            melior::ir::operation::OperationLike::next_in_block,
        )
    }

    fn operations_mut(&self) -> impl Iterator<Item = OperationRefMut<'c, 'a>> {
        std::iter::successors(
            self.first_operation_mut(),
            melior::ir::operation::OperationLike::next_in_block_mut,
        )
    }

    fn last_operation(&self) -> Option<OperationRef<'c, 'a>> {
        self.operations().last()
    }
}
