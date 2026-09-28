//! Iteration over a region's blocks.

use melior::ir::{BlockRef, RegionLike};

/// Iteration over a region's blocks.
pub trait RegionExt<'c: 'a, 'a> {
    fn blocks(&self) -> impl Iterator<Item = BlockRef<'c, 'a>>;
}

impl<'c: 'a, 'a, T: RegionLike<'c, 'a>> RegionExt<'c, 'a> for T {
    fn blocks(&self) -> impl Iterator<Item = BlockRef<'c, 'a>> {
        std::iter::successors(self.first_block(), melior::ir::BlockLike::next_in_region)
    }
}
