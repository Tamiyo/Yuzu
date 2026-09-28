//! What an editor asks about Yuzu source, answered in file ids and text
//! ranges. The protocol is `yuzu_lsp`'s concern; nothing here knows LSP.
//!
//! The shape follows rust-analyzer's `ide` crate: an [`AnalysisHost`] takes
//! each [`Change`], and an [`Analysis`] is an immutable snapshot that a
//! request reads.

use text_size::TextSize;

mod analysis;
mod check;
mod file_structure;
mod folding_ranges;
mod hover;
mod inlay_hints;
mod names;
mod navigation;
mod selection_ranges;
mod syntax_highlighting;
#[cfg(test)]
mod test_support;

pub use analysis::{Analysis, AnalysisHost, Change};
pub use check::Checked;
pub use file_structure::{StructureNode, StructureNodeKind};
pub use folding_ranges::{Fold, FoldKind};
pub use hover::HoverResult;
pub use inlay_hints::InlayHint;
pub use navigation::FileRange;
pub use syntax_highlighting::{Highlight, HlMod, HlMods, HlRange, HlTag};

/// A file the host holds, by the number the server gave it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FileId(pub u32);

/// An offset in a file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FilePosition {
    pub file_id: FileId,
    pub offset: TextSize,
}
