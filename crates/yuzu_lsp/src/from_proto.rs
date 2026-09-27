//! Protocol positions to offsets. A client may send a position past the end
//! of the text it describes, so each conversion can fail.

use line_index::{LineCol, WideEncoding, WideLineCol};
use lsp_types::{Position, Range};
use text_size::{TextRange, TextSize};

use crate::line_index::{LineIndex, PositionEncoding};

pub(crate) fn offset(line_index: &LineIndex, position: Position) -> Option<TextSize> {
    let line_col = match line_index.encoding {
        PositionEncoding::Utf8 => LineCol {
            line: position.line,
            col: position.character,
        },
        PositionEncoding::Utf16 => line_index.index.to_utf8(
            WideEncoding::Utf16,
            WideLineCol {
                line: position.line,
                col: position.character,
            },
        )?,
    };
    line_index.index.offset(line_col)
}

pub(crate) fn text_range(line_index: &LineIndex, range: Range) -> Option<TextRange> {
    let start = offset(line_index, range.start)?;
    let end = offset(line_index, range.end)?;
    (start <= end).then(|| TextRange::new(start, end))
}
