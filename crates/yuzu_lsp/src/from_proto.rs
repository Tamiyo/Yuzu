//! Protocol positions to offsets. A client may send a position past the end
//! of the text it describes, so each conversion can fail.

use line_index::{LineCol, WideEncoding, WideLineCol};
use lsp_types::{Position, Range};
use text_size::{TextRange, TextSize};

use crate::line_index::{LineIndex, PositionEncoding};

/// A position's offset. A character after the end of its line means the end
/// of the line, as the protocol says. A line after the end of the text has
/// no offset.
pub(crate) fn offset(line_index: &LineIndex, position: Position) -> Option<TextSize> {
    let line = line_index.index.line(position.line)?;
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
    let end_of_line = if line.end() < line_index.index.len() {
        line.end() - line_index.line_ending_len(position.line)
    } else {
        line.end()
    };
    let offset = line_index.index.offset(line_col)?;
    Some(offset.min(end_of_line))
}

pub(crate) fn text_range(line_index: &LineIndex, range: Range) -> Option<TextRange> {
    let start = offset(line_index, range.start)?;
    let end = offset(line_index, range.end)?;
    (start <= end).then(|| TextRange::new(start, end))
}

#[cfg(test)]
mod tests {
    use lsp_types::Position;
    use text_size::TextSize;

    use super::offset;
    use crate::line_index::{LineIndex, PositionEncoding};

    #[test]
    fn a_character_past_the_line_means_its_end() {
        for encoding in [PositionEncoding::Utf8, PositionEncoding::Utf16] {
            let index = LineIndex::new("let a = 1\nlet b", encoding);
            assert_eq!(
                offset(&index, Position::new(0, 99)),
                Some(TextSize::from(9))
            );
            assert_eq!(
                offset(&index, Position::new(1, 99)),
                Some(TextSize::from(15))
            );
            assert_eq!(offset(&index, Position::new(2, 0)), None);
        }
    }

    #[test]
    fn a_crlf_line_ends_before_its_carriage_return() {
        let index = LineIndex::new("let a = 1\r\nlet b", PositionEncoding::Utf8);
        assert_eq!(
            offset(&index, Position::new(0, 99)),
            Some(TextSize::from(9))
        );
        assert_eq!(
            offset(&index, Position::new(1, 99)),
            Some(TextSize::from(16))
        );
    }
}
