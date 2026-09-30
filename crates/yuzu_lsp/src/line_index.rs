//! A file's line index, with the position encoding the client agreed to.

use text_size::TextSize;

/// How a position's character counts: bytes, or UTF-16 code units, the
/// protocol's default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PositionEncoding {
    Utf8,
    Utf16,
}

#[derive(Debug)]
pub(crate) struct LineIndex {
    pub(crate) index: line_index::LineIndex,
    pub(crate) encoding: PositionEncoding,
    /// The lines that end in `\r\n`, in order.
    crlf_lines: Vec<u32>,
}

impl LineIndex {
    pub(crate) fn new(text: &str, encoding: PositionEncoding) -> Self {
        let crlf_lines = text
            .split_inclusive('\n')
            .enumerate()
            .filter(|(_, line)| line.ends_with("\r\n"))
            .map(|(at, _)| u32::try_from(at).expect("a document has fewer than 2^32 lines"))
            .collect();
        Self {
            index: line_index::LineIndex::new(text),
            encoding,
            crlf_lines,
        }
    }

    /// How many bytes end a line: two for `\r\n`, one for `\n`.
    pub(crate) fn line_ending_len(&self, line: u32) -> TextSize {
        if self.crlf_lines.binary_search(&line).is_ok() {
            TextSize::from(2)
        } else {
            TextSize::from(1)
        }
    }
}
