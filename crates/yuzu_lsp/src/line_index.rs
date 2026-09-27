//! A file's line index, with the position encoding the client agreed to.

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
}

impl LineIndex {
    pub(crate) fn new(text: &str, encoding: PositionEncoding) -> Self {
        Self {
            index: line_index::LineIndex::new(text),
            encoding,
        }
    }
}
