#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct SourceId(usize);

pub struct LineCol {
    pub line: usize,
    pub col: usize,
}

struct Entry {
    name: String,
    text: String,
    line_starts: Vec<usize>,
}

pub struct SourceMap {
    entries: Vec<Entry>,
}

impl Default for SourceMap {
    fn default() -> Self {
        Self::new()
    }
}

impl SourceMap {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn add(&mut self, name: String, text: String) -> SourceId {
        let line_starts = index_lines(&text);
        self.entries.push(Entry {
            name,
            text,
            line_starts,
        });
        SourceId(self.entries.len())
    }

    pub fn name(&self, source_id: SourceId) -> &str {
        &self.entries[source_id.0 - 1].name
    }

    /// The source added under a name. Positions that reach the compiler from
    /// outside carry the name the file was read under rather than its id, so
    /// this is how they find their way back.
    pub fn id(&self, name: &str) -> Option<SourceId> {
        let index = self.entries.iter().position(|entry| entry.name == name)?;
        Some(SourceId(index + 1))
    }

    /// The offset a line and column name, both counted from one. The inverse
    /// of [`SourceMap::line_col`]. `None` when the position is past the end
    /// of the source, which a position from outside may well be.
    pub fn offset(&self, source_id: SourceId, line: usize, col: usize) -> Option<usize> {
        let entry = self.entries.get(source_id.0.checked_sub(1)?)?;
        let start = *entry.line_starts.get(line.checked_sub(1)?)?;
        let offset = start + col.checked_sub(1)?;
        (offset <= entry.text.len()).then_some(offset)
    }

    pub fn text(&self, source_id: SourceId) -> &str {
        &self.entries[source_id.0 - 1].text
    }

    pub fn line_col(&self, source_id: SourceId, offset: usize) -> LineCol {
        let entry = &self.entries[source_id.0 - 1];
        let line_index = entry.line_starts.partition_point(|&start| start <= offset) - 1;

        LineCol {
            line: line_index + 1,
            col: offset - entry.line_starts[line_index] + 1,
        }
    }

    pub fn line_text(&self, source_id: SourceId, line: usize) -> &str {
        let entry = &self.entries[source_id.0 - 1];

        if line == 0 || line > entry.line_starts.len() {
            return "";
        }

        let start = entry.line_starts[line - 1];
        let end = entry
            .line_starts
            .get(line)
            .copied()
            .unwrap_or(entry.text.len());

        let text = &entry.text[start..end];
        let text = text.strip_suffix('\n').unwrap_or(text);
        text.strip_suffix('\r').unwrap_or(text)
    }
}

fn index_lines(text: &str) -> Vec<usize> {
    let mut starts = vec![0];
    starts.extend(text.match_indices('\n').map(|(i, _)| i + 1));
    starts
}
