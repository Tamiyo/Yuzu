use std::fmt;
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Hash, Debug)]
pub struct SourceId(usize);

#[derive(Debug)]
pub struct LineCol {
    pub line: usize,
    pub col: usize,
}

/// One source: its name, its text and where its lines start. They are
/// shared, so a map made from another one's sources copies no text.
#[derive(Clone)]
struct Entry {
    name: Arc<str>,
    text: Arc<str>,
    line_starts: Arc<[usize]>,
}

#[derive(Clone)]
pub struct SourceMap {
    entries: Vec<Entry>,
}

impl fmt::Debug for SourceMap {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_list()
            .entries(self.entries.iter().map(|entry| &entry.name))
            .finish()
    }
}

impl Default for SourceMap {
    fn default() -> Self {
        Self::new()
    }
}

impl SourceMap {
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn add(&mut self, name: String, text: String) -> SourceId {
        let line_starts = index_lines(&text).into();
        self.entries.push(Entry {
            name: name.into(),
            text: text.into(),
            line_starts,
        });
        SourceId(self.entries.len())
    }

    /// The number of sources added so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Adds the sources `other` holds past this map's own count, sharing
    /// their text, so each keeps the id it has in `other`. `other` must hold
    /// at least as many sources as this map.
    pub fn extend_from(&mut self, other: &SourceMap) {
        let own = self.entries.len();
        debug_assert!(
            own <= other.entries.len(),
            "a map extends only from one that holds at least its own sources"
        );
        self.entries.extend(other.entries.iter().skip(own).cloned());
    }

    #[must_use]
    pub fn name(&self, source_id: SourceId) -> &str {
        &self.entries[source_id.0 - 1].name
    }

    /// The source added under a name. Positions that reach the compiler from
    /// outside carry the name the file was read under rather than its id, so
    /// this is how they find their way back.
    #[must_use]
    pub fn id(&self, name: &str) -> Option<SourceId> {
        let index = self.entries.iter().position(|entry| &*entry.name == name)?;
        Some(SourceId(index + 1))
    }

    /// The offset a line and column name, both counted from one. The inverse
    /// of [`SourceMap::line_col`]. `None` when the position is past the end
    /// of the source, which a position from outside may well be.
    #[must_use]
    pub fn offset(&self, source_id: SourceId, line: usize, col: usize) -> Option<usize> {
        let entry = self.entries.get(source_id.0.checked_sub(1)?)?;
        let start = *entry.line_starts.get(line.checked_sub(1)?)?;
        let offset = start + col.checked_sub(1)?;
        (offset <= entry.text.len()).then_some(offset)
    }

    #[must_use]
    pub fn text(&self, source_id: SourceId) -> &str {
        &self.entries[source_id.0 - 1].text
    }

    #[must_use]
    pub fn line_col(&self, source_id: SourceId, offset: usize) -> LineCol {
        let entry = &self.entries[source_id.0 - 1];
        let line_index = entry.line_starts.partition_point(|&start| start <= offset) - 1;

        LineCol {
            line: line_index + 1,
            col: offset - entry.line_starts[line_index] + 1,
        }
    }

    #[must_use]
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

#[cfg(test)]
mod tests {
    use super::SourceMap;

    #[test]
    fn an_extended_source_keeps_its_id() {
        let mut library = SourceMap::new();
        library.add("<entry>".to_string(), String::new());
        let shared = library.add("lib.yz".to_string(), "a\nb\n".to_string());

        let mut program = SourceMap::new();
        program.add("main.yz".to_string(), "from t\n".to_string());
        program.extend_from(&library);

        assert_eq!(program.name(shared), "lib.yz");
        assert_eq!(program.line_text(shared, 2), "b");
        assert_eq!(program.len(), 2);
    }
}
