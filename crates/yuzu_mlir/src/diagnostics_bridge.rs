//! The bridge from MLIR locations back to the diagnostics engine's spans:
//! the inverse of the locations a conversion mints on the way in.

use melior::ir::Location;
use text_size::{TextRange, TextSize};
use yuzu_diagnostics::diagnostics::Span;
use yuzu_diagnostics::source_map::SourceId;

/// Turns an op's location back into a span the diagnostics printer can
/// render source for.
pub struct DiagnosticsBridge {
    id: SourceId,
    line_starts: Vec<usize>,
    len: usize,
}

impl DiagnosticsBridge {
    pub fn new(id: SourceId, text: &str) -> Self {
        let mut line_starts = vec![0];
        line_starts.extend(text.match_indices('\n').map(|(at, _)| at + 1));
        Self {
            id,
            line_starts,
            len: text.len(),
        }
    }

    /// The span of an op's location. Locations print as `loc("name":line:col)`;
    /// anything else — an unknown location, a fused one — falls back to the
    /// start of the source.
    pub fn span(&self, location: Location) -> Span {
        let offset = self.offset(&location.to_string()).unwrap_or_default();
        let end = (offset + 1).min(self.len);
        Span {
            source_id: self.id,
            range: TextRange::new(TextSize::new(offset as u32), TextSize::new(end as u32)),
        }
    }

    fn offset(&self, printed: &str) -> Option<usize> {
        let (_, tail) = printed.rsplit_once("\":")?;
        let (line, column) = tail.strip_suffix(')')?.split_once(':')?;
        let line: usize = line.parse().ok()?;
        let column: usize = column.parse().ok()?;
        let start = *self.line_starts.get(line.checked_sub(1)?)?;
        Some(start + column - 1)
    }
}
