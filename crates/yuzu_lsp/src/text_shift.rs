//! Maps ranges in the text a check read to a document's newer text. A range
//! before the edit stays in its place. A range after the edit moves with
//! it. A range that touches the edit is dropped until the next check.

use text_size::{TextRange, TextSize};

/// How the text a check read maps onto the text a document has now.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TextShift {
    /// Where the old and new texts first differ.
    prefix: TextSize,
    /// Where the part both texts end with starts, in the old text.
    suffix: TextSize,
    old_len: TextSize,
    new_len: TextSize,
}

impl TextShift {
    pub(crate) fn between(old: &str, new: &str) -> Self {
        let prefix = old
            .bytes()
            .zip(new.bytes())
            .take_while(|(a, b)| a == b)
            .count();
        let limit = old.len().min(new.len()) - prefix;
        let common_suffix = old
            .bytes()
            .rev()
            .zip(new.bytes().rev())
            .take(limit)
            .take_while(|(a, b)| a == b)
            .count();
        let size = |len: usize| TextSize::try_from(len).expect("a document is shorter than 4 GiB");
        Self {
            prefix: size(prefix),
            suffix: size(old.len() - common_suffix),
            old_len: size(old.len()),
            new_len: size(new.len()),
        }
    }

    /// Where `range` of the old text is in the new text, when the edit does
    /// not touch it.
    pub(crate) fn map(&self, range: TextRange) -> Option<TextRange> {
        if range.end() <= self.prefix {
            Some(range)
        } else if range.start() >= self.suffix {
            let from_end = self.old_len - range.end();
            let end = self.new_len - from_end;
            Some(TextRange::at(end - range.len(), range.len()))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use text_size::TextRange;

    use super::TextShift;

    fn map(old: &str, new: &str, word: &str) -> Option<String> {
        let start = old.find(word).unwrap();
        let range = TextRange::at(start.try_into().unwrap(), word.len().try_into().unwrap());
        let mapped = TextShift::between(old, new).map(range)?;
        Some(new[mapped].to_owned())
    }

    #[test]
    fn a_range_before_the_edit_stays() {
        assert_eq!(
            map("let a = 1\nlet b = 2\n", "let a = 1\nlet b = 23\n", "a").as_deref(),
            Some("a")
        );
    }

    #[test]
    fn a_range_after_the_edit_moves_with_it() {
        assert_eq!(
            map("let a = 1\nlet b = 2\n", "let a = 100\nlet b = 2\n", "b").as_deref(),
            Some("b")
        );
    }

    #[test]
    fn a_range_the_edit_touches_is_dropped() {
        assert_eq!(map("let abc = 1\n", "let axc = 1\n", "abc"), None);
    }

    #[test]
    fn an_unchanged_text_keeps_every_range() {
        assert_eq!(
            map("let a = a\n", "let a = a\n", "= a").as_deref(),
            Some("= a")
        );
    }
}
