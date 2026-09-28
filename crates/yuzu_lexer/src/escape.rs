//! The escapes a string literal may hold: `\n`, `\t`, `\r`, `\0`, `\\` and
//! `\"`.

/// The character an escape stands for, by the character after its `\`.
#[must_use]
pub fn escaped(after_backslash: char) -> Option<char> {
    Some(match after_backslash {
        'n' => '\n',
        't' => '\t',
        'r' => '\r',
        '0' => '\0',
        '\\' => '\\',
        '"' => '"',
        _ => return None,
    })
}

/// The text between a string literal's quotes, with each escape replaced
/// by the character it stands for. An unknown escape is kept as written;
/// [`unknown_escapes`] finds it for a report.
#[must_use]
pub fn unescape(inner: &str) -> String {
    let mut text = String::with_capacity(inner.len());
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            text.push(c);
            continue;
        }
        match chars.next() {
            Some(after) => {
                if let Some(replaced) = escaped(after) {
                    text.push(replaced);
                } else {
                    text.push('\\');
                    text.push(after);
                }
            }
            None => text.push('\\'),
        }
    }
    text
}

/// The byte offset in `inner` and the text of each escape that stands for
/// no character.
pub fn unknown_escapes(inner: &str) -> impl Iterator<Item = (usize, &str)> {
    let mut chars = inner.char_indices();
    std::iter::from_fn(move || {
        while let Some((at, c)) = chars.next() {
            if c != '\\' {
                continue;
            }
            let (next_at, after) = chars.next()?;
            if escaped(after).is_none() {
                return Some((at, &inner[at..next_at + after.len_utf8()]));
            }
        }
        None
    })
}

#[cfg(test)]
mod tests {
    use super::{unescape, unknown_escapes};

    #[test]
    fn each_escape_stands_for_its_character() {
        assert_eq!(unescape(r#"a\"b\n\t\r\0\\c"#), "a\"b\n\t\r\0\\c");
    }

    #[test]
    fn an_unknown_escape_is_kept_and_found() {
        assert_eq!(unescape(r"a\qb"), r"a\qb");
        let found: Vec<(usize, &str)> = unknown_escapes(r"a\qb\n\é").collect();
        assert_eq!(found, [(1, r"\q"), (6, r"\é")]);
    }
}
