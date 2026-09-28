//! The ranges that "expand selection" steps through: the token at a
//! position, then each node that holds it, out to the whole file.

use rowan::TokenAtOffset;
use text_size::{TextRange, TextSize};
use yuzu_syntax::{SyntaxNode, SyntaxToken};

pub(crate) fn selection_ranges(root: &SyntaxNode, offset: TextSize) -> Vec<TextRange> {
    let Some(token) = pick_token(root.token_at_offset(offset)) else {
        return vec![root.text_range()];
    };

    let mut ranges = vec![token.text_range()];
    for node in token.parent_ancestors() {
        if ranges.last() != Some(&node.text_range()) {
            ranges.push(node.text_range());
        }
    }
    ranges
}

/// Between two tokens, the one that is not whitespace or a comment.
fn pick_token(tokens: TokenAtOffset<SyntaxToken>) -> Option<SyntaxToken> {
    match tokens {
        TokenAtOffset::None => None,
        TokenAtOffset::Single(token) => Some(token),
        TokenAtOffset::Between(left, right) if right.kind().is_trivia() => Some(left),
        TokenAtOffset::Between(_, right) => Some(right),
    }
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};

    use crate::test_support::{analysis, at, cursor};

    fn check(fixture: &str, expected: &Expect) {
        let (text, offset) = cursor(fixture);
        let rendered: Vec<&str> = analysis(&text)
            .selection_ranges(at(offset))
            .unwrap()
            .iter()
            .map(|&range| &text[range])
            .collect();
        expected.assert_debug_eq(&rendered);
    }

    #[test]
    fn widens_from_the_name_to_the_file() {
        check(
            "def f(x: int64) -> int64 { return x + $0y }\nlet z = 1",
            &expect![[r#"
                [
                    "y",
                    "x + y",
                    "return x + y",
                    "{ return x + y }",
                    "def f(x: int64) -> int64 { return x + y }",
                    "def f(x: int64) -> int64 { return x + y }\nlet z = 1",
                ]
            "#]],
        );
    }
}
