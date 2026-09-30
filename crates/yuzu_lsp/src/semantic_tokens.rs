//! The semantic token legend, and the relative encoding the protocol sends.
//! The builder follows rust-analyzer's `lsp/semantic_tokens.rs`.
//!
//! `boolean`, `table` and `mutable` are not standard; the VS Code extension
//! declares them in its `package.json` beside the type each one falls back to.

use lsp_types::{
    Range, SemanticToken, SemanticTokenModifier, SemanticTokenType, SemanticTokens,
    SemanticTokensEdit,
};
use yuzu_ide::{HlMod, HlMods, HlTag};

const BOOLEAN: SemanticTokenType = SemanticTokenType::new("boolean");
const TABLE: SemanticTokenType = SemanticTokenType::new("table");
const MUTABLE: SemanticTokenModifier = SemanticTokenModifier::new("mutable");

pub(crate) const TYPES: &[SemanticTokenType] = &[
    SemanticTokenType::KEYWORD,
    SemanticTokenType::COMMENT,
    SemanticTokenType::STRING,
    SemanticTokenType::NUMBER,
    BOOLEAN,
    SemanticTokenType::NAMESPACE,
    TABLE,
    SemanticTokenType::STRUCT,
    SemanticTokenType::INTERFACE,
    SemanticTokenType::FUNCTION,
    SemanticTokenType::PARAMETER,
    SemanticTokenType::TYPE_PARAMETER,
    SemanticTokenType::TYPE,
    SemanticTokenType::VARIABLE,
    SemanticTokenType::PROPERTY,
];

pub(crate) const MODIFIERS: &[SemanticTokenModifier] =
    &[SemanticTokenModifier::DECLARATION, MUTABLE];

pub(crate) fn token_type(tag: HlTag) -> u32 {
    let ty = match tag {
        HlTag::Keyword => SemanticTokenType::KEYWORD,
        HlTag::Comment => SemanticTokenType::COMMENT,
        HlTag::StringLiteral => SemanticTokenType::STRING,
        HlTag::NumericLiteral => SemanticTokenType::NUMBER,
        HlTag::BoolLiteral => BOOLEAN,
        HlTag::Module => SemanticTokenType::NAMESPACE,
        HlTag::Table => TABLE,
        HlTag::Struct => SemanticTokenType::STRUCT,
        HlTag::Trait => SemanticTokenType::INTERFACE,
        HlTag::Function => SemanticTokenType::FUNCTION,
        HlTag::Parameter => SemanticTokenType::PARAMETER,
        HlTag::TypeParam => SemanticTokenType::TYPE_PARAMETER,
        HlTag::Type => SemanticTokenType::TYPE,
        HlTag::Local => SemanticTokenType::VARIABLE,
        HlTag::Field => SemanticTokenType::PROPERTY,
    };
    position(TYPES, &ty)
}

pub(crate) fn token_modifiers(mods: HlMods) -> u32 {
    mods.iter().fold(0, |bits, m| {
        let modifier = match m {
            HlMod::Declaration => SemanticTokenModifier::DECLARATION,
            HlMod::Mutable => MUTABLE,
        };
        bits | 1 << position(MODIFIERS, &modifier)
    })
}

fn position<T: PartialEq>(legend: &[T], item: &T) -> u32 {
    let at = legend
        .iter()
        .position(|entry| entry == item)
        .expect("every highlight maps to an entry of the legend");
    u32::try_from(at).expect("the legend has fewer than 2^32 entries")
}

/// Each token is sent relative to the one before it: a line delta, and a
/// column delta when the line did not change.
#[derive(Debug, Default)]
pub(crate) struct SemanticTokensBuilder {
    previous_line: u32,
    previous_start: u32,
    data: Vec<SemanticToken>,
}

impl SemanticTokensBuilder {
    /// `range` lies on one line; the caller splits a token that does not.
    pub(crate) fn push(&mut self, range: Range, token_type: u32, token_modifiers: u32) {
        debug_assert_eq!(
            range.start.line, range.end.line,
            "a pushed token lies on one line"
        );

        let mut delta_line = range.start.line;
        let mut delta_start = range.start.character;
        if !self.data.is_empty() {
            delta_line -= self.previous_line;
            if delta_line == 0 {
                delta_start -= self.previous_start;
            }
        }

        self.data.push(SemanticToken {
            delta_line,
            delta_start,
            length: range.end.character - range.start.character,
            token_type,
            token_modifiers_bitset: token_modifiers,
        });
        self.previous_line = range.start.line;
        self.previous_start = range.start.character;
    }

    pub(crate) fn build(self) -> SemanticTokens {
        SemanticTokens {
            result_id: None,
            data: self.data,
        }
    }
}

/// The one edit that turns `old` into `new`: what lies between the tokens
/// both start with and the tokens both end with. The protocol counts in
/// numbers, and each token is five of them.
pub(crate) fn diff(old: &[SemanticToken], new: &[SemanticToken]) -> Vec<SemanticTokensEdit> {
    let start = new
        .iter()
        .zip(old)
        .take_while(|(new, old)| new == old)
        .count();
    let (old, new) = (&old[start..], &new[start..]);
    let end = new
        .iter()
        .rev()
        .zip(old.iter().rev())
        .take_while(|(new, old)| new == old)
        .count();
    let (old, new) = (&old[..old.len() - end], &new[..new.len() - end]);
    if old.is_empty() && new.is_empty() {
        return Vec::new();
    }
    let numbers = |tokens: usize| u32::try_from(5 * tokens).expect("fewer than 2^32 numbers");
    vec![SemanticTokensEdit {
        start: numbers(start),
        delete_count: numbers(old.len()),
        data: Some(new.to_vec()),
    }]
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use lsp_types::{Position, Range, SemanticToken};

    use super::{SemanticTokensBuilder, diff};

    fn token(delta_start: u32) -> SemanticToken {
        SemanticToken {
            delta_start,
            ..SemanticToken::default()
        }
    }

    #[test]
    fn a_change_in_the_middle_is_one_edit() {
        let old = [token(1), token(2), token(3), token(4)];
        let new = [token(1), token(9), token(9), token(9), token(4)];
        let edits = diff(&old, &new);
        assert_eq!(edits.len(), 1);
        assert_eq!((edits[0].start, edits[0].delete_count), (5, 10));
        assert_eq!(edits[0].data.as_deref(), Some(&new[1..4]));
    }

    #[test]
    fn the_same_tokens_are_no_edit() {
        let tokens = [token(1), token(2)];
        assert!(diff(&tokens, &tokens).is_empty());
    }

    #[test]
    fn tokens_added_at_the_end_delete_nothing() {
        let edits = diff(&[token(1)], &[token(1), token(2)]);
        assert_eq!((edits[0].start, edits[0].delete_count), (5, 0));
    }

    fn on_line(line: u32, start: u32, end: u32) -> Range {
        Range::new(Position::new(line, start), Position::new(line, end))
    }

    #[test]
    fn a_token_is_sent_relative_to_the_one_before() {
        let mut builder = SemanticTokensBuilder::default();
        builder.push(on_line(0, 4, 7), 1, 0);
        builder.push(on_line(0, 10, 12), 2, 1);
        builder.push(on_line(2, 1, 3), 3, 0);
        builder.push(on_line(2, 8, 9), 4, 2);

        let rendered: Vec<String> = builder
            .build()
            .data
            .iter()
            .map(|token| {
                format!(
                    "+{} +{} len {} type {} mods {}",
                    token.delta_line,
                    token.delta_start,
                    token.length,
                    token.token_type,
                    token.token_modifiers_bitset
                )
            })
            .collect();
        expect![[r"
            +0 +4 len 3 type 1 mods 0
            +0 +6 len 2 type 2 mods 1
            +2 +1 len 2 type 3 mods 0
            +0 +7 len 1 type 4 mods 2"]]
        .assert_eq(&rendered.join("\n"));
    }
}
