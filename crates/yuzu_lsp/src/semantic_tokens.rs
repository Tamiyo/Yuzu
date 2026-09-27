//! The semantic token legend, and the relative encoding the protocol sends.
//! The builder follows rust-analyzer's `lsp/semantic_tokens.rs`.
//!
//! `boolean`, `table` and `mutable` are not standard; the VS Code extension
//! declares them in its `package.json` beside the type each one falls back to.

use lsp_types::{Range, SemanticToken, SemanticTokenModifier, SemanticTokenType, SemanticTokens};
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
