use logos::Logos;
use std::convert::TryFrom;
use std::ops::Range as StdRange;
use text_size::{TextRange, TextSize};

use crate::token_kind::TokenKind;

pub struct Lexer<'a> {
    inner: logos::Lexer<'a, TokenKind>,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            inner: TokenKind::lexer(input),
        }
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Token<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let kind = self.inner.next()?.unwrap_or(TokenKind::Error);
        let text = self.inner.slice();

        let range = {
            let StdRange { start, end } = self.inner.span();
            let start = TextSize::try_from(start).unwrap();
            let end = TextSize::try_from(end).unwrap();

            TextRange::new(start, end)
        };

        Some(Self::Item { kind, text, range })
    }
}

#[derive(Debug, PartialEq)]
pub struct Token<'a> {
    pub kind: TokenKind,
    pub text: &'a str,
    pub range: TextRange,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(input: &str) -> Vec<TokenKind> {
        Lexer::new(input).map(|token| token.kind).collect()
    }

    fn one(input: &str) -> TokenKind {
        let kinds = kinds(input);
        assert_eq!(
            kinds.len(),
            1,
            "{input:?} did not lex to a single token: {kinds:?}"
        );
        kinds[0]
    }

    #[test]
    fn keywords_win_over_identifiers() {
        let keywords = [
            ("agg", TokenKind::AggKw),
            ("aggregate", TokenKind::AggregateKw),
            ("external", TokenKind::ExternalKw),
            ("and", TokenKind::AndKw),
            ("as", TokenKind::AsKw),
            ("by", TokenKind::ByKw),
            ("def", TokenKind::DefKw),
            ("distinct", TokenKind::DistinctKw),
            ("drop", TokenKind::DropKw),
            ("extend", TokenKind::ExtendKw),
            ("for", TokenKind::ForKw),
            ("from", TokenKind::FromKw),
            ("full", TokenKind::FullKw),
            ("group", TokenKind::GroupKw),
            ("impl", TokenKind::ImplKw),
            ("in", TokenKind::InKw),
            ("inner", TokenKind::InnerKw),
            ("join", TokenKind::JoinKw),
            ("left", TokenKind::LeftKw),
            ("let", TokenKind::LetKw),
            ("limit", TokenKind::LimitKw),
            ("mut", TokenKind::MutKw),
            ("not", TokenKind::NotKw),
            ("offset", TokenKind::OffsetKw),
            ("on", TokenKind::OnKw),
            ("or", TokenKind::OrKw),
            ("pub", TokenKind::PubKw),
            ("rename", TokenKind::RenameKw),
            ("return", TokenKind::ReturnKw),
            ("right", TokenKind::RightKw),
            ("select", TokenKind::SelectKw),
            ("set", TokenKind::SetKw),
            ("struct", TokenKind::StructKw),
            ("table", TokenKind::TableKw),
            ("trait", TokenKind::TraitKw),
            ("using", TokenKind::UsingKw),
            ("where", TokenKind::WhereKw),
            ("true", TokenKind::BoolLit),
            ("false", TokenKind::BoolLit),
        ];

        for (text, kind) in keywords {
            assert_eq!(one(text), kind, "{text:?} should lex as its keyword");
        }
    }

    #[test]
    fn identifiers() {
        assert_eq!(one("foo"), TokenKind::Identifier);
        assert_eq!(one("_bar"), TokenKind::Identifier);
        assert_eq!(one("Baz123"), TokenKind::Identifier);
        assert_eq!(one("letMeIn"), TokenKind::Identifier);
        assert_eq!(one("trueish"), TokenKind::Identifier);
        assert_eq!(one("joined"), TokenKind::Identifier);
        assert_eq!(one("leftmost"), TokenKind::Identifier);
        assert_eq!(one("rightward"), TokenKind::Identifier);
        assert_eq!(one("innermost"), TokenKind::Identifier);
        assert_eq!(one("fullness"), TokenKind::Identifier);
        assert_eq!(one("onto"), TokenKind::Identifier);
        assert_eq!(one("using_key"), TokenKind::Identifier);
        assert_eq!(one("settings"), TokenKind::Identifier);
        assert_eq!(one("limits"), TokenKind::Identifier);
        assert_eq!(one("offsets"), TokenKind::Identifier);
        assert_eq!(one("aggregates"), TokenKind::Identifier);
        assert_eq!(one("agg_of"), TokenKind::Identifier);
        assert_eq!(one("externals"), TokenKind::Identifier);
        assert_eq!(one("groups"), TokenKind::Identifier);
        assert_eq!(one("bypass"), TokenKind::Identifier);
    }

    #[test]
    fn numeric_literals() {
        assert_eq!(one("42"), TokenKind::IntLit);
        assert_eq!(one("3.14"), TokenKind::FloatLit);
        assert_eq!(one("0xFF"), TokenKind::HexLit);
        assert_eq!(one("0b1010"), TokenKind::BinaryLit);
    }

    #[test]
    fn operators_and_punctuation() {
        assert_eq!(
            kinds("+-**==|>->"),
            [
                TokenKind::Plus,
                TokenKind::Minus,
                TokenKind::StarStar,
                TokenKind::EqEq,
                TokenKind::Pipe,
                TokenKind::Arrow,
            ],
        );
    }

    #[test]
    fn trivia_and_unknown() {
        assert_eq!(
            kinds("a // c\nb"),
            [
                TokenKind::Identifier,
                TokenKind::Space,
                TokenKind::Comment,
                TokenKind::Newline,
                TokenKind::Identifier,
            ],
        );
        assert_eq!(one("@"), TokenKind::Error);
    }

    #[test]
    fn token_carries_text_and_range() {
        let tokens: Vec<_> = Lexer::new("let foo").collect();

        assert_eq!(tokens[0].kind, TokenKind::LetKw);
        assert_eq!(tokens[0].text, "let");
        assert_eq!(tokens[0].range, TextRange::new(0.into(), 3.into()));
        assert_eq!(tokens[2].kind, TokenKind::Identifier);
        assert_eq!(tokens[2].text, "foo");
        assert_eq!(tokens[2].range, TextRange::new(4.into(), 7.into()));
    }
}
