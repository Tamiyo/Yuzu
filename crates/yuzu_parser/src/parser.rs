use std::mem;

use text_size::TextRange;
use yuzu_diagnostics::source_map::SourceId;
use yuzu_lexer::token_kind::TokenKind;
use yuzu_syntax::SyntaxKind;

use crate::{
    grammar::{EXPECT_RECOVERY_SET, STMT_RECOVERY_SET},
    parser::{
        error::ParseError,
        event::Event,
        marker::{CompletedMarker, Marker},
    },
    token_source::TokenSource,
};

pub(crate) mod error;
pub(crate) mod event;
pub(crate) mod marker;

pub(crate) struct Parser<'t, 'input> {
    source: TokenSource<'t, 'input>,
    events: Vec<Event>,
    expected_kinds: Vec<TokenKind>,
    source_id: SourceId,
    consumed: usize,
    /// Where the last error was reported. A second error at the same token
    /// only restates the first in other words.
    last_error_range: Option<TextRange>,
}

impl<'t, 'input> Parser<'t, 'input> {
    pub(crate) fn new(source: TokenSource<'t, 'input>, source_id: SourceId) -> Self {
        Self {
            source,
            events: Vec::new(),
            expected_kinds: Vec::new(),
            source_id,
            consumed: 0,
            last_error_range: None,
        }
    }

    pub(crate) fn start(&mut self) -> Marker {
        let pos = self.events.len();
        self.events.push(Event::Placeholder);
        Marker::new(pos)
    }

    pub(crate) fn complete(&mut self, mut marker: Marker, kind: SyntaxKind) -> CompletedMarker {
        marker.defuse();
        let event = &mut self.events[marker.pos];

        *event = Event::Start {
            kind,
            forward_parent: None,
        };

        self.events.push(Event::Finish);
        CompletedMarker::new(marker.pos)
    }

    pub(crate) fn precede(&mut self, completed_marker: CompletedMarker) -> Marker {
        let marker = self.start();

        if let Event::Start {
            ref mut forward_parent,
            ..
        } = self.events[completed_marker.pos]
        {
            *forward_parent = Some(marker.pos - completed_marker.pos);
        } else {
            unreachable!();
        }

        marker
    }

    pub(crate) fn peek_kind(&mut self) -> Option<TokenKind> {
        self.source.peek_kind()
    }

    pub(crate) fn peek_nth_kind(&mut self, n: usize) -> Option<TokenKind> {
        self.source.peek_nth_kind(n)
    }

    pub(crate) fn at(&mut self, kind: TokenKind) -> bool {
        self.expected_kinds.push(kind);
        self.peek_kind() == Some(kind)
    }

    pub(crate) fn at_end(&mut self) -> bool {
        self.peek_kind().is_none()
    }

    /// Whether an error leaves the next token in place: a statement's start
    /// always, and whatever `set` adds for the error at hand.
    pub(crate) fn at_recovery_set(&mut self, set: &[TokenKind]) -> bool {
        self.peek_kind()
            .is_some_and(|k| set.contains(&k) || STMT_RECOVERY_SET.contains(&k))
    }

    pub(crate) fn error(&mut self, set: &[TokenKind]) {
        self.report_expected_kind();
        if !self.at_recovery_set(set) && !self.at_end() {
            self.bump_as_error();
        }
    }

    /// Reports the next token and takes it whatever it is, for a caller
    /// that knows it is misplaced even where it would start a statement.
    pub(crate) fn error_and_bump(&mut self) {
        self.report_expected_kind();
        if !self.at_end() {
            self.bump_as_error();
        }
    }

    pub(crate) fn error_expression(&mut self, set: &[TokenKind]) {
        let info = inspect_found_token(&mut self.source);
        self.expected_kinds.clear();

        let error = ParseError::ExpectedExpression {
            found: info.text,
            range: info.range,
            source_id: self.source_id,
        };
        self.report(info.range, error);

        if !self.at_recovery_set(set) && !self.at_end() {
            self.bump_as_error();
        }
    }

    pub(crate) fn bump(&mut self) {
        self.expected_kinds.clear();
        self.source.next_token();
        self.events.push(Event::Token);
        self.consumed += 1;
    }

    /// Takes the next token into an error node, for an error already
    /// reported against it.
    pub(crate) fn bump_as_error(&mut self) {
        let m = self.start();
        self.bump();
        self.complete(m, SyntaxKind::Error);
    }

    pub(crate) fn expect(&mut self, kind: TokenKind) {
        if self.at(kind) {
            self.bump();
        } else {
            self.error(&EXPECT_RECOVERY_SET);
        }
    }

    /// How many tokens the parse has taken so far.
    pub(crate) fn consumed(&self) -> usize {
        self.consumed
    }

    pub(crate) fn finish(self) -> Vec<Event> {
        self.events
    }

    fn report_expected_kind(&mut self) {
        let info = inspect_found_token(&mut self.source);
        let error = ParseError::ExpectedKind {
            expected: mem::take(&mut self.expected_kinds),
            found: info.kind,
            range: info.range,
            source_id: self.source_id,
        };
        self.report(info.range, error);
    }

    fn report(&mut self, range: TextRange, error: ParseError) {
        if self.last_error_range != Some(range) {
            self.last_error_range = Some(range);
            self.events.push(Event::Error { error });
        }
    }
}

struct FoundTokenInfo {
    kind: Option<TokenKind>,
    text: Option<String>,
    range: TextRange,
}

fn inspect_found_token(source: &mut TokenSource) -> FoundTokenInfo {
    if let Some(token) = source.peek_token() {
        return FoundTokenInfo {
            kind: Some(token.kind),
            text: Some(token.text.to_string()),
            range: token.range,
        };
    }

    let range = match source.last_token_range() {
        Some(range) => range,
        None => TextRange::new(0.into(), 0.into()),
    };

    FoundTokenInfo {
        kind: None,
        text: None,
        range,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yuzu_diagnostics::source_map::SourceMap;
    use yuzu_lexer::lexer::{Lexer, Token};

    fn source_id() -> SourceId {
        SourceMap::new().add(String::new(), String::new())
    }

    fn parser<'t, 'input>(tokens: &'t [Token<'input>]) -> Parser<'t, 'input> {
        Parser::new(TokenSource::new(tokens), source_id())
    }

    #[test]
    fn at_checks_the_next_kind() {
        let tokens: Vec<_> = Lexer::new("+").collect();
        let mut p = parser(&tokens);

        assert!(p.at(TokenKind::Plus));
        assert!(!p.at(TokenKind::Minus));
    }

    #[test]
    fn at_end_detects_end_of_input() {
        let tokens: Vec<_> = Lexer::new("").collect();
        let mut p = parser(&tokens);

        assert!(p.at_end());
    }

    #[test]
    fn at_recovery_set_matches_the_next_kind() {
        let tokens: Vec<_> = Lexer::new("+").collect();
        let mut p = parser(&tokens);

        assert!(p.at_recovery_set(&[TokenKind::Plus, TokenKind::Minus]));
        assert!(!p.at_recovery_set(&[TokenKind::Minus]));
    }

    #[test]
    fn bump_consumes_a_token_and_records_an_event() {
        let tokens: Vec<_> = Lexer::new("+ -").collect();
        let mut p = parser(&tokens);

        p.bump();
        assert!(p.at(TokenKind::Minus));

        let events = p.finish();
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], Event::Token));
    }

    #[test]
    fn start_then_complete_wraps_in_start_and_finish() {
        let tokens: Vec<_> = Lexer::new("+").collect();
        let mut p = parser(&tokens);

        let m = p.start();
        p.bump();
        p.complete(m, SyntaxKind::Error);

        let events = p.finish();
        assert_eq!(events.len(), 3);
        assert!(matches!(
            events[0],
            Event::Start {
                kind: SyntaxKind::Error,
                forward_parent: None
            }
        ));
        assert!(matches!(events[1], Event::Token));
        assert!(matches!(events[2], Event::Finish));
    }

    #[test]
    fn precede_links_a_forward_parent() {
        let tokens: Vec<_> = Lexer::new("+").collect();
        let mut p = parser(&tokens);

        let inner = {
            let m = p.start();
            p.bump();
            p.complete(m, SyntaxKind::IdentExpr)
        };
        let outer = p.precede(inner);
        p.complete(outer, SyntaxKind::BinaryExpr);

        let events = p.finish();
        // The inner Start now points forward to the outer Start.
        assert!(matches!(
            events[0],
            Event::Start {
                forward_parent: Some(_),
                ..
            }
        ));
        assert!(matches!(
            events[3],
            Event::Start {
                kind: SyntaxKind::BinaryExpr,
                forward_parent: None
            }
        ));
    }

    #[test]
    fn error_records_an_error_event() {
        let tokens: Vec<_> = Lexer::new("-").collect();
        let mut p = parser(&tokens);

        p.at(TokenKind::Plus);
        p.error(&[]);

        let events = p.finish();
        assert!(matches!(events[0], Event::Error { .. }));
    }
}
