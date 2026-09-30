use std::mem;

use text_size::{TextRange, TextSize};
use yuzu_diagnostics::SourceId;
use yuzu_lexer::token_kind::TokenKind;
use yuzu_syntax::SyntaxKind;

use crate::{
    grammar::{EXPECT_RECOVERY_SET, STMT_RECOVERY_SET},
    parser::{
        error::ParseError,
        event::Event,
        marker::{CompletedMarker, Marker},
    },
    token_set::TokenSet,
    token_source::TokenSource,
};

pub(crate) mod error;
pub(crate) mod event;
pub(crate) mod marker;

pub(crate) struct Parser<'t, 'input> {
    source: TokenSource<'t, 'input>,
    events: Vec<Event>,
    expected_kinds: TokenSet,
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
            expected_kinds: TokenSet::EMPTY,
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

    #[expect(
        clippy::needless_pass_by_value,
        reason = "a completed marker is used up when a node is wrapped around it"
    )]
    pub(crate) fn precede(&mut self, completed_marker: CompletedMarker) -> Marker {
        let marker = self.start();

        if let Event::Start {
            ref mut forward_parent,
            ..
        } = self.events[completed_marker.pos]
        {
            *forward_parent = Some(marker.pos - completed_marker.pos);
        } else {
            unreachable!("a completed marker points at the start of its node");
        }

        marker
    }

    /// Drops a node before it is completed, keeping what it held.
    pub(crate) fn abandon(&mut self, mut marker: Marker) {
        marker.defuse();
        if marker.pos == self.events.len() - 1 {
            self.events.pop();
        }
    }

    pub(crate) fn peek_kind(&mut self) -> Option<TokenKind> {
        self.source.peek_kind()
    }

    pub(crate) fn peek_nth_kind(&mut self, n: usize) -> Option<TokenKind> {
        self.source.peek_nth_kind(n)
    }

    pub(crate) fn at(&mut self, kind: TokenKind) -> bool {
        self.expected_kinds.insert(kind);
        self.peek_kind() == Some(kind)
    }

    /// Whether the next token is one of `set`.
    pub(crate) fn at_any(&mut self, set: TokenSet) -> bool {
        self.expected_kinds = self.expected_kinds.union(set);
        self.peek_kind().is_some_and(|kind| set.contains(kind))
    }

    /// Whether a line break comes before the next token.
    pub(crate) fn at_line_start(&mut self) -> bool {
        self.source.at_line_start()
    }

    /// Whether trivia follows the next token.
    pub(crate) fn is_spaced_after(&mut self) -> bool {
        self.source.is_spaced_after()
    }

    pub(crate) fn at_end(&mut self) -> bool {
        self.peek_kind().is_none()
    }

    /// Whether an error leaves the next token in place: a statement's start
    /// always, and whatever `set` adds for the error at hand.
    pub(crate) fn at_recovery_set(&mut self, set: TokenSet) -> bool {
        self.peek_kind()
            .is_some_and(|k| set.contains(k) || STMT_RECOVERY_SET.contains(k))
    }

    pub(crate) fn error(&mut self, set: TokenSet) {
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

    /// Reports what the next token should have been, and leaves it in place
    /// for the syntax after it.
    pub(crate) fn error_in_place(&mut self) {
        self.report_expected_kind();
    }

    pub(crate) fn error_expression(&mut self, set: TokenSet) {
        let (found, range) = self.found();
        self.expected_kinds = TokenSet::EMPTY;

        let error = ParseError::ExpectedExpression {
            found,
            range,
            source_id: self.source_id,
        };
        self.report(range, error);

        if !self.at_recovery_set(set) && !self.at_end() {
            self.bump_as_error();
        }
    }

    /// Reports the token after a `pub` that declares nothing. The token is
    /// left for the statement it starts.
    pub(crate) fn error_declaration(&mut self) {
        let (found, range) = self.found();
        self.expected_kinds = TokenSet::EMPTY;
        let error = ParseError::ExpectedDeclaration {
            found,
            range,
            source_id: self.source_id,
        };
        self.report(range, error);
    }

    /// Reports each escape in the next token, a string, that stands for no
    /// character.
    pub(crate) fn report_unknown_escapes(&mut self) {
        let Some(token) = self.source.peek_token() else {
            return;
        };
        if token.kind != TokenKind::StringLit {
            return;
        }
        let start = token.range.start() + TextSize::of('"');
        let inner = &token.text[1..token.text.len() - 1];
        for (at, escape) in yuzu_lexer::escape::unknown_escapes(inner) {
            let at = start + TextSize::try_from(at).expect("a token is shorter than 4 GiB");
            let range = TextRange::at(at, TextSize::of(escape));
            let error = ParseError::UnknownEscape {
                escape: escape.to_owned(),
                range,
                source_id: self.source_id,
            };
            self.report(range, error);
        }
    }

    /// The kind of the next token and where it is, or the end of input.
    fn found(&mut self) -> (Option<TokenKind>, TextRange) {
        match self.source.peek_token() {
            Some(token) => (Some(token.kind), token.range),
            None => (None, self.end_range()),
        }
    }

    /// An empty range after the last token that is not trivia.
    fn end_range(&self) -> TextRange {
        TextRange::empty(self.source.end_of_last_token())
    }

    pub(crate) fn bump(&mut self) {
        self.expected_kinds = TokenSet::EMPTY;
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
            self.error(EXPECT_RECOVERY_SET);
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
        let (found, range) = self.found();
        let error = ParseError::ExpectedKind {
            expected: mem::take(&mut self.expected_kinds),
            found,
            range,
            source_id: self.source_id,
        };
        self.report(range, error);
    }

    fn report(&mut self, range: TextRange, error: ParseError) {
        if self.last_error_range != Some(range) {
            self.last_error_range = Some(range);
            self.events.push(Event::Error { error });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yuzu_diagnostics::SourceMap;
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

        assert!(p.at_recovery_set(TokenSet::new(&[TokenKind::Plus, TokenKind::Minus])));
        assert!(!p.at_recovery_set(TokenSet::new(&[TokenKind::Minus])));
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
        p.error(TokenSet::EMPTY);

        let events = p.finish();
        assert!(matches!(events[0], Event::Error { .. }));
    }
}
