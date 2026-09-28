use text_size::TextSize;
use yuzu_lexer::{lexer::Token, token_kind::TokenKind};

pub(crate) struct TokenSource<'t, 'input> {
    tokens: &'t [Token<'input>],
    cursor: usize,
}

impl<'t, 'input> TokenSource<'t, 'input> {
    pub(crate) fn new(tokens: &'t [Token<'input>]) -> Self {
        Self { tokens, cursor: 0 }
    }

    pub(crate) fn next_token(&mut self) -> Option<&'t Token<'input>> {
        self.eat_trivia();

        let token = self.tokens.get(self.cursor)?;
        self.cursor += 1;

        Some(token)
    }

    pub(crate) fn peek_kind(&mut self) -> Option<TokenKind> {
        self.eat_trivia();
        self.peek_kind_raw()
    }

    pub(crate) fn peek_nth_kind(&mut self, n: usize) -> Option<TokenKind> {
        self.eat_trivia();
        self.tokens[self.cursor..]
            .iter()
            .map(|token| token.kind)
            .filter(|kind| !kind.is_trivia())
            .nth(n)
    }

    pub(crate) fn peek_token(&mut self) -> Option<&Token<'_>> {
        self.eat_trivia();
        self.peek_token_raw()
    }

    fn eat_trivia(&mut self) {
        while self.at_trivia() {
            self.cursor += 1;
        }
    }

    fn at_trivia(&self) -> bool {
        self.peek_kind_raw().is_some_and(TokenKind::is_trivia)
    }

    /// Where the last token that is not trivia ends: the start when there
    /// is none.
    pub(crate) fn end_of_last_token(&self) -> TextSize {
        self.tokens
            .iter()
            .rev()
            .find(|token| !token.kind.is_trivia())
            .map_or(TextSize::from(0), |token| token.range.end())
    }

    /// Whether the trivia before the next token holds a line break.
    pub(crate) fn newline_before(&mut self) -> bool {
        self.eat_trivia();
        self.tokens[..self.cursor]
            .iter()
            .rev()
            .take_while(|token| token.kind.is_trivia())
            .any(|token| token.kind == TokenKind::Newline)
    }

    fn peek_kind_raw(&self) -> Option<TokenKind> {
        self.peek_token_raw().map(|Token { kind, .. }| *kind)
    }

    fn peek_token_raw(&self) -> Option<&Token<'_>> {
        self.tokens.get(self.cursor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yuzu_lexer::lexer::Lexer;

    fn lex(input: &str) -> Vec<Token<'_>> {
        Lexer::new(input).collect()
    }

    #[test]
    fn peek_kind_skips_trivia_without_consuming() {
        let tokens = lex("  + -");
        let mut source = TokenSource::new(&tokens);

        assert_eq!(source.peek_kind(), Some(TokenKind::Plus));
        assert_eq!(source.peek_kind(), Some(TokenKind::Plus));
    }

    #[test]
    fn next_token_skips_trivia_and_advances() {
        let tokens = lex("  + -");
        let mut source = TokenSource::new(&tokens);

        assert_eq!(source.next_token().map(|t| t.kind), Some(TokenKind::Plus));
        assert_eq!(source.next_token().map(|t| t.kind), Some(TokenKind::Minus));
        assert_eq!(source.next_token().map(|t| t.kind), None);
    }

    #[test]
    fn empty_input_yields_nothing() {
        let tokens = lex("");
        let mut source = TokenSource::new(&tokens);

        assert_eq!(source.peek_kind(), None);
        assert!(source.next_token().is_none());
    }

    #[test]
    fn trivia_only_input_is_treated_as_end() {
        let tokens = lex("   ");
        let mut source = TokenSource::new(&tokens);

        assert_eq!(source.peek_kind(), None);
        assert!(source.next_token().is_none());
    }

    #[test]
    fn the_end_is_after_the_last_token_that_is_not_trivia() {
        let tokens = lex("+ -  // note\n");
        let source = TokenSource::new(&tokens);

        assert_eq!(source.end_of_last_token(), TextSize::from(3));
    }

    #[test]
    fn a_line_break_before_the_next_token_is_seen() {
        let tokens = lex("a b\n  c");
        let mut source = TokenSource::new(&tokens);

        source.next_token();
        assert!(!source.newline_before());
        source.next_token();
        assert!(source.newline_before());
    }
}
