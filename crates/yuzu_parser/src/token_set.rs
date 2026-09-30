use yuzu_lexer::token_kind::TokenKind;

const _: () = assert!(
    TokenKind::ALL.len() <= 128,
    "a token set holds at most 128 kinds"
);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TokenSet(u128);

impl TokenSet {
    pub(crate) const EMPTY: TokenSet = TokenSet(0);

    pub(crate) const fn new(kinds: &[TokenKind]) -> Self {
        let mut bits = 0;
        let mut at = 0;
        while at < kinds.len() {
            bits |= mask(kinds[at]);
            at += 1;
        }
        TokenSet(bits)
    }

    pub(crate) const fn union(self, other: TokenSet) -> Self {
        TokenSet(self.0 | other.0)
    }

    pub(crate) const fn contains(self, kind: TokenKind) -> bool {
        self.0 & mask(kind) != 0
    }

    pub(crate) fn insert(&mut self, kind: TokenKind) {
        self.0 |= mask(kind);
    }

    /// The kinds in the set, in the order the lexer declares them.
    pub(crate) fn iter(self) -> impl Iterator<Item = TokenKind> {
        TokenKind::ALL
            .iter()
            .copied()
            .filter(move |&kind| self.contains(kind))
    }
}

const fn mask(kind: TokenKind) -> u128 {
    1 << kind as u16
}

#[cfg(test)]
mod tests {
    use yuzu_lexer::token_kind::TokenKind;

    use super::TokenSet;

    #[test]
    fn a_set_holds_what_it_was_given_in_lexer_order() {
        let mut set = TokenSet::new(&[TokenKind::Comma, TokenKind::Plus]);
        set.insert(TokenKind::Comma);
        assert!(set.contains(TokenKind::Plus) && !set.contains(TokenKind::Minus));
        let kinds: Vec<TokenKind> = set.iter().collect();
        assert_eq!(kinds, [TokenKind::Plus, TokenKind::Comma]);
        assert_eq!(TokenSet::EMPTY.iter().count(), 0);
    }
}
