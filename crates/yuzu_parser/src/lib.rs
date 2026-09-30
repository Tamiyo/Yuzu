use yuzu_diagnostics::{DiagnosticsEngine, SourceId};
use yuzu_lexer::lexer::{Lexer, Token};
use yuzu_syntax::SyntaxNode;

use crate::{parser::Parser, token_sink::TokenSink, token_source::TokenSource};

mod grammar;
mod parser;
mod token_set;
mod token_sink;
mod token_source;

/// Lexes and parses a whole text.
pub fn parse_text(
    text: &str,
    diagnostics: &mut DiagnosticsEngine,
    source_id: SourceId,
) -> SyntaxNode {
    let tokens: Vec<Token> = Lexer::new(text).collect();
    parse(&tokens, diagnostics, source_id)
}

/// Parses lexed tokens, trivia included, into a tree that holds every one.
pub fn parse(
    tokens: &[Token],
    diagnostics: &mut DiagnosticsEngine,
    source_id: SourceId,
) -> SyntaxNode {
    let token_source = TokenSource::new(tokens);

    let mut parser = Parser::new(token_source, source_id);
    grammar::parse_root(&mut parser);
    let events = parser.finish();

    SyntaxNode::new_root(TokenSink::new(tokens, events, diagnostics).finish())
}
