use std::mem;

use rowan::{GreenNode, GreenNodeBuilder, Language};
use yuzu_diagnostics::diagnostics::{Diagnostic, engine::DiagnosticsEngine};
use yuzu_lexer::{lexer::Token, token_kind::TokenKind};
use yuzu_syntax::{SyntaxKind, YuzuLanguage};

use crate::parser::event::Event;

pub(crate) struct TokenSink<'t, 'input> {
    builder: GreenNodeBuilder<'t>,
    tokens: &'t [Token<'input>],
    cursor: usize,
    events: Vec<Event>,
    diagnostics: &'t mut DiagnosticsEngine,
    /// The kinds of the nodes a start opens, reused from one start to the
    /// next.
    opening: Vec<SyntaxKind>,
}

impl<'t, 'input> TokenSink<'t, 'input> {
    pub(crate) fn new(
        tokens: &'t [Token<'input>],
        events: Vec<Event>,
        diagnostics: &'t mut DiagnosticsEngine,
    ) -> Self {
        Self {
            builder: GreenNodeBuilder::default(),
            tokens,
            cursor: 0,
            events,
            diagnostics,
            opening: Vec::new(),
        }
    }

    pub(crate) fn finish(mut self) -> GreenNode {
        let mut depth = 0usize;
        for idx in 0..self.events.len() {
            match mem::replace(&mut self.events[idx], Event::Placeholder) {
                Event::Start {
                    kind,
                    forward_parent,
                } => {
                    if depth > 0 {
                        self.bump_trivia();
                    }
                    depth += self.start(idx, kind, forward_parent);
                }
                Event::Token => {
                    self.bump_trivia();
                    self.token();
                }
                Event::Finish => {
                    if depth == 1 {
                        self.bump_trivia();
                    }
                    depth -= 1;
                    self.builder.finish_node();
                }
                Event::Error { error } => {
                    let diagnostic: Diagnostic = error.into();
                    self.diagnostics.emit(diagnostic);
                }
                Event::Placeholder => {}
            }
        }

        self.builder.finish()
    }

    fn start(&mut self, idx: usize, kind: SyntaxKind, forward_parent: Option<usize>) -> usize {
        self.opening.clear();
        self.opening.push(kind);

        let mut idx = idx;
        let mut forward_parent = forward_parent;

        // Walk through the forward parent of the forward parent and the forward parent
        // of that, and of that, etc. until we reach a StartNode event without a forward
        // parent.
        while let Some(fp) = forward_parent {
            idx += fp;

            forward_parent = if let Event::Start {
                kind,
                forward_parent,
            } = mem::replace(&mut self.events[idx], Event::Placeholder)
            {
                self.opening.push(kind);
                forward_parent
            } else {
                unreachable!("a forward parent points at the start of a node")
            };
        }

        for &kind in self.opening.iter().rev() {
            self.builder.start_node(YuzuLanguage::kind_to_raw(kind));
        }
        self.opening.len()
    }

    fn token(&mut self) {
        let Token { kind, text, .. } = self.tokens[self.cursor];
        self.builder
            .token(YuzuLanguage::kind_to_raw(kind.into()), text);
        self.cursor += 1;
    }

    fn bump_trivia(&mut self) {
        while self.cursor < self.tokens.len() {
            let token = &self.tokens[self.cursor];
            if !TokenKind::is_trivia(token.kind) {
                break;
            }

            self.token();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use text_size::TextRange;
    use yuzu_diagnostics::source_map::SourceMap;
    use yuzu_lexer::lexer::Lexer;
    use yuzu_syntax::SyntaxNode;

    use crate::parser::error::ParseError;

    #[test]
    fn builds_a_tree_from_events() {
        let tokens: Vec<_> = Lexer::new("+").collect();
        let events = vec![
            Event::Start {
                kind: SyntaxKind::Root,
                forward_parent: None,
            },
            Event::Token,
            Event::Finish,
        ];
        let mut diagnostics = DiagnosticsEngine::new();

        let green = TokenSink::new(&tokens, events, &mut diagnostics).finish();
        let root = SyntaxNode::new_root(green);

        assert_eq!(root.kind(), SyntaxKind::Root);
        assert_eq!(root.text().to_string(), "+");
        assert!(diagnostics.diagnostics().is_empty());
    }

    #[test]
    fn error_events_are_emitted_as_diagnostics() {
        let tokens: Vec<_> = Lexer::new("+").collect();
        let source_id = {
            let mut sources = SourceMap::new();
            sources.add(String::new(), String::new())
        };
        let error = ParseError::ExpectedExpression {
            found: None,
            range: TextRange::default(),
            source_id,
        };
        let events = vec![
            Event::Start {
                kind: SyntaxKind::Root,
                forward_parent: None,
            },
            Event::Error { error },
            Event::Finish,
        ];
        let mut diagnostics = DiagnosticsEngine::new();

        let _ = TokenSink::new(&tokens, events, &mut diagnostics).finish();

        assert_eq!(diagnostics.diagnostics().len(), 1);
    }
}
