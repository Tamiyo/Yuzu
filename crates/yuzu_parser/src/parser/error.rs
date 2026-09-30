use text_size::TextRange;
use yuzu_diagnostics::{Diagnostic, DiagnosticBuilder, SourceId, Span};
use yuzu_lexer::token_kind::TokenKind;

use crate::token_set::TokenSet;

pub(crate) enum ParseError {
    ExpectedKind {
        expected: TokenSet,
        found: Option<TokenKind>,
        range: TextRange,
        source_id: SourceId,
    },
    ExpectedExpression {
        found: Option<TokenKind>,
        range: TextRange,
        source_id: SourceId,
    },
    /// `pub` before something that declares nothing.
    ExpectedDeclaration {
        found: Option<TokenKind>,
        range: TextRange,
        source_id: SourceId,
    },
    /// A `\` in a string that starts no escape the language has.
    UnknownEscape {
        escape: String,
        range: TextRange,
        source_id: SourceId,
    },
}

impl From<ParseError> for Diagnostic {
    fn from(val: ParseError) -> Self {
        let (range, source_id, message) = match val {
            ParseError::ExpectedKind {
                expected,
                found,
                range,
                source_id,
            } => {
                let description = describe_found(found);
                let expected_description =
                    expected.iter().map(describe).collect::<Vec<_>>().join(", ");
                let message = if expected.iter().nth(1).is_none() {
                    format!("expected {expected_description}, found {description}")
                } else {
                    format!("expected one of {expected_description}, found {description}")
                };
                (range, source_id, message)
            }
            ParseError::ExpectedExpression {
                found,
                range,
                source_id,
            } => {
                let message = format!("expected expression, found {}", describe_found(found));
                (range, source_id, message)
            }
            ParseError::ExpectedDeclaration {
                found,
                range,
                source_id,
            } => (
                range,
                source_id,
                format!(
                    "expected a declaration after `pub`, found {}",
                    describe_found(found)
                ),
            ),
            ParseError::UnknownEscape {
                escape,
                range,
                source_id,
            } => (
                range,
                source_id,
                format!("unknown escape `{escape}` in a string"),
            ),
        };
        let span = Span { source_id, range };
        DiagnosticBuilder::error(span, message)
            .primary_label(span, "")
            .build()
    }
}

fn describe_found(found: Option<TokenKind>) -> String {
    match found {
        Some(kind) => describe(kind),
        None => "end of input".to_owned(),
    }
}

fn describe(kind: TokenKind) -> String {
    if kind.is_keyword() || kind.is_symbol() {
        format!("`{kind}`")
    } else {
        kind.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yuzu_diagnostics::{Severity, SourceMap};

    fn source_id() -> SourceId {
        SourceMap::new().add(String::new(), String::new())
    }

    fn message(error: ParseError) -> String {
        let diagnostic: Diagnostic = error.into();
        assert_eq!(diagnostic.severity, Severity::Error);
        diagnostic.message
    }

    #[test]
    fn expected_single_kind() {
        let error = ParseError::ExpectedKind {
            expected: TokenSet::new(&[TokenKind::Plus]),
            found: Some(TokenKind::LetKw),
            range: TextRange::default(),
            source_id: source_id(),
        };
        assert_eq!(message(error), "expected `+`, found `let`");
    }

    #[test]
    fn expected_one_of_several_kinds_each_once() {
        let error = ParseError::ExpectedKind {
            expected: TokenSet::new(&[TokenKind::Comma, TokenKind::Plus, TokenKind::Comma]),
            found: Some(TokenKind::Identifier),
            range: TextRange::default(),
            source_id: source_id(),
        };
        assert_eq!(message(error), "expected one of `+`, `,`, found identifier");
    }

    #[test]
    fn expected_kind_at_end_of_input() {
        let error = ParseError::ExpectedKind {
            expected: TokenSet::new(&[TokenKind::Plus]),
            found: None,
            range: TextRange::default(),
            source_id: source_id(),
        };
        assert_eq!(message(error), "expected `+`, found end of input");
    }

    #[test]
    fn expected_expression_at_end_of_input() {
        let error = ParseError::ExpectedExpression {
            found: None,
            range: TextRange::default(),
            source_id: source_id(),
        };
        assert_eq!(message(error), "expected expression, found end of input");
    }

    #[test]
    fn expected_expression_names_the_token_it_found() {
        let error = ParseError::ExpectedExpression {
            found: Some(TokenKind::ForKw),
            range: TextRange::default(),
            source_id: source_id(),
        };
        assert_eq!(message(error), "expected expression, found `for`");
    }
}
