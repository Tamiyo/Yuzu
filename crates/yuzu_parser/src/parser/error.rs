use text_size::TextRange;
use yuzu_diagnostics::{
    diagnostics::{Diagnostic, Span, builder::DiagnosticBuilder},
    source_map::SourceId,
};
use yuzu_lexer::token_kind::TokenKind;

pub(crate) enum ParseError {
    ExpectedKind {
        expected: Vec<TokenKind>,
        found: Option<TokenKind>,
        range: TextRange,
        source_id: SourceId,
    },
    ExpectedExpression {
        found: Option<String>,
        range: TextRange,
        source_id: SourceId,
    },
}

impl From<ParseError> for Diagnostic {
    fn from(val: ParseError) -> Self {
        match val {
            ParseError::ExpectedKind {
                expected,
                found,
                range,
                source_id,
            } => {
                let span = Span { source_id, range };

                let description = match found {
                    Some(kind) => format!("{kind}"),
                    None => String::from("end of input"),
                };

                let expected_description = expected
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");

                let message = if expected.len() == 1 {
                    format!("expected {expected_description}, found {description}")
                } else {
                    format!("expected one of {expected_description}, found {description}")
                };

                DiagnosticBuilder::error(span, message)
                    .primary_label(span, "")
                    .build()
            }
            ParseError::ExpectedExpression {
                found,
                range,
                source_id,
            } => {
                let span = Span { source_id, range };
                // The token's own text, so the reader sees what is actually
                // there. Every other path built this and then substituted an
                // empty string, leaving the message hanging after `found`.
                let message = match &found {
                    Some(text) => format!("expected expression, found `{text}`"),
                    None => "expected expression, found end of input".to_string(),
                };
                DiagnosticBuilder::error(span, message)
                    .primary_label(span, "")
                    .build()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yuzu_diagnostics::diagnostics::Severity;
    use yuzu_diagnostics::source_map::SourceMap;

    fn source_id() -> SourceId {
        SourceMap::new().add(String::new(), String::new())
    }

    #[test]
    fn expected_single_kind() {
        let error = ParseError::ExpectedKind {
            expected: vec![TokenKind::Plus],
            found: Some(TokenKind::LetKw),
            range: TextRange::default(),
            source_id: source_id(),
        };

        let diagnostic: Diagnostic = error.into();
        assert_eq!(diagnostic.severity, Severity::Error);
        assert_eq!(diagnostic.message, "expected +, found let");
    }

    #[test]
    fn expected_one_of_several_kinds() {
        let error = ParseError::ExpectedKind {
            expected: vec![TokenKind::Plus, TokenKind::Minus],
            found: Some(TokenKind::LetKw),
            range: TextRange::default(),
            source_id: source_id(),
        };

        let diagnostic: Diagnostic = error.into();
        assert_eq!(diagnostic.message, "expected one of +, -, found let");
    }

    #[test]
    fn expected_kind_at_end_of_input() {
        let error = ParseError::ExpectedKind {
            expected: vec![TokenKind::Plus],
            found: None,
            range: TextRange::default(),
            source_id: source_id(),
        };

        let diagnostic: Diagnostic = error.into();
        assert_eq!(diagnostic.message, "expected +, found end of input");
    }

    #[test]
    fn expected_expression_at_end_of_input() {
        let error = ParseError::ExpectedExpression {
            found: None,
            range: TextRange::default(),
            source_id: source_id(),
        };

        let diagnostic: Diagnostic = error.into();
        assert_eq!(
            diagnostic.message,
            "expected expression, found end of input"
        );
    }

    /// The token is carried as its own text, so the reader sees what is
    /// there. Nothing covered this, and the message had been substituting an
    /// empty string and trailing off after `found`.
    #[test]
    fn expected_expression_names_the_token_it_found() {
        let error = ParseError::ExpectedExpression {
            found: Some("for".to_string()),
            range: TextRange::default(),
            source_id: source_id(),
        };

        let diagnostic: Diagnostic = error.into();
        assert_eq!(diagnostic.message, "expected expression, found `for`");
    }
}
