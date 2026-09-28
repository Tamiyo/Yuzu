use yuzu_lexer::token_kind::TokenKind;
use yuzu_syntax::SyntaxKind;

use crate::parser::{Parser, marker::CompletedMarker};

mod expr;
mod rel;
mod stmt;
mod ty;

/// Tokens that start a statement. Yuzu has no statement terminator, so an
/// error never takes one of these: the statement it starts survives the
/// mistake before it. `agg` and `external` also start one, but a name can
/// be misspelt as either, and an error that leaves one in place loses the
/// rest of that statement to it.
pub(crate) const STMT_RECOVERY_SET: [TokenKind; 11] = [
    TokenKind::DefKw,
    TokenKind::StructKw,
    TokenKind::TableKw,
    TokenKind::TraitKw,
    TokenKind::ImplKw,
    TokenKind::LetKw,
    TokenKind::ModKw,
    TokenKind::ImportKw,
    TokenKind::ReturnKw,
    TokenKind::PubKw,
    TokenKind::FromKw,
];

/// What a missing token also leaves in place: a brace, so a body survives
/// the mistake in front of it.
pub(crate) const EXPECT_RECOVERY_SET: [TokenKind; 2] =
    [TokenKind::LeftCurly, TokenKind::RightCurly];

pub(crate) fn parse_root(p: &mut Parser) -> CompletedMarker {
    let m = p.start();
    parse_stmts(p, |p| p.at_end());
    p.complete(m, SyntaxKind::Root)
}

/// Statements until `done`. A statement that takes no token has reported
/// its error at that token already, so the token goes into an error node
/// and the next statement starts past it.
pub(crate) fn parse_stmts(p: &mut Parser, done: impl Fn(&mut Parser) -> bool) {
    while !done(p) {
        let before = p.consumed();
        stmt::parse_stmt(p);
        if p.consumed() == before && !done(p) {
            p.bump_as_error();
        }
    }
}

/// What can follow or close around a name. A missing name leaves these in
/// place for the syntax after it.
const NAME_RECOVERY_SET: [TokenKind; 12] = [
    TokenKind::Eq,
    TokenKind::Colon,
    TokenKind::LeftParen,
    TokenKind::RightParen,
    TokenKind::LeftSquare,
    TokenKind::RightSquare,
    TokenKind::LeftCurly,
    TokenKind::RightCurly,
    TokenKind::Comma,
    TokenKind::Dot,
    TokenKind::Arrow,
    TokenKind::Pipe,
];

/// What comes right after a name that opens a declaration's syntax: a
/// keyword before one of these was meant as the name, as in `let from = 1`.
const NAME_FOLLOWERS: [TokenKind; 6] = [
    TokenKind::Eq,
    TokenKind::Colon,
    TokenKind::LeftParen,
    TokenKind::LeftSquare,
    TokenKind::LeftCurly,
    TokenKind::Comma,
];

/// An `Ident` holds exactly its identifier, so a token that is no name is
/// reported and left outside it, and the node is not built.
pub(crate) fn parse_ident(p: &mut Parser) -> Option<CompletedMarker> {
    if p.at(TokenKind::Identifier) {
        let m = p.start();
        p.bump();
        return Some(p.complete(m, SyntaxKind::Ident));
    }

    if at_misused_name(p) {
        p.error_and_bump();
    } else {
        p.error(&NAME_RECOVERY_SET);
    }
    None
}

/// A keyword or literal where a name goes, followed by what follows a name.
fn at_misused_name(p: &mut Parser) -> bool {
    let misplaced = p
        .peek_kind()
        .is_some_and(|kind| !NAME_RECOVERY_SET.contains(&kind));
    let followed = p
        .peek_nth_kind(1)
        .is_some_and(|next| NAME_FOLLOWERS.contains(&next));
    misplaced && followed
}

#[cfg(test)]
mod test_support {
    use expect_test::Expect;
    use yuzu_diagnostics::diagnostics::engine::DiagnosticsEngine;
    use yuzu_diagnostics::source_map::SourceMap;
    use yuzu_lexer::lexer::{Lexer, Token};
    use yuzu_syntax::SyntaxNode;

    use crate::parser::Parser;
    use crate::token_sink::TokenSink;
    use crate::token_source::TokenSource;

    pub(crate) fn check<R>(input: &str, parse: impl FnOnce(&mut Parser) -> R, expected: Expect) {
        let tokens: Vec<Token> = Lexer::new(input).collect();
        let mut sources = SourceMap::new();
        let source_id = sources.add("test".to_string(), input.to_string());

        let mut parser = Parser::new(TokenSource::new(&tokens), source_id);
        parse(&mut parser);
        let events = parser.finish();

        let mut diagnostics = DiagnosticsEngine::new();
        let result = TokenSink::new(&tokens, events, &mut diagnostics).finish();
        let tree = SyntaxNode::new_root(result.green);

        expected.assert_eq(&format!("{tree:#?}"));
    }

    /// The tree and the errors, for a test of how a parse recovers.
    pub(crate) fn check_recovery(input: &str, expected: Expect) {
        let tokens: Vec<Token> = Lexer::new(input).collect();
        let mut sources = SourceMap::new();
        let source_id = sources.add("test".to_string(), input.to_string());
        let mut diagnostics = DiagnosticsEngine::new();
        let tree = crate::parse(&tokens, &mut diagnostics, source_id);

        let errors: Vec<String> = diagnostics
            .diagnostics()
            .iter()
            .map(|diagnostic| {
                format!(
                    "{:?} {}",
                    diagnostic.labels[0].span.range, diagnostic.message
                )
            })
            .collect();
        expected.assert_eq(&format!("{tree:#?}{}", errors.join("\n")));
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use super::{parse_ident, parse_root};
    use crate::grammar::test_support;

    #[test]
    fn parse_ident_directly() {
        test_support::check(
            "foo",
            parse_ident,
            expect![[r#"
            Ident@0..3
              Identifier@0..3 "foo"
        "#]],
        );
    }

    #[test]
    fn parse_root_directly() {
        test_support::check(
            "",
            parse_root,
            expect![[r#"
            Root@0..0
        "#]],
        );
    }

    #[test]
    fn a_missing_name_keeps_the_next_declaration() {
        test_support::check_recovery(
            "let = 1\nstruct P { x: int64 }",
            expect![[r#"
                Root@0..29
                  LetStmt@0..7
                    LetKw@0..3 "let"
                    Space@3..4 " "
                    Eq@4..5 "="
                    Space@5..6 " "
                    IntLiteral@6..7
                      IntLit@6..7 "1"
                  Newline@7..8 "\n"
                  StructStmt@8..29
                    StructKw@8..14 "struct"
                    Space@14..15 " "
                    Ident@15..16
                      Identifier@15..16 "P"
                    Space@16..17 " "
                    LeftCurly@17..18 "{"
                    Space@18..19 " "
                    StructField@19..27
                      Ident@19..20
                        Identifier@19..20 "x"
                      Colon@20..21 ":"
                      Space@21..22 " "
                      NamedTypeAnnotation@22..27
                        Ident@22..27
                          Identifier@22..27 "int64"
                    Space@27..28 " "
                    RightCurly@28..29 "}"
                4..5 expected one of mut, identifier, found ="#]],
        );
    }

    #[test]
    fn an_empty_stage_keeps_the_next_stage() {
        test_support::check_recovery(
            "from t\n|> where\n|> select a",
            expect![[r#"
                Root@0..27
                  ExprStmt@0..27
                    Pipeline@0..27
                      FromSource@0..6
                        FromKw@0..4 "from"
                        Space@4..5 " "
                        Ident@5..6
                          Identifier@5..6 "t"
                      Newline@6..7 "\n"
                      WhereStage@7..15
                        Pipe@7..9 "|>"
                        Space@9..10 " "
                        WhereKw@10..15 "where"
                      Newline@15..16 "\n"
                      SelectStage@16..27
                        Pipe@16..18 "|>"
                        Space@18..19 " "
                        SelectKw@19..25 "select"
                        Space@25..26 " "
                        SelectItem@26..27
                          IdentExpr@26..27
                            Ident@26..27
                              Identifier@26..27 "a"
                16..18 expected expression, found `|>`"#]],
        );
    }

    #[test]
    fn a_missing_argument_keeps_the_closing_paren() {
        test_support::check_recovery(
            "f(a, )",
            expect![[r#"
            Root@0..6
              ExprStmt@0..6
                CallExpr@0..6
                  IdentExpr@0..1
                    Ident@0..1
                      Identifier@0..1 "f"
                  ArgList@1..6
                    LeftParen@1..2 "("
                    IdentExpr@2..3
                      Ident@2..3
                        Identifier@2..3 "a"
                    Comma@3..4 ","
                    Space@4..5 " "
                    RightParen@5..6 ")"
            5..6 expected expression, found `)`"#]],
        );
    }

    #[test]
    fn a_gap_in_a_list_keeps_the_rest() {
        test_support::check_recovery(
            "[1, , 2]",
            expect![[r#"
            Root@0..8
              ExprStmt@0..8
                ListExpr@0..8
                  LeftSquare@0..1 "["
                  IntLiteral@1..2
                    IntLit@1..2 "1"
                  Comma@2..3 ","
                  Space@3..4 " "
                  Comma@4..5 ","
                  Space@5..6 " "
                  IntLiteral@6..7
                    IntLit@6..7 "2"
                  RightSquare@7..8 "]"
            4..5 expected expression, found `,`"#]],
        );
    }

    #[test]
    fn a_stray_paren_is_skipped_at_file_level() {
        test_support::check_recovery(
            ")\nlet x = 1",
            expect![[r#"
            Root@0..11
              ExprStmt@0..0
              Error@0..1
                RightParen@0..1 ")"
              Newline@1..2 "\n"
              LetStmt@2..11
                LetKw@2..5 "let"
                Space@5..6 " "
                Ident@6..7
                  Identifier@6..7 "x"
                Space@7..8 " "
                Eq@8..9 "="
                Space@9..10 " "
                IntLiteral@10..11
                  IntLit@10..11 "1"
            0..1 expected expression, found `)`"#]],
        );
    }

    #[test]
    fn a_stray_paren_is_skipped_in_a_block() {
        test_support::check_recovery(
            "def f() { ) }\nlet x = 1",
            expect![[r#"
            Root@0..23
              FuncStmt@0..13
                DefKw@0..3 "def"
                Space@3..4 " "
                Ident@4..5
                  Identifier@4..5 "f"
                LeftParen@5..6 "("
                RightParen@6..7 ")"
                Space@7..8 " "
                BlockStmt@8..13
                  LeftCurly@8..9 "{"
                  Space@9..10 " "
                  ExprStmt@10..10
                  Error@10..11
                    RightParen@10..11 ")"
                  Space@11..12 " "
                  RightCurly@12..13 "}"
              Newline@13..14 "\n"
              LetStmt@14..23
                LetKw@14..17 "let"
                Space@17..18 " "
                Ident@18..19
                  Identifier@18..19 "x"
                Space@19..20 " "
                Eq@20..21 "="
                Space@21..22 " "
                IntLiteral@22..23
                  IntLit@22..23 "1"
            10..11 expected expression, found `)`"#]],
        );
    }

    #[test]
    fn a_missing_struct_name_keeps_the_body() {
        test_support::check_recovery(
            "struct { x: int64 }",
            expect![[r#"
                Root@0..19
                  StructStmt@0..19
                    StructKw@0..6 "struct"
                    Space@6..7 " "
                    LeftCurly@7..8 "{"
                    Space@8..9 " "
                    StructField@9..17
                      Ident@9..10
                        Identifier@9..10 "x"
                      Colon@10..11 ":"
                      Space@11..12 " "
                      NamedTypeAnnotation@12..17
                        Ident@12..17
                          Identifier@12..17 "int64"
                    Space@17..18 " "
                    RightCurly@18..19 "}"
                7..8 expected identifier, found {"#]],
        );
    }

    #[test]
    fn a_missing_paren_keeps_the_body() {
        test_support::check_recovery(
            "def f(x: int64 { return x }",
            expect![[r#"
            Root@0..27
              FuncStmt@0..27
                DefKw@0..3 "def"
                Space@3..4 " "
                Ident@4..5
                  Identifier@4..5 "f"
                LeftParen@5..6 "("
                FuncParam@6..14
                  Ident@6..7
                    Identifier@6..7 "x"
                  Colon@7..8 ":"
                  Space@8..9 " "
                  NamedTypeAnnotation@9..14
                    Ident@9..14
                      Identifier@9..14 "int64"
                Space@14..15 " "
                BlockStmt@15..27
                  LeftCurly@15..16 "{"
                  Space@16..17 " "
                  ReturnStmt@17..25
                    ReturnKw@17..23 "return"
                    Space@23..24 " "
                    IdentExpr@24..25
                      Ident@24..25
                        Identifier@24..25 "x"
                  Space@25..26 " "
                  RightCurly@26..27 "}"
            15..16 expected one of [, ,, ), found {"#]],
        );
    }

    #[test]
    fn a_keyword_before_a_name_follower_is_the_name() {
        test_support::check_recovery(
            "let def = 1",
            expect![[r#"
                Root@0..11
                  LetStmt@0..11
                    LetKw@0..3 "let"
                    Space@3..4 " "
                    Error@4..7
                      DefKw@4..7 "def"
                    Space@7..8 " "
                    Eq@8..9 "="
                    Space@9..10 " "
                    IntLiteral@10..11
                      IntLit@10..11 "1"
                4..7 expected one of mut, identifier, found def"#]],
        );
    }

    #[test]
    fn a_keyword_that_starts_a_statement_is_not_the_name() {
        test_support::check_recovery(
            "let\ndef f() { return 1 }",
            expect![[r#"
                Root@0..24
                  LetStmt@0..3
                    LetKw@0..3 "let"
                  Newline@3..4 "\n"
                  FuncStmt@4..24
                    DefKw@4..7 "def"
                    Space@7..8 " "
                    Ident@8..9
                      Identifier@8..9 "f"
                    LeftParen@9..10 "("
                    RightParen@10..11 ")"
                    Space@11..12 " "
                    BlockStmt@12..24
                      LeftCurly@12..13 "{"
                      Space@13..14 " "
                      ReturnStmt@14..22
                        ReturnKw@14..20 "return"
                        Space@20..21 " "
                        IntLiteral@21..22
                          IntLit@21..22 "1"
                      Space@22..23 " "
                      RightCurly@23..24 "}"
                4..7 expected one of mut, identifier, found def"#]],
        );
    }
}
