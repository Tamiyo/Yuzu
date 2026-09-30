use yuzu_lexer::token_kind::TokenKind;
use yuzu_syntax::SyntaxKind;

use crate::parser::{Parser, marker::CompletedMarker};
use crate::token_set::TokenSet;

mod expr;
mod rel;
mod stmt;
mod ty;

/// Tokens that start a statement. Yuzu has no statement terminator, so an
/// error never takes one of these: the statement it starts survives the
/// mistake before it. `agg` and `external` also start one, but a name can
/// be misspelt as either, and an error that leaves one in place loses the
/// rest of that statement to it.
pub(crate) const STMT_RECOVERY_SET: TokenSet = TokenSet::new(&[
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
]);

/// What a missing token also leaves in place: a brace, so a body survives
/// the mistake in front of it.
pub(crate) const EXPECT_RECOVERY_SET: TokenSet =
    TokenSet::new(&[TokenKind::LeftCurly, TokenKind::RightCurly]);

#[expect(
    clippy::redundant_closure_for_method_calls,
    reason = "`Parser::at_end` does not satisfy the closure bound for every lifetime"
)]
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

/// Whether a bracketed list may end in a comma.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Trailing {
    Allowed,
    Forbidden,
}

/// The closing brackets and `{`. A list stops at any of them, so it never
/// takes the close of the syntax around it or the body after it.
const LIST_ENDS: TokenSet = TokenSet::new(&[
    TokenKind::RightParen,
    TokenKind::RightSquare,
    TokenKind::RightCurly,
    TokenKind::LeftCurly,
]);

/// [`delimited`] for a list that needs an item, such as generics: an empty
/// one is reported where its first item should be.
pub(crate) fn delimited_non_empty(
    p: &mut Parser,
    close: TokenKind,
    trailing: Trailing,
    mut item: impl FnMut(&mut Parser),
) {
    // Peeked, not asked: `close` is no item, so it joins no expected set.
    if p.peek_kind() == Some(close) {
        item(p);
    }
    delimited(p, close, trailing, item);
}

/// The items of a bracketed list, up to the `close` the caller then takes.
/// A missing comma is reported and the next item still parses, and a token
/// no item can start is reported and skipped.
pub(crate) fn delimited(
    p: &mut Parser,
    close: TokenKind,
    trailing: Trailing,
    mut item: impl FnMut(&mut Parser),
) {
    if p.at(close) {
        return;
    }
    loop {
        let before = p.consumed();
        item(p);
        if p.at(TokenKind::Comma) {
            p.bump();
            if trailing == Trailing::Allowed && p.at(close) {
                return;
            }
            continue;
        }
        let at_end = p.peek_kind().is_none_or(|kind| LIST_ENDS.contains(kind));
        if p.at(close) || at_end || p.at_recovery_set(TokenSet::EMPTY) {
            return;
        }
        if p.consumed() == before {
            p.error(TokenSet::EMPTY);
        } else {
            p.error_in_place();
        }
    }
}

/// What can follow or close around a name. A missing name leaves these in
/// place for the syntax after it.
const NAME_RECOVERY_SET: TokenSet = TokenSet::new(&[
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
]);

/// What comes right after a name that opens a declaration's syntax: a
/// keyword before one of these was meant as the name, as in `let from = 1`.
const NAME_FOLLOWERS: TokenSet = TokenSet::new(&[
    TokenKind::Eq,
    TokenKind::Colon,
    TokenKind::LeftParen,
    TokenKind::LeftSquare,
    TokenKind::LeftCurly,
    TokenKind::Comma,
]);

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
        p.error(NAME_RECOVERY_SET);
    }
    None
}

/// A keyword or literal where a name goes, followed by what follows a name.
fn at_misused_name(p: &mut Parser) -> bool {
    let misplaced = p
        .peek_kind()
        .is_some_and(|kind| !NAME_RECOVERY_SET.contains(kind));
    let followed = p
        .peek_nth_kind(1)
        .is_some_and(|next| NAME_FOLLOWERS.contains(next));
    misplaced && followed
}

#[cfg(test)]
mod test_support {
    use expect_test::Expect;
    use yuzu_diagnostics::{DiagnosticsEngine, SourceMap};
    use yuzu_lexer::lexer::{Lexer, Token};
    use yuzu_syntax::SyntaxNode;

    use crate::parser::Parser;
    use crate::token_sink::TokenSink;
    use crate::token_source::TokenSource;

    pub(crate) fn check<R>(input: &str, parse: impl FnOnce(&mut Parser) -> R, expected: &Expect) {
        let tokens: Vec<Token> = Lexer::new(input).collect();
        let mut sources = SourceMap::new();
        let source_id = sources.add("test".to_owned(), input.to_owned());

        let mut parser = Parser::new(TokenSource::new(&tokens), source_id);
        parse(&mut parser);
        let events = parser.finish();

        let mut diagnostics = DiagnosticsEngine::new();
        let green = TokenSink::new(&tokens, events, &mut diagnostics).finish();
        let tree = SyntaxNode::new_root(green);

        expected.assert_eq(&format!("{tree:#?}"));
    }

    pub(crate) fn check_recovery(input: &str, expected: &Expect) {
        let mut diagnostics = DiagnosticsEngine::new();
        let source_id = SourceMap::new().add("test".to_owned(), input.to_owned());
        let tree = crate::parse_text(input, &mut diagnostics, source_id);

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

    pub(crate) fn check_outline(input: &str, expected: &Expect) {
        use std::fmt::Write as _;

        let mut diagnostics = DiagnosticsEngine::new();
        let source_id = SourceMap::new().add("test".to_owned(), input.to_owned());
        let tree = crate::parse_text(input, &mut diagnostics, source_id);

        let mut outline = String::new();
        let mut depth = 0;
        for event in tree.preorder() {
            match event {
                rowan::WalkEvent::Enter(node) => {
                    writeln!(outline, "{}{:?}", "  ".repeat(depth), node.kind()).unwrap();
                    depth += 1;
                }
                rowan::WalkEvent::Leave(_) => depth -= 1,
            }
        }
        for diagnostic in diagnostics.diagnostics() {
            writeln!(
                outline,
                "{:?} {}",
                diagnostic.labels[0].span.range, diagnostic.message
            )
            .unwrap();
        }
        expected.assert_eq(&outline);
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use super::{parse_ident, parse_root};
    use crate::grammar::test_support;
    use crate::grammar::test_support::check_outline;

    #[test]
    fn a_missing_comma_keeps_the_next_parameter_and_the_body() {
        check_outline(
            "def f(a: int64 b: int64) -> int64 { return a }\n",
            &expect![[r"
            Root
              FuncStmt
                Ident
                FuncParam
                  Ident
                  NamedTypeAnnotation
                    Ident
                FuncParam
                  Ident
                  NamedTypeAnnotation
                    Ident
                NamedTypeAnnotation
                  Ident
                BlockStmt
                  ReturnStmt
                    IdentExpr
                      Ident
            15..16 expected one of `)`, `[`, `,`, found identifier
        "]],
        );
    }

    #[test]
    fn a_stray_statement_in_an_impl_keeps_the_next_method() {
        check_outline(
            "impl T { let x = 1 def g() -> int64 { return 1 } }\nlet y = 2\n",
            &expect![[r"
                Root
                  ImplStmt
                    Ident
                    Error
                    FuncStmt
                      Ident
                      NamedTypeAnnotation
                        Ident
                      BlockStmt
                        ReturnStmt
                          IntLiteral
                  LetStmt
                    Ident
                    IntLiteral
                9..12 expected one of `}`, `agg`, `def`, `external`, `pub`, found `let`
            "]],
        );
    }

    #[test]
    fn a_bare_alias_stays_on_its_line() {
        check_outline(
            "let q = from t\ng(1)\n",
            &expect![[r"
            Root
              LetStmt
                Ident
                Pipeline
                  FromSource
                    Ident
              ExprStmt
                CallExpr
                  IdentExpr
                    Ident
                  ArgList
                    IntLiteral
        "]],
        );
    }

    #[test]
    fn pub_before_no_declaration_is_reported_at_what_follows() {
        check_outline(
            "pub impl T {}\n",
            &expect![[r"
                Root
                  Error
                  ImplStmt
                    Ident
                4..8 expected a declaration after `pub`, found `impl`
            "]],
        );
    }

    #[test]
    fn a_method_starts_as_a_function_does() {
        check_outline(
            "impl Show for int64 {\n    pub def a() -> int64 { return 1 }\n    agg def b(x: int64) -> int64 { return x }\n    external def c() -> str\n}\n",
            &expect![[r"
                Root
                  ImplStmt
                    TraitRef
                      Ident
                    Ident
                    FuncStmt
                      Ident
                      NamedTypeAnnotation
                        Ident
                      BlockStmt
                        ReturnStmt
                          IntLiteral
                    FuncStmt
                      Ident
                      FuncParam
                        Ident
                        NamedTypeAnnotation
                          Ident
                      NamedTypeAnnotation
                        Ident
                      BlockStmt
                        ReturnStmt
                          IdentExpr
                            Ident
                    FuncStmt
                      Ident
                      NamedTypeAnnotation
                        Ident
            "]],
        );
    }

    #[test]
    fn a_narrowed_pub_before_no_declaration_is_one_error() {
        check_outline(
            "pub(mod) impl T {}\n",
            &expect![[r"
                Root
                  Error
                  ImplStmt
                    Ident
                9..13 expected a declaration after `pub`, found `impl`
            "]],
        );
    }

    #[test]
    fn an_empty_type_parameter_list_expects_a_parameter() {
        check_outline(
            "def f[]() -> int64 { return 1 }\n",
            &expect![[r"
            Root
              FuncStmt
                Ident
                TypeParam
                NamedTypeAnnotation
                  Ident
                BlockStmt
                  ReturnStmt
                    IntLiteral
            6..7 expected identifier, found `]`
        "]],
        );
    }

    #[test]
    fn an_empty_using_list_expects_a_column() {
        check_outline(
            "from t |> join u using ()\n",
            &expect![[r"
            Root
              ExprStmt
                Pipeline
                  FromSource
                    Ident
                  JoinStage
                    Ident
                    JoinUsing
            24..25 expected identifier, found `)`
        "]],
        );
    }

    #[test]
    fn an_empty_type_argument_list_expects_a_type() {
        check_outline(
            "def f(m: Map[]) -> int64 { return 1 }\n",
            &expect![[r"
                Root
                  FuncStmt
                    Ident
                    FuncParam
                      Ident
                      NamedTypeAnnotation
                        Ident
                        NamedTypeAnnotation
                    NamedTypeAnnotation
                      Ident
                    BlockStmt
                      ReturnStmt
                        IntLiteral
                13..14 expected identifier, found `]`
            "]],
        );
    }

    #[test]
    fn a_function_returns_a_query() {
        check_outline(
            "def f() -> int64 { return from t }\n",
            &expect![[r"
            Root
              FuncStmt
                Ident
                NamedTypeAnnotation
                  Ident
                BlockStmt
                  ReturnStmt
                    Pipeline
                      FromSource
                        Ident
        "]],
        );
    }

    #[test]
    fn an_unknown_escape_is_reported_where_it_is() {
        check_outline(
            "let s = \"a\\qb\\n\"\n",
            &expect![[r"
            Root
              LetStmt
                Ident
                StringLiteral
            10..12 unknown escape `\q` in a string
        "]],
        );
    }

    #[test]
    fn the_end_of_input_is_after_the_last_token_not_the_comment() {
        check_outline(
            "let x =\n// a note\n",
            &expect![[r"
            Root
              LetStmt
                Ident
            7..7 expected expression, found end of input
        "]],
        );
    }

    #[test]
    fn tabs_are_whitespace() {
        check_outline(
            "def\tf() -> int64 {\n\treturn 1\n}\n",
            &expect![[r"
            Root
              FuncStmt
                Ident
                NamedTypeAnnotation
                  Ident
                BlockStmt
                  ReturnStmt
                    IntLiteral
        "]],
        );
    }

    #[test]
    fn parse_ident_directly() {
        test_support::check(
            "foo",
            parse_ident,
            &expect![[r#"
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
            &expect![[r"
            Root@0..0
        "]],
        );
    }

    #[test]
    fn a_missing_name_keeps_the_next_declaration() {
        test_support::check_recovery(
            "let = 1\nstruct P { x: int64 }",
            &expect![[r#"
                Root@0..29
                  LetStmt@0..7
                    LetKw@0..3 "let"
                    Whitespace@3..4 " "
                    Eq@4..5 "="
                    Whitespace@5..6 " "
                    IntLiteral@6..7
                      IntLit@6..7 "1"
                  Newline@7..8 "\n"
                  StructStmt@8..29
                    StructKw@8..14 "struct"
                    Whitespace@14..15 " "
                    Ident@15..16
                      Identifier@15..16 "P"
                    Whitespace@16..17 " "
                    LeftCurly@17..18 "{"
                    Whitespace@18..19 " "
                    StructField@19..27
                      Ident@19..20
                        Identifier@19..20 "x"
                      Colon@20..21 ":"
                      Whitespace@21..22 " "
                      NamedTypeAnnotation@22..27
                        Ident@22..27
                          Identifier@22..27 "int64"
                    Whitespace@27..28 " "
                    RightCurly@28..29 "}"
                4..5 expected one of `mut`, identifier, found `=`"#]],
        );
    }

    #[test]
    fn an_empty_stage_keeps_the_next_stage() {
        test_support::check_recovery(
            "from t\n|> where\n|> select a",
            &expect![[r#"
                Root@0..27
                  ExprStmt@0..27
                    Pipeline@0..27
                      FromSource@0..6
                        FromKw@0..4 "from"
                        Whitespace@4..5 " "
                        Ident@5..6
                          Identifier@5..6 "t"
                      Newline@6..7 "\n"
                      WhereStage@7..15
                        Pipe@7..9 "|>"
                        Whitespace@9..10 " "
                        WhereKw@10..15 "where"
                      Newline@15..16 "\n"
                      SelectStage@16..27
                        Pipe@16..18 "|>"
                        Whitespace@18..19 " "
                        SelectKw@19..25 "select"
                        Whitespace@25..26 " "
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
            &expect![[r#"
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
                        Whitespace@4..5 " "
                        RightParen@5..6 ")"
                5..6 expected expression, found `)`"#]],
        );
    }

    #[test]
    fn a_gap_in_a_list_keeps_the_rest() {
        test_support::check_recovery(
            "[1, , 2]",
            &expect![[r#"
                Root@0..8
                  ExprStmt@0..8
                    ListExpr@0..8
                      LeftSquare@0..1 "["
                      IntLiteral@1..2
                        IntLit@1..2 "1"
                      Comma@2..3 ","
                      Whitespace@3..4 " "
                      Comma@4..5 ","
                      Whitespace@5..6 " "
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
            &expect![[r#"
                Root@0..11
                  Error@0..1
                    RightParen@0..1 ")"
                  Newline@1..2 "\n"
                  LetStmt@2..11
                    LetKw@2..5 "let"
                    Whitespace@5..6 " "
                    Ident@6..7
                      Identifier@6..7 "x"
                    Whitespace@7..8 " "
                    Eq@8..9 "="
                    Whitespace@9..10 " "
                    IntLiteral@10..11
                      IntLit@10..11 "1"
                0..1 expected expression, found `)`"#]],
        );
    }

    #[test]
    fn a_stray_paren_is_skipped_in_a_block() {
        test_support::check_recovery(
            "def f() { ) }\nlet x = 1",
            &expect![[r#"
                Root@0..23
                  FuncStmt@0..13
                    DefKw@0..3 "def"
                    Whitespace@3..4 " "
                    Ident@4..5
                      Identifier@4..5 "f"
                    LeftParen@5..6 "("
                    RightParen@6..7 ")"
                    Whitespace@7..8 " "
                    BlockStmt@8..13
                      LeftCurly@8..9 "{"
                      Whitespace@9..10 " "
                      Error@10..11
                        RightParen@10..11 ")"
                      Whitespace@11..12 " "
                      RightCurly@12..13 "}"
                  Newline@13..14 "\n"
                  LetStmt@14..23
                    LetKw@14..17 "let"
                    Whitespace@17..18 " "
                    Ident@18..19
                      Identifier@18..19 "x"
                    Whitespace@19..20 " "
                    Eq@20..21 "="
                    Whitespace@21..22 " "
                    IntLiteral@22..23
                      IntLit@22..23 "1"
                10..11 expected expression, found `)`"#]],
        );
    }

    #[test]
    fn a_missing_struct_name_keeps_the_body() {
        test_support::check_recovery(
            "struct { x: int64 }",
            &expect![[r#"
                Root@0..19
                  StructStmt@0..19
                    StructKw@0..6 "struct"
                    Whitespace@6..7 " "
                    LeftCurly@7..8 "{"
                    Whitespace@8..9 " "
                    StructField@9..17
                      Ident@9..10
                        Identifier@9..10 "x"
                      Colon@10..11 ":"
                      Whitespace@11..12 " "
                      NamedTypeAnnotation@12..17
                        Ident@12..17
                          Identifier@12..17 "int64"
                    Whitespace@17..18 " "
                    RightCurly@18..19 "}"
                7..8 expected identifier, found `{`"#]],
        );
    }

    #[test]
    fn a_missing_paren_keeps_the_body() {
        test_support::check_recovery(
            "def f(x: int64 { return x }",
            &expect![[r#"
                Root@0..27
                  FuncStmt@0..27
                    DefKw@0..3 "def"
                    Whitespace@3..4 " "
                    Ident@4..5
                      Identifier@4..5 "f"
                    LeftParen@5..6 "("
                    FuncParam@6..14
                      Ident@6..7
                        Identifier@6..7 "x"
                      Colon@7..8 ":"
                      Whitespace@8..9 " "
                      NamedTypeAnnotation@9..14
                        Ident@9..14
                          Identifier@9..14 "int64"
                    Whitespace@14..15 " "
                    BlockStmt@15..27
                      LeftCurly@15..16 "{"
                      Whitespace@16..17 " "
                      ReturnStmt@17..25
                        ReturnKw@17..23 "return"
                        Whitespace@23..24 " "
                        IdentExpr@24..25
                          Ident@24..25
                            Identifier@24..25 "x"
                      Whitespace@25..26 " "
                      RightCurly@26..27 "}"
                15..16 expected one of `)`, `[`, `,`, found `{`"#]],
        );
    }

    #[test]
    fn a_keyword_before_a_name_follower_is_the_name() {
        test_support::check_recovery(
            "let def = 1",
            &expect![[r#"
                Root@0..11
                  LetStmt@0..11
                    LetKw@0..3 "let"
                    Whitespace@3..4 " "
                    Error@4..7
                      DefKw@4..7 "def"
                    Whitespace@7..8 " "
                    Eq@8..9 "="
                    Whitespace@9..10 " "
                    IntLiteral@10..11
                      IntLit@10..11 "1"
                4..7 expected one of `mut`, identifier, found `def`"#]],
        );
    }

    #[test]
    fn a_keyword_that_starts_a_statement_is_not_the_name() {
        test_support::check_recovery(
            "let\ndef f() { return 1 }",
            &expect![[r#"
                Root@0..24
                  LetStmt@0..3
                    LetKw@0..3 "let"
                  Newline@3..4 "\n"
                  FuncStmt@4..24
                    DefKw@4..7 "def"
                    Whitespace@7..8 " "
                    Ident@8..9
                      Identifier@8..9 "f"
                    LeftParen@9..10 "("
                    RightParen@10..11 ")"
                    Whitespace@11..12 " "
                    BlockStmt@12..24
                      LeftCurly@12..13 "{"
                      Whitespace@13..14 " "
                      ReturnStmt@14..22
                        ReturnKw@14..20 "return"
                        Whitespace@20..21 " "
                        IntLiteral@21..22
                          IntLit@21..22 "1"
                      Whitespace@22..23 " "
                      RightCurly@23..24 "}"
                4..7 expected one of `mut`, identifier, found `def`"#]],
        );
    }
}
