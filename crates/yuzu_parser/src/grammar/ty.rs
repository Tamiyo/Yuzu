use yuzu_lexer::token_kind::TokenKind;
use yuzu_syntax::SyntaxKind;

use crate::grammar::{Trailing, delimited, parse_ident};
use crate::parser::{Parser, marker::CompletedMarker};

pub(crate) fn parse_type(p: &mut Parser) -> CompletedMarker {
    if p.peek_kind() == Some(TokenKind::LeftParen) {
        parse_func_type(p)
    } else {
        parse_named_type(p)
    }
}

fn parse_named_type(p: &mut Parser) -> CompletedMarker {
    let m = p.start();
    parse_ident(p);

    if p.at(TokenKind::LeftSquare) {
        p.bump();
        // Peeked, not asked: `]` is no type, so it joins no expected set.
        if p.peek_kind() == Some(TokenKind::RightSquare) {
            parse_type(p);
        }
        delimited(p, TokenKind::RightSquare, Trailing::Forbidden, |p| {
            parse_type(p);
        });
        p.expect(TokenKind::RightSquare);
    }

    p.complete(m, SyntaxKind::NamedTypeAnnotation)
}

fn parse_func_type(p: &mut Parser) -> CompletedMarker {
    let m = p.start();

    let params = p.start();
    p.expect(TokenKind::LeftParen);
    delimited(p, TokenKind::RightParen, Trailing::Forbidden, |p| {
        parse_type(p);
    });
    p.expect(TokenKind::RightParen);
    p.complete(params, SyntaxKind::FuncTypeAnnotationParams);

    p.expect(TokenKind::Arrow);
    parse_type(p);
    p.complete(m, SyntaxKind::FuncTypeAnnotation)
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};

    use super::*;
    use crate::grammar::test_support;

    fn check(input: &str, expected: &Expect) {
        test_support::check(input, parse_type, expected);
    }

    #[test]
    fn parse_named_type_directly() {
        test_support::check(
            "Relation[Employee]",
            parse_named_type,
            &expect![[r#"
            NamedTypeAnnotation@0..18
              Ident@0..8
                Identifier@0..8 "Relation"
              LeftSquare@8..9 "["
              NamedTypeAnnotation@9..17
                Ident@9..17
                  Identifier@9..17 "Employee"
              RightSquare@17..18 "]"
        "#]],
        );
    }

    #[test]
    fn parse_func_type_directly() {
        test_support::check(
            "(int, str) -> bool",
            parse_func_type,
            &expect![[r#"
                FuncTypeAnnotation@0..18
                  FuncTypeAnnotationParams@0..10
                    LeftParen@0..1 "("
                    NamedTypeAnnotation@1..4
                      Ident@1..4
                        Identifier@1..4 "int"
                    Comma@4..5 ","
                    Whitespace@5..6 " "
                    NamedTypeAnnotation@6..9
                      Ident@6..9
                        Identifier@6..9 "str"
                    RightParen@9..10 ")"
                  Whitespace@10..11 " "
                  Arrow@11..13 "->"
                  Whitespace@13..14 " "
                  NamedTypeAnnotation@14..18
                    Ident@14..18
                      Identifier@14..18 "bool"
            "#]],
        );
    }

    #[test]
    fn named_type() {
        check(
            "int",
            &expect![[r#"
            NamedTypeAnnotation@0..3
              Ident@0..3
                Identifier@0..3 "int"
        "#]],
        );
    }

    #[test]
    fn generic_type() {
        check(
            "Relation[Employee]",
            &expect![[r#"
            NamedTypeAnnotation@0..18
              Ident@0..8
                Identifier@0..8 "Relation"
              LeftSquare@8..9 "["
              NamedTypeAnnotation@9..17
                Ident@9..17
                  Identifier@9..17 "Employee"
              RightSquare@17..18 "]"
        "#]],
        );
    }

    #[test]
    fn generic_type_with_multiple_args() {
        check(
            "Map[str, int]",
            &expect![[r#"
                NamedTypeAnnotation@0..13
                  Ident@0..3
                    Identifier@0..3 "Map"
                  LeftSquare@3..4 "["
                  NamedTypeAnnotation@4..7
                    Ident@4..7
                      Identifier@4..7 "str"
                  Comma@7..8 ","
                  Whitespace@8..9 " "
                  NamedTypeAnnotation@9..12
                    Ident@9..12
                      Identifier@9..12 "int"
                  RightSquare@12..13 "]"
            "#]],
        );
    }

    #[test]
    fn nested_generic_type() {
        check(
            "Aggregate[decimal, List[int]]",
            &expect![[r#"
                NamedTypeAnnotation@0..29
                  Ident@0..9
                    Identifier@0..9 "Aggregate"
                  LeftSquare@9..10 "["
                  NamedTypeAnnotation@10..17
                    Ident@10..17
                      Identifier@10..17 "decimal"
                  Comma@17..18 ","
                  Whitespace@18..19 " "
                  NamedTypeAnnotation@19..28
                    Ident@19..23
                      Identifier@19..23 "List"
                    LeftSquare@23..24 "["
                    NamedTypeAnnotation@24..27
                      Ident@24..27
                        Identifier@24..27 "int"
                    RightSquare@27..28 "]"
                  RightSquare@28..29 "]"
            "#]],
        );
    }

    #[test]
    fn function_type() {
        check(
            "(int, str) -> bool",
            &expect![[r#"
                FuncTypeAnnotation@0..18
                  FuncTypeAnnotationParams@0..10
                    LeftParen@0..1 "("
                    NamedTypeAnnotation@1..4
                      Ident@1..4
                        Identifier@1..4 "int"
                    Comma@4..5 ","
                    Whitespace@5..6 " "
                    NamedTypeAnnotation@6..9
                      Ident@6..9
                        Identifier@6..9 "str"
                    RightParen@9..10 ")"
                  Whitespace@10..11 " "
                  Arrow@11..13 "->"
                  Whitespace@13..14 " "
                  NamedTypeAnnotation@14..18
                    Ident@14..18
                      Identifier@14..18 "bool"
            "#]],
        );
    }

    #[test]
    fn function_type_without_params() {
        check(
            "() -> int",
            &expect![[r#"
                FuncTypeAnnotation@0..9
                  FuncTypeAnnotationParams@0..2
                    LeftParen@0..1 "("
                    RightParen@1..2 ")"
                  Whitespace@2..3 " "
                  Arrow@3..5 "->"
                  Whitespace@5..6 " "
                  NamedTypeAnnotation@6..9
                    Ident@6..9
                      Identifier@6..9 "int"
            "#]],
        );
    }
}
