use yuzu_lexer::token_kind::TokenKind;
use yuzu_syntax::SyntaxKind;

use crate::grammar::{Trailing, delimited, parse_ident, parse_name};
use crate::parser::{Parser, marker::CompletedMarker};
use crate::token_set::TokenSet;

/// What a missing expression also leaves in place: the close of the list or
/// block around it, and the next pipeline stage.
const EXPR_RECOVERY_SET: TokenSet = TokenSet::new(&[
    TokenKind::RightParen,
    TokenKind::RightSquare,
    TokenKind::RightCurly,
    TokenKind::Comma,
    TokenKind::Pipe,
]);

#[derive(Clone, Copy)]
enum BinOp {
    And,
    Or,
    In,
    NotIn,
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
    Eq,
    Neq,
    Lt,
    Lte,
    Gt,
    Gte,
    ShiftLeft,
    ShiftRight,
}

impl BinOp {
    fn binding_power(self) -> (u8, u8) {
        match self {
            BinOp::Or => (1, 2),
            BinOp::And => (3, 4),
            BinOp::Eq
            | BinOp::Neq
            | BinOp::In
            | BinOp::NotIn
            | BinOp::Lt
            | BinOp::Lte
            | BinOp::Gt
            | BinOp::Gte => (5, 6),
            BinOp::ShiftLeft | BinOp::ShiftRight => (7, 8),
            BinOp::Add | BinOp::Sub => (9, 10),
            BinOp::Mul | BinOp::Div | BinOp::Rem => (11, 12),
            BinOp::Pow => (15, 14),
        }
    }
}

#[derive(Clone, Copy)]
enum UnaryOp {
    Neg,
    Pos,
    Not,
}

impl UnaryOp {
    fn binding_power(self) -> ((), u8) {
        match self {
            UnaryOp::Neg | UnaryOp::Pos => ((), 13),
            UnaryOp::Not => ((), 5),
        }
    }
}

pub(crate) fn parse_expr(p: &mut Parser) -> Option<CompletedMarker> {
    parse_expr_binding_power(p, 0)
}

fn parse_expr_binding_power(p: &mut Parser, minimum_binding_power: u8) -> Option<CompletedMarker> {
    let mut lhs = parse_lhs(p)?;

    // A line break ends the expression before a token that can start one:
    // `(b)` or `-b` on a new line is a new statement. A token no statement
    // starts with, as `.`, `|>` or `and`, carries it on.
    loop {
        if p.at(TokenKind::LeftParen) && !p.at_line_start() {
            let marker = p.precede(lhs);
            parse_arg_list(p);
            lhs = p.complete(marker, SyntaxKind::CallExpr);
            continue;
        }

        if p.at(TokenKind::Dot) {
            let marker = p.precede(lhs);
            p.bump();
            parse_name(p);
            lhs = p.complete(marker, SyntaxKind::FieldAccessExpr);
            continue;
        }

        let Some(op) = parse_bin_op(p) else {
            break;
        };
        // A sign is a prefix unless a space follows it, as in `- b`.
        if matches!(op, BinOp::Add | BinOp::Sub) && p.at_line_start() && !p.is_spaced_after() {
            break;
        }

        let (left_binding_power, right_binding_power) = op.binding_power();
        if left_binding_power < minimum_binding_power {
            break;
        }

        p.bump();
        if matches!(op, BinOp::NotIn) {
            p.bump();
        }

        let marker = p.precede(lhs);
        let rhs = parse_expr_binding_power(p, right_binding_power);
        lhs = p.complete(marker, SyntaxKind::BinaryExpr);

        if rhs.is_none() {
            break;
        }
    }

    Some(lhs)
}

fn parse_lhs(p: &mut Parser) -> Option<CompletedMarker> {
    let Some(kind) = p.peek_kind() else {
        p.error_expression(EXPR_RECOVERY_SET);
        return None;
    };

    let completed = match kind {
        TokenKind::BoolLit
        | TokenKind::IntLit
        | TokenKind::FloatLit
        | TokenKind::HexLit
        | TokenKind::BinaryLit
        | TokenKind::StringLit
        | TokenKind::RawStringLit => parse_literal_expr(p),

        TokenKind::Identifier => {
            if p.peek_nth_kind(1) == Some(TokenKind::LeftCurly) {
                parse_struct_expr(p)
            } else {
                parse_ident_expr(p)
            }
        }

        TokenKind::LeftParen => parse_paren_expr(p),
        TokenKind::LeftSquare => parse_list_expr(p),
        TokenKind::Plus | TokenKind::Minus | TokenKind::NotKw => parse_unary_expr(p),

        _ => {
            p.error_expression(EXPR_RECOVERY_SET);
            return None;
        }
    };

    Some(completed)
}

fn parse_bin_op(p: &mut Parser) -> Option<BinOp> {
    let op = match p.peek_kind()? {
        TokenKind::Plus => BinOp::Add,
        TokenKind::Minus => BinOp::Sub,
        TokenKind::Star => BinOp::Mul,
        TokenKind::StarStar => BinOp::Pow,
        TokenKind::Slash => BinOp::Div,
        TokenKind::Percent => BinOp::Rem,
        TokenKind::EqEq => BinOp::Eq,
        TokenKind::Neq => BinOp::Neq,
        TokenKind::AndKw => BinOp::And,
        TokenKind::OrKw => BinOp::Or,
        TokenKind::Lt => BinOp::Lt,
        TokenKind::Lte => BinOp::Lte,
        TokenKind::Gt => BinOp::Gt,
        TokenKind::Gte => BinOp::Gte,
        TokenKind::Shl => BinOp::ShiftLeft,
        TokenKind::Shr => BinOp::ShiftRight,
        TokenKind::InKw => BinOp::In,
        TokenKind::NotKw if p.peek_nth_kind(1) == Some(TokenKind::InKw) => BinOp::NotIn,
        _ => return None,
    };
    Some(op)
}

fn parse_unary_op(p: &mut Parser) -> Option<UnaryOp> {
    let op = match p.peek_kind()? {
        TokenKind::Plus => UnaryOp::Pos,
        TokenKind::Minus => UnaryOp::Neg,
        TokenKind::NotKw => UnaryOp::Not,
        _ => return None,
    };
    Some(op)
}

fn parse_literal_expr(p: &mut Parser) -> CompletedMarker {
    let ast_kind = match p.peek_kind() {
        Some(TokenKind::IntLit | TokenKind::HexLit | TokenKind::BinaryLit) => {
            SyntaxKind::IntLiteral
        }
        Some(TokenKind::FloatLit) => SyntaxKind::FloatLiteral,
        Some(TokenKind::StringLit | TokenKind::RawStringLit) => SyntaxKind::StringLiteral,
        Some(TokenKind::BoolLit) => SyntaxKind::BoolLiteral,
        _ => unreachable!("parse_literal_expr called without a literal token"),
    };

    p.report_unknown_escapes();
    let m = p.start();
    p.bump();
    p.complete(m, ast_kind)
}

fn parse_ident_expr(p: &mut Parser) -> CompletedMarker {
    let m = p.start();
    parse_ident(p);
    p.complete(m, SyntaxKind::IdentExpr)
}

fn parse_struct_field_init(p: &mut Parser) -> CompletedMarker {
    let m = p.start();
    parse_ident(p);
    p.expect(TokenKind::Colon);
    parse_expr(p);
    p.complete(m, SyntaxKind::StructFieldInit)
}

fn parse_struct_expr(p: &mut Parser) -> CompletedMarker {
    let m = p.start();
    parse_ident(p);
    p.expect(TokenKind::LeftCurly);
    delimited(p, TokenKind::RightCurly, Trailing::Allowed, |p| {
        parse_struct_field_init(p);
    });
    p.expect(TokenKind::RightCurly);
    p.complete(m, SyntaxKind::StructExpr)
}

fn parse_list_expr(p: &mut Parser) -> CompletedMarker {
    let m = p.start();
    p.expect(TokenKind::LeftSquare);
    delimited(p, TokenKind::RightSquare, Trailing::Allowed, |p| {
        parse_expr(p);
    });
    p.expect(TokenKind::RightSquare);
    p.complete(m, SyntaxKind::ListExpr)
}

fn parse_paren_expr(p: &mut Parser) -> CompletedMarker {
    let m = p.start();
    p.expect(TokenKind::LeftParen);
    parse_expr_binding_power(p, 0);
    p.expect(TokenKind::RightParen);
    p.complete(m, SyntaxKind::ParenExpr)
}

fn parse_unary_expr(p: &mut Parser) -> CompletedMarker {
    let op = parse_unary_op(p).expect("parse_unary_expr called without a unary operator");
    let ((), right_binding_power) = op.binding_power();

    let m = p.start();
    p.bump();
    parse_expr_binding_power(p, right_binding_power);
    p.complete(m, SyntaxKind::UnaryExpr)
}

fn parse_arg_list(p: &mut Parser) -> CompletedMarker {
    let m = p.start();
    p.expect(TokenKind::LeftParen);
    delimited(p, TokenKind::RightParen, Trailing::Forbidden, |p| {
        parse_expr(p);
    });
    p.expect(TokenKind::RightParen);
    p.complete(m, SyntaxKind::ArgList)
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};

    use super::*;
    use crate::grammar::test_support;

    fn check(input: &str, expected: &Expect) {
        test_support::check(input, parse_expr, expected);
    }

    #[test]
    fn parse_lhs_directly() {
        test_support::check(
            "foo",
            parse_lhs,
            &expect![[r#"
            IdentExpr@0..3
              Ident@0..3
                Identifier@0..3 "foo"
        "#]],
        );
    }

    #[test]
    fn parse_literal_expr_directly() {
        test_support::check(
            "3.14",
            parse_literal_expr,
            &expect![[r#"
            FloatLiteral@0..4
              FloatLit@0..4 "3.14"
        "#]],
        );
    }

    #[test]
    fn parse_struct_field_init_directly() {
        test_support::check(
            "x: 1",
            parse_struct_field_init,
            &expect![[r#"
                StructFieldInit@0..4
                  Ident@0..1
                    Identifier@0..1 "x"
                  Colon@1..2 ":"
                  Whitespace@2..3 " "
                  IntLiteral@3..4
                    IntLit@3..4 "1"
            "#]],
        );
    }

    #[test]
    fn parse_struct_expr_directly() {
        test_support::check(
            "Point { x: 1, y: 2 }",
            parse_struct_expr,
            &expect![[r#"
                StructExpr@0..20
                  Ident@0..5
                    Identifier@0..5 "Point"
                  Whitespace@5..6 " "
                  LeftCurly@6..7 "{"
                  Whitespace@7..8 " "
                  StructFieldInit@8..12
                    Ident@8..9
                      Identifier@8..9 "x"
                    Colon@9..10 ":"
                    Whitespace@10..11 " "
                    IntLiteral@11..12
                      IntLit@11..12 "1"
                  Comma@12..13 ","
                  Whitespace@13..14 " "
                  StructFieldInit@14..18
                    Ident@14..15
                      Identifier@14..15 "y"
                    Colon@15..16 ":"
                    Whitespace@16..17 " "
                    IntLiteral@17..18
                      IntLit@17..18 "2"
                  Whitespace@18..19 " "
                  RightCurly@19..20 "}"
            "#]],
        );
    }

    #[test]
    fn parse_list_expr_directly() {
        test_support::check(
            "[1, 2, 3]",
            parse_list_expr,
            &expect![[r#"
                ListExpr@0..9
                  LeftSquare@0..1 "["
                  IntLiteral@1..2
                    IntLit@1..2 "1"
                  Comma@2..3 ","
                  Whitespace@3..4 " "
                  IntLiteral@4..5
                    IntLit@4..5 "2"
                  Comma@5..6 ","
                  Whitespace@6..7 " "
                  IntLiteral@7..8
                    IntLit@7..8 "3"
                  RightSquare@8..9 "]"
            "#]],
        );
    }

    #[test]
    fn parse_paren_expr_directly() {
        test_support::check(
            "(1 + 2)",
            parse_paren_expr,
            &expect![[r#"
                ParenExpr@0..7
                  LeftParen@0..1 "("
                  BinaryExpr@1..6
                    IntLiteral@1..2
                      IntLit@1..2 "1"
                    Whitespace@2..3 " "
                    Plus@3..4 "+"
                    Whitespace@4..5 " "
                    IntLiteral@5..6
                      IntLit@5..6 "2"
                  RightParen@6..7 ")"
            "#]],
        );
    }

    #[test]
    fn parse_unary_expr_directly() {
        test_support::check(
            "-5",
            parse_unary_expr,
            &expect![[r#"
            UnaryExpr@0..2
              Minus@0..1 "-"
              IntLiteral@1..2
                IntLit@1..2 "5"
        "#]],
        );
    }

    #[test]
    fn parse_arg_list_directly() {
        test_support::check(
            "(1, 2)",
            parse_arg_list,
            &expect![[r#"
                ArgList@0..6
                  LeftParen@0..1 "("
                  IntLiteral@1..2
                    IntLit@1..2 "1"
                  Comma@2..3 ","
                  Whitespace@3..4 " "
                  IntLiteral@4..5
                    IntLit@4..5 "2"
                  RightParen@5..6 ")"
            "#]],
        );
    }

    #[test]
    fn integer_literal() {
        check(
            "42",
            &expect![[r#"
            IntLiteral@0..2
              IntLit@0..2 "42"
        "#]],
        );
    }

    #[test]
    fn addition() {
        check(
            "1 + 2",
            &expect![[r#"
                BinaryExpr@0..5
                  IntLiteral@0..1
                    IntLit@0..1 "1"
                  Whitespace@1..2 " "
                  Plus@2..3 "+"
                  Whitespace@3..4 " "
                  IntLiteral@4..5
                    IntLit@4..5 "2"
            "#]],
        );
    }

    #[test]
    fn multiplication_binds_tighter_than_addition() {
        check(
            "1 + 2 * 3",
            &expect![[r#"
                BinaryExpr@0..9
                  IntLiteral@0..1
                    IntLit@0..1 "1"
                  Whitespace@1..2 " "
                  Plus@2..3 "+"
                  Whitespace@3..4 " "
                  BinaryExpr@4..9
                    IntLiteral@4..5
                      IntLit@4..5 "2"
                    Whitespace@5..6 " "
                    Star@6..7 "*"
                    Whitespace@7..8 " "
                    IntLiteral@8..9
                      IntLit@8..9 "3"
            "#]],
        );
    }

    #[test]
    fn remainder_binds_like_multiplication() {
        check(
            "1 + 2 % 3 * 4",
            &expect![[r#"
                BinaryExpr@0..13
                  IntLiteral@0..1
                    IntLit@0..1 "1"
                  Whitespace@1..2 " "
                  Plus@2..3 "+"
                  Whitespace@3..4 " "
                  BinaryExpr@4..13
                    BinaryExpr@4..9
                      IntLiteral@4..5
                        IntLit@4..5 "2"
                      Whitespace@5..6 " "
                      Percent@6..7 "%"
                      Whitespace@7..8 " "
                      IntLiteral@8..9
                        IntLit@8..9 "3"
                    Whitespace@9..10 " "
                    Star@10..11 "*"
                    Whitespace@11..12 " "
                    IntLiteral@12..13
                      IntLit@12..13 "4"
            "#]],
        );
    }

    #[test]
    fn power_is_right_associative() {
        check(
            "2 ** 3 ** 4",
            &expect![[r#"
                BinaryExpr@0..11
                  IntLiteral@0..1
                    IntLit@0..1 "2"
                  Whitespace@1..2 " "
                  StarStar@2..4 "**"
                  Whitespace@4..5 " "
                  BinaryExpr@5..11
                    IntLiteral@5..6
                      IntLit@5..6 "3"
                    Whitespace@6..7 " "
                    StarStar@7..9 "**"
                    Whitespace@9..10 " "
                    IntLiteral@10..11
                      IntLit@10..11 "4"
            "#]],
        );
    }

    #[test]
    fn unary_minus_is_looser_than_power() {
        check(
            "-2 ** 2",
            &expect![[r#"
                UnaryExpr@0..7
                  Minus@0..1 "-"
                  BinaryExpr@1..7
                    IntLiteral@1..2
                      IntLit@1..2 "2"
                    Whitespace@2..3 " "
                    StarStar@3..5 "**"
                    Whitespace@5..6 " "
                    IntLiteral@6..7
                      IntLit@6..7 "2"
            "#]],
        );
    }

    #[test]
    fn parentheses_group() {
        check(
            "(1 + 2) * 3",
            &expect![[r#"
                BinaryExpr@0..11
                  ParenExpr@0..7
                    LeftParen@0..1 "("
                    BinaryExpr@1..6
                      IntLiteral@1..2
                        IntLit@1..2 "1"
                      Whitespace@2..3 " "
                      Plus@3..4 "+"
                      Whitespace@4..5 " "
                      IntLiteral@5..6
                        IntLit@5..6 "2"
                    RightParen@6..7 ")"
                  Whitespace@7..8 " "
                  Star@8..9 "*"
                  Whitespace@9..10 " "
                  IntLiteral@10..11
                    IntLit@10..11 "3"
            "#]],
        );
    }

    #[test]
    fn call_of_field_access() {
        check(
            "a.b(c)",
            &expect![[r#"
            CallExpr@0..6
              FieldAccessExpr@0..3
                IdentExpr@0..1
                  Ident@0..1
                    Identifier@0..1 "a"
                Dot@1..2 "."
                Ident@2..3
                  Identifier@2..3 "b"
              ArgList@3..6
                LeftParen@3..4 "("
                IdentExpr@4..5
                  Ident@4..5
                    Identifier@4..5 "c"
                RightParen@5..6 ")"
        "#]],
        );
    }

    #[test]
    fn struct_literal() {
        check(
            "P { x: 1 }",
            &expect![[r#"
                StructExpr@0..10
                  Ident@0..1
                    Identifier@0..1 "P"
                  Whitespace@1..2 " "
                  LeftCurly@2..3 "{"
                  Whitespace@3..4 " "
                  StructFieldInit@4..8
                    Ident@4..5
                      Identifier@4..5 "x"
                    Colon@5..6 ":"
                    Whitespace@6..7 " "
                    IntLiteral@7..8
                      IntLit@7..8 "1"
                  Whitespace@8..9 " "
                  RightCurly@9..10 "}"
            "#]],
        );
    }

    #[test]
    fn not_in_operator() {
        check(
            "a not in b",
            &expect![[r#"
                BinaryExpr@0..10
                  IdentExpr@0..1
                    Ident@0..1
                      Identifier@0..1 "a"
                  Whitespace@1..2 " "
                  NotKw@2..5 "not"
                  Whitespace@5..6 " "
                  InKw@6..8 "in"
                  Whitespace@8..9 " "
                  IdentExpr@9..10
                    Ident@9..10
                      Identifier@9..10 "b"
            "#]],
        );
    }
}
