use yuzu_lexer::token_kind::TokenKind;
use yuzu_syntax::SyntaxKind;

use crate::grammar::expr::parse_expr;
use crate::grammar::parse_ident;
use crate::grammar::rel::parse_query;
use crate::grammar::ty::parse_type;
use crate::parser::{Parser, marker::CompletedMarker};

pub(crate) fn parse_stmt(p: &mut Parser) -> CompletedMarker {
    // `pub` prefixes a declaration, so what follows it decides which one this
    // is. Each declaration bumps the `pub` itself, the way a function bumps
    // its own `external` and `agg`, so the keyword lands inside the node it
    // qualifies.
    if p.at(TokenKind::PubKw) {
        return parse_public_stmt(p);
    }

    // Reserved words, so no lookahead is needed to tell a declaration from an
    // expression that happens to start with the same identifier.
    if p.at(TokenKind::FnKw) || p.at(TokenKind::AggKw) || p.at(TokenKind::ExternalKw) {
        return parse_func_stmt(p);
    }
    if p.at(TokenKind::ImplKw) {
        return parse_impl_stmt(p);
    }
    if p.at(TokenKind::TraitKw) {
        return parse_trait_stmt(p);
    }
    if p.at(TokenKind::LetKw) {
        return parse_let_stmt(p);
    }
    if p.at(TokenKind::ReturnKw) {
        return parse_return_stmt(p);
    }
    if p.at(TokenKind::StructKw) {
        return parse_struct_stmt(p);
    }
    if p.at(TokenKind::TableKw) {
        return parse_table_stmt(p);
    }
    parse_expr_stmt(p)
}

/// A declaration behind `pub`. Only the kinds that carry a name can be
/// exported, so anything else here is reported against the `pub` rather than
/// parsed as a declaration it is not.
fn parse_public_stmt(p: &mut Parser) -> CompletedMarker {
    match p.peek_nth_kind(1) {
        Some(TokenKind::FnKw | TokenKind::AggKw | TokenKind::ExternalKw) => parse_func_stmt(p),
        Some(TokenKind::TraitKw) => parse_trait_stmt(p),
        Some(TokenKind::LetKw) => parse_let_stmt(p),
        Some(TokenKind::StructKw) => parse_struct_stmt(p),
        Some(TokenKind::TableKw) => parse_table_stmt(p),
        _ => {
            let m = p.start();
            p.bump();
            p.error_expression(&[]);
            p.complete(m, SyntaxKind::Error)
        }
    }
}

/// Bumps a leading `pub`, so every declaration parser starts the same way.
fn parse_visibility(p: &mut Parser) {
    if p.at(TokenKind::PubKw) {
        p.bump();
    }
}

fn parse_block_stmt(p: &mut Parser) -> CompletedMarker {
    let m = p.start();

    p.expect(TokenKind::LeftCurly);

    while !p.at(TokenKind::RightCurly) && !p.at_end() {
        parse_stmt(p);
    }

    p.expect(TokenKind::RightCurly);

    p.complete(m, SyntaxKind::BlockStmt)
}

fn parse_func_stmt(p: &mut Parser) -> CompletedMarker {
    let m = p.start();
    parse_visibility(p);

    if p.at(TokenKind::ExternalKw) {
        p.bump();
    }
    if p.at(TokenKind::AggKw) {
        p.bump();
    }
    p.expect(TokenKind::FnKw);
    parse_ident(p);

    if p.at(TokenKind::LeftSquare) {
        p.bump();
        if !p.at(TokenKind::RightSquare) {
            parse_type_param(p);
            while p.at(TokenKind::Comma) {
                p.bump();
                parse_type_param(p);
            }
        }
        p.expect(TokenKind::RightSquare);
    }

    p.expect(TokenKind::LeftParen);
    if !p.at(TokenKind::RightParen) {
        parse_param(p);
        while p.at(TokenKind::Comma) {
            p.bump();
            parse_param(p);
        }
    }
    p.expect(TokenKind::RightParen);

    if p.at(TokenKind::Arrow) {
        p.bump();
        parse_type(p);
    }

    if p.at(TokenKind::WhereKw) {
        p.bump();
        parse_type_bound(p);
        while p.at(TokenKind::Comma) {
            p.bump();
            parse_type_bound(p);
        }
    }

    if p.at(TokenKind::LeftCurly) {
        parse_block_stmt(p);
    }

    p.complete(m, SyntaxKind::FuncStmt)
}

fn parse_impl_stmt(p: &mut Parser) -> CompletedMarker {
    let m = p.start();
    p.expect(TokenKind::ImplKw);

    if p.peek_nth_kind(1) == Some(TokenKind::ForKw) {
        parse_trait_ref(p);
        p.expect(TokenKind::ForKw);
    }

    parse_ident(p);
    p.expect(TokenKind::LeftCurly);
    while p.at(TokenKind::FnKw) {
        parse_func_stmt(p);
    }
    p.expect(TokenKind::RightCurly);

    p.complete(m, SyntaxKind::ImplStmt)
}

fn parse_trait_stmt(p: &mut Parser) -> CompletedMarker {
    let m = p.start();
    parse_visibility(p);
    p.expect(TokenKind::TraitKw);
    parse_ident(p);
    p.expect(TokenKind::LeftCurly);
    while p.at(TokenKind::FnKw) {
        parse_trait_method(p);
    }
    p.expect(TokenKind::RightCurly);

    p.complete(m, SyntaxKind::TraitStmt)
}

fn parse_trait_method(p: &mut Parser) -> CompletedMarker {
    let m = p.start();
    p.expect(TokenKind::FnKw);
    parse_ident(p);
    p.expect(TokenKind::LeftParen);
    if !p.at(TokenKind::RightParen) {
        parse_param(p);
        while p.at(TokenKind::Comma) {
            p.bump();
            parse_param(p);
        }
    }
    p.expect(TokenKind::RightParen);
    if p.at(TokenKind::Arrow) {
        p.bump();
        parse_type(p);
    }

    p.complete(m, SyntaxKind::FuncStmt)
}

fn parse_let_stmt(p: &mut Parser) -> CompletedMarker {
    let m = p.start();
    parse_visibility(p);
    p.expect(TokenKind::LetKw);

    if p.at(TokenKind::MutKw) {
        p.bump();
    }

    parse_ident(p);

    if p.at(TokenKind::Colon) {
        p.bump();
        parse_type(p);
    }

    p.expect(TokenKind::Eq);
    parse_value(p);

    p.complete(m, SyntaxKind::LetStmt)
}

fn parse_return_stmt(p: &mut Parser) -> CompletedMarker {
    let m = p.start();
    p.expect(TokenKind::ReturnKw);

    if !p.at(TokenKind::RightCurly) {
        parse_expr(p);
    }

    p.complete(m, SyntaxKind::ReturnStmt)
}

fn parse_struct_stmt(p: &mut Parser) -> CompletedMarker {
    let m = p.start();
    parse_visibility(p);
    p.expect(TokenKind::StructKw);
    parse_ident(p);
    parse_struct_field_list(p);
    p.complete(m, SyntaxKind::StructStmt)
}

fn parse_table_stmt(p: &mut Parser) -> CompletedMarker {
    let m = p.start();
    parse_visibility(p);
    p.expect(TokenKind::TableKw);
    parse_ident(p);
    p.expect(TokenKind::Eq);
    if p.at(TokenKind::LeftCurly) {
        parse_struct_field_list(p);
    } else {
        parse_ident(p);
    }
    p.complete(m, SyntaxKind::TableStmt)
}

fn parse_expr_stmt(p: &mut Parser) -> CompletedMarker {
    let m = p.start();

    if p.at(TokenKind::FromKw) {
        parse_query(p);
        return p.complete(m, SyntaxKind::ExprStmt);
    }

    parse_expr(p);

    if p.at(TokenKind::Eq) {
        p.bump();
        parse_value(p);
        return p.complete(m, SyntaxKind::AssignStmt);
    }

    p.complete(m, SyntaxKind::ExprStmt)
}

fn parse_value(p: &mut Parser) -> Option<CompletedMarker> {
    if p.at(TokenKind::FromKw) {
        Some(parse_query(p))
    } else {
        parse_expr(p)
    }
}

fn parse_param(p: &mut Parser) -> CompletedMarker {
    let m = p.start();
    parse_ident(p);
    if p.at(TokenKind::Colon) {
        p.bump();
        parse_type(p);
    }
    p.complete(m, SyntaxKind::FuncParam)
}

fn parse_type_param(p: &mut Parser) -> CompletedMarker {
    let m = p.start();
    parse_ident(p);
    p.complete(m, SyntaxKind::TypeParam)
}

fn parse_type_bound(p: &mut Parser) -> CompletedMarker {
    let m = p.start();
    parse_ident(p);
    p.expect(TokenKind::Colon);
    parse_trait_ref(p);
    while p.at(TokenKind::Plus) {
        p.bump();
        parse_trait_ref(p);
    }
    p.complete(m, SyntaxKind::TypeBound)
}

fn parse_trait_ref(p: &mut Parser) -> CompletedMarker {
    let m = p.start();
    parse_ident(p);
    p.complete(m, SyntaxKind::TraitRef)
}

fn parse_struct_field_decl(p: &mut Parser) -> CompletedMarker {
    let m = p.start();
    parse_visibility(p);
    parse_ident(p);
    p.expect(TokenKind::Colon);
    parse_type(p);
    p.complete(m, SyntaxKind::StructField)
}

fn parse_struct_field_list(p: &mut Parser) {
    p.expect(TokenKind::LeftCurly);
    if !p.at(TokenKind::RightCurly) {
        parse_struct_field_decl(p);
        while p.at(TokenKind::Comma) {
            p.bump();
            if p.at(TokenKind::RightCurly) {
                break;
            }
            parse_struct_field_decl(p);
        }
    }
    p.expect(TokenKind::RightCurly);
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};

    use super::*;
    use crate::grammar::test_support;

    fn check(input: &str, expected: Expect) {
        test_support::check(input, parse_stmt, expected);
    }

    #[test]
    fn parse_external_fn_stmt() {
        test_support::check(
            "external fn upper(s: str) -> str",
            parse_stmt,
            expect![[r#"
                FuncStmt@0..32
                  ExternalKw@0..8 "external"
                  Space@8..9 " "
                  FnKw@9..11 "fn"
                  Space@11..12 " "
                  Ident@12..17
                    Identifier@12..17 "upper"
                  LeftParen@17..18 "("
                  FuncParam@18..24
                    Ident@18..19
                      Identifier@18..19 "s"
                    Colon@19..20 ":"
                    Space@20..21 " "
                    NamedTypeAnnotation@21..24
                      Ident@21..24
                        Identifier@21..24 "str"
                  RightParen@24..25 ")"
                  Space@25..26 " "
                  Arrow@26..28 "->"
                  Space@28..29 " "
                  NamedTypeAnnotation@29..32
                    Ident@29..32
                      Identifier@29..32 "str"
            "#]],
        );
    }

    #[test]
    fn parse_external_agg_fn_stmt() {
        test_support::check(
            "external agg fn median(x: int64) -> float64",
            parse_stmt,
            expect![[r#"
                FuncStmt@0..43
                  ExternalKw@0..8 "external"
                  Space@8..9 " "
                  AggKw@9..12 "agg"
                  Space@12..13 " "
                  FnKw@13..15 "fn"
                  Space@15..16 " "
                  Ident@16..22
                    Identifier@16..22 "median"
                  LeftParen@22..23 "("
                  FuncParam@23..31
                    Ident@23..24
                      Identifier@23..24 "x"
                    Colon@24..25 ":"
                    Space@25..26 " "
                    NamedTypeAnnotation@26..31
                      Ident@26..31
                        Identifier@26..31 "int64"
                  RightParen@31..32 ")"
                  Space@32..33 " "
                  Arrow@33..35 "->"
                  Space@35..36 " "
                  NamedTypeAnnotation@36..43
                    Ident@36..43
                      Identifier@36..43 "float64"
            "#]],
        );
    }

    #[test]
    fn parse_agg_func_stmt() {
        test_support::check(
            "agg fn spread(x: int64) -> int64 { return sum(x) }",
            parse_stmt,
            expect![[r#"
                FuncStmt@0..50
                  AggKw@0..3 "agg"
                  Space@3..4 " "
                  FnKw@4..6 "fn"
                  Space@6..7 " "
                  Ident@7..13
                    Identifier@7..13 "spread"
                  LeftParen@13..14 "("
                  FuncParam@14..22
                    Ident@14..15
                      Identifier@14..15 "x"
                    Colon@15..16 ":"
                    Space@16..17 " "
                    NamedTypeAnnotation@17..22
                      Ident@17..22
                        Identifier@17..22 "int64"
                  RightParen@22..23 ")"
                  Space@23..24 " "
                  Arrow@24..26 "->"
                  Space@26..27 " "
                  NamedTypeAnnotation@27..32
                    Ident@27..32
                      Identifier@27..32 "int64"
                  Space@32..33 " "
                  BlockStmt@33..50
                    LeftCurly@33..34 "{"
                    Space@34..35 " "
                    ReturnStmt@35..48
                      ReturnKw@35..41 "return"
                      Space@41..42 " "
                      CallExpr@42..48
                        IdentExpr@42..45
                          Ident@42..45
                            Identifier@42..45 "sum"
                        ArgList@45..48
                          LeftParen@45..46 "("
                          IdentExpr@46..47
                            Ident@46..47
                              Identifier@46..47 "x"
                          RightParen@47..48 ")"
                    Space@48..49 " "
                    RightCurly@49..50 "}"
            "#]],
        );
    }

    #[test]
    fn agg_is_reserved_and_cannot_name_a_binding() {
        test_support::check(
            "let agg = 1",
            parse_stmt,
            expect![[r#"
                LetStmt@0..11
                  LetKw@0..3 "let"
                  Space@3..4 " "
                  Ident@4..7
                    Error@4..7
                      AggKw@4..7 "agg"
                  Space@7..8 " "
                  Eq@8..9 "="
                  Space@9..10 " "
                  IntLiteral@10..11
                    IntLit@10..11 "1"
            "#]],
        );
    }

    #[test]
    fn parse_block_stmt_directly() {
        test_support::check(
            "{ return x }",
            parse_block_stmt,
            expect![[r#"
                BlockStmt@0..12
                  LeftCurly@0..1 "{"
                  Space@1..2 " "
                  ReturnStmt@2..10
                    ReturnKw@2..8 "return"
                    Space@8..9 " "
                    IdentExpr@9..10
                      Ident@9..10
                        Identifier@9..10 "x"
                  Space@10..11 " "
                  RightCurly@11..12 "}"
            "#]],
        );
    }

    #[test]
    fn parse_func_stmt_directly() {
        test_support::check(
            "fn f(x: int) -> int { return x }",
            parse_func_stmt,
            expect![[r#"
                FuncStmt@0..32
                  FnKw@0..2 "fn"
                  Space@2..3 " "
                  Ident@3..4
                    Identifier@3..4 "f"
                  LeftParen@4..5 "("
                  FuncParam@5..11
                    Ident@5..6
                      Identifier@5..6 "x"
                    Colon@6..7 ":"
                    Space@7..8 " "
                    NamedTypeAnnotation@8..11
                      Ident@8..11
                        Identifier@8..11 "int"
                  RightParen@11..12 ")"
                  Space@12..13 " "
                  Arrow@13..15 "->"
                  Space@15..16 " "
                  NamedTypeAnnotation@16..19
                    Ident@16..19
                      Identifier@16..19 "int"
                  Space@19..20 " "
                  BlockStmt@20..32
                    LeftCurly@20..21 "{"
                    Space@21..22 " "
                    ReturnStmt@22..30
                      ReturnKw@22..28 "return"
                      Space@28..29 " "
                      IdentExpr@29..30
                        Ident@29..30
                          Identifier@29..30 "x"
                    Space@30..31 " "
                    RightCurly@31..32 "}"
            "#]],
        );
    }

    #[test]
    fn parse_impl_stmt_directly() {
        test_support::check(
            "impl Point { fn x(self) { return self } }",
            parse_impl_stmt,
            expect![[r#"
                ImplStmt@0..41
                  ImplKw@0..4 "impl"
                  Space@4..5 " "
                  Ident@5..10
                    Identifier@5..10 "Point"
                  Space@10..11 " "
                  LeftCurly@11..12 "{"
                  Space@12..13 " "
                  FuncStmt@13..39
                    FnKw@13..15 "fn"
                    Space@15..16 " "
                    Ident@16..17
                      Identifier@16..17 "x"
                    LeftParen@17..18 "("
                    FuncParam@18..22
                      Ident@18..22
                        Identifier@18..22 "self"
                    RightParen@22..23 ")"
                    Space@23..24 " "
                    BlockStmt@24..39
                      LeftCurly@24..25 "{"
                      Space@25..26 " "
                      ReturnStmt@26..37
                        ReturnKw@26..32 "return"
                        Space@32..33 " "
                        IdentExpr@33..37
                          Ident@33..37
                            Identifier@33..37 "self"
                      Space@37..38 " "
                      RightCurly@38..39 "}"
                  Space@39..40 " "
                  RightCurly@40..41 "}"
            "#]],
        );
    }

    #[test]
    fn parse_trait_stmt_directly() {
        test_support::check(
            "trait Show { fn show(self) -> str }",
            parse_trait_stmt,
            expect![[r#"
                TraitStmt@0..35
                  TraitKw@0..5 "trait"
                  Space@5..6 " "
                  Ident@6..10
                    Identifier@6..10 "Show"
                  Space@10..11 " "
                  LeftCurly@11..12 "{"
                  Space@12..13 " "
                  FuncStmt@13..33
                    FnKw@13..15 "fn"
                    Space@15..16 " "
                    Ident@16..20
                      Identifier@16..20 "show"
                    LeftParen@20..21 "("
                    FuncParam@21..25
                      Ident@21..25
                        Identifier@21..25 "self"
                    RightParen@25..26 ")"
                    Space@26..27 " "
                    Arrow@27..29 "->"
                    Space@29..30 " "
                    NamedTypeAnnotation@30..33
                      Ident@30..33
                        Identifier@30..33 "str"
                  Space@33..34 " "
                  RightCurly@34..35 "}"
            "#]],
        );
    }

    #[test]
    fn parse_trait_method_directly() {
        test_support::check(
            "fn show(self) -> str",
            parse_trait_method,
            expect![[r#"
            FuncStmt@0..20
              FnKw@0..2 "fn"
              Space@2..3 " "
              Ident@3..7
                Identifier@3..7 "show"
              LeftParen@7..8 "("
              FuncParam@8..12
                Ident@8..12
                  Identifier@8..12 "self"
              RightParen@12..13 ")"
              Space@13..14 " "
              Arrow@14..16 "->"
              Space@16..17 " "
              NamedTypeAnnotation@17..20
                Ident@17..20
                  Identifier@17..20 "str"
        "#]],
        );
    }

    #[test]
    fn parse_let_stmt_directly() {
        test_support::check(
            "let mut x: int = 1",
            parse_let_stmt,
            expect![[r#"
                LetStmt@0..18
                  LetKw@0..3 "let"
                  Space@3..4 " "
                  MutKw@4..7 "mut"
                  Space@7..8 " "
                  Ident@8..9
                    Identifier@8..9 "x"
                  Colon@9..10 ":"
                  Space@10..11 " "
                  NamedTypeAnnotation@11..14
                    Ident@11..14
                      Identifier@11..14 "int"
                  Space@14..15 " "
                  Eq@15..16 "="
                  Space@16..17 " "
                  IntLiteral@17..18
                    IntLit@17..18 "1"
            "#]],
        );
    }

    #[test]
    fn parse_return_stmt_directly() {
        test_support::check(
            "return x",
            parse_return_stmt,
            expect![[r#"
            ReturnStmt@0..8
              ReturnKw@0..6 "return"
              Space@6..7 " "
              IdentExpr@7..8
                Ident@7..8
                  Identifier@7..8 "x"
        "#]],
        );
    }

    #[test]
    fn parse_struct_stmt_directly() {
        test_support::check(
            "struct P { x: int }",
            parse_struct_stmt,
            expect![[r#"
                StructStmt@0..19
                  StructKw@0..6 "struct"
                  Space@6..7 " "
                  Ident@7..8
                    Identifier@7..8 "P"
                  Space@8..9 " "
                  LeftCurly@9..10 "{"
                  Space@10..11 " "
                  StructField@11..17
                    Ident@11..12
                      Identifier@11..12 "x"
                    Colon@12..13 ":"
                    Space@13..14 " "
                    NamedTypeAnnotation@14..17
                      Ident@14..17
                        Identifier@14..17 "int"
                  Space@17..18 " "
                  RightCurly@18..19 "}"
            "#]],
        );
    }

    #[test]
    fn parse_table_stmt_directly() {
        test_support::check(
            "table T = Row",
            parse_table_stmt,
            expect![[r#"
                TableStmt@0..13
                  TableKw@0..5 "table"
                  Space@5..6 " "
                  Ident@6..7
                    Identifier@6..7 "T"
                  Space@7..8 " "
                  Eq@8..9 "="
                  Space@9..10 " "
                  Ident@10..13
                    Identifier@10..13 "Row"
            "#]],
        );
    }

    #[test]
    fn parse_expr_stmt_directly() {
        test_support::check(
            "f(x)",
            parse_expr_stmt,
            expect![[r#"
            ExprStmt@0..4
              CallExpr@0..4
                IdentExpr@0..1
                  Ident@0..1
                    Identifier@0..1 "f"
                ArgList@1..4
                  LeftParen@1..2 "("
                  IdentExpr@2..3
                    Ident@2..3
                      Identifier@2..3 "x"
                  RightParen@3..4 ")"
        "#]],
        );
    }

    #[test]
    fn parse_value_with_expr() {
        test_support::check(
            "1 + 2",
            parse_value,
            expect![[r#"
                BinaryExpr@0..5
                  IntLiteral@0..1
                    IntLit@0..1 "1"
                  Space@1..2 " "
                  Plus@2..3 "+"
                  Space@3..4 " "
                  IntLiteral@4..5
                    IntLit@4..5 "2"
            "#]],
        );
    }

    #[test]
    fn parse_value_with_query() {
        test_support::check(
            "from t |> select a",
            parse_value,
            expect![[r#"
                SelectExpr@0..18
                  FromExpr@0..6
                    FromKw@0..4 "from"
                    Space@4..5 " "
                    Ident@5..6
                      Identifier@5..6 "t"
                  Space@6..7 " "
                  Pipe@7..9 "|>"
                  Space@9..10 " "
                  SelectKw@10..16 "select"
                  Space@16..17 " "
                  SelectItem@17..18
                    IdentExpr@17..18
                      Ident@17..18
                        Identifier@17..18 "a"
            "#]],
        );
    }

    #[test]
    fn parse_param_directly() {
        test_support::check(
            "x: int",
            parse_param,
            expect![[r#"
            FuncParam@0..6
              Ident@0..1
                Identifier@0..1 "x"
              Colon@1..2 ":"
              Space@2..3 " "
              NamedTypeAnnotation@3..6
                Ident@3..6
                  Identifier@3..6 "int"
        "#]],
        );
    }

    #[test]
    fn parse_param_without_type() {
        test_support::check(
            "self",
            parse_param,
            expect![[r#"
            FuncParam@0..4
              Ident@0..4
                Identifier@0..4 "self"
        "#]],
        );
    }

    #[test]
    fn parse_type_param_directly() {
        test_support::check(
            "T",
            parse_type_param,
            expect![[r#"
            TypeParam@0..1
              Ident@0..1
                Identifier@0..1 "T"
        "#]],
        );
    }

    #[test]
    fn parse_type_bound_directly() {
        test_support::check(
            "T: Add + Eq",
            parse_type_bound,
            expect![[r#"
                TypeBound@0..11
                  Ident@0..1
                    Identifier@0..1 "T"
                  Colon@1..2 ":"
                  Space@2..3 " "
                  TraitRef@3..6
                    Ident@3..6
                      Identifier@3..6 "Add"
                  Space@6..7 " "
                  Plus@7..8 "+"
                  Space@8..9 " "
                  TraitRef@9..11
                    Ident@9..11
                      Identifier@9..11 "Eq"
            "#]],
        );
    }

    #[test]
    fn parse_trait_ref_directly() {
        test_support::check(
            "Comparable",
            parse_trait_ref,
            expect![[r#"
            TraitRef@0..10
              Ident@0..10
                Identifier@0..10 "Comparable"
        "#]],
        );
    }

    /// `pub` belongs to the declaration it qualifies, so it sits inside the
    /// node rather than beside it, and a field carries its own.
    #[test]
    fn parse_public_struct_with_a_public_field() {
        check(
            "pub struct P { pub x: int, y: int }",
            expect![[r#"
                StructStmt@0..35
                  PubKw@0..3 "pub"
                  Space@3..4 " "
                  StructKw@4..10 "struct"
                  Space@10..11 " "
                  Ident@11..12
                    Identifier@11..12 "P"
                  Space@12..13 " "
                  LeftCurly@13..14 "{"
                  Space@14..15 " "
                  StructField@15..25
                    PubKw@15..18 "pub"
                    Space@18..19 " "
                    Ident@19..20
                      Identifier@19..20 "x"
                    Colon@20..21 ":"
                    Space@21..22 " "
                    NamedTypeAnnotation@22..25
                      Ident@22..25
                        Identifier@22..25 "int"
                  Comma@25..26 ","
                  Space@26..27 " "
                  StructField@27..33
                    Ident@27..28
                      Identifier@27..28 "y"
                    Colon@28..29 ":"
                    Space@29..30 " "
                    NamedTypeAnnotation@30..33
                      Ident@30..33
                        Identifier@30..33 "int"
                  Space@33..34 " "
                  RightCurly@34..35 "}"
            "#]],
        );
    }

    #[test]
    fn parse_struct_field_decl_directly() {
        test_support::check(
            "x: int",
            parse_struct_field_decl,
            expect![[r#"
            StructField@0..6
              Ident@0..1
                Identifier@0..1 "x"
              Colon@1..2 ":"
              Space@2..3 " "
              NamedTypeAnnotation@3..6
                Ident@3..6
                  Identifier@3..6 "int"
        "#]],
        );
    }

    #[test]
    fn let_stmt() {
        check(
            "let x = 1",
            expect![[r#"
                LetStmt@0..9
                  LetKw@0..3 "let"
                  Space@3..4 " "
                  Ident@4..5
                    Identifier@4..5 "x"
                  Space@5..6 " "
                  Eq@6..7 "="
                  Space@7..8 " "
                  IntLiteral@8..9
                    IntLit@8..9 "1"
            "#]],
        );
    }

    #[test]
    fn let_mut_with_type_annotation() {
        check(
            "let mut x: int = 1",
            expect![[r#"
                LetStmt@0..18
                  LetKw@0..3 "let"
                  Space@3..4 " "
                  MutKw@4..7 "mut"
                  Space@7..8 " "
                  Ident@8..9
                    Identifier@8..9 "x"
                  Colon@9..10 ":"
                  Space@10..11 " "
                  NamedTypeAnnotation@11..14
                    Ident@11..14
                      Identifier@11..14 "int"
                  Space@14..15 " "
                  Eq@15..16 "="
                  Space@16..17 " "
                  IntLiteral@17..18
                    IntLit@17..18 "1"
            "#]],
        );
    }

    #[test]
    fn return_stmt() {
        check(
            "return x",
            expect![[r#"
            ReturnStmt@0..8
              ReturnKw@0..6 "return"
              Space@6..7 " "
              IdentExpr@7..8
                Ident@7..8
                  Identifier@7..8 "x"
        "#]],
        );
    }

    #[test]
    fn assign_stmt() {
        check(
            "x = 5",
            expect![[r#"
                AssignStmt@0..5
                  IdentExpr@0..1
                    Ident@0..1
                      Identifier@0..1 "x"
                  Space@1..2 " "
                  Eq@2..3 "="
                  Space@3..4 " "
                  IntLiteral@4..5
                    IntLit@4..5 "5"
            "#]],
        );
    }

    #[test]
    fn func_stmt() {
        check(
            "fn add(x: int, y: int) -> int { return x }",
            expect![[r#"
                FuncStmt@0..42
                  FnKw@0..2 "fn"
                  Space@2..3 " "
                  Ident@3..6
                    Identifier@3..6 "add"
                  LeftParen@6..7 "("
                  FuncParam@7..13
                    Ident@7..8
                      Identifier@7..8 "x"
                    Colon@8..9 ":"
                    Space@9..10 " "
                    NamedTypeAnnotation@10..13
                      Ident@10..13
                        Identifier@10..13 "int"
                  Comma@13..14 ","
                  Space@14..15 " "
                  FuncParam@15..21
                    Ident@15..16
                      Identifier@15..16 "y"
                    Colon@16..17 ":"
                    Space@17..18 " "
                    NamedTypeAnnotation@18..21
                      Ident@18..21
                        Identifier@18..21 "int"
                  RightParen@21..22 ")"
                  Space@22..23 " "
                  Arrow@23..25 "->"
                  Space@25..26 " "
                  NamedTypeAnnotation@26..29
                    Ident@26..29
                      Identifier@26..29 "int"
                  Space@29..30 " "
                  BlockStmt@30..42
                    LeftCurly@30..31 "{"
                    Space@31..32 " "
                    ReturnStmt@32..40
                      ReturnKw@32..38 "return"
                      Space@38..39 " "
                      IdentExpr@39..40
                        Ident@39..40
                          Identifier@39..40 "x"
                    Space@40..41 " "
                    RightCurly@41..42 "}"
            "#]],
        );
    }

    #[test]
    fn generic_func_with_where_clause() {
        check(
            "fn id[T](x: T) -> T where T: Eq { return x }",
            expect![[r#"
                FuncStmt@0..44
                  FnKw@0..2 "fn"
                  Space@2..3 " "
                  Ident@3..5
                    Identifier@3..5 "id"
                  LeftSquare@5..6 "["
                  TypeParam@6..7
                    Ident@6..7
                      Identifier@6..7 "T"
                  RightSquare@7..8 "]"
                  LeftParen@8..9 "("
                  FuncParam@9..13
                    Ident@9..10
                      Identifier@9..10 "x"
                    Colon@10..11 ":"
                    Space@11..12 " "
                    NamedTypeAnnotation@12..13
                      Ident@12..13
                        Identifier@12..13 "T"
                  RightParen@13..14 ")"
                  Space@14..15 " "
                  Arrow@15..17 "->"
                  Space@17..18 " "
                  NamedTypeAnnotation@18..19
                    Ident@18..19
                      Identifier@18..19 "T"
                  Space@19..20 " "
                  WhereKw@20..25 "where"
                  Space@25..26 " "
                  TypeBound@26..31
                    Ident@26..27
                      Identifier@26..27 "T"
                    Colon@27..28 ":"
                    Space@28..29 " "
                    TraitRef@29..31
                      Ident@29..31
                        Identifier@29..31 "Eq"
                  Space@31..32 " "
                  BlockStmt@32..44
                    LeftCurly@32..33 "{"
                    Space@33..34 " "
                    ReturnStmt@34..42
                      ReturnKw@34..40 "return"
                      Space@40..41 " "
                      IdentExpr@41..42
                        Ident@41..42
                          Identifier@41..42 "x"
                    Space@42..43 " "
                    RightCurly@43..44 "}"
            "#]],
        );
    }

    #[test]
    fn struct_stmt() {
        check(
            "struct Point { x: int, y: int }",
            expect![[r#"
                StructStmt@0..31
                  StructKw@0..6 "struct"
                  Space@6..7 " "
                  Ident@7..12
                    Identifier@7..12 "Point"
                  Space@12..13 " "
                  LeftCurly@13..14 "{"
                  Space@14..15 " "
                  StructField@15..21
                    Ident@15..16
                      Identifier@15..16 "x"
                    Colon@16..17 ":"
                    Space@17..18 " "
                    NamedTypeAnnotation@18..21
                      Ident@18..21
                        Identifier@18..21 "int"
                  Comma@21..22 ","
                  Space@22..23 " "
                  StructField@23..29
                    Ident@23..24
                      Identifier@23..24 "y"
                    Colon@24..25 ":"
                    Space@25..26 " "
                    NamedTypeAnnotation@26..29
                      Ident@26..29
                        Identifier@26..29 "int"
                  Space@29..30 " "
                  RightCurly@30..31 "}"
            "#]],
        );
    }

    #[test]
    fn table_stmt_inline() {
        check(
            "table T = { x: int }",
            expect![[r#"
                TableStmt@0..20
                  TableKw@0..5 "table"
                  Space@5..6 " "
                  Ident@6..7
                    Identifier@6..7 "T"
                  Space@7..8 " "
                  Eq@8..9 "="
                  Space@9..10 " "
                  LeftCurly@10..11 "{"
                  Space@11..12 " "
                  StructField@12..18
                    Ident@12..13
                      Identifier@12..13 "x"
                    Colon@13..14 ":"
                    Space@14..15 " "
                    NamedTypeAnnotation@15..18
                      Ident@15..18
                        Identifier@15..18 "int"
                  Space@18..19 " "
                  RightCurly@19..20 "}"
            "#]],
        );
    }

    #[test]
    fn table_stmt_named() {
        check(
            "table T = Row",
            expect![[r#"
                TableStmt@0..13
                  TableKw@0..5 "table"
                  Space@5..6 " "
                  Ident@6..7
                    Identifier@6..7 "T"
                  Space@7..8 " "
                  Eq@8..9 "="
                  Space@9..10 " "
                  Ident@10..13
                    Identifier@10..13 "Row"
            "#]],
        );
    }

    #[test]
    fn impl_trait_for_type() {
        check(
            "impl Show for Point { fn show(self) { return self } }",
            expect![[r#"
                ImplStmt@0..53
                  ImplKw@0..4 "impl"
                  Space@4..5 " "
                  TraitRef@5..9
                    Ident@5..9
                      Identifier@5..9 "Show"
                  Space@9..10 " "
                  ForKw@10..13 "for"
                  Space@13..14 " "
                  Ident@14..19
                    Identifier@14..19 "Point"
                  Space@19..20 " "
                  LeftCurly@20..21 "{"
                  Space@21..22 " "
                  FuncStmt@22..51
                    FnKw@22..24 "fn"
                    Space@24..25 " "
                    Ident@25..29
                      Identifier@25..29 "show"
                    LeftParen@29..30 "("
                    FuncParam@30..34
                      Ident@30..34
                        Identifier@30..34 "self"
                    RightParen@34..35 ")"
                    Space@35..36 " "
                    BlockStmt@36..51
                      LeftCurly@36..37 "{"
                      Space@37..38 " "
                      ReturnStmt@38..49
                        ReturnKw@38..44 "return"
                        Space@44..45 " "
                        IdentExpr@45..49
                          Ident@45..49
                            Identifier@45..49 "self"
                      Space@49..50 " "
                      RightCurly@50..51 "}"
                  Space@51..52 " "
                  RightCurly@52..53 "}"
            "#]],
        );
    }

    #[test]
    fn trait_stmt() {
        check(
            "trait Show { fn show(self) -> str }",
            expect![[r#"
                TraitStmt@0..35
                  TraitKw@0..5 "trait"
                  Space@5..6 " "
                  Ident@6..10
                    Identifier@6..10 "Show"
                  Space@10..11 " "
                  LeftCurly@11..12 "{"
                  Space@12..13 " "
                  FuncStmt@13..33
                    FnKw@13..15 "fn"
                    Space@15..16 " "
                    Ident@16..20
                      Identifier@16..20 "show"
                    LeftParen@20..21 "("
                    FuncParam@21..25
                      Ident@21..25
                        Identifier@21..25 "self"
                    RightParen@25..26 ")"
                    Space@26..27 " "
                    Arrow@27..29 "->"
                    Space@29..30 " "
                    NamedTypeAnnotation@30..33
                      Ident@30..33
                        Identifier@30..33 "str"
                  Space@33..34 " "
                  RightCurly@34..35 "}"
            "#]],
        );
    }
}
