use yuzu_lexer::token_kind::TokenKind;
use yuzu_syntax::SyntaxKind;

use crate::grammar::expr::parse_expr;
use crate::grammar::rel::parse_query;
use crate::grammar::ty::parse_type;
use crate::grammar::{Trailing, delimited, delimited_non_empty, parse_ident, parse_stmts};
use crate::parser::{Parser, marker::CompletedMarker};
use crate::token_set::TokenSet;

/// The tokens a function can start with, its prefixes included.
const FUNC_START: TokenSet = TokenSet::new(&[
    TokenKind::PubKw,
    TokenKind::ExternalKw,
    TokenKind::AggKw,
    TokenKind::DefKw,
]);

pub(crate) fn parse_stmt(p: &mut Parser) -> Option<CompletedMarker> {
    // `pub` prefixes a declaration, so what follows it decides which one this
    // is. Each declaration bumps the `pub` itself, the way a function bumps
    // its own `external` and `agg`, so the keyword lands inside the node it
    // qualifies.
    if p.at(TokenKind::PubKw) {
        return Some(parse_public_stmt(p));
    }

    // Reserved words, so no lookahead is needed to tell a declaration from an
    // expression that happens to start with the same identifier.
    if p.at(TokenKind::DefKw) || p.at(TokenKind::AggKw) || p.at(TokenKind::ExternalKw) {
        return Some(parse_func_stmt(p));
    }
    if p.at(TokenKind::ImplKw) {
        return Some(parse_impl_stmt(p));
    }
    if p.at(TokenKind::TraitKw) {
        return Some(parse_trait_stmt(p));
    }
    if p.at(TokenKind::LetKw) {
        return Some(parse_let_stmt(p));
    }
    if p.at(TokenKind::ReturnKw) {
        return Some(parse_return_stmt(p));
    }
    if p.at(TokenKind::StructKw) {
        return Some(parse_struct_stmt(p));
    }
    if p.at(TokenKind::TableKw) {
        return Some(parse_table_stmt(p));
    }
    if p.at(TokenKind::ModKw) {
        return Some(parse_mod_stmt(p));
    }
    if p.at(TokenKind::ImportKw) {
        return Some(parse_import_stmt(p));
    }
    if at_from_import(p) {
        return Some(parse_from_import_stmt(p));
    }
    parse_expr_stmt(p)
}

/// A declaration behind `pub`. Only the kinds that carry a name can be
/// exported, so anything else here is reported against the `pub` rather than
/// parsed as a declaration it is not.
fn parse_public_stmt(p: &mut Parser) -> CompletedMarker {
    // `pub` alone is one token; `pub(mod)` is four, so the declaration it
    // qualifies sits further along.
    let declaration = if p.peek_nth_kind(1) == Some(TokenKind::LeftParen) {
        4
    } else {
        1
    };

    match p.peek_nth_kind(declaration) {
        Some(TokenKind::DefKw | TokenKind::AggKw | TokenKind::ExternalKw) => parse_func_stmt(p),
        Some(TokenKind::TraitKw) => parse_trait_stmt(p),
        Some(TokenKind::LetKw) => parse_let_stmt(p),
        Some(TokenKind::StructKw) => parse_struct_stmt(p),
        Some(TokenKind::TableKw) => parse_table_stmt(p),
        Some(TokenKind::ModKw) => parse_mod_stmt(p),
        // A query cannot be exported, so a `from` here is an import.
        Some(TokenKind::FromKw) => parse_from_import_stmt(p),
        Some(TokenKind::ImportKw) => parse_import_stmt(p),
        _ => {
            let m = p.start();
            parse_visibility(p);
            p.error_declaration();
            p.complete(m, SyntaxKind::Error)
        }
    }
}

/// Bumps a leading `pub`, and the `(mod)` that narrows it, so every
/// declaration parser starts the same way.
fn parse_visibility(p: &mut Parser) {
    if !p.at(TokenKind::PubKw) {
        return;
    }

    p.bump();
    if p.at(TokenKind::LeftParen) {
        p.bump();
        p.expect(TokenKind::ModKw);
        p.expect(TokenKind::RightParen);
    }
}

/// Whether `from` starts an import rather than a query. Both begin with the
/// same keyword, and the path between them is any number of segments, so the
/// scan runs to the `import` that settles it. A query's `from` is followed by
/// a relation and then a pipe or the end of the statement, never by `import`.
fn at_from_import(p: &mut Parser) -> bool {
    if p.peek_kind() != Some(TokenKind::FromKw) {
        return false;
    }

    let mut at = 1;
    loop {
        if p.peek_nth_kind(at) != Some(TokenKind::Identifier) {
            return false;
        }

        at += 1;
        match p.peek_nth_kind(at) {
            Some(TokenKind::Dot) => at += 1,
            Some(TokenKind::ImportKw) => return true,
            _ => return false,
        }
    }
}

/// `mod name`: a submodule of this one, which is what makes it part of the
/// program. A directory's files are declared rather than discovered, so a
/// file nobody declares is not compiled and cannot be named.
fn parse_mod_stmt(p: &mut Parser) -> CompletedMarker {
    let m = p.start();
    parse_visibility(p);
    p.expect(TokenKind::ModKw);
    parse_ident(p);
    p.complete(m, SyntaxKind::ModStmt)
}

/// `import a.b`, optionally renamed by `as`.
fn parse_import_stmt(p: &mut Parser) -> CompletedMarker {
    let m = p.start();
    parse_visibility(p);
    p.expect(TokenKind::ImportKw);
    parse_module_path(p);
    parse_rename(p);
    p.complete(m, SyntaxKind::ImportStmt)
}

/// `from a.b import x, y as z`.
fn parse_from_import_stmt(p: &mut Parser) -> CompletedMarker {
    let m = p.start();
    parse_visibility(p);
    p.expect(TokenKind::FromKw);
    parse_module_path(p);
    p.expect(TokenKind::ImportKw);
    parse_import_item(p);
    while p.at(TokenKind::Comma) {
        p.bump();
        parse_import_item(p);
    }

    p.complete(m, SyntaxKind::FromImportStmt)
}

/// The dotted name of a module: one segment, or a path through the modules
/// holding it.
fn parse_module_path(p: &mut Parser) -> CompletedMarker {
    let m = p.start();
    parse_ident(p);
    while p.at(TokenKind::Dot) {
        p.bump();
        parse_ident(p);
    }

    p.complete(m, SyntaxKind::ModulePath)
}

fn parse_import_item(p: &mut Parser) -> CompletedMarker {
    let m = p.start();
    parse_ident(p);
    parse_rename(p);
    p.complete(m, SyntaxKind::ImportItem)
}

/// The `as` that names an import something else in this file.
fn parse_rename(p: &mut Parser) {
    if p.at(TokenKind::AsKw) {
        p.bump();
        parse_ident(p);
    }
}

fn parse_block_stmt(p: &mut Parser) -> CompletedMarker {
    let m = p.start();

    p.expect(TokenKind::LeftCurly);
    parse_stmts(p, |p| p.at(TokenKind::RightCurly) || p.at_end());
    p.expect(TokenKind::RightCurly);

    p.complete(m, SyntaxKind::BlockStmt)
}

fn parse_func_stmt(p: &mut Parser) -> CompletedMarker {
    let m = p.start();
    parse_func_prefix(p);
    parse_ident(p);

    if p.at(TokenKind::LeftSquare) {
        p.bump();
        delimited_non_empty(p, TokenKind::RightSquare, Trailing::Forbidden, |p| {
            parse_type_param(p);
        });
        p.expect(TokenKind::RightSquare);
    }

    p.expect(TokenKind::LeftParen);
    delimited(p, TokenKind::RightParen, Trailing::Forbidden, |p| {
        parse_param(p);
    });
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
    parse_methods(p, |p| {
        parse_func_stmt(p);
    });
    p.expect(TokenKind::RightCurly);

    p.complete(m, SyntaxKind::ImplStmt)
}

fn parse_trait_stmt(p: &mut Parser) -> CompletedMarker {
    let m = p.start();
    parse_visibility(p);
    p.expect(TokenKind::TraitKw);
    parse_ident(p);
    p.expect(TokenKind::LeftCurly);
    parse_methods(p, |p| {
        parse_trait_method(p);
    });
    p.expect(TokenKind::RightCurly);

    p.complete(m, SyntaxKind::TraitStmt)
}

/// The methods of an `impl` or a `trait`, up to its `}`. Anything else is
/// reported once and kept in an error node, so the rest of the body stays
/// in it.
fn parse_methods(p: &mut Parser, mut method: impl FnMut(&mut Parser)) {
    while !p.at(TokenKind::RightCurly) && !p.at_end() {
        if p.at_any(FUNC_START) {
            method(p);
            continue;
        }
        p.error_in_place();
        let m = p.start();
        while !p.at_any(FUNC_START) && !p.at(TokenKind::RightCurly) && !p.at_end() {
            p.bump();
        }
        p.complete(m, SyntaxKind::Error);
    }
}

/// `pub`, `external` and `agg`, each when present, then `def`.
fn parse_func_prefix(p: &mut Parser) {
    parse_visibility(p);
    if p.at(TokenKind::ExternalKw) {
        p.bump();
    }
    if p.at(TokenKind::AggKw) {
        p.bump();
    }
    p.expect(TokenKind::DefKw);
}

fn parse_trait_method(p: &mut Parser) -> CompletedMarker {
    let m = p.start();
    parse_func_prefix(p);
    parse_ident(p);
    p.expect(TokenKind::LeftParen);
    delimited(p, TokenKind::RightParen, Trailing::Forbidden, |p| {
        parse_param(p);
    });
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
        parse_value(p);
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

/// An expression, a query or an assignment. A statement with no expression
/// is no statement: the error is reported, and no empty node is left.
fn parse_expr_stmt(p: &mut Parser) -> Option<CompletedMarker> {
    let m = p.start();

    if p.at(TokenKind::FromKw) {
        parse_query(p);
        return Some(p.complete(m, SyntaxKind::ExprStmt));
    }

    let expr = parse_expr(p);

    if p.at(TokenKind::Eq) {
        p.bump();
        parse_value(p);
        return Some(p.complete(m, SyntaxKind::AssignStmt));
    }

    if expr.is_none() {
        p.abandon(m);
        return None;
    }
    Some(p.complete(m, SyntaxKind::ExprStmt))
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
    delimited(p, TokenKind::RightCurly, Trailing::Allowed, |p| {
        parse_struct_field_decl(p);
    });
    p.expect(TokenKind::RightCurly);
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};

    use super::*;
    use crate::grammar::test_support;

    fn check(input: &str, expected: &Expect) {
        test_support::check(input, parse_stmt, expected);
    }

    #[test]
    fn parse_external_fn_stmt() {
        test_support::check(
            "external def upper(s: str) -> str",
            parse_stmt,
            &expect![[r#"
                FuncStmt@0..33
                  ExternalKw@0..8 "external"
                  Whitespace@8..9 " "
                  DefKw@9..12 "def"
                  Whitespace@12..13 " "
                  Ident@13..18
                    Identifier@13..18 "upper"
                  LeftParen@18..19 "("
                  FuncParam@19..25
                    Ident@19..20
                      Identifier@19..20 "s"
                    Colon@20..21 ":"
                    Whitespace@21..22 " "
                    NamedTypeAnnotation@22..25
                      Ident@22..25
                        Identifier@22..25 "str"
                  RightParen@25..26 ")"
                  Whitespace@26..27 " "
                  Arrow@27..29 "->"
                  Whitespace@29..30 " "
                  NamedTypeAnnotation@30..33
                    Ident@30..33
                      Identifier@30..33 "str"
            "#]],
        );
    }

    #[test]
    fn parse_external_agg_fn_stmt() {
        test_support::check(
            "external agg def median(x: int64) -> float64",
            parse_stmt,
            &expect![[r#"
                FuncStmt@0..44
                  ExternalKw@0..8 "external"
                  Whitespace@8..9 " "
                  AggKw@9..12 "agg"
                  Whitespace@12..13 " "
                  DefKw@13..16 "def"
                  Whitespace@16..17 " "
                  Ident@17..23
                    Identifier@17..23 "median"
                  LeftParen@23..24 "("
                  FuncParam@24..32
                    Ident@24..25
                      Identifier@24..25 "x"
                    Colon@25..26 ":"
                    Whitespace@26..27 " "
                    NamedTypeAnnotation@27..32
                      Ident@27..32
                        Identifier@27..32 "int64"
                  RightParen@32..33 ")"
                  Whitespace@33..34 " "
                  Arrow@34..36 "->"
                  Whitespace@36..37 " "
                  NamedTypeAnnotation@37..44
                    Ident@37..44
                      Identifier@37..44 "float64"
            "#]],
        );
    }

    #[test]
    fn parse_agg_func_stmt() {
        test_support::check(
            "agg def spread(x: int64) -> int64 { return sum(x) }",
            parse_stmt,
            &expect![[r#"
                FuncStmt@0..51
                  AggKw@0..3 "agg"
                  Whitespace@3..4 " "
                  DefKw@4..7 "def"
                  Whitespace@7..8 " "
                  Ident@8..14
                    Identifier@8..14 "spread"
                  LeftParen@14..15 "("
                  FuncParam@15..23
                    Ident@15..16
                      Identifier@15..16 "x"
                    Colon@16..17 ":"
                    Whitespace@17..18 " "
                    NamedTypeAnnotation@18..23
                      Ident@18..23
                        Identifier@18..23 "int64"
                  RightParen@23..24 ")"
                  Whitespace@24..25 " "
                  Arrow@25..27 "->"
                  Whitespace@27..28 " "
                  NamedTypeAnnotation@28..33
                    Ident@28..33
                      Identifier@28..33 "int64"
                  Whitespace@33..34 " "
                  BlockStmt@34..51
                    LeftCurly@34..35 "{"
                    Whitespace@35..36 " "
                    ReturnStmt@36..49
                      ReturnKw@36..42 "return"
                      Whitespace@42..43 " "
                      CallExpr@43..49
                        IdentExpr@43..46
                          Ident@43..46
                            Identifier@43..46 "sum"
                        ArgList@46..49
                          LeftParen@46..47 "("
                          IdentExpr@47..48
                            Ident@47..48
                              Identifier@47..48 "x"
                          RightParen@48..49 ")"
                    Whitespace@49..50 " "
                    RightCurly@50..51 "}"
            "#]],
        );
    }

    #[test]
    fn agg_is_reserved_and_cannot_name_a_binding() {
        test_support::check(
            "let agg = 1",
            parse_stmt,
            &expect![[r#"
                LetStmt@0..11
                  LetKw@0..3 "let"
                  Whitespace@3..4 " "
                  Error@4..7
                    AggKw@4..7 "agg"
                  Whitespace@7..8 " "
                  Eq@8..9 "="
                  Whitespace@9..10 " "
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
            &expect![[r#"
                BlockStmt@0..12
                  LeftCurly@0..1 "{"
                  Whitespace@1..2 " "
                  ReturnStmt@2..10
                    ReturnKw@2..8 "return"
                    Whitespace@8..9 " "
                    IdentExpr@9..10
                      Ident@9..10
                        Identifier@9..10 "x"
                  Whitespace@10..11 " "
                  RightCurly@11..12 "}"
            "#]],
        );
    }

    #[test]
    fn parse_func_stmt_directly() {
        test_support::check(
            "def f(x: int) -> int { return x }",
            parse_func_stmt,
            &expect![[r#"
                FuncStmt@0..33
                  DefKw@0..3 "def"
                  Whitespace@3..4 " "
                  Ident@4..5
                    Identifier@4..5 "f"
                  LeftParen@5..6 "("
                  FuncParam@6..12
                    Ident@6..7
                      Identifier@6..7 "x"
                    Colon@7..8 ":"
                    Whitespace@8..9 " "
                    NamedTypeAnnotation@9..12
                      Ident@9..12
                        Identifier@9..12 "int"
                  RightParen@12..13 ")"
                  Whitespace@13..14 " "
                  Arrow@14..16 "->"
                  Whitespace@16..17 " "
                  NamedTypeAnnotation@17..20
                    Ident@17..20
                      Identifier@17..20 "int"
                  Whitespace@20..21 " "
                  BlockStmt@21..33
                    LeftCurly@21..22 "{"
                    Whitespace@22..23 " "
                    ReturnStmt@23..31
                      ReturnKw@23..29 "return"
                      Whitespace@29..30 " "
                      IdentExpr@30..31
                        Ident@30..31
                          Identifier@30..31 "x"
                    Whitespace@31..32 " "
                    RightCurly@32..33 "}"
            "#]],
        );
    }

    #[test]
    fn parse_impl_stmt_directly() {
        test_support::check(
            "impl Point { def x(self) { return self } }",
            parse_impl_stmt,
            &expect![[r#"
                ImplStmt@0..42
                  ImplKw@0..4 "impl"
                  Whitespace@4..5 " "
                  Ident@5..10
                    Identifier@5..10 "Point"
                  Whitespace@10..11 " "
                  LeftCurly@11..12 "{"
                  Whitespace@12..13 " "
                  FuncStmt@13..40
                    DefKw@13..16 "def"
                    Whitespace@16..17 " "
                    Ident@17..18
                      Identifier@17..18 "x"
                    LeftParen@18..19 "("
                    FuncParam@19..23
                      Ident@19..23
                        Identifier@19..23 "self"
                    RightParen@23..24 ")"
                    Whitespace@24..25 " "
                    BlockStmt@25..40
                      LeftCurly@25..26 "{"
                      Whitespace@26..27 " "
                      ReturnStmt@27..38
                        ReturnKw@27..33 "return"
                        Whitespace@33..34 " "
                        IdentExpr@34..38
                          Ident@34..38
                            Identifier@34..38 "self"
                      Whitespace@38..39 " "
                      RightCurly@39..40 "}"
                  Whitespace@40..41 " "
                  RightCurly@41..42 "}"
            "#]],
        );
    }

    #[test]
    fn parse_trait_stmt_directly() {
        test_support::check(
            "trait Show { def show(self) -> str }",
            parse_trait_stmt,
            &expect![[r#"
                TraitStmt@0..36
                  TraitKw@0..5 "trait"
                  Whitespace@5..6 " "
                  Ident@6..10
                    Identifier@6..10 "Show"
                  Whitespace@10..11 " "
                  LeftCurly@11..12 "{"
                  Whitespace@12..13 " "
                  FuncStmt@13..34
                    DefKw@13..16 "def"
                    Whitespace@16..17 " "
                    Ident@17..21
                      Identifier@17..21 "show"
                    LeftParen@21..22 "("
                    FuncParam@22..26
                      Ident@22..26
                        Identifier@22..26 "self"
                    RightParen@26..27 ")"
                    Whitespace@27..28 " "
                    Arrow@28..30 "->"
                    Whitespace@30..31 " "
                    NamedTypeAnnotation@31..34
                      Ident@31..34
                        Identifier@31..34 "str"
                  Whitespace@34..35 " "
                  RightCurly@35..36 "}"
            "#]],
        );
    }

    #[test]
    fn parse_trait_method_directly() {
        test_support::check(
            "def show(self) -> str",
            parse_trait_method,
            &expect![[r#"
                FuncStmt@0..21
                  DefKw@0..3 "def"
                  Whitespace@3..4 " "
                  Ident@4..8
                    Identifier@4..8 "show"
                  LeftParen@8..9 "("
                  FuncParam@9..13
                    Ident@9..13
                      Identifier@9..13 "self"
                  RightParen@13..14 ")"
                  Whitespace@14..15 " "
                  Arrow@15..17 "->"
                  Whitespace@17..18 " "
                  NamedTypeAnnotation@18..21
                    Ident@18..21
                      Identifier@18..21 "str"
            "#]],
        );
    }

    #[test]
    fn parse_let_stmt_directly() {
        test_support::check(
            "let mut x: int = 1",
            parse_let_stmt,
            &expect![[r#"
                LetStmt@0..18
                  LetKw@0..3 "let"
                  Whitespace@3..4 " "
                  MutKw@4..7 "mut"
                  Whitespace@7..8 " "
                  Ident@8..9
                    Identifier@8..9 "x"
                  Colon@9..10 ":"
                  Whitespace@10..11 " "
                  NamedTypeAnnotation@11..14
                    Ident@11..14
                      Identifier@11..14 "int"
                  Whitespace@14..15 " "
                  Eq@15..16 "="
                  Whitespace@16..17 " "
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
            &expect![[r#"
                ReturnStmt@0..8
                  ReturnKw@0..6 "return"
                  Whitespace@6..7 " "
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
            &expect![[r#"
                StructStmt@0..19
                  StructKw@0..6 "struct"
                  Whitespace@6..7 " "
                  Ident@7..8
                    Identifier@7..8 "P"
                  Whitespace@8..9 " "
                  LeftCurly@9..10 "{"
                  Whitespace@10..11 " "
                  StructField@11..17
                    Ident@11..12
                      Identifier@11..12 "x"
                    Colon@12..13 ":"
                    Whitespace@13..14 " "
                    NamedTypeAnnotation@14..17
                      Ident@14..17
                        Identifier@14..17 "int"
                  Whitespace@17..18 " "
                  RightCurly@18..19 "}"
            "#]],
        );
    }

    #[test]
    fn parse_table_stmt_directly() {
        test_support::check(
            "table T = Row",
            parse_table_stmt,
            &expect![[r#"
                TableStmt@0..13
                  TableKw@0..5 "table"
                  Whitespace@5..6 " "
                  Ident@6..7
                    Identifier@6..7 "T"
                  Whitespace@7..8 " "
                  Eq@8..9 "="
                  Whitespace@9..10 " "
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
            &expect![[r#"
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
    fn parse_value_with_query() {
        test_support::check(
            "from t |> select a",
            parse_value,
            &expect![[r#"
                Pipeline@0..18
                  FromSource@0..6
                    FromKw@0..4 "from"
                    Whitespace@4..5 " "
                    Ident@5..6
                      Identifier@5..6 "t"
                  Whitespace@6..7 " "
                  SelectStage@7..18
                    Pipe@7..9 "|>"
                    Whitespace@9..10 " "
                    SelectKw@10..16 "select"
                    Whitespace@16..17 " "
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
            &expect![[r#"
                FuncParam@0..6
                  Ident@0..1
                    Identifier@0..1 "x"
                  Colon@1..2 ":"
                  Whitespace@2..3 " "
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
            &expect![[r#"
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
            &expect![[r#"
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
            &expect![[r#"
                TypeBound@0..11
                  Ident@0..1
                    Identifier@0..1 "T"
                  Colon@1..2 ":"
                  Whitespace@2..3 " "
                  TraitRef@3..6
                    Ident@3..6
                      Identifier@3..6 "Add"
                  Whitespace@6..7 " "
                  Plus@7..8 "+"
                  Whitespace@8..9 " "
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
            &expect![[r#"
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
            &expect![[r#"
                StructStmt@0..35
                  PubKw@0..3 "pub"
                  Whitespace@3..4 " "
                  StructKw@4..10 "struct"
                  Whitespace@10..11 " "
                  Ident@11..12
                    Identifier@11..12 "P"
                  Whitespace@12..13 " "
                  LeftCurly@13..14 "{"
                  Whitespace@14..15 " "
                  StructField@15..25
                    PubKw@15..18 "pub"
                    Whitespace@18..19 " "
                    Ident@19..20
                      Identifier@19..20 "x"
                    Colon@20..21 ":"
                    Whitespace@21..22 " "
                    NamedTypeAnnotation@22..25
                      Ident@22..25
                        Identifier@22..25 "int"
                  Comma@25..26 ","
                  Whitespace@26..27 " "
                  StructField@27..33
                    Ident@27..28
                      Identifier@27..28 "y"
                    Colon@28..29 ":"
                    Whitespace@29..30 " "
                    NamedTypeAnnotation@30..33
                      Ident@30..33
                        Identifier@30..33 "int"
                  Whitespace@33..34 " "
                  RightCurly@34..35 "}"
            "#]],
        );
    }

    /// `pub(mod)` narrows the reach rather than naming a different thing, so
    /// it belongs to the declaration the same way a bare `pub` does.
    #[test]
    fn parse_module_visible_declaration() {
        check(
            "pub(mod) def f(x: int) -> int { return x }",
            &expect![[r#"
                FuncStmt@0..42
                  PubKw@0..3 "pub"
                  LeftParen@3..4 "("
                  ModKw@4..7 "mod"
                  RightParen@7..8 ")"
                  Whitespace@8..9 " "
                  DefKw@9..12 "def"
                  Whitespace@12..13 " "
                  Ident@13..14
                    Identifier@13..14 "f"
                  LeftParen@14..15 "("
                  FuncParam@15..21
                    Ident@15..16
                      Identifier@15..16 "x"
                    Colon@16..17 ":"
                    Whitespace@17..18 " "
                    NamedTypeAnnotation@18..21
                      Ident@18..21
                        Identifier@18..21 "int"
                  RightParen@21..22 ")"
                  Whitespace@22..23 " "
                  Arrow@23..25 "->"
                  Whitespace@25..26 " "
                  NamedTypeAnnotation@26..29
                    Ident@26..29
                      Identifier@26..29 "int"
                  Whitespace@29..30 " "
                  BlockStmt@30..42
                    LeftCurly@30..31 "{"
                    Whitespace@31..32 " "
                    ReturnStmt@32..40
                      ReturnKw@32..38 "return"
                      Whitespace@38..39 " "
                      IdentExpr@39..40
                        Ident@39..40
                          Identifier@39..40 "x"
                    Whitespace@40..41 " "
                    RightCurly@41..42 "}"
            "#]],
        );
    }

    /// A submodule is declared rather than discovered, and carries its own
    /// reach like any other declaration.
    #[test]
    fn parse_module_declarations() {
        check(
            "mod internal",
            &expect![[r#"
                ModStmt@0..12
                  ModKw@0..3 "mod"
                  Whitespace@3..4 " "
                  Ident@4..12
                    Identifier@4..12 "internal"
            "#]],
        );
        check(
            "pub mod math",
            &expect![[r#"
                ModStmt@0..12
                  PubKw@0..3 "pub"
                  Whitespace@3..4 " "
                  ModKw@4..7 "mod"
                  Whitespace@7..8 " "
                  Ident@8..12
                    Identifier@8..12 "math"
            "#]],
        );
    }

    #[test]
    fn parse_import_of_a_path() {
        check(
            "import yuzu.std.math as m",
            &expect![[r#"
                ImportStmt@0..25
                  ImportKw@0..6 "import"
                  Whitespace@6..7 " "
                  ModulePath@7..20
                    Ident@7..11
                      Identifier@7..11 "yuzu"
                    Dot@11..12 "."
                    Ident@12..15
                      Identifier@12..15 "std"
                    Dot@15..16 "."
                    Ident@16..20
                      Identifier@16..20 "math"
                  Whitespace@20..21 " "
                  AsKw@21..23 "as"
                  Whitespace@23..24 " "
                  Ident@24..25
                    Identifier@24..25 "m"
            "#]],
        );
    }

    #[test]
    fn parse_from_import_with_renames() {
        check(
            "from helpers import spread, avg3 as mean",
            &expect![[r#"
                FromImportStmt@0..40
                  FromKw@0..4 "from"
                  Whitespace@4..5 " "
                  ModulePath@5..12
                    Ident@5..12
                      Identifier@5..12 "helpers"
                  Whitespace@12..13 " "
                  ImportKw@13..19 "import"
                  Whitespace@19..20 " "
                  ImportItem@20..26
                    Ident@20..26
                      Identifier@20..26 "spread"
                  Comma@26..27 ","
                  Whitespace@27..28 " "
                  ImportItem@28..40
                    Ident@28..32
                      Identifier@28..32 "avg3"
                    Whitespace@32..33 " "
                    AsKw@33..35 "as"
                    Whitespace@35..36 " "
                    Ident@36..40
                      Identifier@36..40 "mean"
            "#]],
        );
    }

    #[test]
    fn parse_a_public_import() {
        check(
            "pub from helpers import spread",
            &expect![[r#"
                FromImportStmt@0..30
                  PubKw@0..3 "pub"
                  Whitespace@3..4 " "
                  FromKw@4..8 "from"
                  Whitespace@8..9 " "
                  ModulePath@9..16
                    Ident@9..16
                      Identifier@9..16 "helpers"
                  Whitespace@16..17 " "
                  ImportKw@17..23 "import"
                  Whitespace@23..24 " "
                  ImportItem@24..30
                    Ident@24..30
                      Identifier@24..30 "spread"
            "#]],
        );
    }

    /// `from` starts both an import and a query, and the path between them is
    /// any length, so the scan has to reach the `import` to decide. A query
    /// must still come out a query.
    #[test]
    fn a_query_is_not_an_import() {
        check(
            "from t |> select a as v",
            &expect![[r#"
                ExprStmt@0..23
                  Pipeline@0..23
                    FromSource@0..6
                      FromKw@0..4 "from"
                      Whitespace@4..5 " "
                      Ident@5..6
                        Identifier@5..6 "t"
                    Whitespace@6..7 " "
                    SelectStage@7..23
                      Pipe@7..9 "|>"
                      Whitespace@9..10 " "
                      SelectKw@10..16 "select"
                      Whitespace@16..17 " "
                      SelectItem@17..23
                        IdentExpr@17..18
                          Ident@17..18
                            Identifier@17..18 "a"
                        Whitespace@18..19 " "
                        AsKw@19..21 "as"
                        Whitespace@21..22 " "
                        Ident@22..23
                          Identifier@22..23 "v"
            "#]],
        );
    }

    #[test]
    fn parse_struct_field_decl_directly() {
        test_support::check(
            "x: int",
            parse_struct_field_decl,
            &expect![[r#"
                StructField@0..6
                  Ident@0..1
                    Identifier@0..1 "x"
                  Colon@1..2 ":"
                  Whitespace@2..3 " "
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
            &expect![[r#"
                LetStmt@0..9
                  LetKw@0..3 "let"
                  Whitespace@3..4 " "
                  Ident@4..5
                    Identifier@4..5 "x"
                  Whitespace@5..6 " "
                  Eq@6..7 "="
                  Whitespace@7..8 " "
                  IntLiteral@8..9
                    IntLit@8..9 "1"
            "#]],
        );
    }

    #[test]
    fn assign_stmt() {
        check(
            "x = 5",
            &expect![[r#"
                AssignStmt@0..5
                  IdentExpr@0..1
                    Ident@0..1
                      Identifier@0..1 "x"
                  Whitespace@1..2 " "
                  Eq@2..3 "="
                  Whitespace@3..4 " "
                  IntLiteral@4..5
                    IntLit@4..5 "5"
            "#]],
        );
    }

    #[test]
    fn func_stmt() {
        check(
            "def add(x: int, y: int) -> int { return x }",
            &expect![[r#"
                FuncStmt@0..43
                  DefKw@0..3 "def"
                  Whitespace@3..4 " "
                  Ident@4..7
                    Identifier@4..7 "add"
                  LeftParen@7..8 "("
                  FuncParam@8..14
                    Ident@8..9
                      Identifier@8..9 "x"
                    Colon@9..10 ":"
                    Whitespace@10..11 " "
                    NamedTypeAnnotation@11..14
                      Ident@11..14
                        Identifier@11..14 "int"
                  Comma@14..15 ","
                  Whitespace@15..16 " "
                  FuncParam@16..22
                    Ident@16..17
                      Identifier@16..17 "y"
                    Colon@17..18 ":"
                    Whitespace@18..19 " "
                    NamedTypeAnnotation@19..22
                      Ident@19..22
                        Identifier@19..22 "int"
                  RightParen@22..23 ")"
                  Whitespace@23..24 " "
                  Arrow@24..26 "->"
                  Whitespace@26..27 " "
                  NamedTypeAnnotation@27..30
                    Ident@27..30
                      Identifier@27..30 "int"
                  Whitespace@30..31 " "
                  BlockStmt@31..43
                    LeftCurly@31..32 "{"
                    Whitespace@32..33 " "
                    ReturnStmt@33..41
                      ReturnKw@33..39 "return"
                      Whitespace@39..40 " "
                      IdentExpr@40..41
                        Ident@40..41
                          Identifier@40..41 "x"
                    Whitespace@41..42 " "
                    RightCurly@42..43 "}"
            "#]],
        );
    }

    #[test]
    fn generic_func_with_where_clause() {
        check(
            "def id[T](x: T) -> T where T: Eq { return x }",
            &expect![[r#"
                FuncStmt@0..45
                  DefKw@0..3 "def"
                  Whitespace@3..4 " "
                  Ident@4..6
                    Identifier@4..6 "id"
                  LeftSquare@6..7 "["
                  TypeParam@7..8
                    Ident@7..8
                      Identifier@7..8 "T"
                  RightSquare@8..9 "]"
                  LeftParen@9..10 "("
                  FuncParam@10..14
                    Ident@10..11
                      Identifier@10..11 "x"
                    Colon@11..12 ":"
                    Whitespace@12..13 " "
                    NamedTypeAnnotation@13..14
                      Ident@13..14
                        Identifier@13..14 "T"
                  RightParen@14..15 ")"
                  Whitespace@15..16 " "
                  Arrow@16..18 "->"
                  Whitespace@18..19 " "
                  NamedTypeAnnotation@19..20
                    Ident@19..20
                      Identifier@19..20 "T"
                  Whitespace@20..21 " "
                  WhereKw@21..26 "where"
                  Whitespace@26..27 " "
                  TypeBound@27..32
                    Ident@27..28
                      Identifier@27..28 "T"
                    Colon@28..29 ":"
                    Whitespace@29..30 " "
                    TraitRef@30..32
                      Ident@30..32
                        Identifier@30..32 "Eq"
                  Whitespace@32..33 " "
                  BlockStmt@33..45
                    LeftCurly@33..34 "{"
                    Whitespace@34..35 " "
                    ReturnStmt@35..43
                      ReturnKw@35..41 "return"
                      Whitespace@41..42 " "
                      IdentExpr@42..43
                        Ident@42..43
                          Identifier@42..43 "x"
                    Whitespace@43..44 " "
                    RightCurly@44..45 "}"
            "#]],
        );
    }

    #[test]
    fn struct_stmt() {
        check(
            "struct Point { x: int, y: int }",
            &expect![[r#"
                StructStmt@0..31
                  StructKw@0..6 "struct"
                  Whitespace@6..7 " "
                  Ident@7..12
                    Identifier@7..12 "Point"
                  Whitespace@12..13 " "
                  LeftCurly@13..14 "{"
                  Whitespace@14..15 " "
                  StructField@15..21
                    Ident@15..16
                      Identifier@15..16 "x"
                    Colon@16..17 ":"
                    Whitespace@17..18 " "
                    NamedTypeAnnotation@18..21
                      Ident@18..21
                        Identifier@18..21 "int"
                  Comma@21..22 ","
                  Whitespace@22..23 " "
                  StructField@23..29
                    Ident@23..24
                      Identifier@23..24 "y"
                    Colon@24..25 ":"
                    Whitespace@25..26 " "
                    NamedTypeAnnotation@26..29
                      Ident@26..29
                        Identifier@26..29 "int"
                  Whitespace@29..30 " "
                  RightCurly@30..31 "}"
            "#]],
        );
    }

    #[test]
    fn table_stmt_inline() {
        check(
            "table T = { x: int }",
            &expect![[r#"
                TableStmt@0..20
                  TableKw@0..5 "table"
                  Whitespace@5..6 " "
                  Ident@6..7
                    Identifier@6..7 "T"
                  Whitespace@7..8 " "
                  Eq@8..9 "="
                  Whitespace@9..10 " "
                  LeftCurly@10..11 "{"
                  Whitespace@11..12 " "
                  StructField@12..18
                    Ident@12..13
                      Identifier@12..13 "x"
                    Colon@13..14 ":"
                    Whitespace@14..15 " "
                    NamedTypeAnnotation@15..18
                      Ident@15..18
                        Identifier@15..18 "int"
                  Whitespace@18..19 " "
                  RightCurly@19..20 "}"
            "#]],
        );
    }

    #[test]
    fn impl_trait_for_type() {
        check(
            "impl Show for Point { def show(self) { return self } }",
            &expect![[r#"
                ImplStmt@0..54
                  ImplKw@0..4 "impl"
                  Whitespace@4..5 " "
                  TraitRef@5..9
                    Ident@5..9
                      Identifier@5..9 "Show"
                  Whitespace@9..10 " "
                  ForKw@10..13 "for"
                  Whitespace@13..14 " "
                  Ident@14..19
                    Identifier@14..19 "Point"
                  Whitespace@19..20 " "
                  LeftCurly@20..21 "{"
                  Whitespace@21..22 " "
                  FuncStmt@22..52
                    DefKw@22..25 "def"
                    Whitespace@25..26 " "
                    Ident@26..30
                      Identifier@26..30 "show"
                    LeftParen@30..31 "("
                    FuncParam@31..35
                      Ident@31..35
                        Identifier@31..35 "self"
                    RightParen@35..36 ")"
                    Whitespace@36..37 " "
                    BlockStmt@37..52
                      LeftCurly@37..38 "{"
                      Whitespace@38..39 " "
                      ReturnStmt@39..50
                        ReturnKw@39..45 "return"
                        Whitespace@45..46 " "
                        IdentExpr@46..50
                          Ident@46..50
                            Identifier@46..50 "self"
                      Whitespace@50..51 " "
                      RightCurly@51..52 "}"
                  Whitespace@52..53 " "
                  RightCurly@53..54 "}"
            "#]],
        );
    }
}
