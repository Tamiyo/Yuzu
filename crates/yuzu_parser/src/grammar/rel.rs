use yuzu_lexer::token_kind::TokenKind;
use yuzu_syntax::SyntaxKind;

use crate::grammar::expr::parse_expr;
use crate::grammar::parse_ident;
use crate::parser::{Parser, marker::CompletedMarker};

/// `from t |> where … |> select …`: the source, then each stage as a sibling
/// that holds only its own tokens.
pub(crate) fn parse_query(p: &mut Parser) -> CompletedMarker {
    let m = p.start();
    parse_from_source(p);
    while p.at(TokenKind::Pipe) {
        parse_stage(p);
    }

    p.complete(m, SyntaxKind::Pipeline)
}

fn parse_stage(p: &mut Parser) -> CompletedMarker {
    let m = p.start();
    p.bump();

    let kind = if p.at(TokenKind::WhereKw) {
        parse_where_clause(p);
        SyntaxKind::WhereStage
    } else if p.at(TokenKind::DistinctKw) {
        parse_distinct_clause(p);
        SyntaxKind::DistinctStage
    } else if p.at(TokenKind::DropKw) {
        parse_drop_clause(p);
        SyntaxKind::DropStage
    } else if p.at(TokenKind::RenameKw) {
        parse_rename_clause(p);
        SyntaxKind::RenameStage
    } else if p.at(TokenKind::ExtendKw) {
        parse_extend_clause(p);
        SyntaxKind::ExtendStage
    } else if p.at(TokenKind::SetKw) {
        parse_set_clause(p);
        SyntaxKind::SetStage
    } else if p.at(TokenKind::LimitKw) {
        parse_limit_clause(p);
        SyntaxKind::LimitStage
    } else if p.at(TokenKind::AsKw) {
        parse_alias_clause(p);
        SyntaxKind::AliasStage
    } else if p.at(TokenKind::AggregateKw) {
        parse_aggregate_clause(p);
        SyntaxKind::AggregateStage
    } else if at_join_clause(p) {
        parse_join_clause(p);
        SyntaxKind::JoinStage
    } else {
        parse_select_clause(p);
        SyntaxKind::SelectStage
    };

    p.complete(m, kind)
}

fn parse_from_source(p: &mut Parser) -> CompletedMarker {
    let m = p.start();
    p.expect(TokenKind::FromKw);
    parse_ident(p);

    if p.at(TokenKind::AsKw) {
        p.bump();
        parse_ident(p);
    } else if p.at(TokenKind::Identifier) {
        parse_ident(p);
    }

    p.complete(m, SyntaxKind::FromSource)
}

fn parse_select_item(p: &mut Parser) -> CompletedMarker {
    let m = p.start();
    parse_expr(p);

    if p.at(TokenKind::AsKw) {
        p.bump();
        parse_ident(p);
    }

    p.complete(m, SyntaxKind::SelectItem)
}

fn parse_select_clause(p: &mut Parser) {
    p.expect(TokenKind::SelectKw);
    parse_select_item(p);
    while p.at(TokenKind::Comma) {
        p.bump();
        parse_select_item(p);
    }
}

fn parse_where_clause(p: &mut Parser) {
    p.expect(TokenKind::WhereKw);
    parse_expr(p);
}

fn parse_distinct_clause(p: &mut Parser) {
    p.expect(TokenKind::DistinctKw);
}

fn parse_drop_clause(p: &mut Parser) {
    p.expect(TokenKind::DropKw);
    parse_ident(p);
    while p.at(TokenKind::Comma) {
        p.bump();
        parse_ident(p);
    }
}

fn parse_rename_item(p: &mut Parser) {
    let m = p.start();
    parse_ident(p);
    if p.at(TokenKind::Dot) {
        p.bump();
        parse_ident(p);
    }
    p.expect(TokenKind::AsKw);
    parse_ident(p);
    p.complete(m, SyntaxKind::RenameItem);
}

fn parse_rename_clause(p: &mut Parser) {
    p.expect(TokenKind::RenameKw);
    parse_rename_item(p);
    while p.at(TokenKind::Comma) {
        p.bump();
        parse_rename_item(p);
    }
}

fn parse_extend_clause(p: &mut Parser) {
    p.expect(TokenKind::ExtendKw);
    parse_select_item(p);
    while p.at(TokenKind::Comma) {
        p.bump();
        parse_select_item(p);
    }
}

fn parse_set_item(p: &mut Parser) {
    let m = p.start();
    parse_ident(p);
    p.expect(TokenKind::Eq);
    parse_expr(p);
    p.complete(m, SyntaxKind::SetItem);
}

fn parse_set_clause(p: &mut Parser) {
    p.expect(TokenKind::SetKw);
    parse_set_item(p);
    while p.at(TokenKind::Comma) {
        p.bump();
        parse_set_item(p);
    }
}

fn parse_limit_clause(p: &mut Parser) {
    p.expect(TokenKind::LimitKw);
    parse_expr(p);
    if p.at(TokenKind::OffsetKw) {
        p.bump();
        parse_expr(p);
    }
}

/// `|> as t` renames the whole row, so it takes an identifier rather than the
/// `expr as name` an item alias would.
fn parse_alias_clause(p: &mut Parser) {
    p.expect(TokenKind::AsKw);
    parse_ident(p);
}

fn parse_aggregate_item(p: &mut Parser) {
    let m = p.start();
    parse_expr(p);
    if p.at(TokenKind::AsKw) {
        p.bump();
        parse_ident(p);
    }
    p.complete(m, SyntaxKind::AggregateItem);
}

fn parse_group_by_item(p: &mut Parser) {
    let m = p.start();
    parse_ident(p);
    if p.at(TokenKind::Dot) {
        p.bump();
        parse_ident(p);
    }
    if p.at(TokenKind::AsKw) {
        p.bump();
        parse_ident(p);
    }
    p.complete(m, SyntaxKind::GroupByItem);
}

fn parse_group_by(p: &mut Parser) {
    let m = p.start();
    p.expect(TokenKind::GroupKw);
    p.expect(TokenKind::ByKw);
    parse_group_by_item(p);
    while p.at(TokenKind::Comma) {
        p.bump();
        parse_group_by_item(p);
    }
    p.complete(m, SyntaxKind::GroupBy);
}

fn parse_aggregate_clause(p: &mut Parser) {
    p.expect(TokenKind::AggregateKw);
    parse_aggregate_item(p);
    while p.at(TokenKind::Comma) {
        p.bump();
        parse_aggregate_item(p);
    }
    if p.at(TokenKind::GroupKw) {
        parse_group_by(p);
    }
}

const JOIN_TYPES: [TokenKind; 4] = [
    TokenKind::InnerKw,
    TokenKind::LeftKw,
    TokenKind::RightKw,
    TokenKind::FullKw,
];

fn at_join_clause(p: &mut Parser) -> bool {
    p.at(TokenKind::JoinKw) || at_join_type(p)
}

fn at_join_type(p: &mut Parser) -> bool {
    JOIN_TYPES.iter().any(|&kind| p.at(kind))
}

fn parse_join_clause(p: &mut Parser) {
    if at_join_type(p) {
        p.bump();
    }
    p.expect(TokenKind::JoinKw);
    parse_ident(p);

    if p.at(TokenKind::AsKw) {
        p.bump();
        parse_ident(p);
    } else if p.at(TokenKind::Identifier) {
        parse_ident(p);
    }

    if p.at(TokenKind::UsingKw) {
        parse_join_using(p);
    } else {
        parse_join_on(p);
    }
}

fn parse_join_on(p: &mut Parser) {
    let m = p.start();
    p.expect(TokenKind::OnKw);
    parse_expr(p);
    p.complete(m, SyntaxKind::JoinOn);
}

fn parse_join_using(p: &mut Parser) {
    let m = p.start();
    p.expect(TokenKind::UsingKw);
    p.expect(TokenKind::LeftParen);
    // An empty list is left to lowering to report; parsing an identifier here
    // would consume the `)` into an error node and name a column after it.
    if !p.at(TokenKind::RightParen) {
        parse_ident(p);
        while p.at(TokenKind::Comma) {
            p.bump();
            parse_ident(p);
        }
    }
    p.expect(TokenKind::RightParen);
    p.complete(m, SyntaxKind::JoinUsing);
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use super::*;
    use crate::grammar::test_support;

    #[test]
    fn parse_from_source_directly() {
        test_support::check(
            "from employees",
            parse_from_source,
            expect![[r#"
            FromSource@0..14
              FromKw@0..4 "from"
              Space@4..5 " "
              Ident@5..14
                Identifier@5..14 "employees"
        "#]],
        );
    }

    #[test]
    fn parse_from_source_with_alias() {
        test_support::check(
            "from employees as e",
            parse_from_source,
            expect![[r#"
                FromSource@0..19
                  FromKw@0..4 "from"
                  Space@4..5 " "
                  Ident@5..14
                    Identifier@5..14 "employees"
                  Space@14..15 " "
                  AsKw@15..17 "as"
                  Space@17..18 " "
                  Ident@18..19
                    Identifier@18..19 "e"
            "#]],
        );
    }

    #[test]
    fn parse_select_item_directly() {
        test_support::check(
            "name as n",
            parse_select_item,
            expect![[r#"
                SelectItem@0..9
                  IdentExpr@0..4
                    Ident@0..4
                      Identifier@0..4 "name"
                  Space@4..5 " "
                  AsKw@5..7 "as"
                  Space@7..8 " "
                  Ident@8..9
                    Identifier@8..9 "n"
            "#]],
        );
    }

    #[test]
    fn parse_rename_item_directly() {
        test_support::check(
            "old as new",
            parse_rename_item,
            expect![[r#"
                RenameItem@0..10
                  Ident@0..3
                    Identifier@0..3 "old"
                  Space@3..4 " "
                  AsKw@4..6 "as"
                  Space@6..7 " "
                  Ident@7..10
                    Identifier@7..10 "new"
            "#]],
        );
    }

    #[test]
    fn parse_query_select_clause() {
        test_support::check(
            "from t |> select a, b",
            parse_query,
            expect![[r#"
                Pipeline@0..21
                  FromSource@0..6
                    FromKw@0..4 "from"
                    Space@4..5 " "
                    Ident@5..6
                      Identifier@5..6 "t"
                  Space@6..7 " "
                  SelectStage@7..21
                    Pipe@7..9 "|>"
                    Space@9..10 " "
                    SelectKw@10..16 "select"
                    Space@16..17 " "
                    SelectItem@17..18
                      IdentExpr@17..18
                        Ident@17..18
                          Identifier@17..18 "a"
                    Comma@18..19 ","
                    Space@19..20 " "
                    SelectItem@20..21
                      IdentExpr@20..21
                        Ident@20..21
                          Identifier@20..21 "b"
            "#]],
        );
    }

    #[test]
    fn parse_query_where_clause() {
        test_support::check(
            "from t |> where active",
            parse_query,
            expect![[r#"
                Pipeline@0..22
                  FromSource@0..6
                    FromKw@0..4 "from"
                    Space@4..5 " "
                    Ident@5..6
                      Identifier@5..6 "t"
                  Space@6..7 " "
                  WhereStage@7..22
                    Pipe@7..9 "|>"
                    Space@9..10 " "
                    WhereKw@10..15 "where"
                    Space@15..16 " "
                    IdentExpr@16..22
                      Ident@16..22
                        Identifier@16..22 "active"
            "#]],
        );
    }

    #[test]
    fn parse_query_distinct_clause() {
        test_support::check(
            "from t |> distinct",
            parse_query,
            expect![[r#"
                Pipeline@0..18
                  FromSource@0..6
                    FromKw@0..4 "from"
                    Space@4..5 " "
                    Ident@5..6
                      Identifier@5..6 "t"
                  Space@6..7 " "
                  DistinctStage@7..18
                    Pipe@7..9 "|>"
                    Space@9..10 " "
                    DistinctKw@10..18 "distinct"
            "#]],
        );
    }

    #[test]
    fn parse_query_drop_clause() {
        test_support::check(
            "from t |> drop a, b",
            parse_query,
            expect![[r#"
                Pipeline@0..19
                  FromSource@0..6
                    FromKw@0..4 "from"
                    Space@4..5 " "
                    Ident@5..6
                      Identifier@5..6 "t"
                  Space@6..7 " "
                  DropStage@7..19
                    Pipe@7..9 "|>"
                    Space@9..10 " "
                    DropKw@10..14 "drop"
                    Space@14..15 " "
                    Ident@15..16
                      Identifier@15..16 "a"
                    Comma@16..17 ","
                    Space@17..18 " "
                    Ident@18..19
                      Identifier@18..19 "b"
            "#]],
        );
    }

    #[test]
    fn parse_query_rename_clause() {
        test_support::check(
            "from t |> rename a as b",
            parse_query,
            expect![[r#"
                Pipeline@0..23
                  FromSource@0..6
                    FromKw@0..4 "from"
                    Space@4..5 " "
                    Ident@5..6
                      Identifier@5..6 "t"
                  Space@6..7 " "
                  RenameStage@7..23
                    Pipe@7..9 "|>"
                    Space@9..10 " "
                    RenameKw@10..16 "rename"
                    Space@16..17 " "
                    RenameItem@17..23
                      Ident@17..18
                        Identifier@17..18 "a"
                      Space@18..19 " "
                      AsKw@19..21 "as"
                      Space@21..22 " "
                      Ident@22..23
                        Identifier@22..23 "b"
            "#]],
        );
    }

    #[test]
    fn parse_query_extend_clause() {
        test_support::check(
            "from t |> extend a",
            parse_query,
            expect![[r#"
                Pipeline@0..18
                  FromSource@0..6
                    FromKw@0..4 "from"
                    Space@4..5 " "
                    Ident@5..6
                      Identifier@5..6 "t"
                  Space@6..7 " "
                  ExtendStage@7..18
                    Pipe@7..9 "|>"
                    Space@9..10 " "
                    ExtendKw@10..16 "extend"
                    Space@16..17 " "
                    SelectItem@17..18
                      IdentExpr@17..18
                        Ident@17..18
                          Identifier@17..18 "a"
            "#]],
        );
    }

    #[test]
    fn parse_rename_item_qualified() {
        test_support::check(
            "e.id as eid",
            parse_rename_item,
            expect![[r#"
            RenameItem@0..11
              Ident@0..1
                Identifier@0..1 "e"
              Dot@1..2 "."
              Ident@2..4
                Identifier@2..4 "id"
              Space@4..5 " "
              AsKw@5..7 "as"
              Space@7..8 " "
              Ident@8..11
                Identifier@8..11 "eid"
        "#]],
        );
    }

    #[test]
    fn parse_query_set_clause() {
        test_support::check(
            "from t |> set a = 1",
            parse_query,
            expect![[r#"
                Pipeline@0..19
                  FromSource@0..6
                    FromKw@0..4 "from"
                    Space@4..5 " "
                    Ident@5..6
                      Identifier@5..6 "t"
                  Space@6..7 " "
                  SetStage@7..19
                    Pipe@7..9 "|>"
                    Space@9..10 " "
                    SetKw@10..13 "set"
                    Space@13..14 " "
                    SetItem@14..19
                      Ident@14..15
                        Identifier@14..15 "a"
                      Space@15..16 " "
                      Eq@16..17 "="
                      Space@17..18 " "
                      IntLiteral@18..19
                        IntLit@18..19 "1"
            "#]],
        );
    }

    #[test]
    fn parse_query_aggregate_clause() {
        test_support::check(
            "from t |> aggregate sum(a) as s group by b",
            parse_query,
            expect![[r#"
                Pipeline@0..42
                  FromSource@0..6
                    FromKw@0..4 "from"
                    Space@4..5 " "
                    Ident@5..6
                      Identifier@5..6 "t"
                  Space@6..7 " "
                  AggregateStage@7..42
                    Pipe@7..9 "|>"
                    Space@9..10 " "
                    AggregateKw@10..19 "aggregate"
                    Space@19..20 " "
                    AggregateItem@20..31
                      CallExpr@20..26
                        IdentExpr@20..23
                          Ident@20..23
                            Identifier@20..23 "sum"
                        ArgList@23..26
                          LeftParen@23..24 "("
                          IdentExpr@24..25
                            Ident@24..25
                              Identifier@24..25 "a"
                          RightParen@25..26 ")"
                      Space@26..27 " "
                      AsKw@27..29 "as"
                      Space@29..30 " "
                      Ident@30..31
                        Identifier@30..31 "s"
                    Space@31..32 " "
                    GroupBy@32..42
                      GroupKw@32..37 "group"
                      Space@37..38 " "
                      ByKw@38..40 "by"
                      Space@40..41 " "
                      GroupByItem@41..42
                        Ident@41..42
                          Identifier@41..42 "b"
            "#]],
        );
    }

    #[test]
    fn parse_query_aggregate_without_group_by() {
        test_support::check(
            "from t |> aggregate count()",
            parse_query,
            expect![[r#"
                Pipeline@0..27
                  FromSource@0..6
                    FromKw@0..4 "from"
                    Space@4..5 " "
                    Ident@5..6
                      Identifier@5..6 "t"
                  Space@6..7 " "
                  AggregateStage@7..27
                    Pipe@7..9 "|>"
                    Space@9..10 " "
                    AggregateKw@10..19 "aggregate"
                    Space@19..20 " "
                    AggregateItem@20..27
                      CallExpr@20..27
                        IdentExpr@20..25
                          Ident@20..25
                            Identifier@20..25 "count"
                        ArgList@25..27
                          LeftParen@25..26 "("
                          RightParen@26..27 ")"
            "#]],
        );
    }

    #[test]
    fn parse_query_limit_clause() {
        test_support::check(
            "from t |> limit 2 offset 1",
            parse_query,
            expect![[r#"
                Pipeline@0..26
                  FromSource@0..6
                    FromKw@0..4 "from"
                    Space@4..5 " "
                    Ident@5..6
                      Identifier@5..6 "t"
                  Space@6..7 " "
                  LimitStage@7..26
                    Pipe@7..9 "|>"
                    Space@9..10 " "
                    LimitKw@10..15 "limit"
                    Space@15..16 " "
                    IntLiteral@16..17
                      IntLit@16..17 "2"
                    Space@17..18 " "
                    OffsetKw@18..24 "offset"
                    Space@24..25 " "
                    IntLiteral@25..26
                      IntLit@25..26 "1"
            "#]],
        );
    }

    #[test]
    fn parse_query_alias_clause() {
        test_support::check(
            "from t |> as u",
            parse_query,
            expect![[r#"
                Pipeline@0..14
                  FromSource@0..6
                    FromKw@0..4 "from"
                    Space@4..5 " "
                    Ident@5..6
                      Identifier@5..6 "t"
                  Space@6..7 " "
                  AliasStage@7..14
                    Pipe@7..9 "|>"
                    Space@9..10 " "
                    AsKw@10..12 "as"
                    Space@12..13 " "
                    Ident@13..14
                      Identifier@13..14 "u"
            "#]],
        );
    }

    #[test]
    fn parse_query_join_on_clause() {
        test_support::check(
            "from t |> join u as d on a == d.b",
            parse_query,
            expect![[r#"
                Pipeline@0..33
                  FromSource@0..6
                    FromKw@0..4 "from"
                    Space@4..5 " "
                    Ident@5..6
                      Identifier@5..6 "t"
                  Space@6..7 " "
                  JoinStage@7..33
                    Pipe@7..9 "|>"
                    Space@9..10 " "
                    JoinKw@10..14 "join"
                    Space@14..15 " "
                    Ident@15..16
                      Identifier@15..16 "u"
                    Space@16..17 " "
                    AsKw@17..19 "as"
                    Space@19..20 " "
                    Ident@20..21
                      Identifier@20..21 "d"
                    Space@21..22 " "
                    JoinOn@22..33
                      OnKw@22..24 "on"
                      Space@24..25 " "
                      BinaryExpr@25..33
                        IdentExpr@25..26
                          Ident@25..26
                            Identifier@25..26 "a"
                        Space@26..27 " "
                        EqEq@27..29 "=="
                        Space@29..30 " "
                        FieldAccessExpr@30..33
                          IdentExpr@30..31
                            Ident@30..31
                              Identifier@30..31 "d"
                          Dot@31..32 "."
                          Ident@32..33
                            Identifier@32..33 "b"
            "#]],
        );
    }

    #[test]
    fn parse_query_join_using_clause() {
        test_support::check(
            "from t |> left join u using (a, b)",
            parse_query,
            expect![[r#"
                Pipeline@0..34
                  FromSource@0..6
                    FromKw@0..4 "from"
                    Space@4..5 " "
                    Ident@5..6
                      Identifier@5..6 "t"
                  Space@6..7 " "
                  JoinStage@7..34
                    Pipe@7..9 "|>"
                    Space@9..10 " "
                    LeftKw@10..14 "left"
                    Space@14..15 " "
                    JoinKw@15..19 "join"
                    Space@19..20 " "
                    Ident@20..21
                      Identifier@20..21 "u"
                    Space@21..22 " "
                    JoinUsing@22..34
                      UsingKw@22..27 "using"
                      Space@27..28 " "
                      LeftParen@28..29 "("
                      Ident@29..30
                        Identifier@29..30 "a"
                      Comma@30..31 ","
                      Space@31..32 " "
                      Ident@32..33
                        Identifier@32..33 "b"
                      RightParen@33..34 ")"
            "#]],
        );
    }

    #[test]
    fn parse_query_join_bare_alias() {
        test_support::check(
            "from t |> full join u d on a == d.b",
            parse_query,
            expect![[r#"
                Pipeline@0..35
                  FromSource@0..6
                    FromKw@0..4 "from"
                    Space@4..5 " "
                    Ident@5..6
                      Identifier@5..6 "t"
                  Space@6..7 " "
                  JoinStage@7..35
                    Pipe@7..9 "|>"
                    Space@9..10 " "
                    FullKw@10..14 "full"
                    Space@14..15 " "
                    JoinKw@15..19 "join"
                    Space@19..20 " "
                    Ident@20..21
                      Identifier@20..21 "u"
                    Space@21..22 " "
                    Ident@22..23
                      Identifier@22..23 "d"
                    Space@23..24 " "
                    JoinOn@24..35
                      OnKw@24..26 "on"
                      Space@26..27 " "
                      BinaryExpr@27..35
                        IdentExpr@27..28
                          Ident@27..28
                            Identifier@27..28 "a"
                        Space@28..29 " "
                        EqEq@29..31 "=="
                        Space@31..32 " "
                        FieldAccessExpr@32..35
                          IdentExpr@32..33
                            Ident@32..33
                              Identifier@32..33 "d"
                          Dot@33..34 "."
                          Ident@34..35
                            Identifier@34..35 "b"
            "#]],
        );
    }

    #[test]
    fn parse_query_chained() {
        test_support::check(
            "from t |> where a |> select b",
            parse_query,
            expect![[r#"
                Pipeline@0..29
                  FromSource@0..6
                    FromKw@0..4 "from"
                    Space@4..5 " "
                    Ident@5..6
                      Identifier@5..6 "t"
                  Space@6..7 " "
                  WhereStage@7..17
                    Pipe@7..9 "|>"
                    Space@9..10 " "
                    WhereKw@10..15 "where"
                    Space@15..16 " "
                    IdentExpr@16..17
                      Ident@16..17
                        Identifier@16..17 "a"
                  Space@17..18 " "
                  SelectStage@18..29
                    Pipe@18..20 "|>"
                    Space@20..21 " "
                    SelectKw@21..27 "select"
                    Space@27..28 " "
                    SelectItem@28..29
                      IdentExpr@28..29
                        Ident@28..29
                          Identifier@28..29 "b"
            "#]],
        );
    }
}
