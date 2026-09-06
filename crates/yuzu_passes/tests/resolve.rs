//! ResolveNames, exercised from Yuzu source through the converter: answers
//! land as attributes, failures land in the diagnostics engine.

use expect_test::{Expect, expect};
use melior::ir::operation::OperationLike;
use yuzu_diagnostics::diagnostics::engine::DiagnosticsEngine;
use yuzu_diagnostics::diagnostics::printer::DiagnosticPrinter;
use yuzu_diagnostics::source_map::SourceMap;
use yuzu_mlir::DiagnosticsBridge;
use yuzu_passes::resolve_names;

fn check(source: &str, expected: Expect) {
    let context = yuzu_mlir::context();
    let conversion =
        yuzu_lang::convert_source(&context, "test.yz", source).expect("the source converts");
    assert!(
        conversion.unsupported.is_empty(),
        "unsupported constructs: {:?}",
        conversion.unsupported
    );

    let mut sources = SourceMap::new();
    let source_id = sources.add("test.yz".to_string(), source.to_string());
    let mut diagnostics = DiagnosticsEngine::new();
    resolve_names(
        &conversion.module,
        &yuzu_types::Builtins,
        &DiagnosticsBridge::new(source_id, source),
        &mut diagnostics,
    );

    let printer = DiagnosticPrinter::new(&sources);
    let rendered: Vec<String> = diagnostics
        .diagnostics()
        .iter()
        .map(|diagnostic| printer.print(diagnostic))
        .collect();
    let output = if rendered.is_empty() {
        conversion.module.as_operation().to_string()
    } else {
        rendered.join("\n")
    };
    expected.assert_eq(&output);
}

#[test]
fn resolves_columns_params_and_calls() {
    check(
        r#"
struct Row { a: int64, b: int64 }
table t = Row

fn f(x: int64) -> int64 { return x * 3 }

from t
|> where a > 10
|> extend f(a) + b as e
|> aggregate sum(e) as s group by b
"#,
        expect![[r#"
            module {
              yzl.struct @Row !yzr.rel<a: !yz.int64, b: !yz.int64>
              yzl.table @t of @Row
              yzl.fn @f params ["x"] (!yz.int64) -> !yz.int64 {
                %4 = yzl.name "x" : !yzl.var {param = 0 : i64}
                %5 = yz.constant_int 3
                %6 = yz.mul %4, %5 : !yzl.var, !yz.int64 -> !yzl.var
                yzl.return %6 : !yzl.var
              }
              %0 = yzl.from @t
              %1 = yzl.where %0 {
                %4 = yzl.name "a" : !yzl.var {col = 0 : i64}
                %5 = yz.constant_int 10
                %6 = yz.cmp "gt", %4, %5 : !yzl.var, !yz.int64 -> !yzl.var
                yzl.yield %6 : !yzl.var
              }
              %2 = yzl.extend %1 as ["e"] {
                %4 = yzl.name "a" : !yzl.var {col = 0 : i64}
                %5 = yzl.call @f(%4) : (!yzl.var) -> !yzl.var {callee_kind = "fn"}
                %6 = yzl.name "b" : !yzl.var {col = 1 : i64}
                %7 = yz.add %5, %6 : !yzl.var, !yzl.var -> !yzl.var
                yzl.yield %7 : !yzl.var
              }
              %3 = yzl.aggregate %2 group_by ["b"] as ["s"] {
                %4 = yzl.name "e" : !yzl.var {col = 2 : i64}
                %5 = yzl.call @sum(%4) : (!yzl.var) -> !yzl.var {callee_kind = "builtin"}
                yzl.yield %5 : !yzl.var
              } {key_cols = [1]}
              yzl.output %3
            }
        "#]],
    );
}

#[test]
fn resolves_aliases_joins_and_bindings() {
    check(
        r#"
struct Row { id: int64, dept_id: int64 }
table t = Row
struct Dept { id: int64 }
table depts = Dept

let base = from t |> where id > 0

from base
|> inner join depts as d on dept_id == d.id
|> select d.id as out
"#,
        expect![[r#"
            module {
              yzl.struct @Row !yzr.rel<id: !yz.int64, dept_id: !yz.int64>
              yzl.table @t of @Row
              yzl.struct @Dept !yzr.rel<id: !yz.int64>
              yzl.table @depts of @Dept
              yzl.let @base {
                %3 = yzl.from @t
                %4 = yzl.where %3 {
                  %5 = yzl.name "id" : !yzl.var {col = 0 : i64}
                  %6 = yz.constant_int 0
                  %7 = yz.cmp "gt", %5, %6 : !yzl.var, !yz.int64 -> !yzl.var
                  yzl.yield %7 : !yzl.var
                }
                yzl.yield %4 : !yzl.query
              }
              %0 = yzl.from @base
              %1 = yzl.join "inner", %0, @depts as "d" {
                %3 = yzl.name "dept_id" : !yzl.var {col = 1 : i64}
                %4 = yzl.name "d.id" : !yzl.var {col = 2 : i64}
                %5 = yz.cmp "eq", %3, %4 : !yzl.var, !yzl.var -> !yzl.var
                yzl.yield %5 : !yzl.var
              }
              %2 = yzl.select %1 as ["out"] {
                %3 = yzl.name "d.id" : !yzl.var {col = 2 : i64}
                yzl.yield %3 : !yzl.var
              }
              yzl.output %2
            }
        "#]],
    );
}

#[test]
fn reports_unknown_names() {
    check(
        r#"
struct Row { a: int64 }
table t = Row

fn f(x: int64) -> int64 { return y }

from missing
|> where nope > 1
|> set ghost = 2
|> extend wrong(a) as e
"#,
        expect![[r#"
            error: unknown name `y`
             --> test.yz:5:34
              |
            5 | fn f(x: int64) -> int64 { return y }
              |                                  ^

            error: unknown relation `missing`
             --> test.yz:7:1
              |
            7 | from missing
              | ^

            error: unknown column `nope`
             --> test.yz:8:10
              |
            8 | |> where nope > 1
              |          ^

            error: unknown column `ghost`
             --> test.yz:7:1
              |
            7 | from missing
              | ^

            error: unknown column `a`
             --> test.yz:10:17
               |
            10 | |> extend wrong(a) as e
               |                 ^

            error: unknown function `wrong`
             --> test.yz:10:11
               |
            10 | |> extend wrong(a) as e
               |           ^
        "#]],
    );
}

#[test]
fn reports_arity_and_duplicates() {
    check(
        r#"
struct Row { a: int64 }
table t = Row
table t = Row

fn f(x: int64) -> int64 { return x }

from t
|> extend f(a, a) as two, sum() as none as e
"#,
        expect![[r#"
            error: the relation `t` is already defined
             --> test.yz:4:1
              |
            4 | table t = Row
              | ^

            error: `f` expects 1 arguments, got 2
             --> test.yz:9:11
              |
            9 | |> extend f(a, a) as two, sum() as none as e
              |           ^

            error: `sum` expects 1 arguments, got 0
             --> test.yz:9:27
              |
            9 | |> extend f(a, a) as two, sum() as none as e
              |                           ^

            error: unknown name `e`
             --> test.yz:9:44
              |
            9 | |> extend f(a, a) as two, sum() as none as e
              |                                            ^
        "#]],
    );
}

#[test]
fn reports_an_ambiguous_join_column() {
    check(
        r#"
struct Row { id: int64 }
table l = Row
table r = Row

from l
|> inner join r on id == 1
"#,
        expect![[r#"
            error: `id` is ambiguous
             --> test.yz:7:20
              |
            7 | |> inner join r on id == 1
              |                    ^
        "#]],
    );
}
