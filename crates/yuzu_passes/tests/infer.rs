//! InferTypes, exercised from Yuzu source through conversion and resolution:
//! solved types land as `{ty = …}` attributes, conflicts land as diagnostics.

use expect_test::{Expect, expect};
use yuzu_diagnostics::diagnostics::engine::DiagnosticsEngine;
use yuzu_diagnostics::diagnostics::printer::DiagnosticPrinter;
use yuzu_diagnostics::source_map::SourceMap;
use yuzu_mlir::DiagnosticsBridge;
use yuzu_passes::{infer_types, resolve_names};

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
    let bridge = DiagnosticsBridge::new(source_id, source);
    let mut diagnostics = DiagnosticsEngine::new();
    resolve_names(
        &context,
        &conversion.module,
        &yuzu_types::Builtins,
        &bridge,
        &mut diagnostics,
    );
    assert!(
        diagnostics.diagnostics().is_empty(),
        "resolution failed before inference"
    );
    infer_types(&context, &conversion.module, &bridge, &mut diagnostics);

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
fn infers_columns_calls_and_measures() {
    check(
        r#"
struct Row { a: int64, b: int64, rating: float64 }
table t = Row

fn f(x: int64) -> int64 { return x * 3 }

from t
|> where a > 10
|> extend f(a) + b as e
|> aggregate sum(e) as s, avg(rating) as r group by b
"#,
        expect![[r#"
            module {
              yzl.struct @Row !yzr.rel<a: !yz.int64, b: !yz.int64, rating: !yz.float64>
              yzl.table @t of @Row
              yzl.fn @f params ["x"] (!yz.int64) -> !yz.int64 {
                %4 = yzl.name "x" : !yzl.var {param = 0 : i64, ty = !yz.int64}
                %5 = yz.constant_int 3
                %6 = yz.mul %4, %5 : !yzl.var, !yz.int64 -> !yzl.var {ty = !yz.int64}
                yzl.return %6 : !yzl.var
              }
              %0 = yzl.from @t
              %1 = yzl.where %0 {
                %4 = yzl.name "a" : !yzl.var {col = 0 : i64, ty = !yz.int64}
                %5 = yz.constant_int 10
                %6 = yz.cmp "gt", %4, %5 : !yzl.var, !yz.int64 -> !yzl.var {ty = !yz.bool}
                yzl.yield %6 : !yzl.var
              }
              %2 = yzl.extend %1 as ["e"] {
                %4 = yzl.name "a" : !yzl.var {col = 0 : i64, ty = !yz.int64}
                %5 = yzl.call @f(%4) : (!yzl.var) -> !yzl.var {callee_kind = "fn", ty = !yz.int64}
                %6 = yzl.name "b" : !yzl.var {col = 1 : i64, ty = !yz.int64}
                %7 = yz.add %5, %6 : !yzl.var, !yzl.var -> !yzl.var {ty = !yz.int64}
                yzl.yield %7 : !yzl.var
              }
              %3 = yzl.aggregate %2 group_by ["b"] as ["s", "r"] {
                %4 = yzl.name "e" : !yzl.var {col = 3 : i64, ty = !yz.int64}
                %5 = yzl.call @sum(%4) : (!yzl.var) -> !yzl.var {callee_kind = "builtin", ty = !yz.int64}
                %6 = yzl.name "rating" : !yzl.var {col = 2 : i64, ty = !yz.float64}
                %7 = yzl.call @avg(%6) : (!yzl.var) -> !yzl.var {callee_kind = "builtin", ty = !yz.float64}
                yzl.yield %5, %7 : !yzl.var, !yzl.var
              } {key_cols = [1]}
              yzl.output %3
            }
        "#]],
    );
}

#[test]
fn reports_a_comparison_mismatch() {
    check(
        r#"
struct Row { name: str }
table t = Row

from t
|> where name == 1
"#,
        expect![[r#"
            error: expected `!yz.str`, found `!yz.int64`
             --> test.yz:6:10
              |
            6 | |> where name == 1
              |          ^
        "#]],
    );
}

#[test]
fn reports_a_return_mismatch() {
    check(
        r#"
struct Row { a: int64 }
table t = Row

fn f(x: int64) -> bool { return x }

from t
|> extend f(a) as e
"#,
        expect![[r#"
            error: expected `!yz.int64`, found `!yz.bool`
             --> test.yz:5:26
              |
            5 | fn f(x: int64) -> bool { return x }
              |                          ^
        "#]],
    );
}

#[test]
fn reports_a_set_mismatch() {
    check(
        r#"
struct Row { level: int64 }
table t = Row

from t
|> set level = "high"
"#,
        expect![[r#"
            error: expected `!yz.int64`, found `!yz.str`
             --> test.yz:5:1
              |
            5 | from t
              | ^
        "#]],
    );
}
