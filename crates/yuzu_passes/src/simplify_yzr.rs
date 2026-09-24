//! MLIR's canonicalizer and CSE; the `yz` ops declare their own folders.
//! Stage regions are `IsolatedFromAbove`, so what a fold leaves behind stays
//! inside the region that becomes a Substrait expression.

use melior::Context;
use melior::ir::Module;
use melior::pass::{PassManager, transform};

pub fn simplify_yzr(context: &Context, module: &mut Module) {
    let passes = PassManager::new(context);
    passes.add_pass(transform::create_canonicalizer());
    passes.add_pass(transform::create_cse());

    passes
        .run(module)
        .expect("canonicalization runs on any module the lowering builds");
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::test_support::check_simplified;

    #[test]
    fn constant_arithmetic_folds_in_place() {
        check_simplified(
            r#"
struct Row { a: int64 }
table t = Row

from t
|> where a > 2 * 3 + 1
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a"] : [!yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  %1 = yzr.filter %0 : !yz.struct<@Row> {
                  ^bb0(%arg0: !yz.int64):
                    %2 = yz.constant_int 7
                    %3 = yz.cmp "gt", %arg0, %2 : !yz.int64, !yz.int64 -> !yz.bool
                    yzr.yield %3 : !yz.bool
                  }
                  yzr.output %1 : !yz.struct<@Row>
                }
            "#]],
        );
    }

    #[test]
    fn repeated_work_is_computed_once() {
        check_simplified(
            r#"
struct Row { a: int64, b: int64 }
table t = Row

from t
|> extend a + b as x, a + b as y
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a", "b"] : [!yz.int64, !yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  yz.struct @row ["a", "b", "x", "y"] : [!yz.int64, !yz.int64, !yz.int64, !yz.int64]
                  %1 = yzr.extend %0 {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64):
                    %2 = yz.add %arg0, %arg1 : !yz.int64, !yz.int64 -> !yz.int64
                    yzr.yield %2, %2 : !yz.int64, !yz.int64
                  } : !yz.struct<@Row> -> !yz.struct<@row>
                  yzr.output %1 : !yz.struct<@row>
                }
            "#]],
        );
    }

    /// MLIR drops dead operations; that nothing reads a column is a fact
    /// about rows, not SSA, so column pruning stays ours to write.
    #[test]
    fn an_unread_column_is_still_computed() {
        check_simplified(
            r#"
struct Row { a: int64, b: int64 }
table t = Row

from t
|> extend a * b as unused
|> select a as kept
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a", "b"] : [!yz.int64, !yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  yz.struct @row ["a", "b", "unused"] : [!yz.int64, !yz.int64, !yz.int64]
                  %1 = yzr.extend %0 {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64):
                    %3 = yz.mul %arg0, %arg1 : !yz.int64, !yz.int64 -> !yz.int64
                    yzr.yield %3 : !yz.int64
                  } : !yz.struct<@Row> -> !yz.struct<@row>
                  yz.struct @row_0 ["kept"] : [!yz.int64]
                  %2 = yzr.project %1 {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64, %arg2: !yz.int64):
                    yzr.yield %arg0 : !yz.int64
                  } : !yz.struct<@row> -> !yz.struct<@row_0>
                  yzr.output %2 : !yz.struct<@row_0>
                }
            "#]],
        );
    }

    /// These two differ by one and share a double, so folding through one
    /// would answer `false`.
    #[test]
    fn large_integers_compare_exactly() {
        check_simplified(
            r#"
struct Row { a: int64 }
table t = Row

from t
|> where 9007199254740993 > 9007199254740992
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a"] : [!yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  %1 = yzr.filter %0 : !yz.struct<@Row> {
                  ^bb0(%arg0: !yz.int64):
                    %2 = yz.constant_bool true
                    yzr.yield %2 : !yz.bool
                  }
                  yzr.output %1 : !yz.struct<@Row>
                }
            "#]],
        );
    }

    #[test]
    fn constants_reach_each_other_through_a_value() {
        check_simplified(
            r#"
struct Row { a: int64 }
table t = Row

def f(x: int64) -> int64 { return x + 1 + 2 }

from t
|> select f(a) as v
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a"] : [!yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  yz.struct @row ["v"] : [!yz.int64]
                  %1 = yzr.project %0 {
                  ^bb0(%arg0: !yz.int64):
                    %2 = yz.constant_int 3
                    %3 = yz.add %arg0, %2 : !yz.int64, !yz.int64 -> !yz.int64
                    yzr.yield %3 : !yz.int64
                  } : !yz.struct<@Row> -> !yz.struct<@row>
                  yzr.output %1 : !yz.struct<@row>
                }
            "#]],
        );
    }

    #[test]
    fn an_unrepresentable_sum_keeps_its_order() {
        check_simplified(
            r#"
struct Row { a: int64 }
table t = Row

def f(x: int64) -> int64 { return x + 9223372036854775807 + 1 }

from t
|> select f(a) as v
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a"] : [!yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  yz.struct @row ["v"] : [!yz.int64]
                  %1 = yzr.project %0 {
                  ^bb0(%arg0: !yz.int64):
                    %2 = yz.constant_int 1
                    %3 = yz.constant_int 9223372036854775807
                    %4 = yz.add %arg0, %3 : !yz.int64, !yz.int64 -> !yz.int64
                    %5 = yz.add %4, %2 : !yz.int64, !yz.int64 -> !yz.int64
                    yzr.yield %5 : !yz.int64
                  } : !yz.struct<@Row> -> !yz.struct<@row>
                  yzr.output %1 : !yz.struct<@row>
                }
            "#]],
        );
    }

    #[test]
    fn opposite_signs_keep_the_order_they_were_written_in() {
        check_simplified(
            r#"
struct Row { a: int64 }
table t = Row

def f(x: int64) -> int64 { return x + 1 + -1 }

from t
|> select f(a) as v
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a"] : [!yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  yz.struct @row ["v"] : [!yz.int64]
                  %1 = yzr.project %0 {
                  ^bb0(%arg0: !yz.int64):
                    %2 = yz.constant_int -1
                    %3 = yz.constant_int 1
                    %4 = yz.add %arg0, %3 : !yz.int64, !yz.int64 -> !yz.int64
                    %5 = yz.add %4, %2 : !yz.int64, !yz.int64 -> !yz.int64
                    yzr.yield %5 : !yz.int64
                  } : !yz.struct<@Row> -> !yz.struct<@row>
                  yzr.output %1 : !yz.struct<@row>
                }
            "#]],
        );
    }

    #[test]
    fn an_overflowing_sum_is_left_to_the_engine() {
        check_simplified(
            r#"
struct Row { a: int64 }
table t = Row

from t
|> where a > 9223372036854775807 + 1
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a"] : [!yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  %1 = yzr.filter %0 : !yz.struct<@Row> {
                  ^bb0(%arg0: !yz.int64):
                    %2 = yz.constant_int 9223372036854775807
                    %3 = yz.constant_int 1
                    %4 = yz.add %2, %3 : !yz.int64, !yz.int64 -> !yz.int64
                    %5 = yz.cmp "gt", %arg0, %4 : !yz.int64, !yz.int64 -> !yz.bool
                    yzr.yield %5 : !yz.bool
                  }
                  yzr.output %1 : !yz.struct<@Row>
                }
            "#]],
        );
    }

    #[test]
    fn the_least_integer_is_left_to_the_engine() {
        check_simplified(
            r#"
struct Row { a: int64 }
table t = Row

from t
|> where a > -9223372036854775808 / -1
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a"] : [!yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  %1 = yzr.filter %0 : !yz.struct<@Row> {
                  ^bb0(%arg0: !yz.int64):
                    %2 = yz.constant_int -1
                    %3 = yz.constant_int -9223372036854775808
                    %4 = yz.neg %3 : !yz.int64 -> !yz.int64
                    %5 = yz.div %4, %2 : !yz.int64, !yz.int64 -> !yz.int64
                    %6 = yz.cmp "gt", %arg0, %5 : !yz.int64, !yz.int64 -> !yz.bool
                    yzr.yield %6 : !yz.bool
                  }
                  yzr.output %1 : !yz.struct<@Row>
                }
            "#]],
        );
    }

    #[test]
    fn arithmetic_that_fits_still_folds() {
        check_simplified(
            r#"
struct Row { a: int64 }
table t = Row

from t
|> where a > 9223372036854775806 + 1
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a"] : [!yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  %1 = yzr.filter %0 : !yz.struct<@Row> {
                  ^bb0(%arg0: !yz.int64):
                    %2 = yz.constant_int 9223372036854775807
                    %3 = yz.cmp "gt", %arg0, %2 : !yz.int64, !yz.int64 -> !yz.bool
                    yzr.yield %3 : !yz.bool
                  }
                  yzr.output %1 : !yz.struct<@Row>
                }
            "#]],
        );
    }

    /// Symbol DCE takes the struct it declared along with it.
    #[test]
    fn a_relation_the_output_never_reads_is_dropped() {
        check_simplified(
            r#"
struct Row { a: int64, b: int64 }
table t = Row

let never_read = from t |> where a > 1 |> extend a * b as c

from t
|> select a as x
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a", "b"] : [!yz.int64, !yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  yz.struct @row ["x"] : [!yz.int64]
                  %1 = yzr.project %0 {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64):
                    yzr.yield %arg0 : !yz.int64
                  } : !yz.struct<@Row> -> !yz.struct<@row>
                  yzr.output %1 : !yz.struct<@row>
                }
            "#]],
        );
    }

    /// The join decides which rows exist, so column pruning must not take
    /// an unread side for a dead one.
    #[test]
    fn a_join_survives_when_nothing_reads_its_right_side() {
        check_simplified(
            r#"
struct Row { a: int64 }
table t = Row
struct Other { k: int64, extra: str }
table u = Other

from t
|> inner join u on a == k
|> select a as x
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a"] : [!yz.int64]
                  yz.struct @Other ["k", "extra"] : [!yz.int64, !yz.str]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  %1 = yzr.table @u : !yz.struct<@Other>
                  yz.struct @row ["a", "k", "extra"] : [!yz.int64, !yz.int64, !yz.str]
                  %2 = yzr.join "inner", %0, %1 {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64, %arg2: !yz.str):
                    %4 = yz.cmp "eq", %arg0, %arg1 : !yz.int64, !yz.int64 -> !yz.bool
                    yzr.yield %4 : !yz.bool
                  } : !yz.struct<@Row>, !yz.struct<@Other> -> !yz.struct<@row>
                  yz.struct @row_0 ["x"] : [!yz.int64]
                  %3 = yzr.project %2 {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64, %arg2: !yz.str):
                    yzr.yield %arg0 : !yz.int64
                  } : !yz.struct<@row> -> !yz.struct<@row_0>
                  yzr.output %3 : !yz.struct<@row_0>
                }
            "#]],
        );
    }
}
