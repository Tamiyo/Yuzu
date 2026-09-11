//! SimplifyYZR: the work a query does not need to carry. The `yz` ops declare
//! their own folders, so this is MLIR's canonicalizer and CSE rather than a
//! rewrite of our own — constants folded, repeated work shared, and anything
//! nothing reads dropped.
//!
//! Stage regions are `IsolatedFromAbove`, so what a fold leaves behind stays
//! inside the region that becomes a Substrait expression. Without that, the
//! canonicalizer hoists constants to the module and the emitter has to chase
//! values across a boundary Substrait cannot express.

use melior::Context;
use melior::ir::Module;
use melior::ir::operation::OperationLike;
use melior::pass::{PassManager, transform};

/// Expects a lowered yzr module. Diagnostics go through MLIR — run this
/// inside `yuzu_mlir::diagnostics::capture` to collect them.
pub fn simplify_yzr(context: &Context, module: &mut Module) {
    let passes = PassManager::new(context);
    passes.add_pass(transform::create_canonicalizer());
    passes.add_pass(transform::create_cse());

    if let Err(error) = passes.run(module) {
        // MLIR reports why through the handler the capture installed; this
        // says that it happened at all, so a failure is never silent.
        yuzu_mlir::diagnostics::emit_error(
            module.as_operation().location(),
            &format!("simplifying the query failed: {error}"),
        );
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::test_support::check_simplified;

    /// The folders live on the `yz` ops, so constant arithmetic collapses
    /// without a rewrite of our own — and `1 / 0` stands, because folding it
    /// would decide at compile time what the query is entitled to fail on.
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

    /// Two identical computations become one: CSE reaches inside the stage
    /// region, where the repeated work actually is.
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

    /// The limit of a generic DCE, pinned so the gap is visible: MLIR drops
    /// dead *operations*, and the `yzr.extend` here is live — the projection
    /// downstream reads it. That nothing reads its third column is a fact
    /// about rows, not about SSA, so `unused` is still computed. Removing it
    /// is column pruning, and that stays ours to write.
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

    /// Integers compare as integers. These two differ by one and share a
    /// double, so folding through one would answer `false` and silently
    /// return no rows.
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

    /// A sum with no representable answer declines to fold: what overflow
    /// does is the engine's to say, and a wrong constant would not even fail.
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

    /// The range is asymmetric, so the least integer has no negation — and
    /// declining there leaves the division downstream nothing to fold
    /// either, which is the whole expression left to the engine.
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

    /// Arithmetic that does fit still folds, so declining costs nothing that
    /// was ever safe to take.
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

    /// A relation nothing reads costs nothing: the stage ops are `Pure`, so
    /// a binding the output never reaches is dropped whole. The struct the
    /// dead stages declared outlives them — a symbol is not an operation,
    /// and nothing yet collects the ones no type names.
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
                  yz.struct @row ["a", "b", "c"] : [!yz.int64, !yz.int64, !yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  yz.struct @row_0 ["x"] : [!yz.int64]
                  %1 = yzr.project %0 {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64):
                    yzr.yield %arg0 : !yz.int64
                  } : !yz.struct<@Row> -> !yz.struct<@row_0>
                  yzr.output %1 : !yz.struct<@row_0>
                }
            "#]],
        );
    }

    /// A relation can be unused for its values and still decide the answer.
    /// No column of `u` is read here, but the join says which rows exist and
    /// how many: an inner join drops left rows that match nothing, and
    /// multiplies them when the key repeats. So the join stays, and with it
    /// both sides.
    ///
    /// Dropping it would need the right side to be known unique on the key,
    /// and nothing declares keys — a guard rail for column pruning, which
    /// will see these columns go unread and must not conclude from that
    /// alone that the relation is dead.
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
