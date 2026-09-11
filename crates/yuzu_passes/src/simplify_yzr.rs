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
}
