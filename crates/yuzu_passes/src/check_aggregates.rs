//! CheckAggregates: the placement rules for aggregate functions, ported
//! from the old inference. An aggregate call lives only in an `aggregate`
//! item or an `agg fn` body, never in another aggregate's arguments, and an
//! `agg fn` must use an aggregate without calling itself.

use std::collections::{HashMap, HashSet};

use melior::ir::operation::{OperationLike, OperationRef, OperationResult};
use melior::ir::{BlockRef, Location, Module, RegionLike};
use yuzu_mlir::ext::{BlockExt, OperationCast, OperationExt, RegionExt, ValueExt};
use yuzu_mlir::ops::yzl::{CallOp, YzlOp};

/// Where the walk currently is, aggregate-wise.
#[derive(Clone, Copy, PartialEq)]
enum Grouping<'m> {
    /// Row context: aggregate calls may not appear.
    None,
    /// An `aggregate` item region.
    Item,
    /// An `agg fn` body, named to catch self-calls.
    FnBody(&'m str),
}

struct Checker<'c> {
    /// Each group-level value, with the aggregate calls it came from.
    group_values: HashMap<usize, Vec<usize>>,
    /// The location and callee of each aggregate call, by its result.
    aggregate_calls: HashMap<usize, (Location<'c>, String)>,
    /// Calls already reported as nested, so each reports once.
    nested: HashSet<usize>,
    /// Whether the current `agg fn` body used an aggregate.
    saw_aggregate: bool,
}

/// Expects a resolved module: callees are classified by their stamped
/// `callee_kind`. Diagnostics go through MLIR — run this inside
/// `yuzu_mlir::diagnostics::capture` to collect them.
pub fn check_aggregates(module: &Module) {
    let mut checker = Checker {
        group_values: HashMap::new(),
        aggregate_calls: HashMap::new(),
        nested: HashSet::new(),
        saw_aggregate: false,
    };

    checker.check_block(module.body(), Grouping::None);
}

impl<'c> Checker<'c> {
    fn check_block<'m>(&mut self, block: BlockRef<'c, 'm>, grouping: Grouping<'m>) {
        for op in block.operations() {
            match op.as_yzl() {
                Some(YzlOp::Call(call)) => {
                    let callee = call.callee().value();
                    if self.is_aggregate_call(&call) {
                        self.check_aggregate_call(op, callee, grouping);
                    } else {
                        self.propagate_group_values(op);
                    }
                }
                Some(YzlOp::Aggregate(stage)) => {
                    self.check_regions(stage.operation(), Grouping::Item);
                }
                Some(YzlOp::Fn(function)) => {
                    // An `external agg fn` has no body to aggregate in.
                    if function.agg() && !function.external() {
                        let name = function.sym_name().value();
                        let outer = std::mem::replace(&mut self.saw_aggregate, false);
                        self.check_regions(function.operation(), Grouping::FnBody(name));
                        if !self.saw_aggregate {
                            yuzu_mlir::diagnostics::emit_error(
                                Self::returned_location(function.operation())
                                    .unwrap_or_else(|| op.location()),
                                "an `agg fn` must use an aggregate function",
                            );
                        }

                        self.saw_aggregate = outer;
                    } else {
                        self.check_regions(function.operation(), Grouping::None);
                    }
                }
                _ => {
                    self.propagate_group_values(op);
                    self.check_regions(&op, Grouping::None);
                }
            }
        }
    }

    fn check_regions<'m, O: OperationLike<'c, 'm>>(&mut self, op: &O, grouping: Grouping<'m>)
    where
        'c: 'm,
    {
        for region in op.regions() {
            for block in region.blocks() {
                self.check_block(block, grouping);
            }
        }
    }

    fn check_aggregate_call(&mut self, op: OperationRef<'c, '_>, callee: &str, grouping: Grouping) {
        self.saw_aggregate = true;
        if grouping == Grouping::None {
            yuzu_mlir::diagnostics::emit_error(
                op.location(),
                &format!("aggregate function `{callee}` can only be used in an `aggregate` item"),
            );
        }

        if let Grouping::FnBody(name) = grouping
            && callee == name
        {
            yuzu_mlir::diagnostics::emit_error(
                op.location(),
                &format!("`{name}` is an `agg fn` and cannot call itself"),
            );
        }

        let nested = self.operand_aggregates(op);
        if grouping != Grouping::None {
            for &call in &nested {
                if self.nested.insert(call) {
                    let (location, name) = &self.aggregate_calls[&call];
                    yuzu_mlir::diagnostics::emit_error(
                        *location,
                        &format!(
                            "aggregate function `{name}` cannot be nested in another aggregate"
                        ),
                    );
                }
            }
        }

        if let Some(result) = op.try_first_result() {
            let id = result.id();
            let mut calls = nested;
            calls.push(id);
            self.aggregate_calls
                .insert(id, (op.location(), callee.to_string()));
            self.group_values.insert(id, calls);
        }
    }

    /// A value computed from group-level values is group-level too.
    fn propagate_group_values(&mut self, op: OperationRef<'c, '_>) {
        let calls = self.operand_aggregates(op);
        if calls.is_empty() {
            return;
        }

        if let Some(result) = op.try_first_result() {
            self.group_values.insert(result.id(), calls);
        }
    }

    /// The aggregate calls flowing into an op's operands.
    fn operand_aggregates(&self, op: OperationRef<'c, '_>) -> Vec<usize> {
        let mut calls = Vec::new();
        for operand in op.operands() {
            if let Some(through) = self.group_values.get(&operand.id()) {
                for &call in through {
                    if !calls.contains(&call) {
                        calls.push(call);
                    }
                }
            }
        }

        calls
    }

    /// The expression a function body returns — what to blame when an
    /// `agg fn` never aggregates.
    fn returned_location<'m>(op: &impl OperationLike<'c, 'm>) -> Option<Location<'c>>
    where
        'c: 'm,
    {
        let returned = op
            .regions()
            .next()?
            .first_block()?
            .last_operation()?
            .try_first_operand()?;

        Some(OperationResult::try_from(returned).ok()?.owner().location())
    }

    /// Resolution decided this when it built the call, whichever kind of
    /// callee it is, so nothing here asks the registry again.
    fn is_aggregate_call(&self, call: &CallOp<'c, '_>) -> bool {
        call.agg()
    }
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};

    use crate::check_aggregates;
    use crate::test_support;

    fn check(source: &str, expected: Expect) {
        test_support::check_diagnostics(
            source,
            |_context, module| {
                check_aggregates(module);
            },
            expected,
        );
    }

    #[test]
    fn accepts_aggregates_in_their_places() {
        check(
            r#"
struct Row { a: int64, rating: float64 }
table t = Row

def double(x: int64) -> int64 { return x * 2 }
agg def spread(x: float64) -> float64 { return max(x) - min(x) }

from t
|> where a > 1
|> aggregate sum(double(a)) as s, spread(rating) as r group by a
    "#,
            expect!["no diagnostics"],
        );
    }

    #[test]
    fn reports_an_aggregate_outside_an_aggregate_item() {
        check(
            r#"
struct Row { a: int64 }
table t = Row

def double(x: int64) -> int64 { return sum(x) }

from t
|> where sum(a) > 1
    "#,
            expect![[r#"
                error: aggregate function `sum` can only be used in an `aggregate` item
                 --> test.yz:5:40
                  |
                5 | def double(x: int64) -> int64 { return sum(x) }
                  |                                        ^

                error: aggregate function `sum` can only be used in an `aggregate` item
                 --> test.yz:8:10
                  |
                8 | |> where sum(a) > 1
                  |          ^
            "#]],
        );
    }

    /// An `external agg fn` has no body, so the "must aggregate" rule
    /// cannot apply to it.
    #[test]
    fn accepts_an_external_agg_fn() {
        check(
            r#"
struct Row { a: int64 }
table t = Row

external agg def median(x: int64) -> float64

from t
|> aggregate median(a) as m group by a
"#,
            expect!["no diagnostics"],
        );
    }

    /// A join's `on` region is row context, like any other stage's.
    #[test]
    fn reports_an_aggregate_in_a_join_condition() {
        check(
            r#"
struct Row { a: int64 }
table t = Row
struct Other { b: int64 }
table u = Other

from t
|> inner join u on sum(a) == b
"#,
            expect![[r#"
                error: aggregate function `sum` can only be used in an `aggregate` item
                 --> test.yz:8:20
                  |
                8 | |> inner join u on sum(a) == b
                  |                    ^
            "#]],
        );
    }

    #[test]
    fn reports_a_nested_aggregate() {
        check(
            r#"
struct Row { a: int64 }
table t = Row

from t
|> aggregate sum(min(a) + 1) as s group by a
    "#,
            expect![[r#"
            error: aggregate function `min` cannot be nested in another aggregate
             --> test.yz:6:18
              |
            6 | |> aggregate sum(min(a) + 1) as s group by a
              |                  ^
            "#]],
        );
    }

    #[test]
    fn reports_an_agg_fn_calling_itself() {
        check(
            r#"
struct Row { rating: float64 }
table t = Row

agg def spread(x: float64) -> float64 { return spread(x) }

from t
|> aggregate spread(rating) as r group by rating
    "#,
            expect![[r#"
                error: `spread` is an `agg fn` and cannot call itself
                 --> test.yz:5:48
                  |
                5 | agg def spread(x: float64) -> float64 { return spread(x) }
                  |                                                ^
            "#]],
        );
    }

    #[test]
    fn reports_an_agg_fn_without_an_aggregate() {
        check(
            r#"
struct Row { rating: float64 }
table t = Row

agg def spread(x: float64) -> float64 { return x }

from t
|> aggregate spread(rating) as r group by rating
    "#,
            expect![[r#"
                error: an `agg fn` must use an aggregate function
                 --> test.yz:5:1
                  |
                5 | agg def spread(x: float64) -> float64 { return x }
                  | ^
            "#]],
        );
    }
}
