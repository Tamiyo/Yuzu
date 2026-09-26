//! An aggregate call lives only in an `aggregate` item or an `agg fn` body,
//! never in another aggregate's arguments, and an `agg fn` must use an
//! aggregate without calling itself.

use std::collections::{HashMap, HashSet};
use std::mem;

use melior::ir::operation::{OperationLike, OperationRef, OperationResult};
use melior::ir::{BlockRef, Location, Module, RegionLike};
use yuzu_mlir::diagnostics::emit_error;
use yuzu_mlir::ir::block::BlockExt;
use yuzu_mlir::ir::operation::{OperationCast, OperationExt};
use yuzu_mlir::ir::region::RegionExt;
use yuzu_mlir::ir::value::{ValueExt, ValueId};
use yuzu_mlir::ops::yzl::YzlOp;

pub fn check_aggregates(module: &Module) {
    let mut checker = AggregateChecker {
        group_values: HashMap::new(),
        aggregate_calls: HashMap::new(),
        nested: HashSet::new(),
        saw_aggregate: false,
    };

    checker.check_block(module.body(), None);
}

/// Where an aggregate call may appear.
#[derive(Clone, Copy)]
enum Grouping<'c> {
    Item,
    /// Named to catch self-calls.
    FnBody(&'c str),
}

struct AggregateChecker<'c> {
    /// Each group-level value, with the aggregate calls it came from.
    group_values: HashMap<ValueId, Vec<ValueId>>,
    /// The location and callee of each aggregate call, by its result.
    aggregate_calls: HashMap<ValueId, (Location<'c>, &'c str)>,
    nested: HashSet<ValueId>,
    saw_aggregate: bool,
}

impl<'c> AggregateChecker<'c> {
    fn check_block(&mut self, block: BlockRef<'c, '_>, grouping: Option<Grouping<'c>>) {
        for op in block.operations() {
            match op.as_yzl() {
                Some(YzlOp::Call(call)) if call.is_agg() => {
                    self.check_aggregate_call(op, call.callee().value(), grouping);
                }
                Some(YzlOp::Aggregate(stage)) => {
                    self.check_regions(stage.operation(), Some(Grouping::Item));
                }
                // An `external agg fn` has no body to aggregate in.
                Some(YzlOp::Fn(function))
                    if function.is_agg() && function.external_name().is_none() =>
                {
                    let name = function.sym_name().value();
                    let outer = mem::replace(&mut self.saw_aggregate, false);
                    self.check_regions(function.operation(), Some(Grouping::FnBody(name)));
                    if !self.saw_aggregate {
                        emit_error(
                            Self::returned_location(function.operation())
                                .unwrap_or_else(|| op.location()),
                            "an `agg fn` must use an aggregate function",
                        );
                    }

                    self.saw_aggregate = outer;
                }
                Some(YzlOp::Fn(function)) => {
                    self.check_regions(function.operation(), None);
                }
                _ => {
                    self.propagate_group_values(op);
                    self.check_regions(&op, None);
                }
            }
        }
    }

    fn check_regions<'m, O: OperationLike<'c, 'm>>(
        &mut self,
        op: &O,
        grouping: Option<Grouping<'c>>,
    ) where
        'c: 'm,
    {
        for region in op.regions() {
            for block in region.blocks() {
                self.check_block(block, grouping);
            }
        }
    }

    fn check_aggregate_call(
        &mut self,
        op: OperationRef<'c, '_>,
        callee: &'c str,
        grouping: Option<Grouping<'c>>,
    ) {
        self.saw_aggregate = true;
        if grouping.is_none() {
            emit_error(
                op.location(),
                &format!("aggregate function `{callee}` can only be used in an `aggregate` item"),
            );
        }

        if let Some(Grouping::FnBody(name)) = grouping
            && callee == name
        {
            emit_error(
                op.location(),
                &format!("`{name}` is an `agg fn` and cannot call itself"),
            );
        }

        let nested = self.operand_aggregates(op);
        if grouping.is_some() {
            for &call in &nested {
                if self.nested.insert(call) {
                    let (location, name) = self.aggregate_calls[&call];
                    emit_error(
                        location,
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
            self.aggregate_calls.insert(id, (op.location(), callee));
            self.group_values.insert(id, calls);
        }
    }

    fn propagate_group_values(&mut self, op: OperationRef<'c, '_>) {
        let calls = self.operand_aggregates(op);
        if calls.is_empty() {
            return;
        }

        if let Some(result) = op.try_first_result() {
            self.group_values.insert(result.id(), calls);
        }
    }

    fn operand_aggregates(&self, op: OperationRef<'c, '_>) -> Vec<ValueId> {
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
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};

    use crate::check_aggregates;
    use crate::test_support;

    fn check(source: &str, expected: Expect) {
        test_support::check(
            source,
            |context, module| {
                crate::promote_locals(context, module);
                check_aggregates(module);
                "no diagnostics".to_string()
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
                  |                                        ^^^^^^

                error: aggregate function `sum` can only be used in an `aggregate` item
                 --> test.yz:8:10
                  |
                8 | |> where sum(a) > 1
                  |          ^^^^^^
            "#]],
        );
    }

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
                  |                    ^^^^^^
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
                  |                  ^^^^^^
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
                  |                                                ^^^^^^^^^
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
                  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            "#]],
        );
    }
}
