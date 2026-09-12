//! InlineCalls: Substrait has no user-defined functions, so a call to one has
//! to be gone before emission — this is a requirement of the target, not an
//! optimization, and a call left standing is an error rather than a missed
//! opportunity. Builtins and externals stay: both name something the engine
//! already has.
//!
//! A call is replaced by its function's body, with each parameter reference
//! standing for the argument the call supplied. Bodies are copied from the
//! declaration every time rather than consumed, so one function serves every
//! call site, and a body carrying its own calls simply expands again on the
//! next round.
//!
//! Expansion is bounded rather than refused. A call that reduces fully is
//! fine however it got there, so nothing here asks whether a function reaches
//! itself; the budget is what stops one that never finishes.

use std::collections::HashMap;

use melior::Context;
use melior::ir::ValueLike;
use melior::ir::operation::{OperationBuilder, OperationLike, OperationRef};
use melior::ir::{Attribute, BlockRef, Identifier, Module, RegionLike, Value};
use melior::{IrRewriter, RewriterBase, ir::Location};
use yuzu_mlir::ext::{BlockExt, OperationExt, RegionExt};
use yuzu_mlir::ops::yzl::YzlOperationRef;
use yuzu_mlir::{SymbolTable, value_id};

/// How many calls one program may expand. A program whose calls reduce needs
/// far fewer than this; one that does not would never stop on its own.
const BUDGET: usize = 1000;

/// Expects a resolved, inferred module. Rewrites it in place. Diagnostics go
/// through MLIR — run this inside `yuzu_mlir::diagnostics::capture`.
pub fn inline_calls(context: &Context, module: &Module) {
    let rewriter = IrRewriter::new(context);
    let rewriter = rewriter.as_rewriter_base();
    let symbols = SymbolTable::new(module);
    let mut spent = 0;

    loop {
        let mut calls = Vec::new();
        collect_calls(module.body(), &mut calls);
        if calls.is_empty() {
            break;
        }

        if spent + calls.len() > BUDGET {
            report_budget(&calls);
            return;
        }

        spent += calls.len();
        for call in calls {
            if !expand(&rewriter, call, &symbols) {
                return;
            }
        }
    }

    discard_declarations(&rewriter, module.body());
}

/// The calls that have to go, innermost first. Declarations are skipped: a
/// body is a template, and expanding one in place would work through a
/// function that reaches itself without a call site ever asking.
fn collect_calls<'c, 'a>(block: BlockRef<'c, 'a>, out: &mut Vec<OperationRef<'c, 'a>>) {
    for op in block.operations() {
        match YzlOperationRef::of(&op) {
            Some(YzlOperationRef::Fn(_) | YzlOperationRef::Trait(_) | YzlOperationRef::Impl(_)) => {
                continue;
            }
            Some(YzlOperationRef::Call(call)) => {
                if matches!(
                    call.callee_kind().map(|kind| kind.value()),
                    Some("fn") | Some("agg_fn")
                ) {
                    out.push(op);
                }
            }
            _ => {
                for region in op.regions() {
                    for inner in region.blocks() {
                        collect_calls(inner, out);
                    }
                }
            }
        }
    }
}

/// Copies one function body over one call, answering whether it worked.
fn expand<'c, 'a>(
    rewriter: &'a RewriterBase<'c, 'a>,
    call: OperationRef<'c, '_>,
    symbols: &SymbolTable<'c, '_>,
) -> bool {
    let Some(YzlOperationRef::Call(site)) = YzlOperationRef::of(&call) else {
        return error(call.location(), "expected a call to expand");
    };

    let callee = site.callee().value();
    let Some(declaration) = symbols.lookup(callee) else {
        return error(call.location(), &format!("unknown function `{callee}`"));
    };

    let Some(YzlOperationRef::Fn(function)) = YzlOperationRef::of(&declaration) else {
        return error(call.location(), &format!("`{callee}` is not a function"));
    };

    let Some(body) = function.body().first_block() else {
        return error(
            call.location(),
            &format!("`{callee}` has no body to expand here"),
        );
    };

    let arguments: Vec<Value> = call.operands().collect();
    rewriter.set_insertion_point_before(call);

    let mut values: HashMap<usize, Value> = HashMap::new();
    let mut returned = None;
    for op in body.operations() {
        match YzlOperationRef::of(&op) {
            // A parameter reference is the argument, not a copy of anything.
            Some(YzlOperationRef::Name(name)) if name.param().is_some() => {
                let index = name
                    .param()
                    .expect("the parameter stamp is present")
                    .value() as usize;
                let Some(&argument) = arguments.get(index) else {
                    return error(
                        call.location(),
                        &format!("`{callee}` wants an argument this call did not supply"),
                    );
                };

                values.insert(value_id(op.first_result()), argument);
            }
            Some(YzlOperationRef::Return(_)) => {
                returned = op
                    .try_first_operand()
                    .and_then(|value| values.get(&value_id(value)).copied());
            }
            _ => {
                if op.regions().next().is_some() {
                    return error(
                        op.location(),
                        &format!("`{callee}` has a body this expansion cannot copy"),
                    );
                }

                let Some(copied) = copy(rewriter, op, &values) else {
                    return error(
                        op.location(),
                        &format!("`{callee}` has a body that did not copy"),
                    );
                };

                if let Some(result) = op.try_first_result() {
                    values.insert(value_id(result), copied);
                }
            }
        }
    }

    let Some(returned) = returned else {
        return error(
            call.location(),
            &format!("`{callee}` does not return a value to use here"),
        );
    };

    rewriter.replace_all_op_uses_with_values(call, &[returned]);
    rewriter.erase_op(call);
    true
}

/// One body operation, rebuilt against the values its operands became.
fn copy<'c, 'a>(
    rewriter: &'a RewriterBase<'c, 'a>,
    op: OperationRef<'c, '_>,
    values: &HashMap<usize, Value<'c, 'a>>,
) -> Option<Value<'c, 'a>> {
    let operands: Vec<Value> = op
        .operands()
        .map(|operand| values.get(&value_id(operand)).copied().unwrap_or(operand))
        .collect();
    let results: Vec<_> = (0..op.result_count())
        .map(|index| {
            op.result(index)
                .expect("the result index is in range")
                .r#type()
        })
        .collect();
    let attributes: Vec<(Identifier<'c>, Attribute<'c>)> = (0..op.attribute_count())
        .map(|index| {
            op.attribute_at(index)
                .expect("the attribute index is in range")
        })
        .collect();

    let built = OperationBuilder::new(
        op.name()
            .as_string_ref()
            .as_str()
            .expect("op names are utf-8"),
        op.location(),
    )
    .add_operands(&operands)
    .add_results(&results)
    .add_attributes(&attributes)
    .build()
    .ok()?;

    let inserted = rewriter.insert(built);
    inserted.try_first_result()
}

/// Once every call is expanded the declarations describe nothing the module
/// still contains. An external keeps its name on the call rather than here.
fn discard_declarations(rewriter: &RewriterBase, block: BlockRef) {
    let mut declarations = Vec::new();
    for op in block.operations() {
        if matches!(
            YzlOperationRef::of(&op),
            Some(YzlOperationRef::Fn(_) | YzlOperationRef::Trait(_) | YzlOperationRef::Impl(_))
        ) {
            declarations.push(op);
        }
    }

    for declaration in declarations {
        rewriter.erase_op(declaration);
    }
}

/// The budget is spent where the program stopped reducing, so the calls still
/// standing are the ones to name.
fn report_budget(calls: &[OperationRef]) {
    let call = calls
        .first()
        .expect("the budget is reported over some call");
    let name = match YzlOperationRef::of(call) {
        Some(YzlOperationRef::Call(site)) => site.callee().value(),
        _ => "a function",
    };

    error(
        call.location(),
        &format!(
            "expanding `{name}` did not finish within {BUDGET} calls; \
             a function that reaches itself has to reduce to stop"
        ),
    );
}

fn error(location: Location, message: &str) -> bool {
    yuzu_mlir::diagnostics::emit_error(location, message);
    false
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::test_support::check_simplified;

    /// Substrait has no user-defined functions, so the call has to become the
    /// body it names.
    #[test]
    fn a_call_becomes_the_body_it_names() {
        check_simplified(
            r#"
struct Row { a: int64 }
table t = Row

fn double(x: int64) -> int64 { return x * 2 }

from t
|> select double(a) as d
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a"] : [!yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  yz.struct @row ["d"] : [!yz.int64]
                  %1 = yzr.project %0 {
                  ^bb0(%arg0: !yz.int64):
                    %2 = yz.constant_int 2
                    %3 = yz.mul %arg0, %2 : !yz.int64, !yz.int64 -> !yz.int64
                    yzr.yield %3 : !yz.int64
                  } : !yz.struct<@Row> -> !yz.struct<@row>
                  yzr.output %1 : !yz.struct<@row>
                }
            "#]],
        );
    }

    /// A body carrying its own calls expands again on the next round, so
    /// nesting needs no special handling.
    #[test]
    fn a_nested_call_expands_too() {
        check_simplified(
            r#"
struct Row { a: int64 }
table t = Row

fn double(x: int64) -> int64 { return x * 2 }
fn quadruple(x: int64) -> int64 { return double(double(x)) }

from t
|> select quadruple(a) as q
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a"] : [!yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  yz.struct @row ["q"] : [!yz.int64]
                  %1 = yzr.project %0 {
                  ^bb0(%arg0: !yz.int64):
                    %2 = yz.constant_int 2
                    %3 = yz.mul %arg0, %2 : !yz.int64, !yz.int64 -> !yz.int64
                    %4 = yz.mul %3, %2 : !yz.int64, !yz.int64 -> !yz.int64
                    yzr.yield %4 : !yz.int64
                  } : !yz.struct<@Row> -> !yz.struct<@row>
                  yzr.output %1 : !yz.struct<@row>
                }
            "#]],
        );
    }

    /// A builtin names something the engine already has, so it stays a call.
    #[test]
    fn a_builtin_call_is_left_alone() {
        check_simplified(
            r#"
struct Row { a: int64 }
table t = Row

from t
|> select pow(a, 2) as p
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a"] : [!yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  yz.struct @row ["p"] : [!yz.int64]
                  %1 = yzr.project %0 {
                  ^bb0(%arg0: !yz.int64):
                    %2 = yz.constant_int 2
                    %3 = yz.call @pow(%arg0, %2) : (!yz.int64, !yz.int64) -> !yz.int64
                    yzr.yield %3 : !yz.int64
                  } : !yz.struct<@Row> -> !yz.struct<@row>
                  yzr.output %1 : !yz.struct<@row>
                }
            "#]],
        );
    }

    /// Nothing here asks whether a function reaches itself — a call that
    /// reduces is fine however it got there. What stops one that cannot is
    /// the budget, and the error says so where the program stopped.
    #[test]
    fn a_call_that_never_reduces_exhausts_the_budget() {
        check_simplified(
            r#"
struct Row { a: int64 }
table t = Row

fn forever(x: int64) -> int64 { return forever(x) }

from t
|> select forever(a) as f
"#,
            expect![[r#"
                error: expanding `forever` did not finish within 1000 calls; a function that reaches itself has to reduce to stop
                 --> test.yz:5:40
                  |
                5 | fn forever(x: int64) -> int64 { return forever(x) }
                  |                                        ^

                error: `yzl.fn` was not expanded before lowering
                 --> test.yz:5:1
                  |
                5 | fn forever(x: int64) -> int64 { return forever(x) }
                  | ^
            "#]],
        );
    }
}
