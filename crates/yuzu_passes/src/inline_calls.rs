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
use melior::ir::attribute::TypeAttribute;
use melior::ir::operation::{OperationBuilder, OperationLike, OperationRef};
use melior::ir::{Attribute, BlockLike, BlockRef, Identifier, Module, RegionLike, Type, Value};
use melior::{IrRewriter, RewriterBase, ir::Location};
use yuzu_mlir::attributes::CalleeKind;
use yuzu_mlir::ext::{
    ArrayAttributeExt, BlockExt, OperationCast, OperationExt, RegionExt, ValueExt,
};
use yuzu_mlir::ops::yzl::YzlOp;
use yuzu_mlir::types;
use yuzu_mlir::{ParamType, SymbolTable};

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

    discard_declarations(context, &rewriter, module.body());
}

/// The calls that have to go, innermost first. Declarations are skipped: a
/// body is a template, and expanding one in place would work through a
/// function that reaches itself without a call site ever asking.
fn collect_calls<'c, 'a>(block: BlockRef<'c, 'a>, out: &mut Vec<OperationRef<'c, 'a>>) {
    for op in block.operations() {
        match op.as_yzl() {
            Some(YzlOp::Fn(_) | YzlOp::Trait(_) | YzlOp::Impl(_)) => {
                continue;
            }
            Some(YzlOp::Call(call)) => {
                if matches!(
                    call.callee_kind(),
                    Some(CalleeKind::Fn | CalleeKind::AggFn | CalleeKind::Let)
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
    let Some(YzlOp::Call(site)) = call.as_yzl() else {
        return error(call.location(), "expected a call to expand");
    };

    let callee = site.callee().value();
    let Some(declaration) = symbols.lookup(callee) else {
        return error(call.location(), &format!("unknown function `{callee}`"));
    };

    // A `let` is a body with no parameters; it expands exactly as a
    // function does, and yields where a function returns.
    let (body, arguments_types) = match declaration.as_yzl() {
        Some(YzlOp::Fn(function)) => {
            let Some(body) = function.body().first_block() else {
                return error(
                    call.location(),
                    &format!("`{callee}` has no body to expand here"),
                );
            };

            let Some(types) = type_arguments(&function, &site, call.location(), callee) else {
                return false;
            };

            (body, types)
        }
        Some(YzlOp::Let(binding)) => {
            let Some(body) = binding.body().first_block() else {
                return error(
                    call.location(),
                    &format!("`{callee}` has no body to expand here"),
                );
            };

            (body, HashMap::new())
        }
        _ => return error(call.location(), &format!("`{callee}` is not a function")),
    };

    let arguments: Vec<Value> = call.operands().collect();
    rewriter.set_insertion_point_before(call);

    // The body's block arguments are its parameters; each stands for the
    // argument the call supplied, so a use of one copies as a use of that.
    let mut values: HashMap<usize, Value> = HashMap::new();
    if body.argument_count() != arguments.len() {
        return error(
            call.location(),
            &format!(
                "`{callee}` takes {} arguments, got {}",
                body.argument_count(),
                arguments.len()
            ),
        );
    }

    for (index, &argument) in arguments.iter().enumerate() {
        let parameter = body
            .argument(index)
            .expect("the argument index is in range");
        values.insert(parameter.id(), argument);
    }

    let mut returned = None;
    for op in body.operations() {
        match op.as_yzl() {
            Some(YzlOp::Return(_) | YzlOp::Yield(_)) => {
                returned = op
                    .try_first_operand()
                    .and_then(|value| values.get(&value.id()).copied());
            }
            _ => {
                if op.regions().next().is_some() {
                    return error(
                        op.location(),
                        &format!("`{callee}` has a body this expansion cannot copy"),
                    );
                }

                let Some(copied) = copy(rewriter, op, &values, &arguments_types) else {
                    return error(
                        op.location(),
                        &format!("`{callee}` has a body that did not copy"),
                    );
                };

                if let Some(result) = op.try_first_result() {
                    values.insert(result.id(), copied);
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
    types: &HashMap<&str, Type<'c>>,
) -> Option<Value<'c, 'a>> {
    let operands: Vec<Value> = op
        .operands()
        .map(|operand| values.get(&operand.id()).copied().unwrap_or(operand))
        .collect();
    let results: Vec<_> = (0..op.result_count())
        .map(|index| {
            let ty = op
                .result(index)
                .expect("the result index is in range")
                .r#type();
            substitute(ty, types)
        })
        .collect();
    let attributes: Vec<(Identifier<'c>, Attribute<'c>)> = (0..op.attribute_count())
        .map(|index| {
            let (name, attribute) = op
                .attribute_at(index)
                .expect("the attribute index is in range");
            match TypeAttribute::try_from(attribute) {
                Ok(stamp) => (
                    name,
                    TypeAttribute::new(substitute(stamp.value(), types)).into(),
                ),
                Err(_) => (name, attribute),
            }
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

/// What the call chose for each of the function's type parameters. A generic
/// call inference never settled carries no stamp, and expansion is where that
/// can be said usefully — the body would otherwise be copied with a parameter
/// standing in for a type.
fn type_arguments<'c>(
    function: &yuzu_mlir::ops::yzl::FnOp<'c, '_>,
    site: &yuzu_mlir::ops::yzl::CallOp<'c, '_>,
    location: Location<'c>,
    callee: &str,
) -> Option<HashMap<&'c str, Type<'c>>> {
    let Some(parameters) = function.type_params() else {
        return Some(HashMap::new());
    };

    let parameters = parameters.strings();
    let Some(arguments) = site.type_args() else {
        return error(
            location,
            &format!("`{callee}` is generic and this call's types were never settled"),
        )
        .then(HashMap::new);
    };

    let arguments: Vec<Type<'c>> = arguments
        .elements()
        .filter_map(|element| TypeAttribute::try_from(element).ok())
        .map(|attribute| attribute.value())
        .collect();
    if arguments.len() != parameters.len() {
        return error(
            location,
            &format!("`{callee}` takes {} type parameters", parameters.len()),
        )
        .then(HashMap::new);
    }

    Some(parameters.into_iter().zip(arguments).collect())
}

/// A parameter in type position becomes the type the call chose for it.
fn substitute<'c>(ty: Type<'c>, types: &HashMap<&str, Type<'c>>) -> Type<'c> {
    ParamType::from_type(ty)
        .and_then(|param| types.get(param.name()).copied())
        .unwrap_or(ty)
}

/// Once every call is expanded the declarations describe nothing the module
/// still contains. An external keeps its name on the call rather than here,
/// and a `let` bound to a query stays: its stages are rows other queries name.
fn discard_declarations(context: &Context, rewriter: &RewriterBase, block: BlockRef) {
    let mut declarations = Vec::new();
    for op in block.operations() {
        let discard = match op.as_yzl() {
            Some(YzlOp::Fn(_) | YzlOp::Trait(_) | YzlOp::Impl(_)) => true,
            Some(YzlOp::Let(binding)) => !binds_query(context, &binding),
            _ => false,
        };
        if discard {
            declarations.push(op);
        }
    }

    for declaration in declarations {
        rewriter.erase_op(declaration);
    }
}

fn binds_query(context: &Context, binding: &yuzu_mlir::ops::yzl::LetOp) -> bool {
    binding
        .body()
        .first_block()
        .and_then(|block| block.last_operation())
        .and_then(|yielded| yielded.try_first_operand())
        .is_some_and(|value| value.r#type() == types::query(context))
}

/// The budget is spent where the program stopped reducing, so the calls still
/// standing are the ones to name.
fn report_budget(calls: &[OperationRef]) {
    let call = calls
        .first()
        .expect("the budget is reported over some call");
    let name = match call.as_yzl() {
        Some(YzlOp::Call(site)) => site.callee().value(),
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

                error: `forever` was not expanded before lowering
                 --> test.yz:5:40
                  |
                5 | fn forever(x: int64) -> int64 { return forever(x) }
                  |                                        ^
            "#]],
        );
    }

    /// A generic body is copied per call site with the types that call
    /// settled on, so one declaration serves both columns — monomorphizing
    /// falls out of expanding rather than needing a pass of its own.
    #[test]
    fn a_generic_body_takes_the_types_of_its_call() {
        check_simplified(
            r#"
struct Row { a: int64, r: float64 }
table t = Row

fn twice[T](x: T) -> T { return x + x }

from t
|> extend twice(a) as m, twice(r) as n
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a", "r"] : [!yz.int64, !yz.float64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  yz.struct @row ["a", "r", "m", "n"] : [!yz.int64, !yz.float64, !yz.int64, !yz.float64]
                  %1 = yzr.extend %0 {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.float64):
                    %2 = yz.add %arg0, %arg0 : !yz.int64, !yz.int64 -> !yz.int64
                    %3 = yz.add %arg1, %arg1 : !yz.float64, !yz.float64 -> !yz.float64
                    yzr.yield %2, %3 : !yz.int64, !yz.float64
                  } : !yz.struct<@Row> -> !yz.struct<@row>
                  yzr.output %1 : !yz.struct<@row>
                }
            "#]],
        );
    }

    /// Dispatch is the piece that is not written: a trait's methods live in
    /// its implementations, and choosing between them needs the concrete
    /// `Self`. The gap says so rather than claiming the method is unknown.
    #[test]
    fn reports_a_trait_method_it_cannot_dispatch() {
        check_simplified(
            r#"
trait Zero {
    fn zero(x: Self) -> Self
}

impl Zero for int64 {
    fn zero(x: int64) -> int64 { return 0 }
}

fn shift[T](x: T) -> T where T: Zero { return zero(x) }

struct Row { a: int64 }
table t = Row

from t
|> extend shift(a) as z
"#,
            expect![[r#"
                error: `zero` is a trait method, and calling one is not supported yet
                 --> test.yz:10:47
                   |
                10 | fn shift[T](x: T) -> T where T: Zero { return zero(x) }
                   |                                               ^^^^^^^

                error: this part of the query is missing
                 --> test.yz:10:47
                   |
                10 | fn shift[T](x: T) -> T where T: Zero { return zero(x) }
                   |                                               ^
            "#]],
        );
    }

    /// A scalar `let` is a body with no parameters: its uses expand like
    /// calls, and the binding goes with the functions.
    #[test]
    fn a_scalar_let_is_expanded_and_discarded() {
        check_simplified(
            r#"
struct Row { a: int64 }
table t = Row

let ids: List[int64] = [1, 3]

from t
|> where a in ids
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a"] : [!yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  %1 = yzr.filter %0 : !yz.struct<@Row> {
                  ^bb0(%arg0: !yz.int64):
                    %2 = yz.constant_int 1
                    %3 = yz.constant_int 3
                    %4 = yz.list[%2, %3] : (!yz.int64, !yz.int64) -> !yz.list<!yz.int64>
                    %5 = yz.call @in(%arg0, %4) : (!yz.int64, !yz.list<!yz.int64>) -> !yz.bool
                    yzr.yield %5 : !yz.bool
                  }
                  yzr.output %1 : !yz.struct<@Row>
                }
            "#]],
        );
    }
}
