//! Substrait has no user-defined functions, so a call to one is replaced by
//! the body it names, and a call left standing is an error. Nothing asks
//! whether a function reaches itself: a call that reduces is fine, and the
//! budget stops one that never finishes.

use std::collections::VecDeque;

use melior::ir::attribute::TypeAttribute;
use melior::ir::operation::{OperationBuilder, OperationLike, OperationRef};
use melior::ir::{
    Attribute, BlockLike, BlockRef, Identifier, Location, Module, RegionLike, Type, Value,
    ValueLike,
};
use melior::{Context, IrRewriter, RewriterBase};
use rustc_hash::FxHashMap;
use yuzu_mlir::attributes::CalleeSource;
use yuzu_mlir::diagnostics::emit_error;
use yuzu_mlir::ir::attribute::array::ArrayAttributeExt;
use yuzu_mlir::ir::block::BlockExt;
use yuzu_mlir::ir::operation::{OperationCast, OperationExt};
use yuzu_mlir::ir::region::RegionExt;
use yuzu_mlir::ir::value::{ValueExt, ValueId};
use yuzu_mlir::ops::yzl::{CallOp, ConstOp, FnOp, YzlOp};
use yuzu_mlir::types::QueryType;
use yuzu_mlir::{ParamType, SymbolTable};

use crate::operators::Operator;

const BUDGET: usize = 1000;

pub fn inline_calls(context: &Context, module: &mut Module) {
    let rewriter = IrRewriter::new(context);
    let rewriter = rewriter.as_rewriter_base();
    let symbols = SymbolTable::new(module);

    // The module is walked once. An expansion hands back the calls it
    // copied in, so a chain of expansions never walks the module again.
    let mut calls = Vec::new();
    collect_calls(module.body(), &mut calls);
    let mut pending = VecDeque::from(calls);
    let mut spent = 0;
    while let Some(call) = pending.pop_front() {
        if spent == BUDGET {
            report_budget(call);
            return;
        }

        spent += 1;
        let Some(copied) = expand(&rewriter, call, &symbols) else {
            return;
        };

        pending.extend(copied);
    }

    discard_declarations(context, rewriter, module.body());
}

fn collect_calls<'c, 'a>(block: BlockRef<'c, 'a>, out: &mut Vec<OperationRef<'c, 'a>>) {
    for op in block.operations() {
        match op.as_yzl() {
            // An operator's implementation outlives this pass, so what it
            // calls is expanded where it stands.
            Some(YzlOp::Fn(function)) if implements_operator(function) => {
                for region in op.regions() {
                    for inner in region.blocks() {
                        collect_calls(inner, out);
                    }
                }
            }
            Some(YzlOp::Fn(_) | YzlOp::Trait(_) | YzlOp::Impl(_)) => {}
            Some(YzlOp::Call(_)) => {
                if expands(op) {
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

/// A call to a function or a constant, which is what this pass replaces.
fn expands(op: OperationRef) -> bool {
    matches!(
        op.as_yzl(),
        Some(YzlOp::Call(call)) if matches!(
            call.callee_source(),
            Some(CalleeSource::Fn | CalleeSource::Const)
        )
    )
}

/// Replaces a call with a copy of its body, and returns the calls the copy
/// holds, which need expanding in turn.
fn expand<'c, 'a>(
    rewriter: &'a RewriterBase<'c, 'a>,
    call: OperationRef<'c, 'a>,
    symbols: &SymbolTable<'c, '_>,
) -> Option<Vec<OperationRef<'c, 'a>>> {
    let Some(YzlOp::Call(site)) = call.as_yzl() else {
        return error(call.location(), "expected a call to expand");
    };

    let callee = site.callee().value();
    let declaration = symbols
        .lookup(callee)
        .or_else(|| error(call.location(), &format!("unknown function `{callee}`")))?;

    let (body, types) = match declaration.as_yzl() {
        Some(YzlOp::Fn(function)) => (
            function.body().first_block(),
            type_arguments(function, site, call.location(), callee)?,
        ),
        Some(YzlOp::Const(binding)) => (binding.body().first_block(), FxHashMap::default()),
        _ => return error(call.location(), &format!("`{callee}` is not a function")),
    };
    let body = body.or_else(|| {
        error(
            call.location(),
            &format!("`{callee}` has no body to expand here"),
        )
    })?;

    let arguments: Vec<Value> = call.operands().collect();
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

    rewriter.set_insertion_point_before(call);

    let mut values: FxHashMap<ValueId, Value> = body
        .arguments()
        .map(|parameter| parameter.id())
        .zip(arguments)
        .collect();
    let mut returned = None;
    let mut copied_calls = Vec::new();
    for op in body.operations() {
        if let Some(YzlOp::Return(_) | YzlOp::Yield(_)) = op.as_yzl() {
            returned = op
                .try_first_operand()
                .and_then(|value| values.get(&value.id()).copied());
        } else {
            if op.regions().next().is_some() {
                return error(
                    op.location(),
                    &format!("`{callee}` has a body this expansion cannot copy"),
                );
            }

            let copied = copy(rewriter, op, &values, &types).or_else(|| {
                error(
                    op.location(),
                    &format!("`{callee}` has a body that did not copy"),
                )
            })?;
            if expands(copied) {
                copied_calls.push(copied);
            }

            if let (Some(result), Some(value)) = (op.try_first_result(), copied.try_first_result())
            {
                values.insert(result.id(), value);
            }
        }
    }

    let returned = returned.or_else(|| {
        error(
            call.location(),
            &format!("`{callee}` does not return a value to use here"),
        )
    })?;
    rewriter.replace_all_op_uses_with_values(call, &[returned]);
    rewriter.erase_op(call);
    Some(copied_calls)
}

pub(crate) fn copy<'c, 'a>(
    rewriter: &'a RewriterBase<'c, 'a>,
    op: OperationRef<'c, '_>,
    values: &FxHashMap<ValueId, Value<'c, 'a>>,
    types: &FxHashMap<&str, Type<'c>>,
) -> Option<OperationRef<'c, 'a>> {
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

    Some(rewriter.insert(built))
}

fn type_arguments<'c>(
    function: FnOp<'c, '_>,
    site: CallOp<'c, '_>,
    location: Location<'c>,
    callee: &str,
) -> Option<FxHashMap<&'c str, Type<'c>>> {
    let Some(parameters) = function.type_params() else {
        return Some(FxHashMap::default());
    };

    let parameters: Vec<&str> = parameters.strings().collect();
    let arguments = site
        .type_args()
        .or_else(|| {
            error(
                location,
                &format!("`{callee}` is generic and this call's types were never settled"),
            )
        })?
        .types()
        .collect::<Vec<_>>();
    if arguments.len() != parameters.len() {
        return error(
            location,
            &format!("`{callee}` takes {} type parameters", parameters.len()),
        );
    }

    Some(parameters.into_iter().zip(arguments).collect())
}

pub(crate) fn substitute<'c>(ty: Type<'c>, types: &FxHashMap<&str, Type<'c>>) -> Type<'c> {
    ParamType::from_type(ty)
        .and_then(|param| types.get(param.name()).copied())
        .unwrap_or(ty)
}

/// A `let` bound to a query stays: its stages are rows other queries name.
/// An external function stays too: it has no body to expand, and the yzr
/// lowering reads the engine's name from it. So does an operator's
/// implementation, which `legalize_operators` copies in after the folds.
fn discard_declarations(context: &Context, rewriter: RewriterBase, block: BlockRef) {
    let mut declarations = Vec::new();
    for op in block.operations() {
        let discard = match op.as_yzl() {
            Some(YzlOp::Fn(function)) => !function.is_external() && !implements_operator(function),
            Some(YzlOp::Trait(_) | YzlOp::Impl(_)) => true,
            Some(YzlOp::Const(binding)) => !binds_query(context, binding),
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

fn implements_operator(function: FnOp) -> bool {
    Operator::implemented_by(function.sym_name().value()).is_some()
}

fn binds_query(context: &Context, binding: ConstOp) -> bool {
    binding
        .body()
        .first_block()
        .and_then(|block| block.last_operation())
        .and_then(|yielded| yielded.try_first_operand())
        .is_some_and(|value| value.r#type() == QueryType::get(context))
}

fn report_budget(call: OperationRef) {
    let name = match call.as_yzl() {
        Some(YzlOp::Call(site)) => site.callee().value(),
        _ => "a function",
    };

    emit_error(
        call.location(),
        &format!(
            "expanding `{name}` did not finish within {BUDGET} calls; \
             a function that reaches itself has to reduce to stop"
        ),
    );
}

fn error<T>(location: Location, message: &str) -> Option<T> {
    emit_error(location, message);
    None
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::test_support::check_simplified;

    #[test]
    fn a_call_becomes_the_body_it_names() {
        check_simplified(
            r"
struct Row { a: int64 }
table t = Row

def double(x: int64) -> int64 { return x * 2 }

from t
|> select double(a) as d
",
            &expect![[r#"
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

    #[test]
    fn a_nested_call_expands_too() {
        check_simplified(
            r"
struct Row { a: int64 }
table t = Row

def double(x: int64) -> int64 { return x * 2 }
def quadruple(x: int64) -> int64 { return double(double(x)) }

from t
|> select quadruple(a) as q
",
            &expect![[r#"
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

    #[test]
    fn a_builtin_call_is_left_alone() {
        check_simplified(
            r"
struct Row { a: int64 }
table t = Row

from t
|> aggregate count(a) as n
",
            &expect![[r#"
                module {
                  yz.struct @Row ["a"] : [!yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  yz.struct @row ["n"] : [!yz.int64]
                  %1 = yzr.aggregate %0 keys [] {
                  ^bb0(%arg0: !yz.int64):
                    %2 = yzr.agg "count", %arg0 : !yz.int64 -> !yz.int64
                    yzr.yield %2 : !yz.int64
                  } : !yz.struct<@Row> -> !yz.struct<@row>
                  yzr.output %1 : !yz.struct<@row>
                }
            "#]],
        );
    }

    #[test]
    fn a_call_that_never_reduces_exhausts_the_budget() {
        check_simplified(
            r"
struct Row { a: int64 }
table t = Row

def forever(x: int64) -> int64 { return forever(x) }

from t
|> select forever(a) as f
",
            &expect![[r"
                error: expanding `forever` did not finish within 1000 calls; a function that reaches itself has to reduce to stop
                 --> test.yz:5:41
                  |
                5 | def forever(x: int64) -> int64 { return forever(x) }
                  |                                         ^^^^^^^^^^

                error: `yzl.fn` was not expanded before lowering
                 --> test.yz:5:1
                  |
                5 | def forever(x: int64) -> int64 { return forever(x) }
                  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                error: `forever` was not expanded before lowering
                 --> test.yz:5:41
                  |
                5 | def forever(x: int64) -> int64 { return forever(x) }
                  |                                         ^^^^^^^^^^
            "]],
        );
    }

    #[test]
    fn a_generic_body_takes_the_types_of_its_call() {
        check_simplified(
            r"
struct Row { a: int64, r: float64 }
table t = Row

def twice[T](x: T) -> T { return x + x }

from t
|> extend twice(a) as m, twice(r) as n
",
            &expect![[r#"
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

    #[test]
    fn reports_a_trait_method_it_cannot_dispatch() {
        check_simplified(
            r"
trait Zero {
    def zero(x: Self) -> Self
}

impl Zero for int64 {
    def zero(x: int64) -> int64 { return 0 }
}

def shift[T](x: T) -> T where T: Zero { return zero(x) }

struct Row { a: int64 }
table t = Row

from t
|> extend shift(a) as z
",
            &expect![[r"
                error: `zero` is a trait method, and calling one is not supported yet
                 --> test.yz:10:48
                   |
                10 | def shift[T](x: T) -> T where T: Zero { return zero(x) }
                   |                                                ^^^^^^^

                error: this part of the query is missing
                 --> test.yz:10:48
                   |
                10 | def shift[T](x: T) -> T where T: Zero { return zero(x) }
                   |                                                ^^^^^^^
            "]],
        );
    }

    #[test]
    fn a_scalar_let_is_expanded_and_discarded() {
        check_simplified(
            r"
struct Row { a: int64 }
table t = Row

let ids: List[int64] = [1, 3]

from t
|> where a in ids
",
            &expect![[r#"
                module {
                  yz.struct @Row ["a"] : [!yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  %1 = yzr.filter %0 : !yz.struct<@Row> {
                  ^bb0(%arg0: !yz.int64):
                    %2 = yz.constant_int 1
                    %3 = yz.constant_int 3
                    %4 = yz.list[%2, %3] : (!yz.int64, !yz.int64) -> !yz.list<!yz.int64>
                    %5 = yz.in %arg0, %4 : !yz.int64, !yz.list<!yz.int64> -> !yz.bool
                    yzr.yield %5 : !yz.bool
                  }
                  yzr.output %1 : !yz.struct<@Row>
                }
            "#]],
        );
    }
}
