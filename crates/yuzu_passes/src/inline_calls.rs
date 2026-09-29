//! Substrait has no user-defined functions, so a call to one is replaced by
//! the body it names, and a call left standing is an error. The language
//! has no conditional, so a function that reaches itself never stops
//! expanding: such a cycle is reported before anything expands.

use std::collections::VecDeque;

use melior::ir::attribute::{ArrayAttribute, TypeAttribute};
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
use yuzu_mlir::ir::symbol_table::SymbolTable;
use yuzu_mlir::ir::value::{ValueExt, ValueId};
use yuzu_mlir::ops::yzl::{CallOp, ConstOp, FnOp, YzlOp};
use yuzu_mlir::types::{ListType, ParamType, QueryType};

use crate::operators::Operator;
use crate::written_name;

pub fn inline_calls(context: &Context, module: &mut Module) {
    let rewriter = IrRewriter::new(context);
    let rewriter = rewriter.as_rewriter_base();
    let symbols = SymbolTable::new(module);

    let mut calls = Vec::new();
    collect_calls(module.body(), &mut calls);
    let mut walked = FxHashMap::default();
    for &call in &calls {
        if let Some(closing) = find_cycle(callee(call), &symbols, &mut walked) {
            report_cycle(closing);
            return;
        }
    }

    // The module is walked once. An expansion hands back the calls it
    // copied in, so a chain of expansions never walks the module again.
    let mut pending = VecDeque::from(calls);
    while let Some(call) = pending.pop_front() {
        let Some(copied) = expand(context, &rewriter, call, &symbols) else {
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
                collect_region_calls(op, out);
            }
            Some(YzlOp::Fn(_) | YzlOp::Trait(_) | YzlOp::Impl(_)) => {}
            Some(YzlOp::Call(_)) => {
                if expands(op) {
                    out.push(op);
                }
            }
            _ => collect_region_calls(op, out),
        }
    }
}

fn collect_region_calls<'c, 'a>(op: OperationRef<'c, 'a>, out: &mut Vec<OperationRef<'c, 'a>>) {
    for region in op.regions() {
        for inner in region.blocks() {
            collect_calls(inner, out);
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

fn callee<'c>(call: OperationRef<'c, '_>) -> &'c str {
    let Some(YzlOp::Call(site)) = call.as_yzl() else {
        unreachable!("only calls are collected");
    };
    site.callee().value()
}

/// The call that closes a cycle through `symbol`, when the declarations it
/// calls reach back to one still being walked. `walked` holds each symbol
/// seen: `true` while its calls are being walked, `false` once they are.
fn find_cycle<'c, 'a>(
    symbol: &'c str,
    symbols: &SymbolTable<'c, 'a>,
    walked: &mut FxHashMap<&'c str, bool>,
) -> Option<OperationRef<'c, 'a>> {
    if walked.contains_key(symbol) {
        return None;
    }
    walked.insert(symbol, true);
    let mut calls = Vec::new();
    if let Some(declaration) = symbols.lookup(symbol) {
        collect_region_calls(declaration, &mut calls);
    }
    for call in calls {
        let target = callee(call);
        match walked.get(target) {
            Some(true) => return Some(call),
            Some(false) => {}
            None => {
                if let Some(closing) = find_cycle(target, symbols, walked) {
                    return Some(closing);
                }
            }
        }
    }
    walked.insert(symbol, false);
    None
}

fn report_cycle(call: OperationRef) {
    let name = written_name(callee(call));
    emit_error(
        call.location(),
        &format!("`{name}` calls itself here, so expanding it would not end"),
    );
}

/// Replaces a call with a copy of its body, and returns the calls the copy
/// holds, which need expanding in turn. The lowering and inference settled
/// what a call names and takes, so only what a program can still get wrong
/// is reported.
fn expand<'c, 'a>(
    context: &'c Context,
    rewriter: &'a RewriterBase<'c, 'a>,
    call: OperationRef<'c, 'a>,
    symbols: &SymbolTable<'c, '_>,
) -> Option<Vec<OperationRef<'c, 'a>>> {
    let Some(YzlOp::Call(site)) = call.as_yzl() else {
        unreachable!("only calls are expanded");
    };

    let callee = site.callee().value();
    let declaration = symbols
        .lookup(callee)
        .unwrap_or_else(|| panic!("the resolved call `{callee}` names a declaration"));

    let (body, types) = match declaration.as_yzl() {
        Some(YzlOp::Fn(function)) => (
            function.body().first_block(),
            type_arguments(function, site),
        ),
        Some(YzlOp::Const(binding)) => (binding.body().first_block(), FxHashMap::default()),
        _ => panic!("`{callee}` is called as a function and declared as something else"),
    };
    let body = body.unwrap_or_else(|| panic!("`{callee}` has a body to expand"));

    let arguments: Vec<Value> = call.operands().collect();
    assert_eq!(
        body.argument_count(),
        arguments.len(),
        "the call to `{callee}` passes as many arguments as it takes"
    );

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
                    &format!(
                        "`{}` holds a query, and a call to it cannot be expanded yet",
                        written_name(callee)
                    ),
                );
            }

            let copied = copy(context, rewriter, op, &values, &types);
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
            &format!(
                "`{}` does not return a value to use here",
                written_name(callee)
            ),
        )
    })?;
    rewriter.replace_all_op_uses_with_values(call, &[returned]);
    rewriter.erase_op(call);
    Some(copied_calls)
}

/// A copy of an op before the rewriter's insertion point: its operands
/// through `values`, and its types with `types` put for the type
/// parameters they name.
pub(crate) fn copy<'c, 'a>(
    context: &'c Context,
    rewriter: &'a RewriterBase<'c, 'a>,
    op: OperationRef<'c, '_>,
    values: &FxHashMap<ValueId, Value<'c, 'a>>,
    types: &FxHashMap<&str, Type<'c>>,
) -> OperationRef<'c, 'a> {
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
            substitute(context, ty, types)
        })
        .collect();
    let attributes: Vec<(Identifier<'c>, Attribute<'c>)> = (0..op.attribute_count())
        .map(|index| {
            let (name, attribute) = op
                .attribute_at(index)
                .expect("the attribute index is in range");
            (name, substitute_attribute(context, attribute, types))
        })
        .collect();

    let name = op.name();
    let name = name.as_string_ref().as_str().expect("op names are utf-8");
    let built = OperationBuilder::new(name, op.location())
        .add_operands(&operands)
        .add_results(&results)
        .add_attributes(&attributes)
        .build()
        .unwrap_or_else(|_| panic!("a copy of a verified `{name}` builds"));

    rewriter.insert(built)
}

/// An attribute with `types` put for the type parameters it names, inside
/// an array of types as well, as a call's `type_args` holds them.
fn substitute_attribute<'c>(
    context: &'c Context,
    attribute: Attribute<'c>,
    types: &FxHashMap<&str, Type<'c>>,
) -> Attribute<'c> {
    if let Ok(stamp) = TypeAttribute::try_from(attribute) {
        return TypeAttribute::new(substitute(context, stamp.value(), types)).into();
    }
    if let Ok(array) = ArrayAttribute::try_from(attribute) {
        let elements: Vec<Attribute<'c>> = (0..array.len())
            .map(|index| {
                let element = array.element(index).expect("the element index is in range");
                substitute_attribute(context, element, types)
            })
            .collect();
        return ArrayAttribute::new(context, &elements).into();
    }
    attribute
}

fn type_arguments<'c>(
    function: FnOp<'c, '_>,
    site: CallOp<'c, '_>,
) -> FxHashMap<&'c str, Type<'c>> {
    let Some(parameters) = function.type_params() else {
        return FxHashMap::default();
    };

    let parameters: Vec<&str> = parameters.strings().collect();
    let arguments: Vec<Type<'c>> = site
        .type_args()
        .expect("inference settles each generic call's types")
        .types()
        .collect();
    assert_eq!(
        arguments.len(),
        parameters.len(),
        "a generic call has a type for each type parameter"
    );

    parameters.into_iter().zip(arguments).collect()
}

/// A type with `types` put for the type parameters it names, inside a list
/// as well.
pub(crate) fn substitute<'c>(
    context: &'c Context,
    ty: Type<'c>,
    types: &FxHashMap<&str, Type<'c>>,
) -> Type<'c> {
    if let Some(param) = ParamType::from_type(ty) {
        return types.get(param.name()).copied().unwrap_or(ty);
    }
    if let Some(list) = ListType::from_type(ty) {
        return ListType::new(context, substitute(context, list.inner(), types)).into();
    }
    ty
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
        .operation()
        .body_terminator()
        .and_then(|yielded| yielded.try_first_operand())
        .is_some_and(|value| value.r#type() == QueryType::get(context))
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
    fn an_external_call_is_left_alone() {
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
                    %2 = yzr.agg "count"(%arg0) : (!yz.int64) -> !yz.int64
                    yzr.yield %2 : !yz.int64
                  } : !yz.struct<@Row> -> !yz.struct<@row>
                  yzr.output %1 : !yz.struct<@row>
                }
            "#]],
        );
    }

    #[test]
    fn a_function_that_calls_itself_is_reported() {
        check_simplified(
            r"
struct Row { a: int64 }
table t = Row

def forever(x: int64) -> int64 { return forever(x) }

from t
|> select forever(a) as f
",
            &expect![[r"
                error: `forever` calls itself here, so expanding it would not end
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
    fn a_generic_call_inside_a_generic_body_takes_its_types() {
        check_simplified(
            "struct Row { a: int64 }\ntable t = Row\ndef dbl[T](x: T) -> T { return x + x }\ndef quad[T](x: T) -> T { return dbl(x) + dbl(x) }\nfrom t |> select quad(a) as q\n",
            &expect![[r#"
                module {
                  yz.struct @Row ["a"] : [!yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  yz.struct @row ["q"] : [!yz.int64]
                  %1 = yzr.project %0 {
                  ^bb0(%arg0: !yz.int64):
                    %2 = yz.add %arg0, %arg0 : !yz.int64, !yz.int64 -> !yz.int64
                    %3 = yz.add %2, %2 : !yz.int64, !yz.int64 -> !yz.int64
                    yzr.yield %3 : !yz.int64
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
