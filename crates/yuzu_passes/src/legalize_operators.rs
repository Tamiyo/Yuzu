//! An engine takes no `yz.rem`. Each operator the folds left behind becomes
//! a copy of the library function that implements it, with the operator's
//! types in place of the function's type parameters. The copies fold once
//! more, and the implementations go.

use melior::ir::operation::{OperationLike, OperationRef};
use melior::ir::{BlockLike, Module, RegionLike, Type, Value, ValueLike};
use melior::pass::transform;
use melior::{Context, IrRewriter, RewriterBase};
use rustc_hash::FxHashMap;
use yuzu_mlir::diagnostics::emit_error;
use yuzu_mlir::ir::block::BlockExt;
use yuzu_mlir::ir::operation::OperationExt;
use yuzu_mlir::ir::region::RegionExt;
use yuzu_mlir::ir::value::{ValueExt, ValueId};
use yuzu_mlir::{ParamType, SymbolTable};

use crate::inline_calls::copy;
use crate::operators::{Operator, is_named};

pub fn legalize_operators(context: &Context, module: &mut Module) {
    let rewriter = IrRewriter::new(context);
    let rewriter = rewriter.as_rewriter_base();
    let symbols = SymbolTable::new(module);

    let mut operators = Vec::new();
    collect_operators(module.body(), &mut operators);
    for (op, operator) in operators {
        let Some(implementation) = symbols.lookup(&operator.symbol()) else {
            emit_error(
                op.location(),
                &format!(
                    "`{}` has no implementation for this engine",
                    operator.spelling
                ),
            );
            continue;
        };

        expand(&rewriter, op, implementation, operator);
    }

    drop(symbols);
    discard_implementations(&rewriter, module);

    let passes = crate::pass_manager(context);
    passes.add_pass(transform::create_canonicalizer_pass());
    passes
        .run(module)
        .expect("canonicalization runs on any module the lowering builds");
}

/// The operators in every region, outside the implementations themselves.
fn collect_operators<'c, 'a>(
    block: melior::ir::BlockRef<'c, 'a>,
    out: &mut Vec<(OperationRef<'c, 'a>, &'static Operator)>,
) {
    for op in block.operations() {
        if is_implementation(op) {
            continue;
        }

        if let Some(operator) = Operator::of(op) {
            out.push((op, operator));
        }

        for region in op.regions() {
            for inner in region.blocks() {
                collect_operators(inner, out);
            }
        }
    }
}

/// Puts a copy of the implementation's body in place of the operator.
fn expand<'c, 'a>(
    rewriter: &'a RewriterBase<'c, 'a>,
    op: OperationRef<'c, 'a>,
    implementation: OperationRef<'c, '_>,
    operator: &Operator,
) {
    let Some(body) = implementation
        .regions()
        .next()
        .and_then(|region| region.first_block())
    else {
        emit_error(
            op.location(),
            &format!("`{}` has an implementation with no body", operator.spelling),
        );
        return;
    };

    let arguments: Vec<Value<'c, 'a>> = op.operands().collect();
    debug_assert_eq!(
        body.argument_count(),
        arguments.len(),
        "an operator takes as many operands as its implementation takes parameters"
    );

    let types = type_arguments(body, &arguments);
    let mut values: FxHashMap<ValueId, Value<'c, 'a>> = body
        .arguments()
        .map(|parameter| parameter.id())
        .zip(arguments.iter().copied())
        .collect();

    rewriter.set_insertion_point_before(op);
    let mut returned = None;
    for inner in body.operations() {
        if is_named(inner, "yz.return") {
            returned = inner
                .try_first_operand()
                .and_then(|value| values.get(&value.id()).copied());
            continue;
        }

        let Some(copied) = copy(rewriter, inner, &values, &types) else {
            emit_error(
                op.location(),
                &format!(
                    "`{}` has an implementation that did not copy",
                    operator.spelling
                ),
            );
            return;
        };

        if let (Some(result), Some(value)) = (inner.try_first_result(), copied.try_first_result()) {
            values.insert(result.id(), value);
        }
    }

    let Some(returned) = returned else {
        emit_error(
            op.location(),
            &format!(
                "`{}` has an implementation that returns nothing",
                operator.spelling
            ),
        );
        return;
    };

    rewriter.replace_all_op_uses_with_values(op, &[returned]);
    rewriter.erase_op(op);
}

/// Each type parameter, by name, as the operand in its place has it.
fn type_arguments<'c>(
    body: melior::ir::BlockRef<'c, '_>,
    arguments: &[Value<'c, '_>],
) -> FxHashMap<&'c str, Type<'c>> {
    body.arguments()
        .zip(arguments)
        .filter_map(|(parameter, argument)| {
            let param = ParamType::from_type(parameter.r#type())?;
            Some((param.name(), argument.r#type()))
        })
        .collect()
}

fn discard_implementations(rewriter: &RewriterBase, module: &Module) {
    let implementations: Vec<_> = module
        .body()
        .operations()
        .filter(|op| is_implementation(*op))
        .collect();
    for implementation in implementations {
        rewriter.erase_op(implementation);
    }
}

fn is_implementation(op: OperationRef) -> bool {
    is_named(op, "yz.func")
        && op
            .text_attribute("sym_name")
            .is_some_and(|name| Operator::implemented_by(name).is_some())
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use expect_test::expect;
    use melior::ir::Module;

    const PROJECTION: &str = r#"
  yz.struct @Row ["a", "r"] : [!yz.int64, !yz.float64]
  yz.struct @Out ["m", "n"] : [!yz.int64, !yz.float64]
  %0 = yzr.table @t : !yz.struct<@Row>
  %1 = yzr.project %0 {
  ^bb0(%a: !yz.int64, %r: !yz.float64):
    %three = yz.constant_int 3
    %m = yz.rem %a, %three : !yz.int64, !yz.int64 -> !yz.int64
    %two = yz.constant_float 2.0
    %n = yz.rem %r, %two : !yz.float64, !yz.float64 -> !yz.float64
    yzr.yield %m, %n : !yz.int64, !yz.float64
  } : !yz.struct<@Row> -> !yz.struct<@Out>
  yzr.output %1 : !yz.struct<@Out>
"#;

    fn legalized(implementation: &str) -> String {
        let context = yuzu_mlir::context();
        let source = format!("module {{\n{implementation}{PROJECTION}}}");
        let mut module = Module::parse(&context, &source).expect("the module parses");
        super::legalize_operators(&context, &mut module);
        module.as_operation().to_string()
    }

    #[test]
    fn each_operator_becomes_its_implementation_at_its_own_types() {
        expect![[r#"
            module {
              yz.struct @Row ["a", "r"] : [!yz.int64, !yz.float64]
              yz.struct @Out ["m", "n"] : [!yz.int64, !yz.float64]
              %0 = yzr.table @t : !yz.struct<@Row>
              %1 = yzr.project %0 {
              ^bb0(%arg0: !yz.int64, %arg1: !yz.float64):
                %2 = yz.constant_float 2.000000e+00
                %3 = yz.constant_int 3
                %4 = yz.extern_call "modulus"(%arg0, %3) : (!yz.int64, !yz.int64) -> !yz.int64
                %5 = yz.extern_call "modulus"(%arg1, %2) : (!yz.float64, !yz.float64) -> !yz.float64
                yzr.yield %4, %5 : !yz.int64, !yz.float64
              } : !yz.struct<@Row> -> !yz.struct<@Out>
              yzr.output %1 : !yz.struct<@Out>
            }
        "#]]
        .assert_eq(&legalized(
            r#"
  yz.func @yuzu.std.ops.modulo (!yzl.param<"T">, !yzl.param<"T">) -> !yzl.param<"T"> {
  ^bb0(%a: !yzl.param<"T">, %b: !yzl.param<"T">):
    %r = yz.extern_call "modulus"(%a, %b) : (!yzl.param<"T">, !yzl.param<"T">) -> !yzl.param<"T">
    yz.return %r : !yzl.param<"T">
  }
"#,
        ));
    }

    #[test]
    fn an_operator_with_no_implementation_is_reported() {
        let context = yuzu_mlir::context();
        let source = format!("module {{\n{PROJECTION}}}");
        let mut module = Module::parse(&context, &source).expect("the module parses");
        let reported = Rc::new(RefCell::new(Vec::new()));
        let sink = reported.clone();
        let handler = context.attach_diagnostic_handler(move |diagnostic| {
            sink.borrow_mut().push(diagnostic.to_string());
            true
        });
        super::legalize_operators(&context, &mut module);
        context.detach_diagnostic_handler(handler);

        assert_eq!(
            *reported.borrow(),
            ["`%` has no implementation for this engine"; 2]
        );
    }
}
