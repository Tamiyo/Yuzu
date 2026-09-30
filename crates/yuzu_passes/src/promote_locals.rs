//! MLIR's `mem2reg`. Each place a function body declares for a local
//! variable becomes the SSA values stored to it, so no later pass sees one.

use melior::Context;
use melior::ir::Module;
use melior::pass::transform;

/// Promotes each local variable's place to the SSA values stored to it.
///
/// # Panics
///
/// Panics if `mem2reg` fails. It does not fail on a module that the
/// lowering builds.
pub fn promote_locals(context: &Context, module: &mut Module) {
    let passes = crate::pass_manager(context);
    passes.add_pass(transform::create_mem_2_reg());

    passes
        .run(module)
        .expect("mem2reg runs on any module the lowering builds");
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use melior::ir::Module;

    #[test]
    fn a_place_becomes_the_values_stored_to_it() {
        let context = yuzu_mlir::context();
        let mut module = Module::parse(
            &context,
            r#"
module {
  yzl.fn @f params ["x"] (!yz.int64) -> !yz.int64 {
  ^bb0(%arg: !yz.int64):
    %x = yzl.local "x" param : !yzl.ref<!yz.int64>
    %y = yzl.local "y" mut : !yzl.ref<!yz.int64>
    yzl.store %x, %arg : !yzl.ref<!yz.int64>, !yz.int64
    %a = yzl.load %x : !yzl.ref<!yz.int64> -> !yz.int64
    yzl.store %y, %a : !yzl.ref<!yz.int64>, !yz.int64
    %b = yzl.load %y : !yzl.ref<!yz.int64> -> !yz.int64
    %two = yz.constant_int 2
    %c = yz.mul %b, %two : !yz.int64, !yz.int64 -> !yz.int64
    yzl.store %y, %c : !yzl.ref<!yz.int64>, !yz.int64
    %d = yzl.load %y : !yzl.ref<!yz.int64> -> !yz.int64
    yzl.return %d : !yz.int64
  }
}
"#,
        )
        .expect("the module parses");

        super::promote_locals(&context, &mut module);

        expect![[r#"
            module {
              yzl.fn @f params ["x"] (!yz.int64) -> !yz.int64 {
              ^bb0(%arg0: !yz.int64):
                %0 = yz.constant_int 2
                %1 = yz.mul %arg0, %0 : !yz.int64, !yz.int64 -> !yz.int64
                yzl.return %1 : !yz.int64
              }
            }
        "#]]
        .assert_eq(&module.as_operation().to_string());
    }
}
