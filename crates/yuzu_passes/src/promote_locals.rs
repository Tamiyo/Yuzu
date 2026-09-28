//! MLIR's `mem2reg`. Each place a function body declares for a local
//! variable becomes the SSA values stored to it, so no later pass sees one.

use melior::Context;
use melior::ir::Module;
use melior::pass::transform;

/// Promotes each local variable's place to the SSA values stored to it.
///
/// # Panics
///
/// Panics if `mem2reg` fails, which it does not on any module the lowering
/// builds.
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
  ^bb0(%arg: !yzl.unresolved):
    %x = yzl.local "x" param
    %y = yzl.local "y" mut
    yzl.store %x, %arg : !yzl.unresolved
    %a = yzl.load %x : !yzl.unresolved
    yzl.store %y, %a : !yzl.unresolved
    %b = yzl.load %y : !yzl.unresolved
    %two = yz.constant_int 2
    %c = yz.mul %b, %two : !yzl.unresolved, !yz.int64 -> !yzl.unresolved
    yzl.store %y, %c : !yzl.unresolved
    %d = yzl.load %y : !yzl.unresolved
    yzl.return %d : !yzl.unresolved
  }
}
"#,
        )
        .expect("the module parses");

        super::promote_locals(&context, &mut module);

        expect![[r#"
            module {
              yzl.fn @f params ["x"] (!yz.int64) -> !yz.int64 {
              ^bb0(%arg0: !yzl.unresolved):
                %0 = yz.constant_int 2
                %1 = yz.mul %arg0, %0 : !yzl.unresolved, !yz.int64 -> !yzl.unresolved
                yzl.return %1 : !yzl.unresolved
              }
            }
        "#]]
        .assert_eq(&module.as_operation().to_string());
    }
}
