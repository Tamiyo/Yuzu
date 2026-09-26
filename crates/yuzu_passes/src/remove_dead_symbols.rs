use melior::Context;
use melior::ir::Module;
use melior::pass::transform;

pub fn remove_dead_symbols(context: &Context, module: &mut Module) {
    let passes = crate::pass_manager(context);
    passes.add_pass(transform::create_symbol_dce());

    passes
        .run(module)
        .expect("symbol DCE runs on any module the lowering builds");
}

#[cfg(test)]
mod tests {
    use crate::test_support::lower;

    #[test]
    fn a_private_declaration_nothing_reaches_is_removed_and_an_export_stays() {
        let context = yuzu_mlir::context();
        let mut lowered = lower(
            &context,
            &[
                (
                    "helpers.yz",
                    Some("helpers"),
                    "pub def used(x: int64) -> int64 { return helper(x) }\n\
                     pub def exported(x: int64) -> int64 { return x + 1 }\n\
                     def helper(x: int64) -> int64 { return x * 2 }\n\
                     def unused(x: int64) -> int64 { return x + 99 }\n",
                ),
                (
                    "main.yz",
                    None,
                    "from helpers import used\n\nstruct Row { a: int64 }\ntable t = Row\n\nfrom t |> select used(a) as v\n",
                ),
            ],
        );
        super::remove_dead_symbols(&context, &mut lowered.module);
        let module = lowered.module.as_operation().to_string();

        for kept in ["@helpers.used", "@helpers.exported", "@helpers.helper"] {
            assert!(module.contains(kept), "{kept} stays:\n{module}");
        }
        assert!(
            !module.contains("@helpers.unused"),
            "unused goes:\n{module}"
        );
    }
}
