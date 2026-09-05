use melior::Context;

pub mod legality;

/// A context with every Yuzu dialect registered and loaded.
pub fn context() -> Context {
    let context = Context::new();
    yuzu_ir_sys::register_all(context.to_raw());
    context
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use melior::ir::{
        BlockLike, Location, Module, Type,
        operation::{OperationBuilder, OperationLike},
    };

    #[test]
    fn parses_and_prints_yzir_ops() {
        let context = super::context();
        let module = Module::parse(
            &context,
            r#"
module {
  %0 = yzir.const 3
  %1 = yzir.const 4
  %2 = yzir.add %0, %1
  %3 = yzir.mul %2, %0
}
"#,
        )
        .expect("the yzir dialect parses its own syntax");
        expect![[r#"
            module {
              %0 = yzir.const 3
              %1 = yzir.const 4
              %2 = yzir.add %0, %1
              %3 = yzir.mul %2, %0
            }
        "#]]
        .assert_eq(&module.as_operation().to_string());
    }

    #[test]
    fn builds_yzir_ops_programmatically() {
        let context = super::context();
        let location = Location::unknown(&context);
        let int64 = Type::parse(&context, "!yzir.int64").expect("!yzir.int64 parses");

        let module = Module::new(location);
        let block = module.body();
        let three = block.append_operation(
            OperationBuilder::new("yzir.const", location)
                .add_attributes(&[(
                    melior::ir::Identifier::new(&context, "value"),
                    melior::ir::attribute::IntegerAttribute::new(
                        melior::ir::r#type::IntegerType::new(&context, 64).into(),
                        3,
                    )
                    .into(),
                )])
                .add_results(&[int64])
                .build()
                .expect("yzir.const builds"),
        );
        block.append_operation(
            OperationBuilder::new("yzir.add", location)
                .add_operands(&[
                    three.result(0).unwrap().into(),
                    three.result(0).unwrap().into(),
                ])
                .add_results(&[int64])
                .build()
                .expect("yzir.add builds"),
        );

        assert!(module.as_operation().verify());
        expect![[r#"
            module {
              %0 = yzir.const 3
              %1 = yzir.add %0, %0
            }
        "#]]
        .assert_eq(&module.as_operation().to_string());
    }

    #[test]
    fn the_canonicalizer_folds_yzir_constants() {
        let context = super::context();
        let mut module = Module::parse(
            &context,
            r#"
module {
  %t = builtin.unrealized_conversion_cast to !yzr.rel
  %g = yzr.aggregate %t keys [0] {
  ^bb0(%a: !yzir.int64, %b: !yzir.int64):
    %c3 = yzir.const 3
    %c4 = yzir.const 4
    %s = yzir.add %c3, %c4
    %m = yzr.agg "sum", %b : !yzir.int64 -> !yzir.int64
    %v = yzir.mul %m, %s
    yzr.yield %v : !yzir.int64
  }
}
"#,
        )
        .expect("the aggregate parses and verifies");

        let pass_manager = melior::pass::PassManager::new(&context);
        pass_manager.add_pass(melior::pass::transform::create_canonicalizer());
        pass_manager
            .run(&mut module)
            .expect("canonicalization succeeds");

        expect![[r#"
            module {
              %0 = yzir.const 7
              %1 = unrealized_conversion_cast to !yzr.rel
              %2 = yzr.aggregate %1 keys [0] {
              ^bb0(%arg0: !yzir.int64, %arg1: !yzir.int64):
                %3 = yzr.agg "sum", %arg1 : !yzir.int64 -> !yzir.int64
                %4 = yzir.mul %3, %0
                yzr.yield %4 : !yzir.int64
              }
            }
        "#]]
        .assert_eq(&module.as_operation().to_string());
    }

    #[test]
    fn the_grouping_verifier_accepts_an_aggregated_key() {
        let context = super::context();
        assert!(
            Module::parse(
                &context,
                r#"
module {
  %t = builtin.unrealized_conversion_cast to !yzr.rel
  %g = yzr.aggregate %t keys [0] {
  ^bb0(%a: !yzir.int64):
    %m = yzr.agg "sum", %a : !yzir.int64 -> !yzir.int64
    yzr.yield %m : !yzir.int64
  }
}
"#,
            )
            .is_some(),
            "aggregating a key column is legal"
        );
    }

    #[test]
    fn the_grouping_verifier_rejects_a_smuggled_column() {
        let context = super::context();
        assert!(
            Module::parse(
                &context,
                r#"
module {
  %t = builtin.unrealized_conversion_cast to !yzr.rel
  %g = yzr.aggregate %t keys [0] {
  ^bb0(%a: !yzir.int64, %b: !yzir.int64):
    yzr.yield %b : !yzir.int64
  }
}
"#,
            )
            .is_none(),
            "yielding an ungrouped column must fail verification"
        );
    }

    #[test]
    fn the_grouping_verifier_rejects_a_laundered_column() {
        let context = super::context();
        assert!(
            Module::parse(
                &context,
                r#"
module {
  %t = builtin.unrealized_conversion_cast to !yzr.rel
  %g = yzr.aggregate %t keys [0] {
  ^bb0(%a: !yzir.int64, %b: !yzir.int64):
    %c = yzir.const 1
    %x = yzir.add %b, %c
    yzr.yield %x : !yzir.int64
  }
}
"#,
            )
            .is_none(),
            "an ungrouped column laundered through arithmetic must still fail"
        );
    }

    #[test]
    fn the_grouping_verifier_rejects_a_nested_aggregate() {
        let context = super::context();
        assert!(
            Module::parse(
                &context,
                r#"
module {
  %t = builtin.unrealized_conversion_cast to !yzr.rel
  %g = yzr.aggregate %t keys [0] {
  ^bb0(%a: !yzir.int64, %b: !yzir.int64):
    %m = yzr.agg "max", %b : !yzir.int64 -> !yzir.int64
    %n = yzr.agg "sum", %m : !yzir.int64 -> !yzir.int64
    yzr.yield %n : !yzir.int64
  }
}
"#,
            )
            .is_none(),
            "aggregating a measure must fail verification"
        );
    }

    #[test]
    fn yzl_names_and_vars_round_trip() {
        let context = super::context();
        let module = Module::parse(
            &context,
            r#"
module {
  %0 = yzl.name "a" : !yzl.var
  %1 = yzl.name "employees" : !yzl.var
}
"#,
        )
        .expect("the yzl dialect parses its own syntax");
        expect![[r#"
            module {
              %0 = yzl.name "a" : !yzl.var
              %1 = yzl.name "employees" : !yzl.var
            }
        "#]]
        .assert_eq(&module.as_operation().to_string());
    }

    #[test]
    fn the_legality_pass_rejects_an_unsupported_op() {
        let context = super::context();
        let source = r#"
module {
  %0 = yzir.const 2
  %1 = yzir.mul %0, %0
}
"#;

        let mut module = Module::parse(&context, source).unwrap();
        let pass_manager = melior::pass::PassManager::new(&context);
        pass_manager.add_pass(crate::legality::create(vec!["yzir.mul".to_string()]));
        assert!(
            pass_manager.run(&mut module).is_err(),
            "a target without mul must reject the plan"
        );

        let mut module = Module::parse(&context, source).unwrap();
        let pass_manager = melior::pass::PassManager::new(&context);
        pass_manager.add_pass(crate::legality::create(vec!["yzir.shift_left".to_string()]));
        assert!(
            pass_manager.run(&mut module).is_ok(),
            "a target that supports everything present must accept the plan"
        );
    }

    #[test]
    fn the_verifier_rejects_a_mistyped_operand() {
        let context = super::context();
        let module = Module::parse(
            &context,
            r#"
module {
  %0 = "yzir.wrong"() : () -> !yzir.bool
  %1 = yzir.add %0, %0
}
"#,
        );
        assert!(module.is_none(), "yzir.add over !yzir.bool must not parse");
    }
}
