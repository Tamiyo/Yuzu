use melior::Context;

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
