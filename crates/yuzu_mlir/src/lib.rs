use melior::Context;

pub mod legality;
pub mod ods;

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

    fn parse<'c>(context: &'c super::Context, source: &str) -> Option<Module<'c>> {
        Module::parse(context, source)
    }

    #[test]
    fn parses_and_prints_yz_ops() {
        let context = super::context();
        let module = parse(
            &context,
            r#"
module {
  %0 = yz.const 3
  %1 = yz.const 4
  %2 = yz.add %0, %1
  %3 = yz.sub %2, %0
  %4 = yz.mul %3, %1
  %5 = yz.div %4, %1
  %6 = yz.mod %5, %0
  %7 = yz.neg %6
  %8 = yz.cmp "gt", %7, %0
  %9 = yz.not %8
  %10 = yz.and %8, %9
  %11 = yz.or %8, %9
  %12 = yz.const_float 1.500000e+00
  %13 = yz.const_bool true
  %14 = yz.const_str "hello"
}
"#,
        )
        .expect("the yz dialect parses its own syntax");
        expect![[r#"
            module {
              %0 = yz.const 3
              %1 = yz.const 4
              %2 = yz.add %0, %1
              %3 = yz.sub %2, %0
              %4 = yz.mul %3, %1
              %5 = yz.div %4, %1
              %6 = yz.mod %5, %0
              %7 = yz.neg %6
              %8 = yz.cmp "gt", %7, %0
              %9 = yz.not %8
              %10 = yz.and %8, %9
              %11 = yz.or %8, %9
              %12 = yz.const_float 1.500000e+00
              %13 = yz.const_bool true
              %14 = yz.const_str "hello"
            }
        "#]]
        .assert_eq(&module.as_operation().to_string());
    }

    #[test]
    fn builds_yz_ops_programmatically() {
        let context = super::context();
        let location = Location::unknown(&context);
        let int64 = Type::parse(&context, "!yz.int64").expect("!yz.int64 parses");

        let module = Module::new(location);
        let block = module.body();
        let three = block.append_operation(
            OperationBuilder::new("yz.const", location)
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
                .expect("yz.const builds"),
        );
        block.append_operation(
            OperationBuilder::new("yz.add", location)
                .add_operands(&[
                    three.result(0).unwrap().into(),
                    three.result(0).unwrap().into(),
                ])
                .add_results(&[int64])
                .build()
                .expect("yz.add builds"),
        );

        assert!(module.as_operation().verify());
        expect![[r#"
            module {
              %0 = yz.const 3
              %1 = yz.add %0, %0
            }
        "#]]
        .assert_eq(&module.as_operation().to_string());
    }

    #[test]
    fn a_full_pipeline_round_trips() {
        let context = super::context();
        let module = parse(
            &context,
            r#"
module {
  %t = yzr.table @t : !yzr.rel<a: !yz.int64, b: !yz.int64>
  %w = yzr.filter %t : !yzr.rel<a: !yz.int64, b: !yz.int64> {
  ^bb0(%a: !yz.int64, %b: !yz.int64):
    %c10 = yz.const 10
    %p = yz.cmp "gt", %a, %c10
    yzr.yield %p : !yz.bool
  }
  %e = yzr.extend %w {
  ^bb0(%a: !yz.int64, %b: !yz.int64):
    %c3 = yz.const 3
    %0 = yz.mul %a, %c3
    %1 = yz.add %0, %b
    yzr.yield %1 : !yz.int64
  } : !yzr.rel<a: !yz.int64, b: !yz.int64> -> !yzr.rel<a: !yz.int64, b: !yz.int64, e: !yz.int64>
  %g = yzr.aggregate %e keys [1] {
  ^bb0(%a: !yz.int64, %b: !yz.int64, %e0: !yz.int64):
    %m = yzr.agg "sum", %e0 : !yz.int64 -> !yz.int64
    yzr.yield %m : !yz.int64
  } : !yzr.rel<a: !yz.int64, b: !yz.int64, e: !yz.int64> -> !yzr.rel<b: !yz.int64, s: !yz.int64>
  %l = yzr.limit %g, 10 : !yzr.rel<b: !yz.int64, s: !yz.int64>
}
"#,
        )
        .expect("the whole pipeline parses and verifies");
        expect![[r#"
            module {
              %0 = yzr.table @t : <a: !yz.int64, b: !yz.int64>
              %1 = yzr.filter %0 : <a: !yz.int64, b: !yz.int64> {
              ^bb0(%arg0: !yz.int64, %arg1: !yz.int64):
                %5 = yz.const 10
                %6 = yz.cmp "gt", %arg0, %5
                yzr.yield %6 : !yz.bool
              }
              %2 = yzr.extend %1 {
              ^bb0(%arg0: !yz.int64, %arg1: !yz.int64):
                %5 = yz.const 3
                %6 = yz.mul %arg0, %5
                %7 = yz.add %6, %arg1
                yzr.yield %7 : !yz.int64
              } : <a: !yz.int64, b: !yz.int64> -> <a: !yz.int64, b: !yz.int64, e: !yz.int64>
              %3 = yzr.aggregate %2 keys [1] {
              ^bb0(%arg0: !yz.int64, %arg1: !yz.int64, %arg2: !yz.int64):
                %5 = yzr.agg "sum", %arg2 : !yz.int64 -> !yz.int64
                yzr.yield %5 : !yz.int64
              } : <a: !yz.int64, b: !yz.int64, e: !yz.int64> -> <b: !yz.int64, s: !yz.int64>
              %4 = yzr.limit %3, 10 : <b: !yz.int64, s: !yz.int64>
            }
        "#]]
        .assert_eq(&module.as_operation().to_string());

        let printed = module.as_operation().to_string();
        let reparsed = parse(&context, &printed).expect("the printed form parses back");
        assert_eq!(
            printed,
            reparsed.as_operation().to_string(),
            "print -> parse -> print reaches a fixed point"
        );
    }

    #[test]
    fn builds_a_stage_with_generated_constructors() {
        use melior::ir::{
            Block, Region, RegionLike,
            attribute::{FlatSymbolRefAttribute, IntegerAttribute, StringAttribute},
            r#type::IntegerType,
        };

        use crate::ods::{yz, yzl};

        let context = super::context();
        let loc = Location::unknown(&context);
        let query = Type::parse(&context, "!yzl.query").unwrap();
        let int64 = Type::parse(&context, "!yz.int64").unwrap();
        let boolean = Type::parse(&context, "!yz.bool").unwrap();
        let i64 = IntegerType::new(&context, 64).into();

        let module = Module::new(loc);
        let top = module.body();

        let from = top.append_operation(
            yzl::from(
                &context,
                query,
                FlatSymbolRefAttribute::new(&context, "t"),
                loc,
            )
            .into(),
        );

        let region = Region::new();
        let body = region.append_block(Block::new(&[]));
        let a = body.append_operation(
            yzl::_name(&context, int64, StringAttribute::new(&context, "a"), loc).into(),
        );
        let ten = body.append_operation(
            yz::r#const(&context, int64, IntegerAttribute::new(i64, 10), loc).into(),
        );
        let cmp = body.append_operation(
            yz::cmp(
                &context,
                boolean,
                a.result(0).unwrap().into(),
                ten.result(0).unwrap().into(),
                StringAttribute::new(&context, "gt"),
                loc,
            )
            .into(),
        );
        body.append_operation(yzl::r#yield(&context, &[cmp.result(0).unwrap().into()], loc).into());

        top.append_operation(
            yzl::r#where(&context, query, from.result(0).unwrap().into(), region, loc).into(),
        );

        assert!(module.as_operation().verify());
        expect![[r#"
            module {
              %0 = yzl.from @t
              %1 = yzl.where %0 {
                %2 = yzl.name "a" : !yz.int64
                %3 = yz.const 10
                %4 = yz.cmp "gt", %2, %3
                yzl.yield %4 : !yz.bool
              }
            }
        "#]]
        .assert_eq(&module.as_operation().to_string());
    }

    #[test]
    fn the_canonicalizer_folds_across_a_stage_region() {
        let context = super::context();
        let mut module = parse(
            &context,
            r#"
module {
  %t = yzr.table @t : !yzr.rel<a: !yz.int64>
  %w = yzr.filter %t : !yzr.rel<a: !yz.int64> {
  ^bb0(%a: !yz.int64):
    %c3 = yz.const 3
    %c4 = yz.const 4
    %p = yz.cmp "gt", %c4, %c3
    %q = yz.cmp "gt", %a, %c3
    %r = yz.and %p, %q
    yzr.yield %r : !yz.bool
  }
  %d = yzr.filter %w : !yzr.rel<a: !yz.int64> {
  ^bb0(%a: !yz.int64):
    %c0 = yz.const 0
    %c1 = yz.const 1
    %z = yz.div %c1, %c0
    %p = yz.cmp "eq", %z, %c1
    yzr.yield %p : !yz.bool
  }
}
"#,
        )
        .expect("the filters parse and verify");

        let pass_manager = melior::pass::PassManager::new(&context);
        pass_manager.add_pass(melior::pass::transform::create_canonicalizer());
        pass_manager
            .run(&mut module)
            .expect("canonicalization succeeds");

        expect![[r#"
            module {
              %0 = yz.const_bool true
              %1 = yz.const 1
              %2 = yz.const 0
              %3 = yz.const 3
              %4 = yzr.table @t : <a: !yz.int64>
              %5 = yzr.filter %4 : <a: !yz.int64> {
              ^bb0(%arg0: !yz.int64):
                %7 = yz.cmp "gt", %arg0, %3
                %8 = yz.and %0, %7
                yzr.yield %8 : !yz.bool
              }
              %6 = yzr.filter %5 : <a: !yz.int64> {
              ^bb0(%arg0: !yz.int64):
                %7 = yz.div %1, %2
                %8 = yz.cmp "eq", %7, %1
                yzr.yield %8 : !yz.bool
              }
            }
        "#]]
        .assert_eq(&module.as_operation().to_string());
    }

    #[test]
    fn yzl_names_and_vars_round_trip() {
        let context = super::context();
        let module = parse(
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
  %0 = yz.const 2
  %1 = yz.mul %0, %0
}
"#;

        let mut module = parse(&context, source).unwrap();
        let pass_manager = melior::pass::PassManager::new(&context);
        pass_manager.add_pass(crate::legality::create(vec!["yz.mul".to_string()]));
        assert!(
            pass_manager.run(&mut module).is_err(),
            "a target without mul must reject the plan"
        );

        let mut module = parse(&context, source).unwrap();
        let pass_manager = melior::pass::PassManager::new(&context);
        pass_manager.add_pass(crate::legality::create(vec!["yz.shift_left".to_string()]));
        assert!(
            pass_manager.run(&mut module).is_ok(),
            "a target that supports everything present must accept the plan"
        );
    }

    #[test]
    fn the_verifier_rejects_a_mistyped_operand() {
        let context = super::context();
        let module = parse(
            &context,
            r#"
module {
  %0 = "yz.wrong"() : () -> !yz.bool
  %1 = yz.add %0, %0
}
"#,
        );
        assert!(module.is_none(), "yz.add over !yz.bool must not parse");
    }
}
