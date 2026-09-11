//! The dialects, exercised through the public surface: textual round
//! trips, programmatic construction, folding, and the generated
//! constructors.

use expect_test::expect;
use melior::ir::{
    BlockLike, Location, Module, Type,
    operation::{OperationBuilder, OperationLike},
};

fn parse<'c>(context: &'c melior::Context, source: &str) -> Option<Module<'c>> {
    Module::parse(context, source)
}

#[test]
fn parses_and_prints_yz_ops() {
    let context = yuzu_mlir::context();
    let module = parse(
        &context,
        r#"
module {
  %0 = yz.constant_int 3
  %1 = yz.constant_int 4
  %2 = yz.add %0, %1 : !yz.int64, !yz.int64 -> !yz.int64
  %3 = yz.sub %2, %0 : !yz.int64, !yz.int64 -> !yz.int64
  %4 = yz.mul %3, %1 : !yz.int64, !yz.int64 -> !yz.int64
  %5 = yz.div %4, %1 : !yz.int64, !yz.int64 -> !yz.int64
  %6 = yz.rem %5, %0 : !yz.int64, !yz.int64 -> !yz.int64
  %7 = yz.neg %6 : !yz.int64 -> !yz.int64
  %8 = yz.cmp "gt", %7, %0 : !yz.int64, !yz.int64 -> !yz.bool
  %9 = yz.not %8 : !yz.bool -> !yz.bool
  %10 = yz.and %8, %9 : !yz.bool, !yz.bool -> !yz.bool
  %11 = yz.or %8, %9 : !yz.bool, !yz.bool -> !yz.bool
  %12 = yz.constant_float 1.500000e+00
  %f = yz.add %12, %12 : !yz.float64, !yz.float64 -> !yz.float64
  %13 = yz.constant_bool true
  %14 = yz.constant_str "hello"
}
"#,
    )
    .expect("the yz dialect parses its own syntax");
    expect![[r#"
        module {
          %0 = yz.constant_int 3
          %1 = yz.constant_int 4
          %2 = yz.add %0, %1 : !yz.int64, !yz.int64 -> !yz.int64
          %3 = yz.sub %2, %0 : !yz.int64, !yz.int64 -> !yz.int64
          %4 = yz.mul %3, %1 : !yz.int64, !yz.int64 -> !yz.int64
          %5 = yz.div %4, %1 : !yz.int64, !yz.int64 -> !yz.int64
          %6 = yz.rem %5, %0 : !yz.int64, !yz.int64 -> !yz.int64
          %7 = yz.neg %6 : !yz.int64 -> !yz.int64
          %8 = yz.cmp "gt", %7, %0 : !yz.int64, !yz.int64 -> !yz.bool
          %9 = yz.not %8 : !yz.bool -> !yz.bool
          %10 = yz.and %8, %9 : !yz.bool, !yz.bool -> !yz.bool
          %11 = yz.or %8, %9 : !yz.bool, !yz.bool -> !yz.bool
          %12 = yz.constant_float 1.500000e+00
          %13 = yz.add %12, %12 : !yz.float64, !yz.float64 -> !yz.float64
          %14 = yz.constant_bool true
          %15 = yz.constant_str "hello"
        }
    "#]]
    .assert_eq(&module.as_operation().to_string());
}

#[test]
fn builds_yz_ops_programmatically() {
    let context = yuzu_mlir::context();
    let location = Location::unknown(&context);
    let int64 = Type::parse(&context, "!yz.int64").expect("!yz.int64 parses");

    let module = Module::new(location);
    let block = module.body();
    let three = block.append_operation(
        OperationBuilder::new("yz.constant_int", location)
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
            .expect("yz.constant_int builds"),
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
          %0 = yz.constant_int 3
          %1 = yz.add %0, %0 : !yz.int64, !yz.int64 -> !yz.int64
        }
    "#]]
    .assert_eq(&module.as_operation().to_string());
}

#[test]
fn a_full_pipeline_round_trips() {
    let context = yuzu_mlir::context();
    let module = parse(
        &context,
        r#"
module {
  %t = yzr.table @t : !yz.struct<@row_ab>
  %w = yzr.filter %t : !yz.struct<@row_ab> {
  ^bb0(%a: !yz.int64, %b: !yz.int64):
%c10 = yz.constant_int 10
%p = yz.cmp "gt", %a, %c10 : !yz.int64, !yz.int64 -> !yz.bool
yzr.yield %p : !yz.bool
  }
  %e = yzr.extend %w {
  ^bb0(%a: !yz.int64, %b: !yz.int64):
%c3 = yz.constant_int 3
%0 = yz.mul %a, %c3 : !yz.int64, !yz.int64 -> !yz.int64
%1 = yz.add %0, %b : !yz.int64, !yz.int64 -> !yz.int64
yzr.yield %1 : !yz.int64
  } : !yz.struct<@row_ab> -> !yz.struct<@row_e>
  %g = yzr.aggregate %e keys [1] {
  ^bb0(%a: !yz.int64, %b: !yz.int64, %e0: !yz.int64):
%m = yzr.agg "sum", %e0 : !yz.int64 -> !yz.int64
yzr.yield %m : !yz.int64
  } : !yz.struct<@row_e> -> !yz.struct<@agg>
  %l = yzr.limit %g, 10 : !yz.struct<@agg>
}
"#,
    )
    .expect("the whole pipeline parses and verifies");
    expect![[r#"
        module {
          %0 = yzr.table @t : !yz.struct<@row_ab>
          %1 = yzr.filter %0 : !yz.struct<@row_ab> {
          ^bb0(%arg0: !yz.int64, %arg1: !yz.int64):
            %5 = yz.constant_int 10
            %6 = yz.cmp "gt", %arg0, %5 : !yz.int64, !yz.int64 -> !yz.bool
            yzr.yield %6 : !yz.bool
          }
          %2 = yzr.extend %1 {
          ^bb0(%arg0: !yz.int64, %arg1: !yz.int64):
            %5 = yz.constant_int 3
            %6 = yz.mul %arg0, %5 : !yz.int64, !yz.int64 -> !yz.int64
            %7 = yz.add %6, %arg1 : !yz.int64, !yz.int64 -> !yz.int64
            yzr.yield %7 : !yz.int64
          } : !yz.struct<@row_ab> -> !yz.struct<@row_e>
          %3 = yzr.aggregate %2 keys [1] {
          ^bb0(%arg0: !yz.int64, %arg1: !yz.int64, %arg2: !yz.int64):
            %5 = yzr.agg "sum", %arg2 : !yz.int64 -> !yz.int64
            yzr.yield %5 : !yz.int64
          } : !yz.struct<@row_e> -> !yz.struct<@agg>
          %4 = yzr.limit %3, 10 : !yz.struct<@agg>
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

    use yuzu_mlir::ods::{yz, yzl};

    let context = yuzu_mlir::context();
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
        yz::constant_int(&context, int64, IntegerAttribute::new(i64, 10), loc).into(),
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
            %3 = yz.constant_int 10
            %4 = yz.cmp "gt", %2, %3 : !yz.int64, !yz.int64 -> !yz.bool
            yzl.yield %4 : !yz.bool
          }
        }
    "#]]
    .assert_eq(&module.as_operation().to_string());
}

#[test]
fn functions_and_calls_round_trip() {
    let context = yuzu_mlir::context();
    let module = parse(
        &context,
        r#"
module {
  yz.func @triple (!yz.int64) -> !yz.int64 {
  ^bb0(%x: !yz.int64):
%c3 = yz.constant_int 3
%0 = yz.mul %x, %c3 : !yz.int64, !yz.int64 -> !yz.int64
yz.return %0 : !yz.int64
  }
  %a = yz.constant_int 7
  %b = yz.call @triple(%a) : (!yz.int64) -> !yz.int64
  %c = yz.extern_call "upper"(%b) : (!yz.int64) -> !yz.int64
}
"#,
    )
    .expect("functions and calls parse");
    expect![[r#"
        module {
          yz.func @triple (!yz.int64) -> !yz.int64 {
          ^bb0(%arg0: !yz.int64):
            %3 = yz.constant_int 3
            %4 = yz.mul %arg0, %3 : !yz.int64, !yz.int64 -> !yz.int64
            yz.return %4 : !yz.int64
          }
          %0 = yz.constant_int 7
          %1 = yz.call @triple(%0) : (!yz.int64) -> !yz.int64
          %2 = yz.extern_call "upper"(%1) : (!yz.int64) -> !yz.int64
        }
    "#]]
    .assert_eq(&module.as_operation().to_string());
}

#[test]
fn joins_set_ops_and_count_round_trip() {
    let context = yuzu_mlir::context();
    let module = parse(
        &context,
        r#"
module {
  %l = yzr.table @l : !yz.struct<@row>
  %r = yzr.table @r : !yz.struct<@row_b>
  %j = yzr.join "inner", %l, %r {
  ^bb0(%a: !yz.int64, %b: !yz.int64):
%p = yz.cmp "eq", %a, %b : !yz.int64, !yz.int64 -> !yz.bool
yzr.yield %p : !yz.bool
  } : !yz.struct<@row>, !yz.struct<@row_b> -> !yz.struct<@row_ab>
  %u = yzr.union %l, %l : !yz.struct<@row>
  %i = yzr.intersect %l, %u : !yz.struct<@row>
  %e = yzr.except %u, %i : !yz.struct<@row>
  %g = yzr.aggregate %e keys [] {
  ^bb0(%a: !yz.int64):
%n = yzr.count : !yz.int64
yzr.yield %n : !yz.int64
  } : !yz.struct<@row> -> !yz.struct<@counted>
}
"#,
    )
    .expect("joins, set ops, and count parse");
    expect![[r#"
        module {
          %0 = yzr.table @l : !yz.struct<@row>
          %1 = yzr.table @r : !yz.struct<@row_b>
          %2 = yzr.join "inner", %0, %1 {
          ^bb0(%arg0: !yz.int64, %arg1: !yz.int64):
            %7 = yz.cmp "eq", %arg0, %arg1 : !yz.int64, !yz.int64 -> !yz.bool
            yzr.yield %7 : !yz.bool
          } : !yz.struct<@row>, !yz.struct<@row_b> -> !yz.struct<@row_ab>
          %3 = yzr.union %0, %0 : !yz.struct<@row>
          %4 = yzr.intersect %0, %3 : !yz.struct<@row>
          %5 = yzr.except %3, %4 : !yz.struct<@row>
          %6 = yzr.aggregate %5 keys [] {
          ^bb0(%arg0: !yz.int64):
            %7 = yzr.count : !yz.int64
            yzr.yield %7 : !yz.int64
          } : !yz.struct<@row> -> !yz.struct<@counted>
        }
    "#]]
    .assert_eq(&module.as_operation().to_string());
}

/// A stage region is `IsolatedFromAbove` because it becomes a self-contained
/// Substrait expression: the canonicalizer folds freely inside one, but every
/// constant it leaves behind stays where the emitter can still see it.
/// `1 / 0` stands, as a fold that would change the program's meaning.
#[test]
fn the_canonicalizer_keeps_its_folding_inside_the_region() {
    let context = yuzu_mlir::context();
    let mut module = parse(
        &context,
        r#"
module {
  %t = yzr.table @t : !yz.struct<@row>
  %w = yzr.filter %t : !yz.struct<@row> {
  ^bb0(%a: !yz.int64):
%c3 = yz.constant_int 3
%c4 = yz.constant_int 4
%p = yz.cmp "gt", %c4, %c3 : !yz.int64, !yz.int64 -> !yz.bool
%q = yz.cmp "gt", %a, %c3 : !yz.int64, !yz.int64 -> !yz.bool
%r = yz.and %p, %q : !yz.bool, !yz.bool -> !yz.bool
yzr.yield %r : !yz.bool
  }
  %d = yzr.filter %w : !yz.struct<@row> {
  ^bb0(%a: !yz.int64):
%c0 = yz.constant_int 0
%c1 = yz.constant_int 1
%z = yz.div %c1, %c0 : !yz.int64, !yz.int64 -> !yz.int64
%p = yz.cmp "eq", %z, %c1 : !yz.int64, !yz.int64 -> !yz.bool
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
          %0 = yzr.table @t : !yz.struct<@row>
          %1 = yzr.filter %0 : !yz.struct<@row> {
          ^bb0(%arg0: !yz.int64):
            %3 = yz.constant_int 3
            %4 = yz.constant_bool true
            %5 = yz.cmp "gt", %arg0, %3 : !yz.int64, !yz.int64 -> !yz.bool
            %6 = yz.and %5, %4 : !yz.bool, !yz.bool -> !yz.bool
            yzr.yield %6 : !yz.bool
          }
          %2 = yzr.filter %1 : !yz.struct<@row> {
          ^bb0(%arg0: !yz.int64):
            %3 = yz.constant_int 0
            %4 = yz.constant_int 1
            %5 = yz.div %4, %3 : !yz.int64, !yz.int64 -> !yz.int64
            %6 = yz.cmp "eq", %5, %4 : !yz.int64, !yz.int64 -> !yz.bool
            yzr.yield %6 : !yz.bool
          }
        }
    "#]]
    .assert_eq(&module.as_operation().to_string());
}

#[test]
fn yzl_names_and_vars_round_trip() {
    let context = yuzu_mlir::context();
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
fn the_verifier_rejects_a_mistyped_operand() {
    let context = yuzu_mlir::context();
    let module = parse(
        &context,
        r#"
module {
  %0 = "yz.wrong"() : () -> !yz.bool
  %1 = yz.add %0, %0 : !yz.int64, !yz.int64 -> !yz.int64
}
"#,
    );
    assert!(module.is_none(), "yz.add over !yz.bool must not parse");
}

/// melior's generated matching, where it works today: on operations you own.
/// A walk's borrowed refs cannot use this yet — the generated TryFrom
/// consumes an owned Operation — which is what yuzu_mlir::ops covers.
#[test]
fn typed_matching_works_on_owned_operations() {
    use melior::ir::attribute::IntegerAttribute;
    use yuzu_mlir::ods::yz::{self, YzDialectOperation};

    let context = yuzu_mlir::context();
    let location = Location::unknown(&context);
    let operation: melior::ir::operation::Operation = yz::constant_int(
        &context,
        yuzu_mlir::types::int64(&context),
        IntegerAttribute::new(melior::ir::r#type::IntegerType::new(&context, 64).into(), 7),
        location,
    )
    .into();

    match YzDialectOperation::try_new(operation) {
        Ok(YzDialectOperation::ConstantInt(constant)) => {
            let value = constant.value().expect("the value attribute exists");
            assert_eq!(value.value(), 7);
        }
        Ok(other) => panic!("classified as the wrong op: {other}"),
        Err(operation) => panic!("failed to classify {operation}"),
    }
}

/// The struct type is nominal: the symbol is the identity, so equal names
/// unify and different names never do, whatever their fields.
#[test]
fn struct_types_are_nominal() {
    let context = yuzu_mlir::context();
    let parse = |text| Type::parse(&context, text).expect("the struct type parses");
    let row = parse("!yz.struct<@Row>");
    let same = parse("!yz.struct<@Row>");
    let other = parse("!yz.struct<@Other>");
    assert_eq!(row, same);
    assert_ne!(row, other);
    assert_eq!(row.to_string(), "!yz.struct<@Row>");
}

/// The generated borrowed views: typed matching and accessors over a
/// walk's refs, which melior's owned conversions cannot serve.
#[test]
fn borrowed_views_match_and_read_during_walks() {
    use melior::ir::RegionLike;
    use yuzu_mlir::ext::BlockExt;
    use yuzu_mlir::ops::yzl::YzlOperationRef;

    let context = yuzu_mlir::context();
    let module = parse(
        &context,
        r#"
module {
  %0 = yzl.from @t
  %1 = yzl.where %0 {
    %2 = yzl.name "a" : !yzl.var
    yzl.yield %2 : !yzl.var
  }
  yzl.output %1
}
"#,
    )
    .expect("the fixture parses");

    let mut seen = Vec::new();
    let mut source = None;
    for op in module.body().operations() {
        match YzlOperationRef::of(&op) {
            Some(YzlOperationRef::From(from)) => {
                assert_eq!(from.source().value(), "t");
                source = Some(yuzu_mlir::value_id(from.result().into()));
                seen.push("from");
            }
            Some(YzlOperationRef::Where(filter)) => {
                assert_eq!(Some(yuzu_mlir::value_id(filter.input())), source);
                let predicate = filter
                    .body()
                    .first_block()
                    .and_then(|block| block.first_operation())
                    .expect("the where region holds the predicate");
                match YzlOperationRef::of(&predicate) {
                    Some(YzlOperationRef::Name(name)) => assert_eq!(name.name().value(), "a"),
                    _ => panic!("the predicate starts with a yzl.name"),
                }

                seen.push("where");
            }
            Some(YzlOperationRef::Output(_)) => seen.push("output"),
            _ => panic!("unclassified op in the fixture"),
        }
    }

    assert_eq!(seen, ["from", "where", "output"]);
}

/// The remaining accessor shapes: optional attributes, unit attributes, and
/// variadic operands through the generated views.
#[test]
fn views_read_optional_unit_and_variadic_arguments() {
    use melior::ir::attribute::{FlatSymbolRefAttribute, StringAttribute};
    use melior::ir::{Attribute, Identifier, Region, RegionLike};
    use yuzu_mlir::ops::yzl::YzlOperationRef;

    let context = yuzu_mlir::context();
    let location = Location::unknown(&context);

    // Unit attributes and a variadic call, through parsed IR.
    let module = parse(
        &context,
        r#"
module {
  yzl.fn @f params ["x", "y"] (!yz.int64, !yz.int64) -> !yz.int64 {
    %0 = yzl.name "x" : !yzl.var
    %1 = yzl.name "y" : !yzl.var
    %2 = yzl.call @g(%0, %1) : (!yzl.var, !yzl.var) -> !yzl.var
    yzl.return %2 : !yzl.var
  }
}
"#,
    )
    .expect("the fixture parses");

    let function = module.body().first_operation().expect("the fn is present");
    let Some(YzlOperationRef::Fn(function)) = YzlOperationRef::of(&function) else {
        panic!("the first op is the yzl.fn");
    };

    assert!(!function.agg());
    assert!(!function.external());
    let call = function
        .body()
        .first_block()
        .and_then(|block| {
            let mut op = block.first_operation()?;
            while let Some(next) = op.next_in_block() {
                match YzlOperationRef::of(&op) {
                    Some(YzlOperationRef::Call(_)) => break,
                    _ => op = next,
                }
            }

            Some(op)
        })
        .expect("the body holds the call");
    let Some(YzlOperationRef::Call(call)) = YzlOperationRef::of(&call) else {
        panic!("the op is the yzl.call");
    };

    assert_eq!(call.callee().value(), "g");
    assert_eq!(call.operands().count(), 2);

    // Optional attributes, present and absent, on unverified built joins.
    let join = |alias: Option<&str>| {
        let mut attributes = vec![
            (
                Identifier::new(&context, "kind"),
                StringAttribute::new(&context, "inner").into(),
            ),
            (
                Identifier::new(&context, "rhs"),
                FlatSymbolRefAttribute::new(&context, "teams").into(),
            ),
        ];
        if let Some(alias) = alias {
            attributes.push((
                Identifier::new(&context, "rhs_alias"),
                StringAttribute::new(&context, alias).into(),
            ));
        }

        OperationBuilder::new("yzl.join", location)
            .add_attributes(&attributes)
            .add_regions([Region::new()])
            .add_results(&[yuzu_mlir::types::query(&context)])
            .build()
            .expect("the join builds")
    };

    let aliased = join(Some("t"));
    let Some(YzlOperationRef::Join(view)) = YzlOperationRef::of(&aliased) else {
        panic!("the op is the yzl.join");
    };

    assert_eq!(view.rhs_alias().map(|alias| alias.value()), Some("t"));
    assert!(view.using_columns().is_none());

    let bare = join(None);
    let Some(YzlOperationRef::Join(view)) = YzlOperationRef::of(&bare) else {
        panic!("the op is the yzl.join");
    };

    assert!(view.rhs_alias().is_none());

    // A unit attribute that is present reads as true.
    let aggregate_fn = OperationBuilder::new("yzl.fn", location)
        .add_attributes(&[(Identifier::new(&context, "agg"), Attribute::unit(&context))])
        .add_regions([Region::new()])
        .build()
        .expect("the fn builds");
    let Some(YzlOperationRef::Fn(view)) = YzlOperationRef::of(&aggregate_fn) else {
        panic!("the op is the yzl.fn");
    };

    assert!(view.agg());
    assert!(!view.external());
}

/// A view classifies only its own dialect; foreign ops come back as None.
#[test]
fn borrowed_views_reject_foreign_operations() {
    use melior::ir::attribute::IntegerAttribute;
    use yuzu_mlir::ods::yz;
    use yuzu_mlir::ops::yz::YzOperationRef;
    use yuzu_mlir::ops::yzl::YzlOperationRef;

    let context = yuzu_mlir::context();
    let location = Location::unknown(&context);
    let operation: melior::ir::operation::Operation = yz::constant_int(
        &context,
        yuzu_mlir::types::int64(&context),
        IntegerAttribute::new(melior::ir::r#type::IntegerType::new(&context, 64).into(), 7),
        location,
    )
    .into();

    assert!(YzlOperationRef::of(&operation).is_none());
    match YzOperationRef::of(&operation) {
        Some(YzOperationRef::ConstantInt(constant)) => {
            assert_eq!(constant.value().value(), 7);
        }
        _ => panic!("the constant classifies as yz.constant_int"),
    }
}

/// The melior 0.27.6 predicate bug is fixed upstream as of 0.27.7:
/// converting a genuine array succeeds, on the attribute this stack puts on
/// every yzl.fn.
#[test]
fn melior_accepts_a_real_array_attribute() {
    use melior::ir::attribute::{ArrayAttribute, StringAttribute};

    let context = yuzu_mlir::context();
    let attribute: melior::ir::Attribute =
        ArrayAttribute::new(&context, &[StringAttribute::new(&context, "x").into()]).into();

    let array = ArrayAttribute::try_from(attribute).expect("a real array converts");
    assert_eq!(array.len(), 1);
    let element = array.element(0).expect("the element reads");
    assert_eq!(
        StringAttribute::try_from(element)
            .expect("the element is a string")
            .value(),
        "x"
    );
}
