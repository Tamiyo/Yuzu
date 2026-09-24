//! A place without `mut` has one store, its initializer. Each later store
//! is an assignment the program may not make.

use std::collections::HashMap;

use melior::ir::operation::OperationLike;
use melior::ir::{BlockRef, Module};
use yuzu_mlir::diagnostics::emit_error;
use yuzu_mlir::ext::{BlockExt, OperationCast, OperationExt, RegionExt, ValueExt, ValueId};
use yuzu_mlir::ops::yzl::YzlOp;

pub fn check_mutability(module: &Module) {
    let mut checker = MutabilityChecker {
        places: HashMap::new(),
    };

    checker.check_block(module.body());
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Place {
    Immutable,
    Mutable,
    Param,
}

struct Variable<'c> {
    name: &'c str,
    place: Place,
    initialized: bool,
}

struct MutabilityChecker<'c> {
    places: HashMap<ValueId, Variable<'c>>,
}

impl<'c> MutabilityChecker<'c> {
    fn check_block(&mut self, block: BlockRef<'c, '_>) {
        for op in block.operations() {
            match op.as_yzl() {
                Some(YzlOp::Local(local)) => {
                    let place = if local.param() {
                        Place::Param
                    } else if local.is_mut() {
                        Place::Mutable
                    } else {
                        Place::Immutable
                    };

                    self.places.insert(
                        op.first_result().id(),
                        Variable {
                            name: local.var_name().value(),
                            place,
                            initialized: false,
                        },
                    );
                }
                Some(YzlOp::Store(store)) => {
                    let variable = self
                        .places
                        .get_mut(&store.place().id())
                        .expect("a store follows the place it writes");

                    if !variable.initialized {
                        variable.initialized = true;
                        continue;
                    }

                    let name = variable.name;
                    match variable.place {
                        Place::Mutable => {}
                        Place::Immutable => emit_error(
                            op.location(),
                            &format!(
                                "`{name}` is not mutable; declare it with `let mut` to assign it"
                            ),
                        ),
                        Place::Param => emit_error(
                            op.location(),
                            &format!("`{name}` is a parameter and cannot be assigned"),
                        ),
                    }
                }
                _ => {
                    for region in op.regions() {
                        for block in region.blocks() {
                            self.check_block(block);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};

    use crate::check_mutability;
    use crate::test_support;

    fn check(source: &str, expected: Expect) {
        test_support::check(
            source,
            |_, module| {
                check_mutability(module);
                "no diagnostics".to_string()
            },
            expected,
        );
    }

    #[test]
    fn assigning_a_binding_needs_mut() {
        check(
            "struct Row { a: int64 }\ntable t = Row\n\ndef f(x: int64) -> int64 {\n    let n = 0\n    n = n + 1\n    return n\n}\n\nfrom t |> select f(a) as v\n",
            expect![[r#"
                error: `n` is not mutable; declare it with `let mut` to assign it
                 --> test.yz:6:5
                  |
                6 |     n = n + 1
                  |     ^^^^^^^^^
            "#]],
        );
    }

    #[test]
    fn a_mutable_binding_may_be_assigned() {
        check(
            "struct Row { a: int64 }\ntable t = Row\n\ndef f(x: int64) -> int64 {\n    let mut n = 0\n    n = n + 1\n    n = n * 2\n    return n\n}\n\nfrom t |> select f(a) as v\n",
            expect!["no diagnostics"],
        );
    }

    #[test]
    fn a_parameter_is_not_assignable() {
        check(
            "struct Row { a: int64 }\ntable t = Row\n\ndef f(x: int64) -> int64 {\n    x = 1\n    return x\n}\n\nfrom t |> select f(a) as v\n",
            expect![[r#"
                error: `x` is a parameter and cannot be assigned
                 --> test.yz:5:5
                  |
                5 |     x = 1
                  |     ^^^^^
            "#]],
        );
    }

    #[test]
    fn a_shadowing_let_is_a_new_place() {
        check(
            "struct Row { a: int64 }\ntable t = Row\n\ndef f(x: int64) -> int64 {\n    let n = x\n    let n = n + 1\n    return n\n}\n\nfrom t |> select f(a) as v\n",
            expect!["no diagnostics"],
        );
    }
}
