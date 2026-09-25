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
        variables: HashMap::new(),
    };

    checker.check_block(module.body());
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mutability {
    Immutable,
    Mutable,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LocalKind {
    Let(Mutability),
    Param,
}

struct Variable<'c> {
    name: &'c str,
    kind: LocalKind,
    initialized: bool,
}

struct MutabilityChecker<'c> {
    variables: HashMap<ValueId, Variable<'c>>,
}

impl<'c> MutabilityChecker<'c> {
    fn check_block(&mut self, block: BlockRef<'c, '_>) {
        for op in block.operations() {
            match op.as_yzl() {
                Some(YzlOp::Local(local)) => {
                    let kind = if local.is_param() {
                        LocalKind::Param
                    } else if local.is_mut() {
                        LocalKind::Let(Mutability::Mutable)
                    } else {
                        LocalKind::Let(Mutability::Immutable)
                    };

                    self.variables.insert(
                        op.first_result().id(),
                        Variable {
                            name: local.var_name().value(),
                            kind,
                            initialized: false,
                        },
                    );
                }
                Some(YzlOp::Store(store)) => {
                    let variable = self
                        .variables
                        .get_mut(&store.place().id())
                        .expect("a store follows the place it writes");

                    if !variable.initialized {
                        variable.initialized = true;
                        continue;
                    }

                    let name = variable.name;
                    match variable.kind {
                        LocalKind::Let(Mutability::Mutable) => {}
                        LocalKind::Let(Mutability::Immutable) => emit_error(
                            op.location(),
                            &format!(
                                "`{name}` is not mutable; declare it with `let mut` to assign it"
                            ),
                        ),
                        LocalKind::Param => emit_error(
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
            r#"
struct Row { a: int64 }
table t = Row

def f(x: int64) -> int64 {
    let n = 0
    n = n + 1
    return n
}

from t |> select f(a) as v
"#,
            expect![[r#"
                error: `n` is not mutable; declare it with `let mut` to assign it
                 --> test.yz:7:5
                  |
                7 |     n = n + 1
                  |     ^^^^^^^^^
            "#]],
        );
    }

    #[test]
    fn a_mutable_binding_may_be_assigned() {
        check(
            r#"
struct Row { a: int64 }
table t = Row

def f(x: int64) -> int64 {
    let mut n = 0
    n = n + 1
    n = n * 2
    return n
}

from t |> select f(a) as v
"#,
            expect!["no diagnostics"],
        );
    }

    #[test]
    fn a_parameter_is_not_assignable() {
        check(
            r#"
struct Row { a: int64 }
table t = Row

def f(x: int64) -> int64 {
    x = 1
    return x
}

from t |> select f(a) as v
"#,
            expect![[r#"
                error: `x` is a parameter and cannot be assigned
                 --> test.yz:6:5
                  |
                6 |     x = 1
                  |     ^^^^^
            "#]],
        );
    }

    #[test]
    fn a_shadowing_let_is_a_new_place() {
        check(
            r#"
struct Row { a: int64 }
table t = Row

def f(x: int64) -> int64 {
    let n = x
    let n = n + 1
    return n
}

from t |> select f(a) as v
"#,
            expect!["no diagnostics"],
        );
    }
}
