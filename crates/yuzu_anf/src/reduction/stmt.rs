use crate::reduction::term::Term;
use crate::reduction::{AnfReducer, environment::Environment};
use crate::{Atom, AtomId, BindingId, Expr, ExprId, Stmt, StmtId, StructFieldInit, TreeCopier};

impl AnfReducer<'_> {
    /// Evaluates the top-level statement list. Declarations pass through
    /// (copied into the output), lets and assigns fold into the environment,
    /// and observable roots — queries, expression statements, returns — are
    /// emitted along with whatever they demand.
    pub(crate) fn reduce_stmts(&mut self, stmts: &[StmtId], env: &mut Environment) -> Vec<StmtId> {
        let input = self.reduced;

        let mut reduced_stmts = Vec::new();
        for &id in stmts {
            let term = match input.stmt(id) {
                Stmt::Struct { .. } | Stmt::Table { .. } => self.reduce_struct_table_stmt(id),
                Stmt::Func { binding, .. } => self.reduce_func_stmt(*binding, id),
                Stmt::Block { stmts } => match self.reduce_block(stmts, env, &mut reduced_stmts) {
                    term @ Term::Yield(_) => Some(term),
                    _ => None,
                },
                Stmt::Let { binding, expr } => {
                    self.reduce_let_stmt(*binding, *expr, env, &mut reduced_stmts)
                }
                Stmt::Assign { target, value } => {
                    self.reduce_assign_stmt(*target, *value, env, &mut reduced_stmts)
                }
                Stmt::Return { value } => {
                    Some(self.reduce_return_stmt(*value, env, &mut reduced_stmts))
                }
                Stmt::Expr { value } => self.reduce_expr_stmt(*value, env, &mut reduced_stmts),
            };

            match term {
                Some(Term::Stmt(stmt)) => {
                    self.bind_stmt_origin(stmt, id);
                    reduced_stmts.push(stmt);
                }
                Some(Term::Yield(value)) => {
                    let stmt = self.anf.alloc_stmt(Stmt::Return { value });
                    self.bind_stmt_origin(stmt, id);
                    reduced_stmts.push(stmt);
                    break;
                }
                Some(Term::Expr(_) | Term::Atom(_) | Term::Unit) => {
                    unreachable!("a statement reduces to a statement or a control signal")
                }
                None => {}
            }
        }
        reduced_stmts
    }

    pub(crate) fn reduce_block(
        &mut self,
        stmts: &[StmtId],
        env: &mut Environment,
        out: &mut Vec<StmtId>,
    ) -> Term {
        let input = self.reduced;
        let mut tail = Term::Unit;
        for &id in stmts {
            match input.stmt(id) {
                // Function templates are registered up front; type declarations
                // are global and already emitted at the root.
                Stmt::Struct { .. } | Stmt::Table { .. } | Stmt::Func { .. } => {}
                Stmt::Block { stmts } => {
                    if let term @ Term::Yield(_) = self.reduce_block(stmts, env, out) {
                        return term;
                    }
                }
                Stmt::Let { binding, expr } => {
                    let binding = *binding;
                    let expr = *expr;
                    if let Some(computation) = self.reduce_binding(binding, expr, env, out) {
                        let atom = self.bind_to_temp(computation, expr, out);
                        env.bind_atom(binding, atom);
                    }
                }
                Stmt::Assign { target, value } => {
                    self.reduce_assign_stmt(*target, *value, env, out);
                }
                Stmt::Return { value } => {
                    // The first return is the value; anything after it is
                    // unreachable and must not execute.
                    return self.reduce_return_stmt(*value, env, out);
                }
                Stmt::Expr { value } => {
                    tail = match self.reduce_expr(*value, env, out) {
                        Term::Expr(computation) => {
                            Term::Atom(self.bind_to_temp(computation, *value, out))
                        }
                        term @ (Term::Atom(_) | Term::Unit) => term,
                        _ => unreachable!("an expression reduces to a value or a computation"),
                    };
                }
            }
        }
        tail
    }

    fn reduce_struct_table_stmt(&mut self, stmt: StmtId) -> Option<Term> {
        Some(Term::Stmt(self.copy_stmt(stmt)))
    }

    fn reduce_func_stmt(&mut self, binding: BindingId, stmt: StmtId) -> Option<Term> {
        let copied = self.copy_stmt(stmt);
        self.emitted_funcs.insert(binding, copied);
        Some(Term::Stmt(copied))
    }

    fn reduce_let_stmt(
        &mut self,
        binding: BindingId,
        expr: ExprId,
        env: &mut Environment,
        out: &mut Vec<StmtId>,
    ) -> Option<Term> {
        let computation = self.reduce_binding(binding, expr, env, out)?;
        let origin = expr;
        let expr = self.anf.alloc_expr(computation);
        self.bind_expr_origin(expr, origin);
        Some(Term::Stmt(self.anf.alloc_stmt(Stmt::Let { binding, expr })))
    }

    fn reduce_binding(
        &mut self,
        binding: BindingId,
        expr: ExprId,
        env: &mut Environment,
        out: &mut Vec<StmtId>,
    ) -> Option<Expr> {
        match self.reduced.expr(expr) {
            Expr::StructInit { name, fields, ty } => {
                let fields = fields
                    .iter()
                    .map(|field| StructFieldInit {
                        name: field.name,
                        value: self.resolve_atom(field.value, env),
                    })
                    .collect();
                env.bind_struct(binding, *name, fields, *ty, expr);
                None
            }
            Expr::ListInit { elements, ty } => {
                let elements = elements
                    .iter()
                    .map(|&element| self.resolve_atom(element, env))
                    .collect();
                env.bind_list(binding, elements, *ty, expr);
                None
            }
            _ => match self.reduce_expr(expr, env, out) {
                Term::Atom(value) => {
                    env.bind_atom(binding, value);
                    None
                }
                Term::Expr(computation) => Some(computation),
                Term::Unit => None,
                _ => unreachable!("an expression reduces to a value or a computation"),
            },
        }
    }

    fn reduce_return_stmt(
        &mut self,
        value: Option<AtomId>,
        env: &mut Environment,
        out: &mut Vec<StmtId>,
    ) -> Term {
        Term::Yield(value.map(|value| self.reduce_atom(value, env, out)))
    }

    fn reduce_expr_stmt(
        &mut self,
        value: ExprId,
        env: &mut Environment,
        out: &mut Vec<StmtId>,
    ) -> Option<Term> {
        let origin = value;
        let value = match self.reduce_expr(value, env, out) {
            Term::Atom(value) => Expr::Atom { value },
            Term::Expr(computation) => computation,
            Term::Unit => return None,
            _ => unreachable!("an expression reduces to a value or a computation"),
        };
        let value = self.anf.alloc_expr(value);
        self.bind_expr_origin(value, origin);
        Some(Term::Stmt(self.anf.alloc_stmt(Stmt::Expr { value })))
    }

    fn reduce_assign_stmt(
        &mut self,
        target: AtomId,
        value: ExprId,
        env: &mut Environment,
        out: &mut Vec<StmtId>,
    ) -> Option<Term> {
        let Atom::Var { binding } = *self.anf.atom(target) else {
            unreachable!("fields are immutable; an assign target is a name")
        };

        if let Some(computation) = self.reduce_binding(binding, value, env, out) {
            let atom = self.bind_to_temp(computation, value, out);
            env.bind_atom(binding, atom);
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::reduction::test_support::{TABLE, check};

    #[test]
    fn keeps_declarations() {
        check(
            &format!("struct Point {{ x: int64, y: int64 }}\n{TABLE}from t |> select 2 + 2 as v"),
            expect![[r#"
                struct Point { x, y }
                struct Row { a, b }
                table t
                from t
                  |> select 4i64 as v
            "#]],
        );
    }

    #[test]
    fn sequential_reassignments_fold() {
        check(
            &format!("{TABLE}let mut a = 1\na = a + 1\na = a * 3\nfrom t |> select a as w"),
            expect![[r#"
                struct Row { a, b }
                table t
                from t
                  |> select 6i64 as w
            "#]],
        );
    }

    #[test]
    fn assign_between_queries_snapshots_each() {
        check(
            &format!(
                "{TABLE}let mut a = 1\nfrom t |> select a as w\na = 2\nfrom t |> select a as w"
            ),
            expect![[r#"
                struct Row { a, b }
                table t
                from t
                  |> select 1i64 as w
                from t
                  |> select 2i64 as w
            "#]],
        );
    }

    #[test]
    fn assign_of_runtime_computation() {
        check(
            &format!(
                "{TABLE}def fact(n: int64) -> int64 {{ return n * fact(n - 1) }}\nlet mut a = 1\na = fact(3)\nfrom t |> select a as w"
            ),
            expect![[r#"
                struct Row { a, b }
                table t
                fn fact(n) {
                  let %t0 = sub(n, 1i64)
                  let %t1 = fact(%t0)
                  let %t2 = mul(n, %t1)
                  return %t2
                }
                let %r0 = fact(2i64)
                let %r1 = mul(3i64, %r0)
                from t
                  |> select %r1 as w
            "#]],
        );
    }

    #[test]
    fn runtime_residue_binds_at_root() {
        check(
            &format!(
                "{TABLE}def fact(n: int64) -> int64 {{ return n * fact(n - 1) }}\nlet v = fact(3)\nfrom t |> select v as w"
            ),
            expect![[r#"
                struct Row { a, b }
                table t
                fn fact(n) {
                  let %t0 = sub(n, 1i64)
                  let %t1 = fact(%t0)
                  let %t2 = mul(n, %t1)
                  return %t2
                }
                let %r0 = fact(2i64)
                let %r1 = mul(3i64, %r0)
                from t
                  |> select %r1 as w
            "#]],
        );
    }

    #[test]
    fn void_function_effects_fold_into_environment() {
        check(
            &format!(
                "{TABLE}let mut a = 1\ndef bump() {{ a = 2 }}\nbump()\nfrom t |> select a as w"
            ),
            expect![[r#"
                struct Row { a, b }
                table t
                from t
                  |> select 2i64 as w
            "#]],
        );
    }

    #[test]
    fn void_call_chain_absorbs() {
        check(
            &format!(
                "{TABLE}let mut a = 1\ndef nested() {{ a = 2 }}\ndef outer() {{ nested() }}\nouter()\nfrom t |> select a as w"
            ),
            expect![[r#"
                struct Row { a, b }
                table t
                from t
                  |> select 2i64 as w
            "#]],
        );
    }

    #[test]
    fn repeated_void_calls_accumulate() {
        check(
            &format!(
                "{TABLE}let mut a = 1\ndef inc() {{ a = a + 1 }}\ninc()\ninc()\ninc()\nfrom t |> select a as w"
            ),
            expect![[r#"
                struct Row { a, b }
                table t
                from t
                  |> select 4i64 as w
            "#]],
        );
    }

    #[test]
    fn function_with_effect_and_value() {
        check(
            &format!(
                "{TABLE}let mut a = 1\ndef bump() -> int64 {{ a = a + 1\nreturn a }}\nlet v = bump()\nfrom t |> select a as x, v as y"
            ),
            expect![[r#"
                struct Row { a, b }
                table t
                from t
                  |> select 2i64 as x, 2i64 as y
            "#]],
        );
    }

    #[test]
    fn recursive_void_call_is_kept() {
        check(
            &format!("{TABLE}def r() {{ r() }}\nr()\nfrom t |> select a"),
            expect![[r#"
                struct Row { a, b }
                table t
                fn r() {
                  r()
                }
                let %r0 = r()
                %r0
                from t
                  |> select a
            "#]],
        );
    }

    #[test]
    fn code_after_return_is_unreachable() {
        check(
            &format!(
                "{TABLE}let mut a = 1\ndef f() -> int64 {{ return 1\na = 2\nreturn 3 }}\nlet v = f()\nfrom t |> select a as x, v as y"
            ),
            expect![[r#"
                struct Row { a, b }
                table t
                from t
                  |> select 1i64 as x, 1i64 as y
            "#]],
        );
    }

    #[test]
    fn top_level_return_is_emitted() {
        check(
            &format!("{TABLE}from t |> select a\nreturn 1"),
            expect![[r#"
                struct Row { a, b }
                table t
                from t
                  |> select a
                return 1i64
            "#]],
        );
    }

    #[test]
    fn top_level_code_after_return_is_unreachable() {
        check(
            &format!("{TABLE}let mut a = 1\nreturn a\na = 2\nfrom t |> select a as w"),
            expect![[r#"
                struct Row { a, b }
                table t
                return 1i64
            "#]],
        );
    }
}
