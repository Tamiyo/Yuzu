use yuzu_core::adt::SymbolId;
use yuzu_types::TypeId;

use crate::reduction::term::Term;
use crate::reduction::{AnfReducer, environment::Environment, fold::fold};
use crate::{
    Atom, AtomId, BindingId, Const, Expr, ExprId, Ident, Op, RelId, Stmt, StmtId, StructFieldInit,
};

impl<'r> AnfReducer<'r> {
    pub(crate) fn reduce_expr(
        &mut self,
        id: ExprId,
        env: &mut Environment,
        out: &mut Vec<StmtId>,
    ) -> Term {
        match self.reduced.expr(id) {
            Expr::Atom { value } => self.reduce_atom_expr(*value, env, out),
            Expr::Call { op, args, ty } => self.reduce_call_expr(*op, args, *ty, env, out),
            Expr::FuncCall { callee, args, ty } => {
                self.reduce_func_call_expr(*callee, args, *ty, env, out)
            }
            Expr::MethodCall {
                receiver,
                method,
                args,
                ty,
            } => self.reduce_method_call_expr(*receiver, *method, args, *ty, env, out),
            Expr::StructInit { name, fields, ty } => {
                self.reduce_struct_init_expr(*name, fields, *ty, env, out)
            }
            Expr::ListInit { elements, ty } => self.reduce_list_init_expr(elements, *ty, env, out),
            Expr::Rel(rel) => self.reduce_rel_expr(*rel, env),
            Expr::AggCall { func, args, ty } => Term::Expr(Expr::AggCall {
                func: *func,
                args: self.reduce_atoms(args, env, out),
                ty: *ty,
            }),
            Expr::ExternCall { name, args, ty } => Term::Expr(Expr::ExternCall {
                name: *name,
                args: self.reduce_atoms(args, env, out),
                ty: *ty,
            }),
        }
    }

    fn reduce_atom_expr(
        &mut self,
        value: AtomId,
        env: &mut Environment,
        out: &mut Vec<StmtId>,
    ) -> Term {
        Term::Atom(self.reduce_atom(value, env, out))
    }

    fn reduce_method_call_expr(
        &mut self,
        receiver: AtomId,
        method: Ident,
        args: &[AtomId],
        ty: TypeId,
        env: &mut Environment,
        out: &mut Vec<StmtId>,
    ) -> Term {
        Term::Expr(Expr::MethodCall {
            receiver: self.reduce_atom(receiver, env, out),
            method,
            args: self.reduce_atoms(args, env, out),
            ty,
        })
    }

    fn reduce_struct_init_expr(
        &mut self,
        name: SymbolId,
        fields: &[StructFieldInit],
        ty: TypeId,
        env: &mut Environment,
        out: &mut Vec<StmtId>,
    ) -> Term {
        Term::Expr(Expr::StructInit {
            name,
            fields: fields
                .iter()
                .map(|field| StructFieldInit {
                    name: field.name,
                    value: self.reduce_atom(field.value, env, out),
                })
                .collect(),
            ty,
        })
    }

    fn reduce_list_init_expr(
        &mut self,
        elements: &[AtomId],
        ty: TypeId,
        env: &mut Environment,
        out: &mut Vec<StmtId>,
    ) -> Term {
        Term::Expr(Expr::ListInit {
            elements: self.reduce_atoms(elements, env, out),
            ty,
        })
    }

    fn reduce_rel_expr(&mut self, rel: RelId, env: &mut Environment) -> Term {
        Term::Expr(Expr::Rel(self.reduce_rel(rel, env)))
    }

    fn reduce_call_expr(
        &mut self,
        op: Op,
        args: &[AtomId],
        ty: TypeId,
        env: &mut Environment,
        out: &mut Vec<StmtId>,
    ) -> Term {
        if matches!(op, Op::In | Op::NotIn) {
            return self.reduce_membership(op, args, ty, env, out);
        }
        let args = self.reduce_atoms(args, env, out);
        if let Some(folded) = self.try_fold(op, &args) {
            return Term::Atom(self.anf.intern_atom(Atom::Const(folded)));
        }
        Term::Expr(Expr::Call { op, args, ty })
    }

    /// Reduces `x in xs` / `x not in xs`. Both operands resolve without
    /// materializing first, so a fold leaves no dead aggregate behind; they
    /// only materialize if the test survives to runtime.
    fn reduce_membership(
        &mut self,
        op: Op,
        args: &[AtomId],
        ty: TypeId,
        env: &mut Environment,
        out: &mut Vec<StmtId>,
    ) -> Term {
        let [value, list] = args else {
            unreachable!("a membership test has a value and a list")
        };
        let resolved_value = self.resolve_atom(*value, env);
        let resolved_list = self.resolve_atom(*list, env);
        if let Some(folded) = self.fold_membership(op, resolved_value, resolved_list, env) {
            return Term::Atom(self.anf.intern_atom(Atom::Const(folded)));
        }
        let value = self.reduce_atom(*value, env, out);
        let list = self.reduce_atom(*list, env, out);
        Term::Expr(Expr::Call {
            op,
            args: Box::new([value, list]),
            ty,
        })
    }

    fn reduce_func_call_expr(
        &mut self,
        callee: AtomId,
        args: &[AtomId],
        ty: TypeId,
        env: &mut Environment,
        out: &mut Vec<StmtId>,
    ) -> Term {
        let args = self.reduce_atoms(args, env, out);

        if let Some((binding, params, body)) = self.inlinable(callee, args.len()) {
            for (&param, &arg) in params.iter().zip(&args) {
                env.bind_atom(param, arg);
            }
            self.inlining.insert(binding);
            let tail = self.reduce_block(body, env, out);
            self.inlining.remove(&binding);
            return match tail {
                Term::Yield(Some(value)) | Term::Atom(value) => Term::Atom(value),
                Term::Yield(None) | Term::Unit => Term::Unit,
                _ => unreachable!("a body reduces to a value or a return"),
            };
        }

        let callee = self.reduce_atom(callee, env, out);
        Term::Expr(Expr::FuncCall { callee, args, ty })
    }

    /// The inline decision, made entirely up front so a committed inline never
    /// has to roll back environment changes. Yields the callee's identity and
    /// its template's parameters and body statements.
    fn inlinable(
        &self,
        callee: AtomId,
        arity: usize,
    ) -> Option<(BindingId, &'r [BindingId], &'r [StmtId])> {
        // Only a direct reference to a known function can inline.
        let Atom::FuncRef { binding, .. } = *self.anf.atom(callee) else {
            return None;
        };
        let template = self.funcs.func(binding)?;

        // A call to a function already being inlined is recursive; it stays in
        // place, ending the expansion.
        if self.inlining.contains(&binding) {
            return None;
        }

        // The template must have a body block, and the call must supply
        // exactly its parameters.
        let input = self.reduced;
        let Stmt::Func {
            params,
            body: Some(body),
            ..
        } = input.stmt(template)
        else {
            return None;
        };
        if params.len() != arity {
            return None;
        }
        let Stmt::Block { stmts } = input.stmt(*body) else {
            return None;
        };

        Some((binding, params, stmts))
    }

    fn fold_membership(
        &self,
        op: Op,
        value: AtomId,
        list: AtomId,
        env: &Environment,
    ) -> Option<Const> {
        let negate_result = match op {
            Op::In => false,
            Op::NotIn => true,
            _ => unreachable!("unreachable op in fold_membership"),
        };

        // If the right side is not a binding to a list, we can't reduce membership.
        let Atom::Var { binding } = *self.anf.atom(list) else {
            return None;
        };

        // An element equal to the value decides true; all elements provably
        // unequal decide false; anything undecidable keeps the runtime test.
        let mut undecided = false;
        for &element in env.list_elements(binding)? {
            match self.atoms_equal(value, element, env) {
                Some(true) => {
                    return Some(Const::Bool {
                        value: true ^ negate_result,
                    });
                }
                Some(false) => {}
                None => undecided = true,
            }
        }
        if undecided {
            return None;
        }
        Some(Const::Bool {
            value: false ^ negate_result,
        })
    }

    /// Structural equality of two resolved atoms, decided through the
    /// environment: constants compare by value, scalarized aggregates compare
    /// part by part, and a runtime value is undecidable.
    fn atoms_equal(&self, a: AtomId, b: AtomId, env: &Environment) -> Option<bool> {
        if a == b {
            return Some(true);
        }
        match (self.anf.atom(a), self.anf.atom(b)) {
            (Atom::Const(a), Atom::Const(b)) => match fold(Op::Eq, &[*a, *b]) {
                Some(Const::Bool { value }) => Some(value),
                _ => None,
            },
            (Atom::Var { binding: a }, Atom::Var { binding: b }) => {
                self.aggregates_equal(*a, *b, env)
            }
            _ => None,
        }
    }

    fn aggregates_equal(&self, a: BindingId, b: BindingId, env: &Environment) -> Option<bool> {
        if let (Some(a), Some(b)) = (env.list_elements(a), env.list_elements(b)) {
            if a.len() != b.len() {
                return Some(false);
            }
            return self.parts_equal(a.iter().copied().zip(b.iter().copied()), env);
        }
        if let (Some(a), Some(b)) = (env.struct_fields(a), env.struct_fields(b)) {
            let parts = a
                .iter()
                .map(|field| field.value)
                .zip(b.iter().map(|field| field.value));
            return self.parts_equal(parts, env);
        }
        None
    }

    fn parts_equal(
        &self,
        parts: impl Iterator<Item = (AtomId, AtomId)>,
        env: &Environment,
    ) -> Option<bool> {
        let mut undecided = false;
        for (a, b) in parts {
            match self.atoms_equal(a, b, env) {
                Some(false) => return Some(false),
                Some(true) => {}
                None => undecided = true,
            }
        }
        if undecided { None } else { Some(true) }
    }

    fn try_fold(&self, op: Op, args: &[AtomId]) -> Option<Const> {
        let constants: Vec<Const> = args
            .iter()
            .map(|&atom| match self.anf.atom(atom) {
                Atom::Const(constant) => Some(*constant),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()?;
        fold(op, &constants)
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::reduction::test_support::{TABLE, check};

    #[test]
    fn inlines_nested_calls() {
        check(
            &format!(
                "{TABLE}fn inc(x: int32) -> int32 {{ return x + 1 }}\nfn twice(x: int32) -> int32 {{ return inc(inc(x)) }}\nfrom t |> select twice(10) as v"
            ),
            expect![[r#"
                struct Row { a, b }
                table t
                from t
                  |> select 12i32 as v
            "#]],
        );
    }

    #[test]
    fn inlines_function_into_column_and_sweeps_it() {
        check(
            &format!(
                "{TABLE}fn double(x: int32) -> int32 {{ return x * 2 }}\nlet q = from t |> select double(b) as db"
            ),
            expect![[r#"
                struct Row { a, b }
                table t
                let q = from t
                  |> select mul(b, 2i32) as db
            "#]],
        );
    }

    #[test]
    fn inlined_function_resolves_closed_over_binding() {
        check(
            &format!(
                "{TABLE}let base = 5\nfn add_base(x: int64) -> int64 {{ return x + base }}\nfrom t |> select add_base(10) as v"
            ),
            expect![[r#"
                struct Row { a, b }
                table t
                from t
                  |> select 15i64 as v
            "#]],
        );
    }

    #[test]
    fn recursive_call_is_left_in_place() {
        check(
            &format!(
                "{TABLE}fn fact(n: int64) -> int64 {{ return n * fact(n - 1) }}\nfrom t |> select fact(3) as v"
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
                from t
                  |> select %r0 = fact(2i64); %r1 = mul(3i64, %r0); %r1 as v
            "#]],
        );
    }

    #[test]
    fn shadowed_nested_function_inlines_by_identity() {
        check(
            &format!(
                "{TABLE}fn f() -> int64 {{ return 1 }}\nfn g() -> int64 {{ fn f() -> int64 {{ return 2 }}\nreturn f() }}\nfrom t |> select g() as v"
            ),
            expect![[r#"
                struct Row { a, b }
                table t
                from t
                  |> select 2i64 as v
            "#]],
        );
    }

    #[test]
    fn body_tail_expression_is_the_return_value() {
        check(
            &format!("{TABLE}fn f() -> int64 {{ 5 }}\nfrom t |> select f() as w"),
            expect![[r#"
                struct Row { a, b }
                table t
                from t
                  |> select 5i64 as w
            "#]],
        );
    }

    #[test]
    fn folds_membership_in_constant_list() {
        check(
            &format!("{TABLE}let xs = [1, 2]\nfrom t |> where 2 in xs |> where a in xs"),
            expect![[r#"
                struct Row { a, b }
                table t
                from t
                  |> where true
                  |> where xs = [1i32, 2i32]; %r0 = in(a, xs); %r0
            "#]],
        );
    }

    #[test]
    fn reassigned_list_folds_membership() {
        check(
            &format!("{TABLE}let mut xs = [1, 2]\nxs = [3, 4]\nfrom t |> where 3 in xs"),
            expect![[r#"
                struct Row { a, b }
                table t
                from t
                  |> where true
            "#]],
        );
    }

    #[test]
    fn folds_aggregate_membership_by_identity() {
        check(
            &format!("{TABLE}let x = [1]\nlet y = [2]\nlet z = [x, y]\nfrom t |> where x in z"),
            expect![[r#"
                struct Row { a, b }
                table t
                from t
                  |> where true
            "#]],
        );
    }

    #[test]
    fn folds_aggregate_membership_structurally() {
        check(
            &format!("{TABLE}let z = [[1], [2]]\nfrom t |> where [1] in z |> where [3] in z"),
            expect![[r#"
                struct Row { a, b }
                table t
                from t
                  |> where true
                  |> where false
            "#]],
        );
    }

    #[test]
    fn folds_struct_membership_structurally() {
        check(
            &format!(
                "{TABLE}struct P {{ v: int64 }}\nlet ps = [P {{ v: 1 }}, P {{ v: 2 }}]\nfrom t |> where P {{ v: 2 }} in ps |> where P {{ v: 3 }} not in ps"
            ),
            expect![[r#"
                struct Row { a, b }
                table t
                struct P { v }
                from t
                  |> where true
                  |> where true
            "#]],
        );
    }
}
