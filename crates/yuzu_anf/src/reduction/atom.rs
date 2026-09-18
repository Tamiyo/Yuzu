use crate::reduction::AnfReducer;
use crate::reduction::environment::Environment;
use crate::{Atom, AtomId, BindingId, Expr, Stmt, StmtId, StructFieldInit};

impl AnfReducer<'_> {
    pub(crate) fn reduce_atoms(
        &mut self,
        atoms: &[AtomId],
        env: &Environment,
        out: &mut Vec<StmtId>,
    ) -> Box<[AtomId]> {
        atoms
            .iter()
            .map(|&atom| self.reduce_atom(atom, env, out))
            .collect()
    }

    pub(crate) fn reduce_atom(
        &mut self,
        id: AtomId,
        env: &Environment,
        out: &mut Vec<StmtId>,
    ) -> AtomId {
        let value = self.resolve_atom(id, env);
        if let Atom::Var { binding } = *self.anf.atom(value)
            && env.is_aggregate_value(binding)
        {
            return self.materialize_atom(binding, env, out);
        }

        value
    }

    /// A whole-value use of a scalarized aggregate emits its `let` — once, on
    /// first use — and returns a `Var` referring to it. Nested parts emit first.
    ///
    /// ```text
    /// let p = P { v: 1 }
    /// from t |> select p.v as w
    /// p
    /// ```
    /// reduces to
    /// ```text
    /// from t
    ///   |> select 1i64 as w
    /// let p = P { v: 1i64 }
    /// p
    /// ```
    fn materialize_atom(
        &mut self,
        binding: BindingId,
        env: &Environment,
        out: &mut Vec<StmtId>,
    ) -> AtomId {
        if self.materialized.insert(binding) {
            let (expr, origin) = if let Some((name, fields, ty, origin)) = env.struct_(binding) {
                let fields = fields
                    .iter()
                    .map(|field| StructFieldInit {
                        name: field.name,
                        value: self.reduce_atom(field.value, env, out),
                    })
                    .collect();
                (Expr::StructInit { name, fields, ty }, origin)
            } else if let Some((elements, ty, origin)) = env.list(binding) {
                let elements = elements
                    .iter()
                    .map(|&element| self.reduce_atom(element, env, out))
                    .collect();
                (Expr::ListInit { elements, ty }, origin)
            } else {
                unreachable!("materialize called on a non-aggregate binding")
            };

            let expr = self.anf.alloc_expr(expr);
            self.bind_expr_origin(expr, origin);
            let binding_stmt = self.anf.alloc_stmt(Stmt::Let { binding, expr });
            out.push(binding_stmt);
        }

        self.anf.intern_atom(Atom::Var { binding })
    }

    /// Resolves an atom to a concrete value in the environment.
    ///
    /// Constant and function references are resolved as much as they can be
    /// and are returned.
    ///
    /// Variable bindings (temporary, or user-defined) look-up the variable in
    /// the current environment, and return if the variable exists.
    ///
    /// Fields are a unique case where the base can itself be a resolveable atom.
    /// Consider the field `emp.salary`. The base `emp` is resolved to a var that binds
    /// a aggregate type 'struct'. The binding then fetches the name of the field and returns
    /// the resolved value. Otherwise, if we cannot resolve the atom at this point (for a
    /// runtime read, as in the case with queries), a new field is minted for the reduced ANF.
    pub(crate) fn resolve_atom(&mut self, id: AtomId, env: &Environment) -> AtomId {
        match *self.anf.atom(id) {
            Atom::Const(_) | Atom::FuncRef { .. } | Atom::Column { .. } => id,
            Atom::Var { binding } => env.atom(binding).unwrap_or(id),
            Atom::Field { base, field, ty } => {
                let base = self.resolve_atom(base, env);
                if let Atom::Var { binding } = *self.anf.atom(base)
                    && let Some(value) = env.struct_field(binding, field.name)
                {
                    return self.resolve_atom(value, env);
                }
                self.anf.intern_atom(Atom::Field { base, field, ty })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::reduction::test_support::{TABLE, check};

    #[test]
    fn dead_struct_is_dropped() {
        check(
            &format!("{TABLE}struct P {{ v: int64 }}\nlet p = P {{ v: 40 }}\nfrom t |> select a"),
            expect![[r#"
                struct Row { a, b }
                table t
                struct P { v }
                from t
                  |> select a
            "#]],
        );
    }

    #[test]
    fn dead_list_is_dropped() {
        check(
            &format!("{TABLE}let xs = [1, 2]\nfrom t |> select a"),
            expect![[r#"
                struct Row { a, b }
                table t
                from t
                  |> select a
            "#]],
        );
    }

    #[test]
    fn projects_struct_field() {
        check(
            &format!(
                "{TABLE}struct P {{ v: int64 }}\nlet p = P {{ v: 40 }}\nfrom t |> select p.v + 2 as w"
            ),
            expect![[r#"
                struct Row { a, b }
                table t
                struct P { v }
                from t
                  |> select 42i64 as w
            "#]],
        );
    }

    #[test]
    fn partial_projection_drops_struct() {
        check(
            &format!(
                "{TABLE}struct P {{ x: int64, y: int64 }}\nlet p = P {{ x: 3, y: 4 }}\nfrom t |> select p.x as w"
            ),
            expect![[r#"
                struct Row { a, b }
                table t
                struct P { x, y }
                from t
                  |> select 3i64 as w
            "#]],
        );
    }

    #[test]
    fn struct_in_function_body_projects() {
        check(
            &format!(
                "{TABLE}struct P {{ v: int64 }}\ndef f() -> int64 {{ let p = P {{ v: 40 }}\nreturn p.v }}\nfrom t |> select f() as w"
            ),
            expect![[r#"
                struct Row { a, b }
                table t
                struct P { v }
                from t
                  |> select 40i64 as w
            "#]],
        );
    }

    #[test]
    fn reassigned_struct_projects() {
        check(
            &format!(
                "{TABLE}struct P {{ v: int64 }}\nlet mut p = P {{ v: 1 }}\np = P {{ v: 2 }}\nfrom t |> select p.v as w"
            ),
            expect![[r#"
                struct Row { a, b }
                table t
                struct P { v }
                from t
                  |> select 2i64 as w
            "#]],
        );
    }

    #[test]
    fn struct_projected_and_used_whole() {
        check(
            &format!(
                "{TABLE}struct P {{ v: int64 }}\nlet p = P {{ v: 40 }}\nfrom t |> select p.v as w\np"
            ),
            expect![[r#"
                struct Row { a, b }
                table t
                struct P { v }
                from t
                  |> select 40i64 as w
                let p = P { v: 40i64 }
                p
            "#]],
        );
    }

    #[test]
    fn struct_used_whole_twice_is_materialized_once() {
        check(
            &format!(
                "{TABLE}struct P {{ v: int64 }}\nlet p = P {{ v: 40 }}\nfrom t |> select a\np\np"
            ),
            expect![[r#"
                struct Row { a, b }
                table t
                struct P { v }
                from t
                  |> select a
                let p = P { v: 40i64 }
                p
                p
            "#]],
        );
    }

    #[test]
    fn list_used_whole_is_materialized() {
        check(
            &format!("{TABLE}let xs = [1, 2]\nfrom t |> where a in xs"),
            expect![[r#"
                struct Row { a, b }
                table t
                from t
                  |> where xs = [1i32, 2i32]; %r0 = in(a, xs); %r0
            "#]],
        );
    }

    #[test]
    fn list_used_whole_twice_is_materialized_once() {
        check(
            &format!("{TABLE}let xs = [1, 2]\nfrom t |> select a\nxs\nxs"),
            expect![[r#"
                struct Row { a, b }
                table t
                from t
                  |> select a
                let xs = [1i64, 2i64]
                xs
                xs
            "#]],
        );
    }

    #[test]
    fn nested_struct_deep_projection_drops_both() {
        check(
            &format!(
                "{TABLE}struct I {{ x: int64 }}\nstruct O {{ i: I }}\nlet deep = I {{ x: 5 }}\nlet outer = O {{ i: deep }}\nfrom t |> select outer.i.x as w"
            ),
            expect![[r#"
                struct Row { a, b }
                table t
                struct I { x }
                struct O { i }
                from t
                  |> select 5i64 as w
            "#]],
        );
    }

    #[test]
    fn nested_struct_whole_use_materializes_inner_first() {
        check(
            &format!(
                "{TABLE}struct I {{ x: int64 }}\nstruct O {{ i: I }}\nlet deep = I {{ x: 5 }}\nlet outer = O {{ i: deep }}\nfrom t |> select a\nouter"
            ),
            expect![[r#"
                struct Row { a, b }
                table t
                struct I { x }
                struct O { i }
                from t
                  |> select a
                let deep = I { x: 5i64 }
                let outer = O { i: deep }
                outer
            "#]],
        );
    }

    #[test]
    fn dead_struct_with_list_field_drops_both() {
        check(
            &format!(
                "{TABLE}struct S {{ xs: List[int64] }}\nlet s = S {{ xs: [1, 2] }}\nfrom t |> select a"
            ),
            expect![[r#"
                struct Row { a, b }
                table t
                struct S { xs }
                from t
                  |> select a
            "#]],
        );
    }

    #[test]
    fn struct_with_list_field_whole_use_materializes_both() {
        check(
            &format!(
                "{TABLE}struct S {{ xs: List[int64] }}\nlet s = S {{ xs: [1, 2] }}\nfrom t |> select a\ns"
            ),
            expect![[r#"
                struct Row { a, b }
                table t
                struct S { xs }
                from t
                  |> select a
                let %t0 = [1i64, 2i64]
                let s = S { xs: %t0 }
                s
            "#]],
        );
    }

    #[test]
    fn mixed_dead_and_projected_aggregates_all_drop() {
        check(
            &format!(
                "{TABLE}struct P {{ v: int64 }}\nlet dead = P {{ v: 1 }}\nlet live = P {{ v: 2 }}\nlet nums = [9, 9]\nfrom t |> select live.v as w"
            ),
            expect![[r#"
                struct Row { a, b }
                table t
                struct P { v }
                from t
                  |> select 2i64 as w
            "#]],
        );
    }
}
