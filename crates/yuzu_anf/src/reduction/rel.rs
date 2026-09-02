use crate::reduction::{AnfReducer, environment::Environment};
use crate::{AggregateItem, JoinCondition, Rel, RelId, SelectItem, SetItem, Thunk, TreeCopier};

impl AnfReducer<'_> {
    pub(crate) fn reduce_rel(&mut self, id: RelId, env: &mut Environment) -> RelId {
        let input = self.reduced;
        let rel = match input.rel(id) {
            Rel::From { .. } => return self.copy_rel(id),
            Rel::Join {
                left,
                right,
                kind,
                condition,
                ty,
            } => Rel::Join {
                left: self.reduce_rel(*left, env),
                right: self.reduce_rel(*right, env),
                kind: *kind,
                condition: match condition {
                    JoinCondition::On(thunk) => JoinCondition::On(self.reduce_thunk(thunk, env)),
                    JoinCondition::Using(columns) => JoinCondition::Using(columns.clone()),
                },
                ty: *ty,
            },
            Rel::Select {
                input: source,
                items,
                ty,
            } => Rel::Select {
                input: self.reduce_rel(*source, env),
                items: items
                    .iter()
                    .map(|item| self.reduce_select_item(item, env))
                    .collect(),
                ty: *ty,
            },
            Rel::Where {
                input: source,
                predicate,
                ty,
            } => Rel::Where {
                input: self.reduce_rel(*source, env),
                predicate: self.reduce_thunk(predicate, env),
                ty: *ty,
            },
            Rel::Distinct { input: source, ty } => Rel::Distinct {
                input: self.reduce_rel(*source, env),
                ty: *ty,
            },
            Rel::Drop {
                input: source,
                columns,
                ty,
            } => Rel::Drop {
                input: self.reduce_rel(*source, env),
                columns: columns.clone(),
                ty: *ty,
            },
            Rel::Rename {
                input: source,
                items,
                ty,
            } => Rel::Rename {
                input: self.reduce_rel(*source, env),
                items: items.clone(),
                ty: *ty,
            },
            Rel::Set {
                input: source,
                items,
                ty,
            } => Rel::Set {
                input: self.reduce_rel(*source, env),
                items: items
                    .iter()
                    .map(|item| SetItem {
                        column: item.column,
                        value: self.reduce_thunk(&item.value, env),
                    })
                    .collect(),
                ty: *ty,
            },
            Rel::Limit {
                input: source,
                count,
                offset,
                ty,
            } => Rel::Limit {
                input: self.reduce_rel(*source, env),
                count: self.reduce_thunk(count, env),
                offset: offset.as_ref().map(|offset| self.reduce_thunk(offset, env)),
                ty: *ty,
            },
            Rel::Alias {
                input: source,
                alias,
                ty,
            } => Rel::Alias {
                input: self.reduce_rel(*source, env),
                alias: *alias,
                ty: *ty,
            },
            Rel::Extend {
                input: source,
                items,
                ty,
            } => Rel::Extend {
                input: self.reduce_rel(*source, env),
                items: items
                    .iter()
                    .map(|item| self.reduce_select_item(item, env))
                    .collect(),
                ty: *ty,
            },
            Rel::Aggregate {
                input: source,
                items,
                groups,
                ty,
            } => Rel::Aggregate {
                input: self.reduce_rel(*source, env),
                items: items
                    .iter()
                    .map(|item| AggregateItem {
                        body: self.reduce_thunk(&item.body, env),
                        alias: item.alias,
                    })
                    .collect(),
                groups: groups.clone(),
                ty: *ty,
            },
        };

        self.anf.alloc_rel(rel)
    }

    fn reduce_select_item(&mut self, item: &SelectItem, env: &mut Environment) -> SelectItem {
        SelectItem {
            body: self.reduce_thunk(&item.body, env),
            alias: item.alias,
        }
    }

    fn reduce_thunk(&mut self, thunk: &Thunk, env: &mut Environment) -> Thunk {
        let outer_materialized = std::mem::take(&mut self.materialized);
        let mut out = Vec::new();
        self.reduce_block(&thunk.stmts, env, &mut out);
        let value = self.reduce_atom(thunk.value, env, &mut out);
        self.materialized = outer_materialized;
        Thunk {
            stmts: out.into_boxed_slice(),
            value,
        }
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::reduction::test_support::{TABLE, check};

    #[test]
    fn agg_fn_inlines_to_the_handwritten_body() {
        let handwritten = &format!("{TABLE}from t |> aggregate max(a) - min(a) as v group by b");
        let through_fn = &format!(
            "{TABLE}agg fn spread(x: int32) -> int32 {{ return max(x) - min(x) }}\nfrom t |> aggregate spread(a) as v group by b"
        );
        check(
            handwritten,
            expect![[r#"
            struct Row { a, b }
            table t
            from t
              |> aggregate %r0 = max(a); %r1 = min(a); %r2 = sub(%r0, %r1); %r2 as v group by b
        "#]],
        );
        check(
            through_fn,
            expect![[r#"
            struct Row { a, b }
            table t
            from t
              |> aggregate %r0 = max(a); %r1 = min(a); %r2 = sub(%r0, %r1); %r2 as v group by b
        "#]],
        );
    }

    #[test]
    fn count_distinct_is_its_own_function() {
        check(
            &format!("{TABLE}from t |> aggregate count_distinct(a) as kinds group by b"),
            expect![[r#"
                struct Row { a, b }
                table t
                from t
                  |> aggregate count_distinct(a) as kinds group by b
            "#]],
        );
    }

    #[test]
    fn extern_call_survives_with_folded_arguments() {
        check(
            &format!(
                "{TABLE}external fn clamp(x: int32, low: int32) -> int32\nfrom t |> select clamp(a, 2 + 3) as v"
            ),
            expect![[r#"
                struct Row { a, b }
                table t
                from t
                  |> select clamp(a, 5i32) as v
            "#]],
        );
    }

    #[test]
    fn aggregate_folds_inside_measure_arguments() {
        check(
            &format!("{TABLE}from t |> aggregate sum(a * (1 + 1)) as v group by b"),
            expect![[r#"
                struct Row { a, b }
                table t
                from t
                  |> aggregate %r0 = mul(a, 2i32); %r1 = sum(%r0); %r1 as v group by b
            "#]],
        );
    }

    #[test]
    fn aggregate_inlines_a_scalar_function_in_measure_arguments() {
        check(
            &format!(
                "{TABLE}fn double(x: int32) -> int32 {{ return x * 2 }}\nfrom t |> aggregate min(double(a)) as v"
            ),
            expect![[r#"
                struct Row { a, b }
                table t
                from t
                  |> aggregate %r0 = mul(a, 2i32); %r1 = min(%r0); %r1 as v
            "#]],
        );
    }

    #[test]
    fn folds_query_column_constant() {
        check(
            &format!("{TABLE}let q = from t |> select a, 1 + 2 as three"),
            expect![[r#"
                struct Row { a, b }
                table t
                let q = from t
                  |> select a, 3i64 as three
            "#]],
        );
    }

    #[test]
    fn keeps_runtime_predicate() {
        check(
            &format!("{TABLE}let q = from t |> where a > 0 |> select a"),
            expect![[r#"
                struct Row { a, b }
                table t
                let q = from t
                  |> where gt(a, 0i32)
                  |> select a
            "#]],
        );
    }

    const JOIN_TABLES: &str = "struct Other { a: int32, c: int32 }\ntable u = Other\nstruct Codes { c: int32, d: int32 }\ntable v = Codes\n";

    #[test]
    fn folds_constant_in_join_condition() {
        check(
            &format!("{TABLE}{JOIN_TABLES}let q = from t |> join v as x on b == x.c + (1 + 1)"),
            expect![[r#"
                struct Row { a, b }
                table t
                struct Other { a, c }
                table u
                struct Codes { c, d }
                table v
                let q = from t
                  |> inner join from v as x on %r0 = add(c, 2i32); %r1 = eq(b, %r0); %r1
            "#]],
        );
    }

    #[test]
    fn keeps_join_using_columns() {
        check(
            &format!("{TABLE}{JOIN_TABLES}let q = from t |> right join u using (a)"),
            expect![[r#"
                struct Row { a, b }
                table t
                struct Other { a, c }
                table u
                struct Codes { c, d }
                table v
                let q = from t
                  |> right join from u using a
            "#]],
        );
    }

    #[test]
    fn folds_the_value_a_set_assigns() {
        check(
            &format!("{TABLE}let q = from t |> set a = 1 + 2"),
            expect![[r#"
                struct Row { a, b }
                table t
                let q = from t
                  |> set a = 3i64
            "#]],
        );
    }

    #[test]
    fn folds_a_limit_count() {
        check(
            &format!("{TABLE}let q = from t |> limit 2 * 5 offset 1"),
            expect![[r#"
                struct Row { a, b }
                table t
                let q = from t
                  |> limit 10i64 offset 1i64
            "#]],
        );
    }

    #[test]
    fn list_used_whole_across_two_thunks_is_self_contained() {
        check(
            &format!("{TABLE}let xs = [1, 2]\nfrom t |> where a in xs |> where b in xs"),
            expect![[r#"
                struct Row { a, b }
                table t
                from t
                  |> where xs = [1i32, 2i32]; %r0 = in(a, xs); %r0
                  |> where xs = [1i32, 2i32]; %r1 = in(b, xs); %r1
            "#]],
        );
    }
}
