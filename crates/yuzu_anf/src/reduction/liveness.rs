use std::collections::HashSet;

use crate::reduction::AnfReducer;
use crate::{
    Atom, AtomId, BindingId, Expr, ExprId, JoinCondition, Rel, RelId, Stmt, StmtId, Thunk,
    TreeCopier,
};

pub(crate) trait DeadCodeElimination {
    fn sweep_dead_functions(&mut self, stmts: Vec<StmtId>) -> Box<[StmtId]>;
}

pub(crate) trait LivenessAnalysis {
    fn stmt_func_refs(&self, id: StmtId, refs: &mut Vec<BindingId>);
    fn expr_func_refs(&self, id: ExprId, refs: &mut Vec<BindingId>);
    fn rel_func_refs(&self, id: RelId, refs: &mut Vec<BindingId>);
    fn thunk_func_refs(&self, thunk: &Thunk, refs: &mut Vec<BindingId>);
    fn atom_func_refs(&self, id: AtomId, refs: &mut Vec<BindingId>);
}

impl DeadCodeElimination for AnfReducer<'_> {
    fn sweep_dead_functions(&mut self, mut stmts: Vec<StmtId>) -> Box<[StmtId]> {
        let mut refs = Vec::new();
        for &id in &stmts {
            if !matches!(self.anf.stmt(id), Stmt::Func { .. }) {
                self.stmt_func_refs(id, &mut refs);
            }
        }

        let mut live = HashSet::new();
        while let Some(binding) = refs.pop() {
            if !live.insert(binding) {
                continue;
            }
            let def = match self.emitted_funcs.get(&binding).copied() {
                Some(def) => def,
                // A live function declared in a nested scope has no definition
                // at the root yet; emit one.
                None => {
                    let Some(template) = self.funcs.func(binding) else {
                        continue;
                    };
                    let def = self.copy_stmt(template);
                    self.emitted_funcs.insert(binding, def);
                    stmts.push(def);
                    def
                }
            };
            self.stmt_func_refs(def, &mut refs);
        }

        stmts.retain(|&id| match self.anf.stmt(id) {
            Stmt::Func { binding, .. } => live.contains(binding),
            _ => true,
        });
        stmts.into_boxed_slice()
    }
}

impl LivenessAnalysis for AnfReducer<'_> {
    fn stmt_func_refs(&self, id: StmtId, refs: &mut Vec<BindingId>) {
        match self.anf.stmt(id) {
            Stmt::Struct { .. } | Stmt::Table { .. } => {}
            Stmt::Func { body, .. } => {
                if let Some(body) = body {
                    self.stmt_func_refs(*body, refs);
                }
            }
            Stmt::Block { stmts } => {
                for &stmt in stmts.iter() {
                    self.stmt_func_refs(stmt, refs);
                }
            }
            Stmt::Let { expr, .. } => self.expr_func_refs(*expr, refs),
            Stmt::Assign { target, value } => {
                self.atom_func_refs(*target, refs);
                self.expr_func_refs(*value, refs);
            }
            Stmt::Return { value } => {
                if let Some(value) = value {
                    self.atom_func_refs(*value, refs);
                }
            }
            Stmt::Expr { value } => self.expr_func_refs(*value, refs),
        }
    }

    fn expr_func_refs(&self, id: ExprId, refs: &mut Vec<BindingId>) {
        match self.anf.expr(id) {
            Expr::Call { args, .. }
            | Expr::ListInit { elements: args, .. }
            | Expr::AggCall { args, .. }
            | Expr::ExternCall { args, .. } => {
                for &arg in args.iter() {
                    self.atom_func_refs(arg, refs);
                }
            }
            Expr::FuncCall { callee, args, .. } => {
                self.atom_func_refs(*callee, refs);
                for &arg in args.iter() {
                    self.atom_func_refs(arg, refs);
                }
            }
            Expr::MethodCall { receiver, args, .. } => {
                self.atom_func_refs(*receiver, refs);
                for &arg in args.iter() {
                    self.atom_func_refs(arg, refs);
                }
            }
            Expr::StructInit { fields, .. } => {
                for field in fields.iter() {
                    self.atom_func_refs(field.value, refs);
                }
            }
            Expr::Rel(rel) => self.rel_func_refs(*rel, refs),
            Expr::Atom { value } => self.atom_func_refs(*value, refs),
        }
    }

    fn rel_func_refs(&self, id: RelId, refs: &mut Vec<BindingId>) {
        match self.anf.rel(id) {
            Rel::From { .. } => {}
            Rel::Join {
                left,
                right,
                condition,
                ..
            } => {
                self.rel_func_refs(*left, refs);
                self.rel_func_refs(*right, refs);
                if let JoinCondition::On(thunk) = condition {
                    self.thunk_func_refs(thunk, refs);
                }
            }
            Rel::Select { input, items, .. } | Rel::Extend { input, items, .. } => {
                self.rel_func_refs(*input, refs);
                for item in items.iter() {
                    self.thunk_func_refs(&item.body, refs);
                }
            }
            Rel::Aggregate { input, items, .. } => {
                self.rel_func_refs(*input, refs);
                for item in items.iter() {
                    self.thunk_func_refs(&item.body, refs);
                }
            }
            Rel::Where {
                input, predicate, ..
            } => {
                self.rel_func_refs(*input, refs);
                self.thunk_func_refs(predicate, refs);
            }
            Rel::Set { input, items, .. } => {
                self.rel_func_refs(*input, refs);
                for item in items.iter() {
                    self.thunk_func_refs(&item.value, refs);
                }
            }
            Rel::Limit {
                input,
                count,
                offset,
                ..
            } => {
                self.rel_func_refs(*input, refs);
                self.thunk_func_refs(count, refs);
                if let Some(offset) = offset {
                    self.thunk_func_refs(offset, refs);
                }
            }
            Rel::Distinct { input, .. }
            | Rel::Drop { input, .. }
            | Rel::Rename { input, .. }
            | Rel::Alias { input, .. } => {
                self.rel_func_refs(*input, refs);
            }
        }
    }

    fn thunk_func_refs(&self, thunk: &Thunk, refs: &mut Vec<BindingId>) {
        for &stmt in thunk.stmts.iter() {
            self.stmt_func_refs(stmt, refs);
        }
        self.atom_func_refs(thunk.value, refs);
    }

    fn atom_func_refs(&self, id: AtomId, refs: &mut Vec<BindingId>) {
        match *self.anf.atom(id) {
            Atom::FuncRef { binding, .. } => refs.push(binding),
            Atom::Field { base, .. } => self.atom_func_refs(base, refs),
            Atom::Var { .. } | Atom::Column { .. } | Atom::Const(_) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::reduction::test_support::{TABLE, check};

    #[test]
    fn inlines_call_and_drops_dead_function() {
        check(
            &format!(
                "{TABLE}def add_one(x: int32) -> int32 {{ return x + 1 }}\nfrom t |> select add_one(7) as v"
            ),
            expect![[r#"
                struct Row { a, b }
                table t
                from t
                  |> select 8i32 as v
            "#]],
        );
    }

    #[test]
    fn helper_of_kept_function_stays_live() {
        check(
            &format!(
                "{TABLE}def g(n: int64) -> int64 {{ return n + 1 }}\ndef f(n: int64) -> int64 {{ return g(n) * f(n - 1) }}\nfrom t |> select f(3) as v"
            ),
            expect![[r#"
                struct Row { a, b }
                table t
                fn g(n) {
                  let %t0 = add(n, 1i64)
                  return %t0
                }
                fn f(n) {
                  let %t1 = g(n)
                  let %t2 = sub(n, 1i64)
                  let %t3 = f(%t2)
                  let %t4 = mul(%t1, %t3)
                  return %t4
                }
                from t
                  |> select %r0 = f(2i64); %r1 = mul(4i64, %r0); %r1 as v
            "#]],
        );
    }
}
