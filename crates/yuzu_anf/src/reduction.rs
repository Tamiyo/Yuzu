use std::collections::{HashMap, HashSet};

use id_arena::Arena;
use yuzu_core::adt::StringInterner;

use crate::AnfCtx;
use crate::reduction::environment::Environment;
use crate::reduction::liveness::DeadCodeElimination;
use crate::source_map::AnfSourceMap;
use crate::{
    Atom, AtomId, Binding, BindingId, Expr, ExprId, Ident, Rel, RelId, Root, Stmt, StmtId,
    TreeCopier, TreeCopy,
};

mod atom;
mod environment;
mod expr;
mod fold;
mod liveness;
mod rel;
mod stmt;
mod term;
mod value;

pub fn reduce(
    root: &Root,
    anf: &mut AnfCtx,
    interner: &mut StringInterner,
    source_map: &mut AnfSourceMap,
) -> Root {
    let reduced = AnfReduceCtx::take(anf);
    AnfReducer {
        reduced: &reduced,
        anf,
        interner,
        source_map,
        funcs: FuncTable::default(),
        inlining: HashSet::new(),
        materialized: HashSet::new(),
        emitted_funcs: HashMap::new(),
        temp: 0,
    }
    .reduce_root(root)
}

struct AnfReduceCtx {
    stmts: Arena<Stmt>,
    exprs: Arena<Expr>,
    rels: Arena<Rel>,
}

impl AnfReduceCtx {
    fn take(anf: &mut AnfCtx) -> Self {
        Self {
            stmts: std::mem::take(&mut anf.stmts),
            exprs: std::mem::take(&mut anf.exprs),
            rels: std::mem::take(&mut anf.rels),
        }
    }

    fn stmt(&self, id: StmtId) -> &Stmt {
        &self.stmts[id]
    }

    fn expr(&self, id: ExprId) -> &Expr {
        &self.exprs[id]
    }

    fn rel(&self, id: RelId) -> &Rel {
        &self.rels[id]
    }
}

#[derive(Default)]
struct FuncTable {
    funcs: HashMap<BindingId, StmtId>,
}

impl FuncTable {
    fn register(&mut self, binding: BindingId, stmt: StmtId) {
        self.funcs.insert(binding, stmt);
    }

    fn func(&self, binding: BindingId) -> Option<StmtId> {
        self.funcs.get(&binding).copied()
    }
}

struct AnfReducer<'r> {
    reduced: &'r AnfReduceCtx,
    anf: &'r mut AnfCtx,
    interner: &'r mut StringInterner,
    source_map: &'r mut AnfSourceMap,
    funcs: FuncTable,
    inlining: HashSet<BindingId>,
    materialized: HashSet<BindingId>,
    emitted_funcs: HashMap<BindingId, StmtId>,
    temp: usize,
}

impl AnfReducer<'_> {
    fn reduce_root(mut self, root: &Root) -> Root {
        self.register_funcs(&root.stmts);

        let mut env = Environment::default();
        let stmts = self.reduce_stmts(&root.stmts, &mut env);
        let stmts = self.sweep_dead_functions(stmts);
        Root { stmts }
    }

    /// Carries an input node's source position over to the node that replaces
    /// it in the reduced program.
    fn bind_expr_origin(&mut self, id: ExprId, origin: ExprId) {
        if let Some(span) = self.source_map.expr(origin) {
            self.source_map.bind_expr(id, span);
        }
    }

    fn bind_stmt_origin(&mut self, id: StmtId, origin: StmtId) {
        if let Some(span) = self.source_map.stmt(origin) {
            self.source_map.bind_stmt(id, span);
        }
    }

    fn register_funcs(&mut self, stmts: &[StmtId]) {
        let input = self.reduced;
        for &stmt_id in stmts {
            match input.stmt(stmt_id) {
                Stmt::Func { binding, body, .. } => {
                    self.funcs.register(*binding, stmt_id);
                    if let Some(body) = body
                        && let Stmt::Block { stmts } = input.stmt(*body)
                    {
                        self.register_funcs(stmts);
                    }
                }
                Stmt::Block { stmts } => self.register_funcs(stmts),
                _ => {}
            }
        }
    }

    fn bind_to_temp(&mut self, expr: Expr, origin: ExprId, out: &mut Vec<StmtId>) -> AtomId {
        let ty = match expr {
            Expr::Call { ty, .. }
            | Expr::FuncCall { ty, .. }
            | Expr::MethodCall { ty, .. }
            | Expr::StructInit { ty, .. }
            | Expr::ListInit { ty, .. }
            | Expr::AggCall { ty, .. }
            | Expr::ExternCall { ty, .. } => ty,
            Expr::Atom { .. } => unreachable!("bind_to_temp only binds computations"),
            Expr::Rel(_) => unreachable!("a query is a let value, never bound to a temp"),
        };

        let name = self.fresh_temp();
        let binding = self.anf.alloc_binding(Binding { name, ty });
        let expr = self.anf.alloc_expr(expr);
        self.bind_expr_origin(expr, origin);
        out.push(self.anf.alloc_stmt(Stmt::Let { binding, expr }));
        self.anf.intern_atom(Atom::Var { binding })
    }

    fn fresh_temp(&mut self) -> Ident {
        let name = self.interner.intern(&format!("%r{}", self.temp));
        self.temp += 1;
        Ident { name }
    }
}

/// Copying: the reduced program owns its trees, so statements that survive
/// reduction verbatim (declarations, kept function bodies) are rebuilt in the
/// output arenas. Each node's `TreeCopy` walks its fields; these three methods
/// are the id translation — read the input node, rebuild it, allocate it in
/// the output. Atoms and bindings are shared and pass through.
impl TreeCopier for AnfReducer<'_> {
    fn copy_stmt(&mut self, id: StmtId) -> StmtId {
        let input = self.reduced;
        let stmt = input.stmt(id).copy_tree(self);
        self.anf.alloc_stmt(stmt)
    }

    fn copy_expr(&mut self, id: ExprId) -> ExprId {
        let input = self.reduced;
        let expr = input.expr(id).copy_tree(self);
        self.anf.alloc_expr(expr)
    }

    fn copy_rel(&mut self, id: RelId) -> RelId {
        let input = self.reduced;
        let rel = input.rel(id).copy_tree(self);
        self.anf.alloc_rel(rel)
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use expect_test::Expect;
    use yuzu_ast::ast::{AstNode, Root as AstRoot};
    use yuzu_core::adt::StringInterner;
    use yuzu_diagnostics::{diagnostics::engine::DiagnosticsEngine, source_map::SourceMap};
    use yuzu_hir::HirCtx;
    use yuzu_lexer::lexer::{Lexer, Token};
    use yuzu_types::TypeCtx;

    use crate::{AnfCtx, dump, lower, reduce};

    pub(crate) const TABLE: &str = "struct Row { a: int32, b: int32 }\ntable t = Row\n";

    pub(crate) fn check(input: &str, expected: Expect) {
        let mut interner = StringInterner::new();
        let mut diagnostics = DiagnosticsEngine::new();
        let mut sources = SourceMap::new();
        let source_id = sources.add("test".to_string(), input.to_string());

        let tokens: Vec<Token> = Lexer::new(input).collect();
        let syntax = yuzu_parser::parse(&tokens, &mut diagnostics, source_id);
        let ast_root = AstRoot::cast(syntax).expect("root node");

        let mut hir = HirCtx::new();
        let (hir_root, hir_source_map) = yuzu_hir::lower(
            ast_root,
            &mut hir,
            &mut interner,
            &mut diagnostics,
            source_id,
        );

        let mut types = TypeCtx::new();
        let inference = yuzu_hir::infer(
            &hir_root,
            &hir,
            &yuzu_types::Builtins,
            &mut interner,
            &mut types,
            &mut diagnostics,
            &hir_source_map,
        );

        let messages: Vec<&str> = diagnostics
            .diagnostics()
            .iter()
            .map(|d| d.message.as_str())
            .collect();
        assert!(
            messages.is_empty(),
            "program should type-check cleanly, got: {messages:?}"
        );

        let mut anf = AnfCtx::new();
        let (anf_root, mut anf_source_map) = lower(
            &hir_root,
            &hir,
            &inference,
            &types,
            &mut anf,
            &mut interner,
            &mut diagnostics,
            &hir_source_map,
        );

        let reduced = reduce(&anf_root, &mut anf, &mut interner, &mut anf_source_map);
        expected.assert_eq(&dump(&anf, &interner, &reduced));
    }
}
