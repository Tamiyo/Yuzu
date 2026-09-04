use id_arena::Arena;
use rustc_hash::FxHashMap;

mod anf;
mod lowering;
mod printer;
mod reduction;
mod source_map;
mod symbols;

pub use anf::*;
pub use lowering::lower;
pub use printer::dump;
pub use reduction::reduce;
pub use source_map::AnfSourceMap;

/// The program's query: the relational expression statement at its tail.
pub fn find_query(root: &Root, anf: &AnfCtx) -> Option<(RelId, StmtId)> {
    root.stmts.iter().rev().find_map(|&id| match anf.stmt(id) {
        Stmt::Expr { value } => match anf.expr(*value) {
            Expr::Rel(rel) => Some((*rel, id)),
            _ => None,
        },
        _ => None,
    })
}

#[derive(Default)]
pub struct AnfCtx {
    stmts: Arena<Stmt>,
    exprs: Arena<Expr>,
    rels: Arena<Rel>,
    atoms: Arena<Atom>,
    bindings: Arena<Binding>,
    interned: FxHashMap<Atom, AtomId>,
}

impl AnfCtx {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn alloc_stmt(&mut self, stmt: Stmt) -> StmtId {
        self.stmts.alloc(stmt)
    }

    pub fn stmt(&self, id: StmtId) -> &Stmt {
        &self.stmts[id]
    }

    pub fn alloc_expr(&mut self, expr: Expr) -> ExprId {
        self.exprs.alloc(expr)
    }

    pub fn expr(&self, id: ExprId) -> &Expr {
        &self.exprs[id]
    }

    pub fn alloc_rel(&mut self, rel: Rel) -> RelId {
        self.rels.alloc(rel)
    }

    pub fn rel(&self, id: RelId) -> &Rel {
        &self.rels[id]
    }

    pub fn alloc_binding(&mut self, binding: Binding) -> BindingId {
        self.bindings.alloc(binding)
    }

    pub fn binding(&self, id: BindingId) -> &Binding {
        &self.bindings[id]
    }

    pub fn intern_atom(&mut self, atom: Atom) -> AtomId {
        if let Some(&id) = self.interned.get(&atom) {
            return id;
        }
        let id = self.atoms.alloc(atom);
        self.interned.insert(atom, id);
        id
    }

    pub fn atom(&self, id: AtomId) -> &Atom {
        &self.atoms[id]
    }
}
