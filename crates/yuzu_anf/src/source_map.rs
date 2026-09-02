use std::collections::HashMap;

use yuzu_diagnostics::diagnostics::Span;

use crate::{AtomId, ExprId, StmtId};

/// Where each node came from — file and range both, so a program assembled
/// from several files reports every diagnostic against the right source.
#[derive(Default)]
pub struct AnfSourceMap {
    stmts: HashMap<StmtId, Span>,
    exprs: HashMap<ExprId, Span>,
    atoms: HashMap<AtomId, Span>,
}

impl AnfSourceMap {
    pub fn bind_stmt(&mut self, id: StmtId, span: Span) {
        self.stmts.insert(id, span);
    }

    pub fn bind_expr(&mut self, id: ExprId, span: Span) {
        self.exprs.insert(id, span);
    }

    pub fn bind_atom(&mut self, id: AtomId, span: Span) {
        self.atoms.insert(id, span);
    }

    pub fn stmt(&self, id: StmtId) -> Option<Span> {
        self.stmts.get(&id).copied()
    }

    pub fn expr(&self, id: ExprId) -> Option<Span> {
        self.exprs.get(&id).copied()
    }

    pub fn atom(&self, id: AtomId) -> Option<Span> {
        self.atoms.get(&id).copied()
    }
}
