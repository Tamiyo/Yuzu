use yuzu_diagnostics::diagnostics::Span;

use crate::{AtomId, ExprId, StmtId};

/// Where each node came from — file and range both, so a program assembled
/// from several files reports every diagnostic against the right source.
/// Stored as vectors indexed by arena position: ids are dense, so this is
/// cheaper than hashing every node.
#[derive(Default)]
pub struct AnfSourceMap {
    stmts: Vec<Option<Span>>,
    exprs: Vec<Option<Span>>,
    atoms: Vec<Option<Span>>,
}

impl AnfSourceMap {
    pub fn bind_stmt(&mut self, id: StmtId, span: Span) {
        set(&mut self.stmts, id.index(), span);
    }

    pub fn bind_expr(&mut self, id: ExprId, span: Span) {
        set(&mut self.exprs, id.index(), span);
    }

    pub fn bind_atom(&mut self, id: AtomId, span: Span) {
        set(&mut self.atoms, id.index(), span);
    }

    pub fn stmt(&self, id: StmtId) -> Option<Span> {
        get(&self.stmts, id.index())
    }

    pub fn expr(&self, id: ExprId) -> Option<Span> {
        get(&self.exprs, id.index())
    }

    pub fn atom(&self, id: AtomId) -> Option<Span> {
        get(&self.atoms, id.index())
    }
}

fn set(spans: &mut Vec<Option<Span>>, index: usize, span: Span) {
    if spans.len() <= index {
        spans.resize(index + 1, None);
    }
    spans[index] = Some(span);
}

fn get(spans: &[Option<Span>], index: usize) -> Option<Span> {
    spans.get(index).copied().flatten()
}
