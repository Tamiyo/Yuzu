use std::collections::HashMap;

use yuzu_diagnostics::{diagnostics::Span, source_map::SourceId};
use yuzu_syntax::SyntaxNodePtr;

use crate::{ExprId, RelId, StmtId, TypeAnnotationId};

/// Where each node came from — file and range both, so a program assembled
/// from several files reports every diagnostic against the right source.
#[derive(Default)]
pub struct HirSourceMap {
    exprs: HashMap<ExprId, Span>,
    rels: HashMap<RelId, Span>,
    stmts: HashMap<StmtId, Span>,
    annotations: HashMap<TypeAnnotationId, Span>,
}

impl HirSourceMap {
    pub fn bind_expr(&mut self, id: ExprId, source: SourceId, ptr: SyntaxNodePtr) {
        self.exprs.insert(id, span(source, ptr));
    }

    pub fn bind_annotation(&mut self, id: TypeAnnotationId, source: SourceId, ptr: SyntaxNodePtr) {
        self.annotations.insert(id, span(source, ptr));
    }

    pub fn annotation(&self, id: TypeAnnotationId) -> Option<Span> {
        self.annotations.get(&id).copied()
    }

    pub fn bind_rel(&mut self, id: RelId, source: SourceId, ptr: SyntaxNodePtr) {
        self.rels.insert(id, span(source, ptr));
    }

    pub fn bind_stmt(&mut self, id: StmtId, source: SourceId, ptr: SyntaxNodePtr) {
        self.stmts.insert(id, span(source, ptr));
    }

    pub fn expr(&self, id: ExprId) -> Option<Span> {
        self.exprs.get(&id).copied()
    }

    pub fn rel(&self, id: RelId) -> Option<Span> {
        self.rels.get(&id).copied()
    }

    pub fn stmt(&self, id: StmtId) -> Option<Span> {
        self.stmts.get(&id).copied()
    }
}

fn span(source: SourceId, ptr: SyntaxNodePtr) -> Span {
    Span {
        source_id: source,
        range: ptr.text_range(),
    }
}
