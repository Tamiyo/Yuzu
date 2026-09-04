use yuzu_diagnostics::{diagnostics::Span, source_map::SourceId};
use yuzu_syntax::SyntaxNodePtr;

use crate::{ExprId, RelId, StmtId, TypeAnnotationId};

/// Where each node came from — file and range both, so a program assembled
/// from several files reports every diagnostic against the right source.
/// Stored as vectors indexed by arena position: ids are dense, so this is
/// cheaper than hashing every node.
#[derive(Default)]
pub struct HirSourceMap {
    exprs: Vec<Option<Span>>,
    rels: Vec<Option<Span>>,
    stmts: Vec<Option<Span>>,
    annotations: Vec<Option<Span>>,
}

impl HirSourceMap {
    pub fn bind_expr(&mut self, id: ExprId, source: SourceId, ptr: SyntaxNodePtr) {
        set(&mut self.exprs, id.index(), span(source, ptr));
    }

    pub fn bind_annotation(&mut self, id: TypeAnnotationId, source: SourceId, ptr: SyntaxNodePtr) {
        set(&mut self.annotations, id.index(), span(source, ptr));
    }

    pub fn annotation(&self, id: TypeAnnotationId) -> Option<Span> {
        get(&self.annotations, id.index())
    }

    pub fn bind_rel(&mut self, id: RelId, source: SourceId, ptr: SyntaxNodePtr) {
        set(&mut self.rels, id.index(), span(source, ptr));
    }

    pub fn bind_stmt(&mut self, id: StmtId, source: SourceId, ptr: SyntaxNodePtr) {
        set(&mut self.stmts, id.index(), span(source, ptr));
    }

    pub fn expr(&self, id: ExprId) -> Option<Span> {
        get(&self.exprs, id.index())
    }

    pub fn rel(&self, id: RelId) -> Option<Span> {
        get(&self.rels, id.index())
    }

    pub fn stmt(&self, id: StmtId) -> Option<Span> {
        get(&self.stmts, id.index())
    }
}

fn span(source: SourceId, ptr: SyntaxNodePtr) -> Span {
    Span {
        source_id: source,
        range: ptr.text_range(),
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
