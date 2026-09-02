use std::collections::HashMap;

use yuzu_core::adt::StringInterner;
use yuzu_diagnostics::diagnostics::engine::DiagnosticsEngine;
use yuzu_types::{BuiltinFunc, InferKind, TypeCtx, TypeId, TypeUnifier};

use crate::{ExprId, HirCtx, HirSourceMap, RelId, Root, StmtId};

mod coerce;
mod concretize;
mod inference;
mod op;
mod symbols;

use self::inference::TypeInferrer;

#[allow(clippy::too_many_arguments)]
pub fn infer<'i>(
    root: &Root,
    hir: &'i HirCtx,
    registry: &'i dyn yuzu_types::Registry,
    interner: &'i mut StringInterner,
    types: &'i mut TypeCtx,
    diagnostics: &'i mut DiagnosticsEngine,
    source_map: &'i HirSourceMap,
) -> InferenceResult {
    let mut ctx =
        TypeInferrer::new(hir, registry, types, interner, diagnostics, source_map).run(root);
    ctx.concretize();
    ctx.finish()
}

#[derive(Default)]
pub struct InferenceResult {
    expr_types: HashMap<ExprId, TypeId>,
    rel_types: HashMap<RelId, TypeId>,
    columns: HashMap<ExprId, u32>,
    builtin_calls: HashMap<ExprId, BuiltinFunc>,
    group_keys: HashMap<RelId, Box<[u32]>>,
    extern_calls: HashMap<ExprId, yuzu_core::adt::SymbolId>,
    stmt_types: HashMap<StmtId, TypeId>,
    adjustments: HashMap<ExprId, TypeId>,
}

impl InferenceResult {
    pub fn expr_ty(&self, id: ExprId) -> Option<TypeId> {
        self.expr_types.get(&id).copied()
    }

    pub fn rel_ty(&self, id: RelId) -> Option<TypeId> {
        self.rel_types.get(&id).copied()
    }

    /// Which column of its stage's row an expression reads, for the expressions
    /// that are column references. `None` for everything else.
    pub fn column(&self, id: ExprId) -> Option<u32> {
        self.columns.get(&id).copied()
    }

    /// Which builtin a call expression invokes, for the calls that resolved to
    /// one. `None` for everything else.
    pub fn builtin_call(&self, id: ExprId) -> Option<BuiltinFunc> {
        self.builtin_calls.get(&id).copied()
    }

    /// The external function a call resolved to, carried by name to the plan.
    pub fn extern_call(&self, id: ExprId) -> Option<yuzu_core::adt::SymbolId> {
        self.extern_calls.get(&id).copied()
    }

    /// The input-row position of each of an `aggregate` stage's group keys,
    /// in declaration order.
    pub fn group_keys(&self, id: RelId) -> Option<&[u32]> {
        self.group_keys.get(&id).map(|keys| &**keys)
    }

    /// The declared type of a declaration statement (struct, table, or func).
    pub fn stmt_ty(&self, id: StmtId) -> Option<TypeId> {
        self.stmt_types.get(&id).copied()
    }

    pub fn adjustment(&self, id: ExprId) -> Option<TypeId> {
        self.adjustments.get(&id).copied()
    }
}

pub(crate) struct InferCtx<'i> {
    types: &'i mut TypeCtx,
    unifier: TypeUnifier,
    result: InferenceResult,
}

impl<'i> InferCtx<'i> {
    fn new(types: &'i mut TypeCtx) -> Self {
        Self {
            types,
            unifier: TypeUnifier::new(),
            result: InferenceResult::default(),
        }
    }

    fn finish(self) -> InferenceResult {
        self.result
    }

    fn fresh_var(&mut self, kind: InferKind) -> TypeId {
        self.unifier.fresh_var(kind, self.types)
    }

    fn unify(&mut self, a: TypeId, b: TypeId) -> bool {
        self.unifier.unify(a, b, self.types)
    }

    fn resolve(&mut self, ty: TypeId) -> TypeId {
        self.unifier.resolve(ty, self.types)
    }

    fn is_numeric(&mut self, id: TypeId) -> bool {
        let id = self.resolve(id);
        let ty = self.types.ty(id);
        ty.is_int() || ty.is_float()
    }

    fn is_int(&mut self, id: TypeId) -> bool {
        let id = self.resolve(id);
        self.types.ty(id).is_int()
    }

    fn expr_ty(&self, id: ExprId) -> Option<TypeId> {
        self.result.expr_ty(id)
    }

    fn bind_expr_ty(&mut self, id: ExprId, ty: TypeId) -> TypeId {
        self.result.expr_types.insert(id, ty);
        ty
    }

    fn bind_column(&mut self, id: ExprId, column: u32) {
        self.result.columns.insert(id, column);
    }

    fn bind_builtin_call(&mut self, id: ExprId, func: BuiltinFunc) {
        self.result.builtin_calls.insert(id, func);
    }

    fn bind_extern_call(&mut self, id: ExprId, name: yuzu_core::adt::SymbolId) {
        self.result.extern_calls.insert(id, name);
    }

    fn bind_group_keys(&mut self, id: RelId, keys: &[u32]) {
        self.result.group_keys.insert(id, keys.into());
    }

    fn bind_rel_ty(&mut self, id: RelId, ty: TypeId) -> TypeId {
        self.result.rel_types.insert(id, ty);
        ty
    }

    fn bind_stmt_ty(&mut self, id: StmtId, ty: TypeId) -> TypeId {
        self.result.stmt_types.insert(id, ty);
        ty
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use expect_test::Expect;
    use yuzu_core::adt::StringInterner;
    use yuzu_diagnostics::{diagnostics::engine::DiagnosticsEngine, source_map::SourceMap};
    use yuzu_types::TypeCtx;

    use crate::{HirCtx, HirSourceMap, Root};

    pub(crate) fn check(
        build: impl FnOnce(&mut HirCtx, &mut StringInterner) -> Root,
        expected: Expect,
    ) {
        let mut hir = HirCtx::new();
        let mut interner = StringInterner::new();
        let root = build(&mut hir, &mut interner);

        let mut types = TypeCtx::new();
        let mut diagnostics = DiagnosticsEngine::new();
        let source_map = HirSourceMap::default();
        let mut sources = SourceMap::new();
        sources.add("test".into(), String::new());

        super::infer(
            &root,
            &hir,
            &yuzu_types::Builtins,
            &mut interner,
            &mut types,
            &mut diagnostics,
            &source_map,
        );

        let rendered = diagnostics
            .diagnostics()
            .iter()
            .map(|d| d.message.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        expected.assert_eq(&rendered);
    }

    pub(crate) fn check_src(input: &str, expected: Expect) {
        check_src_with(&yuzu_types::Builtins, input, expected);
    }

    pub(crate) fn check_src_with(
        registry: &dyn yuzu_types::Registry,
        input: &str,
        expected: Expect,
    ) {
        use yuzu_ast::ast::{AstNode, Root as AstRoot};
        use yuzu_lexer::lexer::{Lexer, Token};

        let mut hir = HirCtx::new();
        let mut interner = StringInterner::new();
        let mut diagnostics = DiagnosticsEngine::new();
        let mut sources = SourceMap::new();
        let source_id = sources.add("test".into(), input.to_string());

        let tokens: Vec<Token> = Lexer::new(input).collect();
        let syntax = yuzu_parser::parse(&tokens, &mut diagnostics, source_id);
        let ast_root = AstRoot::cast(syntax).expect("root node");

        let (root, source_map) = crate::lower(
            ast_root,
            &mut hir,
            &mut interner,
            &mut diagnostics,
            source_id,
        );

        let mut types = TypeCtx::new();
        super::infer(
            &root,
            &hir,
            registry,
            &mut interner,
            &mut types,
            &mut diagnostics,
            &source_map,
        );

        let rendered = diagnostics
            .diagnostics()
            .iter()
            .map(|d| d.message.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        expected.assert_eq(&rendered);
    }
}
