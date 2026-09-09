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
    registry: &'i dyn yuzu_types::FunctionRegistry,
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
    expr_types: Table<TypeId>,
    rel_types: Table<TypeId>,
    columns: Table<u32>,
    builtin_calls: Table<BuiltinFunc>,
    group_keys: Table<Box<[u32]>>,
    extern_calls: Table<yuzu_core::adt::SymbolId>,
    stmt_types: Table<TypeId>,
    adjustments: Table<TypeId>,
}

/// A side table indexed by arena position — ids are dense, so a vector beats
/// hashing every node.
struct Table<T>(Vec<Option<T>>);

impl<T> Default for Table<T> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<T> Table<T> {
    fn set(&mut self, index: usize, value: T) {
        if self.0.len() <= index {
            self.0.resize_with(index + 1, || None);
        }
        self.0[index] = Some(value);
    }

    fn get(&self, index: usize) -> Option<&T> {
        self.0.get(index).and_then(Option::as_ref)
    }
}

impl<T: Copy> Table<T> {
    fn copied(&self, index: usize) -> Option<T> {
        self.get(index).copied()
    }
}

impl InferenceResult {
    pub fn expr_ty(&self, id: ExprId) -> Option<TypeId> {
        self.expr_types.copied(id.index())
    }

    pub fn rel_ty(&self, id: RelId) -> Option<TypeId> {
        self.rel_types.copied(id.index())
    }

    /// Which column of its stage's row an expression reads, for the expressions
    /// that are column references. `None` for everything else.
    pub fn column(&self, id: ExprId) -> Option<u32> {
        self.columns.copied(id.index())
    }

    /// Which builtin a call expression invokes, for the calls that resolved to
    /// one. `None` for everything else.
    pub fn builtin_call(&self, id: ExprId) -> Option<BuiltinFunc> {
        self.builtin_calls.copied(id.index())
    }

    /// The external function a call resolved to, carried by name to the plan.
    pub fn extern_call(&self, id: ExprId) -> Option<yuzu_core::adt::SymbolId> {
        self.extern_calls.copied(id.index())
    }

    /// The input-row position of each of an `aggregate` stage's group keys,
    /// in declaration order.
    pub fn group_keys(&self, id: RelId) -> Option<&[u32]> {
        self.group_keys.get(id.index()).map(|keys| &**keys)
    }

    /// The declared type of a declaration statement (struct, table, or func).
    pub fn stmt_ty(&self, id: StmtId) -> Option<TypeId> {
        self.stmt_types.copied(id.index())
    }

    pub fn adjustment(&self, id: ExprId) -> Option<TypeId> {
        self.adjustments.copied(id.index())
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
        self.result.expr_types.set(id.index(), ty);
        ty
    }

    fn bind_column(&mut self, id: ExprId, column: u32) {
        self.result.columns.set(id.index(), column);
    }

    fn bind_builtin_call(&mut self, id: ExprId, func: BuiltinFunc) {
        self.result.builtin_calls.set(id.index(), func);
    }

    fn bind_extern_call(&mut self, id: ExprId, name: yuzu_core::adt::SymbolId) {
        self.result.extern_calls.set(id.index(), name);
    }

    fn bind_group_keys(&mut self, id: RelId, keys: &[u32]) {
        self.result.group_keys.set(id.index(), keys.into());
    }

    fn bind_rel_ty(&mut self, id: RelId, ty: TypeId) -> TypeId {
        self.result.rel_types.set(id.index(), ty);
        ty
    }

    fn bind_stmt_ty(&mut self, id: StmtId, ty: TypeId) -> TypeId {
        self.result.stmt_types.set(id.index(), ty);
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
        registry: &dyn yuzu_types::FunctionRegistry,
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
