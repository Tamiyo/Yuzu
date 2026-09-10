use std::collections::HashSet;

use yuzu_core::adt::StringInterner;
use yuzu_diagnostics::diagnostics::{builder::DiagnosticBuilder, engine::DiagnosticsEngine};
use yuzu_types::{AggFunc, BuiltinFunc, Column, Func, InferKind, SymbolId, Type, TypeCtx, TypeId};

use crate::{
    Expr, ExprId, FuncParam, HirCtx, HirSourceMap, Ident, JoinCondition, Literal, Mutability, Op,
    Rel, RelId, RenameItem, Root, SelectItem, SetItem, Stmt, StmtId, StructField, StructFieldInit,
    TypeAnnotation, TypeAnnotationId,
    infer::{
        InferCtx,
        symbols::{Binding, ScopeKind, SymbolTable},
    },
};

/// Where inference stands relative to aggregation. `None` grouping means the
/// walk is in ordinary row code and every aggregate rule is idle.
#[derive(Default)]
struct AggregateScope {
    grouping: Option<Grouping>,
    /// Inside an aggregate call's arguments, where the walk is back in row
    /// context: columns need no key, and group-level values may not appear.
    in_arguments: bool,
    /// Whether any aggregate call appeared — an `agg fn` body without one is
    /// a mismarked function.
    saw_aggregate: bool,
    /// Names `let`-bound at group level. Their values are per-group, so a
    /// reference inside an aggregate's arguments would smuggle a group value
    /// back into row context.
    group_locals: HashSet<SymbolId>,
}

/// What is being aggregated over.
enum Grouping {
    /// The items of one `aggregate` stage: `input` is the row it collapses,
    /// and `keys` are the columns of that row surviving to group level.
    Stage { keys: Vec<ScopeKey> },
    /// An `agg fn` body: grouped by nothing until a call site provides keys.
    /// `stmt` is the declaration, so a call back to it is recognizable.
    FnBody { stmt: StmtId },
}

/// A grouping column: its position in the stage's input row — the space every
/// column reference resolves to while the items are inferred — and its name
/// for diagnostics.
struct ScopeKey {
    name: SymbolId,
    column: u32,
}

pub(crate) struct TypeInferrer<'i> {
    hir: &'i HirCtx,
    registry: &'i dyn yuzu_types::FunctionRegistry,
    infer: InferCtx<'i>,
    symbols: SymbolTable,
    interner: &'i mut StringInterner,
    diagnostics: &'i mut DiagnosticsEngine,
    source_map: &'i HirSourceMap,
    agg: AggregateScope,
}

impl<'i> TypeInferrer<'i> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        hir: &'i HirCtx,
        registry: &'i dyn yuzu_types::FunctionRegistry,
        types: &'i mut TypeCtx,
        interner: &'i mut StringInterner,
        diagnostics: &'i mut DiagnosticsEngine,
        source_map: &'i HirSourceMap,
    ) -> Self {
        Self {
            hir,
            registry,
            infer: InferCtx::new(types),
            symbols: SymbolTable::new(),
            interner,
            diagnostics,
            source_map,
            agg: AggregateScope::default(),
        }
    }

    pub(crate) fn run(mut self, root: &Root) -> InferCtx<'i> {
        // The chain resolves collisions first-wins, but scope binds overwrite,
        // so later entries must not rebind a name an earlier one claimed.
        let mut seen = HashSet::new();
        for entry in self.registry.entries() {
            if !seen.insert(entry.name) {
                continue;
            }
            let name = self.interner.intern(entry.name);
            self.symbols
                .bind_symbol(name, Binding::Builtin { func: entry.func });
        }
        self.hoist_and_infer(&root.stmts);
        self.infer
    }

    /// Registers every declaration in a statement list (so references can be
    /// forward), then types each statement.
    fn hoist_and_infer(&mut self, stmts: &[StmtId]) {
        for &stmt_id in stmts {
            if let Stmt::Struct { name, fields } = self.hir.stmt(stmt_id) {
                self.register_struct(stmt_id, name.symbol, fields);
            }
        }

        for &stmt_id in stmts {
            match self.hir.stmt(stmt_id) {
                Stmt::Table { name, row } => self.register_table(stmt_id, name.symbol, row.symbol),
                Stmt::InlineTable { name, fields } => {
                    self.register_inline_table(stmt_id, name.symbol, fields)
                }
                _ => {}
            }
        }

        for &stmt_id in stmts {
            if let Stmt::Func {
                name,
                params,
                ret_type_annotation,
                is_agg,
                is_external,
                ..
            } = self.hir.stmt(stmt_id)
            {
                self.register_func(
                    stmt_id,
                    name.symbol,
                    params,
                    *ret_type_annotation,
                    *is_agg,
                    *is_external,
                );
            }
        }

        for &stmt_id in stmts {
            self.infer_stmt(stmt_id);
        }
    }

    fn register_struct(&mut self, stmt_id: StmtId, name: SymbolId, fields: &[StructField]) {
        let field_tys = self.resolve_fields(fields);
        let struct_ty = self.infer.types.struct_ty(name, field_tys);
        self.infer.bind_stmt_ty(stmt_id, struct_ty);
        self.symbols.bind_type(name, struct_ty);
        self.symbols.bind_struct(name, stmt_id);
    }

    fn register_table(&mut self, stmt_id: StmtId, name: SymbolId, row: SymbolId) {
        let row_ty = match self.symbols.lookup_type(row) {
            Some(ty) if matches!(self.infer.types.ty(ty), Type::Struct(_)) => ty,
            _ => {
                let message = format!("`{}` is not a struct", self.interner.text(row));
                self.report_stmt(stmt_id, message);
                return;
            }
        };
        let relation_ty = self.infer.types.relation_of_row(row_ty, None);
        self.infer.bind_stmt_ty(stmt_id, relation_ty);
        self.symbols.bind_relation(name, relation_ty);
    }

    fn register_inline_table(&mut self, stmt_id: StmtId, name: SymbolId, fields: &[StructField]) {
        let field_tys = self.resolve_fields(fields);
        let row_ty = self.infer.types.struct_ty(name, field_tys);
        let relation_ty = self.infer.types.relation_of_row(row_ty, None);
        self.infer.bind_stmt_ty(stmt_id, relation_ty);
        self.symbols.bind_relation(name, relation_ty);
    }

    fn resolve_fields(&mut self, fields: &[StructField]) -> Vec<(SymbolId, TypeId)> {
        fields
            .iter()
            .map(|field| {
                (
                    field.name.symbol,
                    self.resolve_annotation(field.type_annotation),
                )
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn register_func(
        &mut self,
        stmt_id: StmtId,
        name: SymbolId,
        params: &[FuncParam],
        ret_annotation: TypeAnnotationId,
        is_agg: bool,
        is_external: bool,
    ) {
        let arg_tys: Vec<TypeId> = params
            .iter()
            .map(|param| self.resolve_annotation(param.type_annotation))
            .collect();

        let ret_ty = self.resolve_return_annotation(ret_annotation);
        let func_ty = self.infer.types.func_ty(arg_tys, ret_ty);
        self.infer.bind_stmt_ty(stmt_id, func_ty);

        self.symbols.bind_symbol(
            name,
            Binding::FuncStmt {
                stmt: stmt_id,
                ty: func_ty,
                is_agg,
                is_external,
            },
        );
    }

    fn infer_stmt(&mut self, stmt_id: StmtId) {
        match self.hir.stmt(stmt_id) {
            Stmt::Struct { .. } => {}
            Stmt::Impl { .. } => todo!(),
            Stmt::Trait { .. } => todo!(),
            Stmt::Func {
                params,
                ret_type_annotation,
                body,
                is_agg,
                is_external,
                ..
            } => self.infer_func_stmt(
                stmt_id,
                params,
                *ret_type_annotation,
                *body,
                *is_agg && !*is_external,
            ),
            Stmt::Block { stmts } => self.infer_block_stmt(stmts),
            Stmt::Table { .. } | Stmt::InlineTable { .. } => {}
            Stmt::Let {
                name,
                mutability,
                type_annotation,
                expr,
            } => self.infer_let_stmt(stmt_id, name.symbol, *mutability, *type_annotation, *expr),
            Stmt::Assign { target, value } => self.infer_assign_stmt(stmt_id, *target, *value),
            Stmt::Return { expr } => self.infer_return_stmt(stmt_id, *expr),
            Stmt::Expr { expr } => self.infer_expr_stmt(*expr),
            Stmt::Missing => {}
        }
    }

    fn infer_block_stmt(&mut self, stmts: &[StmtId]) {
        self.symbols.push_scope(ScopeKind::Block);
        self.hoist_and_infer(stmts);
        self.symbols.pop_scope();
    }

    fn infer_func_stmt(
        &mut self,
        stmt_id: StmtId,
        params: &[FuncParam],
        ret_annotation: TypeAnnotationId,
        body: Option<StmtId>,
        is_agg: bool,
    ) {
        let return_ty = self.resolve_return_annotation(ret_annotation);
        self.symbols.push_scope(ScopeKind::Func { return_ty });
        for param in params {
            let ty = self.resolve_annotation(param.type_annotation);
            self.symbols.bind_symbol(
                param.name.symbol,
                Binding::Param {
                    symbol: param.name.symbol,
                    ty,
                },
            );
        }

        // A function body is a group barrier: whatever aggregate context the
        // declaration sits inside, the body starts clean. An `agg fn` body is
        // then checked at group level — aggregate calls are legal, and its
        // parameters stand for per-row values, reachable only inside those
        // calls' arguments.
        let outer = std::mem::take(&mut self.agg);
        if is_agg {
            self.agg.grouping = Some(Grouping::FnBody { stmt: stmt_id });
        }
        if let Some(body) = body {
            self.infer_stmt(body);
        }
        if is_agg && !self.agg.saw_aggregate {
            self.report_stmt(
                stmt_id,
                "an `agg fn` must use an aggregate function".to_string(),
            );
        }
        self.agg = outer;
        self.symbols.pop_scope();
    }

    fn infer_let_stmt(
        &mut self,
        stmt_id: StmtId,
        name: SymbolId,
        mutability: Mutability,
        type_annotation: Option<TypeAnnotationId>,
        expr: ExprId,
    ) {
        if self.agg.grouping.is_some() && !self.agg.in_arguments {
            self.agg.group_locals.insert(name);
        }
        let expr_ty_id = self.infer_expr(expr);

        let ty_id = if let Some(annotation) = type_annotation {
            let target_ty_id = self.resolve_annotation(annotation);
            if !self.is_assignable(expr, expr_ty_id, target_ty_id) {
                let resolved_expr_ty = self.infer.resolve(expr_ty_id);
                let expected_expr_ty = self.infer.resolve(target_ty_id);
                let message = format!(
                    "value of type `{}` is not assignable to `{}`",
                    self.type_name(resolved_expr_ty),
                    self.type_name(expected_expr_ty),
                );
                self.report_stmt(stmt_id, message);
            }
            target_ty_id
        } else {
            expr_ty_id
        };

        self.symbols.bind_symbol(
            name,
            Binding::LetStmt {
                stmt: stmt_id,
                mutability,
                ty: ty_id,
            },
        );
    }

    fn infer_assign_stmt(&mut self, stmt_id: StmtId, target: ExprId, value: ExprId) {
        let target_ty = self.infer_expr(target);
        let value_ty = self.infer_expr(value);

        self.check_assignable(stmt_id, target);

        if !self.infer.unify(value_ty, target_ty) && !self.infer.coerce(value, value_ty, target_ty)
        {
            let resolved_value = self.infer.resolve(value_ty);
            let resolved_target = self.infer.resolve(target_ty);
            let message = format!(
                "value of type `{:?}` is not assignable to `{:?}`",
                self.infer.types.ty(resolved_value),
                self.infer.types.ty(resolved_target),
            );
            self.report_stmt(stmt_id, message);
        }
    }

    fn check_assignable(&mut self, stmt_id: StmtId, target: ExprId) {
        match self.hir.expr(target) {
            Expr::Ident { value } => {
                let name = value.symbol;
                match self.symbols.lookup_symbol(name).copied() {
                    Some(Binding::LetStmt {
                        mutability: Mutability::Mutable,
                        ..
                    })
                    | None => {}
                    Some(Binding::LetStmt {
                        mutability: Mutability::Immutable,
                        ..
                    }) => {
                        let message = format!(
                            "cannot assign to immutable variable `{}`",
                            self.interner.text(name)
                        );
                        self.report_stmt(stmt_id, message);
                    }
                    Some(_) => {
                        let message = format!("cannot assign to `{}`", self.interner.text(name));
                        self.report_stmt(stmt_id, message);
                    }
                }
            }
            Expr::FieldAccess { base, field } => {
                self.check_field_mutable(stmt_id, *base, field.symbol);
                self.check_place(stmt_id, *base);
            }
            _ => self.report_stmt(stmt_id, "cannot assign to this expression"),
        }
    }

    fn check_place(&mut self, stmt_id: StmtId, target: ExprId) {
        match self.hir.expr(target) {
            Expr::Ident { .. } => {}
            Expr::FieldAccess { base, .. } => self.check_place(stmt_id, *base),
            _ => self.report_stmt(stmt_id, "cannot assign to this expression"),
        }
    }

    fn check_field_mutable(&mut self, stmt_id: StmtId, base: ExprId, field: SymbolId) {
        let Some(base_ty) = self.infer.expr_ty(base) else {
            return;
        };
        let base_ty = self.infer.resolve(base_ty);
        let struct_name = match self.infer.types.ty(base_ty) {
            Type::Struct(s) => s.name,
            _ => return,
        };
        if self.field_is_immutable(struct_name, field) {
            let message = format!(
                "cannot assign to immutable field `{}`",
                self.interner.text(field)
            );
            self.report_stmt(stmt_id, message);
        }
    }

    fn field_is_immutable(&self, struct_name: SymbolId, field: SymbolId) -> bool {
        let Some(decl) = self.symbols.lookup_struct(struct_name) else {
            return false;
        };
        let Stmt::Struct { fields, .. } = self.hir.stmt(decl) else {
            return false;
        };
        fields
            .iter()
            .find(|f| f.name.symbol == field)
            .is_some_and(|f| f.mutability == Mutability::Immutable)
    }

    fn infer_return_stmt(&mut self, stmt_id: StmtId, expr: Option<ExprId>) {
        let value_ty = match expr {
            Some(expr) => self.infer_expr(expr),
            None => self.infer.types.unit_ty(),
        };

        let Some(return_ty) = self.symbols.return_ty() else {
            return;
        };

        // If value_ty can be unified to return_ty, we can return.
        if self.infer.unify(value_ty, return_ty) {
            return;
        }

        if let Some(expr) = expr
            && self.infer.coerce(expr, value_ty, return_ty)
        {
            return;
        }

        let resolved_value = self.infer.resolve(value_ty);
        let resolved_return = self.infer.resolve(return_ty);
        let message = format!(
            "value of type `{:?}` is not assignable to return type `{:?}`",
            self.infer.types.ty(resolved_value),
            self.infer.types.ty(resolved_return),
        );
        self.report_stmt(stmt_id, message);
    }

    fn infer_expr_stmt(&mut self, expr: ExprId) {
        self.infer_expr(expr);
    }

    fn infer_rel(&mut self, id: RelId) -> TypeId {
        let ty = match self.hir.rel(id).clone() {
            Rel::From { relation, alias } => self.infer_from_rel(id, relation.symbol, alias),
            Rel::Join {
                left,
                right,
                condition,
                ..
            } => self.infer_join_rel(id, left, right, &condition),
            Rel::Select { input, items } => self.infer_select_rel(input, &items),
            Rel::Where { input, predicate } => self.infer_where_rel(input, predicate),
            Rel::Distinct { input } => self.infer_rel(input),
            Rel::Drop { input, items } => self.infer_drop_rel(input, &items),
            Rel::Rename { input, items } => self.infer_rename_rel(id, input, &items),
            Rel::Extend { input, items } => self.infer_extend_rel(input, &items),
            Rel::Set { input, items } => self.infer_set_rel(id, input, &items),
            Rel::Limit {
                input,
                count,
                offset,
            } => self.infer_limit_rel(input, count, offset),
            Rel::Alias { input, alias } => self.infer_alias_rel(input, alias),
            Rel::Aggregate {
                input,
                items,
                groups,
            } => self.infer_aggregate_rel(id, input, &items, &groups),
            Rel::Missing => self.infer.types.error_ty(),
        };
        self.infer.bind_rel_ty(id, ty)
    }

    fn infer_from_rel(&mut self, id: RelId, symbol: SymbolId, alias: Option<Ident>) -> TypeId {
        let Some(relation_ty) = self.symbols.lookup_relation(symbol) else {
            let message = format!("`{}` is not a relation", self.interner.text(symbol));
            return self.error_rel(id, message);
        };

        // An alias replaces the relation's own name: `from employees e` is
        // addressed as `e`, never as `employees`.
        let alias = alias.unwrap_or(Ident { symbol }).symbol;
        let Some(columns) = self.columns(relation_ty) else {
            return self.infer.types.error_ty();
        };

        let columns = columns
            .iter()
            .map(|column| Column::new(Some(alias), column.name, column.ty))
            .collect();

        let source_ty = self.infer.types.relation_ty(columns);
        // The binding is what marks `alias.column` as a column reference; which
        // column it is comes from the row it resolves against.
        self.symbols
            .bind_symbol(alias, Binding::Relation { ty: source_ty });

        self.symbols.replace_current_row(source_ty)
    }

    fn infer_join_rel(
        &mut self,
        id: RelId,
        left: RelId,
        right: RelId,
        condition: &JoinCondition,
    ) -> TypeId {
        let left_ty = self.infer_rel(left);
        let right_ty = self.infer_rel(right);

        let Some(left_columns) = self.columns(left_ty).map(|f| f.to_vec()) else {
            return self.infer.types.error_ty();
        };

        let Some(right_columns) = self.columns(right_ty).map(|f| f.to_vec()) else {
            return self.infer.types.error_ty();
        };

        let joined_columns: Vec<Column> = match condition {
            JoinCondition::On(_) => left_columns
                .iter()
                .cloned()
                .chain(right_columns.iter().cloned())
                .collect(),

            JoinCondition::Using(idents) => {
                let mut error_reported = false;
                for ident in idents {
                    let left_col = left_columns.iter().find(|col| col.name == ident.symbol);
                    let right_col = right_columns.iter().find(|col| col.name == ident.symbol);

                    let (Some(left_col), Some(right_col)) = (left_col, right_col) else {
                        let message = format!(
                            "column {} not present in both relations",
                            self.interner.text(ident.symbol)
                        );
                        self.report_rel(id, message);
                        error_reported = true;
                        continue;
                    };

                    // The two are compared, so a plan needs one comparison type.
                    if self.infer.resolve(left_col.ty) != self.infer.resolve(right_col.ty) {
                        let message = format!(
                            "column {} is `{:?}` on the left and `{:?}` on the right",
                            self.interner.text(ident.symbol),
                            self.infer.types.ty(left_col.ty),
                            self.infer.types.ty(right_col.ty),
                        );
                        self.report_rel(id, message);
                        error_reported = true;
                    }
                }

                if error_reported {
                    return self.infer.types.error_ty();
                }

                // `using` only constrains the two rows; both sides' columns
                // carry through, exactly as they do for `on`.
                left_columns
                    .iter()
                    .cloned()
                    .chain(right_columns.iter().cloned())
                    .collect()
            }
        };

        let joined_ty = self
            .symbols
            .replace_current_row(self.infer.types.relation_ty(joined_columns));

        if let JoinCondition::On(expr) = condition {
            let condition_ty = self.infer_expr(*expr);
            self.ensure_boolean(*expr, condition_ty, "`on` condition");
        }

        joined_ty
    }

    fn infer_select_rel(&mut self, input: RelId, items: &[SelectItem]) -> TypeId {
        let input_ty = self.infer_rel(input);
        if self.columns(input_ty).is_none() {
            return self.infer.types.error_ty();
        }

        let mut columns = Vec::with_capacity(items.len());
        let mut anonymous = 0;
        for item in items {
            let ty = self.infer_expr(item.expr);
            let name = self.resolve_column_name(item.expr, item.alias, &mut anonymous);
            columns.push(Column::new(None, name, ty));
        }

        let selected_ty = self.infer.types.relation_ty(columns);

        self.symbols.replace_current_row(selected_ty)
    }

    fn infer_aggregate_rel(
        &mut self,
        id: RelId,
        input: RelId,
        items: &[crate::AggregateItem],
        groups: &[crate::GroupKey],
    ) -> TypeId {
        let input_ty = self.infer_rel(input);
        let Some(input_columns) = self.columns(input_ty) else {
            return self.infer.types.error_ty();
        };
        let input_columns = input_columns.to_vec();

        let mut key_columns = Vec::with_capacity(groups.len());
        let mut key_positions = Vec::with_capacity(groups.len());
        let mut poisoned = false;
        for key in groups {
            let mut matches = input_columns.iter().enumerate().filter(|(_, column)| {
                column.name == key.column.symbol
                    && key
                        .qualifier
                        .as_ref()
                        .is_none_or(|qualifier| column.named_by(qualifier.symbol))
            });
            match (matches.next(), matches.next()) {
                (Some((position, column)), None) => {
                    let name = key.alias.as_ref().unwrap_or(&key.column).symbol;
                    key_columns.push(Column::new(None, name, column.ty));
                    key_positions.push(position as u32);
                }
                (Some(_), Some(_)) => {
                    let message = format!(
                        "group key `{}` is ambiguous; qualify it with a relation alias",
                        self.interner.text(key.column.symbol)
                    );
                    self.error_rel(id, message);
                    poisoned = true;
                }
                _ => {
                    let message = format!(
                        "group key `{}` is not a column of this row",
                        self.interner.text(key.column.symbol)
                    );
                    self.error_rel(id, message);
                    poisoned = true;
                }
            }
        }

        self.infer.bind_group_keys(id, &key_positions);
        self.agg.grouping = Some(Grouping::Stage {
            keys: key_columns
                .iter()
                .zip(&key_positions)
                .map(|(column, &position)| ScopeKey {
                    name: column.name,
                    column: position,
                })
                .collect(),
        });
        let mut columns = key_columns;
        let mut anonymous = 0;
        for item in items {
            let ty = self.infer_expr(item.expr);
            let name = self.resolve_column_name(item.expr, item.alias, &mut anonymous);
            columns.push(Column::new(None, name, ty));
        }
        self.agg.grouping = None;
        self.agg.group_locals.clear();

        if poisoned {
            return self.infer.types.error_ty();
        }

        let aggregated_ty = self.infer.types.relation_ty(columns);
        self.symbols.replace_current_row(aggregated_ty)
    }

    fn infer_where_rel(&mut self, input: RelId, predicate: ExprId) -> TypeId {
        let input_ty = self.infer_rel(input);
        if self.columns(input_ty).is_none() {
            self.infer_expr(predicate);
            return self.infer.types.error_ty();
        }

        let predicate_ty = self.infer_expr(predicate);
        self.ensure_boolean(predicate, predicate_ty, "`where` predicate");

        input_ty
    }

    fn ensure_boolean(&mut self, expr: ExprId, ty: TypeId, subject: &str) {
        let ty = self.infer.resolve(ty);
        let bool_ty = self.infer.types.bool_ty();
        let error_ty = self.infer.types.error_ty();
        if ty == bool_ty || ty == error_ty {
            return;
        }

        let message = format!(
            "expected `{subject}` to be a `bool` type, but found `{:?}`",
            self.infer.types.ty(ty)
        );
        self.report_expr(expr, message);
    }

    fn infer_drop_rel(&mut self, input: RelId, columns: &[Ident]) -> TypeId {
        let input_ty = self.infer_rel(input);
        let Some(input_columns) = self.columns(input_ty) else {
            return self.infer.types.error_ty();
        };

        let dropped: HashSet<SymbolId> = columns.iter().map(|column| column.symbol).collect();

        let remaining = input_columns
            .iter()
            .filter(|column| !dropped.contains(&column.name))
            .cloned()
            .collect();

        let dropped_ty = self.infer.types.relation_ty(remaining);
        self.symbols.replace_current_row(dropped_ty)
    }

    fn infer_rename_rel(&mut self, id: RelId, input: RelId, items: &[RenameItem]) -> TypeId {
        let input_ty = self.infer_rel(input);
        let Some(columns) = self.columns(input_ty) else {
            return self.infer.types.error_ty();
        };
        let mut renamed = columns.to_vec();

        for item in items {
            let Some(column) = self.rename_target(id, item, &renamed) else {
                return self.infer.types.error_ty();
            };
            renamed[column as usize].name = item.to.symbol;
        }

        let renamed_ty = self.infer.types.relation_ty(renamed);
        self.symbols.replace_current_row(renamed_ty)
    }

    /// The one column a rename item names: a qualified item goes through its
    /// alias, a bare one has to match exactly one column of the row.
    fn rename_target(&mut self, id: RelId, item: &RenameItem, columns: &[Column]) -> Option<u32> {
        let name = item.from.symbol;
        let mut matches = columns.iter().enumerate().filter(|(_, column)| {
            column.name == name
                && item
                    .qualifier
                    .is_none_or(|alias| column.named_by(alias.symbol))
        });

        match (matches.next(), matches.next()) {
            (Some((column, _)), None) => Some(column as u32),
            (Some(_), Some(_)) => {
                let message = format!(
                    "column `{}` is ambiguous; qualify it with a relation alias",
                    self.interner.text(name)
                );
                self.report_rel(id, message);
                None
            }
            _ => {
                let message = match item.qualifier {
                    Some(alias) => format!(
                        "`{}` has no column `{}`",
                        self.interner.text(alias.symbol),
                        self.interner.text(name)
                    ),
                    None => format!("column `{}` is not in this row", self.interner.text(name)),
                };
                self.report_rel(id, message);
                None
            }
        }
    }

    /// `set` replaces a column's value where it stands, so the row keeps its
    /// shape and each name must match exactly one column.
    fn infer_set_rel(&mut self, id: RelId, input: RelId, items: &[SetItem]) -> TypeId {
        let input_ty = self.infer_rel(input);
        let Some(columns) = self.columns(input_ty) else {
            return self.infer.types.error_ty();
        };
        let mut columns = columns.to_vec();

        for item in items {
            let ty = self.infer_expr(item.value);
            let name = item.column.symbol;

            let mut matches = columns
                .iter()
                .enumerate()
                .filter(|(_, column)| column.name == name);
            let target = match (matches.next(), matches.next()) {
                (Some((column, _)), None) => column,
                (Some(_), Some(_)) => {
                    let message = format!(
                        "column `{}` is ambiguous; qualify it with a relation alias",
                        self.interner.text(name)
                    );
                    return self.error_rel(id, message);
                }
                _ => {
                    let message =
                        format!("column `{}` is not in this row", self.interner.text(name));
                    return self.error_rel(id, message);
                }
            };
            columns[target].ty = ty;
        }

        let set_ty = self.infer.types.relation_ty(columns);
        self.symbols.replace_current_row(set_ty)
    }

    /// `limit` keeps the row it is given; only the counts are typed.
    fn infer_limit_rel(&mut self, input: RelId, count: ExprId, offset: Option<ExprId>) -> TypeId {
        let input_ty = self.infer_rel(input);
        for bound in [Some(count), offset].into_iter().flatten() {
            let ty = self.infer_expr(bound);
            self.ensure_integer(bound, ty);
        }
        input_ty
    }

    /// `|> as t` renames the whole row: every column is named through `t` from
    /// here on, and the aliases that reached it stop resolving.
    fn infer_alias_rel(&mut self, input: RelId, alias: Ident) -> TypeId {
        let input_ty = self.infer_rel(input);
        let Some(columns) = self.columns(input_ty) else {
            return self.infer.types.error_ty();
        };

        let columns = columns
            .iter()
            .map(|column| Column::new(Some(alias.symbol), column.name, column.ty))
            .collect();

        let aliased_ty = self.infer.types.relation_ty(columns);
        self.symbols
            .bind_symbol(alias.symbol, Binding::Relation { ty: aliased_ty });
        self.symbols.replace_current_row(aliased_ty)
    }

    fn ensure_integer(&mut self, expr: ExprId, ty: TypeId) {
        let ty = self.infer.resolve(ty);
        if self.infer.types.ty(ty).is_integer() || ty == self.infer.types.error_ty() {
            return;
        }
        let message = format!(
            "`limit` needs an integer, but found `{}`",
            self.type_name(ty)
        );
        self.report_expr(expr, message);
    }

    fn infer_extend_rel(&mut self, input: RelId, items: &[SelectItem]) -> TypeId {
        let input_ty = self.infer_rel(input);
        let Some(columns) = self.columns(input_ty) else {
            return self.infer.types.error_ty();
        };

        let mut extended_columns = columns.to_vec();
        let mut anonymous = 0;
        for item in items {
            let ty = self.infer_expr(item.expr);
            let name = self.resolve_column_name(item.expr, item.alias, &mut anonymous);
            extended_columns.push(Column::new(None, name, ty));
        }

        let extended_ty = self.infer.types.relation_ty(extended_columns);
        self.symbols.replace_current_row(extended_ty)
    }

    /// The output column name: the `as` alias, else a bare identifier's or field
    /// access's own name, else a generated `%gN` for an anonymous expression.
    fn resolve_column_name(
        &mut self,
        expr: ExprId,
        alias: Option<Ident>,
        anonymous: &mut usize,
    ) -> SymbolId {
        if let Some(alias) = alias {
            return alias.symbol;
        }
        match self.hir.expr(expr) {
            Expr::Ident { value } => value.symbol,
            Expr::FieldAccess { field, .. } => field.symbol,
            _ => {
                let name = format!("%g{anonymous}");
                *anonymous += 1;
                self.interner.intern(&name)
            }
        }
    }

    fn columns(&self, relation_ty: TypeId) -> Option<&[Column]> {
        match self.infer.types.ty(relation_ty) {
            Type::Relation(relation) => Some(&relation.columns),
            _ => None,
        }
    }

    fn current_columns(&self) -> Option<&[Column]> {
        self.columns(self.symbols.row()?)
    }

    fn resolve_column(&self, name: SymbolId) -> ColumnLookup {
        let Some(columns) = self.current_columns() else {
            return ColumnLookup::Absent;
        };

        let mut matches = columns
            .iter()
            .enumerate()
            .filter(|(_, column)| column.name == name);

        match (matches.next(), matches.next()) {
            (Some((column, found)), None) => ColumnLookup::Unique {
                ty: found.ty,
                column: column as u32,
            },
            (Some(_), Some(_)) => ColumnLookup::Ambiguous,
            _ => ColumnLookup::Absent,
        }
    }

    fn error_rel(&mut self, id: RelId, message: impl Into<String>) -> TypeId {
        self.report_rel(id, message);
        self.infer.types.error_ty()
    }

    fn report_rel(&mut self, id: RelId, message: impl Into<String>) {
        let span = self
            .source_map
            .rel(id)
            .expect("a reported node is always in the source map");
        self.diagnostics
            .emit(DiagnosticBuilder::error(span, message));
    }

    fn infer_expr(&mut self, expr_id: ExprId) -> TypeId {
        match self.hir.expr(expr_id) {
            Expr::Ident { value } => self.infer_ident_expr(expr_id, value),
            Expr::Call { op, args } => self.infer_call_expr(expr_id, *op, args),
            Expr::FuncCall { callee, args } => self.infer_func_call_expr(expr_id, *callee, args),
            Expr::MethodCall {
                receiver,
                method,
                args,
            } => self.infer_method_call_expr(expr_id, *receiver, *method, args),
            Expr::FieldAccess { base, field } => {
                self.infer_field_access_expr(expr_id, *base, field.symbol)
            }
            Expr::StructInit { name, fields } => {
                self.infer_struct_init_expr(expr_id, *name, fields)
            }
            Expr::ListInit { elements } => self.infer_list_init_expr(expr_id, elements),
            Expr::Literal(literal) => self.infer_literal_expr(expr_id, *literal),
            Expr::Rel(rel) => {
                self.symbols.push_scope(ScopeKind::Block);
                let ty = self.infer_rel(*rel);
                self.symbols.pop_scope();
                self.infer.bind_expr_ty(expr_id, ty)
            }
            Expr::Missing => self.poison(expr_id),
        }
    }

    fn infer_ident_expr(&mut self, expr_id: ExprId, value: &Ident) -> TypeId {
        let name = value.symbol;
        if let Some(&binding) = self.symbols.lookup_symbol(name) {
            if self.agg.in_arguments && self.agg.group_locals.contains(&name) {
                let message = format!(
                    "`{}` is a group-level value and cannot be used inside an aggregate's arguments",
                    self.interner.text(name)
                );
                return self.error_expr(expr_id, message);
            }
            match binding {
                Binding::Builtin { func } => {
                    let message = format!("`{}` is a function, not a value", func.name());
                    return self.error_expr(expr_id, message);
                }
                Binding::FuncStmt { is_agg: true, .. } => {
                    let message = format!(
                        "`{}` is an aggregate function, not a value",
                        self.interner.text(name)
                    );
                    return self.error_expr(expr_id, message);
                }
                Binding::Param { .. }
                    if matches!(self.agg.grouping, Some(Grouping::FnBody { .. }))
                        && !self.agg.in_arguments =>
                {
                    let message = format!(
                        "parameter `{}` can only be used inside an aggregate function's arguments",
                        self.interner.text(name)
                    );
                    return self.error_expr(expr_id, message);
                }
                _ => {}
            }
            return self.infer.bind_expr_ty(expr_id, binding.ty());
        }

        match self.resolve_column(name) {
            ColumnLookup::Unique { ty, column } => {
                self.check_group_position(expr_id, name, column);
                self.infer.bind_column(expr_id, column);
                return self.infer.bind_expr_ty(expr_id, ty);
            }
            ColumnLookup::Ambiguous => {
                let message = format!(
                    "column `{}` is ambiguous; qualify it with a relation alias",
                    self.interner.text(name)
                );
                return self.error_expr(expr_id, message);
            }
            ColumnLookup::Absent => {}
        }
        if self.symbols.lookup_relation(name).is_some() {
            let message = format!("`{}` is a relation, not a value", self.interner.text(name));
            return self.error_expr(expr_id, message);
        }

        let message = format!("unresolved identifier `{}`", self.interner.text(name));
        self.error_expr(expr_id, message)
    }

    fn infer_column_expr(&mut self, expr_id: ExprId, alias: Ident, name: SymbolId) -> TypeId {
        let Some(columns) = self.current_columns() else {
            let message = format!("`{}` is not available here", self.interner.text(name));
            return self.error_expr(expr_id, message);
        };

        let found = columns
            .iter()
            .enumerate()
            .find(|(_, column)| column.name == name && column.named_by(alias.symbol));

        let Some((column, found)) = found else {
            let message = format!(
                "`{}` has no column `{}` here",
                self.interner.text(alias.symbol),
                self.interner.text(name)
            );
            return self.error_expr(expr_id, message);
        };

        let ty = found.ty;
        self.check_group_position(expr_id, name, column as u32);
        self.infer.bind_column(expr_id, column as u32);
        self.infer.bind_expr_ty(expr_id, ty)
    }

    /// Inside an `aggregate` item at group level, the input row is gone: a
    /// column is only reachable as a group key or through an aggregate call.
    fn check_group_position(&mut self, expr_id: ExprId, name: SymbolId, column: u32) {
        let Some(Grouping::Stage { keys }) = &self.agg.grouping else {
            return;
        };
        if self.agg.in_arguments || keys.iter().any(|key| key.column == column) {
            return;
        }
        let keys = keys
            .iter()
            .map(|key| format!("`{}`", self.interner.text(key.name)))
            .collect::<Vec<_>>()
            .join(", ");
        let message = if keys.is_empty() {
            format!(
                "column `{}` must be inside an aggregate function; nothing is grouped",
                self.interner.text(name)
            )
        } else {
            format!(
                "column `{}` must be one of the group keys ({keys}) or inside an aggregate function",
                self.interner.text(name)
            )
        };
        self.report_expr(expr_id, message);
    }

    fn infer_call_expr(&mut self, expr_id: ExprId, op: Op, args: &[ExprId]) -> TypeId {
        for &arg in args {
            self.infer_expr(arg);
        }

        let error_ty = self.infer.types.error_ty();
        let arg_tys: Vec<TypeId> = args
            .iter()
            .map(|&arg| self.infer.expr_ty(arg).unwrap_or(error_ty))
            .collect();

        if arg_tys.contains(&error_ty) {
            return self.poison(expr_id);
        }

        let ty = op.resolve(&arg_tys, &mut self.infer);
        if ty == error_ty {
            let operands: Vec<String> = arg_tys
                .iter()
                .map(|&arg| format!("`{}`", self.type_name(arg)))
                .collect();
            self.report_expr(
                expr_id,
                format!(
                    "operator `{}` cannot be applied to {}",
                    op.symbol(),
                    operands.join(" and ")
                ),
            );
        }
        self.infer.bind_expr_ty(expr_id, ty)
    }

    /// A type rendered for a diagnostic: scalars by name, compounds
    /// structurally (`List[Int64]`, a struct by its declared name).
    fn type_name(&mut self, ty: TypeId) -> String {
        let resolved = self.infer.resolve(ty);
        match self.infer.types.ty(resolved).clone() {
            Type::List(list) => format!("List[{}]", self.type_name(list.inner)),
            Type::Relation(relation) => {
                let columns: Vec<String> = relation
                    .columns
                    .iter()
                    .map(|column| self.interner.text(column.name).to_string())
                    .collect();
                format!("Relation[{}]", columns.join(", "))
            }
            Type::Struct(row) => self.interner.text(row.name).to_string(),
            other => format!("{other:?}"),
        }
    }

    fn infer_func_call_expr(&mut self, expr_id: ExprId, callee: ExprId, args: &[ExprId]) -> TypeId {
        if let Some(func) = self.builtin_callee(callee) {
            return self.infer_builtin_call_expr(expr_id, func, args);
        }
        if let Some((name, stmt, ty, external)) = self.agg_fn_callee(callee) {
            return self.infer_agg_fn_call_expr(expr_id, name, stmt, ty, args, external);
        }
        if let Some((name, ty)) = self.extern_callee(callee) {
            let ty = self.check_call_signature(expr_id, ty, args);
            self.infer.bind_extern_call(expr_id, name);
            return ty;
        }

        let callee_ty = self.infer_expr(callee);
        let callee_ty = self.infer.resolve(callee_ty);
        let error_ty = self.infer.types.error_ty();

        let Type::Func(func) = self.infer.types.ty(callee_ty).clone() else {
            for &arg in args {
                self.infer_expr(arg);
            }
            if callee_ty == error_ty {
                return self.infer.bind_expr_ty(expr_id, error_ty);
            }
            let message = format!(
                "type `{:?}` is not callable",
                self.infer.types.ty(callee_ty)
            );
            return self.error_expr(expr_id, message);
        };

        for (index, &arg) in args.iter().enumerate() {
            let arg_ty = self.infer_expr(arg);
            let Some(&param_ty) = func.args.get(index) else {
                continue;
            };

            if !self.is_assignable(arg, arg_ty, param_ty) {
                let resolved_arg = self.infer.resolve(arg_ty);
                let resolved_param = self.infer.resolve(param_ty);
                let message = format!(
                    "argument of type `{:?}` is not assignable to parameter of type `{:?}`",
                    self.infer.types.ty(resolved_arg),
                    self.infer.types.ty(resolved_param),
                );
                self.report_expr(arg, message);
            }
        }

        if args.len() != func.args.len() {
            let message = format!(
                "expected {} argument(s), found {}",
                func.args.len(),
                args.len()
            );
            self.report_expr(expr_id, message);
        }

        self.infer.bind_expr_ty(expr_id, func.ret_type)
    }

    /// The builtin a call's callee names, when nothing in scope shadows it.
    fn builtin_callee(&mut self, callee: ExprId) -> Option<BuiltinFunc> {
        let Expr::Ident { value } = self.hir.expr(callee) else {
            return None;
        };
        match self.symbols.lookup_symbol(value.symbol) {
            Some(Binding::Builtin { func }) => Some(*func),
            _ => None,
        }
    }

    fn infer_method_call_expr(
        &mut self,
        expr_id: ExprId,
        receiver: ExprId,
        method: Ident,
        args: &[ExprId],
    ) -> TypeId {
        let _ = (receiver, method);
        for &arg in args {
            self.infer_expr(arg);
        }
        self.error_expr(expr_id, "method calls are not supported yet")
    }

    /// Argument and arity checks against a resolved function type.
    fn check_call_signature(
        &mut self,
        expr_id: ExprId,
        callee_ty: TypeId,
        args: &[ExprId],
    ) -> TypeId {
        let error_ty = self.infer.types.error_ty();
        let resolved = self.infer.resolve(callee_ty);
        let Type::Func(func) = self.infer.types.ty(resolved).clone() else {
            for &arg in args {
                self.infer_expr(arg);
            }
            if resolved == error_ty {
                return self.infer.bind_expr_ty(expr_id, error_ty);
            }
            let message = format!("type `{:?}` is not callable", self.infer.types.ty(resolved));
            return self.error_expr(expr_id, message);
        };

        for (index, &arg) in args.iter().enumerate() {
            let arg_ty = self.infer_expr(arg);
            let Some(&param_ty) = func.args.get(index) else {
                continue;
            };
            if !self.is_assignable(arg, arg_ty, param_ty) {
                let resolved_arg = self.infer.resolve(arg_ty);
                let resolved_param = self.infer.resolve(param_ty);
                let message = format!(
                    "argument of type `{:?}` is not assignable to parameter of type `{:?}`",
                    self.infer.types.ty(resolved_arg),
                    self.infer.types.ty(resolved_param),
                );
                self.report_expr(arg, message);
            }
        }

        if args.len() != func.args.len() {
            let message = format!(
                "expected {} argument(s), found {}",
                func.args.len(),
                args.len()
            );
            self.report_expr(expr_id, message);
        }

        self.infer.bind_expr_ty(expr_id, func.ret_type)
    }

    fn agg_fn_callee(&mut self, callee: ExprId) -> Option<(SymbolId, StmtId, TypeId, bool)> {
        let Expr::Ident { value } = self.hir.expr(callee) else {
            return None;
        };
        match self.symbols.lookup_symbol(value.symbol) {
            Some(&Binding::FuncStmt {
                stmt,
                ty,
                is_agg: true,
                is_external,
            }) => Some((value.symbol, stmt, ty, is_external)),
            _ => None,
        }
    }

    fn extern_callee(&mut self, callee: ExprId) -> Option<(SymbolId, TypeId)> {
        let Expr::Ident { value } = self.hir.expr(callee) else {
            return None;
        };
        match self.symbols.lookup_symbol(value.symbol) {
            Some(&Binding::FuncStmt {
                ty,
                is_agg: false,
                is_external: true,
                ..
            }) => Some((value.symbol, ty)),
            _ => None,
        }
    }

    /// An `agg fn` call sits exactly where a builtin aggregate call does: its
    /// arguments are per-row values, its result is a group-level value, and it
    /// cannot nest. The body already type-checked at its declaration, so the
    /// call checks like any function call against the signature.
    fn infer_agg_fn_call_expr(
        &mut self,
        expr_id: ExprId,
        name: SymbolId,
        stmt: StmtId,
        callee_ty: TypeId,
        args: &[ExprId],
        external: bool,
    ) -> TypeId {
        let symbol = name;
        let name = self.interner.text(name).to_string();
        if self.agg.grouping.is_none() {
            let message =
                format!("aggregate function `{name}` can only be used in an `aggregate` item");
            return self.error_expr(expr_id, message);
        }
        if self.agg.in_arguments {
            let message =
                format!("aggregate function `{name}` cannot be nested in another aggregate");
            return self.error_expr(expr_id, message);
        }
        if matches!(self.agg.grouping, Some(Grouping::FnBody { stmt: body }) if body == stmt) {
            let message = format!("`{name}` is an `agg fn` and cannot call itself");
            return self.error_expr(expr_id, message);
        }

        let Type::Func(func) = self.infer.types.ty(callee_ty).clone() else {
            unreachable!("a registered function has a function type")
        };

        self.agg.saw_aggregate = true;
        self.agg.in_arguments = true;
        for (index, &arg) in args.iter().enumerate() {
            let arg_ty = self.infer_expr(arg);
            let Some(&param_ty) = func.args.get(index) else {
                continue;
            };
            if !self.is_assignable(arg, arg_ty, param_ty) {
                let resolved_arg = self.infer.resolve(arg_ty);
                let resolved_param = self.infer.resolve(param_ty);
                let message = format!(
                    "argument of type `{:?}` is not assignable to parameter of type `{:?}`",
                    self.infer.types.ty(resolved_arg),
                    self.infer.types.ty(resolved_param),
                );
                self.report_expr(arg, message);
            }
        }
        self.agg.in_arguments = false;

        if args.len() != func.args.len() {
            let message = format!(
                "expected {} argument(s), found {}",
                func.args.len(),
                args.len()
            );
            self.report_expr(expr_id, message);
        }

        if external {
            self.infer
                .bind_builtin_call(expr_id, BuiltinFunc::Aggregate(AggFunc::External(symbol)));
        }
        self.infer.bind_expr_ty(expr_id, func.ret_type)
    }

    fn infer_builtin_call_expr(
        &mut self,
        expr_id: ExprId,
        func: BuiltinFunc,
        args: &[ExprId],
    ) -> TypeId {
        let entry = self
            .registry
            .resolve(func)
            .unwrap_or_else(|| unreachable!("a bound builtin came from the registry"));

        match func {
            BuiltinFunc::Aggregate(agg) => {
                if self.agg.grouping.is_none() {
                    let message = format!(
                        "aggregate function `{}` can only be used in an `aggregate` item",
                        func.name()
                    );
                    return self.error_expr(expr_id, message);
                }
                if self.agg.in_arguments {
                    let message = format!(
                        "aggregate function `{}` cannot be nested in another aggregate",
                        func.name()
                    );
                    return self.error_expr(expr_id, message);
                }

                self.agg.saw_aggregate = true;
                self.agg.in_arguments = true;
                let arg_tys: Vec<TypeId> = args.iter().map(|&arg| self.infer_expr(arg)).collect();
                self.agg.in_arguments = false;

                if args.len() < entry.min_args || args.len() > entry.max_args {
                    let expected = if entry.min_args == entry.max_args {
                        entry.min_args.to_string()
                    } else {
                        format!("{} to {}", entry.min_args, entry.max_args)
                    };
                    let message = format!(
                        "`{}` expects {expected} argument(s), found {}",
                        func.name(),
                        args.len()
                    );
                    return self.error_expr(expr_id, message);
                }

                let ty = self.resolve_agg_ty(expr_id, agg, &arg_tys);
                self.infer.bind_builtin_call(expr_id, func);
                self.infer.bind_expr_ty(expr_id, ty)
            }
            BuiltinFunc::Scalar(scalar) => {
                let arg_tys: Vec<TypeId> = args.iter().map(|&arg| self.infer_expr(arg)).collect();
                if args.len() < entry.min_args || args.len() > entry.max_args {
                    let expected = if entry.min_args == entry.max_args {
                        entry.min_args.to_string()
                    } else {
                        format!("{} to {}", entry.min_args, entry.max_args)
                    };
                    let message = format!(
                        "`{}` expects {expected} argument(s), found {}",
                        func.name(),
                        args.len()
                    );
                    return self.error_expr(expr_id, message);
                }

                let op = match scalar {
                    Func::Power => Op::Pow,
                    Func::ShiftLeft => Op::ShiftLeft,
                    Func::ShiftRight => Op::ShiftRight,
                    Func::In => Op::In,
                    other => {
                        let message = format!("`{}` is not callable by name", other.symbol());
                        return self.error_expr(expr_id, message);
                    }
                };

                let ty = op.resolve(&arg_tys, &mut self.infer);
                self.infer.bind_builtin_call(expr_id, func);
                self.infer.bind_expr_ty(expr_id, ty)
            }
        }
    }

    fn resolve_agg_ty(&mut self, expr_id: ExprId, func: AggFunc, args: &[TypeId]) -> TypeId {
        let error_ty = self.infer.types.error_ty();
        match func {
            AggFunc::Count | AggFunc::CountDistinct => self.infer.types.int64_ty(),
            AggFunc::Sum => {
                let arg = self.infer.resolve(args[0]);
                if arg == error_ty {
                    return error_ty;
                }
                if self.infer.types.ty(arg).is_int() {
                    self.infer.types.int64_ty()
                } else if self.infer.types.ty(arg).is_float() {
                    self.infer.types.float64_ty()
                } else {
                    self.numeric_argument_error(expr_id, func, arg)
                }
            }
            AggFunc::External(_) => {
                unreachable!("an external aggregate checks against its declared signature")
            }
            AggFunc::Min | AggFunc::Max | AggFunc::Avg => {
                let arg = self.infer.resolve(args[0]);
                if arg == error_ty {
                    return error_ty;
                }
                if self.infer.types.ty(arg).is_numeric() {
                    arg
                } else {
                    self.numeric_argument_error(expr_id, func, arg)
                }
            }
        }
    }

    fn numeric_argument_error(&mut self, expr_id: ExprId, func: AggFunc, arg: TypeId) -> TypeId {
        let message = format!(
            "`{}` needs a numeric argument, but found `{:?}`",
            func.name(),
            self.infer.types.ty(arg)
        );
        self.error_expr(expr_id, message)
    }

    fn is_assignable(&mut self, expr: ExprId, value_ty: TypeId, target_ty: TypeId) -> bool {
        self.relations_match(value_ty, target_ty)
            || self.infer.unify(value_ty, target_ty)
            || self.infer.coerce(expr, value_ty, target_ty)
    }

    /// Two relations match on their columns' names and types. A qualifier says
    /// how a column is named inside a query, not what shape the relation has,
    /// so `Relation[Row]` accepts a query however its source was aliased.
    fn relations_match(&self, value_ty: TypeId, target_ty: TypeId) -> bool {
        let (Some(value), Some(target)) = (self.columns(value_ty), self.columns(target_ty)) else {
            return false;
        };
        value.len() == target.len()
            && value
                .iter()
                .zip(target)
                .all(|(value, target)| value.name == target.name && value.ty == target.ty)
    }

    fn infer_field_access_expr(
        &mut self,
        expr_id: ExprId,
        base: ExprId,
        field: SymbolId,
    ) -> TypeId {
        // If the field is of the form <ident>.<expr>, where <ident> is an relation alias, then
        // infer using a column expression instead.
        if let Some(alias) = self.lookup_relation_alias(base) {
            return self.infer_column_expr(expr_id, alias, field);
        }

        // A relation names no value, so it cannot qualify anything until a
        // `from` or `join` gives it an alias.
        if let Expr::Ident { value } = self.hir.expr(base)
            && self.symbols.lookup_relation(value.symbol).is_some()
        {
            let message = format!(
                "`{}` is aliased here; qualify its columns with the alias",
                self.interner.text(value.symbol)
            );
            return self.error_expr(expr_id, message);
        }

        let base_ty = self.infer_expr(base);
        let base_ty = self.infer.resolve(base_ty);
        let error_ty = self.infer.types.error_ty();

        // A poisoned base was already reported; don't cascade.
        if base_ty == error_ty {
            return self.infer.bind_expr_ty(expr_id, error_ty);
        }

        let (struct_name, field_ty) = match self.infer.types.ty(base_ty) {
            Type::Struct(s) => (
                Some(s.name),
                s.fields.iter().find(|(n, _)| *n == field).map(|(_, t)| *t),
            ),
            _ => (None, None),
        };

        if let Some(ty) = field_ty {
            return self.infer.bind_expr_ty(expr_id, ty);
        }

        let message = match struct_name {
            Some(struct_name) => format!(
                "struct `{}` has no field `{}`",
                self.interner.text(struct_name),
                self.interner.text(field),
            ),
            None => format!("type `{:?}` has no fields", self.infer.types.ty(base_ty)),
        };
        self.error_expr(expr_id, message)
    }

    fn lookup_relation_alias(&self, base: ExprId) -> Option<Ident> {
        let Expr::Ident { value } = self.hir.expr(base) else {
            return None;
        };
        match self.symbols.lookup_symbol(value.symbol) {
            Some(Binding::Relation { .. }) => Some(*value),
            _ => None,
        }
    }

    fn infer_struct_init_expr(
        &mut self,
        expr_id: ExprId,
        name: SymbolId,
        fields: &[StructFieldInit],
    ) -> TypeId {
        // The name must resolve to a declared struct.
        let struct_ty = self.symbols.lookup_type(name);
        let declared: Option<Vec<(SymbolId, TypeId)>> = struct_ty.and_then(|ty| {
            if let Type::Struct(s) = self.infer.types.ty(ty) {
                Some(s.fields.clone())
            } else {
                None
            }
        });

        let (Some(struct_ty), Some(declared)) = (struct_ty, declared) else {
            for field in fields {
                self.infer_expr(field.value);
            }
            let message = format!("`{}` is not a struct", self.interner.text(name));
            return self.error_expr(expr_id, message);
        };

        // Each initializer must name a declared field exactly once with an
        // assignable value; every declared field must be initialized.
        let mut seen: HashSet<SymbolId> = HashSet::new();
        for field in fields {
            let field_name = field.name.symbol;
            let value_ty = self.infer_expr(field.value);

            let Some(&(_, declared_ty)) = declared.iter().find(|(n, _)| *n == field_name) else {
                let message = format!(
                    "struct `{}` has no field `{}`",
                    self.interner.text(name),
                    self.interner.text(field_name),
                );
                self.error_expr(expr_id, message);
                continue;
            };

            if seen.contains(&field_name) {
                let message = format!(
                    "field `{}` is initialized more than once",
                    self.interner.text(field_name)
                );
                self.error_expr(expr_id, message);
                continue;
            }
            seen.insert(field_name);

            if !self.infer.unify(value_ty, declared_ty)
                && !self.infer.coerce(field.value, value_ty, declared_ty)
            {
                let message = format!(
                    "value is not assignable to field `{}`",
                    self.interner.text(field_name)
                );
                self.error_expr(expr_id, message);
            }
        }

        for (declared_name, _) in &declared {
            if !seen.contains(declared_name) {
                let message = format!(
                    "missing field `{}` in `{}` literal",
                    self.interner.text(*declared_name),
                    self.interner.text(name),
                );
                self.error_expr(expr_id, message);
            }
        }

        self.infer.bind_expr_ty(expr_id, struct_ty)
    }

    fn infer_list_init_expr(&mut self, expr_id: ExprId, elements: &[ExprId]) -> TypeId {
        if elements.is_empty() {
            let var = self.infer.fresh_var(InferKind::General);
            let list_ty = self.infer.types.list_ty(var);
            return self.infer.bind_expr_ty(expr_id, list_ty);
        }

        let inner_ty = self.infer_expr(elements[0]);

        for element in elements.iter().skip(1) {
            let element_ty = self.infer_expr(*element);
            self.infer.unify(inner_ty, element_ty);
        }

        let list_ty = self.infer.types.list_ty(inner_ty);
        self.infer.bind_expr_ty(expr_id, list_ty)
    }

    fn infer_literal_expr(&mut self, expr_id: ExprId, literal: Literal) -> TypeId {
        let ty = match literal {
            Literal::Bool { .. } => self.infer.types.bool_ty(),
            Literal::String { .. } => self.infer.types.str_ty(),
            Literal::Int { .. } => self.infer.fresh_var(InferKind::Int),
            Literal::Float { .. } => self.infer.fresh_var(InferKind::Float),
            Literal::Missing => self.infer.types.error_ty(),
        };
        self.infer.bind_expr_ty(expr_id, ty)
    }

    fn poison(&mut self, expr_id: ExprId) -> TypeId {
        let error = self.infer.types.error_ty();
        self.infer.bind_expr_ty(expr_id, error)
    }

    fn resolve_return_annotation(&mut self, id: TypeAnnotationId) -> TypeId {
        match self.hir.annotation(id) {
            TypeAnnotation::Missing => self.infer.types.unit_ty(),
            _ => self.resolve_annotation(id),
        }
    }

    fn resolve_annotation(&mut self, id: TypeAnnotationId) -> TypeId {
        match self.hir.annotation(id) {
            TypeAnnotation::Named { name, args } => {
                self.resolve_named_annotation(id, name.symbol, args)
            }
            TypeAnnotation::Func { params, ret } => {
                let param_tys: Vec<TypeId> = params
                    .iter()
                    .map(|&param| self.resolve_annotation(param))
                    .collect();
                let ret_ty = self.resolve_annotation(*ret);
                self.infer.types.func_ty(param_tys, ret_ty)
            }
            TypeAnnotation::Self_ | TypeAnnotation::Missing => self.infer.types.error_ty(),
        }
    }

    fn resolve_named_annotation(
        &mut self,
        id: TypeAnnotationId,
        name: SymbolId,
        args: &[TypeAnnotationId],
    ) -> TypeId {
        let name_str = self.interner.text(name).to_string();

        // Container types take exactly one element type.
        if name_str == "List" || name_str == "Relation" {
            if args.len() != 1 {
                return self.infer.types.error_ty();
            }
            let inner = self.resolve_annotation(args[0]);
            return match name_str.as_str() {
                "List" => self.infer.types.list_ty(inner),
                _ => self.infer.types.relation_of_row(inner, None),
            };
        }

        let scalar = match name_str.as_str() {
            "int64" => Type::Int64,
            "int8" => Type::Int8,
            "int16" => Type::Int16,
            "int32" => Type::Int32,
            "uint8" => Type::UInt8,
            "uint16" => Type::UInt16,
            "uint32" => Type::UInt32,
            "uint64" => Type::UInt64,
            "float64" => Type::Float64,
            "float32" => Type::Float32,
            "bool" => Type::Bool,
            "str" => Type::String,
            "unit" => Type::Unit,
            _ => {
                if let Some(ty) = self.symbols.lookup_type(name) {
                    return ty;
                }
                let message = format!("unknown type `{name_str}`");
                self.report_annotation(id, message);
                return self.infer.types.error_ty();
            }
        };

        // A scalar takes no arguments.
        if !args.is_empty() {
            return self.infer.types.error_ty();
        }

        self.infer.types.intern_ty(scalar)
    }

    fn report_stmt(&mut self, id: StmtId, message: impl Into<String>) {
        let span = self
            .source_map
            .stmt(id)
            .expect("a reported node is always in the source map");

        self.diagnostics
            .emit(DiagnosticBuilder::error(span, message));
    }

    fn report_expr(&mut self, id: ExprId, message: impl Into<String>) {
        let span = self
            .source_map
            .expr(id)
            .expect("a reported node is always in the source map");
        self.diagnostics
            .emit(DiagnosticBuilder::error(span, message));
    }

    fn error_expr(&mut self, id: ExprId, message: impl Into<String>) -> TypeId {
        self.report_expr(id, message);
        self.poison(id)
    }

    fn report_annotation(&mut self, id: TypeAnnotationId, message: impl Into<String>) {
        let span = self
            .source_map
            .annotation(id)
            .expect("a reported node is always in the source map");
        self.diagnostics
            .emit(DiagnosticBuilder::error(span, message));
    }
}

enum ColumnLookup {
    Absent,
    Unique { ty: TypeId, column: u32 },
    Ambiguous,
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use yuzu_core::adt::{Int, StringInterner};

    use crate::{
        Expr, FuncParam, HirCtx, Ident, Literal, Mutability, Root, Stmt, StructField,
        StructFieldInit, TypeAnnotation,
        infer::test_support::{check, check_src},
    };

    fn assign_struct_field(
        hir: &mut HirCtx,
        interner: &mut StringInterner,
        field_mutability: Mutability,
    ) -> Root {
        let s_ty = interner.intern("S");
        let f = interner.intern("f");
        let s = interner.intern("s");
        let int64 = interner.intern("int64");

        let int_ann = hir.alloc_annotation(TypeAnnotation::Named {
            name: Ident { symbol: int64 },
            args: Box::new([]),
        });
        let struct_stmt = hir.alloc_stmt(Stmt::Struct {
            name: Ident { symbol: s_ty },
            fields: Box::new([StructField {
                name: Ident { symbol: f },
                mutability: field_mutability,
                type_annotation: int_ann,
            }]),
        });

        let init = hir.alloc_expr(Expr::Literal(Literal::Int {
            value: Int::from(0u64),
        }));
        let struct_init = hir.alloc_expr(Expr::StructInit {
            name: s_ty,
            fields: Box::new([StructFieldInit {
                name: Ident { symbol: f },
                value: init,
            }]),
        });
        let let_stmt = hir.alloc_stmt(Stmt::Let {
            name: Ident { symbol: s },
            mutability: Mutability::Mutable,
            type_annotation: None,
            expr: struct_init,
        });

        let base = hir.alloc_expr(Expr::Ident {
            value: Ident { symbol: s },
        });
        let target = hir.alloc_expr(Expr::FieldAccess {
            base,
            field: Ident { symbol: f },
        });
        let value = hir.alloc_expr(Expr::Literal(Literal::Int {
            value: Int::from(1u64),
        }));
        let assign = hir.alloc_stmt(Stmt::Assign { target, value });

        Root {
            stmts: Box::new([struct_stmt, let_stmt, assign]),
        }
    }

    fn assign_variable(
        hir: &mut HirCtx,
        interner: &mut StringInterner,
        mutability: Mutability,
    ) -> Root {
        let x = interner.intern("x");
        let init = hir.alloc_expr(Expr::Literal(Literal::Int {
            value: Int::from(0u64),
        }));
        let let_stmt = hir.alloc_stmt(Stmt::Let {
            name: Ident { symbol: x },
            mutability,
            type_annotation: None,
            expr: init,
        });
        let target = hir.alloc_expr(Expr::Ident {
            value: Ident { symbol: x },
        });
        let value = hir.alloc_expr(Expr::Literal(Literal::Int {
            value: Int::from(1u64),
        }));
        let assign = hir.alloc_stmt(Stmt::Assign { target, value });

        Root {
            stmts: Box::new([let_stmt, assign]),
        }
    }

    #[test]
    fn mutable_struct_field_is_assignable() {
        check(
            |hir, interner| assign_struct_field(hir, interner, Mutability::Mutable),
            expect![""],
        );
    }

    #[test]
    fn mutable_variable_is_assignable() {
        check(
            |hir, interner| assign_variable(hir, interner, Mutability::Mutable),
            expect![""],
        );
    }

    fn func_return(
        hir: &mut HirCtx,
        interner: &mut StringInterner,
        param: &str,
        ret: &str,
    ) -> Root {
        let f = interner.intern("f");
        let x = interner.intern("x");
        let param = interner.intern(param);
        let ret = interner.intern(ret);

        let param_ann = hir.alloc_annotation(TypeAnnotation::Named {
            name: Ident { symbol: param },
            args: Box::new([]),
        });
        let ret_ann = hir.alloc_annotation(TypeAnnotation::Named {
            name: Ident { symbol: ret },
            args: Box::new([]),
        });

        let x_ref = hir.alloc_expr(Expr::Ident {
            value: Ident { symbol: x },
        });
        let return_stmt = hir.alloc_stmt(Stmt::Return { expr: Some(x_ref) });
        let body = hir.alloc_stmt(Stmt::Block {
            stmts: Box::new([return_stmt]),
        });

        let func = hir.alloc_stmt(Stmt::Func {
            name: Ident { symbol: f },
            type_params: Box::new([]),
            params: Box::new([FuncParam {
                name: Ident { symbol: x },
                type_annotation: param_ann,
            }]),
            type_bounds: Box::new([]),
            ret_type_annotation: ret_ann,
            body: Some(body),
            is_agg: false,
            is_external: false,
        });

        Root {
            stmts: Box::new([func]),
        }
    }

    #[test]
    fn return_matching_type_is_ok() {
        check(
            |hir, interner| func_return(hir, interner, "int64", "int64"),
            expect![""],
        );
    }

    #[test]
    fn return_widens_to_return_type() {
        check(
            |hir, interner| func_return(hir, interner, "int32", "int64"),
            expect![""],
        );
    }

    fn int64_field(hir: &mut HirCtx, interner: &mut StringInterner, name: &str) -> StructField {
        let name = interner.intern(name);
        let int64 = interner.intern("int64");
        let annotation = hir.alloc_annotation(TypeAnnotation::Named {
            name: Ident { symbol: int64 },
            args: Box::new([]),
        });
        StructField {
            name: Ident { symbol: name },
            mutability: Mutability::Immutable,
            type_annotation: annotation,
        }
    }

    fn named_table(hir: &mut HirCtx, interner: &mut StringInterner, row: &str) -> Root {
        let struct_name = interner.intern("Row");
        let table = interner.intern("T");
        let row = interner.intern(row);

        let field = int64_field(hir, interner, "x");
        let struct_stmt = hir.alloc_stmt(Stmt::Struct {
            name: Ident {
                symbol: struct_name,
            },
            fields: Box::new([field]),
        });
        let table_stmt = hir.alloc_stmt(Stmt::Table {
            name: Ident { symbol: table },
            row: Ident { symbol: row },
        });

        Root {
            stmts: Box::new([struct_stmt, table_stmt]),
        }
    }

    #[test]
    fn named_table_over_struct_is_ok() {
        check(
            |hir, interner| named_table(hir, interner, "Row"),
            expect![""],
        );
    }

    #[test]
    fn inline_table_is_ok() {
        check(
            |hir, interner| {
                let table = interner.intern("T");
                let field = int64_field(hir, interner, "x");
                let table_stmt = hir.alloc_stmt(Stmt::InlineTable {
                    name: Ident { symbol: table },
                    fields: Box::new([field]),
                });
                Root {
                    stmts: Box::new([table_stmt]),
                }
            },
            expect![""],
        );
    }

    // --- infer_stmt: declarations ---

    #[test]
    fn src_struct_registration_ok() {
        check_src("struct P { x: int64 }", expect![""]);
    }

    #[test]
    fn src_named_table_over_struct_ok() {
        check_src("struct R { x: int64 }\ntable T = R", expect![""]);
    }

    #[test]
    fn src_named_table_over_non_struct() {
        check_src("table T = R", expect!["`R` is not a struct"]);
    }

    #[test]
    fn src_inline_table_ok() {
        check_src("table T = { x: int64 }", expect![""]);
    }

    #[test]
    fn src_func_ok() {
        check_src("fn f(x: int64) -> int64 { return x }", expect![""]);
    }

    // --- infer_stmt: let ---

    #[test]
    fn src_let_with_annotation_ok() {
        check_src("let x: int64 = 0", expect![""]);
    }

    #[test]
    fn src_let_widen_coerces() {
        check_src("let a: int32 = 0\nlet b: int64 = a", expect![""]);
    }

    #[test]
    fn src_let_mismatch() {
        check_src(
            "let x: bool = 0",
            expect!["value of type `Int64` is not assignable to `Bool`"],
        );
    }

    #[test]
    fn src_let_without_annotation_ok() {
        check_src("let x = 0", expect![""]);
    }

    // --- infer_stmt: assign ---

    #[test]
    fn src_assign_mutable_var_ok() {
        check_src("let mut x: int64 = 0\nx = 1", expect![""]);
    }

    #[test]
    fn src_assign_immutable_var() {
        check_src(
            "let x: int64 = 0\nx = 1",
            expect!["cannot assign to immutable variable `x`"],
        );
    }

    #[test]
    fn src_assign_value_mismatch() {
        check_src(
            "let mut x: int64 = 0\nx = true",
            expect!["value of type `Bool` is not assignable to `Int64`"],
        );
    }

    #[test]
    fn src_assign_immutable_struct_field() {
        check_src(
            "struct S { f: int64 }\nlet mut s = S { f: 0 }\ns.f = 1",
            expect!["cannot assign to immutable field `f`"],
        );
    }

    #[test]
    fn src_assign_to_non_place() {
        check_src("1 = 2", expect!["cannot assign to this expression"]);
    }

    #[test]
    fn src_assign_field_of_non_place_base() {
        check_src(
            "struct S { x: int64 }\nfn g() -> S { return S { x: 0 } }\ng().x = 1",
            expect![[r#"
                cannot assign to immutable field `x`
                cannot assign to this expression"#]],
        );
    }

    // --- infer_stmt: return ---

    #[test]
    fn src_return_matching_ok() {
        check_src("fn f() -> int64 { return 0 }", expect![""]);
    }

    #[test]
    fn src_return_coerce_widens() {
        check_src(
            "fn f() -> int64 { let a: int32 = 0\nreturn a }",
            expect![""],
        );
    }

    #[test]
    fn src_return_mismatch() {
        check_src(
            "fn f() -> int64 { return true }",
            expect!["value of type `Bool` is not assignable to return type `Int64`"],
        );
    }

    #[test]
    fn src_return_without_value_unit_ok() {
        check_src("fn f() { return }", expect![""]);
    }

    #[test]
    fn src_return_without_value_nonunit() {
        check_src(
            "fn f() -> int64 { return }",
            expect!["value of type `Unit` is not assignable to return type `Int64`"],
        );
    }

    #[test]
    fn src_return_outside_func_ok() {
        check_src("return 0", expect![""]);
    }

    #[test]
    fn src_expr_stmt_ok() {
        check_src("1 + 2", expect![""]);
    }

    // --- infer_expr: literals ---

    #[test]
    fn src_literal_bool_ok() {
        check_src("let a: bool = true", expect![""]);
    }

    #[test]
    fn src_literal_string_ok() {
        check_src("let a: str = \"hi\"", expect![""]);
    }

    #[test]
    fn src_literal_float_ok() {
        check_src("let a: float64 = 0.0", expect![""]);
    }

    // --- infer_expr: ident ---

    #[test]
    fn src_ident_unresolved() {
        check_src("let x = y", expect!["unresolved identifier `y`"]);
    }

    // --- infer_expr: func call ---

    #[test]
    fn src_func_call_ok() {
        check_src(
            "fn f(x: int64) -> int64 { return x }\nlet r: int64 = f(0)",
            expect![""],
        );
    }

    #[test]
    fn src_call_not_callable() {
        check_src(
            "let x: int64 = 0\nlet y = x(1)",
            expect!["type `Int64` is not callable"],
        );
    }

    #[test]
    fn src_call_too_many_args() {
        check_src(
            "fn f(x: int64) -> int64 { return x }\nf(0, 1)",
            expect!["expected 1 argument(s), found 2"],
        );
    }

    #[test]
    fn src_call_too_few_args() {
        check_src(
            "fn f(x: int64) -> int64 { return x }\nf()",
            expect!["expected 1 argument(s), found 0"],
        );
    }

    #[test]
    fn src_call_arg_not_assignable() {
        check_src(
            "fn f(x: bool) -> bool { return x }\nf(0)",
            expect!["argument of type `Int64` is not assignable to parameter of type `Bool`"],
        );
    }

    // --- infer_expr: field access ---

    #[test]
    fn src_field_access_ok() {
        check_src(
            "struct S { x: int64 }\nlet s = S { x: 0 }\nlet v: int64 = s.x",
            expect![""],
        );
    }

    #[test]
    fn src_field_access_unknown_field() {
        check_src(
            "struct S { x: int64 }\nlet s = S { x: 0 }\nlet v = s.y",
            expect!["struct `S` has no field `y`"],
        );
    }

    #[test]
    fn src_field_access_on_non_struct() {
        check_src(
            "let a: int64 = 0\nlet b = a.x",
            expect!["type `Int64` has no fields"],
        );
    }

    // --- infer_expr: struct init ---

    #[test]
    fn src_struct_init_unknown_struct() {
        check_src("let s = M { x: 0 }", expect!["`M` is not a struct"]);
    }

    #[test]
    fn src_struct_init_missing_field() {
        check_src(
            "struct S { x: int64, y: int64 }\nlet s = S { x: 0 }",
            expect!["missing field `y` in `S` literal"],
        );
    }

    #[test]
    fn src_struct_init_unknown_field() {
        check_src(
            "struct S { x: int64 }\nlet s = S { x: 0, y: 1 }",
            expect!["struct `S` has no field `y`"],
        );
    }

    #[test]
    fn src_struct_init_mismatched_field() {
        check_src(
            "struct S { x: bool }\nlet s = S { x: 0 }",
            expect!["value is not assignable to field `x`"],
        );
    }

    #[test]
    fn src_struct_init_duplicate_field() {
        check_src(
            "struct S { x: int64 }\nlet s = S { x: 0, x: 1 }",
            expect!["field `x` is initialized more than once"],
        );
    }

    // --- infer_expr: list init ---

    #[test]
    fn src_list_init_ok() {
        check_src("let l: List[int64] = [1, 2, 3]", expect![""]);
    }

    #[test]
    fn src_list_init_empty_ok() {
        check_src("let l = []", expect![""]);
    }

    // --- coerce: non-numeric source is rejected ---

    #[test]
    fn src_coerce_non_numeric_source_rejected() {
        check_src(
            "let a: bool = true\nlet b: int64 = a",
            expect!["value of type `Bool` is not assignable to `Int64`"],
        );
    }

    // --- infer_rel: queries ---

    const TABLE: &str = "struct Row { a: int64, active: bool }\ntable t = Row\n";

    #[test]
    fn src_select_bare_column_ok() {
        check_src(&format!("{TABLE}let q = from t |> select a"), expect![""]);
    }

    #[test]
    fn src_alias_field_and_where_bool_ok() {
        check_src(
            &format!("{TABLE}let q = from t as e |> where e.active |> select e.a"),
            expect![""],
        );
    }

    #[test]
    fn src_where_non_bool_predicate() {
        check_src(
            &format!("{TABLE}let q = from t |> where a"),
            expect!["expected ``where` predicate` to be a `bool` type, but found `Int64`"],
        );
    }

    #[test]
    fn src_relation_name_qualifies_when_there_is_no_alias() {
        check_src(
            &format!("{TABLE}let q = from t |> select t.a"),
            expect![[r#""#]],
        );
    }

    #[test]
    fn src_alias_shadows_the_relation_name() {
        check_src(
            &format!("{TABLE}let q = from t e |> select t.a"),
            expect!["`t` is aliased here; qualify its columns with the alias"],
        );
    }

    #[test]
    fn src_relation_is_not_a_value() {
        check_src(
            &format!("{TABLE}let q = t"),
            expect!["`t` is a relation, not a value"],
        );
    }

    #[test]
    fn src_from_unknown_table() {
        check_src(
            "let q = from nope |> select x",
            expect!["`nope` is not a relation"],
        );
    }

    #[test]
    fn src_select_unknown_column() {
        check_src(
            &format!("{TABLE}let q = from t |> select missing"),
            expect!["unresolved identifier `missing`"],
        );
    }

    #[test]
    fn src_drop_removes_column_from_row() {
        check_src(
            &format!("{TABLE}let q = from t |> drop a |> select a"),
            expect!["unresolved identifier `a`"],
        );
    }

    #[test]
    fn src_rename_then_select_new_name_ok() {
        check_src(
            &format!("{TABLE}let q = from t |> rename a as x |> select x"),
            expect![""],
        );
    }

    #[test]
    fn src_extend_then_select_computed_ok() {
        check_src(
            &format!("{TABLE}let q = from t |> extend a as b |> select b"),
            expect![""],
        );
    }

    #[test]
    fn src_invalid_operator_application_reports() {
        check_src(
            "let x = true + false",
            expect!["operator `+` cannot be applied to `Bool` and `Bool`"],
        );
    }

    #[test]
    fn src_mismatched_membership_reports() {
        check_src(
            "let f = 1.5\nlet xs = [1, 2]\nlet x = f in xs",
            expect!["operator `in` cannot be applied to `Float64` and `List[Int64]`"],
        );
    }

    #[test]
    fn src_distinct_passthrough_ok() {
        check_src(
            &format!("{TABLE}let q = from t |> distinct |> select a"),
            expect![""],
        );
    }

    // --- infer_join_rel ---

    const JOIN_TABLES: &str = "struct Dept { a: int64, name: str }\ntable d = Dept\nstruct Info { code: int64, label: str }\ntable i = Info\nstruct Tag { a: str, note: str }\ntable g = Tag\n";

    #[test]
    fn src_join_on_ok() {
        check_src(
            &format!(
                "{TABLE}{JOIN_TABLES}let q = from t e |> join i x on e.a == x.code |> select e.a, x.label"
            ),
            expect![""],
        );
    }

    #[test]
    fn src_join_on_sees_both_rows_by_bare_name() {
        check_src(
            &format!("{TABLE}{JOIN_TABLES}let q = from t |> join i on a == code |> select label"),
            expect![""],
        );
    }

    #[test]
    fn src_join_on_shared_column_name_is_allowed() {
        check_src(
            &format!("{TABLE}{JOIN_TABLES}let q = from t e |> join d x on e.a == x.a"),
            expect![[r#""#]],
        );
    }

    #[test]
    fn src_qualified_columns_resolve_on_either_side_of_a_join() {
        check_src(
            &format!(
                "{TABLE}{JOIN_TABLES}let q = from t e |> join d x on e.a == x.a |> select e.a as l, x.a as r, x.name"
            ),
            expect![[r#""#]],
        );
    }

    #[test]
    fn src_bare_column_shared_by_both_sides_is_ambiguous() {
        check_src(
            &format!("{TABLE}{JOIN_TABLES}let q = from t e |> join d x on e.a == x.a |> select a"),
            expect!["column `a` is ambiguous; qualify it with a relation alias"],
        );
    }

    #[test]
    fn src_bare_column_from_one_side_still_resolves() {
        check_src(
            &format!(
                "{TABLE}{JOIN_TABLES}let q = from t e |> join d x on e.a == x.a |> select active, name"
            ),
            expect![[r#""#]],
        );
    }

    #[test]
    fn src_chained_joins_resolve_each_qualified_column() {
        check_src(
            &format!(
                "{TABLE}{JOIN_TABLES}let q = from t e |> join d x on e.a == x.a |> join g y on x.name == y.a |> select e.a as ea, x.a as xa, y.note"
            ),
            expect![[r#""#]],
        );
    }

    #[test]
    fn src_join_on_non_bool_condition() {
        check_src(
            &format!("{TABLE}{JOIN_TABLES}let q = from t e |> join i x on e.a"),
            expect!["expected ``on` condition` to be a `bool` type, but found `Int64`"],
        );
    }

    #[test]
    fn src_join_using_carries_both_sides_columns() {
        check_src(
            &format!(
                "{TABLE}{JOIN_TABLES}let q = from t e |> join d x using (a) |> select e.a, x.a"
            ),
            expect![[r#""#]],
        );
    }

    #[test]
    fn src_bare_using_key_is_ambiguous() {
        check_src(
            &format!("{TABLE}{JOIN_TABLES}let q = from t e |> join d x using (a) |> select a"),
            expect!["column `a` is ambiguous; qualify it with a relation alias"],
        );
    }

    #[test]
    fn src_join_using_unknown_column() {
        check_src(
            &format!("{TABLE}{JOIN_TABLES}let q = from t |> join d using (missing)"),
            expect!["column missing not present in both relations"],
        );
    }

    #[test]
    fn src_join_using_column_missing_on_the_right() {
        check_src(
            &format!("{TABLE}{JOIN_TABLES}let q = from t |> join i using (a)"),
            expect!["column a not present in both relations"],
        );
    }

    #[test]
    fn src_join_using_column_types_must_match() {
        check_src(
            &format!("{TABLE}{JOIN_TABLES}let q = from t |> join g using (a)"),
            expect!["column a is `Int64` on the left and `String` on the right"],
        );
    }

    #[test]
    fn src_using_key_is_named_through_the_left_side_only() {
        check_src(
            &format!("{TABLE}{JOIN_TABLES}let q = from t e |> join d x using (a) |> select e.a"),
            expect![[r#""#]],
        );
    }

    #[test]
    fn src_using_key_is_not_named_through_the_right_side() {
        check_src(
            &format!("{TABLE}{JOIN_TABLES}let q = from t e |> join d x using (a) |> select x.a"),
            expect![[r#""#]],
        );
    }

    #[test]
    fn src_join_using_leaves_a_shared_non_key_column_ambiguous() {
        check_src(
            "struct L { k: int64, v: int64 }\ntable l = L\nstruct R { k: int64, v: int64 }\ntable r = R\nlet q = from l a |> join r b using (k) |> select v",
            expect!["column `v` is ambiguous; qualify it with a relation alias"],
        );
    }

    #[test]
    fn src_join_using_reports_every_bad_column() {
        check_src(
            &format!("{TABLE}{JOIN_TABLES}let q = from t |> join g using (a, active, missing)"),
            expect![[r#"
                column a is `Int64` on the left and `String` on the right
                column active not present in both relations
                column missing not present in both relations"#]],
        );
    }

    #[test]
    fn src_qualified_rename_names_one_side() {
        check_src(
            &format!(
                "{TABLE}{JOIN_TABLES}let q = from t e |> join d x on e.a == x.a |> rename e.a as ea, x.a as xa |> select ea, xa"
            ),
            expect![[r#""#]],
        );
    }

    #[test]
    fn src_bare_rename_of_a_shared_column_is_ambiguous() {
        check_src(
            &format!(
                "{TABLE}{JOIN_TABLES}let q = from t e |> join d x on e.a == x.a |> rename a as z"
            ),
            expect!["column `a` is ambiguous; qualify it with a relation alias"],
        );
    }

    #[test]
    fn src_qualified_rename_of_an_unknown_column() {
        check_src(
            &format!(
                "{TABLE}{JOIN_TABLES}let q = from t e |> join d x on e.a == x.a |> rename e.nope as z"
            ),
            expect!["`e` has no column `nope`"],
        );
    }

    #[test]
    fn src_alias_that_outlived_its_row() {
        check_src(
            &format!("{TABLE}let q = from t e |> select a |> where e.active"),
            expect!["`e` has no column `active` here"],
        );
    }

    #[test]
    fn src_alias_has_no_such_column() {
        check_src(
            &format!("{TABLE}let q = from t e |> where e.nope"),
            expect!["`e` has no column `nope` here"],
        );
    }

    #[test]
    fn src_relation_annotation_accepts_an_aliased_query() {
        check_src(
            &format!("{TABLE}let q: Relation[Row] = from t e |> where e.active"),
            expect![[r#""#]],
        );
    }

    #[test]
    fn src_relation_annotation_ignores_which_alias_was_used() {
        check_src(
            &format!("{TABLE}let q: Relation[Row] = from t |> where active"),
            expect![[r#""#]],
        );
    }

    #[test]
    fn src_relation_annotation_still_checks_the_columns() {
        check_src(
            &format!("{TABLE}let q: Relation[Row] = from t |> select a"),
            expect!["value of type `Relation[a]` is not assignable to `Relation[a, active]`"],
        );
    }

    #[test]
    fn src_set_replaces_a_column_in_place() {
        check_src(
            &format!("{TABLE}let q = from t |> set a = a + 1 |> select a, active"),
            expect![[r#""#]],
        );
    }

    #[test]
    fn src_set_of_an_unknown_column() {
        check_src(
            &format!("{TABLE}let q = from t |> set nope = 1"),
            expect!["column `nope` is not in this row"],
        );
    }

    #[test]
    fn src_set_of_an_ambiguous_column() {
        check_src(
            &format!("{TABLE}{JOIN_TABLES}let q = from t e |> join d x on e.a == x.a |> set a = 1"),
            expect!["column `a` is ambiguous; qualify it with a relation alias"],
        );
    }

    #[test]
    fn src_chained_registry_aliases_a_builtin() {
        use yuzu_types::{AggFunc, BuiltinFunc};
        use yuzu_types::{Builtins, FunctionRegistry, FunctionRegistryEntry, chain};

        struct Total;
        const TOTAL: &[FunctionRegistryEntry] = &[FunctionRegistryEntry {
            name: "total",
            func: BuiltinFunc::Aggregate(AggFunc::Sum),
            min_args: 1,
            max_args: 1,
        }];
        impl FunctionRegistry for Total {
            fn entries(&self) -> &[FunctionRegistryEntry] {
                TOTAL
            }
        }

        let chained = chain(vec![Box::new(Total), Box::new(Builtins)]);
        crate::infer::test_support::check_src_with(
            &chained,
            &format!("{TABLE}let q = from t |> aggregate total(a) as v group by active"),
            expect![""],
        );
    }

    const EXTERNALS: &str =
        "external fn upper(s: str) -> str\nexternal agg fn median(x: int64) -> float64\n";

    #[test]
    fn src_external_scalar_calls_anywhere() {
        check_src(
            &format!("{TABLE}{EXTERNALS}let q = from t |> where upper(\"a\") == \"A\" |> select a"),
            expect![""],
        );
    }

    #[test]
    fn src_external_agg_in_an_aggregate() {
        check_src(
            &format!(
                "{TABLE}{EXTERNALS}let q = from t |> aggregate median(a) as m group by active"
            ),
            expect![""],
        );
    }

    #[test]
    fn src_external_agg_outside_an_aggregate() {
        check_src(
            &format!("{TABLE}{EXTERNALS}let q = from t |> select median(a) as m"),
            expect!["aggregate function `median` can only be used in an `aggregate` item"],
        );
    }

    #[test]
    fn src_external_signature_still_checks() {
        check_src(
            &format!("{TABLE}{EXTERNALS}let q = from t |> select upper(a) as u"),
            expect!["argument of type `Int64` is not assignable to parameter of type `String`"],
        );
    }

    #[test]
    fn src_aggregate_bare_call() {
        check_src(
            &format!("{TABLE}let q = from t |> aggregate sum(a) as s group by active"),
            expect![""],
        );
    }

    #[test]
    fn src_aggregate_full_table() {
        check_src(
            &format!("{TABLE}let q = from t |> aggregate count()"),
            expect![""],
        );
    }

    #[test]
    fn src_aggregate_composite_item() {
        check_src(
            &format!(
                "{TABLE}let q = from t |> aggregate max(a) - min(a) as spread group by active"
            ),
            expect![""],
        );
    }

    #[test]
    fn src_aggregate_key_is_reachable_at_group_level() {
        check_src(
            &format!("{TABLE}let q = from t |> aggregate a + count() as v group by a"),
            expect![""],
        );
    }

    #[test]
    fn src_aggregate_key_alias_names_the_output() {
        check_src(
            &format!(
                "{TABLE}let q = from t |> aggregate count() as n group by active as is_active |> select is_active, n"
            ),
            expect![""],
        );
    }

    #[test]
    fn src_aggregate_outside_aggregate_item() {
        check_src(
            &format!("{TABLE}let q = from t |> select sum(a)"),
            expect!["aggregate function `sum` can only be used in an `aggregate` item"],
        );
    }

    #[test]
    fn src_aggregate_cannot_nest() {
        check_src(
            &format!("{TABLE}let q = from t |> aggregate sum(min(a)) as v"),
            expect!["aggregate function `min` cannot be nested in another aggregate"],
        );
    }

    #[test]
    fn src_group_level_expressions_may_combine_keys() {
        check_src(
            &format!("{TABLE}let q = from t |> aggregate sum(a) as s, a + a as x group by a"),
            expect![""],
        );
    }

    #[test]
    fn src_chained_aggregates_each_resolve_their_own_input() {
        check_src(
            &format!(
                "{TABLE}let q = from t |> aggregate sum(a) as s group by active |> aggregate max(s) as m group by active"
            ),
            expect![""],
        );
    }

    #[test]
    fn src_aggregate_arguments_do_not_need_grouped_columns() {
        check_src(
            &format!("{TABLE}let q = from t |> aggregate sum(a + a) as s group by active"),
            expect![""],
        );
    }

    #[test]
    fn src_every_group_level_column_must_be_a_key() {
        check_src(
            &format!("{TABLE}let q = from t |> aggregate a + 1 as x group by active"),
            expect![
                "column `a` must be one of the group keys (`active`) or inside an aggregate function"
            ],
        );
    }

    #[test]
    fn src_aggregate_non_key_column_at_group_level() {
        check_src(
            &format!("{TABLE}let q = from t |> aggregate a + count() as v group by active"),
            expect![
                "column `a` must be one of the group keys (`active`) or inside an aggregate function"
            ],
        );
    }

    #[test]
    fn src_aggregate_builtin_is_not_a_value() {
        check_src(
            &format!("{TABLE}let q = from t |> select sum as v"),
            expect!["`sum` is a function, not a value"],
        );
    }

    #[test]
    fn src_aggregate_unknown_group_key() {
        check_src(
            &format!("{TABLE}let q = from t |> aggregate count() group by missing"),
            expect!["group key `missing` is not a column of this row"],
        );
    }

    #[test]
    fn src_aggregate_needs_numeric_argument() {
        check_src(
            &format!("{TABLE}let q = from t |> aggregate sum(active) as v"),
            expect!["`sum` needs a numeric argument, but found `Bool`"],
        );
    }

    #[test]
    fn src_aggregate_count_takes_an_optional_argument() {
        check_src(
            &format!("{TABLE}let q = from t |> aggregate count(a) as n"),
            expect![""],
        );
        check_src(
            &format!("{TABLE}let q = from t |> aggregate count(a, active) as n"),
            expect!["`count` expects 0 to 1 argument(s), found 2"],
        );
    }

    #[test]
    fn src_aggregate_count_distinct() {
        check_src(
            &format!("{TABLE}let q = from t |> aggregate count_distinct(a) as kinds"),
            expect![""],
        );
        check_src(
            &format!("{TABLE}let q = from t |> aggregate count_distinct() as kinds"),
            expect!["`count_distinct` expects 1 argument(s), found 0"],
        );
    }

    #[test]
    fn src_aggregate_user_fn_shadows_a_builtin() {
        check_src(
            &format!(
                "{TABLE}fn sum(x: int64) -> int64 {{ return x }}\nlet q = from t |> select sum(a) as v"
            ),
            expect![""],
        );
    }

    #[test]
    fn src_agg_fn_used_in_an_aggregate() {
        check_src(
            &format!(
                "{TABLE}agg fn spread(x: int64) -> int64 {{ return max(x) - min(x) }}\nlet q = from t |> aggregate spread(a) as v group by active"
            ),
            expect![""],
        );
    }

    #[test]
    fn src_agg_fn_outside_an_aggregate() {
        check_src(
            &format!(
                "{TABLE}agg fn spread(x: int64) -> int64 {{ return max(x) - min(x) }}\nlet q = from t |> select spread(a) as v"
            ),
            expect!["aggregate function `spread` can only be used in an `aggregate` item"],
        );
    }

    #[test]
    fn src_agg_fn_cannot_nest_in_an_aggregate() {
        check_src(
            &format!(
                "{TABLE}agg fn spread(x: int64) -> int64 {{ return max(x) - min(x) }}\nlet q = from t |> aggregate sum(spread(a)) as v"
            ),
            expect!["aggregate function `spread` cannot be nested in another aggregate"],
        );
    }

    #[test]
    fn src_group_value_cannot_reenter_an_aggregate() {
        check_src(
            &format!("{TABLE}agg fn sneaky(x: int64) -> int64 {{ let m = max(x) return sum(m) }}"),
            expect![
                "`m` is a group-level value and cannot be used inside an aggregate's arguments"
            ],
        );
    }

    #[test]
    fn src_agg_fn_cannot_call_itself() {
        check_src(
            &format!("{TABLE}agg fn bad(x: int64) -> int64 {{ return sum(x) + bad(x) }}"),
            expect!["`bad` is an `agg fn` and cannot call itself"],
        );
    }

    #[test]
    fn src_agg_fn_needs_an_aggregate_call() {
        check_src(
            &format!("{TABLE}agg fn nothing(x: int64) -> int64 {{ return x + 1 }}"),
            expect![[r#"
                parameter `x` can only be used inside an aggregate function's arguments
                an `agg fn` must use an aggregate function"#]],
        );
    }

    #[test]
    fn src_agg_fn_param_at_group_level() {
        check_src(
            &format!("{TABLE}agg fn off(x: int64) -> int64 {{ return sum(x) + x }}"),
            expect!["parameter `x` can only be used inside an aggregate function's arguments"],
        );
    }

    #[test]
    fn src_plain_fn_cannot_aggregate() {
        check_src(
            &format!("{TABLE}fn sneaky(x: int64) -> int64 {{ return sum(x) }}"),
            expect!["aggregate function `sum` can only be used in an `aggregate` item"],
        );
    }

    #[test]
    fn src_agg_fn_is_not_a_value() {
        check_src(
            &format!(
                "{TABLE}agg fn spread(x: int64) -> int64 {{ return max(x) - min(x) }}\nlet q = from t |> select spread as v"
            ),
            expect!["`spread` is an aggregate function, not a value"],
        );
    }

    #[test]
    fn src_limit_needs_an_integer() {
        check_src(
            &format!("{TABLE}let q = from t |> limit active"),
            expect!["`limit` needs an integer, but found `Bool`"],
        );
    }

    #[test]
    fn src_alias_renames_the_whole_row() {
        check_src(
            &format!("{TABLE}let q = from t e |> as u |> select u.a"),
            expect![[r#""#]],
        );
    }

    #[test]
    fn src_alias_replaces_the_previous_one() {
        check_src(
            &format!("{TABLE}let q = from t e |> as u |> select e.a"),
            expect!["`e` has no column `a` here"],
        );
    }

    #[test]
    fn src_join_unknown_relation() {
        check_src(
            &format!("{TABLE}let q = from t |> join nope on a == 1"),
            expect!["`nope` is not a relation"],
        );
    }
}
