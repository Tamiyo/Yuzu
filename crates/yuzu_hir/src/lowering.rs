use yuzu_diagnostics::{
    diagnostics::{Span, builder::DiagnosticBuilder, engine::DiagnosticsEngine},
    source_map::SourceId,
};

use crate::{HirCtx, HirSourceMap, hir::*};
use yuzu_ast::ast;
use yuzu_core::adt::{Float, Int, StringInterner};
use yuzu_syntax::SyntaxNodePtr;

pub fn lower(
    root: ast::Root,
    ctx: &mut HirCtx,
    interner: &mut StringInterner,
    diagnostics: &mut DiagnosticsEngine,
    source_id: SourceId,
) -> (Root, HirSourceMap) {
    HirLowerer {
        ctx,
        interner,
        diagnostics,
        source_map: HirSourceMap::default(),
        source_id,
    }
    .lower(root)
}

struct HirLowerer<'l> {
    ctx: &'l mut HirCtx,
    interner: &'l mut StringInterner,
    diagnostics: &'l mut DiagnosticsEngine,
    source_map: HirSourceMap,
    source_id: SourceId,
}

impl<'l> HirLowerer<'l> {
    pub fn lower(mut self, root: ast::Root) -> (Root, HirSourceMap) {
        let stmts = root.stmts().map(|stmt| self.lower_stmt(stmt)).collect();
        (Root { stmts }, self.source_map)
    }

    pub fn lower_stmt(&mut self, stmt: ast::Stmt) -> StmtId {
        let ptr = self.ptr_of(&stmt);
        let lowered = match stmt {
            ast::Stmt::StructStmt(struct_stmt) => self.lower_struct_stmt(struct_stmt),
            ast::Stmt::TraitStmt(trait_stmt) => self.lower_trait_stmt(trait_stmt),
            ast::Stmt::ImplStmt(impl_stmt) => self.lower_impl_stmt(impl_stmt),
            ast::Stmt::FuncStmt(func_stmt) => self.lower_func_stmt(func_stmt),
            ast::Stmt::TableStmt(table_stmt) => self.lower_table_stmt(table_stmt),
            ast::Stmt::BlockStmt(block_stmt) => return self.lower_block_stmt(block_stmt),
            ast::Stmt::LetStmt(let_stmt) => self.lower_let_stmt(let_stmt),
            ast::Stmt::AssignStmt(assign_stmt) => self.lower_assign_stmt(assign_stmt),
            ast::Stmt::ReturnStmt(return_stmt) => self.lower_return_stmt(return_stmt),
            ast::Stmt::ExprStmt(expr_stmt) => self.lower_expr_stmt(expr_stmt),
        };

        let id = self.ctx.alloc_stmt(lowered);
        self.source_map.bind_stmt(id, self.source_id, ptr);
        id
    }

    fn lower_block_stmt(&mut self, stmt: ast::BlockStmt) -> StmtId {
        let ptr = self.ptr_of(&stmt);
        let mut stmts = Vec::new();
        for inner in stmt.stmts() {
            stmts.push(self.lower_stmt(inner));
        }

        let id = self.ctx.alloc_stmt(Stmt::Block {
            stmts: stmts.into_boxed_slice(),
        });
        self.source_map.bind_stmt(id, self.source_id, ptr);
        id
    }

    fn lower_struct_stmt(&mut self, stmt: ast::StructStmt) -> Stmt {
        let Some(name) = stmt.name() else {
            self.error(&stmt, "struct is missing its name");
            return Stmt::Missing;
        };
        let name = self.lower_ident(name);

        let mut fields = Vec::new();
        for field in stmt.fields() {
            if let Some(field) = self.lower_struct_field_decl(field) {
                fields.push(field);
            }
        }

        Stmt::Struct {
            name,
            fields: fields.into_boxed_slice(),
        }
    }

    fn lower_struct_field_decl(&mut self, field: ast::StructField) -> Option<StructField> {
        let Some(name) = field.name() else {
            self.error(&field, "struct field is incomplete");
            return None;
        };
        let Some(ty) = field.ty() else {
            self.error(&field, "struct field is incomplete");
            return None;
        };

        Some(StructField {
            name: self.lower_ident(name),
            mutability: lower_mutability(field.mutability()),
            type_annotation: self.lower_type_annotation(ty),
        })
    }

    fn lower_table_stmt(&mut self, stmt: ast::TableStmt) -> Stmt {
        let Some(name) = stmt.name() else {
            self.error(&stmt, "table is missing its name");
            return Stmt::Missing;
        };
        let name = self.lower_ident(name);

        match stmt.row_struct() {
            Some(row) => Stmt::Table {
                name,
                row: self.lower_ident(row),
            },
            None => {
                let mut fields = Vec::new();
                for field in stmt.inline_fields() {
                    if let Some(field) = self.lower_struct_field_decl(field) {
                        fields.push(field);
                    }
                }
                Stmt::InlineTable {
                    name,
                    fields: fields.into_boxed_slice(),
                }
            }
        }
    }

    fn lower_func_stmt(&mut self, stmt: ast::FuncStmt) -> Stmt {
        let Some(name) = stmt.name() else {
            self.error(&stmt, "function is missing its name");
            return Stmt::Missing;
        };
        let name = self.lower_ident(name);

        let mut type_params = Vec::new();
        for type_param in stmt.type_params() {
            if let Some(type_param) = type_param.name() {
                type_params.push(self.lower_ident(type_param));
            }
        }

        let mut params = Vec::new();
        for param in stmt.params() {
            if let Some(param) = self.lower_param(param) {
                params.push(param);
            }
        }

        let mut type_bounds = Vec::new();
        for bound in stmt.bounds() {
            if let Some(bound) = self.lower_type_bound(bound) {
                type_bounds.push(bound);
            }
        }

        let ret_type_annotation = self.lower_type_annotation_opt(stmt.result());
        let is_external = stmt.is_external();
        let body = stmt.body().map(|block| self.lower_block_stmt(block));
        if is_external && body.is_some() {
            self.error(&stmt, "an external function cannot have a body");
        }
        if !is_external && body.is_none() {
            self.error(&stmt, "function is missing its body");
        }

        Stmt::Func {
            name,
            type_params: type_params.into_boxed_slice(),
            params: params.into_boxed_slice(),
            type_bounds: type_bounds.into_boxed_slice(),
            ret_type_annotation,
            body,
            is_agg: stmt.is_agg(),
            is_external,
        }
    }

    fn lower_param(&mut self, param: ast::FuncParam) -> Option<FuncParam> {
        let Some(name) = param.name() else {
            self.error(&param, "parameter is missing its name");
            return None;
        };

        let type_annotation = match param.ty() {
            Some(ty) => self.lower_type_annotation(ty),
            None if name.text().as_deref() == Some("self") => {
                self.ctx.alloc_annotation(TypeAnnotation::Self_)
            }
            None => {
                self.error(&param, "parameter is missing its type");
                self.ctx.alloc_annotation(TypeAnnotation::Missing)
            }
        };

        Some(FuncParam {
            name: self.lower_ident(name),
            type_annotation,
        })
    }

    fn lower_impl_stmt(&mut self, stmt: ast::ImplStmt) -> Stmt {
        let Some(name) = stmt.ty() else {
            self.error(&stmt, "`impl` is missing its type name");
            return Stmt::Missing;
        };
        let name = self.lower_ident(name);
        let trait_ref = stmt
            .trait_()
            .and_then(|trait_ref| self.lower_trait_ref(trait_ref));

        let mut methods = Vec::new();
        for method in stmt.methods() {
            methods.push(self.lower_stmt(ast::Stmt::FuncStmt(method)));
        }

        Stmt::Impl {
            trait_ref,
            name,
            methods: methods.into_boxed_slice(),
        }
    }

    fn lower_trait_stmt(&mut self, stmt: ast::TraitStmt) -> Stmt {
        let Some(name) = stmt.name() else {
            self.error(&stmt, "trait is missing its name");
            return Stmt::Missing;
        };
        let name = self.lower_ident(name);

        let mut methods = Vec::new();
        for method in stmt.methods() {
            methods.push(self.lower_stmt(ast::Stmt::FuncStmt(method)));
        }

        Stmt::Trait {
            name,
            methods: methods.into_boxed_slice(),
        }
    }

    fn lower_let_stmt(&mut self, stmt: ast::LetStmt) -> Stmt {
        let Some(name) = stmt.name() else {
            self.error(&stmt, "let binding is missing its name");
            return Stmt::Missing;
        };
        let name = self.lower_ident(name);

        let Some(expr) = stmt.expr() else {
            self.error(&stmt, "let binding is missing its expression");
            return Stmt::Missing;
        };
        let expr = self.lower_expr(expr);

        let type_annotation = stmt
            .type_annotation()
            .map(|annotation| self.lower_type_annotation(annotation));

        Stmt::Let {
            name,
            mutability: lower_mutability(stmt.mutability()),
            type_annotation,
            expr,
        }
    }

    fn lower_assign_stmt(&mut self, stmt: ast::AssignStmt) -> Stmt {
        let Some(target) = stmt.target() else {
            self.error(&stmt, "assignment is missing its target");
            return Stmt::Missing;
        };
        let target = self.lower_expr(target);

        let Some(value) = stmt.value() else {
            self.error(&stmt, "assignment is missing its value");
            return Stmt::Missing;
        };
        let value = self.lower_expr(value);

        Stmt::Assign { target, value }
    }

    fn lower_return_stmt(&mut self, stmt: ast::ReturnStmt) -> Stmt {
        let expr = stmt.expr().map(|expr| self.lower_expr(expr));
        Stmt::Return { expr }
    }

    fn lower_expr_stmt(&mut self, stmt: ast::ExprStmt) -> Stmt {
        let Some(expr) = stmt.expr() else {
            self.error(&stmt, "expression statement is incomplete");
            return Stmt::Missing;
        };

        Stmt::Expr {
            expr: self.lower_expr(expr),
        }
    }

    fn lower_type_bound(&mut self, bound: ast::TypeBound) -> Option<TypeBound> {
        let Some(subject) = bound.subject() else {
            self.error(&bound, "type bound is missing its subject");
            return None;
        };
        let subject = self.lower_ident(subject);

        let mut traits = Vec::new();
        for trait_ref in bound.traits() {
            if let Some(trait_ref) = self.lower_trait_ref(trait_ref) {
                traits.push(trait_ref);
            }
        }

        Some(TypeBound {
            subject,
            traits: traits.into_boxed_slice(),
        })
    }

    fn lower_trait_ref(&mut self, trait_ref: ast::TraitRef) -> Option<TraitRef> {
        let Some(name) = trait_ref.name() else {
            self.error(&trait_ref, "trait reference is missing its name");
            return None;
        };

        Some(TraitRef {
            name: self.lower_ident(name),
        })
    }

    fn lower_type_annotation(&mut self, annotation: ast::TypeAnnotation) -> TypeAnnotationId {
        let ptr = self.ptr_of(&annotation);
        let lowered = match annotation {
            ast::TypeAnnotation::NamedTypeAnnotation(named) => self.lower_named_type(named),
            ast::TypeAnnotation::FuncTypeAnnotation(func) => self.lower_func_type(func),
        };

        let id = self.ctx.alloc_annotation(lowered);
        self.source_map.bind_annotation(id, self.source_id, ptr);
        id
    }

    fn lower_type_annotation_opt(
        &mut self,
        annotation: Option<ast::TypeAnnotation>,
    ) -> TypeAnnotationId {
        match annotation {
            Some(annotation) => self.lower_type_annotation(annotation),
            None => self.ctx.alloc_annotation(TypeAnnotation::Missing),
        }
    }

    fn lower_named_type(&mut self, annotation: ast::NamedTypeAnnotation) -> TypeAnnotation {
        let Some(name) = annotation.name() else {
            self.error(&annotation, "type is missing its name");
            return TypeAnnotation::Missing;
        };
        let name = self.lower_ident(name);

        let mut args = Vec::new();
        for arg in annotation.args() {
            args.push(self.lower_type_annotation(arg));
        }

        TypeAnnotation::Named {
            name,
            args: args.into_boxed_slice(),
        }
    }

    fn lower_func_type(&mut self, annotation: ast::FuncTypeAnnotation) -> TypeAnnotation {
        let mut params = Vec::new();
        if let Some(param_list) = annotation.params() {
            for param in param_list.params() {
                params.push(self.lower_type_annotation(param));
            }
        }

        let ret = self.lower_type_annotation_opt(annotation.result());

        TypeAnnotation::Func {
            params: params.into_boxed_slice(),
            ret,
        }
    }

    pub fn lower_expr(&mut self, expr: ast::Expr) -> ExprId {
        let ptr = self.ptr_of(&expr);
        let lowered = match expr {
            ast::Expr::IdentExpr(ident_expr) => self.lower_ident_expr(ident_expr),
            ast::Expr::CallExpr(call_expr) => self.lower_call_expr(call_expr),
            ast::Expr::FieldAccessExpr(field_access_expr) => {
                self.lower_field_access_expr(field_access_expr)
            }
            ast::Expr::StructExpr(struct_expr) => self.lower_struct_expr(struct_expr),
            ast::Expr::ListExpr(list_expr) => self.lower_list_expr(list_expr),
            ast::Expr::BinaryExpr(binary_expr) => self.lower_binary_expr(binary_expr),
            ast::Expr::UnaryExpr(unary_expr) => self.lower_unary_expr(unary_expr),
            ast::Expr::ParenExpr(paren_expr) => return self.lower_paren_expr(paren_expr),
            ast::Expr::Literal(literal) => Expr::Literal(self.lower_literal(literal)),
            ast::Expr::Rel(rel) => Expr::Rel(self.lower_rel(rel)),
        };

        let id = self.ctx.alloc_expr(lowered);
        self.source_map.bind_expr(id, self.source_id, ptr);
        id
    }

    fn lower_ident_expr(&mut self, expr: ast::IdentExpr) -> Expr {
        let Some(name) = expr.name() else {
            self.error(&expr, "identifier expression is missing its name");
            return Expr::Missing;
        };
        Expr::Ident {
            value: self.lower_ident(name),
        }
    }

    fn lower_binary_expr(&mut self, expr: ast::BinaryExpr) -> Expr {
        let Some(lhs) = expr.lhs() else {
            self.error(&expr, "binary expression is missing its left operand");
            return Expr::Missing;
        };

        let Some(op) = expr.op() else {
            self.error(&expr, "binary expression is missing its operator");
            return Expr::Missing;
        };

        let Some(rhs) = expr.rhs() else {
            self.error(&expr, "binary expression is missing its right operand");
            return Expr::Missing;
        };

        let lhs = self.lower_expr(lhs);
        let rhs = self.lower_expr(rhs);
        Expr::Call {
            op: lower_bin_op(op),
            args: Box::new([lhs, rhs]),
        }
    }

    fn lower_unary_expr(&mut self, expr: ast::UnaryExpr) -> Expr {
        let Some(op) = expr.op() else {
            self.error(&expr, "unary expression is missing its operator");
            return Expr::Missing;
        };

        let Some(operand) = expr.expr() else {
            self.error(&expr, "unary expression is missing its operand");
            return Expr::Missing;
        };

        let operand = self.lower_expr(operand);
        Expr::Call {
            op: lower_unary_op(op),
            args: Box::new([operand]),
        }
    }

    fn lower_paren_expr(&mut self, expr: ast::ParenExpr) -> ExprId {
        let Some(inner) = expr.expr() else {
            self.error(
                &expr,
                "parenthesized expression is missing its inner expression",
            );
            return self.ctx.alloc_expr(Expr::Missing);
        };

        self.lower_expr(inner)
    }

    fn lower_call_expr(&mut self, expr: ast::CallExpr) -> Expr {
        let Some(callee) = expr.callee() else {
            self.error(&expr, "call is missing its callee");
            return Expr::Missing;
        };

        if let ast::Expr::FieldAccessExpr(field_access) = callee {
            let Some(receiver) = field_access.base() else {
                self.error(&field_access, "method call is missing its receiver");
                return Expr::Missing;
            };
            let Some(method) = field_access.field() else {
                self.error(&field_access, "method call is missing its method");
                return Expr::Missing;
            };
            let receiver = self.lower_expr(receiver);
            let method = self.lower_ident(method);
            let args = self.lower_args(expr.args());
            return Expr::MethodCall {
                receiver,
                method,
                args,
            };
        }

        let callee = self.lower_expr(callee);
        let args = self.lower_args(expr.args());
        Expr::FuncCall { callee, args }
    }

    fn lower_args(&mut self, args: Option<ast::ArgList>) -> Box<[ExprId]> {
        let mut lowered = Vec::new();
        if let Some(list) = args {
            for arg in list.args() {
                lowered.push(self.lower_expr(arg));
            }
        }
        lowered.into_boxed_slice()
    }

    fn lower_field_access_expr(&mut self, expr: ast::FieldAccessExpr) -> Expr {
        let Some(base) = expr.base() else {
            self.error(&expr, "field access is missing its base");
            return Expr::Missing;
        };

        let Some(field) = expr.field() else {
            self.error(&expr, "field access is missing its field");
            return Expr::Missing;
        };

        let base = self.lower_expr(base);
        let field = self.lower_ident(field);
        Expr::FieldAccess { base, field }
    }

    fn lower_struct_expr(&mut self, expr: ast::StructExpr) -> Expr {
        let Some(name) = expr.name() else {
            self.error(&expr, "struct literal is missing its type name");
            return Expr::Missing;
        };

        let name = self.lower_ident(name).symbol;

        let mut fields = Vec::new();
        for field in expr.fields() {
            if let Some(field) = self.lower_struct_field_init(field) {
                fields.push(field);
            }
        }

        Expr::StructInit {
            name,
            fields: fields.into_boxed_slice(),
        }
    }

    fn lower_struct_field_init(&mut self, field: ast::StructFieldInit) -> Option<StructFieldInit> {
        let Some(name) = field.name() else {
            self.error(&field, "struct field initializer is missing its name");
            return None;
        };
        let Some(value) = field.value() else {
            self.error(&field, "struct field initializer is missing its value");
            return None;
        };

        Some(StructFieldInit {
            name: self.lower_ident(name),
            value: self.lower_expr(value),
        })
    }

    fn lower_list_expr(&mut self, expr: ast::ListExpr) -> Expr {
        let mut elements = Vec::new();
        for element in expr.elements() {
            elements.push(self.lower_expr(element));
        }
        Expr::ListInit {
            elements: elements.into_boxed_slice(),
        }
    }

    fn lower_ident(&mut self, ident: ast::Ident) -> Ident {
        let name = self.interner.intern(&ident.text().unwrap_or_default());
        Ident { symbol: name }
    }

    pub fn lower_rel(&mut self, rel: ast::Rel) -> RelId {
        let ptr = self.ptr_of(&rel);
        let lowered = match rel {
            ast::Rel::FromExpr(from_expr) => self.lower_from_rel(from_expr),
            ast::Rel::SelectExpr(select_expr) => self.lower_select_rel(select_expr),
            ast::Rel::WhereExpr(where_expr) => self.lower_where_rel(where_expr),
            ast::Rel::DistinctExpr(distinct_expr) => self.lower_distinct_rel(distinct_expr),
            ast::Rel::DropExpr(drop_expr) => self.lower_drop_rel(drop_expr),
            ast::Rel::RenameExpr(rename_expr) => self.lower_rename_rel(rename_expr),
            ast::Rel::ExtendExpr(extend_expr) => self.lower_extend_rel(extend_expr),
            ast::Rel::JoinExpr(join_expr) => self.lower_join_rel(join_expr),
            ast::Rel::SetExpr(set_expr) => self.lower_set_rel(set_expr),
            ast::Rel::LimitExpr(limit_expr) => self.lower_limit_rel(limit_expr),
            ast::Rel::AliasExpr(alias_expr) => self.lower_alias_rel(alias_expr),
            ast::Rel::AggregateExpr(aggregate_expr) => self.lower_aggregate_rel(aggregate_expr),
        };

        let id = self.ctx.alloc_rel(lowered);
        self.source_map.bind_rel(id, self.source_id, ptr);
        id
    }

    fn lower_from_rel(&mut self, expr: ast::FromExpr) -> Rel {
        let Some(relation) = expr.relation() else {
            self.error(&expr, "`from` is missing its relation");
            return Rel::Missing;
        };

        let relation = self.lower_ident(relation);
        let alias = expr.alias().map(|alias| self.lower_ident(alias));
        Rel::From { relation, alias }
    }

    fn lower_join_rel(&mut self, expr: ast::JoinExpr) -> Rel {
        let Some(input) = expr.input() else {
            self.error(&expr, "`join` is missing its input relation");
            return Rel::Missing;
        };
        let Some(relation) = expr.relation() else {
            self.error(&expr, "`join` is missing its relation");
            return Rel::Missing;
        };
        let left = self.lower_rel_input(input);
        let right = self.lower_joined_rel(relation, expr.alias());
        let Some(condition) = self.lower_join_condition(&expr) else {
            return Rel::Missing;
        };
        Rel::Join {
            left,
            right,
            kind: lower_join_kind(expr.kind()),
            condition,
        }
    }

    fn lower_joined_rel(&mut self, relation: ast::Ident, alias: Option<ast::Ident>) -> RelId {
        let ptr = self.ptr_of(&relation);
        let relation = self.lower_ident(relation);
        let alias = alias.map(|alias| self.lower_ident(alias));
        let id = self.ctx.alloc_rel(Rel::From { relation, alias });
        self.source_map.bind_rel(id, self.source_id, ptr);
        id
    }

    fn lower_join_condition(&mut self, expr: &ast::JoinExpr) -> Option<JoinCondition> {
        if let Some(using) = expr.using() {
            let columns: Box<[Ident]> = using
                .columns()
                .map(|column| self.lower_ident(column))
                .collect();
            if columns.is_empty() {
                self.error(&using, "`using` needs at least one column");
                return None;
            }
            return Some(JoinCondition::Using(columns));
        }

        let Some(on) = expr.on() else {
            self.error(expr, "`join` is missing its `on` or `using` clause");
            return None;
        };
        let Some(condition) = on.condition() else {
            self.error(&on, "`on` is missing its condition");
            return None;
        };
        Some(JoinCondition::On(self.lower_expr(condition)))
    }

    fn lower_select_rel(&mut self, expr: ast::SelectExpr) -> Rel {
        let Some(input) = expr.input() else {
            self.error(&expr, "`select` is missing its input relation");
            return Rel::Missing;
        };

        let input = self.lower_rel_input(input);
        let mut items = Vec::new();
        for item in expr.items() {
            if let Some(item) = self.lower_select_item(item) {
                items.push(item);
            }
        }

        Rel::Select {
            input,
            items: items.into_boxed_slice(),
        }
    }

    fn lower_select_item(&mut self, item: ast::SelectItem) -> Option<SelectItem> {
        let Some(expr) = item.expr() else {
            self.error(&item, "select item is missing its expression");
            return None;
        };

        let expr = self.lower_expr(expr);
        let alias = item.alias().map(|alias| self.lower_ident(alias));
        Some(SelectItem { expr, alias })
    }

    fn lower_where_rel(&mut self, expr: ast::WhereExpr) -> Rel {
        let Some(input) = expr.input() else {
            self.error(&expr, "`where` is missing its input relation");
            return Rel::Missing;
        };
        let Some(predicate) = expr.predicate() else {
            self.error(&expr, "`where` is missing its predicate");
            return Rel::Missing;
        };

        let input = self.lower_rel_input(input);
        let predicate = self.lower_expr(predicate);
        Rel::Where { input, predicate }
    }

    fn lower_distinct_rel(&mut self, expr: ast::DistinctExpr) -> Rel {
        let Some(input) = expr.input() else {
            self.error(&expr, "`distinct` is missing its input relation");
            return Rel::Missing;
        };

        Rel::Distinct {
            input: self.lower_rel_input(input),
        }
    }

    fn lower_drop_rel(&mut self, expr: ast::DropExpr) -> Rel {
        let Some(input) = expr.input() else {
            self.error(&expr, "`drop` is missing its input relation");
            return Rel::Missing;
        };

        let input = self.lower_rel_input(input);
        let mut items = Vec::new();
        for column in expr.columns() {
            items.push(self.lower_ident(column));
        }

        Rel::Drop {
            input,
            items: items.into_boxed_slice(),
        }
    }

    fn lower_rename_rel(&mut self, expr: ast::RenameExpr) -> Rel {
        let Some(input) = expr.input() else {
            self.error(&expr, "`rename` is missing its input relation");
            return Rel::Missing;
        };

        let input = self.lower_rel_input(input);
        let mut items = Vec::new();
        for item in expr.items() {
            if let Some(item) = self.lower_rename_item(item) {
                items.push(item);
            }
        }

        Rel::Rename {
            input,
            items: items.into_boxed_slice(),
        }
    }

    fn lower_rename_item(&mut self, item: ast::RenameItem) -> Option<RenameItem> {
        let Some(from) = item.from() else {
            self.error(&item, "rename item is missing its `from` name");
            return None;
        };
        let Some(to) = item.to() else {
            self.error(&item, "rename item is missing its `to` name");
            return None;
        };

        let qualifier = item.qualifier().map(|alias| self.lower_ident(alias));
        Some(RenameItem {
            qualifier,
            from: self.lower_ident(from),
            to: self.lower_ident(to),
        })
    }

    fn lower_extend_rel(&mut self, expr: ast::ExtendExpr) -> Rel {
        let Some(input) = expr.input() else {
            self.error(&expr, "`extend` is missing its input relation");
            return Rel::Missing;
        };

        let input = self.lower_rel_input(input);
        let mut items = Vec::new();
        for item in expr.items() {
            if let Some(item) = self.lower_select_item(item) {
                items.push(item);
            }
        }

        Rel::Extend {
            input,
            items: items.into_boxed_slice(),
        }
    }

    fn lower_set_rel(&mut self, expr: ast::SetExpr) -> Rel {
        let Some(input) = expr.input() else {
            self.error(&expr, "`set` is missing its input relation");
            return Rel::Missing;
        };

        let input = self.lower_rel_input(input);
        let mut items = Vec::new();
        for item in expr.items() {
            let (Some(column), Some(value)) = (item.column(), item.value()) else {
                self.error(&item, "set item is incomplete");
                continue;
            };
            items.push(SetItem {
                column: self.lower_ident(column),
                value: self.lower_expr(value),
            });
        }

        Rel::Set {
            input,
            items: items.into_boxed_slice(),
        }
    }

    fn lower_aggregate_rel(&mut self, expr: ast::AggregateExpr) -> Rel {
        let Some(input) = expr.input() else {
            self.error(&expr, "`aggregate` is missing its input relation");
            return Rel::Missing;
        };

        let input = self.lower_rel_input(input);
        let mut items = Vec::new();
        for item in expr.items() {
            let Some(item_expr) = item.expr() else {
                self.error(&item, "aggregate item is missing its expression");
                continue;
            };
            items.push(AggregateItem {
                expr: self.lower_expr(item_expr),
                alias: item.alias().map(|alias| self.lower_ident(alias)),
            });
        }

        let mut groups = Vec::new();
        let keys: Vec<ast::GroupByItem> = expr
            .group_by()
            .map(|list| list.items().collect())
            .unwrap_or_default();
        for key in keys {
            let Some(column) = key.column() else {
                self.error(&key, "group by key is missing its column");
                continue;
            };
            groups.push(GroupKey {
                qualifier: key.qualifier().map(|qualifier| self.lower_ident(qualifier)),
                column: self.lower_ident(column),
                alias: key.alias().map(|alias| self.lower_ident(alias)),
            });
        }

        Rel::Aggregate {
            input,
            items: items.into_boxed_slice(),
            groups: groups.into_boxed_slice(),
        }
    }

    fn lower_limit_rel(&mut self, expr: ast::LimitExpr) -> Rel {
        let Some(input) = expr.input() else {
            self.error(&expr, "`limit` is missing its input relation");
            return Rel::Missing;
        };
        let Some(count) = expr.count() else {
            self.error(&expr, "`limit` is missing its row count");
            return Rel::Missing;
        };

        let input = self.lower_rel_input(input);
        Rel::Limit {
            input,
            count: self.lower_expr(count),
            offset: expr.offset().map(|offset| self.lower_expr(offset)),
        }
    }

    fn lower_alias_rel(&mut self, expr: ast::AliasExpr) -> Rel {
        let Some(input) = expr.input() else {
            self.error(&expr, "`as` is missing its input relation");
            return Rel::Missing;
        };
        let Some(alias) = expr.alias() else {
            self.error(&expr, "`as` is missing its alias");
            return Rel::Missing;
        };

        Rel::Alias {
            input: self.lower_rel_input(input),
            alias: self.lower_ident(alias),
        }
    }

    fn lower_rel_input(&mut self, input: ast::Expr) -> RelId {
        match input {
            ast::Expr::Rel(rel) => self.lower_rel(rel),
            other => {
                self.error(&other, "expected a relation as the pipe input");
                self.ctx.alloc_rel(Rel::Missing)
            }
        }
    }

    fn lower_literal(&mut self, literal: ast::Literal) -> Literal {
        match literal {
            ast::Literal::BoolLiteral(lit) => {
                let Some(value) = lit.value() else {
                    self.error(&lit, "bool literal is missing its value");
                    return Literal::Missing;
                };
                Literal::Bool { value }
            }
            ast::Literal::IntLiteral(lit) => {
                let Some(value) = lit.value() else {
                    self.error(&lit, "integer literal is missing its value");
                    return Literal::Missing;
                };
                Literal::Int {
                    value: Int::from(value),
                }
            }
            ast::Literal::FloatLiteral(lit) => {
                let Some(value) = lit.value() else {
                    self.error(&lit, "float literal is missing its value");
                    return Literal::Missing;
                };
                Literal::Float {
                    value: Float::from(value),
                }
            }
            ast::Literal::StringLiteral(lit) => {
                let Some(value) = lit.value() else {
                    self.error(&lit, "string literal is missing its value");
                    return Literal::Missing;
                };
                Literal::String {
                    value: self.interner.intern(&value),
                }
            }
        }
    }

    fn error(&mut self, node: &impl ast::AstNode, message: &str) {
        let range = node.syntax().text_range();
        let span = Span {
            source_id: self.source_id,
            range,
        };

        let diagnostic = DiagnosticBuilder::error(span, message).build();
        self.diagnostics.emit(diagnostic);
    }

    fn ptr_of(&self, node: &impl ast::AstNode) -> SyntaxNodePtr {
        SyntaxNodePtr::new(node.syntax())
    }
}

fn lower_join_kind(kind: Option<ast::JoinKind>) -> JoinKind {
    match kind {
        Some(ast::JoinKind::Left) => JoinKind::Left,
        Some(ast::JoinKind::Right) => JoinKind::Right,
        Some(ast::JoinKind::Full) => JoinKind::Full,
        Some(ast::JoinKind::Inner) | None => JoinKind::Inner,
    }
}

fn lower_bin_op(op: ast::BinOp) -> Op {
    match op {
        ast::BinOp::Add => Op::Add,
        ast::BinOp::Sub => Op::Sub,
        ast::BinOp::Mul => Op::Mul,
        ast::BinOp::Div => Op::Div,
        ast::BinOp::Pow => Op::Pow,
        ast::BinOp::And => Op::And,
        ast::BinOp::Or => Op::Or,
        ast::BinOp::In => Op::In,
        ast::BinOp::NotIn => Op::NotIn,
        ast::BinOp::Eq => Op::Eq,
        ast::BinOp::Neq => Op::Neq,
        ast::BinOp::Lt => Op::Lt,
        ast::BinOp::Lte => Op::Lte,
        ast::BinOp::Gt => Op::Gt,
        ast::BinOp::Gte => Op::Gte,
        ast::BinOp::ShiftLeft => Op::ShiftLeft,
        ast::BinOp::ShiftRight => Op::ShiftRight,
    }
}

fn lower_unary_op(op: ast::UnaryOp) -> Op {
    match op {
        ast::UnaryOp::Pos => Op::UnaryPos,
        ast::UnaryOp::Neg => Op::UnaryNeg,
        ast::UnaryOp::Not => Op::UnaryNot,
    }
}

fn lower_mutability(mutability: ast::Mutability) -> Mutability {
    match mutability {
        ast::Mutability::Mutable => Mutability::Mutable,
        ast::Mutability::Immutable => Mutability::Immutable,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::printer::dump;
    use expect_test::{Expect, expect};
    use yuzu_ast::ast::{AstNode, Root as AstRoot};
    use yuzu_diagnostics::source_map::SourceMap;
    use yuzu_lexer::lexer::{Lexer, Token};

    fn check(input: &str, expected: Expect) {
        let mut hir = HirCtx::new();
        let mut interner = StringInterner::new();
        let mut diagnostics = DiagnosticsEngine::new();
        let mut sources = SourceMap::new();
        let source_id = sources.add("test".to_string(), input.to_string());

        let tokens: Vec<Token> = Lexer::new(input).collect();
        let syntax = yuzu_parser::parse(&tokens, &mut diagnostics, source_id);
        let ast_root = AstRoot::cast(syntax).expect("root node");

        let lowerer = HirLowerer {
            ctx: &mut hir,
            interner: &mut interner,
            diagnostics: &mut diagnostics,
            source_map: HirSourceMap::default(),
            source_id,
        };
        let (root, _source_map) = lowerer.lower(ast_root);

        expected.assert_eq(&dump(&hir, &interner, &root));
    }

    #[test]
    fn struct_stmt() {
        check(
            "struct Point { x: int, y: int }",
            expect![[r#"
            Struct "Point"
              field "x":
                Named "int"
              field "y":
                Named "int"
        "#]],
        );
    }

    #[test]
    fn table_stmt_named() {
        check(
            "table Employees = Employee",
            expect![[r#"
            Table "Employees" row "Employee"
        "#]],
        );
    }

    #[test]
    fn inline_table_stmt() {
        check(
            "table T = { x: int }",
            expect![[r#"
            InlineTable "T"
              field "x":
                Named "int"
        "#]],
        );
    }

    #[test]
    fn func_stmt() {
        check(
            "fn add(x: int, y: int) -> int { return x }",
            expect![[r#"
            Func "add"
              param "x":
                Named "int"
              param "y":
                Named "int"
              ret:
                Named "int"
              body:
                Block
                  Return
                    Ident "x"
        "#]],
        );
    }

    #[test]
    fn generic_func_with_bound() {
        check(
            "fn id[T](x: T) -> T where T: Eq { return x }",
            expect![[r#"
            Func "id"
              type_param "T"
              param "x":
                Named "T"
              bound "T"
                trait "Eq"
              ret:
                Named "T"
              body:
                Block
                  Return
                    Ident "x"
        "#]],
        );
    }

    #[test]
    fn impl_stmt() {
        check(
            "impl Point { fn x(self) { return self } }",
            expect![[r#"
                Impl "Point"
                  Func "x"
                    param "self":
                      self
                    ret:
                      Missing
                    body:
                      Block
                        Return
                          Ident "self"
            "#]],
        );
    }

    #[test]
    fn impl_trait_for_stmt() {
        check(
            "impl Show for Point { fn show(self) { return self } }",
            expect![[r#"
                Impl "Show" for "Point"
                  Func "show"
                    param "self":
                      self
                    ret:
                      Missing
                    body:
                      Block
                        Return
                          Ident "self"
            "#]],
        );
    }

    #[test]
    fn self_receiver_param() {
        check(
            "impl Counter { fn value(self) -> int { return 0 } }",
            expect![[r#"
                Impl "Counter"
                  Func "value"
                    param "self":
                      self
                    ret:
                      Named "int"
                    body:
                      Block
                        Return
                          Literal Int 0u64
            "#]],
        );
    }

    #[test]
    fn trait_stmt() {
        check(
            "trait Show { fn show(self) -> str }",
            expect![[r#"
                Trait "Show"
                  Func "show"
                    param "self":
                      self
                    ret:
                      Named "str"
                    body:
            "#]],
        );
    }

    #[test]
    fn trait_method_without_body() {
        check(
            "trait Greet { fn hello(self) }",
            expect![[r#"
            Trait "Greet"
              Func "hello"
                param "self":
                  self
                ret:
                  Missing
                body:
        "#]],
        );
    }

    #[test]
    fn let_stmt() {
        check(
            "let x: int = 1 + 2",
            expect![[r#"
                Let "x" Immutable
                  type:
                    Named "int"
                  expr:
                    Call Add
                      Literal Int 1u64
                      Literal Int 2u64
            "#]],
        );
    }

    #[test]
    fn let_mut_stmt() {
        check(
            "let mut x = -1",
            expect![[r#"
                Let "x" Mutable
                  expr:
                    Call UnaryNeg
                      Literal Int 1u64
            "#]],
        );
    }

    #[test]
    fn assign_stmt() {
        check(
            "p.x = 5",
            expect![[r#"
                Assign
                  FieldAccess "x"
                    Ident "p"
                  Literal Int 5u64
            "#]],
        );
    }

    #[test]
    fn return_stmt() {
        check(
            "return f(x)",
            expect![[r#"
                Return
                  FuncCall
                    Ident "f"
                    Ident "x"
            "#]],
        );
    }

    #[test]
    fn expr_stmt() {
        check(
            "[1, 2, 3]",
            expect![[r#"
                Expr
                  ListInit
                    Literal Int 1u64
                    Literal Int 2u64
                    Literal Int 3u64
            "#]],
        );
    }

    #[test]
    fn query_stmt() {
        check(
            "from employees |> where active |> select name",
            expect![[r#"
            Expr
              Rel
                Select
                  Where
                    From "employees"
                    Ident "active"
                  item:
                    Ident "name"
        "#]],
        );
    }

    // --- Statement arms not covered above ---

    #[test]
    fn func_without_return_annotation() {
        check(
            "fn f() { return 1 }",
            expect![[r#"
            Func "f"
              ret:
                Missing
              body:
                Block
                  Return
                    Literal Int 1u64
        "#]],
        );
    }

    #[test]
    fn let_immutable_without_annotation() {
        check(
            "let x = 1",
            expect![[r#"
            Let "x" Immutable
              expr:
                Literal Int 1u64
        "#]],
        );
    }

    #[test]
    fn let_mut_with_annotation() {
        check(
            "let mut x: int = 1",
            expect![[r#"
            Let "x" Mutable
              type:
                Named "int"
              expr:
                Literal Int 1u64
        "#]],
        );
    }

    #[test]
    fn return_without_value() {
        check(
            "fn f() { return }",
            expect![[r#"
            Func "f"
              ret:
                Missing
              body:
                Block
                  Return
        "#]],
        );
    }

    // --- Expression arms ---

    #[test]
    fn method_call_expr() {
        check(
            "a.b(c)",
            expect![[r#"
            Expr
              MethodCall "b"
                Ident "a"
                Ident "c"
        "#]],
        );
    }

    #[test]
    fn struct_init_expr() {
        check(
            "P { x: 1, y: 2 }",
            expect![[r#"
            Expr
              StructInit "P"
                field "x":
                  Literal Int 1u64
                field "y":
                  Literal Int 2u64
        "#]],
        );
    }

    #[test]
    fn field_access_expr() {
        check(
            "a.b.c",
            expect![[r#"
            Expr
              FieldAccess "c"
                FieldAccess "b"
                  Ident "a"
        "#]],
        );
    }

    #[test]
    fn paren_expr_is_unwrapped() {
        check(
            "(1 + 2)",
            expect![[r#"
            Expr
              Call Add
                Literal Int 1u64
                Literal Int 2u64
        "#]],
        );
    }

    // --- Binary operators (every arm of lower_bin_op) ---

    #[test]
    fn binary_arithmetic_operators() {
        check(
            "[1 + 2, 1 - 2, 1 * 2, 1 / 2, 2 ** 3]",
            expect![[r#"
                Expr
                  ListInit
                    Call Add
                      Literal Int 1u64
                      Literal Int 2u64
                    Call Sub
                      Literal Int 1u64
                      Literal Int 2u64
                    Call Mul
                      Literal Int 1u64
                      Literal Int 2u64
                    Call Div
                      Literal Int 1u64
                      Literal Int 2u64
                    Call Pow
                      Literal Int 2u64
                      Literal Int 3u64
            "#]],
        );
    }

    #[test]
    fn binary_comparison_operators() {
        check(
            "[a == b, a != b, a < b, a <= b, a > b, a >= b]",
            expect![[r#"
                Expr
                  ListInit
                    Call Eq
                      Ident "a"
                      Ident "b"
                    Call Neq
                      Ident "a"
                      Ident "b"
                    Call Lt
                      Ident "a"
                      Ident "b"
                    Call Lte
                      Ident "a"
                      Ident "b"
                    Call Gt
                      Ident "a"
                      Ident "b"
                    Call Gte
                      Ident "a"
                      Ident "b"
            "#]],
        );
    }

    #[test]
    fn binary_logical_operators() {
        check(
            "[a and b, a or b]",
            expect![[r#"
            Expr
              ListInit
                Call And
                  Ident "a"
                  Ident "b"
                Call Or
                  Ident "a"
                  Ident "b"
        "#]],
        );
    }

    #[test]
    fn binary_membership_operators() {
        check(
            "[a in b, a not in b]",
            expect![[r#"
            Expr
              ListInit
                Call In
                  Ident "a"
                  Ident "b"
                Call NotIn
                  Ident "a"
                  Ident "b"
        "#]],
        );
    }

    #[test]
    fn binary_shift_operators() {
        check(
            "[a << b, a >> b]",
            expect![[r#"
            Expr
              ListInit
                Call ShiftLeft
                  Ident "a"
                  Ident "b"
                Call ShiftRight
                  Ident "a"
                  Ident "b"
        "#]],
        );
    }

    // --- Unary operators (every arm of lower_unary_op) ---

    #[test]
    fn unary_pos_operator() {
        check(
            "+1",
            expect![[r#"
            Expr
              Call UnaryPos
                Literal Int 1u64
        "#]],
        );
    }

    #[test]
    fn unary_neg_operator() {
        check(
            "-1",
            expect![[r#"
            Expr
              Call UnaryNeg
                Literal Int 1u64
        "#]],
        );
    }

    #[test]
    fn unary_not_operator() {
        check(
            "not a",
            expect![[r#"
            Expr
              Call UnaryNot
                Ident "a"
        "#]],
        );
    }

    // --- Literal kinds ---

    #[test]
    fn int_literal() {
        check(
            "42",
            expect![[r#"
            Expr
              Literal Int 42u64
        "#]],
        );
    }

    #[test]
    fn float_literal() {
        check(
            "3.14",
            expect![[r#"
            Expr
              Literal Float 3.14f64
        "#]],
        );
    }

    #[test]
    fn bool_literal() {
        check(
            "true",
            expect![[r#"
            Expr
              Literal Bool true
        "#]],
        );
    }

    #[test]
    fn string_literal() {
        check(
            "\"hello\"",
            expect![[r#"
            Expr
              Literal String "hello"
        "#]],
        );
    }

    // --- Type annotations ---

    #[test]
    fn named_type_with_args() {
        check(
            "let v: List[int] = xs",
            expect![[r#"
            Let "v" Immutable
              type:
                Named "List"
                  Named "int"
              expr:
                Ident "xs"
        "#]],
        );
    }

    #[test]
    fn function_type_annotation() {
        check(
            "let f: (int) -> int = g",
            expect![[r#"
            Let "f" Immutable
              type:
                Func
                  Named "int"
                  ret:
                    Named "int"
              expr:
                Ident "g"
        "#]],
        );
    }

    // --- Relational operators (every arm of lower_rel) ---

    #[test]
    fn rel_from_with_alias() {
        check(
            "from employees as e |> select e",
            expect![[r#"
            Expr
              Rel
                Select
                  From "employees" as "e"
                  item:
                    Ident "e"
        "#]],
        );
    }

    #[test]
    fn rel_distinct() {
        check(
            "from t |> distinct",
            expect![[r#"
            Expr
              Rel
                Distinct
                  From "t"
        "#]],
        );
    }

    #[test]
    fn rel_drop() {
        check(
            "from t |> drop a, b",
            expect![[r#"
            Expr
              Rel
                Drop
                  From "t"
                  column "a"
                  column "b"
        "#]],
        );
    }

    #[test]
    fn rel_rename() {
        check(
            "from t |> rename a as b",
            expect![[r#"
            Expr
              Rel
                Rename
                  From "t"
                  rename "a" -> "b"
        "#]],
        );
    }

    #[test]
    fn rel_extend() {
        check(
            "from t |> extend a",
            expect![[r#"
            Expr
              Rel
                Extend
                  From "t"
                  item:
                    Ident "a"
        "#]],
        );
    }

    #[test]
    fn rel_rename_qualified() {
        check(
            "from t |> rename e.id as eid, b as c",
            expect![[r#"
            Expr
              Rel
                Rename
                  From "t"
                  rename "e"."id" -> "eid"
                  rename "b" -> "c"
        "#]],
        );
    }

    #[test]
    fn rel_set() {
        check(
            "from t |> set a = a + 1, b = 2",
            expect![[r#"
            Expr
              Rel
                Set
                  From "t"
                  set "a":
                    Call Add
                      Ident "a"
                      Literal Int 1u64
                  set "b":
                    Literal Int 2u64
        "#]],
        );
    }

    #[test]
    fn rel_limit() {
        check(
            "from t |> limit 10 offset 5",
            expect![[r#"
            Expr
              Rel
                Limit
                  From "t"
                  count:
                    Literal Int 10u64
                  offset:
                    Literal Int 5u64
        "#]],
        );
    }

    #[test]
    fn external_func_stmts() {
        check(
            "external fn upper(s: str) -> str\nexternal agg fn median(x: int64) -> float64",
            expect![[r#"
                External Func "upper"
                  param "s":
                    Named "str"
                  ret:
                    Named "str"
                  body:
                External Agg Func "median"
                  param "x":
                    Named "int64"
                  ret:
                    Named "float64"
                  body:
            "#]],
        );
    }

    #[test]
    fn agg_func_stmt() {
        check(
            "agg fn spread(x: int64) -> int64 { return sum(x) }",
            expect![[r#"
                Agg Func "spread"
                  param "x":
                    Named "int64"
                  ret:
                    Named "int64"
                  body:
                    Block
                      Return
                        FuncCall
                          Ident "sum"
                          Ident "x"
            "#]],
        );
    }

    #[test]
    fn rel_aggregate() {
        check(
            "from t |> aggregate sum(a) as s, count() group by b, e.c as k",
            expect![[r#"
                Expr
                  Rel
                    Aggregate
                      From "t"
                      item as "s":
                        FuncCall
                          Ident "sum"
                          Ident "a"
                      item:
                        FuncCall
                          Ident "count"
                      group by "b"
                      group by "e"."c" as "k"
            "#]],
        );
    }

    #[test]
    fn rel_aggregate_without_group_by() {
        check(
            "from t |> aggregate count()",
            expect![[r#"
            Expr
              Rel
                Aggregate
                  From "t"
                  item:
                    FuncCall
                      Ident "count"
        "#]],
        );
    }

    #[test]
    fn rel_alias() {
        check(
            "from t |> as u",
            expect![[r#"
            Expr
              Rel
                Alias "u"
                  From "t"
        "#]],
        );
    }

    #[test]
    fn rel_join_on() {
        check(
            "from t |> join u as d on a == d.b",
            expect![[r#"
            Expr
              Rel
                Join inner
                  From "t"
                  From "u" as "d"
                  on:
                    Call Eq
                      Ident "a"
                      FieldAccess "b"
                        Ident "d"
        "#]],
        );
    }

    #[test]
    fn rel_join_using() {
        check(
            "from t |> left join u using (a, b)",
            expect![[r#"
            Expr
              Rel
                Join left
                  From "t"
                  From "u"
                  using "a"
                  using "b"
        "#]],
        );
    }

    #[test]
    fn rel_join_after_a_stage_with_its_own_idents() {
        check(
            "from t |> drop a, b |> join u as d on c == d.e",
            expect![[r#"
            Expr
              Rel
                Join inner
                  Drop
                    From "t"
                    column "a"
                    column "b"
                  From "u" as "d"
                  on:
                    Call Eq
                      Ident "c"
                      FieldAccess "e"
                        Ident "d"
        "#]],
        );
    }

    #[test]
    fn rel_join_chained() {
        check(
            "from t |> left join u on a == b |> join v as x on c == x.d",
            expect![[r#"
                Expr
                  Rel
                    Join inner
                      Join left
                        From "t"
                        From "u"
                        on:
                          Call Eq
                            Ident "a"
                            Ident "b"
                      From "v" as "x"
                      on:
                        Call Eq
                          Ident "c"
                          FieldAccess "d"
                            Ident "x"
            "#]],
        );
    }

    #[test]
    fn rel_join_input_alias_is_not_the_joined_relation() {
        check(
            "from t as e |> join u on a == b",
            expect![[r#"
            Expr
              Rel
                Join inner
                  From "t" as "e"
                  From "u"
                  on:
                    Call Eq
                      Ident "a"
                      Ident "b"
        "#]],
        );
    }

    #[test]
    fn rel_join_using_without_columns_lowers_to_missing() {
        check(
            "from t |> join u using ()",
            expect![[r#"
            Expr
              Rel
                Missing
        "#]],
        );
    }

    #[test]
    fn rel_join_without_condition_lowers_to_missing() {
        check(
            "from t |> join u",
            expect![[r#"
            Expr
              Rel
                Missing
        "#]],
        );
    }

    #[test]
    fn rel_select_with_alias() {
        check(
            "from t |> select x as y",
            expect![[r#"
            Expr
              Rel
                Select
                  From "t"
                  item as "y":
                    Ident "x"
        "#]],
        );
    }

    #[test]
    fn rel_chained_all_clauses() {
        check(
            "from t |> where a |> drop b |> rename c as d |> distinct |> extend e |> select f",
            expect![[r#"
                Expr
                  Rel
                    Select
                      Extend
                        Distinct
                          Rename
                            Drop
                              Where
                                From "t"
                                Ident "a"
                              column "b"
                            rename "c" -> "d"
                        item:
                          Ident "e"
                      item:
                        Ident "f"
            "#]],
        );
    }

    // --- Error recovery / Missing nodes ---

    #[test]
    fn missing_assign_value() {
        check(
            "x =",
            expect![[r#"
            Missing
        "#]],
        );
    }

    #[test]
    fn missing_let_expression() {
        check(
            "let x =",
            expect![[r#"
            Missing
        "#]],
        );
    }

    #[test]
    fn missing_binary_rhs() {
        check(
            "1 +",
            expect![[r#"
            Expr
              Missing
        "#]],
        );
    }

    #[test]
    fn missing_param_type() {
        check(
            "fn f(x) { return x }",
            expect![[r#"
            Func "f"
              param "x":
                Missing
              ret:
                Missing
              body:
                Block
                  Return
                    Ident "x"
        "#]],
        );
    }
}
