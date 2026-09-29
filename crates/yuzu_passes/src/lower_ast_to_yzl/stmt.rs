use melior::ir::attribute::{
    ArrayAttribute, FlatSymbolRefAttribute, StringAttribute, TypeAttribute,
};
use melior::ir::r#type::FunctionType;
use melior::ir::{
    Attribute, Block, BlockLike, BlockRef, Location, Operation, Region, RegionLike, Type, Value,
};
use text_size::TextRange;
use yuzu_ast::ast::{self, AstNode, Mutability, Visibility};
use yuzu_mlir::attributes::CalleeSource;
use yuzu_mlir::ir::attribute::array::ArrayAttributeExt;
use yuzu_mlir::ir::block::BlockExt;
use yuzu_mlir::ir::operation::{OperationCast, OperationExt, OperationMutExt};
use yuzu_mlir::ods::yzl;
use yuzu_mlir::ops::yzl::YzlOp;
use yuzu_mlir::types::{
    self, ErrorType, ListType, ParamType, RefType, StructType, UnitType, UnresolvedType,
};

use crate::lower_ast_to_yzl::symbols::{
    Binding, BindingKind, Declared, Field, FunctionKind, Lookup, Method, ModulePath, Overload,
    Reference, Row, Target,
};
use crate::lower_ast_to_yzl::{AstToYzl, Locals};

/// Where a `fn` is written, which decides the generics its surroundings
/// supply, whether it needs a body, and the symbol it is built under. A
/// method carries the methods of its trait, which say whether its name is
/// overloaded.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Site<'m, 'c> {
    AtModule,
    InTrait(&'m [Method<'c>]),
    InImpl(&'m [Method<'c>]),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LocalKind {
    Let(Mutability),
    Param,
}

impl Site<'_, '_> {
    fn generics(self) -> &'static [&'static str] {
        match self {
            Site::AtModule => &[],
            Site::InTrait(_) | Site::InImpl(_) => &["Self"],
        }
    }
}

impl<'c> AstToYzl<'c, '_> {
    pub(super) fn convert_stmt<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        locals: &mut Locals<'c, 'a>,
        stmt: &ast::Stmt,
    ) {
        match stmt {
            ast::Stmt::StructStmt(decl) => self.convert_struct(block, decl),
            ast::Stmt::TableStmt(decl) => self.convert_table(block, decl),
            ast::Stmt::FuncStmt(decl) => self.convert_fn(block, decl),
            ast::Stmt::TraitStmt(decl) => self.convert_trait(block, decl),
            ast::Stmt::ImplStmt(decl) => self.convert_impl(block, decl),
            ast::Stmt::LetStmt(decl) => self.convert_let(block, locals, decl),
            ast::Stmt::AssignStmt(stmt) => self.convert_assign(block, locals, stmt),
            ast::Stmt::ReturnStmt(stmt) => self.convert_return(block, locals, stmt),
            ast::Stmt::BlockStmt(stmt) => self.convert_nested_block(block, locals, stmt),
            ast::Stmt::ExprStmt(stmt) => self.convert_expr_stmt(block, locals, stmt),
            ast::Stmt::ImportStmt(stmt) => self.convert_import(stmt),
            ast::Stmt::FromImportStmt(stmt) => self.convert_import(stmt),
            ast::Stmt::ModStmt(stmt) => self.convert_import(stmt),
        }
    }

    fn convert_struct<'a>(&mut self, block: BlockRef<'c, 'a>, decl: &ast::StructStmt) {
        if !self.check_declaration_at_file_level(decl) {
            return;
        }

        let Some(name) = self.read_ident(decl.name()) else {
            self.assert_syntax_error("struct is missing its name");
            return;
        };

        if !self.was_hoisted(decl, name) {
            return;
        }

        let name = self.symbols.symbol_here(name);
        let loc = self.location(decl);
        let fields = self.read_fields(decl.fields());
        self.emit_struct(block, name, fields, decl.visibility(), loc);
    }

    fn convert_table<'a>(&mut self, block: BlockRef<'c, 'a>, decl: &ast::TableStmt) {
        if !self.check_declaration_at_file_level(decl) {
            return;
        }

        let Some(name) = self.read_ident(decl.name()) else {
            self.assert_syntax_error("table is missing its name");
            return;
        };

        if !self.was_hoisted(decl, name) {
            return;
        }

        // An imported struct is held under the module that declared it,
        // whatever an `as` renamed it to here.
        let row = if let Some(written) = decl.struct_name()
            && let Some(struct_name) = self.read_ident(Some(written.clone()))
        {
            if let Some((symbol, target)) = self.symbols.struct_symbol(struct_name) {
                self.record(written.syntax().text_range(), struct_name, target);
                symbol
            } else {
                debug_assert!(
                    self.diagnostics.has_errors(),
                    "`{struct_name}` passed the hoist as a struct and is not one now"
                );
                return;
            }
        } else {
            let symbol = self
                .symbols
                .symbol_here(self.symbols.intern_fmt(format_args!("{name}_row")));

            let loc = self.location(decl);
            let fields = self.read_fields(decl.inline_fields());
            self.emit_struct(block, symbol, fields, decl.visibility(), loc);

            symbol
        };

        let name = self.symbols.symbol_here(name);
        let loc = self.location(decl);

        let mut table: Operation<'c> = yzl::table(
            self.context,
            StringAttribute::new(self.context, name),
            FlatSymbolRefAttribute::new(self.context, row),
            loc,
        )
        .into();

        if decl.visibility() != Visibility::Public {
            table.set_private(self.context);
        }

        block.append_operation(table);
    }

    fn convert_fn<'a>(&mut self, block: BlockRef<'c, 'a>, decl: &ast::FuncStmt) {
        if !self.check_declaration_at_file_level(decl) {
            return;
        }

        if let Some(name) = self.read_ident(decl.name())
            && !self.was_hoisted(decl, name)
        {
            return;
        }

        self.convert_method(block, decl, Site::AtModule);
    }

    fn convert_method<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        decl: &ast::FuncStmt,
        site: Site<'_, 'c>,
    ) {
        let loc = self.location(decl);

        let Some(name) = self.read_ident(decl.name()) else {
            self.assert_syntax_error("function is missing its name");
            return;
        };

        match (decl.is_external(), decl.body().is_some()) {
            (true, true) => self.report(decl, "an external function cannot have a body"),
            (false, false) if !matches!(site, Site::InTrait(_)) => {
                self.report(decl, "function is missing its body");
            }
            _ => {}
        }

        let mut generics: Vec<&'c str> = site.generics().to_vec();
        generics.extend(
            decl.type_params()
                .filter_map(|param| self.read_ident(param.name())),
        );

        let op = self.in_type_params(generics.clone(), |this| {
            this.build_fn(decl, site, name, &generics, loc)
        });

        if let Some(op) = op {
            block.append_operation(op);
        }
    }

    /// The `yzl.fn` for a declaration, with its type parameters in scope.
    /// `None` when a parameter could not be read.
    fn build_fn(
        &mut self,
        decl: &ast::FuncStmt,
        site: Site<'_, 'c>,
        name: &'c str,
        generics: &[&'c str],
        loc: Location<'c>,
    ) -> Option<Operation<'c>> {
        // Trait bounds.
        let (bound_params, bound_traits) = self.read_bounds(decl);

        // Params.
        let mut param_names = Vec::new();
        let mut param_ranges = Vec::new();
        let mut param_types = Vec::new();
        let mut has_error = false;
        for param in decl.params() {
            let Some(name) = self.read_ident(param.name()) else {
                self.assert_syntax_error("parameter is missing its name");
                has_error = true;
                continue;
            };

            let ty = if let Some(ty) = param.ty() {
                self.read_type_annotation(ty)
            } else {
                self.report(&param, "parameter is missing its type");
                has_error = true;
                continue;
            };

            param_names.push(name);
            param_ranges.push(param.syntax().text_range());
            param_types.push(ty);
        }

        // Early exit if there are any parameter errors.
        if has_error {
            return None;
        }

        // A function that declares no result returns unit.
        let result = match decl.result() {
            Some(result) => self.read_type_annotation(result),
            None => UnitType::new(self.context).into(),
        };

        let body = Region::new();
        if let Some(block) = decl.body() {
            let ty = UnresolvedType::new(self.context).into();
            let arguments = vec![(ty, loc); param_names.len()];
            let entry = body.append_block(Block::new(&arguments));

            let mut locals = Locals::new();
            self.in_block(|this| {
                // Make a place for each parameter, with the type of the parameter.
                // Put the argument in the place. The name of the parameter then gives the
                // place, as the name of a `let` does.
                for (index, name) in param_names.iter().enumerate() {
                    let argument = entry
                        .argument(index)
                        .expect("the entry block has one argument per parameter")
                        .into();

                    let place = this.emit_local(
                        entry,
                        name,
                        LocalKind::Param,
                        param_types[index],
                        argument,
                        loc,
                    );

                    this.bind_local(&mut locals, name, param_ranges[index], place);
                }

                this.convert_block(entry, &mut locals, &block);
                this.emit_final_return(entry, decl, name, result, &block);
            });
        }

        let symbol = self.symbol_in(site, name, param_names.len());
        let signature = FunctionType::new(self.context, &param_types, &[result]);
        let mut builder = yzl::FnOperationBuilder::new(self.context, loc)
            .sym_name(StringAttribute::new(self.context, symbol))
            .params(ArrayAttribute::from_strings(self.context, param_names))
            .signature(TypeAttribute::new(signature.into()))
            .body(body);

        if decl.is_agg() {
            builder = builder.is_agg(Attribute::unit(self.context));
        }

        if let Some(external_name) = decl.is_external().then_some(name) {
            builder = builder.external_name(StringAttribute::new(self.context, external_name));
        }

        if !generics.is_empty() {
            builder = builder.type_params(ArrayAttribute::from_strings(self.context, generics));
        }

        if !bound_params.is_empty() {
            builder = builder
                .bound_params(ArrayAttribute::from_strings(self.context, &bound_params))
                .bound_traits(ArrayAttribute::from_symbols(self.context, &bound_traits));
        }

        let mut op: Operation<'c> = builder.build().into();
        if site == Site::AtModule && decl.visibility() != Visibility::Public {
            op.set_private(self.context);
        }

        Some(op)
    }

    fn convert_trait<'a>(&mut self, block: BlockRef<'c, 'a>, decl: &ast::TraitStmt) {
        if !self.check_declaration_at_file_level(decl) {
            return;
        }

        let Some(name) = self.read_ident(decl.name()) else {
            self.assert_syntax_error("trait is missing its name");
            return;
        };

        if !self.was_hoisted(decl, name) {
            return;
        }

        let region = Region::new();
        let body = region.append_block(Block::new(&[]));
        let methods = self.read_methods(decl.methods());
        for method in self.check_methods(decl.methods()) {
            self.convert_method(body, &method, Site::InTrait(&methods));
        }

        let name = self.symbols.symbol_here(name);
        let mut trait_: Operation<'c> =
            yzl::TraitOperationBuilder::new(self.context, self.location(decl))
                .sym_name(StringAttribute::new(self.context, name))
                .body(region)
                .build()
                .into();

        if decl.visibility() != Visibility::Public {
            trait_.set_private(self.context);
        }

        block.append_operation(trait_);
    }

    fn convert_impl<'a>(&mut self, block: BlockRef<'c, 'a>, decl: &ast::ImplStmt) {
        if !self.check_declaration_at_file_level(decl) {
            return;
        }

        let Some(trait_name) = decl
            .trait_()
            .and_then(|trait_ref| self.read_ident(trait_ref.name()))
        else {
            self.assert_syntax_error("`impl` is missing its trait");
            return;
        };

        let Some(target) = self.read_ident(decl.ty()) else {
            self.assert_syntax_error("`impl` is missing its type name");
            return;
        };

        // A method is overloaded when its trait overloads it, so the two
        // agree on its symbol.
        let methods = match self.symbols.kind(trait_name) {
            Some(BindingKind::Trait { methods }) => methods.clone(),
            _ => self.read_methods(decl.methods()),
        };

        let trait_symbol = if let Some((symbol, declared)) = self.symbols.trait_symbol(trait_name) {
            if let Some(written) = decl.trait_().and_then(|trait_ref| trait_ref.name()) {
                self.record(written.syntax().text_range(), trait_name, declared);
            }
            Some(symbol)
        } else {
            self.report(decl, &format!("unknown trait `{trait_name}`"));
            None
        };

        let target = if let Some((symbol, declared)) = self.symbols.struct_symbol(target) {
            if let Some(written) = decl.ty() {
                self.record(written.syntax().text_range(), target, declared);
            }
            symbol
        } else {
            if types::scalar(self.context, target).is_none() {
                self.report(decl, &format!("unknown type `{target}`"));
            }

            target
        };

        let region = Region::new();
        let body = region.append_block(Block::new(&[]));
        for method in self.check_methods(decl.methods()) {
            self.convert_method(body, &method, Site::InImpl(&methods));
        }

        // The methods are still checked, but an implementation of no trait
        // is not built.
        let Some(trait_symbol) = trait_symbol else {
            return;
        };

        block.append_operation(
            yzl::ImplOperationBuilder::new(self.context, self.location(decl))
                .r#trait(FlatSymbolRefAttribute::new(self.context, trait_symbol))
                .target(FlatSymbolRefAttribute::new(self.context, target))
                .body(region)
                .build()
                .into(),
        );
    }

    /// In a function body a `let` is a local value; at the file level it is
    /// a symbol, since a stage region cannot reach a value outside itself.
    fn convert_let<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        locals: &mut Locals<'c, 'a>,
        decl: &ast::LetStmt,
    ) {
        let loc = self.location(decl);

        let Some(name) = self.read_ident(decl.name()) else {
            self.assert_syntax_error("let binding is missing its name");
            return;
        };

        let Some(expr) = decl.expr() else {
            self.assert_syntax_error("let binding is missing its expression");
            return;
        };

        if self.symbols.is_in_body() {
            let element = match decl.type_annotation() {
                Some(annotation) => self.read_type_annotation(annotation),
                None => UnresolvedType::new(self.context).into(),
            };
            let kind = LocalKind::Let(decl.mutability());
            let value = self.convert_expr(block, locals, &expr);
            let place = self.emit_local(block, name, kind, element, value, loc);
            self.bind_local(locals, name, decl.syntax().text_range(), place);
            return;
        }

        // A file-level binding is copied into each use, so nothing could see
        // a change to it.
        if let Some(token) = decl.mut_token() {
            let diagnostic = self
                .diagnostic_at(token.text_range(), "a file-level binding cannot be `mut`")
                .note("a file-level binding is a constant; move it into a function to change it");

            self.diagnostics.emit(diagnostic);
        }

        if !self.was_hoisted(decl, name) {
            return;
        }

        let annotation = decl
            .type_annotation()
            .map(|annotation| self.read_type_annotation(annotation));

        let region = Region::new();
        let body = region.append_block(Block::new(&[]));
        let (value, row) = match &expr {
            ast::Expr::Pipeline(pipeline) => {
                let (value, row) = self.convert_query(body, pipeline);
                (value, Some(row))
            }
            _ => (self.convert_expr(body, &Locals::new(), &expr), None),
        };

        body.append_operation(yzl::r#yield(self.context, &[value], loc).into());

        let symbol = self.symbols.symbol_here(name);
        let mut builder = yzl::ConstOperationBuilder::new(self.context, loc)
            .sym_name(StringAttribute::new(self.context, symbol))
            .body(region);

        if let Some(annotation) = annotation {
            builder = builder.annotation(TypeAttribute::new(annotation));
        }

        let mut constant: Operation<'c> = builder.build().into();
        if decl.visibility() != Visibility::Public {
            constant.set_private(self.context);
        }

        block.append_operation(constant);

        let kind = match row {
            Some(row) => BindingKind::Relation { row },
            None => BindingKind::Let,
        };

        self.symbols.bind(
            name,
            Binding {
                kind,
                text_range: decl.syntax().text_range(),
                visibility: decl.visibility(),
            },
        );
    }

    fn convert_assign<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        locals: &mut Locals<'c, 'a>,
        assign: &ast::AssignStmt,
    ) {
        if !self.check_in_body(assign, "an assignment is not a top-level statement") {
            return;
        }

        let target = match assign.target() {
            Some(ast::Expr::IdentExpr(ident)) => self
                .read_ident(ident.name())
                .map(|name| (name, ident.syntax().text_range())),
            Some(target) => {
                self.report(&target, "only a name can be assigned to");
                return;
            }
            None => None,
        };

        let Some((name, written)) = target else {
            self.assert_syntax_error("assignment is missing its target");
            return;
        };

        let Some(value) = assign.value() else {
            self.assert_syntax_error("assignment is missing its value");
            return;
        };

        let Some(slot) = self.resolve_place(assign, name, written) else {
            return;
        };

        let value = self.convert_expr(block, locals, &value);
        let loc = self.location(assign);
        block.append_operation(yzl::store(self.context, locals[slot], value, loc).into());
    }

    fn convert_return<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        locals: &mut Locals<'c, 'a>,
        ret: &ast::ReturnStmt,
    ) {
        if !self.check_in_body(ret, "a return is not a top-level statement") {
            return;
        }

        let values: Vec<Value> = ret
            .expr()
            .map(|expr| self.convert_expr(block, locals, &expr))
            .into_iter()
            .collect();
        block.append_operation(yzl::r#return(self.context, &values, self.location(ret)).into());
    }

    fn convert_nested_block<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        locals: &mut Locals<'c, 'a>,
        body: &ast::BlockStmt,
    ) {
        if !self.check_in_body(body, "a block is not a top-level statement") {
            return;
        }

        self.convert_block(block, locals, body);
    }

    fn convert_block<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        locals: &mut Locals<'c, 'a>,
        body: &ast::BlockStmt,
    ) {
        self.in_block(|this| {
            for stmt in body.stmts() {
                if ends_in_return(block) {
                    this.report(&stmt, "nothing can follow a `return`");
                    break;
                }
                this.convert_stmt(block, locals, &stmt);
            }
        });
    }

    fn convert_expr_stmt<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        locals: &mut Locals<'c, 'a>,
        stmt: &ast::ExprStmt,
    ) {
        let Some(expr) = stmt.expr() else {
            self.assert_syntax_error("expression statement is missing its expression");
            return;
        };

        // The program's query is the entry file's.
        let is_query = matches!(expr, ast::Expr::Pipeline(_));
        if is_query && !self.symbols.is_in_body() && !self.symbols.module().is_entry() {
            self.report(stmt, "a module cannot hold a query");
            return;
        }

        self.convert_expr(block, locals, &expr);
    }

    fn convert_import(&mut self, node: &impl AstNode) {
        self.check_at_file_level(node, "this belongs at the top of the file");
    }

    pub(super) fn hoist_declarations(&mut self, root: &ast::Root) {
        for stmt in root.stmts() {
            match &stmt {
                ast::Stmt::StructStmt(decl) => self.hoist_struct(decl),
                ast::Stmt::TraitStmt(decl) => self.hoist_trait(decl),
                ast::Stmt::FuncStmt(decl) => self.hoist_fn(decl),
                ast::Stmt::TableStmt(_)
                | ast::Stmt::ImplStmt(_)
                | ast::Stmt::LetStmt(_)
                | ast::Stmt::ExprStmt(_)
                | ast::Stmt::BlockStmt(_)
                | ast::Stmt::AssignStmt(_)
                | ast::Stmt::ReturnStmt(_)
                | ast::Stmt::ImportStmt(_)
                | ast::Stmt::FromImportStmt(_)
                | ast::Stmt::ModStmt(_) => {}
            }
        }

        // A table's row is its struct's fields, so tables go in last.
        for stmt in root.stmts() {
            match &stmt {
                ast::Stmt::TableStmt(decl) => self.hoist_table(decl),
                ast::Stmt::LetStmt(decl) => self.hoist_let(decl),
                ast::Stmt::StructStmt(_)
                | ast::Stmt::TraitStmt(_)
                | ast::Stmt::FuncStmt(_)
                | ast::Stmt::ImplStmt(_)
                | ast::Stmt::ExprStmt(_)
                | ast::Stmt::BlockStmt(_)
                | ast::Stmt::AssignStmt(_)
                | ast::Stmt::ReturnStmt(_)
                | ast::Stmt::ImportStmt(_)
                | ast::Stmt::FromImportStmt(_)
                | ast::Stmt::ModStmt(_) => {}
            }
        }
    }

    fn hoist_struct(&mut self, decl: &ast::StructStmt) {
        let Some(name) = self.read_ident(decl.name()) else {
            return;
        };

        let fields = self.read_field_names(decl.fields());
        self.bind_or_report(
            decl,
            name,
            BindingKind::Struct { fields },
            decl.visibility(),
        );
    }

    fn hoist_table(&mut self, decl: &ast::TableStmt) {
        let Some(name) = self.read_ident(decl.name()) else {
            return;
        };

        let row = if decl.inline_fields().next().is_some() {
            Row::from(self.read_field_names(decl.inline_fields()))
        } else {
            let Some(declared) = self.read_ident(decl.struct_name()) else {
                return;
            };

            // The struct may be imported, and an import names where it was
            // written rather than repeating it.
            if let Some(BindingKind::Struct { fields, .. }) = self.symbols.kind(declared) {
                Row::from(fields.clone())
            } else {
                self.report(decl, &format!("`{declared}` is not a struct"));
                return;
            }
        };

        self.bind_or_report(decl, name, BindingKind::Relation { row }, decl.visibility());
    }

    /// Each field's name, and the field that declares it.
    fn read_field_names(&self, fields: impl Iterator<Item = ast::StructField>) -> Vec<Field<'c>> {
        fields
            .filter_map(|field| {
                Some(Field {
                    name: self.read_ident(field.name())?,
                    declared: Some(self.span(field.syntax().text_range())),
                })
            })
            .collect()
    }

    fn hoist_trait(&mut self, decl: &ast::TraitStmt) {
        let Some(name) = self.read_ident(decl.name()) else {
            return;
        };

        let methods = self.read_methods(decl.methods());
        self.bind_or_report(
            decl,
            name,
            BindingKind::Trait { methods },
            decl.visibility(),
        );
    }

    fn hoist_fn(&mut self, decl: &ast::FuncStmt) {
        let Some(name) = self.read_ident(decl.name()) else {
            return;
        };

        let source = if decl.is_external() {
            CalleeSource::External
        } else {
            CalleeSource::Fn
        };
        let kind = if decl.is_agg() {
            FunctionKind::Aggregate
        } else {
            FunctionKind::Scalar
        };
        let overload = Overload {
            source,
            arity: decl.params().count(),
            text_range: decl.syntax().text_range(),
            visibility: decl.visibility(),
        };

        if let Some(Binding {
            kind: BindingKind::Func { .. },
            ..
        }) = self.symbols.binding(name)
        {
            self.bind_overload(decl, name, kind, overload);
            return;
        }

        self.bind_or_report(
            decl,
            name,
            BindingKind::Func {
                kind,
                overloads: vec![overload],
            },
            decl.visibility(),
        );
    }

    fn hoist_let(&mut self, decl: &ast::LetStmt) {
        let Some(name) = self.read_ident(decl.name()) else {
            return;
        };

        self.bind_or_report(decl, name, BindingKind::Pending, decl.visibility());
    }

    pub(super) fn bind_imports(&mut self, root: &ast::Root) {
        for stmt in root.stmts() {
            match &stmt {
                ast::Stmt::FromImportStmt(import) => {
                    let Some(written) = import.path() else {
                        continue;
                    };
                    let path = self.read_path(&written);
                    self.record_module(written.syntax().text_range(), path, path);

                    for item in import.items() {
                        self.bind_import(path, &item, import.visibility());
                    }
                }
                ast::Stmt::ImportStmt(import) => {
                    let Some(written) = import.path() else {
                        continue;
                    };
                    let path = self.read_path(&written);
                    self.record_module(written.syntax().text_range(), path, path);
                    if let Some(alias) = import.alias()
                        && let Some(spelling) = self.read_ident(Some(alias.clone()))
                    {
                        self.record_module(alias.syntax().text_range(), spelling, path);
                    }

                    let last = path
                        .rsplit('.')
                        .next()
                        .expect("a split yields at least one piece");

                    let name = self.read_ident(import.alias()).unwrap_or(last);
                    self.bind_or_report(
                        import,
                        name,
                        BindingKind::Module { path },
                        import.visibility(),
                    );
                }
                ast::Stmt::ModStmt(decl) => {
                    let Some(name) = self.read_ident(decl.name()) else {
                        self.assert_syntax_error("module declaration is missing its name");
                        continue;
                    };

                    let path = self.symbols.symbol_here(name);
                    self.bind_or_report(
                        decl,
                        name,
                        BindingKind::Module { path },
                        decl.visibility(),
                    );
                }
                ast::Stmt::StructStmt(_)
                | ast::Stmt::TableStmt(_)
                | ast::Stmt::FuncStmt(_)
                | ast::Stmt::TraitStmt(_)
                | ast::Stmt::ImplStmt(_)
                | ast::Stmt::LetStmt(_)
                | ast::Stmt::ExprStmt(_)
                | ast::Stmt::BlockStmt(_)
                | ast::Stmt::AssignStmt(_)
                | ast::Stmt::ReturnStmt(_) => {}
            }
        }
    }

    /// `visibility` is the import's own: a module passes on what it imports
    /// only when `pub` says so.
    fn bind_import(&mut self, path: &'c str, item: &ast::ImportItem, visibility: Visibility) {
        let Some(name) = self.read_ident(item.name()) else {
            self.assert_syntax_error("import item is missing its name");
            return;
        };

        let Some((from, exported)) = self.resolve_export(item, path, name) else {
            return;
        };

        let declared = Target {
            at: from,
            range: exported.text_range,
        };
        let alias = item.alias();
        for written in item.name().into_iter().chain(alias.clone()) {
            if let Some(spelling) = self.read_ident(Some(written.clone())) {
                self.record(written.syntax().text_range(), spelling, declared);
            }
        }

        let local = self.read_ident(alias).unwrap_or(name);

        if self.symbols.binding(local).is_some() {
            self.check_duplicate(item, exported.kind.name(), local);
            return;
        }

        self.symbols.bind(
            local,
            Binding {
                kind: BindingKind::Import { from },
                text_range: item.syntax().text_range(),
                visibility,
            },
        );
    }

    fn bind_or_report(
        &mut self,
        node: &impl AstNode,
        name: &'c str,
        kind: BindingKind<'c>,
        visibility: Visibility,
    ) {
        if self.symbols.binding(name).is_some() {
            self.check_duplicate(node, kind.name(), name);
            return;
        }

        self.symbols.bind(
            name,
            Binding {
                kind,
                text_range: node.syntax().text_range(),
                visibility,
            },
        );
    }

    /// Adds a declaration to a function name this module declares already.
    /// Every overload has the same kind, and each takes a different number
    /// of arguments, so a call names one overload.
    fn bind_overload(
        &mut self,
        decl: &ast::FuncStmt,
        name: &'c str,
        kind: FunctionKind,
        overload: Overload,
    ) {
        let Some(Binding {
            kind:
                BindingKind::Func {
                    kind: declared,
                    overloads,
                },
            ..
        }) = self.symbols.binding(name)
        else {
            unreachable!("`{name}` is bound to a function before an overload is added")
        };

        let declared = *declared;
        let first = overloads[0].text_range;
        let same_arity = overloads
            .iter()
            .find(|other| other.arity == overload.arity)
            .map(|other| other.text_range);

        if declared != kind {
            let message = format!("every overload of `{name}` must be a {}", declared.name());
            self.report_duplicate(decl, &message, first);
            return;
        }

        if let Some(other) = same_arity {
            let message = format!(
                "the function `{name}` with {} parameter(s) is already defined",
                overload.arity
            );
            self.report_duplicate(decl, &message, other);
            return;
        }

        self.symbols.add_overload(name, overload);
    }

    /// Binds a name to a place the body has declared at `declared`.
    fn bind_local<'a>(
        &mut self,
        locals: &mut Locals<'c, 'a>,
        name: &'c str,
        declared: TextRange,
        place: Value<'c, 'a>,
    ) {
        self.symbols.bind_local(name, locals.len(), declared);
        locals.push(place);
    }

    /// The declaration a module exports under `name`, followed through the
    /// module's own imports to the file that wrote it.
    pub(super) fn resolve_export(
        &mut self,
        at: &impl AstNode,
        path: &'c str,
        name: &str,
    ) -> Option<(Declared<'c>, Binding<'c>)> {
        let module = ModulePath::from_path(path);
        if !self.symbols.contains_module(module) {
            self.report(at, &format!("`{path}` is not a module this file reads"));
            return None;
        }

        // The module's own entry decides: an import it did not mark `pub`
        // is not exported, whatever the origin's visibility.
        let Some(own) = self.symbols.declared_binding(module, name) else {
            self.report(at, &format!("`{path}` does not declare `{name}`"));
            return None;
        };

        if own.visibility != Visibility::Public {
            let is_import = matches!(own.kind, BindingKind::Import { .. });
            let mut diagnostic = self.diagnostic_at(
                at.syntax().text_range(),
                &format!("`{name}` is not public; `{path}` keeps it to itself"),
            );
            if is_import {
                diagnostic = diagnostic.note(format!(
                    "`{path}` imports `{name}`; `pub from` would export it again"
                ));
            }

            self.diagnostics.emit(diagnostic);
            return None;
        }

        let found = self
            .symbols
            .find_declared(module, name)
            .map(|(declared, binding)| (declared, binding.clone()))
            .expect("an import points at a declaration that exists");
        Some(found)
    }

    /// One entry per `(parameter, trait)` pair.
    fn read_bounds(&mut self, decl: &ast::FuncStmt) -> (Vec<&'c str>, Vec<&'c str>) {
        let mut subjects = Vec::new();
        let mut traits = Vec::new();
        for bound in decl.bounds() {
            let Some(subject) = self.read_ident(bound.subject()) else {
                self.assert_syntax_error("type bound is missing its subject");
                continue;
            };

            if !self.symbols.is_type_param(subject) {
                self.report(&bound, &format!("unknown type parameter `{subject}`"));
            }

            for trait_ref in bound.traits() {
                let Some(name) = self.read_ident(trait_ref.name()) else {
                    self.assert_syntax_error("trait reference is missing its name");
                    continue;
                };

                // A bound names the trait's symbol, the one its `impl`s are
                // recorded under: in a module the two are not spelled alike.
                let Some((bound_trait, declared)) = self.symbols.trait_symbol(name) else {
                    self.report(&trait_ref, &format!("unknown trait `{name}`"));
                    continue;
                };
                if let Some(written) = trait_ref.name() {
                    self.record(written.syntax().text_range(), name, declared);
                }

                subjects.push(subject);
                traits.push(bound_trait);
            }
        }

        (subjects, traits)
    }

    fn read_methods(&self, methods: impl Iterator<Item = ast::FuncStmt>) -> Vec<Method<'c>> {
        methods
            .filter_map(|method| {
                Some(Method {
                    name: self.read_ident(method.name())?,
                    arity: method.params().count(),
                })
            })
            .collect()
    }

    /// The methods of a trait or an implementation, less each one that
    /// takes as many arguments as an earlier method of its name.
    fn check_methods(
        &mut self,
        methods: impl Iterator<Item = ast::FuncStmt>,
    ) -> Vec<ast::FuncStmt> {
        let mut kept = Vec::new();
        let mut seen: Vec<(Method<'c>, TextRange)> = Vec::new();
        for decl in methods {
            // `convert_method` reports a missing name.
            if let Some(name) = self.read_ident(decl.name()) {
                let method = Method {
                    name,
                    arity: decl.params().count(),
                };

                if let Some(&(_, other)) = seen.iter().find(|(earlier, _)| *earlier == method) {
                    let message = format!(
                        "the method `{name}` with {} parameter(s) is already defined",
                        method.arity
                    );
                    self.report_duplicate(&decl, &message, other);
                    continue;
                }

                seen.push((method, decl.syntax().text_range()));
            }

            kept.push(decl);
        }

        kept
    }

    /// `!yzl.error` for an annotation that names no type, so inference takes
    /// it as the error it is rather than as a type to infer.
    fn read_type_annotation(&mut self, annotation: ast::TypeAnnotation) -> Type<'c> {
        let named = match annotation {
            ast::TypeAnnotation::NamedTypeAnnotation(named) => named,
            ast::TypeAnnotation::FuncTypeAnnotation(func) => {
                self.report(&func, "function types are not supported yet");
                return ErrorType::new(self.context).into();
            }
        };

        let Some(name) = self.read_ident(named.name()) else {
            self.assert_syntax_error("type is missing its name");
            return ErrorType::new(self.context).into();
        };

        if self.symbols.is_type_param(name) {
            return ParamType::new(self.context, name).into();
        }

        if name == "List" {
            let mut args = named.args();
            let (Some(inner), None) = (args.next(), args.next()) else {
                self.report(&named, "`List` takes exactly one type argument");
                return ErrorType::new(self.context).into();
            };

            let inner = self.read_type_annotation(inner);
            return ListType::new(self.context, inner).into();
        }

        if let Some(scalar) = types::scalar(self.context, name) {
            return scalar;
        }

        if let Some((symbol, declared)) = self.symbols.struct_symbol(name) {
            if let Some(written) = named.name() {
                self.record(written.syntax().text_range(), name, declared);
            }
            return StructType::new(self.context, symbol).into();
        }

        self.report(&named, &format!("unknown type `{name}`"));
        ErrorType::new(self.context).into()
    }

    fn read_path(&self, path: &ast::ModulePath) -> &'c str {
        self.symbols.intern(&path.to_dotted())
    }

    /// The slot of the place an assignment writes, when the name written at
    /// `written` is one.
    fn resolve_place(
        &mut self,
        node: &impl AstNode,
        name: &str,
        written: TextRange,
    ) -> Option<usize> {
        let message = match self.symbols.lookup(Reference::unqualified(name)) {
            Lookup::Local { slot, declared } => {
                self.record_local(written, name, declared);
                return Some(slot);
            }
            Lookup::Lost => return None,
            Lookup::Column { .. } | Lookup::Ambiguous | Lookup::NarrowedAway => {
                format!("`{name}` is a column; `set` is how a query writes one")
            }
            Lookup::Let(..) => {
                format!("`{name}` is a module-level binding and cannot be assigned")
            }
            Lookup::NotAValue(what) => format!("`{name}` is a {what}, not a binding"),
            Lookup::NotYet => format!("`{name}` is bound further down the file"),
            Lookup::Unknown => format!("unresolved identifier `{name}`"),
        };

        self.report(node, &message);
        None
    }

    /// Declares a place for a local variable and stores its first value.
    /// `element` is the type the place holds: what the program wrote, or
    /// `!yzl.unresolved` for inference.
    fn emit_local<'a>(
        &self,
        block: BlockRef<'c, 'a>,
        name: &str,
        kind: LocalKind,
        element: Type<'c>,
        value: Value<'c, 'a>,
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        let builder = yzl::LocalOperationBuilder::new(self.context, loc)
            .place(RefType::new(self.context, element).into())
            .var_name(StringAttribute::new(self.context, name));

        let builder = match kind {
            LocalKind::Let(Mutability::Mutable) => builder.is_mut(Attribute::unit(self.context)),
            LocalKind::Let(Mutability::Immutable) => builder,
            LocalKind::Param => builder.is_param(Attribute::unit(self.context)),
        };

        let place = block
            .append_operation(builder.build().into())
            .first_result();

        block.append_operation(yzl::store(self.context, place, value, loc).into());
        place
    }

    fn read_fields(
        &mut self,
        fields: impl Iterator<Item = ast::StructField>,
    ) -> (Vec<&'c str>, Vec<Type<'c>>) {
        let mut names = Vec::new();
        let mut types = Vec::new();
        for field in fields {
            let (Some(name), Some(ty)) = (self.read_ident(field.name()), field.ty()) else {
                self.assert_syntax_error("struct field is missing its name or type");
                continue;
            };

            names.push(name);
            types.push(self.read_type_annotation(ty));
        }

        (names, types)
    }

    /// The `return` a body that does not end in one gets: none of a value in
    /// a unit function, and a hole after a report in any other.
    fn emit_final_return<'a>(
        &mut self,
        entry: BlockRef<'c, 'a>,
        decl: &ast::FuncStmt,
        name: &str,
        result: Type<'c>,
        body: &ast::BlockStmt,
    ) {
        if ends_in_return(entry) {
            return;
        }

        let range = body.syntax().text_range();
        let values = if UnitType::from_type(result).is_some() {
            Vec::new()
        } else {
            if ErrorType::from_type(result).is_none() {
                let message = format!(
                    "`{name}` must end with a `return` of `{}`",
                    types::name(result)
                );
                self.report(decl, &message);
            }
            vec![self.emit_hole(entry, range, ErrorType::new(self.context).into())]
        };

        let loc = self.location_at(range);
        entry.append_operation(yzl::r#return(self.context, &values, loc).into());
    }

    fn emit_struct<'a>(
        &self,
        block: BlockRef<'c, 'a>,
        symbol: &str,
        (names, types): (Vec<&'c str>, Vec<Type<'c>>),
        visibility: Visibility,
        loc: Location<'c>,
    ) {
        let mut struct_: Operation<'c> = yzl::r#struct(
            self.context,
            StringAttribute::new(self.context, symbol),
            ArrayAttribute::from_strings(self.context, &names),
            ArrayAttribute::from_types(self.context, types),
            loc,
        )
        .into();

        if visibility != Visibility::Public {
            struct_.set_private(self.context);
        }

        block.append_operation(struct_);
    }

    fn check_declaration_at_file_level(&mut self, node: &impl AstNode) -> bool {
        self.check_at_file_level(node, "declarations inside functions are not supported yet")
    }

    fn check_at_file_level(&mut self, node: &impl AstNode, message: &str) -> bool {
        if self.symbols.is_in_body() {
            self.report(node, message);
            return false;
        }

        true
    }

    fn check_in_body(&mut self, node: &impl AstNode, message: &str) -> bool {
        if !self.symbols.is_in_body() {
            self.report(node, message);
            return false;
        }

        true
    }

    fn check_duplicate(&mut self, node: &impl AstNode, what: &str, name: &str) {
        let text_range = self
            .symbols
            .binding(name)
            .expect("a duplicate is checked against a bound name")
            .text_range;

        self.report_duplicate(
            node,
            &format!("the {what} `{name}` is already defined"),
            text_range,
        );
    }

    fn report_duplicate(&mut self, node: &impl AstNode, message: &str, other: TextRange) {
        let other = self.text_at_range(other);
        let diagnostic = self
            .diagnostic_at(node.syntax().text_range(), message)
            .note(format!("also declared at {other}"));

        self.diagnostics.emit(diagnostic);
    }

    /// The symbol a function is built under. A trait or an implementation
    /// is a symbol table of its own, so a method keeps its bare name, with
    /// its parameter count when its trait overloads it.
    fn symbol_in(&self, site: Site<'_, 'c>, name: &'c str, arity: usize) -> &'c str {
        match site {
            Site::AtModule => self
                .symbols
                .function_symbol(self.symbols.module().declares(name), arity),
            Site::InTrait(methods) | Site::InImpl(methods) => {
                let overloads = methods.iter().filter(|method| method.name == name);
                self.symbols
                    .overload_symbol(name, arity, overloads.count() > 1)
            }
        }
    }

    /// Whether the hoist bound this declaration. When it did not, it
    /// reported why.
    fn was_hoisted(&self, node: &impl AstNode, name: &str) -> bool {
        let hoisted = self
            .symbols
            .binding(name)
            .is_some_and(|binding| binding.is_declared_at(node.syntax().text_range()));
        debug_assert!(
            hoisted || self.diagnostics.has_errors(),
            "`{name}` was not bound by the hoist and nothing reported why"
        );
        hoisted
    }
}

/// Whether a block's last op is a `return`, after which nothing may come.
fn ends_in_return(block: BlockRef<'_, '_>) -> bool {
    block
        .last_operation()
        .is_some_and(|op| matches!(op.as_yzl(), Some(YzlOp::Return(_))))
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::test_support::{lower, lowered, lowered_program, reported};

    #[test]
    fn a_file_level_let_cannot_be_mut() {
        expect![[r"
            error: a file-level binding cannot be `mut`
             --> test.yz:4:5
              |
            4 | let mut cap = 1
              |     ^^^
              = note: a file-level binding is a constant; move it into a function to change it
        "]]
        .assert_eq(&reported(
            "struct Row { a: int64 }\ntable t = Row\n\nlet mut cap = 1\n\nfrom t |> select a + cap as v\n",
        ));
    }

    #[test]
    fn a_body_statement_at_the_file_level_is_reported() {
        expect![[r"
            error: a return is not a top-level statement
             --> test.yz:4:1
              |
            4 | return 1
              | ^^^^^^^^

            error: an assignment is not a top-level statement
             --> test.yz:5:1
              |
            5 | x = 2
              | ^^^^^
        "]]
        .assert_eq(&reported(
            "struct Row { a: int64 }\ntable t = Row\n\nreturn 1\nx = 2\n\nfrom t |> select a as v\n",
        ));
    }

    #[test]
    fn an_import_in_a_body_is_reported() {
        expect![[r"
            error: this belongs at the top of the file
             --> test.yz:5:3
              |
            5 |   import helpers
              |   ^^^^^^^^^^^^^^
        "]]
        .assert_eq(&reported(
            "struct Row { a: int64 }\ntable t = Row\n\ndef f(x: int64) -> int64 {\n  import helpers\n  return x\n}\n\nfrom t |> select f(a) as v\n",
        ));
    }

    #[test]
    fn a_type_parameter_leaves_scope_when_its_function_fails() {
        expect![[r"
            error: parameter is missing its type
             --> test.yz:1:10
              |
            1 | def f[T](x) -> T { return x }
              |          ^

            error: unknown type `T`
             --> test.yz:2:10
              |
            2 | def g(x: T) -> int64 { return 1 }
              |          ^
        "]]
        .assert_eq(&reported(
            "def f[T](x) -> T { return x }\ndef g(x: T) -> int64 { return 1 }\n",
        ));
    }

    #[test]
    fn an_implementation_of_an_unknown_trait_is_not_built() {
        let context = yuzu_mlir::context();
        let lowered = lower(
            &context,
            &[(
                "test.yz",
                None,
                "impl Missing for int64 {\n  def zero(x: int64) -> int64 { return 0 }\n}\n",
            )],
        );
        expect![[r"
            error: unknown trait `Missing`
             --> test.yz:1:1
              |
            1 | impl Missing for int64 {
              | ^^^^^^^^^^^^^^^^^^^^^^^^
            module {
            }
        "]]
        .assert_eq(&format!(
            "{}{}",
            crate::test_support::rendered(&lowered.sources, &lowered.diagnostics),
            lowered.module.as_operation()
        ));
    }

    #[test]
    fn only_a_name_can_be_assigned_to() {
        expect![[r"
            error: only a name can be assigned to
             --> test.yz:4:3
              |
            4 |   r.a = 1
              |   ^^^
        "]].assert_eq(&reported(
            "struct Row { a: int64 }\ndef f(x: int64) -> int64 {\n  let mut r = x\n  r.a = 1\n  return r\n}\n",
        ));
    }

    #[test]
    fn an_external_in_a_module_keeps_its_written_name() {
        expect![[r#"
            module {
              yzl.fn @helpers.median params ["x"] (!yz.float64) -> !yz.float64 external "median" {
              }
              yzl.struct @Row ["r"] : [!yz.float64] {sym_visibility = "private"}
              yzl.table @t of @Row {sym_visibility = "private"}
              %0 = yzl.from @t
              %1 = yzl.select %0 as ["m"] {
              ^bb0(%arg0: !yzl.unresolved):
                %2 = yzl.call @helpers.median(%arg0) : (!yzl.unresolved) -> !yzl.unresolved {callee_source = "external"}
                yzl.yield %2 : !yzl.unresolved
              }
              yzl.output %1
            }
        "#]].assert_eq(&lowered_program(&[
            (
                "helpers.yz",
                Some("helpers"),
                "pub external def median(x: float64) -> float64\n",
            ),
            (
                "main.yz",
                None,
                "from helpers import median\nstruct Row { r: float64 }\ntable t = Row\nfrom t |> select median(r) as m\n",
            ),
        ]));
    }

    #[test]
    fn a_module_level_let_cannot_take_a_name_again() {
        expect![[r"
            error: the binding `cap` is already defined
             --> test.yz:5:1
              |
            5 | let cap = 2
              | ^^^^^^^^^^^
              = note: also declared at test.yz:4:1
        "]]
        .assert_eq(&reported(
            "struct Row { a: int64 }\ntable t = Row\n\nlet cap = 1\nlet cap = 2\n\nfrom t\n|> where a > cap\n",
        ));
    }

    #[test]
    fn converts_declarations() {
        expect![[r#"
            module {
              yzl.fn @f params ["x"] (!yz.int64) -> !yz.int64 {
              ^bb0(%arg0: !yzl.unresolved):
                %0 = yzl.local "x" param : !yzl.ref<!yz.int64>
                yzl.store %0, %arg0 : !yzl.ref<!yz.int64>, !yzl.unresolved
                %1 = yzl.load %0 : !yzl.ref<!yz.int64> -> !yzl.unresolved
                %2 = yz.constant_int 2
                %3 = yz.mul %1, %2 : !yzl.unresolved, !yz.int64 -> !yzl.unresolved
                %4 = yzl.local "doubled" : !yzl.ref<!yzl.unresolved>
                yzl.store %4, %3 : !yzl.ref<!yzl.unresolved>, !yzl.unresolved
                %5 = yzl.load %4 : !yzl.ref<!yzl.unresolved> -> !yzl.unresolved
                %6 = yz.constant_int 1
                %7 = yz.add %5, %6 : !yzl.unresolved, !yz.int64 -> !yzl.unresolved
                yzl.return %7 : !yzl.unresolved
              } {sym_visibility = "private"}
              yzl.fn @spread params ["x"] (!yz.int64) -> !yz.int64 agg {
              ^bb0(%arg0: !yzl.unresolved):
                %0 = yzl.local "x" param : !yzl.ref<!yz.int64>
                yzl.store %0, %arg0 : !yzl.ref<!yz.int64>, !yzl.unresolved
                %1 = yzl.load %0 : !yzl.ref<!yz.int64> -> !yzl.unresolved
                %2 = yzl.call @yuzu.prelude.max(%1) : (!yzl.unresolved) -> !yzl.unresolved {callee_source = "external", is_agg}
                %3 = yzl.load %0 : !yzl.ref<!yz.int64> -> !yzl.unresolved
                %4 = yzl.call @yuzu.prelude.min(%3) : (!yzl.unresolved) -> !yzl.unresolved {callee_source = "external", is_agg}
                %5 = yz.sub %2, %4 : !yzl.unresolved, !yzl.unresolved -> !yzl.unresolved
                yzl.return %5 : !yzl.unresolved
              } {sym_visibility = "private"}
              yzl.fn @upper params ["s"] (!yz.str) -> !yz.str external "upper" {
              } {sym_visibility = "private"}
              yzl.fn @yuzu.prelude.max generics ["T"] params ["x"] (!yzl.param<"T">) -> !yzl.param<"T"> agg external "max" {
              }
              yzl.fn @yuzu.prelude.min generics ["T"] params ["x"] (!yzl.param<"T">) -> !yzl.param<"T"> agg external "min" {
              }
            }
        "#]]
        .assert_eq(&lowered(
            r"
def f(x: int64) -> int64 {
    let doubled = x * 2
    return doubled + 1
}

agg def spread(x: int64) -> int64 {
    return max(x) - min(x)
}

external def upper(s: str) -> str
",
        ));
    }

    #[test]
    fn converts_expression_statements_in_bodies() {
        expect![[r#"
            module {
              yzl.struct @Row ["a"] : [!yz.int64] {sym_visibility = "private"}
              yzl.table @t of @Row {sym_visibility = "private"}
              yzl.fn @f params ["x"] (!yz.int64) -> !yz.int64 {
              ^bb0(%arg0: !yzl.unresolved):
                %1 = yzl.local "x" param : !yzl.ref<!yz.int64>
                yzl.store %1, %arg0 : !yzl.ref<!yz.int64>, !yzl.unresolved
                %2 = yzl.load %1 : !yzl.ref<!yz.int64> -> !yzl.unresolved
                %3 = yz.constant_int 1
                %4 = yz.add %2, %3 : !yzl.unresolved, !yz.int64 -> !yzl.unresolved
                %5 = yzl.load %1 : !yzl.ref<!yz.int64> -> !yzl.unresolved
                %6 = yz.constant_int 2
                %7 = yz.mul %5, %6 : !yzl.unresolved, !yz.int64 -> !yzl.unresolved
                %8 = yzl.local "y" : !yzl.ref<!yzl.unresolved>
                yzl.store %8, %7 : !yzl.ref<!yzl.unresolved>, !yzl.unresolved
                %9 = yzl.load %8 : !yzl.ref<!yzl.unresolved> -> !yzl.unresolved
                yzl.return %9 : !yzl.unresolved
              } {sym_visibility = "private"}
              %0 = yzl.from @t
              yzl.output %0
            }
        "#]].assert_eq(&lowered(
            "struct Row { a: int64 }\ntable t = Row\n\ndef f(x: int64) -> int64 {\n    x + 1\n    let y = x * 2\n    return y\n}\n\nfrom t\n",
        ));
    }

    #[test]
    fn converts_traits_and_generics() {
        expect![[r#"
            module {
              yzl.trait @Add {
                yzl.fn @add generics ["Self"] params ["x", "y"] (!yzl.param<"Self">, !yzl.param<"Self">) -> !yzl.param<"Self"> {
                }
              } {sym_visibility = "private"}
              yzl.impl @Add for @int64 {
                yzl.fn @add generics ["Self"] params ["x", "y"] (!yz.int64, !yz.int64) -> !yz.int64 {
                ^bb0(%arg0: !yzl.unresolved, %arg1: !yzl.unresolved):
                  %0 = yzl.local "x" param : !yzl.ref<!yz.int64>
                  yzl.store %0, %arg0 : !yzl.ref<!yz.int64>, !yzl.unresolved
                  %1 = yzl.local "y" param : !yzl.ref<!yz.int64>
                  yzl.store %1, %arg1 : !yzl.ref<!yz.int64>, !yzl.unresolved
                  %2 = yzl.load %0 : !yzl.ref<!yz.int64> -> !yzl.unresolved
                  %3 = yzl.load %1 : !yzl.ref<!yz.int64> -> !yzl.unresolved
                  %4 = yz.add %2, %3 : !yzl.unresolved, !yzl.unresolved -> !yzl.unresolved
                  yzl.return %4 : !yzl.unresolved
                }
              }
              yzl.fn @id generics ["T"] where ["T"] : [@Add] params ["x"] (!yzl.param<"T">) -> !yzl.param<"T"> {
              ^bb0(%arg0: !yzl.unresolved):
                %0 = yzl.local "x" param : !yzl.ref<!yzl.param<"T">>
                yzl.store %0, %arg0 : !yzl.ref<!yzl.param<"T">>, !yzl.unresolved
                %1 = yzl.load %0 : !yzl.ref<!yzl.param<"T">> -> !yzl.unresolved
                yzl.return %1 : !yzl.unresolved
              } {sym_visibility = "private"}
            }
        "#]].assert_eq(&lowered(
        "trait Add {\n    def add(x: Self, y: Self) -> Self\n}\n\nimpl Add for int64 {\n    def add(x: int64, y: int64) -> int64 { return x + y }\n}\n\ndef id[T](x: T) -> T where T: Add { return x }\n",
    ));
    }

    #[test]
    fn reports_unsupported_constructs() {
        expect![[r"
        error: declarations inside functions are not supported yet
         --> test.yz:2:5
          |
        2 |     struct S { a: int64 }
          |     ^^^^^^^^^^^^^^^^^^^^^
    "]]
        .assert_eq(&reported(
            "def f(x: int64) -> int64 {\n    struct S { a: int64 }\n    return x\n}\n",
        ));
    }

    #[test]
    fn converts_an_annotated_let() {
        expect![[r#"
            module {
              yzl.const @ids : !yz.list<!yz.int64> {
                %0 = yz.constant_list [1, 3] : <!yz.int64>
                yzl.yield %0 : !yz.list<!yz.int64>
              } {sym_visibility = "private"}
            }
        "#]]
        .assert_eq(&lowered("let ids: List[int64] = [1, 3]\n"));
    }

    #[test]
    fn a_redeclaration_says_where_the_other_one_is() {
        expect![[r"
            error: the relation `Row` is already defined
             --> test.yz:3:1
              |
            3 | table Row = Row
              | ^^^^^^^^^^^^^^^
              = note: also declared at test.yz:1:1
        "]]
        .assert_eq(&reported("struct Row { a: int64 }\n\ntable Row = Row\n"));
    }

    #[test]
    fn overloads_are_told_apart_by_their_parameter_count() {
        expect![[r#"
            module {
              yzl.fn @f.1 params ["x"] (!yz.int64) -> !yz.int64 {
              ^bb0(%arg0: !yzl.unresolved):
                %2 = yzl.local "x" param : !yzl.ref<!yz.int64>
                yzl.store %2, %arg0 : !yzl.ref<!yz.int64>, !yzl.unresolved
                %3 = yzl.load %2 : !yzl.ref<!yz.int64> -> !yzl.unresolved
                yzl.return %3 : !yzl.unresolved
              } {sym_visibility = "private"}
              yzl.fn @f.2 params ["x", "y"] (!yz.int64, !yz.int64) -> !yz.int64 {
              ^bb0(%arg0: !yzl.unresolved, %arg1: !yzl.unresolved):
                %2 = yzl.local "x" param : !yzl.ref<!yz.int64>
                yzl.store %2, %arg0 : !yzl.ref<!yz.int64>, !yzl.unresolved
                %3 = yzl.local "y" param : !yzl.ref<!yz.int64>
                yzl.store %3, %arg1 : !yzl.ref<!yz.int64>, !yzl.unresolved
                %4 = yzl.load %2 : !yzl.ref<!yz.int64> -> !yzl.unresolved
                %5 = yzl.load %3 : !yzl.ref<!yz.int64> -> !yzl.unresolved
                %6 = yz.add %4, %5 : !yzl.unresolved, !yzl.unresolved -> !yzl.unresolved
                yzl.return %6 : !yzl.unresolved
              } {sym_visibility = "private"}
              yzl.struct @Row ["a"] : [!yz.int64] {sym_visibility = "private"}
              yzl.table @t of @Row {sym_visibility = "private"}
              %0 = yzl.from @t
              %1 = yzl.select %0 as ["one", "two"] {
              ^bb0(%arg0: !yzl.unresolved):
                %2 = yzl.call @f.1(%arg0) : (!yzl.unresolved) -> !yzl.unresolved {callee_source = "fn"}
                %3 = yzl.call @f.2(%arg0, %arg0) : (!yzl.unresolved, !yzl.unresolved) -> !yzl.unresolved {callee_source = "fn"}
                yzl.yield %2, %3 : !yzl.unresolved, !yzl.unresolved
              }
              yzl.output %1
            }
        "#]].assert_eq(&lowered(
            "def f(x: int64) -> int64 { return x }\ndef f(x: int64, y: int64) -> int64 { return x + y }\n\nstruct Row { a: int64 }\ntable t = Row\n\nfrom t |> select f(a) as one, f(a, a) as two\n",
        ));
    }

    #[test]
    fn an_overload_with_the_same_parameter_count_is_reported() {
        expect![[r"
            error: the function `f` with 1 parameter(s) is already defined
             --> test.yz:2:1
              |
            2 | def f(y: int64) -> int64 { return y }
              | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              = note: also declared at test.yz:1:1
        "]]
        .assert_eq(&reported(
            "def f(x: int64) -> int64 { return x }\ndef f(y: int64) -> int64 { return y }\n",
        ));
    }

    #[test]
    fn every_overload_has_the_same_kind() {
        expect![[r"
            error: every overload of `f` must be a scalar function
             --> test.yz:2:1
              |
            2 | agg def f(x: int64, y: int64) -> int64 { return x }
              | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              = note: also declared at test.yz:1:1
        "]].assert_eq(&reported(
            "def f(x: int64) -> int64 { return x }\nagg def f(x: int64, y: int64) -> int64 { return x }\n",
        ));
    }

    #[test]
    fn a_trait_method_may_overload() {
        expect![[r#"
            module {
              yzl.trait @Round {
                yzl.fn @round.1 generics ["Self"] params ["x"] (!yzl.param<"Self">) -> !yzl.param<"Self"> {
                }
                yzl.fn @round.2 generics ["Self"] params ["x", "digits"] (!yzl.param<"Self">, !yz.int64) -> !yzl.param<"Self"> {
                }
              } {sym_visibility = "private"}
              yzl.impl @Round for @int64 {
                yzl.fn @round.1 generics ["Self"] params ["x"] (!yz.int64) -> !yz.int64 {
                ^bb0(%arg0: !yzl.unresolved):
                  %0 = yzl.local "x" param : !yzl.ref<!yz.int64>
                  yzl.store %0, %arg0 : !yzl.ref<!yz.int64>, !yzl.unresolved
                  %1 = yzl.load %0 : !yzl.ref<!yz.int64> -> !yzl.unresolved
                  yzl.return %1 : !yzl.unresolved
                }
                yzl.fn @round.2 generics ["Self"] params ["x", "digits"] (!yz.int64, !yz.int64) -> !yz.int64 {
                ^bb0(%arg0: !yzl.unresolved, %arg1: !yzl.unresolved):
                  %0 = yzl.local "x" param : !yzl.ref<!yz.int64>
                  yzl.store %0, %arg0 : !yzl.ref<!yz.int64>, !yzl.unresolved
                  %1 = yzl.local "digits" param : !yzl.ref<!yz.int64>
                  yzl.store %1, %arg1 : !yzl.ref<!yz.int64>, !yzl.unresolved
                  %2 = yzl.load %0 : !yzl.ref<!yz.int64> -> !yzl.unresolved
                  yzl.return %2 : !yzl.unresolved
                }
              }
            }
        "#]].assert_eq(&lowered(
            "trait Round {\n    def round(x: Self) -> Self\n    def round(x: Self, digits: int64) -> Self\n}\n\nimpl Round for int64 {\n    def round(x: int64) -> int64 { return x }\n    def round(x: int64, digits: int64) -> int64 { return x }\n}\n",
        ));
    }

    #[test]
    fn a_method_with_the_same_parameter_count_is_reported() {
        expect![[r"
            error: the method `round` with 1 parameter(s) is already defined
             --> test.yz:3:5
              |
            3 |     def round(y: Self) -> Self
              |     ^^^^^^^^^^^^^^^^^^^^^^^^^^
              = note: also declared at test.yz:2:5
        "]]
        .assert_eq(&reported(
            "trait Round {\n    def round(x: Self) -> Self\n    def round(y: Self) -> Self\n}\n",
        ));
    }

    #[test]
    fn a_redeclaration_leaves_the_first_one_standing() {
        expect![[r"
            error: the struct `Row` is already defined
             --> test.yz:2:1
              |
            2 | struct Row { b: int64 }
              | ^^^^^^^^^^^^^^^^^^^^^^^
              = note: also declared at test.yz:1:1
        "]]
        .assert_eq(&reported(
            "struct Row { a: int64 }\nstruct Row { b: int64 }\ntable t = Row\n\nfrom t\n|> select a as x\n",
        ));
    }

    #[test]
    fn an_import_of_an_unloaded_module_is_reported() {
        expect![[r"
            error: `helpers` is not a module this file reads
             --> test.yz:1:21
              |
            1 | from helpers import spread
              |                     ^^^^^^
        "]]
        .assert_eq(&reported("from helpers import spread\n"));
    }

    #[test]
    fn a_redeclared_name_contributes_one_declaration() {
        let context = yuzu_mlir::context();
        let module = lower(
            &context,
            &[(
                "test.yz",
                None,
                "struct Row { a: int64 }\nstruct Row { b: int64 }\n",
            )],
        )
        .module
        .as_operation()
        .to_string();
        assert_eq!(
            module.matches("yzl.struct @Row").count(),
            1,
            "the duplicate is left out:\n{module}"
        );
        assert!(
            module.as_str().contains(r#"["a"]"#),
            "the first declaration is the one that stands:\n{module}"
        );
    }

    #[test]
    fn a_table_can_name_a_struct_declared_below_it() {
        expect![[r#"
            module {
              yzl.table @t of @Row {sym_visibility = "private"}
              yzl.struct @Row ["a"] : [!yz.int64] {sym_visibility = "private"}
              %0 = yzl.from @t
              %1 = yzl.select %0 as ["a"] {
              ^bb0(%arg0: !yzl.unresolved):
                yzl.yield %arg0 : !yzl.unresolved
              }
              yzl.output %1
            }
        "#]]
        .assert_eq(&lowered(
            "table t = Row\nstruct Row { a: int64 }\n\nfrom t |> select a\n",
        ));
    }

    #[test]
    fn a_body_is_required_unless_the_target_or_a_trait_supplies_it() {
        expect![[r"
            error: an external function cannot have a body
             --> test.yz:1:1
              |
            1 | external def upper(s: str) -> str { return s }
              | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

            error: function is missing its body
             --> test.yz:2:1
              |
            2 | def lower(s: str) -> str
              | ^^^^^^^^^^^^^^^^^^^^^^^^
        "]]
        .assert_eq(&reported(
            "external def upper(s: str) -> str { return s }\ndef lower(s: str) -> str\n",
        ));

        expect![[r"
            error: function is missing its body
             --> test.yz:5:23
              |
            5 | impl Show for int64 { def show(x: Self) -> str }
              |                       ^^^^^^^^^^^^^^^^^^^^^^^^
        "]]
        .assert_eq(&reported(
            "trait Show {\n    def show(x: Self) -> str\n}\n\nimpl Show for int64 { def show(x: Self) -> str }\n",
        ));
    }

    #[test]
    fn reports_an_unknown_type() {
        expect![[r"
            error: unknown type `Nope`
             --> test.yz:1:10
              |
            1 | def f(x: Nope) -> int64 { return 1 }
              |          ^^^^
        "]]
        .assert_eq(&reported("def f(x: Nope) -> int64 { return 1 }\n"));
    }

    #[test]
    fn a_unit_function_may_end_without_a_return() {
        expect![[r#"
            module {
              yzl.fn @g params ["x"] (!yz.int64) -> !yz.unit {
              ^bb0(%arg0: !yzl.unresolved):
                %0 = yzl.local "x" param : !yzl.ref<!yz.int64>
                yzl.store %0, %arg0 : !yzl.ref<!yz.int64>, !yzl.unresolved
                %1 = yzl.load %0 : !yzl.ref<!yz.int64> -> !yzl.unresolved
                %2 = yzl.local "y" : !yzl.ref<!yzl.unresolved>
                yzl.store %2, %1 : !yzl.ref<!yzl.unresolved>, !yzl.unresolved
                yzl.return
              } {sym_visibility = "private"}
            }
        "#]].assert_eq(&lowered("def g(x: int64) {\n    let y = x\n}\n"));
    }

    #[test]
    fn a_function_with_a_result_ends_in_a_return() {
        expect![[r"
            error: `f` must end with a `return` of `int64`
             --> test.yz:1:1
              |
            1 | def f(x: int64) -> int64 {
              | ^^^^^^^^^^^^^^^^^^^^^^^^^^
        "]].assert_eq(&reported("def f(x: int64) -> int64 {\n    let y = x\n}\n"));
    }

    #[test]
    fn nothing_follows_a_return() {
        expect![[r"
            error: nothing can follow a `return`
             --> test.yz:3:5
              |
            3 |     let y = x
              |     ^^^^^^^^^
        "]].assert_eq(&reported(
            "def f(x: int64) -> int64 {\n    return x\n    let y = x\n}\n",
        ));
    }
}
