use melior::ir::attribute::{
    ArrayAttribute, FlatSymbolRefAttribute, StringAttribute, TypeAttribute,
};
use melior::ir::r#type::FunctionType;
use melior::ir::{
    Attribute, Block, BlockLike, BlockRef, Location, Operation, Region, RegionLike, Type, Value,
};
use yuzu_ast::ast::Mutability;
use yuzu_ast::{AstNode, Visibility, ast};
use yuzu_mlir::attributes::CalleeSource;
use yuzu_mlir::ir::attribute::array::ArrayAttributeExt;
use yuzu_mlir::ir::operation::{OperationExt, OperationMutExt};
use yuzu_mlir::ods::yzl;
use yuzu_mlir::types::{self, RefType, UnresolvedType};
use yuzu_mlir::{ListType, ParamType, StructType};

use crate::lower_ast_to_yzl::symbols::{
    Binding, BindingKind, Declared, FunctionKind, Lookup, ModulePath, Reference, Row,
};
use crate::lower_ast_to_yzl::{AstToYzl, Locals};

/// Where a `fn` is written, which decides the generics its surroundings
/// supply and whether it needs a body.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Site {
    AtModule,
    InTrait,
    InImpl,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LocalKind {
    Let(Mutability),
    Param,
}

impl Site {
    fn generics(self) -> &'static [&'static str] {
        match self {
            Site::AtModule => &[],
            Site::InTrait | Site::InImpl => &["Self"],
        }
    }
}

impl<'c, 'd> AstToYzl<'c, 'd> {
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
        if !self.check_at_file_level(decl, "declarations inside functions are not supported yet") {
            return;
        }

        let Some(name) = self.read_ident(decl.name()) else {
            self.report(decl, "struct is missing its name");
            return;
        };

        if !self.is_bound(decl, name) {
            debug_assert!(
                self.diagnostics.has_errors(),
                "`{name}` was not bound by the hoist and nothing reported why"
            );
            return;
        }

        let name = self.symbols.symbol_here(name);
        let loc = self.location(decl);
        self.emit_struct(block, name, decl.fields(), decl.visibility(), loc);
    }

    fn convert_table<'a>(&mut self, block: BlockRef<'c, 'a>, decl: &ast::TableStmt) {
        if !self.check_at_file_level(decl, "declarations inside functions are not supported yet") {
            return;
        }

        let Some(name) = self.read_ident(decl.name()) else {
            self.report(decl, "table is missing its name");
            return;
        };

        if !self.is_bound(decl, name) {
            debug_assert!(
                self.diagnostics.has_errors(),
                "`{name}` was not bound by the hoist and nothing reported why"
            );
            return;
        }

        // An imported struct is held under the module that declared it,
        // whatever an `as` renamed it to here.
        let row = match self.read_ident(decl.struct_name()) {
            Some(struct_name) => match self.symbols.struct_symbol(struct_name) {
                Some(symbol) => symbol,
                None => {
                    debug_assert!(
                        self.diagnostics.has_errors(),
                        "`{struct_name}` passed the hoist as a struct and is not one now"
                    );
                    return;
                }
            },
            None => {
                let symbol = self
                    .symbols
                    .symbol_here(self.symbols.intern(&format!("{name}_row")));

                let loc = self.location(decl);
                self.emit_struct(block, symbol, decl.inline_fields(), decl.visibility(), loc);

                symbol
            }
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
        if !self.check_at_file_level(decl, "declarations inside functions are not supported yet") {
            return;
        }

        // `convert_method` reports a missing name.
        if let Some(name) = self.read_ident(decl.name())
            && !self.is_bound(decl, name)
        {
            debug_assert!(
                self.diagnostics.has_errors(),
                "`{name}` was not bound by the hoist and nothing reported why"
            );
            return;
        }

        self.convert_method(block, decl, Site::AtModule);
    }

    fn convert_method<'a>(&mut self, block: BlockRef<'c, 'a>, decl: &ast::FuncStmt, site: Site) {
        let loc = self.location(decl);

        let Some(name) = self.read_ident(decl.name()) else {
            self.report(decl, "function is missing its name");
            return;
        };

        match (decl.is_external(), decl.body().is_some()) {
            (true, true) => self.report(decl, "an external function cannot have a body"),
            (false, false) if site != Site::InTrait => {
                self.report(decl, "function is missing its body");
            }
            _ => {}
        }

        let mut generics: Vec<&'c str> = site.generics().to_vec();
        generics.extend(
            decl.type_params()
                .filter_map(|param| self.read_ident(param.name())),
        );

        self.symbols.enter_type_params(generics.clone());

        let mut param_names = Vec::new();
        let mut param_types = Vec::new();
        let mut param_count = 0;
        let mut has_error = false;
        for param in decl.params() {
            let Some(name) = self.read_ident(param.name()) else {
                self.report(&param, "parameter is missing its name");
                has_error = true;
                continue;
            };

            let ty = match param.ty() {
                Some(ty) => self.read_type_annotation(ty),
                None => {
                    self.report(&param, "parameter is missing its type");
                    has_error = true;
                    continue;
                }
            };

            param_names.push(name);
            param_types.push(ty);
            param_count += 1;
        }

        if has_error {
            return;
        }

        let result = match decl.result() {
            Some(result) => self.read_type_annotation(result),
            None => UnresolvedType::get(self.context),
        };

        let signature = FunctionType::new(self.context, &param_types, &[result]);

        let (bound_params, bound_traits) = self.read_bounds(decl);

        let external_name = decl.is_external().then_some(name);

        // A trait or an implementation is a symbol table of its own, so a
        // method keeps its bare name.
        let name = match site {
            Site::AtModule => self.symbols.symbol_here(name),
            Site::InTrait | Site::InImpl => name,
        };

        let body = Region::new();
        if let Some(block) = decl.body() {
            let ty = UnresolvedType::get(self.context);
            let arguments = vec![(ty, loc); param_count];
            let entry = body.append_block(Block::new(&arguments));

            let mut locals = Locals::new();
            self.symbols.enter_block();
            for (index, name) in param_names.iter().enumerate() {
                let argument = entry
                    .argument(index)
                    .expect("the entry block has one argument per parameter")
                    .into();

                let place = self.emit_local(entry, name, LocalKind::Param, argument, loc);
                self.bind_local(&mut locals, name, place);
            }

            self.convert_block(entry, &mut locals, &block);
            self.symbols.leave();
        }

        let mut builder = yzl::FnOperationBuilder::new(self.context, loc)
            .sym_name(StringAttribute::new(self.context, name))
            .params(ArrayAttribute::from_strings(self.context, param_names))
            .signature(TypeAttribute::new(signature.into()))
            .body(body);

        if decl.is_agg() {
            builder = builder.is_agg(Attribute::unit(self.context));
        }

        if let Some(external_name) = &external_name {
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

        self.symbols.leave();
        block.append_operation(op);
    }

    fn convert_trait<'a>(&mut self, block: BlockRef<'c, 'a>, decl: &ast::TraitStmt) {
        if !self.check_at_file_level(decl, "declarations inside functions are not supported yet") {
            return;
        }

        let Some(name) = self.read_ident(decl.name()) else {
            self.report(decl, "trait is missing its name");
            return;
        };

        if !self.is_bound(decl, name) {
            debug_assert!(
                self.diagnostics.has_errors(),
                "`{name}` was not bound by the hoist and nothing reported why"
            );
            return;
        }

        let region = Region::new();
        let body = region.append_block(Block::new(&[]));
        for method in decl.methods() {
            self.convert_method(body, &method, Site::InTrait);
        }

        let name = self.symbols.symbol_here(name);
        let mut r#trait: Operation<'c> =
            yzl::TraitOperationBuilder::new(self.context, self.location(decl))
                .sym_name(StringAttribute::new(self.context, name))
                .body(region)
                .build()
                .into();

        if decl.visibility() != Visibility::Public {
            r#trait.set_private(self.context);
        }

        block.append_operation(r#trait);
    }

    fn convert_impl<'a>(&mut self, block: BlockRef<'c, 'a>, decl: &ast::ImplStmt) {
        if !self.check_at_file_level(decl, "declarations inside functions are not supported yet") {
            return;
        }

        let Some(trait_name) = decl
            .trait_()
            .and_then(|trait_ref| self.read_ident(trait_ref.name()))
        else {
            self.report(decl, "`impl` is missing its trait");
            return;
        };

        let Some(target) = self.read_ident(decl.ty()) else {
            self.report(decl, "`impl` is missing its type name");
            return;
        };

        let trait_name = match self.symbols.trait_symbol(trait_name) {
            Some(symbol) => symbol,
            None => {
                self.report(decl, &format!("unknown trait `{trait_name}`"));
                trait_name
            }
        };

        let target = match self.symbols.struct_symbol(target) {
            Some(symbol) => symbol,
            None => {
                if types::scalar(self.context, target).is_none() {
                    self.report(decl, &format!("unknown type `{target}`"));
                }

                target
            }
        };

        let region = Region::new();
        let body = region.append_block(Block::new(&[]));
        for method in decl.methods() {
            self.convert_method(body, &method, Site::InImpl);
        }

        block.append_operation(
            yzl::ImplOperationBuilder::new(self.context, self.location(decl))
                .r#trait(FlatSymbolRefAttribute::new(self.context, trait_name))
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
            self.report(decl, "let binding is missing its name");
            return;
        };

        let Some(expr) = decl.expr() else {
            self.report(decl, "let binding is missing its expression");
            return;
        };

        if self.symbols.in_body() {
            let kind = LocalKind::Let(decl.mutability());
            let value = self.convert_expr(block, locals, &expr);
            let place = self.emit_local(block, name, kind, value, loc);
            self.bind_local(locals, name, place);
            return;
        }

        // A file-level binding is copied into each use, so nothing could see
        // a change to it.
        if let Some(token) = decl.mut_token() {
            let diagnostic = self
                .error_at(token.text_range(), "a file-level binding cannot be `mut`")
                .note("a file-level binding is a constant; move it into a function to change it");

            self.diagnostics.emit(diagnostic);
        }

        if !self.is_bound(decl, name) {
            debug_assert!(
                self.diagnostics.has_errors(),
                "`{name}` was not bound by the hoist and nothing reported why"
            );
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

        let target = assign.target().and_then(|target| match target {
            ast::Expr::IdentExpr(ident) => self.read_ident(ident.name()),
            _ => None,
        });

        let Some(name) = target else {
            self.report(assign, "assignment is missing its target");
            return;
        };

        let Some(value) = assign.value() else {
            self.report(assign, "assignment is missing its value");
            return;
        };

        let Some(slot) = self.read_place(assign, name) else {
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
        self.symbols.enter_block();
        for stmt in body.stmts() {
            self.convert_stmt(block, locals, &stmt);
        }

        self.symbols.leave();
    }

    fn convert_expr_stmt<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        locals: &mut Locals<'c, 'a>,
        stmt: &ast::ExprStmt,
    ) {
        let Some(expr) = stmt.expr() else {
            return;
        };

        // The program's query is the entry file's.
        let is_query = matches!(expr, ast::Expr::Pipeline(_));
        if is_query && !self.symbols.in_body() && !self.symbols.module().is_entry() {
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

        let fields = decl
            .fields()
            .filter_map(|f| self.read_ident(f.name()))
            .collect();
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
            Row::from(
                decl.inline_fields()
                    .filter_map(|field| self.read_ident(field.name()))
                    .collect::<Vec<_>>(),
            )
        } else {
            let Some(declared) = self.read_ident(decl.struct_name()) else {
                return;
            };

            // The struct may be imported, and an import names where it was
            // written rather than repeating it.
            match self.symbols.kind(declared) {
                Some(BindingKind::Struct { fields, .. }) => Row::from(fields.clone()),
                _ => {
                    self.report(decl, &format!("`{declared}` is not a struct"));
                    return;
                }
            }
        };

        self.bind_or_report(decl, name, BindingKind::Relation { row }, decl.visibility());
    }

    fn hoist_trait(&mut self, decl: &ast::TraitStmt) {
        let Some(name) = self.read_ident(decl.name()) else {
            return;
        };

        let methods = decl
            .methods()
            .filter_map(|m| self.read_ident(m.name()))
            .collect();
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

        self.bind_or_report(
            decl,
            name,
            BindingKind::Func {
                source,
                kind,
                arity: decl.params().count(),
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
                    let Some(path) = import.path().map(|path| self.read_path(&path)) else {
                        continue;
                    };

                    for item in import.items() {
                        self.bind_import(path, &item);
                    }
                }
                ast::Stmt::ImportStmt(import) => {
                    let Some(path) = import.path().map(|path| self.read_path(&path)) else {
                        continue;
                    };

                    let last = path
                        .rsplit('.')
                        .next()
                        .expect("a split yields at least one piece");

                    let name = self.read_ident(import.alias()).unwrap_or(last);
                    self.bind_or_report(
                        import,
                        name,
                        BindingKind::Module { path },
                        Visibility::Private,
                    );
                }
                ast::Stmt::ModStmt(decl) => {
                    let Some(name) = self.read_ident(decl.name()) else {
                        self.report(decl, "module declaration is missing its name");
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

    fn bind_import(&mut self, path: &'c str, item: &ast::ImportItem) {
        let Some(name) = self.read_ident(item.name()) else {
            self.report(item, "import item is missing its name");
            return;
        };

        let Some((from, exported)) = self.read_export(item, path, name) else {
            return;
        };

        let local = self.read_ident(item.alias()).unwrap_or(name);

        if self.symbols.binding(local).is_some() {
            self.check_duplicate(item, exported.kind.name(), local);
            return;
        }

        self.symbols.bind(
            local,
            Binding {
                kind: BindingKind::Import { from },
                text_range: item.syntax().text_range(),
                visibility: exported.visibility,
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

    /// Binds a name to a place the body has declared.
    fn bind_local<'a>(&mut self, locals: &mut Locals<'c, 'a>, name: &'c str, place: Value<'c, 'a>) {
        self.symbols.bind_local(name, locals.len());
        locals.push(place);
    }

    pub(super) fn read_export(
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

        let Some(binding) = self
            .symbols
            .find_declared(module, name)
            .map(|(_, b)| b.clone())
        else {
            self.report(at, &format!("`{path}` does not declare `{name}`"));
            return None;
        };

        if binding.visibility != Visibility::Public {
            self.report(
                at,
                &format!("`{name}` is not public; `{path}` keeps it to itself"),
            );
            return None;
        }

        // A module may pass on what it imported, so the origin is followed
        // to the file that wrote it.
        let (at, origin) = self
            .symbols
            .find_declared(module, name)
            .expect("an import points at a declaration that exists");

        Some((at, origin.clone()))
    }

    /// One entry per `(parameter, trait)` pair.
    fn read_bounds(&mut self, decl: &ast::FuncStmt) -> (Vec<&'c str>, Vec<&'c str>) {
        let mut subjects = Vec::new();
        let mut traits = Vec::new();
        for bound in decl.bounds() {
            let Some(subject) = self.read_ident(bound.subject()) else {
                self.report(&bound, "type bound is missing its subject");
                continue;
            };

            if !self.symbols.is_type_param(subject) {
                self.report(&bound, &format!("unknown type parameter `{subject}`"));
            }

            for trait_ref in bound.traits() {
                let Some(name) = self.read_ident(trait_ref.name()) else {
                    self.report(&trait_ref, "trait reference is missing its name");
                    continue;
                };

                if self.symbols.trait_symbol(name).is_none() {
                    self.report(&trait_ref, &format!("unknown trait `{name}`"));
                }

                subjects.push(subject);
                traits.push(name);
            }
        }

        (subjects, traits)
    }

    fn read_type_annotation(&mut self, annotation: ast::TypeAnnotation) -> Type<'c> {
        let named = match annotation {
            ast::TypeAnnotation::NamedTypeAnnotation(named) => named,
            ast::TypeAnnotation::FuncTypeAnnotation(func) => {
                self.report(&func, "function types are not supported yet");
                return UnresolvedType::get(self.context);
            }
        };

        let Some(name) = self.read_ident(named.name()) else {
            self.report(&named, "type is missing its name");
            return UnresolvedType::get(self.context);
        };

        if self.symbols.is_type_param(name) {
            return ParamType::new(self.context, name).into();
        }

        if name == "List" {
            let mut args = named.args();
            let (Some(inner), None) = (args.next(), args.next()) else {
                self.report(&named, "`List` takes exactly one type argument");
                return UnresolvedType::get(self.context);
            };

            let inner = self.read_type_annotation(inner);
            return ListType::new(self.context, inner).into();
        }

        if let Some(scalar) = types::scalar(self.context, name) {
            return scalar;
        }

        if let Some(symbol) = self.symbols.struct_symbol(name) {
            return StructType::new(self.context, symbol).into();
        }

        self.report(&named, &format!("unknown type `{name}`"));
        UnresolvedType::get(self.context)
    }

    fn read_path(&self, path: &ast::ModulePath) -> &'c str {
        self.symbols
            .intern(&path.segments_text().collect::<Vec<_>>().join("."))
    }

    /// The slot of the place an assignment writes, when the name is one.
    fn read_place(&mut self, node: &impl AstNode, name: &str) -> Option<usize> {
        let message = match self.symbols.lookup(Reference::unqualified(name)) {
            Lookup::Local(slot) => return Some(slot),
            Lookup::Column(_) | Lookup::Ambiguous | Lookup::NarrowedAway => {
                format!("`{name}` is a column; `set` is how a query writes one")
            }
            Lookup::Let(_) => {
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
    fn emit_local<'a>(
        &self,
        block: BlockRef<'c, 'a>,
        name: &str,
        kind: LocalKind,
        value: Value<'c, 'a>,
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        let builder = yzl::LocalOperationBuilder::new(self.context, loc)
            .place(RefType::get(self.context))
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

    fn emit_struct<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        symbol: &str,
        fields: impl Iterator<Item = ast::StructField>,
        visibility: Visibility,
        loc: Location<'c>,
    ) {
        let mut names = Vec::new();
        let mut types = Vec::new();

        for field in fields {
            let (Some(name), Some(ty)) = (self.read_ident(field.name()), field.ty()) else {
                self.report(&field, "struct field is incomplete");
                continue;
            };

            names.push(name);
            types.push(self.read_type_annotation(ty));
        }

        let mut r#struct: Operation<'c> = yzl::r#struct(
            self.context,
            StringAttribute::new(self.context, symbol),
            ArrayAttribute::from_strings(self.context, &names),
            ArrayAttribute::from_types(self.context, types),
            loc,
        )
        .into();

        if visibility != Visibility::Public {
            r#struct.set_private(self.context);
        }

        block.append_operation(r#struct);
    }

    fn check_at_file_level(&mut self, node: &impl AstNode, message: &str) -> bool {
        if self.symbols.in_body() {
            self.report(node, message);
            return false;
        }

        true
    }

    fn check_in_body(&mut self, node: &impl AstNode, message: &str) -> bool {
        if !self.symbols.in_body() {
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

        let other = self.text_at_range(text_range);
        let diagnostic = self
            .error_at(
                node.syntax().text_range(),
                &format!("the {what} `{name}` is already defined"),
            )
            .note(format!("also declared at {other}"));

        self.diagnostics.emit(diagnostic);
    }

    fn is_bound(&self, node: &impl AstNode, name: &str) -> bool {
        self.symbols
            .binding(name)
            .is_some_and(|binding| binding.text_range == node.syntax().text_range())
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::test_support::{lower, lowered, reported};

    #[test]
    fn a_file_level_let_cannot_be_mut() {
        expect![[r#"
            error: a file-level binding cannot be `mut`
             --> test.yz:4:5
              |
            4 | let mut cap = 1
              |     ^^^
              = note: a file-level binding is a constant; move it into a function to change it
        "#]]
        .assert_eq(&reported(
            "struct Row { a: int64 }\ntable t = Row\n\nlet mut cap = 1\n\nfrom t |> select a + cap as v\n",
        ));
    }

    #[test]
    fn a_body_statement_at_the_file_level_is_reported() {
        expect![[r#"
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
        "#]]
        .assert_eq(&reported(
            "struct Row { a: int64 }\ntable t = Row\n\nreturn 1\nx = 2\n\nfrom t |> select a as v\n",
        ));
    }

    #[test]
    fn an_import_in_a_body_is_reported() {
        expect![[r#"
            error: this belongs at the top of the file
             --> test.yz:5:3
              |
            5 |   import helpers
              |   ^^^^^^^^^^^^^^
        "#]]
        .assert_eq(&reported(
            "struct Row { a: int64 }\ntable t = Row\n\ndef f(x: int64) -> int64 {\n  import helpers\n  return x\n}\n\nfrom t |> select f(a) as v\n",
        ));
    }

    #[test]
    fn a_module_level_let_cannot_take_a_name_again() {
        expect![[r#"
            error: the binding `cap` is already defined
             --> test.yz:5:1
              |
            5 | let cap = 2
              | ^^^^^^^^^^^
              = note: also declared at test.yz:4:1
        "#]]
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
                %0 = yzl.local "x" param
                yzl.store %0, %arg0 : !yzl.unresolved
                %1 = yzl.load %0 : !yzl.unresolved
                %2 = yz.constant_int 2
                %3 = yz.mul %1, %2 : !yzl.unresolved, !yz.int64 -> !yzl.unresolved
                %4 = yzl.local "doubled"
                yzl.store %4, %3 : !yzl.unresolved
                %5 = yzl.load %4 : !yzl.unresolved
                %6 = yz.constant_int 1
                %7 = yz.add %5, %6 : !yzl.unresolved, !yz.int64 -> !yzl.unresolved
                yzl.return %7 : !yzl.unresolved
              } {sym_visibility = "private"}
              yzl.fn @spread params ["x"] (!yz.int64) -> !yz.int64 agg {
              ^bb0(%arg0: !yzl.unresolved):
                %0 = yzl.local "x" param
                yzl.store %0, %arg0 : !yzl.unresolved
                %1 = yzl.load %0 : !yzl.unresolved
                %2 = yzl.call @max(%1) : (!yzl.unresolved) -> !yzl.unresolved {callee_source = "builtin", is_agg}
                %3 = yzl.load %0 : !yzl.unresolved
                %4 = yzl.call @min(%3) : (!yzl.unresolved) -> !yzl.unresolved {callee_source = "builtin", is_agg}
                %5 = yz.sub %2, %4 : !yzl.unresolved, !yzl.unresolved -> !yzl.unresolved
                yzl.return %5 : !yzl.unresolved
              } {sym_visibility = "private"}
              yzl.fn @upper params ["s"] (!yz.str) -> !yz.str external "upper" {
              } {sym_visibility = "private"}
            }
        "#]]
        .assert_eq(&lowered(
            r#"
def f(x: int64) -> int64 {
    let doubled = x * 2
    return doubled + 1
}

agg def spread(x: int64) -> int64 {
    return max(x) - min(x)
}

external def upper(s: str) -> str
"#,
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
                %1 = yzl.local "x" param
                yzl.store %1, %arg0 : !yzl.unresolved
                %2 = yzl.load %1 : !yzl.unresolved
                %3 = yz.constant_int 1
                %4 = yz.add %2, %3 : !yzl.unresolved, !yz.int64 -> !yzl.unresolved
                %5 = yzl.load %1 : !yzl.unresolved
                %6 = yz.constant_int 2
                %7 = yz.mul %5, %6 : !yzl.unresolved, !yz.int64 -> !yzl.unresolved
                %8 = yzl.local "y"
                yzl.store %8, %7 : !yzl.unresolved
                %9 = yzl.load %8 : !yzl.unresolved
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
                  %0 = yzl.local "x" param
                  yzl.store %0, %arg0 : !yzl.unresolved
                  %1 = yzl.local "y" param
                  yzl.store %1, %arg1 : !yzl.unresolved
                  %2 = yzl.load %0 : !yzl.unresolved
                  %3 = yzl.load %1 : !yzl.unresolved
                  %4 = yz.add %2, %3 : !yzl.unresolved, !yzl.unresolved -> !yzl.unresolved
                  yzl.return %4 : !yzl.unresolved
                }
              }
              yzl.fn @id generics ["T"] where ["T"] : [@Add] params ["x"] (!yzl.param<"T">) -> !yzl.param<"T"> {
              ^bb0(%arg0: !yzl.unresolved):
                %0 = yzl.local "x" param
                yzl.store %0, %arg0 : !yzl.unresolved
                %1 = yzl.load %0 : !yzl.unresolved
                yzl.return %1 : !yzl.unresolved
              } {sym_visibility = "private"}
            }
        "#]].assert_eq(&lowered(
        "trait Add {\n    def add(x: Self, y: Self) -> Self\n}\n\nimpl Add for int64 {\n    def add(x: int64, y: int64) -> int64 { return x + y }\n}\n\ndef id[T](x: T) -> T where T: Add { return x }\n",
    ));
    }

    #[test]
    fn reports_unsupported_constructs() {
        expect![[r#"
        error: declarations inside functions are not supported yet
         --> test.yz:2:5
          |
        2 |     struct S { a: int64 }
          |     ^^^^^^^^^^^^^^^^^^^^^
    "#]]
        .assert_eq(&reported(
            "def f(x: int64) -> int64 {\n    struct S { a: int64 }\n    return x\n}\n",
        ));
    }

    #[test]
    fn converts_an_annotated_let() {
        expect![[r#"
            module {
              yzl.const @ids : !yz.list<!yz.int64> {
                %0 = yz.constant_int 1
                %1 = yz.constant_int 3
                %2 = yzl.list[%0, %1] : (!yz.int64, !yz.int64) -> !yzl.unresolved
                yzl.yield %2 : !yzl.unresolved
              } {sym_visibility = "private"}
            }
        "#]]
        .assert_eq(&lowered("let ids: List[int64] = [1, 3]\n"));
    }

    #[test]
    fn a_redeclaration_says_where_the_other_one_is() {
        expect![[r#"
            error: the relation `Row` is already defined
             --> test.yz:3:1
              |
            3 | table Row = Row
              | ^^^^^^^^^^^^^^^
              = note: also declared at test.yz:1:1
        "#]]
        .assert_eq(&reported("struct Row { a: int64 }\n\ntable Row = Row\n"));
    }

    #[test]
    fn a_redeclaration_leaves_the_first_one_standing() {
        expect![[r#"
            error: the struct `Row` is already defined
             --> test.yz:2:1
              |
            2 | struct Row { b: int64 }
              | ^^^^^^^^^^^^^^^^^^^^^^^
              = note: also declared at test.yz:1:1
        "#]]
        .assert_eq(&reported(
            "struct Row { a: int64 }\nstruct Row { b: int64 }\ntable t = Row\n\nfrom t\n|> select a as x\n",
        ));
    }

    #[test]
    fn an_import_of_an_unloaded_module_is_reported() {
        expect![[r#"
            error: `helpers` is not a module this file reads
             --> test.yz:1:21
              |
            1 | from helpers import spread
              |                     ^^^^^^
        "#]]
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
        expect![[r#"
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
        "#]]
        .assert_eq(&reported(
            "external def upper(s: str) -> str { return s }\ndef lower(s: str) -> str\n",
        ));

        expect![[r#"
            error: function is missing its body
             --> test.yz:5:23
              |
            5 | impl Show for int64 { def show(x: Self) -> str }
              |                       ^^^^^^^^^^^^^^^^^^^^^^^^
        "#]]
        .assert_eq(&reported(
            "trait Show {\n    def show(x: Self) -> str\n}\n\nimpl Show for int64 { def show(x: Self) -> str }\n",
        ));
    }

    #[test]
    fn reports_an_unknown_type() {
        expect![[r#"
            error: unknown type `Nope`
             --> test.yz:1:10
              |
            1 | def f(x: Nope) -> int64 { return 1 }
              |          ^^^^
        "#]]
        .assert_eq(&reported("def f(x: Nope) -> int64 { return 1 }\n"));
    }
}
