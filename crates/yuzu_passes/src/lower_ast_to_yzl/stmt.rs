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
use yuzu_mlir::ext::{ArrayAttributeExt, OperationMutExt};
use yuzu_mlir::ods::yzl;
use yuzu_mlir::types::{self, VarType};
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

impl Site {
    fn generics(self) -> &'static [&'static str] {
        match self {
            Site::AtModule => &[],
            Site::InTrait | Site::InImpl => &["Self"],
        }
    }
}

impl<'c, 'd> AstToYzl<'c, 'd> {
    /// Imports bind before the file's own declarations, so the two collide
    /// the way two declarations do.
    pub(super) fn bind_imports(&mut self, root: &ast::Root) {
        for stmt in root.stmts() {
            match &stmt {
                ast::Stmt::FromImportStmt(import) => {
                    let Some(path) = import.path().map(|path| self.path_text(&path)) else {
                        continue;
                    };

                    for item in import.items() {
                        self.bind_import(&path, &item);
                    }
                }
                ast::Stmt::ImportStmt(import) => {
                    let Some(path) = import.path().map(|path| self.path_text(&path)) else {
                        continue;
                    };

                    let last = path
                        .rsplit('.')
                        .next()
                        .expect("a split yields at least one piece");

                    let name = import.alias_text().unwrap_or(last.to_string());
                    self.bind_or_report(
                        import,
                        &name,
                        BindingKind::Module { path },
                        Visibility::Private,
                    );
                }
                ast::Stmt::ModStmt(decl) => {
                    let Some(name) = decl.name_text() else {
                        self.report(decl, "module declaration is missing its name");
                        continue;
                    };

                    let path = self.symbols.module().qualify(&name).unwrap_or(name.clone());
                    self.bind_or_report(
                        decl,
                        &name,
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

    /// Converts any statement. Each statement checks that it is valid where
    /// the walk is, so this dispatches without conditions.
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

    /// Declarations and imports belong to the file.
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

    /// The first walk binds imports and modules, so the second has nothing
    /// to build for them.
    fn convert_import(&mut self, node: &impl AstNode) {
        self.check_at_file_level(node, "this belongs at the top of the file");
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
        let is_query = matches!(expr, ast::Expr::Rel(_));
        if is_query && !self.symbols.in_body() && !self.symbols.module().is_entry() {
            self.report(stmt, "a module cannot hold a query");
            return;
        }

        self.convert_expr(block, locals, &expr);
    }

    /// `pub(mod)` reaches the enclosing module's files. Until a module holds
    /// more than one file, nobody asking from outside is one of them.
    pub(super) fn read_export(
        &mut self,
        at: &impl AstNode,
        path: &str,
        name: &str,
    ) -> Option<(Declared, Binding)> {
        let module = ModulePath::from_path(path);
        if !self.symbols.contains_module(&module) {
            self.report(at, &format!("`{path}` is not a module this file reads"));
            return None;
        }

        let Some(binding) = self
            .symbols
            .find_in(module.declares(name))
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
            .find_in(module.declares(name))
            .expect("an import points at a declaration that exists");

        Some((at, origin.clone()))
    }

    fn bind_import(&mut self, path: &str, item: &ast::ImportItem) {
        let Some(name) = item.name_text() else {
            self.report(item, "import item is missing its name");
            return;
        };

        let Some((from, exported)) = self.read_export(item, path, &name) else {
            return;
        };

        let local = item.alias_text().unwrap_or(name);
        self.declare_imported(item, &local, &exported, from);
    }

    /// Binds where the declaration was written rather than a copy of what it
    /// says: a `let` says nothing yet during this walk.
    fn declare_imported(
        &mut self,
        node: &impl AstNode,
        local: &str,
        exported: &Binding,
        from: Declared,
    ) {
        if self.symbols.binding(local).is_some() {
            self.check_duplicate(node, exported.kind.name(), local);
            return;
        }

        self.symbols.bind(
            local,
            Binding {
                kind: BindingKind::Import { from },
                declared: node.syntax().text_range(),
                visibility: exported.visibility,
            },
        );
    }

    fn hoist_struct(&mut self, decl: &ast::StructStmt) {
        let Some(name) = decl.name_text() else {
            return;
        };

        let fields = decl.fields().filter_map(|f| f.name_text()).collect();
        self.bind_or_report(
            decl,
            &name,
            BindingKind::Struct { fields },
            decl.visibility(),
        );
    }

    fn hoist_table(&mut self, decl: &ast::TableStmt) {
        let Some(name) = decl.name_text() else {
            return;
        };

        let row = if decl.inline_fields().next().is_some() {
            Row::from(
                decl.inline_fields()
                    .filter_map(|field| field.name_text())
                    .collect::<Vec<_>>(),
            )
        } else {
            let Some(declared) = decl.struct_name_text() else {
                return;
            };

            // The struct may be imported, and an import names where it was
            // written rather than repeating it.
            match self.symbols.kind(&declared) {
                Some(BindingKind::Struct { fields, .. }) => Row::from(fields.clone()),
                _ => {
                    self.report(decl, &format!("`{declared}` is not a struct"));
                    return;
                }
            }
        };

        self.bind_or_report(
            decl,
            &name,
            BindingKind::Relation { row },
            decl.visibility(),
        );
    }

    fn hoist_trait(&mut self, decl: &ast::TraitStmt) {
        let Some(name) = decl.name_text() else {
            return;
        };

        let methods = decl.methods().filter_map(|m| m.name_text()).collect();
        self.bind_or_report(
            decl,
            &name,
            BindingKind::Trait { methods },
            decl.visibility(),
        );
    }

    fn hoist_fn(&mut self, decl: &ast::FuncStmt) {
        let Some(name) = decl.name_text() else {
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
            &name,
            BindingKind::Func {
                source,
                kind,
                arity: decl.params().count(),
            },
            decl.visibility(),
        );
    }

    /// Registers the name and who may see it, which is all an importer
    /// needs. What the `let` is bound to is the second walk's answer.
    fn hoist_let(&mut self, decl: &ast::LetStmt) {
        let Some(name) = decl.name_text() else {
            return;
        };

        self.bind_or_report(decl, &name, BindingKind::Pending, decl.visibility());
    }

    fn bind_or_report(
        &mut self,
        node: &impl AstNode,
        name: &str,
        kind: BindingKind,
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
                declared: node.syntax().text_range(),
                visibility,
            },
        );
    }

    fn check_duplicate(&mut self, node: &impl AstNode, what: &str, name: &str) {
        let site = self
            .symbols
            .binding(name)
            .expect("a duplicate is checked against a bound name")
            .declared;

        let other = self.position_text(site);
        let diagnostic = self
            .error_at(
                node.syntax().text_range(),
                &format!("the {what} `{name}` is already defined"),
            )
            .note(format!("also declared at {other}"));

        self.diagnostics.emit(diagnostic);
    }

    fn convert_struct<'a>(&mut self, block: BlockRef<'c, 'a>, decl: &ast::StructStmt) {
        if !self.check_at_file_level(decl, "declarations inside functions are not supported yet") {
            return;
        }

        let Some(name) = decl.name_text() else {
            self.report(decl, "struct is missing its name");
            return;
        };

        if !self.is_bound(decl, &name) {
            debug_assert!(
                self.diagnostics.has_errors(),
                "`{name}` was not bound by the hoist and nothing reported why"
            );
            return;
        }

        let name = self.symbols.module().declares(&name).symbol();
        self.emit_struct(
            block,
            &name,
            decl.fields(),
            decl.visibility(),
            self.location(decl),
        );
    }

    fn convert_table<'a>(&mut self, block: BlockRef<'c, 'a>, decl: &ast::TableStmt) {
        if !self.check_at_file_level(decl, "declarations inside functions are not supported yet") {
            return;
        }

        let Some(name) = decl.name_text() else {
            self.report(decl, "table is missing its name");
            return;
        };

        if !self.is_bound(decl, &name) {
            debug_assert!(
                self.diagnostics.has_errors(),
                "`{name}` was not bound by the hoist and nothing reported why"
            );
            return;
        }

        // An imported struct is held under the module that declared it,
        // whatever an `as` renamed it to here.
        let row = match decl.struct_name_text() {
            Some(struct_name) => match self.symbols.struct_symbol(&struct_name) {
                Some(symbol) => symbol.to_string(),
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
                    .module()
                    .declares(&format!("{name}_row"))
                    .symbol();
                self.emit_struct(
                    block,
                    &symbol,
                    decl.inline_fields(),
                    decl.visibility(),
                    self.location(decl),
                );
                symbol
            }
        };

        let name = self.symbols.module().declares(&name).symbol();
        let mut table: Operation<'c> = yzl::table(
            self.context,
            StringAttribute::new(self.context, &name),
            FlatSymbolRefAttribute::new(self.context, &row),
            self.location(decl),
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
        if let Some(name) = decl.name_text()
            && !self.is_bound(decl, &name)
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
        let location = self.location(decl);

        let Some(name) = decl.name_text() else {
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

        let generics: Vec<String> = site
            .generics()
            .iter()
            .map(|generic| generic.to_string())
            .chain(decl.type_params().filter_map(|param| param.name_text()))
            .collect();

        self.symbols.enter_type_params(generics.clone());

        let mut param_names = Vec::new();
        let mut param_types = Vec::new();
        let mut param_count = 0;
        let mut has_error = false;
        for param in decl.params() {
            let Some(name) = param.name_text() else {
                self.report(&param, "parameter is missing its name");
                has_error = true;
                continue;
            };

            let ty = match param.ty() {
                Some(ty) => self.read_type(ty),
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
            Some(result) => self.read_type(result),
            None => VarType::get(self.context),
        };

        let signature = FunctionType::new(self.context, &param_types, &[result]);

        let (bound_params, bound_traits) = self.read_bounds(decl);

        // A trait or an implementation is a symbol table of its own, so a
        // method keeps its bare name.
        let name = match site {
            Site::AtModule => self.symbols.module().declares(&name).symbol(),
            Site::InTrait | Site::InImpl => name,
        };

        let body = Region::new();
        if let Some(block) = decl.body() {
            let ty = VarType::get(self.context);
            let arguments = vec![(ty, location); param_count];
            let entry = body.append_block(Block::new(&arguments));

            self.symbols.enter_function(param_names.clone());
            self.convert_block(entry, &mut Locals::new(), &block);
            self.symbols.leave();
        }

        let mut builder = yzl::FnOperationBuilder::new(self.context, location)
            .sym_name(StringAttribute::new(self.context, &name))
            .params(ArrayAttribute::from_strings(self.context, param_names))
            .signature(TypeAttribute::new(signature.into()))
            .body(body);

        if decl.is_agg() {
            builder = builder.agg(Attribute::unit(self.context));
        }

        if decl.is_external() {
            builder = builder.external(Attribute::unit(self.context));
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

        let Some(name) = decl.name_text() else {
            self.report(decl, "trait is missing its name");
            return;
        };

        if !self.is_bound(decl, &name) {
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

        let name = self.symbols.module().declares(&name).symbol();
        let mut r#trait: Operation<'c> =
            yzl::TraitOperationBuilder::new(self.context, self.location(decl))
                .sym_name(StringAttribute::new(self.context, &name))
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

        let Some(trait_name) = decl.trait_().and_then(|trait_ref| trait_ref.name_text()) else {
            self.report(decl, "`impl` is missing its trait");
            return;
        };

        let Some(target) = decl.ty_text() else {
            self.report(decl, "`impl` is missing its type name");
            return;
        };

        let trait_name = match self.symbols.trait_symbol(&trait_name) {
            Some(symbol) => symbol.to_string(),
            None => {
                self.report(decl, &format!("unknown trait `{trait_name}`"));
                trait_name
            }
        };

        let target = match self.symbols.struct_symbol(&target) {
            Some(symbol) => symbol.to_string(),
            None => {
                if types::scalar(self.context, &target).is_none() {
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
                .r#trait(FlatSymbolRefAttribute::new(self.context, &trait_name))
                .target(FlatSymbolRefAttribute::new(self.context, &target))
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
        let location = self.location(decl);

        let Some(name) = decl.name_text() else {
            self.report(decl, "let binding is missing its name");
            return;
        };

        let Some(expr) = decl.expr() else {
            self.report(decl, "let binding is missing its expression");
            return;
        };

        if self.symbols.in_body() {
            let value = self.convert_expr(block, locals, &expr);
            self.bind_local(locals, &name, value, decl.mutability());
            return;
        }

        if !self.is_bound(decl, &name) {
            debug_assert!(
                self.diagnostics.has_errors(),
                "`{name}` was not bound by the hoist and nothing reported why"
            );
            return;
        }

        let annotation = decl
            .type_annotation()
            .map(|annotation| self.read_type(annotation));

        let region = Region::new();
        let body = region.append_block(Block::new(&[]));
        let (value, row) = match &expr {
            ast::Expr::Rel(rel) => {
                let (value, row) = self.convert_query(body, rel);
                (value, Some(row))
            }
            _ => (self.convert_expr(body, &Locals::new(), &expr), None),
        };

        body.append_operation(yzl::r#yield(self.context, &[value], location).into());

        let symbol = self.symbols.module().declares(&name).symbol();
        let mut builder = yzl::LetOperationBuilder::new(self.context, location)
            .sym_name(StringAttribute::new(self.context, &symbol))
            .body(region);

        if let Some(annotation) = annotation {
            builder = builder.annotation(TypeAttribute::new(annotation));
        }

        let mut r#let: Operation<'c> = builder.build().into();
        if decl.visibility() != Visibility::Public {
            r#let.set_private(self.context);
        }

        block.append_operation(r#let);

        let kind = match row {
            Some(row) => BindingKind::Relation { row },
            None => BindingKind::Let,
        };

        self.symbols.bind(
            &name,
            Binding {
                kind,
                declared: decl.syntax().text_range(),
                visibility: decl.visibility(),
            },
        );
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

    fn convert_nested_block<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        locals: &mut Locals<'c, 'a>,
        body: &ast::BlockStmt,
    ) {
        if self.check_in_body(body, "a block is not a top-level statement") {
            self.convert_block(block, locals, body);
        }
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
            ast::Expr::IdentExpr(ident) => ident.name_text(),
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

        if !self.check_assignable(assign, &name) {
            return;
        }

        let value = self.convert_expr(block, locals, &value);
        self.bind_local(locals, &name, value, Mutability::Mutable);
    }

    fn convert_return<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        locals: &mut Locals<'c, 'a>,
        ret: &ast::ReturnStmt,
    ) {
        if self.symbols.in_function() {
            let values: Vec<Value> = ret
                .expr()
                .map(|expr| self.convert_expr(block, locals, &expr))
                .into_iter()
                .collect();
            block.append_operation(yzl::r#return(self.context, &values, self.location(ret)).into());
            return;
        }

        self.report(ret, "a return is not a top-level statement");
    }

    /// One entry per `(parameter, trait)` pair.
    fn read_bounds(&mut self, decl: &ast::FuncStmt) -> (Vec<String>, Vec<String>) {
        let mut subjects = Vec::new();
        let mut traits = Vec::new();
        for bound in decl.bounds() {
            let Some(subject) = bound.subject_text() else {
                self.report(&bound, "type bound is missing its subject");
                continue;
            };

            if !self.symbols.is_type_param(&subject) {
                self.report(&bound, &format!("unknown type parameter `{subject}`"));
            }

            for trait_ref in bound.traits() {
                let Some(name) = trait_ref.name_text() else {
                    self.report(&trait_ref, "trait reference is missing its name");
                    continue;
                };

                if self.symbols.trait_symbol(&name).is_none() {
                    self.report(&trait_ref, &format!("unknown trait `{name}`"));
                }

                subjects.push(subject.clone());
                traits.push(name);
            }
        }

        (subjects, traits)
    }

    fn read_type(&mut self, annotation: ast::TypeAnnotation) -> Type<'c> {
        let named = match annotation {
            ast::TypeAnnotation::NamedTypeAnnotation(named) => named,
            ast::TypeAnnotation::FuncTypeAnnotation(func) => {
                self.report(&func, "function types are not supported yet");
                return VarType::get(self.context);
            }
        };

        let Some(name) = named.name_text() else {
            self.report(&named, "type is missing its name");
            return VarType::get(self.context);
        };

        if self.symbols.is_type_param(&name) {
            return ParamType::new(self.context, &name).into();
        }

        if name == "List" {
            let mut args = named.args();
            let (Some(inner), None) = (args.next(), args.next()) else {
                self.report(&named, "`List` takes exactly one type argument");
                return VarType::get(self.context);
            };

            let inner = self.read_type(inner);
            return ListType::new(self.context, inner).into();
        }

        if let Some(scalar) = types::scalar(self.context, &name) {
            return scalar;
        }

        if let Some(symbol) = self.symbols.struct_symbol(&name) {
            return StructType::new(self.context, &symbol).into();
        }

        self.report(&named, &format!("unknown type `{name}`"));
        VarType::get(self.context)
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
            let (Some(name), Some(ty)) = (field.name_text(), field.ty()) else {
                self.report(&field, "struct field is incomplete");
                continue;
            };

            names.push(name);
            types.push(self.read_type(ty));
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

    fn check_assignable(&mut self, node: &impl AstNode, name: &str) -> bool {
        let message = match self.symbols.lookup(Reference::unqualified(name)) {
            Lookup::Local {
                mutability: Mutability::Mutable,
                ..
            } => return true,
            Lookup::Local { .. } => {
                format!("`{name}` is not mutable; declare it with `let mut` to assign it")
            }
            Lookup::Param(_) => format!("`{name}` is a parameter and cannot be assigned"),
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
        false
    }

    fn bind_local<'a>(
        &mut self,
        locals: &mut Locals<'c, 'a>,
        name: &str,
        value: Value<'c, 'a>,
        mutability: Mutability,
    ) {
        self.symbols.bind_local(name, locals.len(), mutability);
        locals.push(value);
    }

    fn is_bound(&self, node: &impl AstNode, name: &str) -> bool {
        self.symbols
            .binding(name)
            .is_some_and(|binding| binding.declared == node.syntax().text_range())
    }

    fn path_text(&self, path: &ast::ModulePath) -> String {
        path.segments_text().collect::<Vec<_>>().join(".")
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::test_support::{lower, lowered, reported};

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
              ^bb0(%arg0: !yzl.var):
                %0 = yz.constant_int 2
                %1 = yz.mul %arg0, %0 : !yzl.var, !yz.int64 -> !yzl.var
                %2 = yz.constant_int 1
                %3 = yz.add %1, %2 : !yzl.var, !yz.int64 -> !yzl.var
                yzl.return %3 : !yzl.var
              } {sym_visibility = "private"}
              yzl.fn @spread params ["x"] (!yz.int64) -> !yz.int64 agg {
              ^bb0(%arg0: !yzl.var):
                %0 = yzl.call @max(%arg0) : (!yzl.var) -> !yzl.var {agg, callee_source = "builtin"}
                %1 = yzl.call @min(%arg0) : (!yzl.var) -> !yzl.var {agg, callee_source = "builtin"}
                %2 = yz.sub %0, %1 : !yzl.var, !yzl.var -> !yzl.var
                yzl.return %2 : !yzl.var
              } {sym_visibility = "private"}
              yzl.fn @upper params ["s"] (!yz.str) -> !yz.str external {
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
              ^bb0(%arg0: !yzl.var):
                %1 = yz.constant_int 1
                %2 = yz.add %arg0, %1 : !yzl.var, !yz.int64 -> !yzl.var
                %3 = yz.constant_int 2
                %4 = yz.mul %arg0, %3 : !yzl.var, !yz.int64 -> !yzl.var
                yzl.return %4 : !yzl.var
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
                ^bb0(%arg0: !yzl.var, %arg1: !yzl.var):
                  %0 = yz.add %arg0, %arg1 : !yzl.var, !yzl.var -> !yzl.var
                  yzl.return %0 : !yzl.var
                }
              }
              yzl.fn @id generics ["T"] where ["T"] : [@Add] params ["x"] (!yzl.param<"T">) -> !yzl.param<"T"> {
              ^bb0(%arg0: !yzl.var):
                yzl.return %arg0 : !yzl.var
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
              yzl.let @ids : !yz.list<!yz.int64> {
                %0 = yz.constant_int 1
                %1 = yz.constant_int 3
                %2 = yzl.list[%0, %1] : (!yz.int64, !yz.int64) -> !yzl.var
                yzl.yield %2 : !yzl.var
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
    fn assigning_a_binding_needs_mut() {
        expect![[r#"
            error: `n` is not mutable; declare it with `let mut` to assign it
             --> test.yz:6:5
              |
            6 |     n = n + 1
              |     ^^^^^^^^^
        "#]]
        .assert_eq(&reported(
            "struct Row { a: int64 }\ntable t = Row\n\ndef f(x: int64) -> int64 {\n    let n = 0\n    n = n + 1\n    return n\n}\n\nfrom t |> select f(a) as v\n",
        ));
    }

    #[test]
    fn a_mutable_binding_may_be_assigned() {
        let module = lowered(
            "struct Row { a: int64 }\ntable t = Row\n\ndef f(x: int64) -> int64 {\n    let mut n = 0\n    n = n + 1\n    return n\n}\n\nfrom t |> select f(a) as v\n",
        );
        assert!(module.contains("yzl.return"), "the body lowers:\n{module}");
    }

    #[test]
    fn a_parameter_is_not_assignable() {
        expect![[r#"
            error: `x` is a parameter and cannot be assigned
             --> test.yz:5:5
              |
            5 |     x = 1
              |     ^^^^^
        "#]]
        .assert_eq(&reported(
            "struct Row { a: int64 }\ntable t = Row\n\ndef f(x: int64) -> int64 {\n    x = 1\n    return x\n}\n\nfrom t |> select f(a) as v\n",
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
              ^bb0(%arg0: !yzl.var):
                yzl.yield %arg0 : !yzl.var
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
