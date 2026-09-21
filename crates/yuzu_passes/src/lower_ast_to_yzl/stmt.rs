//! Statements: imports, the hoist that registers every declaration, and
//! the conversion of each declaration into its `yzl` op.

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
use yuzu_mlir::ext::OperationMutExt;
use yuzu_mlir::ods::yzl;
use yuzu_mlir::types;
use yuzu_mlir::{ListType, ParamType, StructType};

use crate::lower_ast_to_yzl::symbols::{
    Binding, BindingKind, Callable, FunctionKind, Lookup, ModulePath, Reference, Row,
};
use crate::lower_ast_to_yzl::{AstToYzl, Locals};
use yuzu_mlir::SymbolTable as MlirSymbolTable;

/// Where a `fn` is written, which decides the generics its surroundings
/// supply and whether it needs a body.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Declared {
    AtModule,
    InTrait,
    InImpl,
}

impl Declared {
    fn generics(self) -> &'static [&'static str] {
        match self {
            Declared::AtModule => &[],
            Declared::InTrait | Declared::InImpl => &["Self"],
        }
    }
}

/// A function declaration, read and checked, before its op is built.
struct Function<'c> {
    symbol: &'c str,
    declared: Declared,
    generics: Vec<&'c str>,
    params: Vec<&'c str>,
    signature: FunctionType<'c>,
    bound_params: Vec<Attribute<'c>>,
    bound_traits: Vec<Attribute<'c>>,
    kind: FunctionKind,
    source: CalleeSource,
    visibility: Visibility,
    location: Location<'c>,
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
                    let last = self.intern(last);

                    let name = self.read_name(import.alias()).unwrap_or(last);
                    let path = self.intern(&path);
                    self.bind_or_report(
                        import,
                        name,
                        BindingKind::Module { path },
                        Visibility::Private,
                    );
                }
                ast::Stmt::ModStmt(decl) => {
                    let Some(name) = self.read_name(decl.name()) else {
                        self.report(decl, "module declaration is missing its name");
                        continue;
                    };

                    let path = self.built_symbol(name);
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
                ast::Stmt::StructStmt(_)
                | ast::Stmt::TraitStmt(_)
                | ast::Stmt::FuncStmt(_)
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
    }

    pub(super) fn convert_stmt<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        stmt: &ast::Stmt,
        mlir_symbols: &mut MlirSymbolTable<'c, '_>,
    ) {
        match stmt {
            ast::Stmt::StructStmt(decl) => self.convert_struct(block, decl),
            ast::Stmt::TableStmt(decl) => self.convert_table(block, decl),
            ast::Stmt::FuncStmt(decl) => self.convert_fn(block, decl),
            ast::Stmt::TraitStmt(decl) => self.convert_trait(block, decl),
            ast::Stmt::ImplStmt(decl) => self.convert_impl(block, decl),
            ast::Stmt::LetStmt(decl) => self.convert_let(decl, mlir_symbols),
            ast::Stmt::ExprStmt(stmt) => {
                if !self.symbols.module().is_entry()
                    && matches!(stmt.expr(), Some(ast::Expr::Rel(_)))
                {
                    self.report(stmt, "a module cannot hold a query");
                    return;
                }

                if let Some(expr) = stmt.expr() {
                    self.convert_expr(block, &Locals::new(), &expr);
                }
            }
            ast::Stmt::BlockStmt(stmt) => self.report(stmt, "a block is not a top-level statement"),
            ast::Stmt::AssignStmt(stmt) => {
                self.report(stmt, "an assignment is not a top-level statement")
            }
            ast::Stmt::ReturnStmt(stmt) => {
                self.report(stmt, "a return is not a top-level statement")
            }
            ast::Stmt::ImportStmt(_) | ast::Stmt::FromImportStmt(_) | ast::Stmt::ModStmt(_) => {}
        }
    }

    /// `pub(mod)` reaches the enclosing module's files. Until a module holds
    /// more than one file, nobody asking from outside is one of them.
    pub(super) fn read_export(
        &mut self,
        at: &impl AstNode,
        path: &str,
        name: &'c str,
    ) -> Option<Binding<'c>> {
        let module = ModulePath::of(self.intern(path));
        if !self.symbols.contains_module(module) {
            self.report(at, &format!("`{path}` is not a module this file reads"));
            return None;
        }

        let Some(binding) = self.symbols.declared(module.declares(name)).cloned() else {
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

        Some(binding)
    }

    fn bind_import(&mut self, path: &str, item: &ast::ImportItem) {
        let Some(name) = self.read_name(item.name()) else {
            self.report(item, "import item is missing its name");
            return;
        };

        let Some(binding) = self.read_export(item, path, name) else {
            return;
        };

        let local = self.read_name(item.alias()).unwrap_or(name);
        self.declare_imported(item, local, binding);
    }

    fn declare_imported(&mut self, node: &impl AstNode, name: &'c str, binding: Binding<'c>) {
        if self.symbols.binding(name).is_some() {
            self.check_duplicate(node, binding.kind.what(), name);
            return;
        }

        self.symbols.bind(
            name,
            Binding {
                declared: node.syntax().text_range(),
                ..binding
            },
        );
    }

    fn hoist_struct(&mut self, decl: &ast::StructStmt) {
        let Some(name) = self.read_name(decl.name()) else {
            return;
        };

        let fields = decl
            .fields()
            .filter_map(|f| self.read_name(f.name()))
            .collect();
        let symbol = self.built_symbol(name);
        self.bind_or_report(
            decl,
            name,
            BindingKind::Struct { fields, symbol },
            decl.visibility(),
        );
    }

    fn hoist_table(&mut self, decl: &ast::TableStmt) {
        let Some(name) = self.read_name(decl.name()) else {
            return;
        };

        let row = if decl.inline_fields().next().is_some() {
            Row::from(
                decl.inline_fields()
                    .filter_map(|field| self.read_name(field.name()))
                    .collect::<Vec<_>>(),
            )
        } else {
            let Some(declared) = self.read_name(decl.row_struct()) else {
                return;
            };

            match self.symbols.kind(declared) {
                Some(BindingKind::Struct { fields, .. }) => Row::from(fields.clone()),
                _ => {
                    self.report(decl, &format!("`{declared}` is not a struct"));
                    return;
                }
            }
        };

        let symbol = self.built_symbol(name);
        self.bind_or_report(
            decl,
            name,
            BindingKind::Relation { row, symbol },
            decl.visibility(),
        );
    }

    fn hoist_trait(&mut self, decl: &ast::TraitStmt) {
        let Some(name) = self.read_name(decl.name()) else {
            return;
        };

        let methods = decl
            .methods()
            .filter_map(|m| self.read_name(m.name()))
            .collect();
        let symbol = self.built_symbol(name);
        self.bind_or_report(
            decl,
            name,
            BindingKind::Trait { methods, symbol },
            decl.visibility(),
        );
    }

    fn hoist_fn(&mut self, decl: &ast::FuncStmt) {
        let Some(name) = self.read_name(decl.name()) else {
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

        let arity = decl.params().count();
        let symbol = self.built_symbol(name);
        self.bind_or_report(
            decl,
            name,
            BindingKind::Func(Callable {
                symbol,
                source,
                kind,
                min_args: arity,
                max_args: arity,
            }),
            decl.visibility(),
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
            self.check_duplicate(node, kind.what(), name);
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

    /// The note says where the other declaration is rather than which came
    /// first: hoisting takes the declarations out of source order.
    fn check_duplicate(&mut self, node: &impl AstNode, what: &str, name: &str) {
        let declared = self
            .symbols
            .binding(name)
            .expect("a duplicate is checked against a bound name")
            .declared;

        let other = self.position_text(declared);
        let diagnostic = self
            .error_at(
                node.syntax().text_range(),
                &format!("the {what} `{name}` is already defined"),
            )
            .note(format!("also declared at {other}"));
        self.diagnostics.emit(diagnostic);
    }

    fn convert_struct<'a>(&mut self, block: BlockRef<'c, 'a>, decl: &ast::StructStmt) {
        let Some(name) = self.read_name(decl.name()) else {
            self.report(decl, "struct is missing its name");
            return;
        };

        if !self.is_declaration_of(decl, name) {
            debug_assert!(
                self.diagnostics.has_errors(),
                "`{name}` was not bound by the hoist and nothing reported why"
            );
            return;
        }

        let symbol = self.built_symbol(name);
        self.emit_struct(
            block,
            symbol,
            decl.fields(),
            decl.visibility(),
            self.location(decl),
        );
    }

    fn convert_table<'a>(&mut self, block: BlockRef<'c, 'a>, decl: &ast::TableStmt) {
        let Some(name) = self.read_name(decl.name()) else {
            self.report(decl, "table is missing its name");
            return;
        };

        if !self.is_declaration_of(decl, name) {
            debug_assert!(
                self.diagnostics.has_errors(),
                "`{name}` was not bound by the hoist and nothing reported why"
            );
            return;
        }

        // An imported struct is held under the module that declared it,
        // whatever an `as` renamed it to here.
        let row = match self.read_name(decl.row_struct()) {
            Some(row) => self.symbols.struct_symbol(row).unwrap_or(row),
            None => {
                let row = self.built_symbol(&format!("{name}_row"));
                self.emit_struct(
                    block,
                    row,
                    decl.inline_fields(),
                    decl.visibility(),
                    self.location(decl),
                );
                row
            }
        };

        let symbol = self.built_symbol(name);
        self.emit_table(block, symbol, row, decl.visibility(), self.location(decl));
    }

    fn convert_fn<'a>(&mut self, block: BlockRef<'c, 'a>, decl: &ast::FuncStmt) {
        // A nameless `fn` still lowers, so the missing name is what gets
        // reported.
        if let Some(name) = self.read_name(decl.name())
            && !self.is_declaration_of(decl, name)
        {
            return;
        }

        self.convert_method(block, decl, Declared::AtModule);
    }

    fn convert_method<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        decl: &ast::FuncStmt,
        declared: Declared,
    ) {
        let Some(function) = self.read_function(decl, declared) else {
            return;
        };

        let body = self.convert_function_body(&function, decl.body());
        block.append_operation(self.emit_function(&function, body));
    }

    fn read_function(&mut self, decl: &ast::FuncStmt, declared: Declared) -> Option<Function<'c>> {
        let Some(name) = self.read_name(decl.name()) else {
            self.report(decl, "function is missing its name");
            return None;
        };

        match (decl.is_external(), decl.body().is_some()) {
            (true, true) => self.report(decl, "an external function cannot have a body"),
            (false, false) if declared != Declared::InTrait => {
                self.report(decl, "function is missing its body");
            }
            _ => {}
        }

        let generics: Vec<&'c str> = declared
            .generics()
            .iter()
            .copied()
            .chain(
                decl.type_params()
                    .filter_map(|param| self.read_name(param.name())),
            )
            .collect();

        let mut params: Vec<&'c str> = Vec::new();
        let mut param_types = Vec::new();
        for param in decl.params() {
            match self.read_name(param.name()) {
                Some(name) => params.push(name),
                None => self.report(&param, "parameter is missing its name"),
            }

            param_types.push(match param.ty() {
                Some(ty) => self.read_type(ty, &generics),
                None => {
                    self.report(&param, "parameter is missing its type");
                    types::var(self.context)
                }
            });
        }

        let result = match decl.result() {
            Some(result) => self.read_type(result, &generics),
            None => types::var(self.context),
        };
        let signature = FunctionType::new(self.context, &param_types, &[result]);
        let (bound_params, bound_traits) = self.read_bounds(decl, &generics);

        // A trait or an implementation is a symbol table of its own, so a
        // method keeps its bare name.
        let symbol = match declared {
            Declared::AtModule => self.built_symbol(name),
            Declared::InTrait | Declared::InImpl => name,
        };

        Some(Function {
            symbol,
            declared,
            generics,
            params,
            signature,
            bound_params,
            bound_traits,
            kind: if decl.is_agg() {
                FunctionKind::Aggregate
            } else {
                FunctionKind::Scalar
            },
            source: if decl.is_external() {
                CalleeSource::External
            } else {
                CalleeSource::Fn
            },
            visibility: decl.visibility(),
            location: self.location(decl),
        })
    }

    fn convert_function_body(
        &mut self,
        function: &Function<'c>,
        body: Option<ast::BlockStmt>,
    ) -> Region<'c> {
        let region = Region::new();
        let Some(body) = body else {
            return region;
        };

        let arguments: Vec<(Type<'c>, Location<'c>)> = (0..function.signature.input_count())
            .map(|_| (types::var(self.context), function.location))
            .collect();
        let entry = region.append_block(Block::new(&arguments));
        self.symbols.enter_function(function.params.clone());
        self.convert_block(entry, &mut Locals::new(), &body);
        self.symbols.leave();
        region
    }

    fn emit_function(&self, function: &Function<'c>, body: Region<'c>) -> Operation<'c> {
        let params = ArrayAttribute::new(self.context, &self.string_attributes(&function.params));
        let mut builder = yzl::FnOperationBuilder::new(self.context, function.location)
            .sym_name(StringAttribute::new(self.context, function.symbol))
            .params(params)
            .signature(TypeAttribute::new(function.signature.into()))
            .body(body);

        if function.kind == FunctionKind::Aggregate {
            builder = builder.agg(Attribute::unit(self.context));
        }

        if function.source == CalleeSource::External {
            builder = builder.external(Attribute::unit(self.context));
        }

        if !function.generics.is_empty() {
            let names = self.string_attributes(&function.generics);
            builder = builder.type_params(ArrayAttribute::new(self.context, &names));
        }

        if !function.bound_params.is_empty() {
            builder = builder
                .bound_params(ArrayAttribute::new(self.context, &function.bound_params))
                .bound_traits(ArrayAttribute::new(self.context, &function.bound_traits));
        }

        let mut op: Operation<'c> = builder.build().into();
        if function.declared == Declared::AtModule && function.visibility != Visibility::Public {
            op.set_private(self.context);
        }

        op
    }

    fn convert_trait<'a>(&mut self, block: BlockRef<'c, 'a>, decl: &ast::TraitStmt) {
        let Some(name) = self.read_name(decl.name()) else {
            self.report(decl, "trait is missing its name");
            return;
        };

        if !self.is_declaration_of(decl, name) {
            debug_assert!(
                self.diagnostics.has_errors(),
                "`{name}` was not bound by the hoist and nothing reported why"
            );
            return;
        }

        let region = Region::new();
        let body = region.append_block(Block::new(&[]));
        for method in decl.methods() {
            self.convert_method(body, &method, Declared::InTrait);
        }

        let symbol = self.built_symbol(name);
        let mut r#trait: Operation<'c> =
            yzl::TraitOperationBuilder::new(self.context, self.location(decl))
                .sym_name(StringAttribute::new(self.context, symbol))
                .body(region)
                .build()
                .into();
        if decl.visibility() != Visibility::Public {
            r#trait.set_private(self.context);
        }
        block.append_operation(r#trait);
    }

    fn convert_impl<'a>(&mut self, block: BlockRef<'c, 'a>, decl: &ast::ImplStmt) {
        let Some(trait_name) = decl
            .trait_()
            .and_then(|trait_ref| self.read_name(trait_ref.name()))
        else {
            self.report(decl, "`impl` is missing its trait");
            return;
        };

        let Some(target) = self.read_name(decl.ty()) else {
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
            self.convert_method(body, &method, Declared::InImpl);
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

    /// The symbol table places the operation, so this takes no block.
    fn convert_let(&mut self, decl: &ast::LetStmt, mlir_symbols: &mut MlirSymbolTable<'c, '_>) {
        let Some(name) = self.read_name(decl.name()) else {
            self.report(decl, "let binding is missing its name");
            return;
        };

        let Some(expr) = decl.expr() else {
            self.report(decl, "let binding is missing its expression");
            return;
        };

        let annotation = decl
            .type_annotation()
            .map(|annotation| self.read_type(annotation, &[]));
        let location = self.location(decl);
        let (body, row) = self.convert_let_body(&expr, location);
        let symbol = self.emit_let(
            mlir_symbols,
            self.built_symbol(name),
            annotation,
            body,
            decl.visibility(),
            location,
        );

        let kind = match row {
            Some(row) => BindingKind::Relation { row, symbol },
            None => BindingKind::Let { symbol },
        };
        self.symbols.bind(
            name,
            Binding {
                kind,
                declared: decl.syntax().text_range(),
                visibility: decl.visibility(),
            },
        );
    }

    /// The region, and the row when the expression is a query. A query's
    /// row has to be taken before its scope is left.
    fn convert_let_body(
        &mut self,
        expr: &ast::Expr,
        location: Location<'c>,
    ) -> (Region<'c>, Option<Row<'c>>) {
        let region = Region::new();
        let body = region.append_block(Block::new(&[]));
        let (value, row) = match expr {
            ast::Expr::Rel(rel) => {
                let value = self.convert_rel(body, rel);
                let row = self.symbols.row().clone();
                self.symbols.leave();
                (value, Some(row))
            }
            _ => (self.convert_expr(body, &Locals::new(), expr), None),
        };

        body.append_operation(yzl::r#yield(self.context, &[value], location).into());
        (region, row)
    }

    /// A `let` may take a name something else already holds: the code
    /// between the two reads the earlier one, so both stay in the module and
    /// the symbol table renames the second. What it assigned is returned,
    /// since that is what the binding reaches.
    fn emit_let(
        &self,
        mlir_symbols: &mut MlirSymbolTable<'c, '_>,
        symbol: &'c str,
        annotation: Option<Type<'c>>,
        body: Region<'c>,
        visibility: Visibility,
        location: Location<'c>,
    ) -> &'c str {
        let mut builder = yzl::LetOperationBuilder::new(self.context, location)
            .sym_name(StringAttribute::new(self.context, symbol))
            .body(body);
        if let Some(annotation) = annotation {
            builder = builder.annotation(TypeAttribute::new(annotation));
        }

        let mut r#let: Operation<'c> = builder.build().into();
        if visibility != Visibility::Public {
            r#let.set_private(self.context);
        }

        let assigned = mlir_symbols.insert(r#let);
        StringAttribute::try_from(assigned)
            .expect("a symbol name is a string")
            .value()
    }

    fn convert_block<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        locals: &mut Locals<'c, 'a>,
        body: &ast::BlockStmt,
    ) {
        self.symbols.enter_block();
        for stmt in body.stmts() {
            self.convert_body_stmt(block, locals, &stmt);
        }

        self.symbols.leave();
    }

    fn convert_body_stmt<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        locals: &mut Locals<'c, 'a>,
        stmt: &ast::Stmt,
    ) {
        match stmt {
            ast::Stmt::LetStmt(binding) => {
                let Some(name) = self.read_name(binding.name()) else {
                    self.report(binding, "let binding is missing its name");
                    return;
                };

                let Some(expr) = binding.expr() else {
                    self.report(binding, "let binding is missing its expression");
                    return;
                };

                let value = self.convert_expr(block, locals, &expr);
                self.bind_local(locals, name, value, binding.mutability());
            }
            ast::Stmt::AssignStmt(assign) => {
                let target = assign.target().and_then(|target| match target {
                    ast::Expr::IdentExpr(ident) => self.read_name(ident.name()),
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

                if !self.check_assignable(assign, name) {
                    return;
                }

                let value = self.convert_expr(block, locals, &value);
                self.bind_local(locals, name, value, Mutability::Mutable);
            }
            ast::Stmt::ReturnStmt(ret) => {
                let values: Vec<Value> = ret
                    .expr()
                    .map(|expr| self.convert_expr(block, locals, &expr))
                    .into_iter()
                    .collect();
                block.append_operation(
                    yzl::r#return(self.context, &values, self.location(ret)).into(),
                );
            }
            ast::Stmt::ExprStmt(expr_stmt) => {
                if let Some(expr) = expr_stmt.expr() {
                    self.convert_expr(block, locals, &expr);
                }
            }
            ast::Stmt::BlockStmt(nested) => self.convert_block(block, locals, nested),
            ast::Stmt::StructStmt(_)
            | ast::Stmt::TraitStmt(_)
            | ast::Stmt::ImplStmt(_)
            | ast::Stmt::FuncStmt(_)
            | ast::Stmt::TableStmt(_) => {
                self.report(stmt, "declarations inside functions are not supported yet");
            }
            ast::Stmt::ImportStmt(_) | ast::Stmt::FromImportStmt(_) | ast::Stmt::ModStmt(_) => {
                self.report(stmt, "this belongs at the top of the file");
            }
        }
    }

    /// One entry per `(parameter, trait)` pair.
    fn read_bounds(
        &mut self,
        decl: &ast::FuncStmt,
        generics: &[&'c str],
    ) -> (Vec<Attribute<'c>>, Vec<Attribute<'c>>) {
        let mut subjects = Vec::new();
        let mut traits = Vec::new();
        for bound in decl.bounds() {
            let Some(subject) = self.read_name(bound.subject()) else {
                self.report(&bound, "type bound is missing its subject");
                continue;
            };

            if !generics.contains(&subject) {
                self.report(&bound, &format!("unknown type parameter `{subject}`"));
            }

            for trait_ref in bound.traits() {
                let Some(name) = self.read_name(trait_ref.name()) else {
                    self.report(&trait_ref, "trait reference is missing its name");
                    continue;
                };

                if self.symbols.trait_symbol(name).is_none() {
                    self.report(&trait_ref, &format!("unknown trait `{name}`"));
                }

                subjects.push(StringAttribute::new(self.context, subject).into());
                traits.push(FlatSymbolRefAttribute::new(self.context, name).into());
            }
        }

        (subjects, traits)
    }

    fn read_fields(
        &mut self,
        fields: impl Iterator<Item = ast::StructField>,
    ) -> (ArrayAttribute<'c>, ArrayAttribute<'c>) {
        let mut names = Vec::new();
        let mut types = Vec::new();
        for field in fields {
            let (Some(name), Some(ty)) = (self.read_name(field.name()), field.ty()) else {
                self.report(&field, "struct field is incomplete");
                continue;
            };

            names.push(StringAttribute::new(self.context, name).into());
            types.push(TypeAttribute::new(self.read_type(ty, &[])).into());
        }

        (
            ArrayAttribute::new(self.context, &names),
            ArrayAttribute::new(self.context, &types),
        )
    }

    fn read_type(&mut self, annotation: ast::TypeAnnotation, generics: &[&'c str]) -> Type<'c> {
        let named = match annotation {
            ast::TypeAnnotation::NamedTypeAnnotation(named) => named,
            ast::TypeAnnotation::FuncTypeAnnotation(func) => {
                self.report(&func, "function types are not supported yet");
                return types::var(self.context);
            }
        };

        let Some(name) = self.read_name(named.name()) else {
            self.report(&named, "type is missing its name");
            return types::var(self.context);
        };

        if generics.contains(&name) {
            return ParamType::new(self.context, name).into();
        }

        if name == "List" {
            let mut args = named.args();
            let (Some(inner), None) = (args.next(), args.next()) else {
                self.report(&named, "`List` takes exactly one type argument");
                return types::var(self.context);
            };

            let inner = self.read_type(inner, generics);
            return ListType::new(self.context, inner).into();
        }

        if let Some(scalar) = types::scalar(self.context, name) {
            return scalar;
        }

        if let Some(symbol) = self.symbols.struct_symbol(name) {
            return StructType::new(self.context, symbol).into();
        }

        self.report(&named, &format!("unknown type `{name}`"));
        types::var(self.context)
    }

    fn emit_struct<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        symbol: &'c str,
        fields: impl Iterator<Item = ast::StructField>,
        visibility: Visibility,
        loc: Location<'c>,
    ) {
        let (names, types) = self.read_fields(fields);
        let mut r#struct: Operation<'c> = yzl::r#struct(
            self.context,
            StringAttribute::new(self.context, symbol),
            names,
            types,
            loc,
        )
        .into();

        if visibility != Visibility::Public {
            r#struct.set_private(self.context);
        }

        block.append_operation(r#struct);
    }

    fn emit_table<'a>(
        &self,
        block: BlockRef<'c, 'a>,
        symbol: &'c str,
        row: &'c str,
        visibility: Visibility,
        location: Location<'c>,
    ) {
        let mut table: Operation<'c> = yzl::table(
            self.context,
            StringAttribute::new(self.context, symbol),
            FlatSymbolRefAttribute::new(self.context, row),
            location,
        )
        .into();
        if visibility != Visibility::Public {
            table.set_private(self.context);
        }

        block.append_operation(table);
    }

    fn check_assignable(&mut self, node: &impl AstNode, name: &'c str) -> bool {
        let message = match self.symbols.lookup(Reference::bare(name)) {
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
            Lookup::Unknown => format!("unresolved identifier `{name}`"),
        };

        self.report(node, &message);
        false
    }

    fn bind_local<'a>(
        &mut self,
        locals: &mut Locals<'c, 'a>,
        name: &'c str,
        value: Value<'c, 'a>,
        mutability: Mutability,
    ) {
        self.symbols.bind_local(name, locals.len(), mutability);
        locals.push(value);
    }

    /// The second declaration of a name is reported by the hoist and left
    /// out of the module, so one name stays one symbol.
    fn is_declaration_of(&self, node: &impl AstNode, name: &str) -> bool {
        self.symbols
            .binding(name)
            .is_some_and(|binding| binding.declared == node.syntax().text_range())
    }

    fn path_text(&self, path: &ast::ModulePath) -> String {
        path.segments()
            .filter_map(|segment| segment.text())
            .collect::<Vec<_>>()
            .join(".")
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::test_support::{lower, lowered, reported};

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
    fn a_let_may_take_a_name_an_earlier_one_holds() {
        expect![[r#"
            module {
              yzl.struct @Row ["a"] : [!yz.int64] {sym_visibility = "private"}
              yzl.table @t of @Row {sym_visibility = "private"}
              yzl.let @cap {
                %2 = yz.constant_int 1
                yzl.yield %2 : !yz.int64
              } {sym_visibility = "private"}
              yzl.let @step {
                %2 = yzl.call @cap() : () -> !yzl.var {callee_source = "let"}
                %3 = yz.constant_int 10
                %4 = yz.add %2, %3 : !yzl.var, !yz.int64 -> !yzl.var
                yzl.yield %4 : !yzl.var
              } {sym_visibility = "private"}
              yzl.let @cap_0 {
                %2 = yz.constant_int 2
                yzl.yield %2 : !yz.int64
              } {sym_visibility = "private"}
              %0 = yzl.from @t
              %1 = yzl.where %0 {
              ^bb0(%arg0: !yzl.var):
                %2 = yzl.call @cap_0() : () -> !yzl.var {callee_source = "let"}
                %3 = yz.cmp "gt", %arg0, %2 : !yzl.var, !yzl.var -> !yzl.var
                yzl.yield %3 : !yzl.var
              }
              yzl.output %1
            }
        "#]]
        .assert_eq(&lowered(
            "struct Row { a: int64 }\ntable t = Row\n\nlet cap = 1\nlet step = cap + 10\nlet cap = 2\n\nfrom t\n|> where a > cap\n",
        ));
    }

    #[test]
    fn a_rebound_query_is_the_one_from_names() {
        let module = lowered(
            "struct Row { a: int64 }\ntable t = Row\n\nlet base = from t\nlet base = from t |> where a > 1\n\nfrom base\n|> select a as out\n",
        );
        assert!(
            module.contains("yzl.let @base_0") && module.contains("yzl.from @base_0"),
            "`from` names the rebinding:\n{module}"
        );
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
