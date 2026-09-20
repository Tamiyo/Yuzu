use melior::ir::attribute::{
    ArrayAttribute, FlatSymbolRefAttribute, StringAttribute, TypeAttribute,
};
use melior::ir::r#type::FunctionType;
use melior::ir::{
    Attribute, Block, BlockLike, BlockRef, Location, Region, RegionLike, Type, Value,
};
use yuzu_ast::ast::Mutability;
use yuzu_ast::{AstNode, Visibility, ast};
use yuzu_mlir::attributes::CalleeKind;
use yuzu_mlir::ods::yzl;
use yuzu_mlir::types;
use yuzu_mlir::{ListType, ParamType, StructType};

use crate::lower_ast_to_yzl::symbols::{Binding, Callable, Kind, Lookup, Reference, Row};
use crate::lower_ast_to_yzl::{AstToYzl, Locals};

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

impl<'c, 'd> AstToYzl<'c, 'd> {
    /// Imports bind before the file's own declarations, so the two collide
    /// the way two declarations do.
    pub(super) fn bind_imports(&mut self, root: &ast::Root) {
        for stmt in root.stmts() {
            match &stmt {
                ast::Stmt::FromImportStmt(import) => {
                    let Some(path) = import.path().map(|path| self.module_path(&path)) else {
                        continue;
                    };

                    for item in import.items() {
                        self.bind_import(&path, &item);
                    }
                }
                ast::Stmt::ImportStmt(import) => {
                    let Some(path) = import.path().map(|path| self.module_path(&path)) else {
                        continue;
                    };

                    let Some(last) = path.rsplit('.').next().map(|last| self.intern(last)) else {
                        continue;
                    };

                    let name = self.ident(import.alias()).unwrap_or(last);
                    let path = self.intern(&path);
                    self.declare(import, name, Kind::Module { path }, Visibility::Private);
                }
                ast::Stmt::ModStmt(decl) => {
                    let Some(name) = self.ident(decl.name()) else {
                        continue;
                    };

                    let path = self.symbol_for(name);
                    self.declare(decl, name, Kind::Module { path }, decl.visibility());
                }
                _ => {}
            }
        }
    }

    fn bind_import(&mut self, path: &str, item: &ast::ImportItem) {
        let Some(name) = self.ident(item.name()) else {
            self.report(item, "import item is missing its name");
            return;
        };

        let Some(binding) = self.exported(item, path, name) else {
            return;
        };

        let local = self.ident(item.alias()).unwrap_or(name);
        self.declare_imported(item, local, binding);
    }

    /// `pub(mod)` reaches the enclosing module's files. Until a module holds
    /// more than one file, nobody asking from outside is one of them.
    pub(super) fn exported(
        &mut self,
        at: &impl AstNode,
        path: &str,
        name: &str,
    ) -> Option<Binding<'c>> {
        let Some(module) = self.exports.get(&Some(path)) else {
            self.report(at, &format!("`{path}` is not a module this file reads"));
            return None;
        };

        let Some(binding) = module.get(name).cloned() else {
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

    fn module_path(&self, path: &ast::ModulePath) -> String {
        path.segments()
            .filter_map(|segment| segment.text())
            .collect::<Vec<_>>()
            .join(".")
    }

    /// A `let` binds in order instead, once its body is emitted. Structs go
    /// in before tables, because a table's row is its struct's fields.
    pub(super) fn hoist(&mut self, root: &ast::Root) {
        for stmt in root.stmts() {
            match &stmt {
                ast::Stmt::StructStmt(decl) => self.hoist_struct(decl),
                ast::Stmt::TraitStmt(decl) => self.hoist_trait(decl),
                ast::Stmt::FuncStmt(decl) => self.hoist_fn(decl),
                _ => {}
            }
        }

        for stmt in root.stmts() {
            if let ast::Stmt::TableStmt(decl) = &stmt {
                self.hoist_table(decl);
            }
        }
    }

    fn hoist_struct(&mut self, decl: &ast::StructStmt) {
        let Some(name) = self.ident(decl.name()) else {
            return;
        };

        let fields = decl.fields().filter_map(|f| self.ident(f.name())).collect();
        let symbol = self.symbol_for(name);
        self.declare(
            decl,
            name,
            Kind::Struct { fields, symbol },
            decl.visibility(),
        );
    }

    fn hoist_table(&mut self, decl: &ast::TableStmt) {
        let Some(name) = self.ident(decl.name()) else {
            return;
        };

        let row = if decl.inline_fields().next().is_some() {
            Row::from(
                decl.inline_fields()
                    .filter_map(|field| self.ident(field.name()))
                    .collect::<Vec<_>>(),
            )
        } else {
            let Some(declared) = self.ident(decl.row_struct()) else {
                return;
            };

            match self.symbols.kind(declared) {
                Some(Kind::Struct { fields, .. }) => Row::from(fields.clone()),
                _ => {
                    self.report(decl, &format!("`{declared}` is not a struct"));
                    return;
                }
            }
        };

        let symbol = self.symbol_for(name);
        self.declare(
            decl,
            name,
            Kind::Relation { row, symbol },
            decl.visibility(),
        );
    }

    fn hoist_trait(&mut self, decl: &ast::TraitStmt) {
        let Some(name) = self.ident(decl.name()) else {
            return;
        };

        let methods = decl
            .methods()
            .filter_map(|m| self.ident(m.name()))
            .collect();
        let symbol = self.symbol_for(name);
        self.declare(
            decl,
            name,
            Kind::Trait { methods, symbol },
            decl.visibility(),
        );
    }

    fn hoist_fn(&mut self, decl: &ast::FuncStmt) {
        let Some(name) = self.ident(decl.name()) else {
            return;
        };

        let kind = if decl.is_external() {
            CalleeKind::External
        } else if decl.is_agg() {
            CalleeKind::AggFn
        } else {
            CalleeKind::Fn
        };

        let arity = decl.params().count();
        let symbol = self.symbol_for(name);
        self.declare(
            decl,
            name,
            Kind::Func(Callable {
                symbol,
                kind,
                min_args: arity,
                max_args: arity,
                is_agg: decl.is_agg(),
            }),
            decl.visibility(),
        );
    }

    fn declare(
        &mut self,
        node: &impl AstNode,
        name: &'c str,
        kind: Kind<'c>,
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
        let Some(declared) = self.symbols.binding(name).map(|binding| binding.declared) else {
            return;
        };

        let other = self.position(declared);
        let diagnostic = self
            .diagnostic(
                node.syntax().text_range(),
                &format!("the {what} `{name}` is already defined"),
            )
            .note(format!("also declared at {other}"));
        self.diagnostics.emit(diagnostic);
    }

    pub(super) fn convert_stmt<'a>(&mut self, block: BlockRef<'c, 'a>, stmt: &ast::Stmt) {
        match stmt {
            ast::Stmt::StructStmt(decl) => self.convert_struct(block, decl),
            ast::Stmt::TableStmt(decl) => self.convert_table(block, decl),
            ast::Stmt::FuncStmt(decl) => self.convert_fn(block, decl),
            ast::Stmt::TraitStmt(decl) => self.convert_trait(block, decl),
            ast::Stmt::ImplStmt(decl) => self.convert_impl(block, decl),
            ast::Stmt::LetStmt(decl) => self.convert_let(block, decl),
            ast::Stmt::ExprStmt(stmt) => {
                if self.module.is_some() && matches!(stmt.expr(), Some(ast::Expr::Rel(_))) {
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

    fn convert_struct<'a>(&mut self, block: BlockRef<'c, 'a>, decl: &ast::StructStmt) {
        let Some(name) = self.ident(decl.name()) else {
            self.report(decl, "struct is missing its name");
            return;
        };

        if !self.declares(decl, name) {
            return;
        }

        let symbol = self.symbol_for(name);
        self.append_struct(block, symbol, decl.fields(), self.location(decl));
    }

    fn convert_table<'a>(&mut self, block: BlockRef<'c, 'a>, decl: &ast::TableStmt) {
        let Some(name) = self.ident(decl.name()) else {
            self.report(decl, "table is missing its name");
            return;
        };

        if !self.declares(decl, name) {
            return;
        }

        // An imported struct is held under the module that declared it,
        // whatever an `as` renamed it to here.
        let row = match self.ident(decl.row_struct()) {
            Some(row) => self.symbols.struct_symbol(row).unwrap_or(row),
            None => {
                let row = self.symbol_for(self.intern(&format!("{name}_row")));
                self.append_struct(block, row, decl.inline_fields(), self.location(decl));
                row
            }
        };

        let symbol = self.symbol_for(name);
        block.append_operation(
            yzl::table(
                self.context,
                StringAttribute::new(self.context, symbol),
                FlatSymbolRefAttribute::new(self.context, row),
                self.location(decl),
            )
            .into(),
        );
    }

    fn append_struct<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        symbol: &'c str,
        fields: impl Iterator<Item = ast::StructField>,
        loc: Location<'c>,
    ) {
        let (names, types) = self.field_attrs(fields);
        block.append_operation(
            yzl::r#struct(
                self.context,
                StringAttribute::new(self.context, symbol),
                names,
                types,
                loc,
            )
            .into(),
        );
    }

    fn convert_fn<'a>(&mut self, block: BlockRef<'c, 'a>, decl: &ast::FuncStmt) {
        // A nameless `fn` still lowers, so the missing name is what gets
        // reported.
        if let Some(name) = self.ident(decl.name())
            && !self.declares(decl, name)
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
        let Some(name) = self.ident(decl.name()) else {
            self.report(decl, "function is missing its name");
            return;
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
                    .filter_map(|param| self.ident(param.name())),
            )
            .collect();

        let mut names = Vec::new();
        let mut param_types = Vec::new();
        for param in decl.params() {
            match self.ident(param.name()) {
                Some(name) => names.push(name),
                None => self.report(&param, "parameter is missing its name"),
            }

            param_types.push(match param.ty() {
                Some(ty) => self.annotation_type(ty, &generics),
                None => {
                    self.report(&param, "parameter is missing its type");
                    types::var(self.context)
                }
            });
        }

        let result = match decl.result() {
            Some(result) => self.annotation_type(result, &generics),
            None => types::var(self.context),
        };
        let signature = FunctionType::new(self.context, &param_types, &[result]);
        let (subjects, traits) = self.bound_attrs(decl, &generics);
        let params = ArrayAttribute::new(self.context, &self.string_attrs(&names));

        let region = Region::new();
        if let Some(body) = decl.body() {
            let loc = self.location(decl);
            let arguments: Vec<(Type<'c>, Location<'c>)> = param_types
                .iter()
                .map(|_| (types::var(self.context), loc))
                .collect();
            let entry = region.append_block(Block::new(&arguments));
            self.symbols.enter_function(names);
            self.convert_block(entry, &mut Locals::new(), &body);
            self.symbols.leave();
        }

        // A trait or an implementation is a symbol table of its own, so a
        // method keeps its bare name.
        let symbol = match declared {
            Declared::AtModule => self.symbol_for(name),
            Declared::InTrait | Declared::InImpl => name,
        };
        let mut builder = yzl::FnOperationBuilder::new(self.context, self.location(decl))
            .sym_name(StringAttribute::new(self.context, symbol))
            .params(params)
            .signature(TypeAttribute::new(signature.into()))
            .body(region);

        if decl.is_agg() {
            builder = builder.agg(Attribute::unit(self.context));
        }

        if decl.is_external() {
            builder = builder.external(Attribute::unit(self.context));
        }

        if !generics.is_empty() {
            let names = self.string_attrs(&generics);
            builder = builder.type_params(ArrayAttribute::new(self.context, &names));
        }

        if !subjects.is_empty() {
            builder = builder
                .bound_params(ArrayAttribute::new(self.context, &subjects))
                .bound_traits(ArrayAttribute::new(self.context, &traits));
        }

        block.append_operation(builder.build().into());
    }

    /// One entry per `(parameter, trait)` pair.
    fn bound_attrs(
        &mut self,
        decl: &ast::FuncStmt,
        generics: &[&'c str],
    ) -> (Vec<Attribute<'c>>, Vec<Attribute<'c>>) {
        let mut subjects = Vec::new();
        let mut traits = Vec::new();
        for bound in decl.bounds() {
            let Some(subject) = self.ident(bound.subject()) else {
                self.report(&bound, "type bound is missing its subject");
                continue;
            };

            if !generics.contains(&subject) {
                self.report(&bound, &format!("unknown type parameter `{subject}`"));
            }

            for trait_ref in bound.traits() {
                let Some(name) = self.ident(trait_ref.name()) else {
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

    fn convert_trait<'a>(&mut self, block: BlockRef<'c, 'a>, decl: &ast::TraitStmt) {
        let Some(name) = self.ident(decl.name()) else {
            self.report(decl, "trait is missing its name");
            return;
        };

        if !self.declares(decl, name) {
            return;
        }

        let region = Region::new();
        let body = region.append_block(Block::new(&[]));
        for method in decl.methods() {
            self.convert_method(body, &method, Declared::InTrait);
        }

        let symbol = self.symbol_for(name);
        block.append_operation(
            yzl::TraitOperationBuilder::new(self.context, self.location(decl))
                .sym_name(StringAttribute::new(self.context, symbol))
                .body(region)
                .build()
                .into(),
        );
    }

    fn convert_impl<'a>(&mut self, block: BlockRef<'c, 'a>, decl: &ast::ImplStmt) {
        let Some(trait_name) = decl
            .trait_()
            .and_then(|trait_ref| self.ident(trait_ref.name()))
        else {
            self.report(decl, "`impl` is missing its trait");
            return;
        };

        let Some(target) = self.ident(decl.ty()) else {
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

    fn convert_let<'a>(&mut self, block: BlockRef<'c, 'a>, decl: &ast::LetStmt) {
        let Some(name) = self.ident(decl.name()) else {
            self.report(decl, "let binding is missing its name");
            return;
        };

        let Some(expr) = decl.expr() else {
            self.report(decl, "let binding is missing its expression");
            return;
        };

        let symbol = self.rebound_symbol(name);
        let annotation = decl
            .type_annotation()
            .map(|annotation| self.annotation_type(annotation, &[]));
        let region = Region::new();
        let body = region.append_block(Block::new(&[]));
        let value = match &expr {
            ast::Expr::Rel(rel) => {
                let value = self.convert_rel(body, rel);
                let row = self.symbols.row().clone();
                self.symbols.leave();
                self.symbols.bind(
                    name,
                    Binding {
                        kind: Kind::Relation { row, symbol },
                        declared: decl.syntax().text_range(),
                        visibility: decl.visibility(),
                    },
                );
                value
            }
            _ => {
                let value = self.convert_expr(body, &Locals::new(), &expr);
                self.symbols.bind(
                    name,
                    Binding {
                        kind: Kind::Let { symbol },
                        declared: decl.syntax().text_range(),
                        visibility: decl.visibility(),
                    },
                );
                value
            }
        };

        body.append_operation(yzl::r#yield(self.context, &[value], self.location(decl)).into());
        let mut builder = yzl::LetOperationBuilder::new(self.context, self.location(decl))
            .sym_name(StringAttribute::new(self.context, symbol))
            .body(region);
        if let Some(annotation) = annotation {
            builder = builder.annotation(TypeAttribute::new(annotation));
        }

        block.append_operation(builder.build().into());
    }

    /// A `let` taking a name something else already holds gets a symbol of
    /// its own: the code between the two reads the earlier one, and both
    /// stay in the module.
    fn rebound_symbol(&mut self, name: &'c str) -> &'c str {
        if self.symbols.binding(name).is_none() {
            return self.symbol_for(name);
        }

        self.rebound += 1;
        let taken = self.intern(&format!("{name}.{}", self.rebound));
        self.symbol_for(taken)
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
                let Some(name) = self.ident(binding.name()) else {
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
                    ast::Expr::IdentExpr(ident) => self.ident(ident.name()),
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

                if !self.assignable(assign, name) {
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

    fn assignable(&mut self, node: &impl AstNode, name: &'c str) -> bool {
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
    fn declares(&self, node: &impl AstNode, name: &str) -> bool {
        self.symbols
            .binding(name)
            .is_some_and(|binding| binding.declared == node.syntax().text_range())
    }

    fn field_attrs(
        &mut self,
        fields: impl Iterator<Item = ast::StructField>,
    ) -> (ArrayAttribute<'c>, ArrayAttribute<'c>) {
        let mut names = Vec::new();
        let mut types = Vec::new();
        for field in fields {
            let (Some(name), Some(ty)) = (self.ident(field.name()), field.ty()) else {
                self.report(&field, "struct field is incomplete");
                continue;
            };

            names.push(StringAttribute::new(self.context, name).into());
            types.push(TypeAttribute::new(self.annotation_type(ty, &[])).into());
        }

        (
            ArrayAttribute::new(self.context, &names),
            ArrayAttribute::new(self.context, &types),
        )
    }

    fn annotation_type(
        &mut self,
        annotation: ast::TypeAnnotation,
        generics: &[&'c str],
    ) -> Type<'c> {
        let named = match annotation {
            ast::TypeAnnotation::NamedTypeAnnotation(named) => named,
            ast::TypeAnnotation::FuncTypeAnnotation(func) => {
                self.report(&func, "function types are not supported yet");
                return types::var(self.context);
            }
        };

        let Some(name) = self.ident(named.name()) else {
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

            let inner = self.annotation_type(inner, generics);
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
              }
              yzl.fn @spread params ["x"] (!yz.int64) -> !yz.int64 agg {
              ^bb0(%arg0: !yzl.var):
                %0 = yzl.call @max(%arg0) : (!yzl.var) -> !yzl.var {agg, callee_kind = "builtin"}
                %1 = yzl.call @min(%arg0) : (!yzl.var) -> !yzl.var {agg, callee_kind = "builtin"}
                %2 = yz.sub %0, %1 : !yzl.var, !yzl.var -> !yzl.var
                yzl.return %2 : !yzl.var
              }
              yzl.fn @upper params ["s"] (!yz.str) -> !yz.str external {
              }
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
              yzl.struct @Row ["a"] : [!yz.int64]
              yzl.table @t of @Row
              yzl.fn @f params ["x"] (!yz.int64) -> !yz.int64 {
              ^bb0(%arg0: !yzl.var):
                %1 = yz.constant_int 1
                %2 = yz.add %arg0, %1 : !yzl.var, !yz.int64 -> !yzl.var
                %3 = yz.constant_int 2
                %4 = yz.mul %arg0, %3 : !yzl.var, !yz.int64 -> !yzl.var
                yzl.return %4 : !yzl.var
              }
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
              }
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
              }
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
              }
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
              yzl.struct @Row ["a"] : [!yz.int64]
              yzl.table @t of @Row
              yzl.let @cap {
                %2 = yz.constant_int 1
                yzl.yield %2 : !yz.int64
              }
              yzl.let @step {
                %2 = yzl.call @cap() : () -> !yzl.var {callee_kind = "let"}
                %3 = yz.constant_int 10
                %4 = yz.add %2, %3 : !yzl.var, !yz.int64 -> !yzl.var
                yzl.yield %4 : !yzl.var
              }
              yzl.let @cap.1 {
                %2 = yz.constant_int 2
                yzl.yield %2 : !yz.int64
              }
              %0 = yzl.from @t
              %1 = yzl.where %0 {
              ^bb0(%arg0: !yzl.var):
                %2 = yzl.call @cap.1() : () -> !yzl.var {callee_kind = "let"}
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
            module.contains("yzl.let @base.1") && module.contains("yzl.from @base.1"),
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
              yzl.table @t of @Row
              yzl.struct @Row ["a"] : [!yz.int64]
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
