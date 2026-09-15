use melior::ir::{
    Attribute, Block, BlockLike, BlockRef, Location, Region, RegionLike, Type, Value,
    attribute::{ArrayAttribute, FlatSymbolRefAttribute, StringAttribute, TypeAttribute},
};
use yuzu_ast::{AstNode, ast};
use yuzu_mlir::attributes::CalleeKind;
use yuzu_mlir::ods::yzl;
use yuzu_mlir::{ListType, ParamType, StructType};

use crate::lower_ast_to_yzl::symbols::{Callable, Kind, Row};
use crate::lower_ast_to_yzl::{AstToYzl, Locals};
use yuzu_mlir::types;

/// Where a `fn` is written. A trait and its implementations declare `Self`
/// implicitly, and a trait's methods are the one kind that stands as a
/// signature with nothing to run.
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
    /// Registers every declaration up front so references can be forward. A
    /// `let` binds in order instead, once its body is emitted.
    ///
    /// Structs go in before tables, because a table's row is the fields of
    /// the struct it names — one pass in source order would only find a
    /// struct declared above the table that reads it.
    pub(super) fn hoist(&mut self, root: &ast::Root) {
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
                | ast::Stmt::ReturnStmt(_) => {}
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
        self.declare(decl, name, Kind::Struct { fields });
    }

    fn hoist_table(&mut self, decl: &ast::TableStmt) {
        let Some(name) = self.ident(decl.name()) else {
            return;
        };

        // An inline table declares its row shape in place; a named one
        // names a struct the program declares.
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
                Some(Kind::Struct { fields }) => Row::from(fields.clone()),
                Some(_) | None => {
                    self.error(decl, &format!("`{declared}` is not a struct"));
                    return;
                }
            }
        };

        self.declare(decl, name, Kind::Relation { row });
    }

    fn hoist_trait(&mut self, decl: &ast::TraitStmt) {
        let Some(name) = self.ident(decl.name()) else {
            return;
        };

        let methods = decl
            .methods()
            .filter_map(|m| self.ident(m.name()))
            .collect();
        self.declare(decl, name, Kind::Trait { methods });
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
        self.declare(
            decl,
            name,
            Kind::Func(Callable {
                kind,
                min_args: arity,
                max_args: arity,
                agg: decl.is_agg(),
            }),
        );
    }

    /// Binds a module-level name, reporting a second declaration of one.
    fn declare(&mut self, node: &impl AstNode, name: &'c str, kind: Kind<'c>) {
        self.check_duplicate(node, kind.what(), name);
        self.symbols.bind(name, kind, node.syntax().text_range());
    }

    /// The note says where the other declaration is rather than which came
    /// first: hoisting takes the declarations out of source order, so which
    /// of the two is reported is not the order they were written in.
    fn check_duplicate(&mut self, node: &impl AstNode, what: &str, name: &str) {
        let Some(declared) = self.symbols.binding(name).map(|binding| binding.declared) else {
            return;
        };

        let other = self.position(declared);
        self.error_at_noting(
            node.syntax().text_range(),
            &format!("the {what} `{name}` is already defined"),
            format!("also declared at {other}"),
        );
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
                if let Some(expr) = stmt.expr() {
                    self.convert_expr(block, &Locals::new(), &expr);
                }
            }
            ast::Stmt::BlockStmt(stmt) => self.error(stmt, "a block is not a top-level statement"),
            ast::Stmt::AssignStmt(stmt) => {
                self.error(stmt, "an assignment is not a top-level statement")
            }
            ast::Stmt::ReturnStmt(stmt) => {
                self.error(stmt, "a return is not a top-level statement")
            }
        }
    }

    fn convert_struct<'a>(&mut self, block: BlockRef<'c, 'a>, decl: &ast::StructStmt) {
        let Some(name) = self.ident(decl.name()) else {
            self.error(decl, "struct is missing its name");
            return;
        };

        let (names, types) = self.field_attrs(decl.fields());
        block.append_operation(
            yzl::r#struct(
                self.context,
                StringAttribute::new(self.context, name),
                names,
                types,
                self.location(decl),
            )
            .into(),
        );
    }

    fn convert_table<'a>(&mut self, block: BlockRef<'c, 'a>, decl: &ast::TableStmt) {
        let Some(name) = self.ident(decl.name()) else {
            self.error(decl, "table is missing its name");
            return;
        };

        let row = match self.ident(decl.row_struct()) {
            Some(row) => row,
            None => {
                let row = self.intern(&format!("{name}_row"));
                let (names, types) = self.field_attrs(decl.inline_fields());
                block.append_operation(
                    yzl::r#struct(
                        self.context,
                        StringAttribute::new(self.context, row),
                        names,
                        types,
                        self.location(decl),
                    )
                    .into(),
                );
                row
            }
        };

        block.append_operation(
            yzl::table(
                self.context,
                StringAttribute::new(self.context, name),
                FlatSymbolRefAttribute::new(self.context, row),
                self.location(decl),
            )
            .into(),
        );
    }

    fn convert_fn<'a>(&mut self, block: BlockRef<'c, 'a>, decl: &ast::FuncStmt) {
        self.convert_method(block, decl, Declared::AtModule);
    }

    /// A function, and where it is written: that decides the type parameters
    /// its surroundings supply, and whether it needs a body.
    fn convert_method<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        decl: &ast::FuncStmt,
        declared: Declared,
    ) {
        let Some(name) = self.ident(decl.name()) else {
            self.error(decl, "function is missing its name");
            return;
        };

        // `external` says the target has the function, so there is nothing
        // to write; everything else is a definition, and the one place a
        // signature stands alone is a trait, which declares what its
        // implementations must supply.
        match (decl.is_external(), decl.body().is_some()) {
            (true, true) => self.error(decl, "an external function cannot have a body"),
            (false, false) if declared != Declared::InTrait => {
                self.error(decl, "function is missing its body");
            }
            (true, false) | (false, true) | (false, false) => {}
        }

        let mut params: Vec<Attribute> = Vec::new();
        for param in decl.params() {
            match self.ident(param.name()) {
                Some(name) => params.push(StringAttribute::new(self.context, name).into()),
                None => self.error(&param, "parameter is missing its name"),
            }
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

        let signature = {
            let mut param_types: Vec<Type> = Vec::new();
            for param in decl.params() {
                let ty = match param.ty() {
                    Some(ty) => self.annotation_type(ty, &generics),
                    None => {
                        self.error(&param, "parameter is missing its type");
                        types::var(self.context)
                    }
                };
                param_types.push(ty);
            }

            let result = match decl.result() {
                Some(result) => self.annotation_type(result, &generics),
                None => types::var(self.context),
            };
            melior::ir::r#type::FunctionType::new(self.context, &param_types, &[result]).into()
        };

        for bound in decl.bounds() {
            if let Some(subject) = self.ident(bound.subject())
                && !generics.contains(&subject)
            {
                self.error(&bound, &format!("unknown type parameter `{subject}`"));
            }

            for trait_ref in bound.traits() {
                if let Some(trait_name) = self.ident(trait_ref.name())
                    && !matches!(self.symbols.kind(trait_name), Some(Kind::Trait { .. }))
                {
                    self.error(&trait_ref, &format!("unknown trait `{trait_name}`"));
                }
            }
        }

        let region = Region::new();
        if let Some(body) = decl.body() {
            let loc = self.location(decl);
            let arguments: Vec<(Type<'c>, Location<'c>)> = decl
                .params()
                .map(|_| (types::var(self.context), loc))
                .collect();
            let entry = region.append_block(Block::new(&arguments));
            let names: Vec<&'c str> = decl.params().filter_map(|p| self.ident(p.name())).collect();
            self.symbols.enter_function(names);
            self.convert_block(entry, &mut Locals::new(), &body);
            self.symbols.leave();
        }

        let mut builder = yzl::FnOperationBuilder::new(self.context, self.location(decl))
            .sym_name(StringAttribute::new(self.context, name))
            .params(ArrayAttribute::new(self.context, &params))
            .signature(TypeAttribute::new(signature))
            .body(region);

        if decl.is_agg() {
            builder = builder.agg(Attribute::unit(self.context));
        }

        if decl.is_external() {
            builder = builder.external(Attribute::unit(self.context));
        }

        if !generics.is_empty() {
            let names: Vec<Attribute> = generics
                .iter()
                .map(|name| StringAttribute::new(self.context, name).into())
                .collect();
            builder = builder.type_params(ArrayAttribute::new(self.context, &names));
        }

        let (subjects, traits) = self.bound_attrs(decl);
        if !subjects.is_empty() {
            builder = builder
                .bound_params(ArrayAttribute::new(self.context, &subjects))
                .bound_traits(ArrayAttribute::new(self.context, &traits));
        }

        block.append_operation(builder.build().into());
    }

    /// One entry per (parameter, trait) pair.
    fn bound_attrs(&mut self, decl: &ast::FuncStmt) -> (Vec<Attribute<'c>>, Vec<Attribute<'c>>) {
        let mut subjects = Vec::new();
        let mut traits = Vec::new();
        for bound in decl.bounds() {
            let Some(subject) = self.ident(bound.subject()) else {
                self.error(&bound, "type bound is missing its subject");
                continue;
            };

            for trait_ref in bound.traits() {
                let Some(name) = self.ident(trait_ref.name()) else {
                    self.error(&trait_ref, "trait reference is missing its name");
                    continue;
                };

                subjects.push(StringAttribute::new(self.context, subject).into());
                traits.push(FlatSymbolRefAttribute::new(self.context, name).into());
            }
        }

        (subjects, traits)
    }

    fn convert_trait<'a>(&mut self, block: BlockRef<'c, 'a>, decl: &ast::TraitStmt) {
        let Some(name) = self.ident(decl.name()) else {
            self.error(decl, "trait is missing its name");
            return;
        };

        let region = Region::new();
        let body = region.append_block(Block::new(&[]));
        for method in decl.methods() {
            self.convert_method(body, &method, Declared::InTrait);
        }

        block.append_operation(
            yzl::TraitOperationBuilder::new(self.context, self.location(decl))
                .sym_name(StringAttribute::new(self.context, name))
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
            self.error(decl, "`impl` is missing its trait");
            return;
        };

        let Some(target) = self.ident(decl.ty()) else {
            self.error(decl, "`impl` is missing its type name");
            return;
        };

        if !matches!(self.symbols.kind(trait_name), Some(Kind::Trait { .. })) {
            self.error(decl, &format!("unknown trait `{trait_name}`"));
        }

        if self.scalar_type(target).is_none() && !self.symbols.is_struct(target) {
            self.error(decl, &format!("unknown type `{target}`"));
        }

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

    /// A lexical block: its `let`s bind for as long as it lasts, so it is a
    /// scope of its own — which is what makes a nested block's bindings go
    /// out of scope at its end.
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
                    self.error(binding, "let binding is missing its name");
                    return;
                };

                let Some(expr) = binding.expr() else {
                    self.error(binding, "let binding is missing its expression");
                    return;
                };

                let value = self.convert_expr(block, locals, &expr);
                self.bind_local(locals, name, value);
            }
            ast::Stmt::AssignStmt(assign) => {
                let target = assign.target().and_then(|target| match target {
                    ast::Expr::IdentExpr(ident) => self.ident(ident.name()),
                    _ => None,
                });

                let Some(name) = target else {
                    self.error(assign, "assignment is missing its target");
                    return;
                };

                let Some(value) = assign.value() else {
                    self.error(assign, "assignment is missing its value");
                    return;
                };

                let value = self.convert_expr(block, locals, &value);
                self.bind_local(locals, name, value);
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
                self.error(stmt, "declarations inside functions are not supported yet");
            }
        }
    }

    /// The value goes in the traversal's stack, and the name goes in the
    /// scope, pointing at the slot it landed in.
    fn bind_local<'a>(&mut self, locals: &mut Locals<'c, 'a>, name: &'c str, value: Value<'c, 'a>) {
        self.symbols.bind_local(name, locals.len());
        locals.push(value);
    }

    fn convert_let<'a>(&mut self, block: BlockRef<'c, 'a>, decl: &ast::LetStmt) {
        let Some(name) = self.ident(decl.name()) else {
            self.error(decl, "let binding is missing its name");
            return;
        };

        let Some(expr) = decl.expr() else {
            self.error(decl, "let binding is missing its expression");
            return;
        };

        self.check_duplicate(decl, "binding", name);
        let annotation = decl
            .type_annotation()
            .map(|annotation| self.annotation_type(annotation, &[]));
        let region = Region::new();
        let body = region.append_block(Block::new(&[]));
        // A query binds its row, for `from`; a value binds its name.
        let value = match &expr {
            ast::Expr::Rel(rel) => {
                let value = self.convert_rel(body, rel);
                let row = self.symbols.row().clone();
                self.symbols.leave();
                self.symbols
                    .bind(name, Kind::Relation { row }, decl.syntax().text_range());
                value
            }
            _ => {
                let value = self.convert_expr(body, &Locals::new(), &expr);
                self.symbols
                    .bind(name, Kind::Let, decl.syntax().text_range());
                value
            }
        };

        body.append_operation(yzl::r#yield(self.context, &[value], self.location(decl)).into());
        let mut builder = yzl::LetOperationBuilder::new(self.context, self.location(decl))
            .sym_name(StringAttribute::new(self.context, name))
            .body(region);
        if let Some(annotation) = annotation {
            builder = builder.annotation(TypeAttribute::new(annotation));
        }

        block.append_operation(builder.build().into());
    }

    fn field_attrs(
        &mut self,
        fields: impl Iterator<Item = ast::StructField>,
    ) -> (ArrayAttribute<'c>, ArrayAttribute<'c>) {
        let mut names = Vec::new();
        let mut types = Vec::new();
        for field in fields {
            let (Some(name), Some(ty)) = (self.ident(field.name()), field.ty()) else {
                self.error(&field, "struct field is incomplete");
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
                self.error(&func, "function types are not supported yet");
                return types::var(self.context);
            }
        };

        let Some(name) = self.ident(named.name()) else {
            self.error(&named, "type is missing its name");
            return types::var(self.context);
        };

        if generics.contains(&name) {
            return ParamType::new(self.context, name).into();
        }

        if name == "List" {
            let mut args = named.args();
            let (Some(inner), None) = (args.next(), args.next()) else {
                self.error(&named, "`List` takes exactly one type argument");
                return types::var(self.context);
            };

            let inner = self.annotation_type(inner, generics);
            return ListType::new(self.context, inner).into();
        }

        if let Some(scalar) = self.scalar_type(name) {
            return scalar;
        }

        if self.symbols.is_struct(name) {
            return StructType::new(self.context, name).into();
        }

        self.error(&named, &format!("unknown type `{name}`"));
        types::var(self.context)
    }

    /// The type a primitive's name stands for — the one list of them.
    fn scalar_type(&self, name: &str) -> Option<Type<'c>> {
        Some(match name {
            "int64" => types::int64(self.context),
            "float64" => types::float64(self.context),
            "bool" => types::boolean(self.context),
            "str" => types::str(self.context),
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::lower_ast_to_yzl::test_support::{convert, converted};

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
        .assert_eq(&converted(
            r#"
fn f(x: int64) -> int64 {
    let doubled = x * 2
    return doubled + 1
}

agg fn spread(x: int64) -> int64 {
    return max(x) - min(x)
}

external fn upper(s: str) -> str
"#,
        ));
    }

    /// Function bodies take expression statements, not just bindings and
    /// returns.
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
        "#]].assert_eq(&converted(
            "struct Row { a: int64 }\ntable t = Row\n\nfn f(x: int64) -> int64 {\n    x + 1\n    let y = x * 2\n    return y\n}\n\nfrom t\n",
        ));
    }

    /// Traits, implementations, and bounded generics reach the substrate:
    /// `Self` is the parameter a trait declares implicitly, and an `impl`'s
    /// target is a name for resolution to bind.
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
        "#]].assert_eq(&converted(
        "trait Add {\n    fn add(x: Self, y: Self) -> Self\n}\n\nimpl Add for int64 {\n    fn add(x: int64, y: int64) -> int64 { return x + y }\n}\n\nfn id[T](x: T) -> T where T: Add { return x }\n",
    ));
    }

    /// An unsupported construct is an error, never a silent note.
    #[test]
    fn reports_unsupported_constructs() {
        use yuzu_diagnostics::diagnostics::printer::DiagnosticPrinter;

        let context = yuzu_mlir::context();
        let source = "fn f(x: int64) -> int64 {\n    struct S { a: int64 }\n    return x\n}\n";
        let (_, sources, diagnostics) = convert(&context, "test.yz", source);
        let printer = DiagnosticPrinter::new(&sources);
        let rendered: Vec<String> = diagnostics
            .diagnostics()
            .iter()
            .map(|diagnostic| printer.print(diagnostic))
            .collect();
        expect![[r#"
        error: declarations inside functions are not supported yet
         --> test.yz:2:5
          |
        2 |     struct S { a: int64 }
          |     ^^^^^^^^^^^^^^^^^^^^^
    "#]]
        .assert_eq(&rendered.join("\n"));
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
        .assert_eq(&converted("let ids: List[int64] = [1, 3]\n"));
    }

    /// A name can only be declared once, and the note says where it was —
    /// the printer underlines one line, and the first declaration is rarely
    /// on it.
    #[test]
    fn a_redeclaration_says_where_the_other_one_is() {
        use crate::lower_ast_to_yzl::test_support::reported;

        let context = yuzu_mlir::context();
        expect![[r#"
            error: the relation `Row` is already defined
             --> test.yz:3:1
              |
            3 | table Row = Row
              | ^^^^^^^^^^^^^^^
              = note: also declared at test.yz:1:1
        "#]]
        .assert_eq(&reported(
            &context,
            "struct Row { a: int64 }\n\ntable Row = Row\n",
        ));
    }

    /// Declarations are hoisted, so a table may name a struct declared
    /// below it — one pass in source order reported three errors for this.
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
        .assert_eq(&converted(
            "table t = Row\nstruct Row { a: int64 }\n\nfrom t |> select a\n",
        ));
    }

    /// `external` says the target has the function, and a signature with no
    /// body is a declaration only a trait can make — so neither shape is
    /// expressible anywhere else.
    #[test]
    fn a_body_is_required_unless_the_target_or_a_trait_supplies_it() {
        use crate::lower_ast_to_yzl::test_support::reported;

        let context = yuzu_mlir::context();
        expect![[r#"
            error: an external function cannot have a body
             --> test.yz:1:1
              |
            1 | external fn upper(s: str) -> str { return s }
              | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

            error: function is missing its body
             --> test.yz:2:1
              |
            2 | fn lower(s: str) -> str
              | ^^^^^^^^^^^^^^^^^^^^^^^
        "#]]
        .assert_eq(&reported(
            &context,
            "external fn upper(s: str) -> str { return s }\nfn lower(s: str) -> str\n",
        ));

        // A trait's methods are signatures, and an implementation of one is
        // a definition like any other.
        expect![[r#"
            error: function is missing its body
             --> test.yz:5:23
              |
            5 | impl Show for int64 { fn show(x: Self) -> str }
              |                       ^^^^^^^^^^^^^^^^^^^^^^^
        "#]]
        .assert_eq(&reported(
            &context,
            "trait Show {\n    fn show(x: Self) -> str\n}\n\nimpl Show for int64 { fn show(x: Self) -> str }\n",
        ));
    }

    #[test]
    fn reports_an_unknown_type() {
        use yuzu_diagnostics::diagnostics::printer::DiagnosticPrinter;

        let context = yuzu_mlir::context();
        let source = "fn f(x: Nope) -> int64 { return 1 }\n";
        let (_, sources, diagnostics) = convert(&context, "test.yz", source);
        let printer = DiagnosticPrinter::new(&sources);
        let rendered: Vec<String> = diagnostics
            .diagnostics()
            .iter()
            .map(|diagnostic| printer.print(diagnostic))
            .collect();
        expect![[r#"
        error: unknown type `Nope`
         --> test.yz:1:9
          |
        1 | fn f(x: Nope) -> int64 { return 1 }
          |         ^^^^
    "#]]
        .assert_eq(&rendered.join("\n"));
    }
}
