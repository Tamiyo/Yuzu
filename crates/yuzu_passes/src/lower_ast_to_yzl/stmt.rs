use melior::ir::{
    Attribute, Block, BlockLike, BlockRef, Location, Region, RegionLike, Type, Value,
    attribute::{ArrayAttribute, FlatSymbolRefAttribute, StringAttribute, TypeAttribute},
};
use yuzu_ast::{AstNode, ast};
use yuzu_mlir::attributes::CalleeKind;
use yuzu_mlir::ods::yzl;
use yuzu_mlir::{ListType, ParamType, StructType};

use crate::lower_ast_to_yzl::symbols::{Binding, Callable, unqualified};
use crate::lower_ast_to_yzl::{AstToYzl, Locals};
use yuzu_mlir::types;

impl<'c, 'd> AstToYzl<'c, 'd> {
    /// Registers every declaration up front so references can be forward. A
    /// `let` binds in order instead, once its body is emitted.
    pub(super) fn hoist(&mut self, root: &ast::Root) {
        for stmt in root.stmts() {
            match &stmt {
                ast::Stmt::StructStmt(decl) => {
                    let Some(name) = self.ident(decl.name()) else {
                        continue;
                    };
                    let fields = decl.fields().filter_map(|f| self.ident(f.name())).collect();
                    self.declare(decl, name, Binding::Struct { fields });
                }
                ast::Stmt::TableStmt(decl) => {
                    let Some(name) = self.ident(decl.name()) else {
                        continue;
                    };

                    // An inline table declares its row shape in place; a
                    // named one names a struct the program declares.
                    let row = if decl.inline_fields().next().is_some() {
                        unqualified(
                            decl.inline_fields()
                                .filter_map(|field| self.ident(field.name()))
                                .collect(),
                        )
                    } else {
                        let Some(declared) = self.ident(decl.row_struct()) else {
                            continue;
                        };

                        match self.symbols.binding(declared) {
                            Some(Binding::Struct { fields }) => unqualified(fields.clone()),
                            Some(_) | None => {
                                self.error(decl, &format!("`{declared}` is not a struct"));
                                continue;
                            }
                        }
                    };

                    self.declare(decl, name, Binding::Relation { row });
                }
                ast::Stmt::TraitStmt(decl) => {
                    let Some(name) = self.ident(decl.name()) else {
                        continue;
                    };
                    let methods = decl
                        .methods()
                        .filter_map(|m| self.ident(m.name()))
                        .collect();
                    self.declare(decl, name, Binding::Trait { methods });
                }
                ast::Stmt::FuncStmt(decl) => {
                    let Some(name) = self.ident(decl.name()) else {
                        continue;
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
                        Binding::Func(Callable {
                            kind,
                            min_args: arity,
                            max_args: arity,
                        }),
                    );
                }
                ast::Stmt::ImplStmt(_)
                | ast::Stmt::LetStmt(_)
                | ast::Stmt::ExprStmt(_)
                | ast::Stmt::BlockStmt(_)
                | ast::Stmt::AssignStmt(_)
                | ast::Stmt::ReturnStmt(_) => {}
            }
        }
    }

    /// Binds a module-level name, reporting a second declaration of one.
    fn declare(&mut self, node: &impl AstNode, name: &'c str, binding: Binding<'c>) {
        self.check_duplicate(node, binding.what(), name);
        self.symbols.bind(name, binding);
    }

    fn check_duplicate(&mut self, node: &impl AstNode, what: &str, name: &str) {
        if self.symbols.binding(name).is_some() {
            self.error(node, &format!("the {what} `{name}` is already defined"));
        }
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
                    if matches!(expr, ast::Expr::Rel(_)) {
                        self.symbols.leave();
                    }
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
        self.convert_method(block, decl, &[]);
    }

    /// A function, with any type parameters its surroundings supply — a
    /// trait and its implementations declare `Self` implicitly.
    fn convert_method<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        decl: &ast::FuncStmt,
        implicit: &[&'c str],
    ) {
        let Some(name) = self.ident(decl.name()) else {
            self.error(decl, "function is missing its name");
            return;
        };

        let mut params: Vec<Attribute> = Vec::new();
        for param in decl.params() {
            match self.ident(param.name()) {
                Some(name) => params.push(StringAttribute::new(self.context, name).into()),
                None => self.error(&param, "parameter is missing its name"),
            }
        }

        let generics: Vec<&'c str> = implicit
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
                    && !matches!(
                        self.symbols.binding(trait_name),
                        Some(Binding::Trait { .. })
                    )
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
            let mut locals = Locals::new();
            for stmt in body.stmts() {
                self.convert_body_stmt(entry, &mut locals, &stmt);
            }

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
            self.convert_method(body, &method, &["Self"]);
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

        if !matches!(
            self.symbols.binding(trait_name),
            Some(Binding::Trait { .. })
        ) {
            self.error(decl, &format!("unknown trait `{trait_name}`"));
        }

        if !self.symbols.is_type_name(target) {
            self.error(decl, &format!("unknown type `{target}`"));
        }

        let region = Region::new();
        let body = region.append_block(Block::new(&[]));
        for method in decl.methods() {
            self.convert_method(body, &method, &["Self"]);
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
                locals.insert(name, value);
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
                locals.insert(name, value);
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
            ast::Stmt::BlockStmt(nested) => {
                let mut scope = locals.clone();
                for stmt in nested.stmts() {
                    self.convert_body_stmt(block, &mut scope, &stmt);
                }
            }
            ast::Stmt::StructStmt(_)
            | ast::Stmt::TraitStmt(_)
            | ast::Stmt::ImplStmt(_)
            | ast::Stmt::FuncStmt(_)
            | ast::Stmt::TableStmt(_) => {
                self.error(stmt, "declarations inside functions are not supported yet");
            }
        }
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
                self.symbols.bind(name, Binding::Relation { row });
                value
            }
            _ => {
                let value = self.convert_expr(body, &Locals::new(), &expr);
                self.symbols.bind(name, Binding::Let);
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

        match name {
            "int64" => types::int64(self.context),
            "float64" => types::float64(self.context),
            "bool" => types::boolean(self.context),
            "str" => types::str(self.context),
            _ if self.symbols.is_type_name(name) => StructType::new(self.context, name).into(),
            _ => {
                self.error(&named, &format!("unknown type `{name}`"));
                types::var(self.context)
            }
        }
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
                %0 = yzl.call @max(%arg0) : (!yzl.var) -> !yzl.var {callee_kind = "builtin"}
                %1 = yzl.call @min(%arg0) : (!yzl.var) -> !yzl.var {callee_kind = "builtin"}
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
