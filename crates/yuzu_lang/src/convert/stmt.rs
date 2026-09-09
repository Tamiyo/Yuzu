use melior::ir::{
    Attribute, Block, BlockLike, BlockRef, Region, RegionLike, Type, Value,
    attribute::{ArrayAttribute, FlatSymbolRefAttribute, StringAttribute, TypeAttribute},
};
use yuzu_ast::ast;
use yuzu_mlir::ods::yzl;

use crate::convert::{AstToYzl, Locals, ident_text};

impl<'c, 'd> AstToYzl<'c, 'd> {
    pub(super) fn convert_stmt<'a>(&self, block: BlockRef<'c, 'a>, stmt: &ast::Stmt) {
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
            unsupported => self.error(unsupported, "this statement is not supported yet"),
        }
    }

    fn convert_struct<'a>(&self, block: BlockRef<'c, 'a>, decl: &ast::StructStmt) {
        let Some(name) = ident_text(decl.name()) else {
            self.error(decl, "struct is missing its name");
            return;
        };

        let (names, types) = self.field_attrs(decl.fields());
        block.append_operation(
            yzl::r#struct(
                self.context,
                StringAttribute::new(self.context, &name),
                names,
                types,
                self.location(decl),
            )
            .into(),
        );
    }

    fn convert_table<'a>(&self, block: BlockRef<'c, 'a>, decl: &ast::TableStmt) {
        let Some(name) = ident_text(decl.name()) else {
            self.error(decl, "table is missing its name");
            return;
        };

        let row = match ident_text(decl.row_struct()) {
            Some(row) => row,
            // An inline table declares its row shape in place; give the shape
            // a struct of its own so the table can point at it.
            None => {
                let row = format!("{name}_row");
                let (names, types) = self.field_attrs(decl.inline_fields());
                block.append_operation(
                    yzl::r#struct(
                        self.context,
                        StringAttribute::new(self.context, &row),
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
                StringAttribute::new(self.context, &name),
                FlatSymbolRefAttribute::new(self.context, &row),
                self.location(decl),
            )
            .into(),
        );
    }

    fn convert_fn<'a>(&self, block: BlockRef<'c, 'a>, decl: &ast::FuncStmt) {
        self.convert_method(block, decl, &[]);
    }

    /// A function, with any type parameters its surroundings supply — a
    /// trait and its implementations declare `Self` implicitly.
    fn convert_method<'a>(&self, block: BlockRef<'c, 'a>, decl: &ast::FuncStmt, implicit: &[&str]) {
        let Some(name) = ident_text(decl.name()) else {
            self.error(decl, "function is missing its name");
            return;
        };

        let mut params: Vec<Attribute> = Vec::new();
        for param in decl.params() {
            match ident_text(param.name()) {
                Some(name) => params.push(StringAttribute::new(self.context, &name).into()),
                None => self.error(&param, "parameter is missing its name"),
            }
        }

        let generics: Vec<String> = implicit
            .iter()
            .map(|name| name.to_string())
            .chain(
                decl.type_params()
                    .filter_map(|param| ident_text(param.name())),
            )
            .collect();
        let signature = {
            let params: Vec<Type> = decl
                .params()
                .map(|param| self.annotation_type(param.ty(), &generics))
                .collect();
            let result = self.annotation_type(decl.result(), &generics);
            melior::ir::r#type::FunctionType::new(self.context, &params, &[result]).into()
        };

        let region = Region::new();
        if let Some(body) = decl.body() {
            let entry = region.append_block(Block::new(&[]));
            let mut locals = Locals::new();
            for stmt in body.stmts() {
                self.convert_body_stmt(entry, &mut locals, &stmt);
            }
        }

        let mut builder = yzl::FnOperationBuilder::new(self.context, self.location(decl))
            .sym_name(StringAttribute::new(self.context, &name))
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

        // One entry per (parameter, trait) pair, the way `rename` pairs its
        // from- and to-columns.
        let (subjects, traits) = self.bound_attrs(decl);
        if !subjects.is_empty() {
            builder = builder
                .bound_params(ArrayAttribute::new(self.context, &subjects))
                .bound_traits(ArrayAttribute::new(self.context, &traits));
        }

        block.append_operation(builder.build().into());
    }

    fn bound_attrs(&self, decl: &ast::FuncStmt) -> (Vec<Attribute<'c>>, Vec<Attribute<'c>>) {
        let mut subjects = Vec::new();
        let mut traits = Vec::new();
        for bound in decl.bounds() {
            let Some(subject) = ident_text(bound.subject()) else {
                self.error(&bound, "type bound is missing its subject");
                continue;
            };

            for trait_ref in bound.traits() {
                let Some(name) = ident_text(trait_ref.name()) else {
                    self.error(&trait_ref, "trait reference is missing its name");
                    continue;
                };

                subjects.push(StringAttribute::new(self.context, &subject).into());
                traits.push(FlatSymbolRefAttribute::new(self.context, &name).into());
            }
        }

        (subjects, traits)
    }

    /// A trait's methods are body-less `yzl.fn`s typed against `Self`, the
    /// parameter every trait declares implicitly.
    fn convert_trait<'a>(&self, block: BlockRef<'c, 'a>, decl: &ast::TraitStmt) {
        let Some(name) = ident_text(decl.name()) else {
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
                .sym_name(StringAttribute::new(self.context, &name))
                .body(region)
                .build()
                .into(),
        );
    }

    fn convert_impl<'a>(&self, block: BlockRef<'c, 'a>, decl: &ast::ImplStmt) {
        let Some(trait_name) = decl
            .trait_()
            .and_then(|trait_ref| ident_text(trait_ref.name()))
        else {
            self.error(decl, "`impl` is missing its trait");
            return;
        };

        let Some(target) = ident_text(decl.ty()) else {
            self.error(decl, "`impl` is missing its type name");
            return;
        };

        let region = Region::new();
        let body = region.append_block(Block::new(&[]));
        for method in decl.methods() {
            self.convert_method(body, &method, &["Self"]);
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

    /// A statement inside a function body: the full set the language
    /// takes there, except the declarations that do not nest yet.
    fn convert_body_stmt<'a>(
        &self,
        block: BlockRef<'c, 'a>,
        locals: &mut Locals<'c, 'a>,
        stmt: &ast::Stmt,
    ) {
        match stmt {
            ast::Stmt::LetStmt(binding) => {
                let Some(name) = ident_text(binding.name()) else {
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
                    ast::Expr::IdentExpr(ident) => ident_text(ident.name()),
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
            unsupported => {
                self.error(
                    unsupported,
                    "declarations inside functions are not supported yet",
                );
            }
        }
    }

    fn convert_let<'a>(&self, block: BlockRef<'c, 'a>, decl: &ast::LetStmt) {
        let Some(name) = ident_text(decl.name()) else {
            self.error(decl, "let binding is missing its name");
            return;
        };

        let Some(expr) = decl.expr() else {
            self.error(decl, "let binding is missing its expression");
            return;
        };

        let region = Region::new();
        let body = region.append_block(Block::new(&[]));
        let value = self.convert_expr(body, &Locals::new(), &expr);
        body.append_operation(yzl::r#yield(self.context, &[value], self.location(decl)).into());
        block.append_operation(
            yzl::r#let(
                self.context,
                region,
                StringAttribute::new(self.context, &name),
                self.location(decl),
            )
            .into(),
        );
    }

    fn field_attrs(
        &self,
        fields: impl Iterator<Item = ast::StructField>,
    ) -> (ArrayAttribute<'c>, ArrayAttribute<'c>) {
        let mut names = Vec::new();
        let mut types = Vec::new();
        for field in fields {
            let Some(name) = ident_text(field.name()) else {
                self.error(&field, "struct field is missing its name");
                continue;
            };

            names.push(StringAttribute::new(self.context, &name).into());
            types.push(TypeAttribute::new(self.annotation_type(field.ty(), &[])).into());
        }

        (
            ArrayAttribute::new(self.context, &names),
            ArrayAttribute::new(self.context, &types),
        )
    }

    fn annotation_type(
        &self,
        annotation: Option<ast::TypeAnnotation>,
        generics: &[String],
    ) -> Type<'c> {
        let name = annotation
            .and_then(|annotation| match annotation {
                ast::TypeAnnotation::NamedTypeAnnotation(named) => ident_text(named.name()),
                _ => None,
            })
            .unwrap_or_default();
        if generics.iter().any(|param| param == &name) {
            return yuzu_mlir::ParamType::new(self.context, &name).into();
        }

        match name.as_str() {
            "int64" => self.types.int64,
            "float64" => self.types.float64,
            "bool" => self.types.boolean,
            "str" => self.types.str,
            _ => self.types.var,
        }
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::convert::test_support::{convert, converted};

    #[test]
    fn converts_declarations() {
        expect![[r#"
            module {
              yzl.fn @f params ["x"] (!yz.int64) -> !yz.int64 {
                %0 = yzl.name "x" : !yzl.var
                %1 = yz.constant_int 2
                %2 = yz.mul %0, %1 : !yzl.var, !yz.int64 -> !yzl.var
                %3 = yz.constant_int 1
                %4 = yz.add %2, %3 : !yzl.var, !yz.int64 -> !yzl.var
                yzl.return %4 : !yzl.var
              }
              yzl.fn @spread params ["x"] (!yz.int64) -> !yz.int64 agg {
                %0 = yzl.name "x" : !yzl.var
                %1 = yzl.call @max(%0) : (!yzl.var) -> !yzl.var
                %2 = yzl.name "x" : !yzl.var
                %3 = yzl.call @min(%2) : (!yzl.var) -> !yzl.var
                %4 = yz.sub %1, %3 : !yzl.var, !yzl.var -> !yzl.var
                yzl.return %4 : !yzl.var
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
            %1 = yzl.name "x" : !yzl.var
            %2 = yz.constant_int 1
            %3 = yz.add %1, %2 : !yzl.var, !yz.int64 -> !yzl.var
            %4 = yzl.name "x" : !yzl.var
            %5 = yz.constant_int 2
            %6 = yz.mul %4, %5 : !yzl.var, !yz.int64 -> !yzl.var
            yzl.return %6 : !yzl.var
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
                  %0 = yzl.name "x" : !yzl.var
                  %1 = yzl.name "y" : !yzl.var
                  %2 = yz.add %0, %1 : !yzl.var, !yzl.var -> !yzl.var
                  yzl.return %2 : !yzl.var
                }
              }
              yzl.fn @id generics ["T"] where ["T"] : [@Add] params ["x"] (!yzl.param<"T">) -> !yzl.param<"T"> {
                %0 = yzl.name "x" : !yzl.var
                yzl.return %0 : !yzl.var
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
}
