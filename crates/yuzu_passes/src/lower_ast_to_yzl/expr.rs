use melior::ir::{
    Attribute, BlockLike, BlockRef, Location, Type, Value,
    attribute::{FlatSymbolRefAttribute, FloatAttribute, IntegerAttribute, StringAttribute},
};
use yuzu_ast::{AstNode, BinOp, UnaryOp, ast};
use yuzu_mlir::ods::{yz, yzl};

use crate::lower_ast_to_yzl::{AstToYzl, Locals};
use melior::ir::r#type::IntegerType;
use yuzu_mlir::attributes::{CalleeKind, CmpPredicate};

use crate::lower_ast_to_yzl::resolve::{CallError, Lookup};
use yuzu_mlir::ext::OperationExt;
use yuzu_mlir::types;

impl<'c, 'd> AstToYzl<'c, 'd> {
    pub(super) fn convert_expr<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        locals: &Locals<'c, 'a>,
        expr: &ast::Expr,
    ) -> Value<'c, 'a> {
        let loc = self.location(expr);
        match expr {
            ast::Expr::Literal(literal) => self.convert_literal(block, literal),
            ast::Expr::IdentExpr(ident) => {
                let Some(name) = self.ident(ident.name()) else {
                    return self.missing(
                        block,
                        ident,
                        "identifier expression is missing its name",
                        types::var(self.context),
                    );
                };

                if let Some(&value) = locals.get(name) {
                    return value;
                }

                self.name_ref(block, ident, name, loc)
            }
            ast::Expr::FieldAccessExpr(access) => {
                // `t.a` is one qualified reference, not a load: keep the
                // dotted name whole for resolution.
                let base = match access.base() {
                    Some(ast::Expr::IdentExpr(ident)) => self.ident(ident.name()),
                    Some(_) => {
                        return self.missing(
                            block,
                            access,
                            "field access on an expression is not supported yet",
                            types::var(self.context),
                        );
                    }
                    None => None,
                };

                let Some(base) = base else {
                    return self.missing(
                        block,
                        access,
                        "field access is missing its base",
                        types::var(self.context),
                    );
                };

                let Some(field) = self.ident(access.field()) else {
                    return self.missing(
                        block,
                        access,
                        "field access is missing its field",
                        types::var(self.context),
                    );
                };

                self.name_ref(block, access, &format!("{base}.{field}"), loc)
            }
            ast::Expr::BinaryExpr(binary) => self.convert_binary(block, locals, binary),
            ast::Expr::UnaryExpr(unary) => {
                let value = match unary.expr() {
                    Some(expr) => self.convert_expr(block, locals, &expr),
                    None => {
                        return self.missing(
                            block,
                            unary,
                            "unary expression is missing its operand",
                            types::var(self.context),
                        );
                    }
                };

                let result = match unary.op() {
                    Some(UnaryOp::Neg) => {
                        yz::neg(self.context, types::var(self.context), value, loc).into()
                    }
                    Some(UnaryOp::Not) => {
                        yz::not(self.context, types::var(self.context), value, loc).into()
                    }
                    Some(UnaryOp::Pos) => return value,
                    None => {
                        return self.missing(
                            block,
                            unary,
                            "unary expression is missing its operator",
                            types::var(self.context),
                        );
                    }
                };

                block.append_operation(result).first_result()
            }
            ast::Expr::CallExpr(call) => {
                let callee = match call.callee() {
                    Some(ast::Expr::IdentExpr(ident)) => match self.ident(ident.name()) {
                        Some(callee) => callee,
                        None => {
                            return self.missing(
                                block,
                                call,
                                "call is missing its callee",
                                types::var(self.context),
                            );
                        }
                    },

                    Some(_) => {
                        return self.missing(
                            block,
                            call,
                            "calling an expression is not supported yet",
                            types::var(self.context),
                        );
                    }
                    None => {
                        return self.missing(
                            block,
                            call,
                            "call is missing its callee",
                            types::var(self.context),
                        );
                    }
                };

                let args: Vec<ast::Expr> = call
                    .args()
                    .into_iter()
                    .flat_map(|args| args.args())
                    .collect();
                let operands: Vec<Value> = args
                    .iter()
                    .map(|arg| self.convert_expr(block, locals, arg))
                    .collect();
                let callable = self.resolver.callable(callee, self.registry);
                let kind = match callable {
                    Ok(callable) => {
                        let (min, max) = (callable.min_args, callable.max_args);
                        if operands.len() < min || operands.len() > max {
                            let expected = if min == max {
                                format!("{min}")
                            } else {
                                format!("{min} to {max}")
                            };
                            self.error(
                                call,
                                &format!(
                                    "`{callee}` expects {expected} arguments, got {}",
                                    operands.len()
                                ),
                            );
                        }

                        callable.kind
                    }
                    // A trait's methods live in its implementations, so no
                    // module-level name reaches one. Choosing between them is
                    // dispatch, which nothing does yet — and "unknown" would
                    // send the reader looking for a declaration that is there.
                    Err(CallError::TraitMethod) => {
                        return self.missing(
                            block,
                            call,
                            &format!(
                                "`{callee}` is a trait method, and calling one is not supported yet"
                            ),
                            types::var(self.context),
                        );
                    }
                    Err(CallError::Unknown) => {
                        return self.missing(
                            block,
                            call,
                            &format!("unknown function `{callee}`"),
                            types::var(self.context),
                        );
                    }
                };

                self.call(block, callee, kind, &operands, loc)
            }
            ast::Expr::ListExpr(list) => {
                let elements: Vec<ast::Expr> = list.elements().collect();
                let values: Vec<Value> = elements
                    .iter()
                    .map(|element| self.convert_expr(block, locals, element))
                    .collect();
                block
                    .append_operation(
                        yzl::list(self.context, types::var(self.context), &values, loc).into(),
                    )
                    .first_result()
            }
            ast::Expr::ParenExpr(paren) => match paren.expr() {
                Some(inner) => self.convert_expr(block, locals, &inner),
                None => self.missing(
                    block,
                    paren,
                    "parenthesized expression is missing its inner expression",
                    types::var(self.context),
                ),
            },

            ast::Expr::Rel(rel) => self.convert_rel(block, rel),
            ast::Expr::StructExpr(literal) => self.missing(
                block,
                literal,
                "struct literals are not supported yet",
                types::var(self.context),
            ),
        }
    }

    fn convert_binary<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        locals: &Locals<'c, 'a>,
        binary: &ast::BinaryExpr,
    ) -> Value<'c, 'a> {
        let loc = self.location(binary);
        let lhs = match binary.lhs() {
            Some(expr) => self.convert_expr(block, locals, &expr),
            None => {
                return self.missing(
                    block,
                    binary,
                    "binary expression is missing its left operand",
                    types::var(self.context),
                );
            }
        };

        let rhs = match binary.rhs() {
            Some(expr) => self.convert_expr(block, locals, &expr),
            None => {
                return self.missing(
                    block,
                    binary,
                    "binary expression is missing its right operand",
                    types::var(self.context),
                );
            }
        };

        let context = self.context;
        let cmp = |predicate: CmpPredicate| {
            yz::cmp(
                context,
                types::var(self.context),
                lhs,
                rhs,
                StringAttribute::new(context, predicate.as_str()),
                loc,
            )
            .into()
        };

        // Operators without a yz op yet lower as calls by name, which is what
        // they are before the registry resolves them.

        let operation = match binary.op() {
            Some(BinOp::Add) => yz::add(context, types::var(self.context), lhs, rhs, loc).into(),
            Some(BinOp::Sub) => yz::sub(context, types::var(self.context), lhs, rhs, loc).into(),
            Some(BinOp::Mul) => yz::mul(context, types::var(self.context), lhs, rhs, loc).into(),
            Some(BinOp::Div) => yz::div(context, types::var(self.context), lhs, rhs, loc).into(),
            Some(BinOp::And) => yz::and(context, types::var(self.context), lhs, rhs, loc).into(),
            Some(BinOp::Or) => yz::or(context, types::var(self.context), lhs, rhs, loc).into(),
            Some(BinOp::Eq) => cmp(CmpPredicate::Equal),
            Some(BinOp::Neq) => cmp(CmpPredicate::NotEqual),
            Some(BinOp::Lt) => cmp(CmpPredicate::Less),
            Some(BinOp::Lte) => cmp(CmpPredicate::LessOrEqual),
            Some(BinOp::Gt) => cmp(CmpPredicate::Greater),
            Some(BinOp::Gte) => cmp(CmpPredicate::GreaterOrEqual),
            Some(BinOp::Pow) => return self.builtin(block, "pow", &[lhs, rhs], loc),
            Some(BinOp::ShiftLeft) => return self.builtin(block, "shift_left", &[lhs, rhs], loc),
            Some(BinOp::ShiftRight) => {
                return self.builtin(block, "shift_right", &[lhs, rhs], loc);
            }
            Some(BinOp::In) => return self.builtin(block, "in", &[lhs, rhs], loc),
            Some(BinOp::NotIn) => {
                let contains = self.builtin(block, "in", &[lhs, rhs], loc);
                yz::not(context, types::var(self.context), contains, loc).into()
            }
            None => {
                return self.missing(
                    block,
                    binary,
                    "binary expression is missing its operator",
                    types::var(self.context),
                );
            }
        };

        block.append_operation(operation).first_result()
    }

    fn convert_literal<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        literal: &ast::Literal,
    ) -> Value<'c, 'a> {
        let loc = self.location(literal);
        let operation = match literal {
            ast::Literal::IntLiteral(int) => yz::constant_int(
                self.context,
                types::int64(self.context),
                IntegerAttribute::new(
                    IntegerType::new(self.context, 64).into(),
                    int.value().unwrap_or_default() as i64,
                ),
                loc,
            )
            .into(),
            ast::Literal::FloatLiteral(float) => yz::constant_float(
                self.context,
                types::float64(self.context),
                FloatAttribute::new(
                    self.context,
                    Type::float64(self.context),
                    float.value().unwrap_or_default(),
                ),
                loc,
            )
            .into(),
            ast::Literal::BoolLiteral(boolean) => yz::constant_bool(
                self.context,
                types::boolean(self.context),
                Attribute::parse(
                    self.context,
                    if boolean.value().unwrap_or_default() {
                        "true"
                    } else {
                        "false"
                    },
                )
                .expect("a bool attribute parses"),
                loc,
            )
            .into(),
            ast::Literal::StringLiteral(string) => yz::constant_str(
                self.context,
                types::str(self.context),
                StringAttribute::new(self.context, &string.value().unwrap_or_default()),
                loc,
            )
            .into(),
        };

        block.append_operation(operation).first_result()
    }

    /// A name in expression position, resolved where it stands. A column or
    /// parameter is a block argument of the block being built; a module-level
    /// `let` is a call for expansion to inline, since this block cannot reach
    /// a value outside itself.
    fn name_ref<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        node: &impl AstNode,
        name: &str,
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        let (lookup, in_query) = {
            let resolver = &self.resolver;
            (resolver.lookup(name), resolver.in_query())
        };
        match lookup {
            Lookup::Column(index) | Lookup::Param(index) => block
                .argument(index)
                .expect("the resolver answered from the row this block was built for")
                .into(),
            Lookup::Let(symbol) => self.call(block, symbol, CalleeKind::Let, &[], loc),
            Lookup::Ambiguous => self.missing(
                block,
                node,
                &format!("`{name}` is ambiguous"),
                types::var(self.context),
            ),
            Lookup::NarrowedAway => self.missing(
                block,
                node,
                &format!("`{name}` is no longer in the row: an earlier stage narrowed it away"),
                types::var(self.context),
            ),
            Lookup::Unknown => {
                let what = if in_query { "column" } else { "name" };
                self.missing(
                    block,
                    node,
                    &format!("unknown {what} `{name}`"),
                    types::var(self.context),
                )
            }
        }
    }

    /// A call the emitter has already classified: the kind is decided here,
    /// so every pass after reads it rather than looking the callee up again.
    fn call<'a>(
        &self,
        block: BlockRef<'c, 'a>,
        callee: &str,
        kind: CalleeKind,
        operands: &[Value<'c, 'a>],
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        block
            .append_operation(
                yzl::CallOperationBuilder::new(self.context, loc)
                    .result(types::var(self.context))
                    .operands(operands)
                    .callee(FlatSymbolRefAttribute::new(self.context, callee))
                    .callee_kind(StringAttribute::new(self.context, kind.as_str()))
                    .build()
                    .into(),
            )
            .first_result()
    }

    /// The operators the parser lowers to calls are registry builtins.
    fn builtin<'a>(
        &self,
        block: BlockRef<'c, 'a>,
        callee: &str,
        operands: &[Value<'c, 'a>],
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        self.call(block, callee, CalleeKind::Builtin, operands, loc)
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::lower_ast_to_yzl::test_support::convert;

    /// The HIR lowerer's missing-piece mechanics, ported: a hole in the
    /// parse converts to a reported diagnostic and a `yzl.missing` value.
    #[test]
    fn reports_missing_pieces() {
        use yuzu_diagnostics::diagnostics::printer::DiagnosticPrinter;

        let source = "struct Row { a: int64 }\ntable t = Row\n\nfrom t\n|> where a >\n";
        let context = yuzu_mlir::context();
        let (module, sources, diagnostics) = convert(&context, "test.yz", source);
        let printer = DiagnosticPrinter::new(&sources);
        let rendered: Vec<String> = diagnostics
            .diagnostics()
            .iter()
            .map(|diagnostic| printer.print(diagnostic))
            .collect();
        expect![[r#"
            error: expected expression, found end of input
             --> test.yz:5:13
              |
            5 | |> where a >
              | 

            error: binary expression is missing its right operand
             --> test.yz:5:10
              |
            5 | |> where a >
              |          ^^^
        "#]]
        .assert_eq(&rendered.join("\n"));
        assert!(
            module.as_operation().to_string().contains("yzl.missing"),
            "the hole converts to a yzl.missing value"
        );
    }
}
