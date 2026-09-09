use melior::ir::{
    Attribute, BlockLike, BlockRef, Location, Type, Value,
    attribute::{FlatSymbolRefAttribute, FloatAttribute, IntegerAttribute, StringAttribute},
};
use yuzu_ast::{BinOp, UnaryOp, ast};
use yuzu_mlir::ods::{yz, yzl};

use crate::convert::{AstToYzl, Locals, first_result, ident_text};

impl<'c, 'd> AstToYzl<'c, 'd> {
    pub(super) fn convert_expr<'a>(
        &self,
        block: BlockRef<'c, 'a>,
        locals: &Locals<'c, 'a>,
        expr: &ast::Expr,
    ) -> Value<'c, 'a> {
        let loc = self.location(expr);
        match expr {
            ast::Expr::Literal(literal) => self.convert_literal(block, literal),
            ast::Expr::IdentExpr(ident) => {
                let Some(name) = ident_text(ident.name()) else {
                    return self.missing(
                        block,
                        ident,
                        "identifier expression is missing its name",
                        self.types.var,
                    );
                };

                if let Some(&value) = locals.get(&name) {
                    return value;
                }

                self.name_ref(block, &name, loc)
            }
            ast::Expr::FieldAccessExpr(access) => {
                // `t.a` is one qualified reference, not a load: keep the
                // dotted name whole for resolution.
                let base = match access.base() {
                    Some(ast::Expr::IdentExpr(ident)) => ident_text(ident.name()),
                    Some(_) => {
                        return self.missing(
                            block,
                            access,
                            "field access on an expression is not supported yet",
                            self.types.var,
                        );
                    }
                    None => None,
                };

                let Some(base) = base else {
                    return self.missing(
                        block,
                        access,
                        "field access is missing its base",
                        self.types.var,
                    );
                };

                let Some(field) = ident_text(access.field()) else {
                    return self.missing(
                        block,
                        access,
                        "field access is missing its field",
                        self.types.var,
                    );
                };

                self.name_ref(block, &format!("{base}.{field}"), loc)
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
                            self.types.var,
                        );
                    }
                };

                let result = match unary.op() {
                    Some(UnaryOp::Neg) => yz::neg(self.context, self.types.var, value, loc).into(),
                    Some(UnaryOp::Not) => yz::not(self.context, self.types.var, value, loc).into(),
                    Some(UnaryOp::Pos) => return value,
                    None => {
                        return self.missing(
                            block,
                            unary,
                            "unary expression is missing its operator",
                            self.types.var,
                        );
                    }
                };

                first_result(block.append_operation(result))
            }
            ast::Expr::CallExpr(call) => {
                let callee = match call.callee() {
                    Some(ast::Expr::IdentExpr(ident)) => match ident_text(ident.name()) {
                        Some(callee) => callee,
                        None => {
                            return self.missing(
                                block,
                                call,
                                "call is missing its callee",
                                self.types.var,
                            );
                        }
                    },

                    Some(_) => {
                        return self.missing(
                            block,
                            call,
                            "calling an expression is not supported yet",
                            self.types.var,
                        );
                    }
                    None => {
                        return self.missing(
                            block,
                            call,
                            "call is missing its callee",
                            self.types.var,
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
                first_result(
                    block.append_operation(
                        yzl::call(
                            self.context,
                            self.types.var,
                            &operands,
                            FlatSymbolRefAttribute::new(self.context, &callee),
                            loc,
                        )
                        .into(),
                    ),
                )
            }
            ast::Expr::ListExpr(list) => {
                let elements: Vec<ast::Expr> = list.elements().collect();
                let values: Vec<Value> = elements
                    .iter()
                    .map(|element| self.convert_expr(block, locals, element))
                    .collect();
                first_result(
                    block.append_operation(
                        yzl::list(self.context, self.types.var, &values, loc).into(),
                    ),
                )
            }
            ast::Expr::ParenExpr(paren) => match paren.expr() {
                Some(inner) => self.convert_expr(block, locals, &inner),
                None => self.missing(
                    block,
                    paren,
                    "parenthesized expression is missing its inner expression",
                    self.types.var,
                ),
            },

            ast::Expr::Rel(rel) => self.convert_rel(block, rel),
            unsupported => self.missing(
                block,
                unsupported,
                "this expression is not supported yet",
                self.types.var,
            ),
        }
    }

    fn convert_binary<'a>(
        &self,
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
                    self.types.var,
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
                    self.types.var,
                );
            }
        };

        let context = self.context;
        let cmp = |predicate| {
            yz::cmp(
                context,
                self.types.var,
                lhs,
                rhs,
                StringAttribute::new(context, predicate),
                loc,
            )
            .into()
        };

        // Operators without a yz op yet lower as calls by name, which is what
        // they are before the registry resolves them.
        let call = |callee| {
            yzl::call(
                context,
                self.types.var,
                &[lhs, rhs],
                FlatSymbolRefAttribute::new(context, callee),
                loc,
            )
            .into()
        };

        let operation = match binary.op() {
            Some(BinOp::Add) => yz::add(context, self.types.var, lhs, rhs, loc).into(),
            Some(BinOp::Sub) => yz::sub(context, self.types.var, lhs, rhs, loc).into(),
            Some(BinOp::Mul) => yz::mul(context, self.types.var, lhs, rhs, loc).into(),
            Some(BinOp::Div) => yz::div(context, self.types.var, lhs, rhs, loc).into(),
            Some(BinOp::And) => yz::and(context, self.types.var, lhs, rhs, loc).into(),
            Some(BinOp::Or) => yz::or(context, self.types.var, lhs, rhs, loc).into(),
            Some(BinOp::Eq) => cmp("eq"),
            Some(BinOp::Neq) => cmp("ne"),
            Some(BinOp::Lt) => cmp("lt"),
            Some(BinOp::Lte) => cmp("le"),
            Some(BinOp::Gt) => cmp("gt"),
            Some(BinOp::Gte) => cmp("ge"),
            Some(BinOp::Pow) => call("pow"),
            Some(BinOp::ShiftLeft) => call("shift_left"),
            Some(BinOp::ShiftRight) => call("shift_right"),
            Some(BinOp::In) => call("in"),
            Some(BinOp::NotIn) => {
                let contains = first_result(block.append_operation(call("in")));
                yz::not(context, self.types.var, contains, loc).into()
            }
            None => {
                return self.missing(
                    block,
                    binary,
                    "binary expression is missing its operator",
                    self.types.var,
                );
            }
        };

        first_result(block.append_operation(operation))
    }

    fn convert_literal<'a>(
        &self,
        block: BlockRef<'c, 'a>,
        literal: &ast::Literal,
    ) -> Value<'c, 'a> {
        let loc = self.location(literal);
        let operation = match literal {
            ast::Literal::IntLiteral(int) => yz::constant_int(
                self.context,
                self.types.int64,
                IntegerAttribute::new(self.types.i64, int.value().unwrap_or_default() as i64),
                loc,
            )
            .into(),
            ast::Literal::FloatLiteral(float) => yz::constant_float(
                self.context,
                self.types.float64,
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
                self.types.boolean,
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
                self.types.str,
                StringAttribute::new(self.context, &string.value().unwrap_or_default()),
                loc,
            )
            .into(),
        };

        first_result(block.append_operation(operation))
    }

    fn name_ref<'a>(
        &self,
        block: BlockRef<'c, 'a>,
        name: &str,
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        first_result(
            block.append_operation(
                yzl::_name(
                    self.context,
                    self.types.var,
                    StringAttribute::new(self.context, name),
                    loc,
                )
                .into(),
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::convert::test_support::convert;

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
