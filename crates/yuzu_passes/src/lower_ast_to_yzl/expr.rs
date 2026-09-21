//! Expressions: each becomes `yz` scalar ops, or a `yzl.call` when a name
//! resolves to something callable.

use melior::ir::attribute::{
    FlatSymbolRefAttribute, FloatAttribute, IntegerAttribute, StringAttribute,
};
use melior::ir::r#type::IntegerType;
use melior::ir::{Attribute, BlockLike, BlockRef, Location, Type, Value};
use yuzu_ast::{AstNode, BinOp, UnaryOp, ast};
use yuzu_mlir::attributes::CmpPredicate;
use yuzu_mlir::ext::OperationExt;
use yuzu_mlir::ods::{yz, yzl};
use yuzu_mlir::types;

use crate::lower_ast_to_yzl::symbols::{BindingKind, Callable, FunctionKind, Lookup, Reference};
use crate::lower_ast_to_yzl::{AstToYzl, Locals};

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
            ast::Expr::IdentExpr(ident) => self.convert_ident(block, locals, ident, loc),
            ast::Expr::FieldAccessExpr(access) => {
                self.convert_field_access(block, locals, access, loc)
            }
            ast::Expr::BinaryExpr(binary) => self.convert_binary(block, locals, binary),
            ast::Expr::UnaryExpr(unary) => self.convert_unary(block, locals, unary, loc),
            ast::Expr::CallExpr(call) => self.convert_call(block, locals, call, loc),
            ast::Expr::ListExpr(list) => self.convert_list(block, locals, list, loc),
            ast::Expr::ParenExpr(paren) => match paren.expr() {
                Some(inner) => self.convert_expr(block, locals, &inner),
                None => self.report_and_hole(
                    block,
                    paren,
                    "parenthesized expression is missing its inner expression",
                    types::var(self.context),
                ),
            },
            // A query is a scope, and it ends where the expression ends.
            ast::Expr::Rel(rel) => {
                let value = self.convert_rel(block, rel);
                self.symbols.leave();
                value
            }
            ast::Expr::StructExpr(literal) => self.report_and_hole(
                block,
                literal,
                "struct literals are not supported yet",
                types::var(self.context),
            ),
        }
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

    fn convert_ident<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        locals: &Locals<'c, 'a>,
        ident: &ast::IdentExpr,
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        let Some(name) = self.read_name(ident.name()) else {
            return self.report_and_hole(
                block,
                ident,
                "identifier expression is missing its name",
                types::var(self.context),
            );
        };

        self.name_ref(block, locals, ident, Reference::bare(name), loc)
    }

    /// `t.a` is a qualified column reference, not a load.
    fn convert_field_access<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        locals: &Locals<'c, 'a>,
        access: &ast::FieldAccessExpr,
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        let base = match access.base() {
            Some(ast::Expr::IdentExpr(ident)) => self.read_name(ident.name()),
            Some(_) => {
                return self.report_and_hole(
                    block,
                    access,
                    "field access on an expression is not supported yet",
                    types::var(self.context),
                );
            }
            None => None,
        };

        let Some(base) = base else {
            return self.report_and_hole(
                block,
                access,
                "field access is missing its base",
                types::var(self.context),
            );
        };

        let Some(field) = self.read_name(access.field()) else {
            return self.report_and_hole(
                block,
                access,
                "field access is missing its field",
                types::var(self.context),
            );
        };

        let reference = Reference {
            qualifier: Some(base),
            name: field,
        };
        self.name_ref(block, locals, access, reference, loc)
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
                return self.report_and_hole(
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
                return self.report_and_hole(
                    block,
                    binary,
                    "binary expression is missing its right operand",
                    types::var(self.context),
                );
            }
        };

        let context = self.context;
        let var = types::var(context);
        let cmp = |predicate: CmpPredicate| {
            yz::cmp(
                context,
                var,
                lhs,
                rhs,
                StringAttribute::new(context, predicate.as_str()),
                loc,
            )
            .into()
        };

        let operation = match binary.op() {
            Some(BinOp::Add) => yz::add(context, var, lhs, rhs, loc).into(),
            Some(BinOp::Sub) => yz::sub(context, var, lhs, rhs, loc).into(),
            Some(BinOp::Mul) => yz::mul(context, var, lhs, rhs, loc).into(),
            Some(BinOp::Div) => yz::div(context, var, lhs, rhs, loc).into(),
            Some(BinOp::And) => yz::and(context, var, lhs, rhs, loc).into(),
            Some(BinOp::Or) => yz::or(context, var, lhs, rhs, loc).into(),
            Some(BinOp::Eq) => cmp(CmpPredicate::Equal),
            Some(BinOp::Neq) => cmp(CmpPredicate::NotEqual),
            Some(BinOp::Lt) => cmp(CmpPredicate::Less),
            Some(BinOp::Lte) => cmp(CmpPredicate::LessOrEqual),
            Some(BinOp::Gt) => cmp(CmpPredicate::Greater),
            Some(BinOp::Gte) => cmp(CmpPredicate::GreaterOrEqual),
            Some(BinOp::Pow) => return self.operator(block, binary, "pow", &[lhs, rhs], loc),
            Some(BinOp::ShiftLeft) => {
                return self.operator(block, binary, "shift_left", &[lhs, rhs], loc);
            }
            Some(BinOp::ShiftRight) => {
                return self.operator(block, binary, "shift_right", &[lhs, rhs], loc);
            }
            Some(BinOp::In) => return self.operator(block, binary, "in", &[lhs, rhs], loc),
            Some(BinOp::NotIn) => {
                let contains = self.operator(block, binary, "in", &[lhs, rhs], loc);
                yz::not(context, var, contains, loc).into()
            }
            None => {
                return self.report_and_hole(
                    block,
                    binary,
                    "binary expression is missing its operator",
                    var,
                );
            }
        };

        block.append_operation(operation).first_result()
    }

    fn convert_unary<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        locals: &Locals<'c, 'a>,
        unary: &ast::UnaryExpr,
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        let value = match unary.expr() {
            Some(expr) => self.convert_expr(block, locals, &expr),
            None => {
                return self.report_and_hole(
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
                return self.report_and_hole(
                    block,
                    unary,
                    "unary expression is missing its operator",
                    types::var(self.context),
                );
            }
        };

        block.append_operation(result).first_result()
    }

    fn convert_call<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        locals: &Locals<'c, 'a>,
        call: &ast::CallExpr,
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        let callee = match call.callee() {
            Some(ast::Expr::IdentExpr(ident)) => match self.read_name(ident.name()) {
                Some(callee) => callee,
                None => {
                    return self.report_and_hole(
                        block,
                        call,
                        "call is missing its callee",
                        types::var(self.context),
                    );
                }
            },
            Some(ast::Expr::FieldAccessExpr(access)) => {
                return self.convert_module_call(block, locals, call, &access, loc);
            }
            Some(_) => {
                return self.report_and_hole(
                    block,
                    call,
                    "calling an expression is not supported yet",
                    types::var(self.context),
                );
            }
            None => {
                return self.report_and_hole(
                    block,
                    call,
                    "call is missing its callee",
                    types::var(self.context),
                );
            }
        };

        let operands: Vec<Value> = call
            .args()
            .into_iter()
            .flat_map(|args| args.args())
            .map(|arg| self.convert_expr(block, locals, &arg))
            .collect();
        let Some(callable) = self.symbols.callable(callee, self.registry) else {
            let message = match self.symbols.kind(callee) {
                Some(kind) => format!("`{callee}` is a {}, not a function", kind.what()),
                None if self.symbols.is_method(callee) => {
                    format!("`{callee}` is a trait method, and calling one is not supported yet")
                }
                None => format!("unresolved identifier `{callee}`"),
            };
            return self.report_and_hole(block, call, &message, types::var(self.context));
        };

        self.check_arity(call, callee, callable, operands.len());
        self.call(block, callable, &operands, loc)
    }

    fn convert_module_call<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        locals: &Locals<'c, 'a>,
        call: &ast::CallExpr,
        access: &ast::FieldAccessExpr,
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        let base = match access.base() {
            Some(ast::Expr::IdentExpr(ident)) => self.read_name(ident.name()),
            _ => None,
        };

        let (Some(base), Some(name)) = (base, self.read_name(access.field())) else {
            return self.report_and_hole(
                block,
                call,
                "calling an expression is not supported yet",
                types::var(self.context),
            );
        };

        let Some(path) = self.symbols.module_of(base) else {
            return self.report_and_hole(
                block,
                call,
                &format!("`{base}` is not a module"),
                types::var(self.context),
            );
        };

        let operands: Vec<Value> = call
            .args()
            .into_iter()
            .flat_map(|args| args.args())
            .map(|arg| self.convert_expr(block, locals, &arg))
            .collect();

        let Some(binding) = self.read_export(call, path, name) else {
            return self.emit_hole(block, call.syntax().text_range(), types::var(self.context));
        };

        let BindingKind::Func(callable) = binding.kind else {
            return self.report_and_hole(
                block,
                call,
                &format!("`{name}` is a {}, not a function", binding.kind.what()),
                types::var(self.context),
            );
        };

        self.check_arity(call, name, callable, operands.len());
        self.call(block, callable, &operands, loc)
    }

    fn convert_list<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        locals: &Locals<'c, 'a>,
        list: &ast::ListExpr,
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        let values: Vec<Value> = list
            .elements()
            .map(|element| self.convert_expr(block, locals, &element))
            .collect();
        block
            .append_operation(
                yzl::list(self.context, types::var(self.context), &values, loc).into(),
            )
            .first_result()
    }

    /// A module-level `let` becomes a call for expansion to inline, since a
    /// stage region is isolated.
    fn name_ref<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        locals: &Locals<'c, 'a>,
        node: &impl AstNode,
        reference: Reference<'c>,
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        let name = reference.name;
        let message = match self.symbols.lookup(reference) {
            Lookup::Column(index) | Lookup::Param(index) => {
                return block
                    .argument(index)
                    .expect("the scope answered from the row this block was built for")
                    .into();
            }
            Lookup::Local { slot, .. } => return locals[slot],
            Lookup::Let(symbol) => {
                return self.call(block, Callable::let_binding(symbol), &[], loc);
            }
            Lookup::Ambiguous => {
                format!("column `{name}` is ambiguous; qualify it with a relation alias")
            }
            Lookup::NarrowedAway => {
                format!(
                    "column `{name}` is no longer in the row: an earlier stage narrowed it away"
                )
            }
            Lookup::NotAValue(what) => format!("`{reference}` is a {what}, not a value"),
            Lookup::Unknown => format!("unresolved identifier `{reference}`"),
        };

        self.unresolved_column(node, &message);
        self.emit_hole(block, node.syntax().text_range(), types::var(self.context))
    }

    fn call<'a>(
        &self,
        block: BlockRef<'c, 'a>,
        callable: Callable<'c>,
        operands: &[Value<'c, 'a>],
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        let mut builder = yzl::CallOperationBuilder::new(self.context, loc)
            .result(types::var(self.context))
            .operands(operands)
            .callee(FlatSymbolRefAttribute::new(self.context, callable.symbol))
            .callee_source(StringAttribute::new(self.context, callable.source.as_str()));
        if callable.kind == FunctionKind::Aggregate {
            builder = builder.agg(Attribute::unit(self.context));
        }

        block
            .append_operation(builder.build().into())
            .first_result()
    }

    fn operator<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        node: &impl AstNode,
        callee: &'c str,
        operands: &[Value<'c, 'a>],
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        let Some(callable) = self.symbols.operator(callee, self.registry) else {
            return self.report_and_hole(
                block,
                node,
                &format!("`{callee}` is not available"),
                types::var(self.context),
            );
        };

        self.check_arity(node, callee, callable, operands.len());
        self.call(block, callable, operands, loc)
    }

    fn check_arity(&mut self, call: &impl AstNode, callee: &str, callable: Callable, given: usize) {
        let (min, max) = (callable.min_args, callable.max_args);
        if given < min || given > max {
            let expected = if min == max {
                format!("{min}")
            } else {
                format!("{min} to {max}")
            };
            self.report(
                call,
                &format!("`{callee}` expects {expected} argument(s), found {given}"),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::test_support::{lower, lowered, rendered, reported};

    #[test]
    fn a_column_reference_says_what_the_row_carries() {
        expect![[r#"
            error: column `id` is ambiguous; qualify it with a relation alias
             --> test.yz:9:11
              |
            9 | |> select id
              |           ^^
              = note: the row carries `e.id`, `e.dept_id`, `d.id`, `d.name`
        "#]]
        .assert_eq(&reported(
            r#"
struct Employee { id: str, dept_id: int64 }
table employees = Employee
struct Department { id: str, name: str }
table departments = Department

from employees as e
|> inner join departments as d on e.dept_id == d.id
|> select id
"#,
        ));
    }

    #[test]
    fn a_narrowed_column_says_what_is_left() {
        expect![[r#"
            error: column `level` is no longer in the row: an earlier stage narrowed it away
             --> test.yz:7:10
              |
            7 | |> where level > 1
              |          ^^^^^
              = note: the row carries `id`
        "#]]
        .assert_eq(&reported(
            r#"
struct Row { id: str, level: int64 }
table t = Row

from t
|> select id
|> where level > 1
"#,
        ));
    }

    #[test]
    fn an_operator_and_its_name_resolve_together() {
        expect![[r#"
            module {
              yzl.struct @Row ["a"] : [!yz.int64] {sym_visibility = "private"}
              yzl.table @t of @Row {sym_visibility = "private"}
              yzl.fn @shift_left params ["x", "y"] (!yz.int64, !yz.int64) -> !yz.int64 {
              ^bb0(%arg0: !yzl.var, %arg1: !yzl.var):
                %2 = yz.add %arg0, %arg1 : !yzl.var, !yzl.var -> !yzl.var
                yzl.return %2 : !yzl.var
              } {sym_visibility = "private"}
              %0 = yzl.from @t
              %1 = yzl.select %0 as ["named", "operator"] {
              ^bb0(%arg0: !yzl.var):
                %2 = yz.constant_int 2
                %3 = yzl.call @shift_left(%arg0, %2) : (!yzl.var, !yz.int64) -> !yzl.var {callee_source = "fn"}
                %4 = yz.constant_int 2
                %5 = yzl.call @shift_left(%arg0, %4) : (!yzl.var, !yz.int64) -> !yzl.var {callee_source = "fn"}
                yzl.yield %3, %5 : !yzl.var, !yzl.var
              }
              yzl.output %1
            }
        "#]]
        .assert_eq(&lowered(
            r#"
struct Row { a: int64 }
table t = Row

def shift_left(x: int64, y: int64) -> int64 { return x + y }

from t
|> select shift_left(a, 2) as named, a << 2 as operator
"#,
        ));
    }

    #[test]
    fn a_value_does_not_stand_for_an_operator() {
        let module = lowered(
            r#"
struct Row { a: int64 }
table t = Row

let pow = 2

from t
|> select a ** 2 as p
"#,
        );
        assert!(
            module.contains(r#"yzl.call @pow(%arg0, %2)"#)
                && module.contains(r#"callee_source = "builtin""#),
            "the operator still reaches the builtin:\n{module}"
        );
    }

    #[test]
    fn reports_missing_pieces() {
        let context = yuzu_mlir::context();
        let lowered = lower(
            &context,
            &[(
                "test.yz",
                None,
                "struct Row { a: int64 }\ntable t = Row\n\nfrom t\n|> where a >\n",
            )],
        );
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
        .assert_eq(&rendered(&lowered.sources, &lowered.diagnostics));
        assert!(
            lowered
                .module
                .as_operation()
                .to_string()
                .contains("yzl.missing"),
            "the hole lowers to a yzl.missing value"
        );
    }
}
