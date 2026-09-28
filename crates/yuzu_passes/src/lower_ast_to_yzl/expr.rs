//! Expressions: each becomes `yz` scalar ops, or a `yzl.call` when a name
//! resolves to something callable.

use melior::ir::attribute::{
    BoolAttribute, FlatSymbolRefAttribute, FloatAttribute, IntegerAttribute, StringAttribute,
};
use melior::ir::{Attribute, BlockLike, BlockRef, Location, Type, Value};
use yuzu_ast::{AstNode, BinOp, UnaryOp, ast};
use yuzu_mlir::attributes::CmpPredicate;
use yuzu_mlir::ir::attribute::integer::IntegerAttributeExt;
use yuzu_mlir::ir::operation::OperationExt;
use yuzu_mlir::ods::{yz, yzl};
use yuzu_mlir::types::{BoolType, Float64Type, Int64Type, StrType, UnresolvedType};

use crate::lower_ast_to_yzl::symbols::{BindingKind, Callable, FunctionKind, Lookup, Reference};
use crate::lower_ast_to_yzl::{AstToYzl, Locals};
use crate::operators;

impl<'c> AstToYzl<'c, '_> {
    pub(super) fn convert_expr<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        locals: &Locals<'c, 'a>,
        expr: &ast::Expr,
    ) -> Value<'c, 'a> {
        match expr {
            ast::Expr::Literal(literal) => self.convert_literal(block, literal),
            ast::Expr::IdentExpr(ident) => self.convert_ident(block, locals, ident),
            ast::Expr::FieldAccessExpr(access) => self.convert_field_access(block, locals, access),
            ast::Expr::BinaryExpr(binary) => self.convert_binary(block, locals, binary),
            ast::Expr::UnaryExpr(unary) => self.convert_unary(block, locals, unary),
            ast::Expr::CallExpr(call) => self.convert_call(block, locals, call),
            ast::Expr::ListExpr(list) => self.convert_list(block, locals, list),
            ast::Expr::ParenExpr(paren) => self.convert_paren_expr(block, locals, paren),
            ast::Expr::Pipeline(pipeline) => self.convert_query(block, pipeline).0,
            ast::Expr::StructExpr(literal) => self.report_and_hole(
                block,
                literal,
                "struct literals are not supported yet",
                UnresolvedType::get(self.context),
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
                Int64Type::get(self.context),
                IntegerAttribute::from_i64(self.context, int.value().unwrap_or_default() as i64),
                loc,
            )
            .into(),
            ast::Literal::FloatLiteral(float) => yz::constant_float(
                self.context,
                Float64Type::get(self.context),
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
                BoolType::get(self.context),
                BoolAttribute::new(self.context, boolean.value().unwrap_or_default()),
                loc,
            )
            .into(),
            ast::Literal::StringLiteral(string) => yz::constant_str(
                self.context,
                StrType::get(self.context),
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
    ) -> Value<'c, 'a> {
        let Some(name) = self.read_ident(ident.name()) else {
            return self.parser_hole(
                block,
                ident,
                "identifier expression is missing its name",
                UnresolvedType::get(self.context),
            );
        };

        let loc = self.location(ident);
        self.name_ref(block, locals, ident, Reference::unqualified(name), loc)
    }

    /// `t.a` is a qualified column reference, not a load.
    fn convert_field_access<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        locals: &Locals<'c, 'a>,
        access: &ast::FieldAccessExpr,
    ) -> Value<'c, 'a> {
        let loc = self.location(access);

        let base = match access.base() {
            Some(ast::Expr::IdentExpr(ident)) => self.read_ident(ident.name()),
            Some(_) => {
                return self.report_and_hole(
                    block,
                    access,
                    "field access on an expression is not supported yet",
                    UnresolvedType::get(self.context),
                );
            }
            None => None,
        };

        let Some(base) = base else {
            return self.parser_hole(
                block,
                access,
                "field access is missing its base",
                UnresolvedType::get(self.context),
            );
        };

        let Some(field) = self.read_ident(access.field()) else {
            return self.parser_hole(
                block,
                access,
                "field access is missing its field",
                UnresolvedType::get(self.context),
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
                return self.parser_hole(
                    block,
                    binary,
                    "binary expression is missing its left operand",
                    UnresolvedType::get(self.context),
                );
            }
        };

        let rhs = match binary.rhs() {
            Some(expr) => self.convert_expr(block, locals, &expr),
            None => {
                return self.parser_hole(
                    block,
                    binary,
                    "binary expression is missing its right operand",
                    UnresolvedType::get(self.context),
                );
            }
        };

        let var = UnresolvedType::get(self.context);
        let cmp = |predicate: CmpPredicate| {
            yz::cmp(
                self.context,
                var,
                lhs,
                rhs,
                StringAttribute::new(self.context, predicate.as_str()),
                loc,
            )
            .into()
        };

        let operation = match binary.op() {
            Some(BinOp::Add) => yz::add(self.context, var, lhs, rhs, loc).into(),
            Some(BinOp::Sub) => yz::sub(self.context, var, lhs, rhs, loc).into(),
            Some(BinOp::Mul) => yz::mul(self.context, var, lhs, rhs, loc).into(),
            Some(BinOp::Div) => yz::div(self.context, var, lhs, rhs, loc).into(),
            Some(BinOp::Rem) => {
                self.symbols.refer_operator(&operators::REM);
                yz::rem(self.context, var, lhs, rhs, loc).into()
            }
            Some(BinOp::And) => yz::and(self.context, var, lhs, rhs, loc).into(),
            Some(BinOp::Or) => yz::or(self.context, var, lhs, rhs, loc).into(),
            Some(BinOp::Eq) => cmp(CmpPredicate::Equal),
            Some(BinOp::Neq) => cmp(CmpPredicate::NotEqual),
            Some(BinOp::Lt) => cmp(CmpPredicate::Less),
            Some(BinOp::Lte) => cmp(CmpPredicate::LessOrEqual),
            Some(BinOp::Gt) => cmp(CmpPredicate::Greater),
            Some(BinOp::Gte) => cmp(CmpPredicate::GreaterOrEqual),
            Some(BinOp::Pow) => {
                self.symbols.refer_operator(&operators::POW);
                yz::pow(self.context, var, lhs, rhs, loc).into()
            }
            Some(BinOp::ShiftLeft) => {
                self.symbols.refer_operator(&operators::SHL);
                yz::shl(self.context, var, lhs, rhs, loc).into()
            }
            Some(BinOp::ShiftRight) => {
                self.symbols.refer_operator(&operators::SHR);
                yz::shr(self.context, var, lhs, rhs, loc).into()
            }
            Some(BinOp::In) => return self.operator(block, binary, "in", &[lhs, rhs], loc),
            Some(BinOp::NotIn) => {
                let contains = self.operator(block, binary, "in", &[lhs, rhs], loc);
                yz::not(self.context, var, contains, loc).into()
            }
            None => {
                return self.parser_hole(
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
    ) -> Value<'c, 'a> {
        let loc = self.location(unary);

        let value = match unary.expr() {
            Some(expr) => self.convert_expr(block, locals, &expr),
            None => {
                return self.parser_hole(
                    block,
                    unary,
                    "unary expression is missing its operand",
                    UnresolvedType::get(self.context),
                );
            }
        };

        let result = match unary.op() {
            Some(UnaryOp::Neg) => {
                yz::neg(self.context, UnresolvedType::get(self.context), value, loc).into()
            }
            Some(UnaryOp::Not) => {
                yz::not(self.context, UnresolvedType::get(self.context), value, loc).into()
            }
            Some(UnaryOp::Pos) => return value,
            None => {
                return self.parser_hole(
                    block,
                    unary,
                    "unary expression is missing its operator",
                    UnresolvedType::get(self.context),
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
    ) -> Value<'c, 'a> {
        let loc = self.location(call);

        let callee = match call.callee() {
            Some(ast::Expr::IdentExpr(ident)) => match self.read_ident(ident.name()) {
                Some(callee) => callee,
                None => {
                    return self.parser_hole(
                        block,
                        call,
                        "call is missing its callee",
                        UnresolvedType::get(self.context),
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
                    UnresolvedType::get(self.context),
                );
            }
            None => {
                return self.parser_hole(
                    block,
                    call,
                    "call is missing its callee",
                    UnresolvedType::get(self.context),
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
                Some(BindingKind::Pending) => format!("`{callee}` is bound further down the file"),
                Some(kind) => format!("`{callee}` is a {kind}, not a function"),
                None if self.symbols.is_method(callee) => {
                    format!("`{callee}` is a trait method, and calling one is not supported yet")
                }
                None => format!("unresolved identifier `{callee}`"),
            };
            return self.report_and_hole(block, call, &message, UnresolvedType::get(self.context));
        };

        self.check_arity(call, callee, &callable, operands.len());
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
            Some(ast::Expr::IdentExpr(ident)) => self.read_ident(ident.name()),
            _ => None,
        };

        let (Some(base), Some(name)) = (base, self.read_ident(access.field())) else {
            return self.report_and_hole(
                block,
                call,
                "calling an expression is not supported yet",
                UnresolvedType::get(self.context),
            );
        };

        let Some(path) = self.symbols.module_of(base) else {
            return self.report_and_hole(
                block,
                call,
                &format!("`{base}` is not a module"),
                UnresolvedType::get(self.context),
            );
        };

        let operands: Vec<Value> = call
            .args()
            .into_iter()
            .flat_map(|args| args.args())
            .map(|arg| self.convert_expr(block, locals, &arg))
            .collect();

        let Some((at, binding)) = self.read_export(call, path, name) else {
            return self.emit_hole(
                block,
                call.syntax().text_range(),
                UnresolvedType::get(self.context),
            );
        };

        let Some(callable) = self.symbols.callable_in(at) else {
            return self.report_and_hole(
                block,
                call,
                &format!("`{name}` is a {}, not a function", binding.kind),
                UnresolvedType::get(self.context),
            );
        };

        self.check_arity(call, name, &callable, operands.len());
        self.call(block, callable, &operands, loc)
    }

    fn convert_list<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        locals: &Locals<'c, 'a>,
        list: &ast::ListExpr,
    ) -> Value<'c, 'a> {
        let loc = self.location(list);

        let values: Vec<Value> = list
            .elements()
            .map(|element| self.convert_expr(block, locals, &element))
            .collect();
        block
            .append_operation(
                yzl::list(
                    self.context,
                    UnresolvedType::get(self.context),
                    &values,
                    loc,
                )
                .into(),
            )
            .first_result()
    }

    fn convert_paren_expr<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        locals: &Locals<'c, 'a>,
        paren: &ast::ParenExpr,
    ) -> Value<'c, 'a> {
        match paren.expr() {
            Some(inner) => self.convert_expr(block, locals, &inner),
            None => self.parser_hole(
                block,
                paren,
                "parenthesized expression is missing its inner expression",
                UnresolvedType::get(self.context),
            ),
        }
    }

    /// A module-level `let` becomes a call for expansion to inline, since a
    /// stage region is isolated.
    fn name_ref<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        locals: &Locals<'c, 'a>,
        node: &impl AstNode,
        reference: Reference<'_>,
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        let name = reference.name;
        let message = match self.symbols.lookup(reference) {
            Lookup::Column(index) => {
                return block
                    .argument(index)
                    .expect("the scope answered from the row this block was built for")
                    .into();
            }
            Lookup::Local(slot) => {
                let load = yzl::load(
                    self.context,
                    UnresolvedType::get(self.context),
                    locals[slot],
                    loc,
                );
                return block.append_operation(load.into()).first_result();
            }
            Lookup::Let(symbol) => {
                return self.call(block, Callable::constant(symbol), &[], loc);
            }
            Lookup::Lost => {
                return self.emit_hole(
                    block,
                    node.syntax().text_range(),
                    UnresolvedType::get(self.context),
                );
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
            Lookup::NotYet => format!("`{reference}` is bound further down the file"),
            Lookup::Unknown => format!("unresolved identifier `{reference}`"),
        };

        self.unresolved_column(node, &message);
        self.emit_hole(
            block,
            node.syntax().text_range(),
            UnresolvedType::get(self.context),
        )
    }

    fn call<'a>(
        &self,
        block: BlockRef<'c, 'a>,
        callable: Callable<'c>,
        operands: &[Value<'c, 'a>],
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        let mut builder = yzl::CallOperationBuilder::new(self.context, loc)
            .result(UnresolvedType::get(self.context))
            .operands(operands)
            .callee(FlatSymbolRefAttribute::new(self.context, callable.symbol))
            .callee_source(StringAttribute::new(self.context, callable.source.as_str()));
        if callable.kind == FunctionKind::Aggregate {
            builder = builder.is_agg(Attribute::unit(self.context));
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
                UnresolvedType::get(self.context),
            );
        };

        self.check_arity(node, callee, &callable, operands.len());
        self.call(block, callable, operands, loc)
    }

    fn check_arity(
        &mut self,
        call: &impl AstNode,
        callee: &str,
        callable: &Callable<'c>,
        given: usize,
    ) {
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
        expect![[r"
            error: column `id` is ambiguous; qualify it with a relation alias
             --> test.yz:9:11
              |
            9 | |> select id
              |           ^^
              = note: the row carries `e.id`, `e.dept_id`, `d.id`, `d.name`
        "]]
        .assert_eq(&reported(
            r"
struct Employee { id: str, dept_id: int64 }
table employees = Employee
struct Department { id: str, name: str }
table departments = Department

from employees as e
|> inner join departments as d on e.dept_id == d.id
|> select id
",
        ));
    }

    #[test]
    fn a_narrowed_column_says_what_is_left() {
        expect![[r"
            error: column `level` is no longer in the row: an earlier stage narrowed it away
             --> test.yz:7:10
              |
            7 | |> where level > 1
              |          ^^^^^
              = note: the row carries `id`
        "]]
        .assert_eq(&reported(
            r"
struct Row { id: str, level: int64 }
table t = Row

from t
|> select id
|> where level > 1
",
        ));
    }

    #[test]
    fn an_operator_is_its_op_whatever_its_name_is_bound_to() {
        let module = lowered(
            r"
struct Row { a: int64 }
table t = Row

def shift_left(x: int64, y: int64) -> int64 { return x + y }
let pow = 2

from t
|> select shift_left(a, 2) as named, a << 2 as shifted, a ** 2 as squared
",
        );
        assert_eq!(
            module.matches("yzl.call @shift_left(").count(),
            1,
            "{module}"
        );
        assert!(module.contains("yz.shl %arg0"), "{module}");
        assert!(module.contains("yz.pow %arg0"), "{module}");
    }

    #[test]
    fn a_missing_piece_lowers_to_a_hole() {
        let context = yuzu_mlir::context();
        let lowered = lower(
            &context,
            &[(
                "test.yz",
                None,
                "struct Row { a: int64 }\ntable t = Row\n\nfrom t\n|> where a >\n",
            )],
        );
        expect![[r"
            error: expected expression, found end of input
             --> test.yz:5:13
              |
            5 | |> where a >
              | 
        "]]
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
