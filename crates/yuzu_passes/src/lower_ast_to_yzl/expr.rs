//! Expressions: each becomes `yz` scalar ops, or a `yzl.call` when a name
//! resolves to something callable.

use melior::Context;
use melior::ir::attribute::{
    ArrayAttribute, BoolAttribute, FlatSymbolRefAttribute, FloatAttribute, IntegerAttribute,
    StringAttribute,
};
use melior::ir::{Attribute, BlockLike, BlockRef, Location, Type, Value};
use text_size::TextRange;
use yuzu_ast::ast::{self, AstNode, BinOp, UnaryOp};
use yuzu_mlir::attributes::CmpPredicate;
use yuzu_mlir::ir::attribute::integer::IntegerAttributeExt;
use yuzu_mlir::ir::operation::OperationExt;
use yuzu_mlir::ods::{yz, yzl};
use yuzu_mlir::types::{BoolType, Float64Type, Int64Type, ListType, StrType, UnresolvedType};

use crate::lower_ast_to_yzl::symbols::{BindingKind, Callable, FunctionKind, Lookup, Reference};
use crate::lower_ast_to_yzl::{AstToYzl, Locals};
use crate::operators;

/// The value a literal writes, as the attribute of the constant op that
/// holds it.
#[derive(Clone, Copy)]
enum Constant<'c> {
    Int(IntegerAttribute<'c>),
    Float(FloatAttribute<'c>),
    Bool(BoolAttribute<'c>),
    Str(StringAttribute<'c>),
}

impl<'c> Constant<'c> {
    fn ty(self, context: &'c Context) -> Type<'c> {
        match self {
            Constant::Int(_) => Int64Type::new(context).into(),
            Constant::Float(_) => Float64Type::new(context).into(),
            Constant::Bool(_) => BoolType::new(context).into(),
            Constant::Str(_) => StrType::new(context).into(),
        }
    }
}

impl<'c> From<Constant<'c>> for Attribute<'c> {
    fn from(constant: Constant<'c>) -> Self {
        match constant {
            Constant::Int(value) => value.into(),
            Constant::Float(value) => value.into(),
            Constant::Bool(value) => value.into(),
            Constant::Str(value) => value.into(),
        }
    }
}

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
            ast::Expr::StructExpr(literal) => self.hole_and_report(
                block,
                literal,
                "struct literals are not supported yet",
                UnresolvedType::new(self.context).into(),
            ),
        }
    }

    fn convert_literal<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        literal: &ast::Literal,
    ) -> Value<'c, 'a> {
        let loc = self.location(literal);
        if let ast::Literal::IntLiteral(int) = literal
            && self.read_int64(int).is_none()
        {
            return self.emit_hole(
                block,
                int.syntax().text_range(),
                UnresolvedType::new(self.context).into(),
            );
        }
        let constant = self
            .read_constant(literal)
            .expect("an integer literal was read as an `int64` above");

        let ty = constant.ty(self.context);
        let operation = match constant {
            Constant::Int(value) => yz::constant_int(self.context, ty, value, loc).into(),
            Constant::Float(value) => yz::constant_float(self.context, ty, value, loc).into(),
            Constant::Bool(value) => yz::constant_bool(self.context, ty, value, loc).into(),
            Constant::Str(value) => yz::constant_str(self.context, ty, value, loc).into(),
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
            return self.hole_and_assert(
                block,
                ident,
                "identifier expression is missing its name",
                UnresolvedType::new(self.context).into(),
            );
        };

        let loc = self.location(ident);
        let used = ident.syntax().text_range();
        self.convert_reference(
            block,
            locals,
            ident,
            used,
            Reference::unqualified(name),
            loc,
        )
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
                return self.hole_and_report(
                    block,
                    access,
                    "field access on an expression is not supported yet",
                    UnresolvedType::new(self.context).into(),
                );
            }
            None => None,
        };

        let Some(base) = base else {
            return self.hole_and_assert(
                block,
                access,
                "field access is missing its base",
                UnresolvedType::new(self.context).into(),
            );
        };

        let Some((field, used)) = self.read_ident_with_range(access.field()) else {
            return self.hole_and_assert(
                block,
                access,
                "field access is missing its field",
                UnresolvedType::new(self.context).into(),
            );
        };

        let reference = Reference {
            qualifier: Some(base),
            name: field,
        };
        self.convert_reference(block, locals, access, used, reference, loc)
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
                return self.hole_and_assert(
                    block,
                    binary,
                    "binary expression is missing its left operand",
                    UnresolvedType::new(self.context).into(),
                );
            }
        };

        let rhs = match binary.rhs() {
            Some(expr) => self.convert_expr(block, locals, &expr),
            None => {
                return self.hole_and_assert(
                    block,
                    binary,
                    "binary expression is missing its right operand",
                    UnresolvedType::new(self.context).into(),
                );
            }
        };

        let var = UnresolvedType::new(self.context).into();
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
            Some(BinOp::In) => yz::r#in(self.context, var, lhs, rhs, loc).into(),
            Some(BinOp::NotIn) => {
                let contains = block
                    .append_operation(yz::r#in(self.context, var, lhs, rhs, loc).into())
                    .first_result();
                yz::not(self.context, var, contains, loc).into()
            }
            None => {
                return self.hole_and_assert(
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

        // `int64`'s least value is written as the negation of a literal one
        // past its greatest, which alone is out of range.
        if unary.op() == Some(UnaryOp::Neg)
            && let Some(ast::Expr::Literal(ast::Literal::IntLiteral(int))) = unary.expr()
            && int.value() == Some(i64::MIN.unsigned_abs())
        {
            let least = yz::constant_int(
                self.context,
                Int64Type::new(self.context).into(),
                IntegerAttribute::from_i64(self.context, i64::MIN),
                loc,
            );
            return block.append_operation(least.into()).first_result();
        }

        let value = match unary.expr() {
            Some(expr) => self.convert_expr(block, locals, &expr),
            None => {
                return self.hole_and_assert(
                    block,
                    unary,
                    "unary expression is missing its operand",
                    UnresolvedType::new(self.context).into(),
                );
            }
        };

        let result = match unary.op() {
            Some(UnaryOp::Neg) => yz::neg(
                self.context,
                UnresolvedType::new(self.context).into(),
                value,
                loc,
            )
            .into(),
            Some(UnaryOp::Not) => yz::not(
                self.context,
                UnresolvedType::new(self.context).into(),
                value,
                loc,
            )
            .into(),
            Some(UnaryOp::Pos) => return value,
            None => {
                return self.hole_and_assert(
                    block,
                    unary,
                    "unary expression is missing its operator",
                    UnresolvedType::new(self.context).into(),
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

        let (callee, callee_range) = match call.callee() {
            Some(ast::Expr::IdentExpr(ident)) => match self.read_ident(ident.name()) {
                Some(callee) => (callee, ident.syntax().text_range()),
                None => {
                    return self.hole_and_assert(
                        block,
                        call,
                        "call is missing its callee",
                        UnresolvedType::new(self.context).into(),
                    );
                }
            },
            Some(ast::Expr::FieldAccessExpr(access)) => {
                return self.convert_module_call(block, locals, call, &access, loc);
            }
            Some(_) => {
                return self.hole_and_report(
                    block,
                    call,
                    "calling an expression is not supported yet",
                    UnresolvedType::new(self.context).into(),
                );
            }
            None => {
                return self.hole_and_assert(
                    block,
                    call,
                    "call is missing its callee",
                    UnresolvedType::new(self.context).into(),
                );
            }
        };

        let operands: Vec<Value> = call
            .args()
            .into_iter()
            .flat_map(|args| args.args())
            .map(|arg| self.convert_expr(block, locals, &arg))
            .collect();

        let given = operands.len();
        let Some(callable) = self.symbols.callable(callee, given) else {
            // A call that takes the wrong number of arguments still names the
            // function, as it does while the arguments are being typed.
            if let Some(target) = self.symbols.target_of(callee) {
                self.record(callee_range, callee, target);
            }
            let message = if let Some(arities) = self.symbols.arities(callee) {
                arity_mismatch(callee, &arities, given)
            } else {
                match self.symbols.kind(callee) {
                    Some(BindingKind::Pending) => {
                        format!("`{callee}` is bound further down the file")
                    }
                    Some(kind) => format!("`{callee}` is a {kind}, not a function"),
                    None if self.symbols.is_method(callee) => {
                        format!(
                            "`{callee}` is a trait method, and calling one is not supported yet"
                        )
                    }
                    None => format!("unresolved identifier `{callee}`"),
                }
            };
            return self.hole_and_report(
                block,
                call,
                &message,
                UnresolvedType::new(self.context).into(),
            );
        };

        self.record(callee_range, callee, callable.target);
        self.emit_call(block, callable, &operands, loc)
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
            Some(ast::Expr::IdentExpr(ident)) => self
                .read_ident(ident.name())
                .map(|base| (base, ident.syntax().text_range())),
            Some(_) => {
                return self.hole_and_report(
                    block,
                    call,
                    "calling an expression is not supported yet",
                    UnresolvedType::new(self.context).into(),
                );
            }
            None => None,
        };

        let (Some((base, base_range)), Some((name, used))) =
            (base, self.read_ident_with_range(access.field()))
        else {
            return self.hole_and_assert(
                block,
                call,
                "module call is missing its module or its function",
                UnresolvedType::new(self.context).into(),
            );
        };

        let Some(path) = self.symbols.module_of(base) else {
            return self.hole_and_report(
                block,
                call,
                &format!("`{base}` is not a module"),
                UnresolvedType::new(self.context).into(),
            );
        };
        self.record_module(base_range, base, path);

        let operands: Vec<Value> = call
            .args()
            .into_iter()
            .flat_map(|args| args.args())
            .map(|arg| self.convert_expr(block, locals, &arg))
            .collect();

        let Some((at, kind)) = self.resolve_export(call, path, name) else {
            return self.emit_hole(
                block,
                call.syntax().text_range(),
                UnresolvedType::new(self.context).into(),
            );
        };

        let given = operands.len();
        let Some(callable) = self.symbols.callable_in(at, given) else {
            if let Some(target) = self.symbols.target_in(at) {
                self.record_through(used, name, target, None);
            }
            let message = match self.symbols.arities_in(at) {
                Some(arities) => arity_mismatch(name, &arities, given),
                None => format!("`{name}` is a {kind}, not a function"),
            };
            return self.hole_and_report(
                block,
                call,
                &message,
                UnresolvedType::new(self.context).into(),
            );
        };

        self.record_through(used, name, callable.target, None);
        self.emit_call(block, callable, &operands, loc)
    }

    fn convert_list<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        locals: &Locals<'c, 'a>,
        list: &ast::ListExpr,
    ) -> Value<'c, 'a> {
        let loc = self.location(list);
        if let Some((element, values)) = self.read_constant_list(list) {
            return self.emit_constant_list(block, element, &values, loc);
        }

        let values: Vec<Value> = list
            .elements()
            .map(|element| self.convert_expr(block, locals, &element))
            .collect();
        block
            .append_operation(
                yzl::list(
                    self.context,
                    UnresolvedType::new(self.context).into(),
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
            None => self.hole_and_assert(
                block,
                paren,
                "parenthesized expression is missing its inner expression",
                UnresolvedType::new(self.context).into(),
            ),
        }
    }

    /// A module-level `let` becomes a call for expansion to inline, since a
    /// stage region is isolated.
    fn convert_reference<'a>(
        &mut self,
        block: BlockRef<'c, 'a>,
        locals: &Locals<'c, 'a>,
        node: &impl AstNode,
        used: TextRange,
        reference: Reference<'_>,
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        let name = reference.name;
        let message = match self.symbols.lookup(reference) {
            Lookup::Column { index, declared } => {
                if let Some(declared) = declared {
                    self.record_declared(used, name, declared);
                }
                return block
                    .argument(index)
                    .expect("the scope answered from the row this block was built for")
                    .into();
            }
            Lookup::Local { slot, declared } => {
                self.record_local(used, name, declared);
                let load = yzl::load(
                    self.context,
                    UnresolvedType::new(self.context).into(),
                    locals[slot],
                    loc,
                );
                return block.append_operation(load.into()).first_result();
            }
            Lookup::Let { symbol, target } => {
                self.record(used, name, target);
                return self.emit_call(block, Callable::constant(symbol, target), &[], loc);
            }
            Lookup::Lost => {
                return self.emit_hole(
                    block,
                    node.syntax().text_range(),
                    UnresolvedType::new(self.context).into(),
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
            Lookup::NotAValue(what) => {
                // The name still names a declaration. Record it for the editor.
                if let Some(path) = self.symbols.module_of(name) {
                    self.record_module(used, name, path);
                } else if let Some(target) = self.symbols.target_of(name) {
                    self.record(used, name, target);
                }
                format!("`{reference}` is a {what}, not a value")
            }
            Lookup::NotYet => format!("`{reference}` is bound further down the file"),
            Lookup::Unknown => format!("unresolved identifier `{reference}`"),
        };

        self.report_unresolved(node, &message);
        self.emit_hole(
            block,
            node.syntax().text_range(),
            UnresolvedType::new(self.context).into(),
        )
    }

    /// The values of a list whose elements are all literals of one kind,
    /// with the kind's type. Any other list is lowered element by element,
    /// so inference reports a mix of kinds as it reports any other.
    fn read_constant_list(&self, list: &ast::ListExpr) -> Option<(Type<'c>, Vec<Attribute<'c>>)> {
        let mut element: Option<Type<'c>> = None;
        let mut values = Vec::new();
        for expr in list.elements() {
            let ast::Expr::Literal(literal) = expr else {
                return None;
            };

            let constant = self.read_constant(&literal)?;
            let ty = constant.ty(self.context);
            if element.is_some_and(|element| element != ty) {
                return None;
            }

            element = Some(ty);
            values.push(constant.into());
        }

        Some((element?, values))
    }

    /// The value a literal writes. `None` for an integer that `int64` cannot
    /// hold.
    fn read_constant(&self, literal: &ast::Literal) -> Option<Constant<'c>> {
        Some(match literal {
            ast::Literal::IntLiteral(int) => {
                let value = i64::try_from(int.value()?).ok()?;
                Constant::Int(IntegerAttribute::from_i64(self.context, value))
            }
            ast::Literal::FloatLiteral(float) => Constant::Float(FloatAttribute::new(
                self.context,
                Type::float64(self.context),
                float.value().unwrap_or_default(),
            )),
            ast::Literal::BoolLiteral(boolean) => Constant::Bool(BoolAttribute::new(
                self.context,
                boolean.value().unwrap_or_default(),
            )),
            ast::Literal::StringLiteral(string) => Constant::Str(StringAttribute::new(
                self.context,
                &string.to_value().unwrap_or_default(),
            )),
        })
    }

    fn emit_constant_list<'a>(
        &self,
        block: BlockRef<'c, 'a>,
        element: Type<'c>,
        values: &[Attribute<'c>],
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        block
            .append_operation(
                yz::constant_list(
                    self.context,
                    ListType::new(self.context, element).into(),
                    ArrayAttribute::new(self.context, values),
                    loc,
                )
                .into(),
            )
            .first_result()
    }

    fn emit_call<'a>(
        &self,
        block: BlockRef<'c, 'a>,
        callable: Callable<'c>,
        operands: &[Value<'c, 'a>],
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        let mut builder = yzl::CallOperationBuilder::new(self.context, loc)
            .result(UnresolvedType::new(self.context).into())
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
}

/// The message for a call to a function none of whose overloads takes
/// `given` arguments.
fn arity_mismatch(name: &str, arities: &[usize], given: usize) -> String {
    let expected = match arities {
        [] => unreachable!("`{name}` has an overload wherever its name is visible"),
        [only] => only.to_string(),
        [rest @ .., last] => {
            let rest: Vec<String> = rest.iter().map(ToString::to_string).collect();
            format!("{} or {last}", rest.join(", "))
        }
    };

    format!("`{name}` expects {expected} argument(s), found {given}")
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::test_support::{lower, lowered, rendered, reported};

    #[test]
    fn a_call_no_overload_takes_is_reported() {
        expect![[r"
            error: `f` expects 1 or 2 argument(s), found 0
             --> test.yz:7:18
              |
            7 | from t |> select f() as v
              |                  ^^^
        "]].assert_eq(&reported(
            "def f(x: int64) -> int64 { return x }\ndef f(x: int64, y: int64) -> int64 { return x }\n\nstruct Row { a: int64 }\ntable t = Row\n\nfrom t |> select f() as v\n",
        ));
    }

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
    fn an_integer_literal_past_int64_is_reported() {
        expect![[r"
            error: integer literal is out of range for `int64`
             --> test.yz:6:14
              |
            6 | |> where a > 9223372036854775808
              |              ^^^^^^^^^^^^^^^^^^^
        "]]
        .assert_eq(&reported(
            r"
struct Row { a: int64 }
table t = Row

from t
|> where a > 9223372036854775808
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
              |             ^
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
