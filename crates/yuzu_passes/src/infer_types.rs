//! Unification over the values of a verified module. Every `!yzl.var` value
//! is a type variable, and the answers are written onto the values, so
//! `!yzl.var` is gone by the end.

use std::collections::{HashMap, HashSet};
use std::mem;

use melior::Context;
use melior::ir::attribute::{ArrayAttribute, TypeAttribute};
use melior::ir::operation::{OperationLike, OperationMutLike, OperationRef, OperationRefMut};
use melior::ir::r#type::FunctionType;
use melior::ir::{Attribute, BlockRef, Location, Module, RegionLike, Type, Value, ValueLike};
use yuzu_mlir::attributes::CalleeSource;
use yuzu_mlir::diagnostics::emit_error;
use yuzu_mlir::ext::{
    ArrayAttributeExt, BlockExt, OperationCast, OperationExt, RegionExt, ValueExt, ValueId,
};
use yuzu_mlir::ops::yz::YzOp;
use yuzu_mlir::ops::yzl::{FnOp, YzlOp};
use yuzu_mlir::types::{self, BoolType, Float64Type, Int64Type, VarType};
use yuzu_mlir::{ListType, ParamType};
use yuzu_types::{AggFunc, BuiltinFunc, Func, FunctionRegistry};

pub fn infer_types<'c>(
    context: &'c Context,
    module: &mut Module<'c>,
    registry: &dyn FunctionRegistry,
) {
    let declared = Declarations::of(module.body());
    let mut inferrer = TypeInferrer {
        context,
        declared: &declared,
        registry,
        filled: Vec::new(),
        vars: HashMap::new(),
        rows: HashMap::new(),
        bindings: HashMap::new(),
        relations: HashMap::new(),
        pending: Vec::new(),
        instances: HashMap::new(),
    };

    inferrer.infer_block(module.body());
    inferrer.check_pending_bounds();
    inferrer.stamp_block(module.body());
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct TypeVar(usize);

/// `List` holds the variable of its element type.
#[derive(Clone, Copy)]
enum Term<'c> {
    Concrete(Type<'c>),
    Var(TypeVar),
    List(TypeVar),
}

type Row<'c> = Vec<Term<'c>>;

struct Signature<'c> {
    params: Vec<Type<'c>>,
    result: Type<'c>,
    type_params: Vec<&'c str>,
    bounds: Vec<(&'c str, &'c str)>,
}

#[derive(Default)]
struct Declarations<'c> {
    signatures: HashMap<&'c str, Signature<'c>>,
    /// The `(trait, type)` pairs the implementations supply.
    impls: HashSet<(&'c str, &'c str)>,
    rows: HashMap<&'c str, Row<'c>>,
}

impl<'c> Declarations<'c> {
    fn of(block: BlockRef<'c, '_>) -> Self {
        let mut declared = Self::default();
        for op in block.operations() {
            match op.as_yzl() {
                Some(YzlOp::Fn(function)) => {
                    if let Some(signature) = parse_signature(&function) {
                        declared
                            .signatures
                            .insert(function.sym_name().value(), signature);
                    }
                }
                Some(YzlOp::Impl(item)) => {
                    declared
                        .impls
                        .insert((item.r#trait().value(), item.target().value()));
                }
                Some(YzlOp::Struct(item)) => {
                    let row = item
                        .types()
                        .types()
                        .into_iter()
                        .map(Term::Concrete)
                        .collect();
                    declared.rows.insert(item.sym_name().value(), row);
                }
                Some(YzlOp::Table(table)) => {
                    let row = declared
                        .rows
                        .get(table.row().value())
                        .cloned()
                        .unwrap_or_default();
                    declared.rows.insert(table.sym_name().value(), row);
                }
                _ => {}
            }
        }

        declared
    }
}

/// A bound checked once the variable standing for its parameter resolves.
#[derive(Clone, Copy)]
struct PendingBound<'c> {
    var: TypeVar,
    trait_: &'c str,
    callee: &'c str,
    location: Location<'c>,
}

struct TypeInferrer<'c, 'd> {
    context: &'c Context,
    declared: &'d Declarations<'c>,
    registry: &'d dyn FunctionRegistry,
    /// One slot per type variable: unbound, substituted by another variable,
    /// or filled with its concrete type.
    filled: Vec<Option<Term<'c>>>,
    vars: HashMap<ValueId, TypeVar>,
    rows: HashMap<ValueId, Row<'c>>,
    /// What each `let` yields, by symbol: the types a call to it gives.
    bindings: HashMap<&'c str, Row<'c>>,
    /// The row of each `let` that yields a query, by symbol: what `from`
    /// reads. A query `let` yields one value, the query, so its row is not
    /// its yielded types.
    relations: HashMap<&'c str, Row<'c>>,
    pending: Vec<PendingBound<'c>>,
    /// The type variables each generic call minted, in declaration order.
    instances: HashMap<ValueId, Vec<TypeVar>>,
}

impl<'c, 'd> TypeInferrer<'c, 'd> {
    fn term_of(&mut self, value: Value<'c, '_>) -> Term<'c> {
        let ty = value.r#type();
        if ty != VarType::get(self.context) {
            return Term::Concrete(ty);
        }

        let key = value.id();
        if let Some(&var) = self.vars.get(&key) {
            return Term::Var(var);
        }

        let var = self.fresh();
        self.vars.insert(key, var);
        Term::Var(var)
    }

    fn fresh(&mut self) -> TypeVar {
        self.filled.push(None);
        TypeVar(self.filled.len() - 1)
    }

    fn find(&mut self, var: TypeVar) -> TypeVar {
        match self.filled[var.0] {
            Some(Term::Var(next)) => {
                let root = self.find(next);
                self.filled[var.0] = Some(Term::Var(root));
                root
            }
            _ => var,
        }
    }

    fn shallow(&mut self, term: Term<'c>) -> Term<'c> {
        match term {
            Term::Var(var) => {
                let root = self.find(var);
                self.filled[root.0].unwrap_or(Term::Var(root))
            }
            term => term,
        }
    }

    fn unify(&mut self, op: OperationRef<'c, '_>, a: Term<'c>, b: Term<'c>) {
        let a = self.shallow(a);
        let b = self.shallow(b);
        match (a, b) {
            (Term::Var(a), Term::Var(b)) if a == b => {}
            (Term::Var(var), term) | (term, Term::Var(var)) => self.filled[var.0] = Some(term),
            (Term::List(_), Term::List(_) | Term::Concrete(_))
            | (Term::Concrete(_), Term::List(_)) => match (element(a), element(b)) {
                (Some(a), Some(b)) => self.unify(op, a, b),
                _ => self.mismatch(op, a, b),
            },
            (Term::Concrete(x), Term::Concrete(y)) => {
                if x != y {
                    self.mismatch(op, a, b);
                }
            }
        }
    }

    fn mismatch(&mut self, op: OperationRef<'c, '_>, expected: Term<'c>, found: Term<'c>) {
        let (expected, found) = (self.display(expected), self.display(found));
        self.report(op, &format!("expected `{expected}`, found `{found}`"));
    }

    /// Names the stage in the complaint, rather than reporting a bare
    /// mismatch.
    fn expect_yield(&mut self, op: OperationRef<'c, '_>, expected: Type<'c>, what: &str) {
        let Some(value) = last_region_op(op).and_then(|end| end.try_first_operand()) else {
            return;
        };

        let term = self.term_of(value);
        match self.resolve(term) {
            Some(found) if found != expected => {
                let (expected, found) = (
                    self.display(Term::Concrete(expected)),
                    self.display(Term::Concrete(found)),
                );
                self.report(
                    op,
                    &format!("expected the {what} to be `{expected}`, found `{found}`"),
                );
            }
            None => self.unify(op, term, Term::Concrete(expected)),
            Some(_) => {}
        }
    }

    fn resolve(&mut self, term: Term<'c>) -> Option<Type<'c>> {
        match self.shallow(term) {
            Term::Concrete(ty) => Some(ty),
            Term::Var(_) => None,
            Term::List(inner) => {
                let inner = self.resolve(Term::Var(inner))?;
                Some(ListType::new(self.context, inner).into())
            }
        }
    }

    fn display(&mut self, term: Term<'c>) -> String {
        match self.shallow(term) {
            Term::Concrete(ty) => types::name(self.context, ty),
            Term::Var(_) => "_".to_string(),
            Term::List(inner) => format!("List[{}]", self.display(Term::Var(inner))),
        }
    }

    fn infer_block(&mut self, block: BlockRef<'c, '_>) {
        for op in block.operations() {
            self.infer_op(op);
        }
    }

    fn infer_op(&mut self, op: OperationRef<'c, '_>) {
        let declared = self.declared;
        match op.as_yzl() {
            Some(YzlOp::Call(call)) => {
                let callee = call.callee().value();
                match call.callee_source() {
                    Some(CalleeSource::Builtin) => self.resolve_builtin_ty(op, callee),
                    Some(CalleeSource::Let) => {
                        let yielded = self
                            .bindings
                            .get(callee)
                            .and_then(|row| row.first().copied());
                        if let Some(yielded) = yielded {
                            let term = self.term_of(op.first_result());
                            self.unify(op, term, yielded);
                        }
                    }
                    Some(CalleeSource::Fn | CalleeSource::External) | None => {
                        let Some(signature) = declared.signatures.get(callee) else {
                            return;
                        };

                        let bindings = self.instantiate(op, callee, signature);
                        for (argument, parameter) in op.operands().zip(&signature.params) {
                            let term = self.term_of(argument);
                            let expected = self.substitute(*parameter, &bindings);
                            self.unify(op, term, expected);
                        }

                        let term = self.term_of(op.first_result());
                        let expected = self.substitute(signature.result, &bindings);
                        self.unify(op, term, expected);
                    }
                }
            }
            Some(YzlOp::Fn(function)) => {
                let Some(signature) = declared.signatures.get(function.sym_name().value()) else {
                    return;
                };

                self.infer_regions(op, &Row::new(), &signature.params);
                self.unify_returns(op, signature.result);
            }
            Some(YzlOp::From(from)) => {
                let source = from.source().value();
                let row = declared
                    .rows
                    .get(source)
                    .or_else(|| self.relations.get(source))
                    .cloned()
                    .unwrap_or_default();
                self.record_row(op, row);
            }
            Some(YzlOp::Let(binding)) => {
                self.infer_regions(op, &Row::new(), &[]);
                let name = binding.sym_name().value();
                let query_row = last_region_op(op)
                    .and_then(|terminator| terminator.try_first_operand())
                    .and_then(|query| self.rows.get(&query.id()).cloned());
                if let Some(query_row) = query_row {
                    self.relations.insert(name, query_row);
                }

                let row = self.yield_terms(op);
                if let Some(annotation) = binding.annotation()
                    && let Some(&yielded) = row.first()
                {
                    self.unify(op, Term::Concrete(annotation.value()), yielded);
                }

                self.bindings.insert(name, row);
            }
            Some(YzlOp::List(_)) => {
                let inner = self.fresh();
                for element in op.operands() {
                    let term = self.term_of(element);
                    self.unify(op, term, Term::Var(inner));
                }

                let out = self.term_of(op.first_result());
                self.unify(op, out, Term::List(inner));
            }
            Some(
                stage @ (YzlOp::Where(_)
                | YzlOp::Distinct(_)
                | YzlOp::Limit(_)
                | YzlOp::Alias(_)
                | YzlOp::Rename(_)
                | YzlOp::Drop(_)),
            ) => {
                let row = self.input_row(op);
                self.infer_regions(op, &row, &[]);
                if matches!(stage, YzlOp::Where(_)) {
                    let boolean = BoolType::get(self.context);
                    self.expect_yield(op, boolean, "`where` predicate");
                }

                self.record_row(op, row);
            }
            Some(YzlOp::Set(stage)) => {
                let row = self.input_row(op);
                self.infer_regions(op, &row, &[]);
                let yields = self.yield_terms(op);
                let columns = stage.set_cols().map(|columns| columns.indices());
                for (index, term) in columns.unwrap_or_default().into_iter().zip(yields) {
                    if let Some(&column) = row.get(index) {
                        self.unify(op, column, term);
                    }
                }

                self.record_row(op, row);
            }
            Some(YzlOp::Select(_)) => {
                let input = self.input_row(op);
                self.infer_regions(op, &input, &[]);
                let row = self.yield_terms(op);
                self.record_row(op, row);
            }
            Some(YzlOp::Extend(_)) => {
                let mut row = self.input_row(op);
                self.infer_regions(op, &row, &[]);
                row.extend(self.yield_terms(op));
                self.record_row(op, row);
            }
            Some(YzlOp::Aggregate(stage)) => {
                let input = self.input_row(op);
                self.infer_regions(op, &input, &[]);
                let keys = stage.key_cols().map(|keys| keys.indices());
                let mut row: Row = keys
                    .unwrap_or_default()
                    .into_iter()
                    .filter_map(|index| input.get(index).copied())
                    .collect();
                row.extend(self.yield_terms(op));
                self.record_row(op, row);
            }
            Some(YzlOp::Join(_)) => {
                let row = self.input_row(op);
                self.infer_regions(op, &row, &[]);
                let boolean = BoolType::get(self.context);
                self.expect_yield(op, boolean, "`on` condition");
                self.record_row(op, row);
            }
            Some(_) => self.infer_regions(op, &Row::new(), &[]),
            None => self.infer_yz_op(op),
        }
    }

    fn infer_yz_op(&mut self, op: OperationRef<'c, '_>) {
        let boolean = Term::Concrete(BoolType::get(self.context));
        match op.as_yz() {
            Some(YzOp::Add(_) | YzOp::Sub(_) | YzOp::Mul(_) | YzOp::Div(_) | YzOp::Rem(_)) => {
                let (lhs, rhs) = (self.operand_term(op, 0), self.operand_term(op, 1));
                let out = self.term_of(op.first_result());
                self.unify(op, lhs, rhs);
                self.unify(op, lhs, out);
            }
            Some(YzOp::Neg(_)) => {
                let value = self.operand_term(op, 0);
                let out = self.term_of(op.first_result());
                self.unify(op, value, out);
            }
            Some(YzOp::Cmp(_)) => {
                let (lhs, rhs) = (self.operand_term(op, 0), self.operand_term(op, 1));
                let out = self.term_of(op.first_result());
                self.unify(op, lhs, rhs);
                self.unify(op, out, boolean);
            }
            Some(YzOp::And(_) | YzOp::Or(_)) => {
                let (lhs, rhs) = (self.operand_term(op, 0), self.operand_term(op, 1));
                let out = self.term_of(op.first_result());
                self.unify(op, lhs, boolean);
                self.unify(op, rhs, boolean);
                self.unify(op, out, boolean);
            }
            Some(YzOp::Not(_)) => {
                let value = self.operand_term(op, 0);
                let out = self.term_of(op.first_result());
                self.unify(op, value, boolean);
                self.unify(op, out, boolean);
            }
            _ => self.infer_regions(op, &Row::new(), &[]),
        }
    }

    /// One fresh variable per type parameter, so every call solves its own
    /// instance.
    fn instantiate(
        &mut self,
        op: OperationRef<'c, '_>,
        callee: &'c str,
        signature: &Signature<'c>,
    ) -> HashMap<&'c str, TypeVar> {
        let mut bindings = HashMap::new();
        let mut ordered = Vec::new();
        for name in &signature.type_params {
            let var = self.fresh();
            bindings.insert(*name, var);
            ordered.push(var);
        }

        if !ordered.is_empty()
            && let Some(result) = op.try_first_result()
        {
            self.instances.insert(result.id(), ordered);
        }

        for (subject, trait_) in &signature.bounds {
            if let Some(&var) = bindings.get(subject) {
                self.pending.push(PendingBound {
                    var,
                    trait_,
                    callee,
                    location: op.location(),
                });
            }
        }

        bindings
    }

    fn substitute(&self, ty: Type<'c>, bindings: &HashMap<&'c str, TypeVar>) -> Term<'c> {
        if let Some(param) = ParamType::from_type(ty)
            && let Some(&var) = bindings.get(param.name())
        {
            return Term::Var(var);
        }

        Term::Concrete(ty)
    }

    fn check_pending_bounds(&mut self) {
        for bound in mem::take(&mut self.pending) {
            let Some(resolved) = self.resolve(Term::Var(bound.var)) else {
                continue;
            };

            let Some(name) = types::scalar_name(self.context, resolved) else {
                continue;
            };

            if !self.declared.impls.contains(&(bound.trait_, name)) {
                emit_error(
                    bound.location,
                    &format!(
                        "`{name}` does not implement `{}`, required by `{}`",
                        bound.trait_, bound.callee
                    ),
                );
            }
        }
    }

    fn operand_term(&mut self, op: OperationRef<'c, '_>, index: usize) -> Term<'c> {
        match op.operand(index) {
            Ok(value) => self.term_of(value),
            Err(_) => Term::Concrete(VarType::get(self.context)),
        }
    }

    /// The builtins are polymorphic, so each carries its own typing rule.
    fn resolve_builtin_ty(&mut self, op: OperationRef<'c, '_>, callee: &str) {
        let Some(entry) = self
            .registry
            .entries()
            .iter()
            .find(|entry| entry.name == callee)
        else {
            return;
        };

        let int64 = Term::Concrete(Int64Type::get(self.context));
        let boolean = Term::Concrete(BoolType::get(self.context));
        let out = self.term_of(op.first_result());
        match entry.func {
            BuiltinFunc::Aggregate(AggFunc::Count | AggFunc::CountDistinct) => {
                self.unify(op, out, int64);
            }
            BuiltinFunc::Aggregate(AggFunc::Sum) => {
                if let Some(argument) = op.try_first_operand() {
                    let term = self.term_of(argument);
                    if let Some(ty) = self.resolve(term) {
                        let result = if ty == Float64Type::get(self.context) {
                            Term::Concrete(ty)
                        } else {
                            int64
                        };

                        self.unify(op, out, result);
                    }
                }
            }
            BuiltinFunc::Aggregate(AggFunc::Min | AggFunc::Max | AggFunc::Avg) => {
                if let Some(argument) = op.try_first_operand() {
                    let term = self.term_of(argument);
                    self.unify(op, out, term);
                }
            }
            BuiltinFunc::Scalar(Func::In) => {
                let (lhs, rhs) = (self.operand_term(op, 0), self.operand_term(op, 1));
                let inner = self.fresh();
                self.unify(op, Term::List(inner), rhs);
                self.unify(op, lhs, Term::Var(inner));
                self.unify(op, out, boolean);
            }
            BuiltinFunc::Scalar(Func::Power) => {
                let (lhs, rhs) = (self.operand_term(op, 0), self.operand_term(op, 1));
                self.unify(op, lhs, rhs);
                self.unify(op, lhs, out);
            }
            BuiltinFunc::Scalar(Func::ShiftLeft | Func::ShiftRight) => {
                let (lhs, rhs) = (self.operand_term(op, 0), self.operand_term(op, 1));
                self.unify(op, lhs, int64);
                self.unify(op, rhs, int64);
                self.unify(op, out, int64);
            }
            BuiltinFunc::Aggregate(AggFunc::External(_)) | BuiltinFunc::Scalar(_) => {}
        }
    }

    /// The block arguments are the row's columns or the function's
    /// parameters, by position.
    fn infer_regions(&mut self, op: OperationRef<'c, '_>, columns: &Row<'c>, params: &[Type<'c>]) {
        for region in op.regions() {
            for block in region.blocks() {
                for (index, argument) in block.arguments().enumerate() {
                    let term = self.term_of(argument.into());
                    if let Some(&column) = columns.get(index) {
                        self.unify(op, term, column);
                    } else if let Some(&param) = params.get(index) {
                        self.unify(op, term, Term::Concrete(param));
                    }
                }

                self.infer_block(block);
            }
        }
    }

    fn input_row(&mut self, op: OperationRef<'c, '_>) -> Row<'c> {
        op.try_first_operand()
            .and_then(|input| self.rows.get(&input.id()).cloned())
            .unwrap_or_default()
    }

    fn record_row(&mut self, op: OperationRef<'c, '_>, row: Row<'c>) {
        if let Some(result) = op.try_first_result() {
            self.rows.insert(result.id(), row);
        }
    }

    fn yield_terms(&mut self, op: OperationRef<'c, '_>) -> Row<'c> {
        let Some(terminator) = last_region_op(op) else {
            return Row::new();
        };

        terminator
            .operands()
            .map(|value| self.term_of(value))
            .collect()
    }

    fn unify_returns(&mut self, op: OperationRef<'c, '_>, ret: Type<'c>) {
        if let Some(terminator) = last_region_op(op)
            && matches!(terminator.as_yzl(), Some(YzlOp::Return(_)))
            && let Some(value) = terminator.try_first_operand()
        {
            let term = self.term_of(value);
            self.unify(terminator, term, Term::Concrete(ret));
        }
    }

    /// A column or parameter is only as open as the expressions reading it,
    /// so an unresolved one is left for them to report.
    fn stamp_block(&mut self, block: BlockRef<'c, '_>) {
        for argument in block.arguments() {
            let term = self.term_of(argument.into());
            if let Some(ty) = self.resolve(term) {
                argument.set_type(ty);
            }
        }

        for mut op in block.operations_mut() {
            for region in op.regions() {
                for inner in region.blocks() {
                    self.stamp_block(inner);
                }
            }

            let Some(result) = op.try_first_result() else {
                continue;
            };

            let term = self.term_of(result);
            match self.resolve(term) {
                Some(ty) => result.set_type(ty),
                // A hole was already reported by the parse.
                None if matches!(op.as_yzl(), Some(YzlOp::Missing(_))) => {}
                None => emit_error(
                    op.location(),
                    "the type of this expression could not be inferred",
                ),
            }

            self.stamp_type_args(&mut op, result.id());
        }
    }

    /// A partial answer is worse than none, so a call whose parameters did
    /// not all resolve is left unstamped.
    fn stamp_type_args(&mut self, op: &mut OperationRefMut<'c, '_>, result: ValueId) {
        let Some(vars) = self.instances.remove(&result) else {
            return;
        };

        let resolved: Vec<Attribute<'c>> = vars
            .iter()
            .filter_map(|&var| self.resolve(Term::Var(var)))
            .map(|ty| TypeAttribute::new(ty).into())
            .collect();
        if resolved.len() == vars.len() {
            op.set_attribute(
                "type_args",
                ArrayAttribute::new(self.context, &resolved).into(),
            );
        }
    }

    fn report(&self, op: OperationRef<'c, '_>, message: &str) {
        emit_error(op.location(), message);
    }
}

fn element(term: Term<'_>) -> Option<Term<'_>> {
    match term {
        Term::List(inner) => Some(Term::Var(inner)),
        Term::Concrete(ty) => ListType::from_type(ty).map(|list| Term::Concrete(list.inner())),
        Term::Var(_) => None,
    }
}

fn last_region_op<'c, 'a>(op: OperationRef<'c, 'a>) -> Option<OperationRef<'c, 'a>> {
    op.regions().next()?.first_block()?.last_operation()
}

fn parse_signature<'c>(function: &FnOp<'c, '_>) -> Option<Signature<'c>> {
    let signature = FunctionType::try_from(function.signature().value()).ok()?;
    let params = (0..signature.input_count())
        .filter_map(|index| signature.input(index).ok())
        .collect();
    let subjects = function
        .bound_params()
        .map(|names| names.strings())
        .unwrap_or_default();
    let traits = function
        .bound_traits()
        .map(|names| names.symbols())
        .unwrap_or_default();

    Some(Signature {
        params,
        result: signature.result(0).ok()?,
        type_params: function
            .type_params()
            .map(|names| names.strings())
            .unwrap_or_default(),
        bounds: subjects.into_iter().zip(traits).collect(),
    })
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};

    use crate::infer_types;
    use crate::test_support;

    fn check(source: &str, expected: Expect) {
        test_support::check(
            source,
            |context, module| {
                infer_types(context, module, &yuzu_types::Builtins);
                module.as_operation().to_string()
            },
            expected,
        );
    }

    /// A query `let` yields one value, the query, and `from` reads its row
    /// of columns, not that one value.
    #[test]
    fn a_query_let_gives_its_row_to_from() {
        check(
            "struct Row { a: int64 }\ntable t = Row\n\nlet cap = 42\nlet small = from t |> where a < 10\n\nfrom small |> select a + cap as v\n",
            expect![[r#"
                module {
                  yzl.struct @Row ["a"] : [!yz.int64] {sym_visibility = "private"}
                  yzl.table @t of @Row {sym_visibility = "private"}
                  yzl.let @cap {
                    %2 = yz.constant_int 42
                    yzl.yield %2 : !yz.int64
                  } {sym_visibility = "private"}
                  yzl.let @small {
                    %2 = yzl.from @t
                    %3 = yzl.where %2 {
                    ^bb0(%arg0: !yz.int64):
                      %4 = yz.constant_int 10
                      %5 = yz.cmp "lt", %arg0, %4 : !yz.int64, !yz.int64 -> !yz.bool
                      yzl.yield %5 : !yz.bool
                    }
                    yzl.yield %3 : !yzl.query
                  } {sym_visibility = "private"}
                  %0 = yzl.from @small
                  %1 = yzl.select %0 as ["v"] {
                  ^bb0(%arg0: !yz.int64):
                    %2 = yzl.call @cap() : () -> !yz.int64 {callee_source = "let"}
                    %3 = yz.add %arg0, %2 : !yz.int64, !yz.int64 -> !yz.int64
                    yzl.yield %3 : !yz.int64
                  }
                  yzl.output %1
                }
            "#]],
        );
    }

    #[test]
    fn infers_columns_calls_and_measures() {
        check(
            r#"
struct Row { a: int64, b: int64, rating: float64 }
table t = Row

def f(x: int64) -> int64 { return x * 3 }

from t
|> where a > 10
|> extend f(a) + b as e
|> aggregate sum(e) as s, avg(rating) as r group by b
    "#,
            expect![[r#"
                module {
                  yzl.struct @Row ["a", "b", "rating"] : [!yz.int64, !yz.int64, !yz.float64] {sym_visibility = "private"}
                  yzl.table @t of @Row {sym_visibility = "private"}
                  yzl.fn @f params ["x"] (!yz.int64) -> !yz.int64 {
                  ^bb0(%arg0: !yz.int64):
                    %4 = yz.constant_int 3
                    %5 = yz.mul %arg0, %4 : !yz.int64, !yz.int64 -> !yz.int64
                    yzl.return %5 : !yz.int64
                  } {sym_visibility = "private"}
                  %0 = yzl.from @t
                  %1 = yzl.where %0 {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64, %arg2: !yz.float64):
                    %4 = yz.constant_int 10
                    %5 = yz.cmp "gt", %arg0, %4 : !yz.int64, !yz.int64 -> !yz.bool
                    yzl.yield %5 : !yz.bool
                  }
                  %2 = yzl.extend %1 as ["e"] {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64, %arg2: !yz.float64):
                    %4 = yzl.call @f(%arg0) : (!yz.int64) -> !yz.int64 {callee_source = "fn"}
                    %5 = yz.add %4, %arg1 : !yz.int64, !yz.int64 -> !yz.int64
                    yzl.yield %5 : !yz.int64
                  }
                  %3 = yzl.aggregate %2 group_by ["b"] as ["s", "r"] {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64, %arg2: !yz.float64, %arg3: !yz.int64):
                    %4 = yzl.call @sum(%arg3) : (!yz.int64) -> !yz.int64 {agg, callee_source = "builtin"}
                    %5 = yzl.call @avg(%arg2) : (!yz.float64) -> !yz.float64 {agg, callee_source = "builtin"}
                    yzl.yield %4, %5 : !yz.int64, !yz.float64
                  } {key_cols = [1]}
                  yzl.output %3
                }
            "#]],
        );
    }

    #[test]
    fn instantiates_a_generic_call_per_site() {
        check(
            r#"
trait Numeric {
    def zero(x: Self) -> Self
}

impl Numeric for int64 {
    def zero(x: int64) -> int64 { return 0 }
}

impl Numeric for float64 {
    def zero(x: float64) -> float64 { return 0.0 }
}

def id[T](x: T) -> T where T: Numeric { return x }

struct Row { a: int64, r: float64 }
table t = Row

from t
|> extend id(a) as m, id(r) as n
"#,
            expect![[r#"
                module {
                  yzl.trait @Numeric {
                    yzl.fn @zero generics ["Self"] params ["x"] (!yzl.param<"Self">) -> !yzl.param<"Self"> {
                    }
                  } {sym_visibility = "private"}
                  yzl.impl @Numeric for @int64 {
                    yzl.fn @zero generics ["Self"] params ["x"] (!yz.int64) -> !yz.int64 {
                    ^bb0(%arg0: !yzl.var):
                      %2 = yz.constant_int 0
                      yzl.return %2 : !yz.int64
                    }
                  }
                  yzl.impl @Numeric for @float64 {
                    yzl.fn @zero generics ["Self"] params ["x"] (!yz.float64) -> !yz.float64 {
                    ^bb0(%arg0: !yzl.var):
                      %2 = yz.constant_float 0.000000e+00
                      yzl.return %2 : !yz.float64
                    }
                  }
                  yzl.fn @id generics ["T"] where ["T"] : [@Numeric] params ["x"] (!yzl.param<"T">) -> !yzl.param<"T"> {
                  ^bb0(%arg0: !yzl.param<"T">):
                    yzl.return %arg0 : !yzl.param<"T">
                  } {sym_visibility = "private"}
                  yzl.struct @Row ["a", "r"] : [!yz.int64, !yz.float64] {sym_visibility = "private"}
                  yzl.table @t of @Row {sym_visibility = "private"}
                  %0 = yzl.from @t
                  %1 = yzl.extend %0 as ["m", "n"] {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.float64):
                    %2 = yzl.call @id(%arg0) : (!yz.int64) -> !yz.int64 {callee_source = "fn", type_args = [!yz.int64]}
                    %3 = yzl.call @id(%arg1) : (!yz.float64) -> !yz.float64 {callee_source = "fn", type_args = [!yz.float64]}
                    yzl.yield %2, %3 : !yz.int64, !yz.float64
                  }
                  yzl.output %1
                }
            "#]],
        );
    }

    #[test]
    fn an_external_call_takes_its_declared_type() {
        check(
            r#"
external def median(x: float64) -> float64

struct Row { rating: float64 }
table t = Row

from t
|> extend median(rating) as m
"#,
            expect![[r#"
                module {
                  yzl.fn @median params ["x"] (!yz.float64) -> !yz.float64 external {
                  } {sym_visibility = "private"}
                  yzl.struct @Row ["rating"] : [!yz.float64] {sym_visibility = "private"}
                  yzl.table @t of @Row {sym_visibility = "private"}
                  %0 = yzl.from @t
                  %1 = yzl.extend %0 as ["m"] {
                  ^bb0(%arg0: !yz.float64):
                    %2 = yzl.call @median(%arg0) : (!yz.float64) -> !yz.float64 {callee_source = "external"}
                    yzl.yield %2 : !yz.float64
                  }
                  yzl.output %1
                }
            "#]],
        );
    }

    #[test]
    fn reports_an_unsatisfied_bound() {
        check(
            r#"
trait Numeric {
    def zero(x: Self) -> Self
}

impl Numeric for int64 {
    def zero(x: int64) -> int64 { return 0 }
}

def id[T](x: T) -> T where T: Numeric { return x }

struct Row { name: str }
table t = Row

from t
|> extend id(name) as n
"#,
            expect![[r#"
                error: `str` does not implement `Numeric`, required by `id`
                 --> test.yz:16:11
                   |
                16 | |> extend id(name) as n
                   |           ^^^^^^^^
            "#]],
        );
    }

    #[test]
    fn reports_a_comparison_mismatch() {
        check(
            r#"
struct Row { name: str }
table t = Row

from t
|> where name == 1
    "#,
            expect![[r#"
                error: expected `str`, found `int64`
                 --> test.yz:6:10
                  |
                6 | |> where name == 1
                  |          ^^^^^^^^^
            "#]],
        );
    }

    #[test]
    fn reports_a_return_mismatch() {
        check(
            r#"
struct Row { a: int64 }
table t = Row

def f(x: int64) -> bool { return x }

from t
|> extend f(a) as e
    "#,
            expect![[r#"
                error: expected `int64`, found `bool`
                 --> test.yz:5:27
                  |
                5 | def f(x: int64) -> bool { return x }
                  |                           ^^^^^^^^
            "#]],
        );
    }

    #[test]
    fn reports_a_set_mismatch() {
        check(
            r#"
struct Row { level: int64 }
table t = Row

from t
|> set level = "high"
    "#,
            expect![[r#"
                error: expected `int64`, found `str`
                 --> test.yz:6:1
                  |
                6 | |> set level = "high"
                  | ^^^^^^^^^^^^^^^^^^^^^
            "#]],
        );
    }

    #[test]
    fn infers_a_list_and_membership() {
        check(
            r#"
struct Row { a: int64 }
table t = Row

from t
|> where a in [1, 3]
"#,
            expect![[r#"
                module {
                  yzl.struct @Row ["a"] : [!yz.int64] {sym_visibility = "private"}
                  yzl.table @t of @Row {sym_visibility = "private"}
                  %0 = yzl.from @t
                  %1 = yzl.where %0 {
                  ^bb0(%arg0: !yz.int64):
                    %2 = yz.constant_int 1
                    %3 = yz.constant_int 3
                    %4 = yzl.list[%2, %3] : (!yz.int64, !yz.int64) -> !yz.list<!yz.int64>
                    %5 = yzl.call @in(%arg0, %4) : (!yz.int64, !yz.list<!yz.int64>) -> !yz.bool {callee_source = "builtin"}
                    yzl.yield %5 : !yz.bool
                  }
                  yzl.output %1
                }
            "#]],
        );
    }

    #[test]
    fn an_annotated_let_holds_its_body_to_the_annotation() {
        check(
            r#"
struct Row { a: int64 }
table t = Row

let ids: List[str] = [1, 3]

from t
|> where a in ids
"#,
            expect![[r#"
                error: expected `str`, found `int64`
                 --> test.yz:5:1
                  |
                5 | let ids: List[str] = [1, 3]
                  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^
            "#]],
        );
    }

    #[test]
    fn a_stage_names_what_it_expected() {
        check(
            r#"
struct Row { level: int64, name: str }
table t = Row
struct Dept { level: int64 }
table depts = Dept

from t
|> where level
"#,
            expect![[r#"
                error: expected the `where` predicate to be `bool`, found `int64`
                 --> test.yz:8:1
                  |
                8 | |> where level
                  | ^^^^^^^^^^^^^^
            "#]],
        );
    }

    #[test]
    fn a_join_condition_has_to_be_a_predicate() {
        check(
            r#"
struct Row { level: int64 }
table t = Row
struct Dept { level: int64 }
table depts = Dept

from t as e
|> inner join depts as d on e.level
"#,
            expect![[r#"
                error: expected the `on` condition to be `bool`, found `int64`
                 --> test.yz:8:1
                  |
                8 | |> inner join depts as d on e.level
                  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            "#]],
        );
    }

    #[test]
    fn membership_needs_a_list() {
        check(
            r#"
struct Row { a: int64 }
table t = Row

from t
|> where a in 1
"#,
            expect![[r#"
                error: expected `List[_]`, found `int64`
                 --> test.yz:6:10
                  |
                6 | |> where a in 1
                  |          ^^^^^^
            "#]],
        );
    }

    #[test]
    fn reports_an_expression_nothing_pinned_down() {
        check(
            r#"
struct Row { a: int64 }
table t = Row

let xs = []

from t
|> where a > 1
"#,
            expect![[r#"
                error: the type of this expression could not be inferred
                 --> test.yz:5:10
                  |
                5 | let xs = []
                  |          ^^
            "#]],
        );
    }

    #[test]
    fn a_list_takes_the_type_of_its_elements() {
        check(
            r#"
struct Row { a: int64 }
table t = Row

let xs = [1, 2]

from t
|> where a > 1
"#,
            expect![[r#"
                module {
                  yzl.struct @Row ["a"] : [!yz.int64] {sym_visibility = "private"}
                  yzl.table @t of @Row {sym_visibility = "private"}
                  yzl.let @xs {
                    %2 = yz.constant_int 1
                    %3 = yz.constant_int 2
                    %4 = yzl.list[%2, %3] : (!yz.int64, !yz.int64) -> !yz.list<!yz.int64>
                    yzl.yield %4 : !yz.list<!yz.int64>
                  } {sym_visibility = "private"}
                  %0 = yzl.from @t
                  %1 = yzl.where %0 {
                  ^bb0(%arg0: !yz.int64):
                    %2 = yz.constant_int 1
                    %3 = yz.cmp "gt", %arg0, %2 : !yz.int64, !yz.int64 -> !yz.bool
                    yzl.yield %3 : !yz.bool
                  }
                  yzl.output %1
                }
            "#]],
        );
    }
}
