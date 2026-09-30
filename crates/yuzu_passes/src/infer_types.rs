//! Unification over the values of a verified module. Every `!yzl.unresolved` value
//! is a type variable, and the answers are written onto the values, so
//! `!yzl.unresolved` is gone by the end.

use std::mem;

use melior::Context;
use melior::ir::attribute::{ArrayAttribute, TypeAttribute};
use melior::ir::operation::{OperationLike, OperationMutLike, OperationRef, OperationRefMut};
use melior::ir::r#type::FunctionType;
use melior::ir::{Attribute, BlockRef, Location, Module, Type, Value, ValueLike};
use rustc_hash::{FxHashMap, FxHashSet};
use yuzu_mlir::attributes::CalleeSource;
use yuzu_mlir::diagnostics::emit_error;
use yuzu_mlir::ir::attribute::array::ArrayAttributeExt;
use yuzu_mlir::ir::block::BlockExt;
use yuzu_mlir::ir::operation::{OperationCast, OperationExt};
use yuzu_mlir::ir::region::RegionExt;
use yuzu_mlir::ir::value::{ValueExt, ValueId, op_result};
use yuzu_mlir::ops::yz::YzOp;
use yuzu_mlir::ops::yzl::{FnOp, YzlOp};
use yuzu_mlir::types::{
    self, BoolType, ErrorType, Int64Type, ListType, ParamType, RefType, UnitType, UnresolvedType,
};

pub fn infer_types<'c>(context: &'c Context, module: &mut Module<'c>) {
    let declared = Declarations::of(module.body());
    let mut inferrer = TypeInferrer {
        context,
        declared: &declared,
        filled: Vec::new(),
        vars: FxHashMap::default(),
        rows: FxHashMap::default(),
        bindings: FxHashMap::default(),
        relations: FxHashMap::default(),
        pending: Vec::new(),
        caller: None,
        instances: FxHashMap::default(),
        reported_unresolved: false,
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
    bounds: Vec<Bound<'c>>,
}

/// `subject: trait_` on a function's type parameter.
struct Bound<'c> {
    subject: &'c str,
    trait_: &'c str,
}

/// `impl trait_ for target`.
#[derive(PartialEq, Eq, Hash)]
struct Implementation<'c> {
    trait_: &'c str,
    target: &'c str,
}

#[derive(Default)]
struct Declarations<'c> {
    signatures: FxHashMap<&'c str, Signature<'c>>,
    impls: FxHashSet<Implementation<'c>>,
    rows: FxHashMap<&'c str, Row<'c>>,
}

impl<'c> Declarations<'c> {
    fn of(block: BlockRef<'c, '_>) -> Self {
        let mut declared = Self::default();
        for op in block.operations() {
            match op.as_yzl() {
                Some(YzlOp::Fn(function)) => {
                    if let Some(signature) = parse_signature(function) {
                        declared
                            .signatures
                            .insert(function.sym_name().value(), signature);
                    }
                }
                Some(YzlOp::Impl(item)) => {
                    declared.impls.insert(Implementation {
                        trait_: item.r#trait().value(),
                        target: item.target().value(),
                    });
                }
                Some(YzlOp::Struct(item)) => {
                    let row = item.types().types().map(Term::Concrete).collect();
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

/// The type parameters one generic call solves, by the call's result.
struct Instance<'c> {
    callee: &'c str,
    params: Vec<(&'c str, TypeVar)>,
}

/// A bound checked once the variable standing for its parameter resolves.
#[derive(Clone, Copy)]
struct PendingBound<'c> {
    var: TypeVar,
    trait_: &'c str,
    callee: &'c str,
    /// The function the call is in, whose own bounds its type parameters
    /// satisfy.
    caller: Option<&'c str>,
    location: Location<'c>,
}

struct TypeInferrer<'c, 'd> {
    context: &'c Context,
    declared: &'d Declarations<'c>,
    /// One slot per type variable: unbound, substituted by another variable,
    /// or filled with its concrete type.
    filled: Vec<Option<Term<'c>>>,
    vars: FxHashMap<ValueId, TypeVar>,
    rows: FxHashMap<ValueId, Row<'c>>,
    /// What each `let` yields, by symbol: the types a call to it gives.
    bindings: FxHashMap<&'c str, Row<'c>>,
    /// The row of each `let` that yields a query, by symbol: what `from`
    /// reads. A query `let` yields one value, the query, so its row is not
    /// its yielded types.
    relations: FxHashMap<&'c str, Row<'c>>,
    pending: Vec<PendingBound<'c>>,
    /// The function whose body the pass infers.
    caller: Option<&'c str>,
    /// The type variables each generic call minted, in declaration order.
    instances: FxHashMap<ValueId, Instance<'c>>,
    /// Whether stamping reported a value it could not type.
    reported_unresolved: bool,
}

impl<'c> TypeInferrer<'c, '_> {
    fn term_of(&mut self, value: Value<'c, '_>) -> Term<'c> {
        let ty = value.r#type();
        if UnresolvedType::from_type(ty).is_none() {
            return Term::Concrete(ty);
        }

        if is_hole(value) {
            return Term::Concrete(ErrorType::new(self.context).into());
        }

        Term::Var(self.var_for(value.id()))
    }

    /// The type of the value a place holds: its annotation, or a variable
    /// for the place.
    fn place_term(&mut self, place: Value<'c, '_>) -> Term<'c> {
        let element = RefType::from_type(place.r#type())
            .expect("a place has a `!yzl.ref` type")
            .element();
        if UnresolvedType::from_type(element).is_none() {
            return Term::Concrete(element);
        }

        Term::Var(self.var_for(place.id()))
    }

    /// The variable a value stands for, made on the first request.
    fn var_for(&mut self, key: ValueId) -> TypeVar {
        if let Some(&var) = self.vars.get(&key) {
            return var;
        }

        let var = self.fresh();
        self.vars.insert(key, var);
        var
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

    /// An error matches every type. A variable it meets takes it, so the
    /// values that read the variable are not reported again.
    fn unify(&mut self, op: OperationRef<'c, '_>, a: Term<'c>, b: Term<'c>) {
        let a = self.shallow(a);
        let b = self.shallow(b);
        if let Some(error) = [a, b].into_iter().find(|&term| is_error(term)) {
            for term in [a, b] {
                if let Term::Var(var) = term {
                    self.filled[var.0] = Some(error);
                }
            }
            return;
        }

        match (a, b) {
            (Term::Var(a), Term::Var(b)) if a == b => {}
            (Term::Var(var), term) | (term, Term::Var(var)) => {
                if self.occurs(var, term) {
                    report(op, "a list cannot contain itself");
                    self.filled[var.0] = Some(Term::Concrete(ErrorType::new(self.context).into()));
                } else {
                    self.filled[var.0] = Some(term);
                }
            }
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

    /// Whether a variable is inside a term, which binding it to that term
    /// would make a list of itself.
    fn occurs(&mut self, var: TypeVar, term: Term<'c>) -> bool {
        match self.shallow(term) {
            Term::Var(other) => self.find(other) == var,
            Term::List(inner) => self.occurs(var, Term::Var(inner)),
            Term::Concrete(_) => false,
        }
    }

    fn mismatch(&mut self, op: OperationRef<'c, '_>, expected: Term<'c>, found: Term<'c>) {
        let (expected, found) = (self.display(expected), self.display(found));
        report(op, &format!("expected `{expected}`, found `{found}`"));
    }

    /// Names the stage in the complaint, rather than reporting a bare
    /// mismatch.
    fn expect_yield(&mut self, op: OperationRef<'c, '_>, expected: Type<'c>, what: &str) {
        let Some(value) = op.body_terminator().and_then(|end| end.try_first_operand()) else {
            return;
        };

        let term = self.term_of(value);
        match self.resolve(term) {
            Some(found) if found != expected && ErrorType::from_type(found).is_none() => {
                let (expected, found) = (
                    self.display(Term::Concrete(expected)),
                    self.display(Term::Concrete(found)),
                );
                report(
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
            Term::Concrete(ty) => types::name(ty),
            Term::Var(_) => "_".to_string(),
            Term::List(inner) => format!("List[{}]", self.display(Term::Var(inner))),
        }
    }

    fn infer_block(&mut self, block: BlockRef<'c, '_>) {
        for op in block.operations() {
            self.infer_op(op);
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "one arm for each yzl op the inference types"
    )]
    fn infer_op(&mut self, op: OperationRef<'c, '_>) {
        let declared = self.declared;
        match op.as_yzl() {
            Some(YzlOp::Call(call)) => {
                let callee = call.callee().value();
                match call.callee_source() {
                    Some(CalleeSource::Const) => {
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
                            let expected = substitute(*parameter, &bindings);
                            self.unify(op, term, expected);
                        }

                        let term = self.term_of(op.first_result());
                        let expected = substitute(signature.result, &bindings);
                        self.unify(op, term, expected);
                    }
                }
            }
            Some(YzlOp::Fn(function)) => {
                let Some(signature) = declared.signatures.get(function.sym_name().value()) else {
                    return;
                };

                self.caller = Some(function.sym_name().value());
                self.infer_regions(op, &Row::new(), &signature.params);
                self.unify_returns(op, signature.result);
                self.caller = None;
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
            Some(YzlOp::Const(binding)) => {
                self.infer_regions(op, &Row::new(), &[]);
                let name = binding.sym_name().value();
                let query_row = op
                    .body_terminator()
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
            Some(YzlOp::Store(store)) => {
                let place = self.place_term(store.place());
                let value = self.term_of(store.value());
                self.unify(op, place, value);
            }
            Some(YzlOp::Load(load)) => {
                let place = self.place_term(load.place());
                let result = self.term_of(op.first_result());
                self.unify(op, place, result);
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
                | YzlOp::Rename(_)),
            ) => {
                let row = self.input_row(op);
                self.infer_regions(op, &row, &[]);
                if matches!(stage, YzlOp::Where(_)) {
                    let boolean = BoolType::new(self.context).into();
                    self.expect_yield(op, boolean, "`where` predicate");
                }

                self.record_row(op, row);
            }
            Some(YzlOp::Drop(stage)) => {
                let dropped: Vec<usize> = stage.drop_cols().indices().collect();
                let row = self
                    .input_row(op)
                    .into_iter()
                    .enumerate()
                    .filter(|(index, _)| !dropped.contains(index))
                    .map(|(_, column)| column)
                    .collect();
                self.record_row(op, row);
            }
            Some(YzlOp::Set(stage)) => {
                let row = self.input_row(op);
                self.infer_regions(op, &row, &[]);
                let yields = self.yield_terms(op);
                let columns = stage
                    .set_cols()
                    .into_iter()
                    .flat_map(|columns| columns.indices());
                for (index, term) in columns.zip(yields) {
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
                let mut row: Row = stage
                    .key_cols()
                    .into_iter()
                    .flat_map(|keys| keys.indices())
                    .filter_map(|index| input.get(index).copied())
                    .collect();
                row.extend(self.yield_terms(op));
                self.record_row(op, row);
            }
            Some(YzlOp::Join(join)) => {
                let left = self.input_row(op);
                let rhs = join.rhs().value();
                let right = declared
                    .rows
                    .get(rhs)
                    .or_else(|| self.relations.get(rhs))
                    .cloned()
                    .unwrap_or_default();
                let (left_width, right_width) = (left.len(), right.len());
                let mut row = left;
                row.extend(right);

                if let Some((left_keys, right_keys)) = join.using_keys() {
                    for (&left_key, &right_key) in left_keys.iter().zip(&right_keys) {
                        if let (Some(&lhs), Some(&rhs)) =
                            (row.get(left_key), row.get(left_width + right_key))
                        {
                            self.unify(op, lhs, rhs);
                        }
                    }
                    let order = yuzu_mlir::ops::using_join_order(
                        left_width,
                        right_width,
                        &left_keys,
                        &right_keys,
                    );
                    let joined = order
                        .iter()
                        .filter_map(|&at| row.get(at).copied())
                        .collect();
                    self.record_row(op, joined);
                    return;
                }

                self.infer_regions(op, &row, &[]);
                let boolean = BoolType::new(self.context).into();
                self.expect_yield(op, boolean, "`on` condition");
                self.record_row(op, row);
            }
            Some(_) => self.infer_regions(op, &Row::new(), &[]),
            None => self.infer_yz_op(op),
        }
    }

    /// An operator with an error operand gives an error, whatever its rule
    /// would give: `nosuch + 1` is not an `int64` for having a `1`.
    fn infer_yz_op(&mut self, op: OperationRef<'c, '_>) {
        let error = Term::Concrete(ErrorType::new(self.context).into());
        if let Some(out) = op.try_first_result()
            && op.operands().any(|operand| {
                let term = self.term_of(operand);
                is_error(self.shallow(term))
            })
        {
            let out = self.term_of(out);
            self.unify(op, out, error);
            return;
        }

        let boolean = Term::Concrete(BoolType::new(self.context).into());
        match op.as_yz() {
            Some(
                YzOp::Add(_)
                | YzOp::Sub(_)
                | YzOp::Mul(_)
                | YzOp::Div(_)
                | YzOp::Rem(_)
                | YzOp::Pow(_),
            ) => {
                let (lhs, rhs) = (self.operand_term(op, 0), self.operand_term(op, 1));
                let out = self.term_of(op.first_result());
                self.unify(op, lhs, rhs);
                self.unify(op, lhs, out);
            }
            Some(YzOp::Shl(_) | YzOp::Shr(_)) => {
                let int64 = Term::Concrete(Int64Type::new(self.context).into());
                let (lhs, rhs) = (self.operand_term(op, 0), self.operand_term(op, 1));
                let out = self.term_of(op.first_result());
                self.unify(op, lhs, int64);
                self.unify(op, rhs, int64);
                self.unify(op, out, int64);
            }
            Some(YzOp::In(_)) => {
                let (value, list) = (self.operand_term(op, 0), self.operand_term(op, 1));
                let out = self.term_of(op.first_result());
                let inner = self.fresh();
                self.unify(op, Term::List(inner), list);
                self.unify(op, value, Term::Var(inner));
                self.unify(op, out, boolean);
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
    ) -> FxHashMap<&'c str, TypeVar> {
        let mut bindings = FxHashMap::default();
        let mut params = Vec::new();
        for &name in &signature.type_params {
            let var = self.fresh();
            bindings.insert(name, var);
            params.push((name, var));
        }

        if !params.is_empty()
            && let Some(result) = op.try_first_result()
        {
            self.instances
                .insert(result.id(), Instance { callee, params });
        }

        for bound in &signature.bounds {
            if let Some(&var) = bindings.get(bound.subject) {
                self.pending.push(PendingBound {
                    var,
                    trait_: bound.trait_,
                    callee,
                    caller: self.caller,
                    location: op.location(),
                });
            }
        }

        bindings
    }

    /// Whether a function requires a trait of one of its type parameters.
    fn is_bounded(&self, function: &str, subject: &str, trait_: &str) -> bool {
        self.declared
            .signatures
            .get(function)
            .is_some_and(|signature| {
                signature
                    .bounds
                    .iter()
                    .any(|bound| bound.subject == subject && bound.trait_ == trait_)
            })
    }

    fn check_pending_bounds(&mut self) {
        for bound in mem::take(&mut self.pending) {
            let Some(resolved) = self.resolve(Term::Var(bound.var)) else {
                continue;
            };

            if ErrorType::from_type(resolved).is_some() {
                continue;
            }

            let name = types::name(resolved);
            let implemented = if let Some(param) = ParamType::from_type(resolved) {
                bound
                    .caller
                    .is_some_and(|caller| self.is_bounded(caller, param.name(), bound.trait_))
            } else {
                self.declared.impls.contains(&Implementation {
                    trait_: bound.trait_,
                    target: &name,
                })
            };
            if !implemented {
                emit_error(
                    bound.location,
                    &format!(
                        "`{name}` does not implement `{}`, required by `{}`",
                        crate::written_name(bound.trait_),
                        crate::written_name(bound.callee)
                    ),
                );
            }
        }
    }

    fn operand_term(&mut self, op: OperationRef<'c, '_>, index: usize) -> Term<'c> {
        match op.operand(index) {
            Ok(value) => self.term_of(value),
            Err(_) => Term::Concrete(UnresolvedType::new(self.context).into()),
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
        let Some(terminator) = op.body_terminator() else {
            return Row::new();
        };

        terminator
            .operands()
            .map(|value| self.term_of(value))
            .collect()
    }

    /// A `return` with no value returns unit.
    fn unify_returns(&mut self, op: OperationRef<'c, '_>, ret: Type<'c>) {
        if let Some(terminator) = op.body_terminator()
            && matches!(terminator.as_yzl(), Some(YzlOp::Return(_)))
        {
            let term = match terminator.try_first_operand() {
                Some(value) => self.term_of(value),
                None => Term::Concrete(UnitType::new(self.context).into()),
            };
            self.unify(terminator, Term::Concrete(ret), term);
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

            if matches!(op.as_yzl(), Some(YzlOp::Local(_))) {
                let term = self.place_term(result);
                match self.resolve(term) {
                    Some(element) => result.set_type(RefType::new(self.context, element).into()),
                    // The place takes its initializer's type. The initializer
                    // comes first, so the report is already on it.
                    None => debug_assert!(
                        self.reported_unresolved,
                        "a place is left untyped only after its initializer is reported"
                    ),
                }
                continue;
            }

            let term = self.term_of(result);
            match self.resolve(term) {
                Some(ty) => result.set_type(ty),
                // A hole was already reported by the parse.
                None if matches!(op.as_yzl(), Some(YzlOp::Missing(_))) => {}
                None => {
                    self.reported_unresolved = true;
                    emit_error(
                        op.location(),
                        "the type of this expression could not be inferred",
                    );
                }
            }

            self.stamp_type_args(&mut op, result.id());
        }
    }

    /// A partial answer is worse than none, so a call whose parameters did
    /// not all resolve is left unstamped.
    fn stamp_type_args(&mut self, op: &mut OperationRefMut<'c, '_>, result: ValueId) {
        let Some(Instance { callee, params }) = self.instances.remove(&result) else {
            return;
        };

        let mut resolved: Vec<Attribute<'c>> = Vec::with_capacity(params.len());
        for (name, var) in params {
            match self.resolve(Term::Var(var)) {
                Some(ty) => resolved.push(TypeAttribute::new(ty).into()),
                // A parameter no argument and no use pins down.
                None => emit_error(
                    op.location(),
                    &format!(
                        "the type parameter `{name}` of `{}` could not be inferred",
                        crate::written_name(callee)
                    ),
                ),
            }
        }

        if resolved.len() == resolved.capacity() {
            op.set_attribute(
                "type_args",
                ArrayAttribute::new(self.context, &resolved).into(),
            );
        }
    }
}

fn substitute<'c>(ty: Type<'c>, bindings: &FxHashMap<&'c str, TypeVar>) -> Term<'c> {
    if let Some(param) = ParamType::from_type(ty)
        && let Some(&var) = bindings.get(param.name())
    {
        return Term::Var(var);
    }

    Term::Concrete(ty)
}

fn report(op: OperationRef<'_, '_>, message: &str) {
    emit_error(op.location(), message);
}

fn is_error(term: Term<'_>) -> bool {
    matches!(term, Term::Concrete(ty) if ErrorType::from_type(ty).is_some())
}

/// The result of a `yzl.missing`: what the lowering stood in for what it
/// could not lower.
fn is_hole(value: Value<'_, '_>) -> bool {
    op_result(value)
        .is_some_and(|result| matches!(result.owner().as_yzl(), Some(YzlOp::Missing(_))))
}

fn element(term: Term<'_>) -> Option<Term<'_>> {
    match term {
        Term::List(inner) => Some(Term::Var(inner)),
        Term::Concrete(ty) => ListType::from_type(ty).map(|list| Term::Concrete(list.inner())),
        Term::Var(_) => None,
    }
}

fn parse_signature<'c>(function: FnOp<'c, '_>) -> Option<Signature<'c>> {
    let signature = FunctionType::try_from(function.signature().value()).ok()?;
    let params = (0..signature.input_count())
        .filter_map(|index| signature.input(index).ok())
        .collect();
    let subjects = function
        .bound_params()
        .into_iter()
        .flat_map(|names| names.strings());
    let traits = function
        .bound_traits()
        .into_iter()
        .flat_map(|names| names.symbols());

    Some(Signature {
        params,
        result: signature.result(0).ok()?,
        type_params: function
            .type_params()
            .map(|names| names.strings().collect())
            .unwrap_or_default(),
        bounds: subjects
            .zip(traits)
            .map(|(subject, trait_)| Bound { subject, trait_ })
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};

    use crate::infer_types;
    use crate::test_support;

    fn check(source: &str, expected: &Expect) {
        test_support::check(
            source,
            |context, module| {
                infer_types(context, module);
                crate::promote_locals(context, module);
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
            &expect![[r#"
                module {
                  yzl.struct @Row ["a"] : [!yz.int64] {sym_visibility = "private"}
                  yzl.table @t of @Row {sym_visibility = "private"}
                  yzl.const @cap {
                    %2 = yz.constant_int 42
                    yzl.yield %2 : !yz.int64
                  } {sym_visibility = "private"}
                  yzl.const @small {
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
                    %2 = yzl.call @cap() : () -> !yz.int64 {callee_source = "const"}
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
            r"
struct Row { a: int64, b: int64, rating: float64 }
table t = Row

def f(x: int64) -> int64 { return x * 3 }

from t
|> where a > 10
|> extend f(a) + b as e
|> aggregate sum(e) as s, avg(rating) as r group by b
    ",
            &expect![[r#"
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
                    %4 = yzl.call @yuzu.prelude.sum(%arg3) : (!yz.int64) -> !yz.int64 {callee_source = "external", is_agg, type_args = [!yz.int64]}
                    %5 = yzl.call @yuzu.prelude.avg(%arg2) : (!yz.float64) -> !yz.float64 {callee_source = "external", is_agg, type_args = [!yz.float64]}
                    yzl.yield %4, %5 : !yz.int64, !yz.float64
                  } {key_cols = [1]}
                  yzl.fn @yuzu.prelude.sum generics ["T"] params ["x"] (!yzl.param<"T">) -> !yzl.param<"T"> agg external "sum" {
                  }
                  yzl.fn @yuzu.prelude.avg generics ["T"] params ["x"] (!yzl.param<"T">) -> !yzl.param<"T"> agg external "avg" {
                  }
                  yzl.output %3
                }
            "#]],
        );
    }

    #[test]
    fn instantiates_a_generic_call_per_site() {
        check(
            r"
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
",
            &expect![[r#"
                module {
                  yzl.trait @Numeric {
                    yzl.fn @zero generics ["Self"] params ["x"] (!yzl.param<"Self">) -> !yzl.param<"Self"> {
                    }
                  } {sym_visibility = "private"}
                  yzl.impl @Numeric for @int64 {
                    yzl.fn @zero generics ["Self"] params ["x"] (!yz.int64) -> !yz.int64 {
                    ^bb0(%arg0: !yzl.unresolved):
                      %2 = yz.constant_int 0
                      yzl.return %2 : !yz.int64
                    }
                  }
                  yzl.impl @Numeric for @float64 {
                    yzl.fn @zero generics ["Self"] params ["x"] (!yz.float64) -> !yz.float64 {
                    ^bb0(%arg0: !yzl.unresolved):
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
            r"
external def median(x: float64) -> float64

struct Row { rating: float64 }
table t = Row

from t
|> extend median(rating) as m
",
            &expect![[r#"
                module {
                  yzl.fn @median params ["x"] (!yz.float64) -> !yz.float64 external "median" {
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
            r"
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
",
            &expect![[r"
                error: `str` does not implement `Numeric`, required by `id`
                 --> test.yz:16:11
                   |
                16 | |> extend id(name) as n
                   |           ^^^^^^^^
            "]],
        );
    }

    #[test]
    fn a_list_that_would_contain_itself_is_reported() {
        check(
            "struct Row { a: int64 }\ntable t = Row\nlet xs = []\nfrom t |> where xs in xs\n",
            &expect![[r"
                error: a list cannot contain itself
                 --> test.yz:4:17
                  |
                4 | from t |> where xs in xs
                  |                 ^^^^^^^^
            "]],
        );
    }

    #[test]
    fn a_list_is_checked_against_a_bound() {
        check(
            "trait Numeric {\n    def zero(x: Self) -> Self\n}\nimpl Numeric for int64 {\n    def zero(x: int64) -> int64 { return 0 }\n}\ndef id[T](x: T) -> T where T: Numeric { return x }\nstruct Row { a: int64 }\ntable t = Row\nfrom t |> select id([1, 2]) as v\n",
            &expect![[r"
                error: `List[int64]` does not implement `Numeric`, required by `id`
                 --> test.yz:10:18
                   |
                10 | from t |> select id([1, 2]) as v
                   |                  ^^^^^^^^^^
            "]],
        );
    }

    #[test]
    fn a_type_parameter_satisfies_a_bound_its_function_requires() {
        check(
            "trait Numeric {\n    def zero(x: Self) -> Self\n}\ndef id[T](x: T) -> T where T: Numeric { return x }\ndef twice[T](x: T) -> T where T: Numeric { return id(x) }\ndef loose[T](x: T) -> T { return id(x) }\n",
            &expect![[r"
                error: `T` does not implement `Numeric`, required by `id`
                 --> test.yz:6:34
                  |
                6 | def loose[T](x: T) -> T { return id(x) }
                  |                                  ^^^^^
            "]],
        );
    }

    #[test]
    fn a_local_is_held_to_its_annotation() {
        check(
            "def f(x: int64) -> str {\n  let y: str = x\n  return y\n}\n",
            &expect![[r"
                error: expected `str`, found `int64`
                 --> test.yz:2:3
                  |
                2 |   let y: str = x
                  |   ^^^^^^^^^^^^^^
            "]],
        );
    }

    #[test]
    fn a_local_takes_the_type_of_its_value() {
        test_support::check(
            "def f(x: int64) -> int64 {\n  let mut y = x\n  y = y + 1\n  return y\n}\n",
            |context, module| {
                infer_types(context, module);
                module.as_operation().to_string()
            },
            &expect![[r#"
                module {
                  yzl.fn @f params ["x"] (!yz.int64) -> !yz.int64 {
                  ^bb0(%arg0: !yz.int64):
                    %0 = yzl.local "x" param : !yzl.ref<!yz.int64>
                    yzl.store %0, %arg0 : !yzl.ref<!yz.int64>, !yz.int64
                    %1 = yzl.load %0 : !yzl.ref<!yz.int64> -> !yz.int64
                    %2 = yzl.local "y" mut : !yzl.ref<!yz.int64>
                    yzl.store %2, %1 : !yzl.ref<!yz.int64>, !yz.int64
                    %3 = yzl.load %2 : !yzl.ref<!yz.int64> -> !yz.int64
                    %4 = yz.constant_int 1
                    %5 = yz.add %3, %4 : !yz.int64, !yz.int64 -> !yz.int64
                    yzl.store %2, %5 : !yzl.ref<!yz.int64>, !yz.int64
                    %6 = yzl.load %2 : !yzl.ref<!yz.int64> -> !yz.int64
                    yzl.return %6 : !yz.int64
                  } {sym_visibility = "private"}
                }
            "#]],
        );
    }

    #[test]
    fn reports_a_comparison_mismatch() {
        check(
            r"
struct Row { name: str }
table t = Row

from t
|> where name == 1
    ",
            &expect![[r"
                error: expected `str`, found `int64`
                 --> test.yz:6:10
                  |
                6 | |> where name == 1
                  |          ^^^^^^^^^
            "]],
        );
    }

    #[test]
    fn reports_a_return_mismatch() {
        check(
            r"
struct Row { a: int64 }
table t = Row

def f(x: int64) -> bool { return x }

from t
|> extend f(a) as e
    ",
            &expect![[r"
                error: expected `bool`, found `int64`
                 --> test.yz:5:27
                  |
                5 | def f(x: int64) -> bool { return x }
                  |                           ^^^^^^^^
            "]],
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
            &expect![[r#"
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
            r"
struct Row { a: int64 }
table t = Row

from t
|> where a in [1, 3]
",
            &expect![[r#"
                module {
                  yzl.struct @Row ["a"] : [!yz.int64] {sym_visibility = "private"}
                  yzl.table @t of @Row {sym_visibility = "private"}
                  %0 = yzl.from @t
                  %1 = yzl.where %0 {
                  ^bb0(%arg0: !yz.int64):
                    %2 = yz.constant_list [1, 3] : <!yz.int64>
                    %3 = yz.in %arg0, %2 : !yz.int64, !yz.list<!yz.int64> -> !yz.bool
                    yzl.yield %3 : !yz.bool
                  }
                  yzl.output %1
                }
            "#]],
        );
    }

    #[test]
    fn a_type_parameter_nothing_pins_down_is_reported() {
        check(
            r"
def mk[T]() -> int64 { return 1 }

struct Row { a: int64 }
table t = Row

from t |> select mk() as v
",
            &expect![[r"
                error: the type parameter `T` of `mk` could not be inferred
                 --> test.yz:7:18
                  |
                7 | from t |> select mk() as v
                  |                  ^^^^
            "]],
        );
    }

    #[test]
    fn a_function_without_a_result_type_returns_unit() {
        check(
            r"
def f(x: int64) {
    return x
}
",
            &expect![[r"
                error: expected `unit`, found `int64`
                 --> test.yz:3:5
                  |
                3 |     return x
                  |     ^^^^^^^^
            "]],
        );
    }

    #[test]
    fn a_bare_return_returns_unit() {
        check(
            r"
def f(x: int64) -> int64 {
    return
}
",
            &expect![[r"
                error: expected `int64`, found `unit`
                 --> test.yz:3:5
                  |
                3 |     return
                  |     ^^^^^^
            "]],
        );
    }

    #[test]
    fn an_annotated_let_holds_its_body_to_the_annotation() {
        check(
            r"
struct Row { a: int64 }
table t = Row

let ids: List[str] = [1, 3]

from t
|> where a in ids
",
            &expect![[r"
                error: expected `List[str]`, found `List[int64]`
                 --> test.yz:5:1
                  |
                5 | let ids: List[str] = [1, 3]
                  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^
            "]],
        );
    }

    #[test]
    fn a_stage_names_what_it_expected() {
        check(
            r"
struct Row { level: int64, name: str }
table t = Row
struct Dept { level: int64 }
table depts = Dept

from t
|> where level
",
            &expect![[r"
                error: expected the `where` predicate to be `bool`, found `int64`
                 --> test.yz:8:1
                  |
                8 | |> where level
                  | ^^^^^^^^^^^^^^
            "]],
        );
    }

    #[test]
    fn a_join_condition_has_to_be_a_predicate() {
        check(
            r"
struct Row { level: int64 }
table t = Row
struct Dept { level: int64 }
table depts = Dept

from t as e
|> inner join depts as d on e.level
",
            &expect![[r"
                error: expected the `on` condition to be `bool`, found `int64`
                 --> test.yz:8:1
                  |
                8 | |> inner join depts as d on e.level
                  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            "]],
        );
    }

    #[test]
    fn membership_needs_a_list() {
        check(
            r"
struct Row { a: int64 }
table t = Row

from t
|> where a in 1
",
            &expect![[r"
                error: expected `List[_]`, found `int64`
                 --> test.yz:6:10
                  |
                6 | |> where a in 1
                  |          ^^^^^^
            "]],
        );
    }

    #[test]
    fn reports_an_expression_nothing_pinned_down() {
        check(
            r"
struct Row { a: int64 }
table t = Row

let xs = []

from t
|> where a > 1
",
            &expect![[r"
                error: the type of this expression could not be inferred
                 --> test.yz:5:10
                  |
                5 | let xs = []
                  |          ^^
            "]],
        );
    }

    #[test]
    fn a_local_nothing_pinned_down_is_reported() {
        check(
            r"
struct Row { a: int64 }
table t = Row

def f() -> int64 {
    let xs = []
    return 1
}

from t
|> select f() as v
",
            &expect![[r"
                error: the type of this expression could not be inferred
                 --> test.yz:6:14
                  |
                6 |     let xs = []
                  |              ^^
            "]],
        );
    }

    #[test]
    fn an_empty_list_takes_its_annotation() {
        check(
            r"
struct Row { a: int64 }
table t = Row

let xs: List[int64] = []

from t
|> where a in xs
",
            &expect![[r#"
                module {
                  yzl.struct @Row ["a"] : [!yz.int64] {sym_visibility = "private"}
                  yzl.table @t of @Row {sym_visibility = "private"}
                  yzl.const @xs : !yz.list<!yz.int64> {
                    %2 = yzl.list[] : () -> !yz.list<!yz.int64>
                    yzl.yield %2 : !yz.list<!yz.int64>
                  } {sym_visibility = "private"}
                  %0 = yzl.from @t
                  %1 = yzl.where %0 {
                  ^bb0(%arg0: !yz.int64):
                    %2 = yzl.call @xs() : () -> !yz.list<!yz.int64> {callee_source = "const"}
                    %3 = yz.in %arg0, %2 : !yz.int64, !yz.list<!yz.int64> -> !yz.bool
                    yzl.yield %3 : !yz.bool
                  }
                  yzl.output %1
                }
            "#]],
        );
    }

    #[test]
    fn a_list_takes_the_type_of_its_elements() {
        check(
            r"
struct Row { a: int64 }
table t = Row

let xs = [1, 2]

from t
|> where a > 1
",
            &expect![[r#"
                module {
                  yzl.struct @Row ["a"] : [!yz.int64] {sym_visibility = "private"}
                  yzl.table @t of @Row {sym_visibility = "private"}
                  yzl.const @xs {
                    %2 = yz.constant_list [1, 2] : <!yz.int64>
                    yzl.yield %2 : !yz.list<!yz.int64>
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
