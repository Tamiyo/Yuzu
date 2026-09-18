//! InferTypes: unification over MLIR values. Every `!yzl.var`-typed value is
//! a type variable; op semantics, function signatures, and the field types
//! flowing through the stages fill the substitutions. The answers are written
//! onto the values themselves, so `!yzl.var` is gone by the end and every
//! later pass reads a type rather than a stamp beside one.

use std::collections::{HashMap, HashSet};

use melior::Context;
use melior::ir::attribute::{ArrayAttribute, IntegerAttribute, TypeAttribute};
use melior::ir::operation::{OperationLike, OperationMutLike, OperationRef, OperationRefMut};
use melior::ir::r#type::FunctionType;
use melior::ir::{
    Attribute, BlockLike, BlockRef, Location, Module, RegionLike, Type, Value, ValueLike,
};
use yuzu_mlir::ListType;
use yuzu_mlir::attributes::CalleeKind;
use yuzu_mlir::ext::{
    ArrayAttributeExt, BlockExt, OperationCast, OperationExt, RegionExt, ValueExt,
};
use yuzu_mlir::ops::yz::YzOp;
use yuzu_mlir::ops::yzl::YzlOp;
use yuzu_mlir::types;

/// A type either known or still being solved for: a concrete MLIR type, a
/// type variable whose substitution is still being filled, or a list whose
/// inner type is the variable's.
#[derive(Clone, Copy)]
enum Term<'c> {
    Concrete(Type<'c>),
    Var(usize),
    List(usize),
}

type Row<'c> = Vec<Term<'c>>;

/// A function's declared shape: the types as written, plus the parameters
/// a call instantiates and the bounds those parameters carry.
#[derive(Clone)]
struct Signature<'c> {
    params: Vec<Type<'c>>,
    result: Type<'c>,
    type_params: Vec<&'c str>,
    bounds: Vec<(&'c str, &'c str)>,
}

/// A bound one instantiation owes: the variable standing for the parameter,
/// the trait it must satisfy, and where to report if it does not.
struct PendingBound<'c> {
    var: usize,
    r#trait: &'c str,
    callee: &'c str,
    location: Location<'c>,
}

struct TypeInferrer<'c> {
    context: &'c Context,
    /// One slot per type variable: unbound, substituted by another variable,
    /// or filled with its concrete type.
    filled: Vec<Option<Term<'c>>>,
    /// The type variable of each `!yzl.var` value, keyed by value identity —
    /// a hash map because MLIR values are pointers, not arena indices.
    vars: HashMap<usize, usize>,
    /// The row of column terms flowing out of each stage value.
    rows: HashMap<usize, Row<'c>>,
    signatures: HashMap<&'c str, Signature<'c>>,
    /// The `(trait, type)` pairs the module's implementations supply.
    impls: HashSet<(&'c str, &'c str)>,
    /// A bound to check once the type variable standing for its parameter
    /// resolves — deferred, because the argument may resolve after the call.
    pending: Vec<PendingBound<'c>>,
    /// The type variables a generic call minted, in the order its function
    /// declared them — resolved and stamped once inference settles.
    instances: HashMap<usize, Vec<usize>>,
    relations: HashMap<&'c str, Row<'c>>,
}

/// Expects a verified module: required ODS attributes are read through
/// typed accessors that panic when absent. Diagnostics go through MLIR —
/// run this inside `yuzu_mlir::diagnostics::capture` to collect them.
pub fn infer_types<'c>(context: &'c Context, module: &Module<'c>) {
    let mut inferrer = TypeInferrer {
        context,
        filled: Vec::new(),
        vars: HashMap::new(),
        rows: HashMap::new(),
        signatures: HashMap::new(),
        impls: HashSet::new(),
        pending: Vec::new(),
        instances: HashMap::new(),
        relations: HashMap::new(),
    };

    inferrer.hoist(module.body());
    inferrer.infer_block(module.body(), &Row::new(), &[]);
    inferrer.check_pending_bounds();
    inferrer.stamp_block(module.body());
}

impl<'c> TypeInferrer<'c> {
    /// The term of a value: its concrete type, or the type variable minted
    /// for it — a `!yzl.var` value is its own variable.
    fn term_of(&mut self, value: Value<'c, '_>) -> Term<'c> {
        let ty = value.r#type();
        if ty != types::var(self.context) {
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

    fn fresh(&mut self) -> usize {
        self.filled.push(None);
        self.filled.len() - 1
    }

    /// Follows the substitution chain to a variable's root, compressing the
    /// chain on the way.
    fn find(&mut self, var: usize) -> usize {
        match self.filled[var] {
            Some(Term::Var(next)) => {
                let root = self.find(next);
                self.filled[var] = Some(Term::Var(root));
                root
            }
            _ => var,
        }
    }

    /// The root of a variable's chain, or what the root is filled with.
    fn shallow(&mut self, term: Term<'c>) -> Term<'c> {
        match term {
            Term::Var(var) => {
                let root = self.find(var);
                self.filled[root].unwrap_or(Term::Var(root))
            }
            term => term,
        }
    }

    fn unify(&mut self, op: OperationRef<'c, '_>, a: Term<'c>, b: Term<'c>) {
        let a = self.shallow(a);
        let b = self.shallow(b);
        match (a, b) {
            (Term::Var(a), Term::Var(b)) if a == b => {}
            (Term::Var(var), term) | (term, Term::Var(var)) => self.filled[var] = Some(term),
            (Term::Concrete(a), Term::Concrete(b)) => {
                if a != b {
                    let (a, b) = (
                        self.display(Term::Concrete(a)),
                        self.display(Term::Concrete(b)),
                    );
                    self.error(op, format!("expected `{a}`, found `{b}`"));
                }
            }
            (Term::List(a), Term::List(b)) => self.unify(op, Term::Var(a), Term::Var(b)),
            (Term::List(inner), Term::Concrete(ty)) => match ListType::from_type(ty) {
                Some(list) => self.unify(op, Term::Var(inner), Term::Concrete(list.inner())),
                None => {
                    let (expected, found) = (
                        self.display(Term::List(inner)),
                        self.display(Term::Concrete(ty)),
                    );
                    self.error(op, format!("expected `{expected}`, found `{found}`"));
                }
            },
            (Term::Concrete(ty), Term::List(inner)) => match ListType::from_type(ty) {
                Some(list) => self.unify(op, Term::Concrete(list.inner()), Term::Var(inner)),
                None => {
                    let (expected, found) = (
                        self.display(Term::Concrete(ty)),
                        self.display(Term::List(inner)),
                    );
                    self.error(op, format!("expected `{expected}`, found `{found}`"));
                }
            },
        }
    }

    /// The value a stage's region yields, held to the type the stage needs.
    /// Whatever it settled on is named for the reader, so the complaint is
    /// about the predicate they wrote rather than about a type variable.
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
                self.error(
                    op,
                    format!("expected the {what} to be `{expected}`, found `{found}`"),
                );
            }
            // Nothing pinned it down, so the stage is what says what it is.
            None => self.unify(op, term, Term::Concrete(expected)),
            Some(_) => {}
        }
    }

    /// Resolves (collapses) a term to its final type, when it has one.
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

    /// A term as the program would have written it, with `_` where nothing
    /// is known yet.
    fn display(&mut self, term: Term<'c>) -> String {
        match self.shallow(term) {
            Term::Concrete(ty) => types::name(self.context, ty),
            Term::Var(_) => "_".to_string(),
            Term::List(inner) => format!("List[{}]", self.display(Term::Var(inner))),
        }
    }

    fn hoist(&mut self, block: BlockRef<'c, '_>) {
        for op in block.operations() {
            match op.as_yzl() {
                Some(YzlOp::Fn(function)) => {
                    if let Some(signature) = parse_signature(&function) {
                        self.signatures
                            .insert(function.sym_name().value(), signature);
                    }
                }
                Some(YzlOp::Impl(item)) => {
                    self.impls
                        .insert((item.r#trait().value(), item.target().value()));
                }
                Some(YzlOp::Struct(item)) => {
                    let row = field_row(item.types());
                    self.relations.insert(item.sym_name().value(), row);
                }
                Some(YzlOp::Table(table)) => {
                    let row = self
                        .relations
                        .get(table.row().value())
                        .cloned()
                        .unwrap_or_default();

                    self.relations.insert(table.sym_name().value(), row);
                }
                _ => {}
            }
        }
    }

    fn infer_block(&mut self, block: BlockRef<'c, '_>, columns: &Row<'c>, params: &[Type<'c>]) {
        for op in block.operations() {
            self.infer_op(op, columns, params);
        }
    }

    fn infer_op(&mut self, op: OperationRef<'c, '_>, columns: &Row<'c>, params: &[Type<'c>]) {
        match op.as_yzl() {
            Some(YzlOp::Call(call)) => {
                let callee = call.callee().value();
                match call.callee_kind() {
                    Some(CalleeKind::Builtin) => self.resolve_builtin_ty(op, callee),
                    // An external declares its signature like any other
                    // function; only the body is missing, and a call does not
                    // need one. Skipping it left the call's type unresolved
                    // and carried a `!yzl.var` all the way into yzr.
                    Some(CalleeKind::Let) => {
                        let yielded = self
                            .relations
                            .get(callee)
                            .and_then(|row| row.first().copied());
                        if let Some(yielded) = yielded {
                            let term = self.term_of(op.first_result());
                            self.unify(op, term, yielded);
                        }
                    }
                    Some(CalleeKind::Fn | CalleeKind::AggFn | CalleeKind::External) | None => {
                        let Some(signature) = self.signatures.get(callee).cloned() else {
                            return;
                        };

                        let bindings = self.instantiate(op, callee, &signature);
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
                let Some(signature) = self.signatures.get(function.sym_name().value()) else {
                    return;
                };

                let (parameters, result) = (signature.params.clone(), signature.result);
                self.infer_regions(op, &Row::new(), &parameters);
                self.unify_returns(op, result);
            }
            Some(YzlOp::From(from)) => {
                let row = self
                    .relations
                    .get(from.source().value())
                    .cloned()
                    .unwrap_or_default();
                self.record_row(op, row);
            }
            Some(YzlOp::Let(binding)) => {
                self.infer_regions(op, &Row::new(), &[]);
                let row = self.yield_terms(op);
                if let Some(annotation) = binding.annotation()
                    && let Some(&yielded) = row.first()
                {
                    self.unify(op, Term::Concrete(annotation.value()), yielded);
                }

                self.relations.insert(binding.sym_name().value(), row);
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
                    let boolean = types::boolean(self.context);
                    self.expect_yield(op, boolean, "`where` predicate");
                }

                self.record_row(op, row);
            }
            Some(YzlOp::Set(stage)) => {
                let row = self.input_row(op);
                self.infer_regions(op, &row, &[]);
                let yields = self.yield_terms(op);
                for (index, term) in indices(stage.set_cols()).into_iter().zip(yields) {
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
                let mut row: Row = indices(stage.key_cols())
                    .into_iter()
                    .filter_map(|index| input.get(index).copied())
                    .collect();
                row.extend(self.yield_terms(op));
                self.record_row(op, row);
            }
            Some(YzlOp::Join(_)) => {
                let row = self.input_row(op);
                self.infer_regions(op, &row, &[]);
                // `using` names its columns instead, and leaves no region.
                let boolean = types::boolean(self.context);
                self.expect_yield(op, boolean, "`on` condition");
                self.record_row(op, row);
            }
            Some(_) => self.infer_regions(op, columns, params),
            None => self.infer_yz_op(op, columns, params),
        }
    }

    fn infer_yz_op(&mut self, op: OperationRef<'c, '_>, columns: &Row<'c>, params: &[Type<'c>]) {
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
                self.unify(op, out, Term::Concrete(types::boolean(self.context)));
            }
            Some(YzOp::And(_) | YzOp::Or(_)) => {
                let (lhs, rhs) = (self.operand_term(op, 0), self.operand_term(op, 1));
                let out = self.term_of(op.first_result());
                self.unify(op, lhs, Term::Concrete(types::boolean(self.context)));
                self.unify(op, rhs, Term::Concrete(types::boolean(self.context)));
                self.unify(op, out, Term::Concrete(types::boolean(self.context)));
            }
            Some(YzOp::Not(_)) => {
                let value = self.operand_term(op, 0);
                let out = self.term_of(op.first_result());
                self.unify(op, value, Term::Concrete(types::boolean(self.context)));
                self.unify(op, out, Term::Concrete(types::boolean(self.context)));
            }
            _ => self.infer_regions(op, columns, params),
        }
    }

    /// Mints one fresh variable per type parameter, so every call to a
    /// generic function solves its own instance, and records the bounds that
    /// instance owes.
    fn instantiate(
        &mut self,
        op: OperationRef<'c, '_>,
        callee: &'c str,
        signature: &Signature<'c>,
    ) -> HashMap<&'c str, usize> {
        let mut bindings = HashMap::new();
        let mut ordered = Vec::new();
        for name in &signature.type_params {
            let var = self.fresh();
            bindings.insert(*name, var);
            ordered.push(var);
        }

        // Which type the call chose for each parameter is inference's answer,
        // and expansion needs it to pick an implementation. Recorded now,
        // stamped once the variables resolve.
        if !ordered.is_empty()
            && let Some(result) = op.try_first_result()
        {
            self.instances.insert(result.id(), ordered);
        }

        for (subject, r#trait) in &signature.bounds {
            if let Some(&var) = bindings.get(subject) {
                self.pending.push(PendingBound {
                    var,
                    r#trait,
                    callee,
                    location: op.location(),
                });
            }
        }

        bindings
    }

    /// A declared type with this instance's parameters swapped in.
    fn substitute(&self, ty: Type<'c>, bindings: &HashMap<&'c str, usize>) -> Term<'c> {
        if let Some(param) = yuzu_mlir::ParamType::from_type(ty)
            && let Some(&var) = bindings.get(param.name())
        {
            return Term::Var(var);
        }

        Term::Concrete(ty)
    }

    /// Every instantiation owes its bounds once its type is known.
    fn check_pending_bounds(&mut self) {
        for index in 0..self.pending.len() {
            let bound = &self.pending[index];
            let (var, r#trait, callee, location) =
                (bound.var, bound.r#trait, bound.callee, bound.location);
            let Some(resolved) = self.resolve(Term::Var(var)) else {
                continue;
            };

            let Some(name) = self.type_name(resolved) else {
                continue;
            };

            if !self.impls.contains(&(r#trait, name)) {
                yuzu_mlir::diagnostics::emit_error(
                    location,
                    &format!("`{name}` does not implement `{trait}`, required by `{callee}`"),
                );
            }
        }
    }

    /// The name an `impl` would target this type by.
    fn type_name(&self, ty: Type<'c>) -> Option<&'static str> {
        if ty == types::int64(self.context) {
            return Some("int64");
        }
        if ty == types::float64(self.context) {
            return Some("float64");
        }
        if ty == types::boolean(self.context) {
            return Some("bool");
        }
        if ty == types::str(self.context) {
            return Some("str");
        }

        None
    }

    fn operand_term(&mut self, op: OperationRef<'c, '_>, index: usize) -> Term<'c> {
        match op.operand(index) {
            Ok(value) => self.term_of(value),
            Err(_) => Term::Concrete(types::var(self.context)),
        }
    }

    /// The aggregate builtins are polymorphic; these are the old
    /// `resolve_agg_ty` rules over terms.
    fn resolve_builtin_ty(&mut self, op: OperationRef<'c, '_>, callee: &str) {
        let out = self.term_of(op.first_result());
        match callee {
            "count" | "count_distinct" => {
                self.unify(op, out, Term::Concrete(types::int64(self.context)));
            }
            "sum" => {
                if let Some(argument) = op.try_first_operand() {
                    let term = self.term_of(argument);
                    if let Some(ty) = self.resolve(term) {
                        let result = if ty == types::float64(self.context) {
                            types::float64(self.context)
                        } else {
                            types::int64(self.context)
                        };

                        self.unify(op, out, Term::Concrete(result));
                    }
                }
            }
            "in" => {
                let (lhs, rhs) = (self.operand_term(op, 0), self.operand_term(op, 1));
                let inner = self.fresh();
                self.unify(op, Term::List(inner), rhs);
                self.unify(op, lhs, Term::Var(inner));
                self.unify(op, out, Term::Concrete(types::boolean(self.context)));
            }
            "min" | "max" | "avg" => {
                if let Some(argument) = op.try_first_operand() {
                    let term = self.term_of(argument);
                    self.unify(op, out, term);
                }
            }
            // The scalar builtins the parser lowers operators to: `**` keeps
            // its operands' type, the shifts are integer-only.
            "pow" => {
                let (lhs, rhs) = (self.operand_term(op, 0), self.operand_term(op, 1));
                self.unify(op, lhs, rhs);
                self.unify(op, lhs, out);
            }
            "shift_left" | "shift_right" => {
                let (lhs, rhs) = (self.operand_term(op, 0), self.operand_term(op, 1));
                self.unify(op, lhs, Term::Concrete(types::int64(self.context)));
                self.unify(op, rhs, Term::Concrete(types::int64(self.context)));
                self.unify(op, out, Term::Concrete(types::int64(self.context)));
            }
            _ => {}
        }
    }

    /// A region's block arguments are the row's columns or the function's
    /// parameters, by position; each takes that term before the body is
    /// inferred against it.
    fn infer_regions(&mut self, op: OperationRef<'c, '_>, columns: &Row<'c>, params: &[Type<'c>]) {
        for region in op.regions() {
            for block in region.blocks() {
                for index in 0..block.argument_count() {
                    let argument = block
                        .argument(index)
                        .expect("the argument index is in range");
                    let term = self.term_of(argument.into());
                    if let Some(&column) = columns.get(index) {
                        self.unify(op, term, column);
                    } else if let Some(&param) = params.get(index) {
                        self.unify(op, term, Term::Concrete(param));
                    }
                }

                self.infer_block(block, columns, params);
            }
        }
    }

    // --- rows and yields ---

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

    /// Inference's answers, written onto the values themselves. `!yzl.var` is
    /// this pass's construct, so no later pass should meet one: a value that
    /// resolved takes the type it resolved to, and one that did not is
    /// reported here, where the expression that stayed open is still in hand.
    fn stamp_block(&mut self, block: BlockRef<'c, '_>) {
        for index in 0..block.argument_count() {
            let argument = block
                .argument(index)
                .expect("the argument index is in range");
            let term = self.term_of(argument.into());
            // A column or parameter is only ever as open as the expressions
            // reading it, and those are what the program wrote — so an
            // unresolved one is left for them to report.
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
                // A hole stands for an expression the program never got to
                // write, and the parse error above it already said so.
                None if matches!(op.as_yzl(), Some(YzlOp::Missing(_))) => {}
                None => self.error_at(
                    op.location(),
                    "the type of this expression could not be inferred".to_string(),
                ),
            }

            self.stamp_type_args(&mut op, result.id());
        }
    }

    /// The types a generic call settled on, in declaration order. A partial
    /// answer is worse than none, so a call whose parameters did not all
    /// resolve is left unstamped for expansion to report.
    fn stamp_type_args(&mut self, op: &mut OperationRefMut<'c, '_>, result: usize) {
        let Some(vars) = self.instances.get(&result).cloned() else {
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

    fn error(&mut self, op: OperationRef<'c, '_>, message: String) {
        self.error_at(op.location(), message);
    }

    fn error_at(&mut self, location: Location<'c>, message: String) {
        yuzu_mlir::diagnostics::emit_error(location, &message);
    }
}

fn last_region_op<'c, 'a>(op: OperationRef<'c, 'a>) -> Option<OperationRef<'c, 'a>> {
    op.regions().next()?.first_block()?.last_operation()
}

fn parse_signature<'c>(function: &yuzu_mlir::ops::yzl::FnOp<'c, '_>) -> Option<Signature<'c>> {
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

/// The indices an optional stamp carries.
pub(crate) fn indices(stamp: Option<melior::ir::attribute::ArrayAttribute>) -> Vec<usize> {
    stamp
        .map(|array| {
            array
                .elements()
                .filter_map(|element| IntegerAttribute::try_from(element).ok())
                .map(|index| index.value() as usize)
                .collect()
        })
        .unwrap_or_default()
}

/// The field types of a struct declaration's `types` array.
fn field_row<'c>(types: melior::ir::attribute::ArrayAttribute<'c>) -> Row<'c> {
    types
        .elements()
        .filter_map(|element| TypeAttribute::try_from(element).ok())
        .map(|ty| Term::Concrete(ty.value()))
        .collect()
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
                infer_types(context, module);
            },
            expected,
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
                  yzl.struct @Row ["a", "b", "rating"] : [!yz.int64, !yz.int64, !yz.float64]
                  yzl.table @t of @Row
                  yzl.fn @f params ["x"] (!yz.int64) -> !yz.int64 {
                  ^bb0(%arg0: !yz.int64):
                    %4 = yz.constant_int 3
                    %5 = yz.mul %arg0, %4 : !yz.int64, !yz.int64 -> !yz.int64
                    yzl.return %5 : !yz.int64
                  }
                  %0 = yzl.from @t
                  %1 = yzl.where %0 {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64, %arg2: !yz.float64):
                    %4 = yz.constant_int 10
                    %5 = yz.cmp "gt", %arg0, %4 : !yz.int64, !yz.int64 -> !yz.bool
                    yzl.yield %5 : !yz.bool
                  }
                  %2 = yzl.extend %1 as ["e"] {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64, %arg2: !yz.float64):
                    %4 = yzl.call @f(%arg0) : (!yz.int64) -> !yz.int64 {callee_kind = "fn"}
                    %5 = yz.add %4, %arg1 : !yz.int64, !yz.int64 -> !yz.int64
                    yzl.yield %5 : !yz.int64
                  }
                  %3 = yzl.aggregate %2 group_by ["b"] as ["s", "r"] {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64, %arg2: !yz.float64, %arg3: !yz.int64):
                    %4 = yzl.call @sum(%arg3) : (!yz.int64) -> !yz.int64 {agg, callee_kind = "builtin"}
                    %5 = yzl.call @avg(%arg2) : (!yz.float64) -> !yz.float64 {agg, callee_kind = "builtin"}
                    yzl.yield %4, %5 : !yz.int64, !yz.float64
                  } {key_cols = [1]}
                  yzl.output %3
                }
            "#]],
        );
    }

    /// Each call to a generic function solves its own instance, so one
    /// declaration serves both column types — and each call carries what it
    /// settled on, which is how expansion later knows `@id` at `int64` wants
    /// the `int64` implementation.
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
                  }
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
                  }
                  yzl.struct @Row ["a", "r"] : [!yz.int64, !yz.float64]
                  yzl.table @t of @Row
                  %0 = yzl.from @t
                  %1 = yzl.extend %0 as ["m", "n"] {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.float64):
                    %2 = yzl.call @id(%arg0) : (!yz.int64) -> !yz.int64 {callee_kind = "fn", type_args = [!yz.int64]}
                    %3 = yzl.call @id(%arg1) : (!yz.float64) -> !yz.float64 {callee_kind = "fn", type_args = [!yz.float64]}
                    yzl.yield %2, %3 : !yz.int64, !yz.float64
                  }
                  yzl.output %1
                }
            "#]],
        );
    }

    /// An external has no body, but it has a signature, and a call only ever
    /// needed that. Skipping it left the call's own type unsolved.
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
                  }
                  yzl.struct @Row ["rating"] : [!yz.float64]
                  yzl.table @t of @Row
                  %0 = yzl.from @t
                  %1 = yzl.extend %0 as ["m"] {
                  ^bb0(%arg0: !yz.float64):
                    %2 = yzl.call @median(%arg0) : (!yz.float64) -> !yz.float64 {callee_kind = "external"}
                    yzl.yield %2 : !yz.float64
                  }
                  yzl.output %1
                }
            "#]],
        );
    }

    /// A bound is owed by the instance, and reported where the call is.
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
                   |           ^
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
                  |          ^
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
                  |                           ^
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
                  | ^
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
                  yzl.struct @Row ["a"] : [!yz.int64]
                  yzl.table @t of @Row
                  %0 = yzl.from @t
                  %1 = yzl.where %0 {
                  ^bb0(%arg0: !yz.int64):
                    %2 = yz.constant_int 1
                    %3 = yz.constant_int 3
                    %4 = yzl.list[%2, %3] : (!yz.int64, !yz.int64) -> !yz.list<!yz.int64>
                    %5 = yzl.call @in(%arg0, %4) : (!yz.int64, !yz.list<!yz.int64>) -> !yz.bool {callee_kind = "builtin"}
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
                  | ^
            "#]],
        );
    }

    /// The stage a value is yielded to is what says what it must be, so the
    /// complaint names the predicate rather than a type variable.
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
                  | ^
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
                  | ^
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
                  |          ^
            "#]],
        );
    }

    /// Nothing says what an empty list holds, and the language has no way to
    /// write it down. Inference owns `!yzl.var`, so it says so here rather
    /// than handing a later pass a type it cannot read.
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
                  |          ^
            "#]],
        );
    }

    /// The same list, given an element to take its type from, resolves — and
    /// the answer is on the value, not beside it.
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
                  yzl.struct @Row ["a"] : [!yz.int64]
                  yzl.table @t of @Row
                  yzl.let @xs {
                    %2 = yz.constant_int 1
                    %3 = yz.constant_int 2
                    %4 = yzl.list[%2, %3] : (!yz.int64, !yz.int64) -> !yz.list<!yz.int64>
                    yzl.yield %4 : !yz.list<!yz.int64>
                  }
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
