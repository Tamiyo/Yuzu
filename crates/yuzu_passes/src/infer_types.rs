//! InferTypes: unification over MLIR values. Every `!yzl.var`-typed value is
//! a type variable; op semantics, function signatures, and the field types
//! flowing through the stages fill the substitutions. Answers are stamped as
//! `{ty = …}` attributes — LowerYZL applies them during its rebuild, so
//! inference itself rewrites nothing.

use std::collections::{HashMap, HashSet};

use melior::Context;
use melior::ir::attribute::{ArrayAttribute, IntegerAttribute, TypeAttribute};
use melior::ir::operation::{OperationLike, OperationMutLike, OperationRef, OperationRefMut};
use melior::ir::r#type::FunctionType;
use melior::ir::{Attribute, BlockRef, Location, Module, RegionLike, Type, Value, ValueLike};
use yuzu_mlir::attributes::CalleeKind;
use yuzu_mlir::ext::{ArrayAttributeExt, BlockExt, OperationExt, RegionExt};
use yuzu_mlir::ops::yz::YzOperationRef;
use yuzu_mlir::ops::yzl::YzlOperationRef;
use yuzu_mlir::types;

/// A type either known or still being solved for: a concrete MLIR type, or a
/// type variable whose substitution is still being filled.
#[derive(Clone, Copy)]
enum Term<'c> {
    Concrete(Type<'c>),
    Var(usize),
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

        let key = yuzu_mlir::value_id(value);
        if let Some(&var) = self.vars.get(&key) {
            return Term::Var(var);
        }

        let var = self.filled.len();
        self.filled.push(None);
        self.vars.insert(key, var);
        Term::Var(var)
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

    fn unify(&mut self, op: OperationRef<'c, '_>, a: Term<'c>, b: Term<'c>) {
        match (a, b) {
            (Term::Concrete(a), Term::Concrete(b)) => {
                if a != b {
                    self.error(op, format!("expected `{a}`, found `{b}`"));
                }
            }
            (Term::Var(var), Term::Concrete(ty)) | (Term::Concrete(ty), Term::Var(var)) => {
                self.fill(op, var, ty);
            }
            (Term::Var(a), Term::Var(b)) => self.merge(op, a, b),
        }
    }

    /// Fills a type variable's substitution with a concrete type.
    fn fill(&mut self, op: OperationRef<'c, '_>, var: usize, ty: Type<'c>) {
        let root = self.find(var);
        match self.filled[root] {
            Some(Term::Concrete(filled)) if filled != ty => {
                self.error(op, format!("expected `{filled}`, found `{ty}`"));
            }
            _ => self.filled[root] = Some(Term::Concrete(ty)),
        }
    }

    /// Merges the substitutions of two type variables.
    fn merge(&mut self, op: OperationRef<'c, '_>, a: usize, b: usize) {
        let a = self.find(a);
        let b = self.find(b);
        if a == b {
            return;
        }

        let merged = match (self.filled[a], self.filled[b]) {
            (Some(Term::Concrete(left)), Some(Term::Concrete(right))) if left != right => {
                self.error(op, format!("expected `{left}`, found `{right}`"));
                Some(left)
            }
            (Some(Term::Concrete(ty)), _) | (_, Some(Term::Concrete(ty))) => Some(ty),
            _ => None,
        };

        self.filled[b] = Some(Term::Var(a));
        if let Some(ty) = merged {
            self.filled[a] = Some(Term::Concrete(ty));
        }
    }

    /// Resolves (collapses) a term to its final type, when it has one.
    fn resolve(&mut self, term: Term<'c>) -> Option<Type<'c>> {
        match term {
            Term::Concrete(ty) => Some(ty),
            Term::Var(var) => {
                let root = self.find(var);
                match self.filled[root] {
                    Some(Term::Concrete(ty)) => Some(ty),
                    _ => None,
                }
            }
        }
    }

    fn hoist(&mut self, block: BlockRef<'c, '_>) {
        for op in block.operations() {
            match YzlOperationRef::of(&op) {
                Some(YzlOperationRef::Fn(function)) => {
                    if let Some(signature) = parse_signature(&function) {
                        self.signatures
                            .insert(function.sym_name().value(), signature);
                    }
                }
                Some(YzlOperationRef::Impl(item)) => {
                    self.impls
                        .insert((item.r#trait().value(), item.target().value()));
                }
                Some(YzlOperationRef::Struct(item)) => {
                    let row = field_row(item.types());
                    self.relations.insert(item.sym_name().value(), row);
                }
                Some(YzlOperationRef::Table(table)) => {
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
        match YzlOperationRef::of(&op) {
            Some(YzlOperationRef::Name(name)) => {
                let term = self.term_of(op.first_result());
                if let Some(index) = name.col().map(|col| col.value() as usize) {
                    if let Some(&column) = columns.get(index) {
                        self.unify(op, term, column);
                    }
                } else if let Some(index) = name.param().map(|param| param.value() as usize)
                    && let Some(&param) = params.get(index)
                {
                    self.unify(op, term, Term::Concrete(param));
                }
            }
            Some(YzlOperationRef::Call(call)) => {
                let callee = call.callee().value();
                match call.callee_kind() {
                    Some(CalleeKind::Builtin) => self.resolve_builtin_ty(op, callee),
                    Some(CalleeKind::External) => {}
                    _ => {
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
            Some(YzlOperationRef::Fn(function)) => {
                let Some(signature) = self.signatures.get(function.sym_name().value()) else {
                    return;
                };

                let (parameters, result) = (signature.params.clone(), signature.result);
                self.infer_regions(op, &Row::new(), &parameters);
                self.unify_returns(op, result);
            }
            Some(YzlOperationRef::From(from)) => {
                let row = self
                    .relations
                    .get(from.source().value())
                    .cloned()
                    .unwrap_or_default();
                self.record_row(op, row);
            }
            Some(YzlOperationRef::Let(binding)) => {
                self.infer_regions(op, &Row::new(), &[]);
                let row = self.yield_terms(op);
                self.relations.insert(binding.sym_name().value(), row);
            }
            Some(
                stage @ (YzlOperationRef::Where(_)
                | YzlOperationRef::Distinct(_)
                | YzlOperationRef::Limit(_)
                | YzlOperationRef::Alias(_)
                | YzlOperationRef::Rename(_)
                | YzlOperationRef::Drop(_)),
            ) => {
                let row = self.input_row(op);
                self.infer_regions(op, &row, &[]);
                if matches!(stage, YzlOperationRef::Where(_)) {
                    self.unify_yield(op, Term::Concrete(types::boolean(self.context)));
                }

                self.record_row(op, row);
            }
            Some(YzlOperationRef::Set(stage)) => {
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
            Some(YzlOperationRef::Select(_)) => {
                let input = self.input_row(op);
                self.infer_regions(op, &input, &[]);
                let row = self.yield_terms(op);
                self.record_row(op, row);
            }
            Some(YzlOperationRef::Extend(_)) => {
                let mut row = self.input_row(op);
                self.infer_regions(op, &row, &[]);
                row.extend(self.yield_terms(op));
                self.record_row(op, row);
            }
            Some(YzlOperationRef::Aggregate(stage)) => {
                let input = self.input_row(op);
                self.infer_regions(op, &input, &[]);
                let mut row: Row = indices(stage.key_cols())
                    .into_iter()
                    .filter_map(|index| input.get(index).copied())
                    .collect();
                row.extend(self.yield_terms(op));
                self.record_row(op, row);
            }
            Some(YzlOperationRef::Join(_)) => {
                let row = self.input_row(op);
                self.infer_regions(op, &row, &[]);
                self.record_row(op, row);
            }
            Some(_) => self.infer_regions(op, columns, params),
            None => self.infer_yz_op(op, columns, params),
        }
    }

    fn infer_yz_op(&mut self, op: OperationRef<'c, '_>, columns: &Row<'c>, params: &[Type<'c>]) {
        match YzOperationRef::of(&op) {
            Some(
                YzOperationRef::Add(_)
                | YzOperationRef::Sub(_)
                | YzOperationRef::Mul(_)
                | YzOperationRef::Div(_)
                | YzOperationRef::Rem(_),
            ) => {
                let (lhs, rhs) = (self.operand_term(op, 0), self.operand_term(op, 1));
                let out = self.term_of(op.first_result());
                self.unify(op, lhs, rhs);
                self.unify(op, lhs, out);
            }
            Some(YzOperationRef::Neg(_)) => {
                let value = self.operand_term(op, 0);
                let out = self.term_of(op.first_result());
                self.unify(op, value, out);
            }
            Some(YzOperationRef::Cmp(_)) => {
                let (lhs, rhs) = (self.operand_term(op, 0), self.operand_term(op, 1));
                let out = self.term_of(op.first_result());
                self.unify(op, lhs, rhs);
                self.unify(op, out, Term::Concrete(types::boolean(self.context)));
            }
            Some(YzOperationRef::And(_) | YzOperationRef::Or(_)) => {
                let (lhs, rhs) = (self.operand_term(op, 0), self.operand_term(op, 1));
                let out = self.term_of(op.first_result());
                self.unify(op, lhs, Term::Concrete(types::boolean(self.context)));
                self.unify(op, rhs, Term::Concrete(types::boolean(self.context)));
                self.unify(op, out, Term::Concrete(types::boolean(self.context)));
            }
            Some(YzOperationRef::Not(_)) => {
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
            let var = self.filled.len();
            self.filled.push(None);
            bindings.insert(*name, var);
            ordered.push(var);
        }

        // Which type the call chose for each parameter is inference's answer,
        // and expansion needs it to pick an implementation. Recorded now,
        // stamped once the variables resolve.
        if !ordered.is_empty()
            && let Some(result) = op.try_first_result()
        {
            self.instances.insert(yuzu_mlir::value_id(result), ordered);
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

    fn infer_regions(&mut self, op: OperationRef<'c, '_>, columns: &Row<'c>, params: &[Type<'c>]) {
        for region in op.regions() {
            for block in region.blocks() {
                self.infer_block(block, columns, params);
            }
        }
    }

    // --- rows and yields ---

    fn input_row(&mut self, op: OperationRef<'c, '_>) -> Row<'c> {
        op.try_first_operand()
            .and_then(|input| self.rows.get(&yuzu_mlir::value_id(input)).cloned())
            .unwrap_or_default()
    }

    fn record_row(&mut self, op: OperationRef<'c, '_>, row: Row<'c>) {
        if let Some(result) = op.try_first_result() {
            self.rows.insert(yuzu_mlir::value_id(result), row);
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

    fn unify_yield(&mut self, op: OperationRef<'c, '_>, expected: Term<'c>) {
        if let Some(terminator) = last_region_op(op)
            && let Some(value) = terminator.try_first_operand()
        {
            let term = self.term_of(value);
            self.unify(op, term, expected);
        }
    }

    fn unify_returns(&mut self, op: OperationRef<'c, '_>, ret: Type<'c>) {
        if let Some(terminator) = last_region_op(op)
            && matches!(
                YzlOperationRef::of(&terminator),
                Some(YzlOperationRef::Return(_))
            )
            && let Some(value) = terminator.try_first_operand()
        {
            let term = self.term_of(value);
            self.unify(terminator, term, Term::Concrete(ret));
        }
    }

    fn stamp_block(&mut self, block: BlockRef<'c, '_>) {
        for mut op in block.operations_mut() {
            for region in op.regions() {
                for inner in region.blocks() {
                    self.stamp_block(inner);
                }
            }

            if let Some(result) = op.try_first_result() {
                let term = self.term_of(result);
                if let Term::Var(_) = term
                    && let Some(ty) = self.resolve(term)
                {
                    op.set_attribute("ty", TypeAttribute::new(ty).into());
                }

                self.stamp_type_args(&mut op, yuzu_mlir::value_id(result));
            }
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
        yuzu_mlir::diagnostics::emit_error(op.location(), &message);
    }
}

fn last_region_op<'c, 'a>(op: OperationRef<'c, 'a>) -> Option<OperationRef<'c, 'a>> {
    op.regions().next()?.first_block()?.last_operation()
}

fn parse_signature<'c>(
    function: &yuzu_mlir::ops::yzl::FnOperationRef<'c, '_>,
) -> Option<Signature<'c>> {
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

    use crate::test_support;
    use crate::{infer_types, resolve_names};

    fn check(source: &str, expected: Expect) {
        test_support::check(
            source,
            |context, module| {
                resolve_names(context, module, &yuzu_types::Builtins);
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

fn f(x: int64) -> int64 { return x * 3 }

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
                %4 = yzl.name "x" : !yzl.var {param = 0 : i64, ty = !yz.int64}
                %5 = yz.constant_int 3
                %6 = yz.mul %4, %5 : !yzl.var, !yz.int64 -> !yzl.var {ty = !yz.int64}
                yzl.return %6 : !yzl.var
              }
              %0 = yzl.from @t
              %1 = yzl.where %0 {
                %4 = yzl.name "a" : !yzl.var {col = 0 : i64, ty = !yz.int64}
                %5 = yz.constant_int 10
                %6 = yz.cmp "gt", %4, %5 : !yzl.var, !yz.int64 -> !yzl.var {ty = !yz.bool}
                yzl.yield %6 : !yzl.var
              }
              %2 = yzl.extend %1 as ["e"] {
                %4 = yzl.name "a" : !yzl.var {col = 0 : i64, ty = !yz.int64}
                %5 = yzl.call @f(%4) : (!yzl.var) -> !yzl.var {callee_kind = "fn", ty = !yz.int64}
                %6 = yzl.name "b" : !yzl.var {col = 1 : i64, ty = !yz.int64}
                %7 = yz.add %5, %6 : !yzl.var, !yzl.var -> !yzl.var {ty = !yz.int64}
                yzl.yield %7 : !yzl.var
              }
              %3 = yzl.aggregate %2 group_by ["b"] as ["s", "r"] {
                %4 = yzl.name "e" : !yzl.var {col = 3 : i64, ty = !yz.int64}
                %5 = yzl.call @sum(%4) : (!yzl.var) -> !yzl.var {callee_kind = "builtin", ty = !yz.int64}
                %6 = yzl.name "rating" : !yzl.var {col = 2 : i64, ty = !yz.float64}
                %7 = yzl.call @avg(%6) : (!yzl.var) -> !yzl.var {callee_kind = "builtin", ty = !yz.float64}
                yzl.yield %5, %7 : !yzl.var, !yzl.var
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
    fn zero(x: Self) -> Self
}

impl Numeric for int64 {
    fn zero(x: int64) -> int64 { return 0 }
}

impl Numeric for float64 {
    fn zero(x: float64) -> float64 { return 0.0 }
}

fn id[T](x: T) -> T where T: Numeric { return x }

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
                      %2 = yz.constant_int 0
                      yzl.return %2 : !yz.int64
                    }
                  }
                  yzl.impl @Numeric for @float64 {
                    yzl.fn @zero generics ["Self"] params ["x"] (!yz.float64) -> !yz.float64 {
                      %2 = yz.constant_float 0.000000e+00
                      yzl.return %2 : !yz.float64
                    }
                  }
                  yzl.fn @id generics ["T"] where ["T"] : [@Numeric] params ["x"] (!yzl.param<"T">) -> !yzl.param<"T"> {
                    %2 = yzl.name "x" : !yzl.var {param = 0 : i64, ty = !yzl.param<"T">}
                    yzl.return %2 : !yzl.var
                  }
                  yzl.struct @Row ["a", "r"] : [!yz.int64, !yz.float64]
                  yzl.table @t of @Row
                  %0 = yzl.from @t
                  %1 = yzl.extend %0 as ["m", "n"] {
                    %2 = yzl.name "a" : !yzl.var {col = 0 : i64, ty = !yz.int64}
                    %3 = yzl.call @id(%2) : (!yzl.var) -> !yzl.var {callee_kind = "fn", ty = !yz.int64, type_args = [!yz.int64]}
                    %4 = yzl.name "r" : !yzl.var {col = 1 : i64, ty = !yz.float64}
                    %5 = yzl.call @id(%4) : (!yzl.var) -> !yzl.var {callee_kind = "fn", ty = !yz.float64, type_args = [!yz.float64]}
                    yzl.yield %3, %5 : !yzl.var, !yzl.var
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
    fn zero(x: Self) -> Self
}

impl Numeric for int64 {
    fn zero(x: int64) -> int64 { return 0 }
}

fn id[T](x: T) -> T where T: Numeric { return x }

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
            error: expected `!yz.str`, found `!yz.int64`
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

fn f(x: int64) -> bool { return x }

from t
|> extend f(a) as e
    "#,
            expect![[r#"
            error: expected `!yz.int64`, found `!yz.bool`
             --> test.yz:5:26
              |
            5 | fn f(x: int64) -> bool { return x }
              |                          ^
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
            error: expected `!yz.int64`, found `!yz.str`
             --> test.yz:5:1
              |
            5 | from t
              | ^
            "#]],
        );
    }
}
