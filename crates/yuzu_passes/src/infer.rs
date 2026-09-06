//! InferTypes: unification over MLIR values. Every `!yzl.var`-typed value is
//! a variable; concrete types anchor classes; op semantics, function
//! signatures, and the schema types flowing through the stages supply the
//! equations. Answers are stamped as `{ty = …}` attributes — LowerYZL applies
//! them during its rebuild, so inference itself rewrites nothing.

use std::collections::HashMap;

use melior::ir::attribute::{
    AttributeLike, FlatSymbolRefAttribute, IntegerAttribute, StringAttribute, TypeAttribute,
};
use melior::ir::operation::{OperationLike, OperationRef};
use melior::ir::r#type::FunctionType;
use melior::ir::{BlockLike, BlockRef, Module, RegionLike, Type, Value, ValueLike};
use yuzu_diagnostics::diagnostics::builder::DiagnosticBuilder;
use yuzu_diagnostics::diagnostics::engine::DiagnosticsEngine;
use yuzu_mlir::DiagnosticsBridge;

/// A type either known or still being solved for: a concrete MLIR type, or a
/// union-find class shared by every value that must agree.
#[derive(Clone, Copy)]
enum Term<'c> {
    Concrete(Type<'c>),
    Class(usize),
}

type Row<'c> = Vec<Term<'c>>;

struct Inferrer<'c, 'a, 'e> {
    source: &'a DiagnosticsBridge,
    diagnostics: &'e mut DiagnosticsEngine,
    /// Union-find parents; a root's entry in `resolved` is its answer.
    parents: Vec<usize>,
    resolved: HashMap<usize, Type<'c>>,
    /// The class of each var-typed value, keyed by the value's identity.
    classes: HashMap<usize, usize>,
    /// The row of column terms flowing out of each stage value.
    rows: HashMap<usize, Row<'c>>,
    signatures: HashMap<String, (Vec<Type<'c>>, Type<'c>)>,
    relations: HashMap<String, Row<'c>>,
    types: yuzu_mlir::Types<'c>,
}

pub fn infer_types(
    module: &Module,
    source: &DiagnosticsBridge,
    diagnostics: &mut DiagnosticsEngine,
) {
    let context = unsafe { module.context().to_ref() };
    let mut inferrer = Inferrer {
        source,
        diagnostics,
        parents: Vec::new(),
        resolved: HashMap::new(),
        classes: HashMap::new(),
        rows: HashMap::new(),
        signatures: HashMap::new(),
        relations: HashMap::new(),
        types: yuzu_mlir::Types::new(context),
    };
    inferrer.declare_signatures(module.body());
    inferrer.infer_block(module.body(), &Row::new(), &[]);
    inferrer.stamp_block(module.body());
}

impl<'c, 'a, 'e> Inferrer<'c, 'a, 'e> {
    // --- union-find ---

    fn term_of(&mut self, value: Value<'c, '_>) -> Term<'c> {
        let ty = value.r#type();
        if ty != self.types.var {
            return Term::Concrete(ty);
        }
        let key = value.to_raw().ptr as usize;
        if let Some(&class) = self.classes.get(&key) {
            return Term::Class(class);
        }
        let class = self.parents.len();
        self.parents.push(class);
        self.classes.insert(key, class);
        Term::Class(class)
    }

    fn root(&mut self, class: usize) -> usize {
        let parent = self.parents[class];
        if parent == class {
            return class;
        }
        let root = self.root(parent);
        self.parents[class] = root;
        root
    }

    fn unify(&mut self, op: OperationRef<'c, '_>, a: Term<'c>, b: Term<'c>) {
        match (a, b) {
            (Term::Concrete(a), Term::Concrete(b)) => {
                if a != b {
                    self.error(op, format!("expected `{a}`, found `{b}`"));
                }
            }
            (Term::Class(class), Term::Concrete(ty)) | (Term::Concrete(ty), Term::Class(class)) => {
                let root = self.root(class);
                match self.resolved.get(&root) {
                    Some(&resolved) if resolved != ty => {
                        self.error(op, format!("expected `{resolved}`, found `{ty}`"));
                    }
                    _ => {
                        self.resolved.insert(root, ty);
                    }
                }
            }
            (Term::Class(a), Term::Class(b)) => {
                let a = self.root(a);
                let b = self.root(b);
                if a == b {
                    return;
                }
                let merged = match (self.resolved.get(&a), self.resolved.get(&b)) {
                    (Some(&left), Some(&right)) if left != right => {
                        self.error(op, format!("expected `{left}`, found `{right}`"));
                        Some(left)
                    }
                    (Some(&ty), _) | (_, Some(&ty)) => Some(ty),
                    (None, None) => None,
                };
                self.parents[b] = a;
                if let Some(ty) = merged {
                    self.resolved.insert(a, ty);
                }
            }
        }
    }

    fn resolve(&mut self, term: Term<'c>) -> Option<Type<'c>> {
        match term {
            Term::Concrete(ty) => Some(ty),
            Term::Class(class) => {
                let root = self.root(class);
                self.resolved.get(&root).copied()
            }
        }
    }

    // --- the walk ---

    fn declare_signatures(&mut self, block: BlockRef<'c, '_>) {
        let mut operation = block.first_operation();
        while let Some(op) = operation {
            match op_name(op).as_str() {
                "yzl.fn" => {
                    if let Some(name) = text_attribute(op, "sym_name")
                        && let Ok(attribute) = op.attribute("signature")
                        && let Some(signature) = parse_signature(attribute)
                    {
                        self.signatures.insert(name, signature);
                    }
                }
                "yzl.struct" => {
                    if let Some(name) = text_attribute(op, "sym_name")
                        && let Ok(schema) = op.attribute("schema")
                    {
                        let row = parse_schema_row(schema);
                        self.relations.insert(name, row);
                    }
                }
                "yzl.table" => {
                    if let Some(name) = text_attribute(op, "sym_name") {
                        let row = self
                            .relations
                            .get(&symbol_attribute(op, "row"))
                            .cloned()
                            .unwrap_or_default();
                        self.relations.insert(name, row);
                    }
                }
                _ => {}
            }
            operation = op.next_in_block();
        }
    }

    fn infer_block(&mut self, block: BlockRef<'c, '_>, columns: &Row<'c>, params: &[Type<'c>]) {
        let mut operation = block.first_operation();
        while let Some(op) = operation {
            self.infer_op(op, columns, params);
            operation = op.next_in_block();
        }
    }

    fn infer_op(&mut self, op: OperationRef<'c, '_>, columns: &Row<'c>, params: &[Type<'c>]) {
        match op_name(op).as_str() {
            "yzl.name" => {
                let term = self.term_of(result(op));
                if let Some(index) = index_attribute(op, "col") {
                    if let Some(&column) = columns.get(index) {
                        self.unify(op, term, column);
                    }
                } else if let Some(index) = index_attribute(op, "param")
                    && let Some(&param) = params.get(index)
                {
                    self.unify(op, term, Term::Concrete(param));
                }
            }
            "yzl.call" => {
                let callee = symbol_attribute(op, "callee");
                match text_attribute(op, "callee_kind").as_deref() {
                    Some("builtin") => self.infer_builtin(op, &callee),
                    Some("external") => {}
                    _ => {
                        let Some((parameters, ret)) = self.signatures.get(&callee).cloned() else {
                            return;
                        };
                        for (index, parameter) in parameters.iter().enumerate() {
                            if let Ok(argument) = op.operand(index) {
                                let term = self.term_of(argument);
                                self.unify(op, term, Term::Concrete(*parameter));
                            }
                        }
                        let term = self.term_of(result(op));
                        self.unify(op, term, Term::Concrete(ret));
                    }
                }
            }
            "yzl.fn" => {
                let Some((parameters, ret)) =
                    text_attribute(op, "sym_name").and_then(|name| self.signatures.get(&name))
                else {
                    return;
                };
                let (parameters, ret) = (parameters.clone(), *ret);
                self.infer_regions(op, &Row::new(), &parameters);
                self.unify_returns(op, ret);
            }
            "yzl.from" => {
                let row = self
                    .relations
                    .get(&symbol_attribute(op, "source"))
                    .cloned()
                    .unwrap_or_default();
                self.record_row(op, row);
            }
            "yzl.let" => {
                self.infer_regions(op, &Row::new(), &[]);
                if let Some(name) = text_attribute(op, "sym_name") {
                    let row = self.yield_terms(op);
                    self.relations.insert(name, row);
                }
            }
            "yzl.where" | "yzl.distinct" | "yzl.limit" | "yzl.alias" | "yzl.rename"
            | "yzl.drop" => {
                let row = self.input_row(op);
                self.infer_regions(op, &row, &[]);
                if op_name(op) == "yzl.where" {
                    self.unify_yield(op, Term::Concrete(self.types.boolean));
                }
                self.record_row(op, row);
            }
            "yzl.set" => {
                let row = self.input_row(op);
                self.infer_regions(op, &row, &[]);
                let yields = self.yield_terms(op);
                for (index, term) in index_array_attribute(op, "set_cols")
                    .into_iter()
                    .zip(yields)
                {
                    if let Some(&column) = row.get(index) {
                        self.unify(op, column, term);
                    }
                }
                self.record_row(op, row);
            }
            "yzl.select" => {
                let input = self.input_row(op);
                self.infer_regions(op, &input, &[]);
                let row = self.yield_terms(op);
                self.record_row(op, row);
            }
            "yzl.extend" => {
                let mut row = self.input_row(op);
                self.infer_regions(op, &row, &[]);
                row.extend(self.yield_terms(op));
                self.record_row(op, row);
            }
            "yzl.aggregate" => {
                let input = self.input_row(op);
                self.infer_regions(op, &input, &[]);
                let mut row: Row = index_array_attribute(op, "key_cols")
                    .into_iter()
                    .filter_map(|index| input.get(index).copied())
                    .collect();
                row.extend(self.yield_terms(op));
                self.record_row(op, row);
            }
            "yzl.join" => {
                let row = self.input_row(op);
                self.infer_regions(op, &row, &[]);
                self.record_row(op, row);
            }
            "yz.add" | "yz.sub" | "yz.mul" | "yz.div" | "yz.rem" => {
                let (lhs, rhs) = (self.operand_term(op, 0), self.operand_term(op, 1));
                let out = self.term_of(result(op));
                self.unify(op, lhs, rhs);
                self.unify(op, lhs, out);
            }
            "yz.neg" => {
                let value = self.operand_term(op, 0);
                let out = self.term_of(result(op));
                self.unify(op, value, out);
            }
            "yz.cmp" => {
                let (lhs, rhs) = (self.operand_term(op, 0), self.operand_term(op, 1));
                let out = self.term_of(result(op));
                self.unify(op, lhs, rhs);
                self.unify(op, out, Term::Concrete(self.types.boolean));
            }
            "yz.and" | "yz.or" => {
                let (lhs, rhs) = (self.operand_term(op, 0), self.operand_term(op, 1));
                let out = self.term_of(result(op));
                self.unify(op, lhs, Term::Concrete(self.types.boolean));
                self.unify(op, rhs, Term::Concrete(self.types.boolean));
                self.unify(op, out, Term::Concrete(self.types.boolean));
            }
            "yz.not" => {
                let value = self.operand_term(op, 0);
                let out = self.term_of(result(op));
                self.unify(op, value, Term::Concrete(self.types.boolean));
                self.unify(op, out, Term::Concrete(self.types.boolean));
            }
            _ => self.infer_regions(op, columns, params),
        }
    }

    fn operand_term(&mut self, op: OperationRef<'c, '_>, index: usize) -> Term<'c> {
        match op.operand(index) {
            Ok(value) => self.term_of(value),
            Err(_) => Term::Concrete(self.types.var),
        }
    }

    /// The aggregate builtins are polymorphic; these are the old
    /// `resolve_agg_ty` rules over terms.
    fn infer_builtin(&mut self, op: OperationRef<'c, '_>, callee: &str) {
        let out = self.term_of(result(op));
        match callee {
            "count" | "count_distinct" => {
                self.unify(op, out, Term::Concrete(self.types.int64));
            }
            "sum" => {
                if let Ok(argument) = op.operand(0) {
                    let term = self.term_of(argument);
                    if let Some(ty) = self.resolve(term) {
                        let result = if ty == self.types.float64 {
                            self.types.float64
                        } else {
                            self.types.int64
                        };
                        self.unify(op, out, Term::Concrete(result));
                    }
                }
            }
            "in" => {
                self.unify(op, out, Term::Concrete(self.types.boolean));
            }
            "min" | "max" | "avg" => {
                if let Ok(argument) = op.operand(0) {
                    let term = self.term_of(argument);
                    self.unify(op, out, term);
                }
            }
            _ => {}
        }
    }

    fn infer_regions(&mut self, op: OperationRef<'c, '_>, columns: &Row<'c>, params: &[Type<'c>]) {
        for region in op.regions() {
            let mut block = region.first_block();
            while let Some(current) = block {
                self.infer_block(current, columns, params);
                block = current.next_in_region();
            }
        }
    }

    // --- rows and yields ---

    fn input_row(&mut self, op: OperationRef<'c, '_>) -> Row<'c> {
        op.operand(0)
            .ok()
            .and_then(|input| self.rows.get(&key(input)).cloned())
            .unwrap_or_default()
    }

    fn record_row(&mut self, op: OperationRef<'c, '_>, row: Row<'c>) {
        if let Ok(result) = op.result(0) {
            self.rows.insert(key(result.into()), row);
        }
    }

    fn yield_terms(&mut self, op: OperationRef<'c, '_>) -> Row<'c> {
        let Some(terminator) = last_region_op(op) else {
            return Row::new();
        };
        (0..terminator.operand_count())
            .filter_map(|index| terminator.operand(index).ok())
            .map(|value| self.term_of(value))
            .collect()
    }

    fn unify_yield(&mut self, op: OperationRef<'c, '_>, expected: Term<'c>) {
        if let Some(terminator) = last_region_op(op)
            && let Ok(value) = terminator.operand(0)
        {
            let term = self.term_of(value);
            self.unify(op, term, expected);
        }
    }

    fn unify_returns(&mut self, op: OperationRef<'c, '_>, ret: Type<'c>) {
        if let Some(terminator) = last_region_op(op)
            && op_name(terminator) == "yzl.return"
            && let Ok(value) = terminator.operand(0)
        {
            let term = self.term_of(value);
            self.unify(terminator, term, Term::Concrete(ret));
        }
    }

    // --- stamping ---

    fn stamp_block(&mut self, block: BlockRef<'c, '_>) {
        let mut operation = block.first_operation();
        while let Some(op) = operation {
            for region in op.regions() {
                let mut inner = region.first_block();
                while let Some(current) = inner {
                    self.stamp_block(current);
                    inner = current.next_in_region();
                }
            }
            if let Ok(result) = op.result(0) {
                let term = self.term_of(result.into());
                if let Term::Class(_) = term
                    && let Some(ty) = self.resolve(term)
                {
                    set_attribute(op, "ty", TypeAttribute::new(ty).into());
                }
            }
            operation = op.next_in_block();
        }
    }

    fn error(&mut self, op: OperationRef<'c, '_>, message: String) {
        let span = self.source.span(op.location());
        self.diagnostics
            .emit(DiagnosticBuilder::error(span, message));
    }
}

// --- op reading helpers ---

fn op_name(op: OperationRef) -> String {
    op.name()
        .as_string_ref()
        .as_str()
        .unwrap_or_default()
        .to_string()
}

fn result<'c, 'a>(op: OperationRef<'c, 'a>) -> Value<'c, 'a> {
    op.result(0).expect("the op has a result").into()
}

fn key(value: Value) -> usize {
    value.to_raw().ptr as usize
}

fn last_region_op<'c, 'a>(op: OperationRef<'c, 'a>) -> Option<OperationRef<'c, 'a>> {
    let block = op.regions().next()?.first_block()?;
    let mut last = block.first_operation()?;
    while let Some(next) = last.next_in_block() {
        last = next;
    }
    Some(last)
}

fn text_attribute(op: OperationRef, name: &str) -> Option<String> {
    let attribute = op.attribute(name).ok()?;
    StringAttribute::try_from(attribute)
        .ok()
        .map(|string| string.value().to_string())
}

fn symbol_attribute(op: OperationRef, name: &str) -> String {
    let Ok(attribute) = op.attribute(name) else {
        return String::new();
    };
    FlatSymbolRefAttribute::try_from(attribute)
        .map(|symbol| symbol.value().to_string())
        .or_else(|_| StringAttribute::try_from(attribute).map(|string| string.value().to_string()))
        .unwrap_or_default()
}

fn index_attribute(op: OperationRef, name: &str) -> Option<usize> {
    let attribute = op.attribute(name).ok()?;
    IntegerAttribute::try_from(attribute)
        .ok()
        .map(|index| index.value() as usize)
}

fn index_array_attribute(op: OperationRef, name: &str) -> Vec<usize> {
    let Ok(attribute) = op.attribute(name) else {
        return Vec::new();
    };
    yuzu_mlir::array_elements(attribute)
        .into_iter()
        .filter_map(|element| IntegerAttribute::try_from(element).ok())
        .map(|index| index.value() as usize)
        .collect()
}

fn parse_signature<'c>(attribute: melior::ir::Attribute<'c>) -> Option<(Vec<Type<'c>>, Type<'c>)> {
    let signature =
        FunctionType::try_from(TypeAttribute::try_from(attribute).ok()?.value()).ok()?;
    let params = (0..signature.input_count())
        .filter_map(|index| signature.input(index).ok())
        .collect();
    Some((params, signature.result(0).ok()?))
}

fn set_attribute(op: OperationRef, name: &str, attribute: melior::ir::Attribute) {
    unsafe {
        mlir_sys::mlirOperationSetAttributeByName(
            op.to_raw(),
            melior::StringRef::new(name).to_raw(),
            attribute.to_raw(),
        );
    }
}

/// The column types of the struct's `!yzr.rel` schema attribute.
fn parse_schema_row<'c>(attribute: melior::ir::Attribute<'c>) -> Row<'c> {
    let Ok(schema) = TypeAttribute::try_from(attribute) else {
        return Row::new();
    };
    let Some(rel) = yuzu_mlir::RelType::from_type(schema.value()) else {
        return Row::new();
    };
    (0..rel.column_count())
        .map(|index| Term::Concrete(rel.column_type(index)))
        .collect()
}
