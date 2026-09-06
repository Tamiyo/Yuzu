//! ResolveNames: binds every name in a yzl module — relations, columns,
//! parameters, callees — and records each answer as an attribute for the
//! conversions downstream. Column resolution needs only the *names* flowing
//! through the query, which the syntax fully determines; the types stay
//! unresolved for InferTypes.

use std::collections::HashMap;

use melior::Context;
use melior::ir::attribute::{
    ArrayAttribute, FlatSymbolRefAttribute, IntegerAttribute, StringAttribute, TypeAttribute,
};
use melior::ir::operation::{OperationLike, OperationMutLike, OperationRefMut};
use melior::ir::{Attribute, BlockLike, BlockRef, Module, RegionLike};
use yuzu_diagnostics::diagnostics::builder::DiagnosticBuilder;
use yuzu_diagnostics::diagnostics::engine::DiagnosticsEngine;
use yuzu_mlir::ops::Yzl;
use yuzu_mlir::{DiagnosticsBridge, value_id};
use yuzu_types::Registry;

/// A column the query carries at some stage: its name, and the alias
/// qualifying it when an `alias` stage or a join has named its side.
#[derive(Clone, PartialEq)]
struct Column {
    qualifier: Option<String>,
    name: String,
}

impl Column {
    fn matches(&self, reference: &str) -> bool {
        match reference.split_once('.') {
            Some((qualifier, name)) => {
                self.qualifier.as_deref() == Some(qualifier) && self.name == name
            }
            None => self.name == reference,
        }
    }
}

type Schema = Vec<Column>;

/// What a callable name resolved to, recorded as the op's `callee_kind`.
struct Callable {
    kind: &'static str,
    min_args: usize,
    max_args: usize,
}

/// The columns a region's names may refer to.
enum Ambient<'a> {
    Columns(&'a Schema),
    Params(&'a [String]),
    None,
}

struct Resolver<'c, 'a, 'e> {
    context: &'c Context,
    registry: &'a dyn Registry,
    source: &'a DiagnosticsBridge,
    diagnostics: &'e mut DiagnosticsEngine,
    structs: HashMap<String, Schema>,
    relations: HashMap<String, Schema>,
    callables: HashMap<String, Callable>,
    schemas: HashMap<usize, Schema>,
    types: yuzu_mlir::Types<'c>,
}

pub fn resolve_names<'c>(
    context: &'c Context,
    module: &Module<'c>,
    registry: &dyn Registry,
    source: &DiagnosticsBridge,
    diagnostics: &mut DiagnosticsEngine,
) {
    let mut resolver = Resolver {
        context,
        registry,
        source,
        diagnostics,
        structs: HashMap::new(),
        relations: HashMap::new(),
        callables: HashMap::new(),
        schemas: HashMap::new(),
        types: yuzu_mlir::Types::new(context),
    };
    resolver.declare(module.body());
    resolver.resolve_block(module.body(), &Ambient::None);
}

impl<'c> Resolver<'c, '_, '_> {
    /// First walk: every declaration registers before anything resolves, so
    /// order between declarations does not matter.
    fn declare(&mut self, block: BlockRef<'c, '_>) {
        let mut operation = block.first_operation();
        while let Some(op) = operation {
            match Yzl::of(&op) {
                Some(Yzl::Struct) => {
                    if let Some(name) = symbol_name(&op) {
                        let schema = schema_columns(&op);
                        self.declare_named(&op, "struct", &name);
                        self.structs.insert(name, schema);
                    }
                }
                Some(Yzl::Table) => {
                    if let Some(name) = symbol_name(&op) {
                        let row = symbol_text(&op, "row");
                        if let Some(schema) = self.structs.get(&row).cloned() {
                            self.declare_named(&op, "relation", &name);
                            self.relations.insert(name, schema);
                        } else {
                            self.error(&op, format!("unknown struct `{row}`"));
                        }
                    }
                }
                Some(Yzl::Fn) => {
                    if let Some(name) = symbol_name(&op) {
                        let params = string_array(&op, "params").len();
                        let kind = if op.attribute("external").is_ok() {
                            "external"
                        } else if op.attribute("agg").is_ok() {
                            "agg_fn"
                        } else {
                            "fn"
                        };
                        self.declare_named(&op, "function", &name);
                        self.callables.insert(
                            name,
                            Callable {
                                kind,
                                min_args: params,
                                max_args: params,
                            },
                        );
                    }
                }
                _ => {}
            }
            operation = op.next_in_block();
        }
    }

    fn declare_named<'a>(&mut self, op: &impl OperationLike<'c, 'a>, what: &str, name: &str)
    where
        'c: 'a,
    {
        if self.structs.contains_key(name)
            || self.relations.contains_key(name)
            || self.callables.contains_key(name)
        {
            self.error(op, format!("the {what} `{name}` is already defined"));
        }
    }

    fn resolve_block(&mut self, block: BlockRef<'c, '_>, ambient: &Ambient) {
        let mut operation = block.first_operation_mut();
        while let Some(mut op) = operation {
            self.resolve_op(&mut op, ambient);
            operation = op.next_in_block_mut();
        }
    }

    fn resolve_op(&mut self, op: &mut OperationRefMut<'c, '_>, ambient: &Ambient) {
        match Yzl::of(op) {
            Some(Yzl::Name) => self.resolve_name(op, ambient),
            Some(Yzl::Call) => self.resolve_call(op, ambient),
            Some(Yzl::Fn) => {
                let params = string_array(op, "params");
                self.resolve_regions(op, &Ambient::Params(&params));
            }
            Some(Yzl::Let) => {
                self.resolve_regions(op, &Ambient::None);
                if let Some(name) = symbol_name(op) {
                    let schema = self.yielded_schema(op);
                    self.declare_named(op, "binding", &name);
                    self.relations.insert(name, schema);
                }
            }
            Some(Yzl::From) => {
                let source = symbol_text(op, "source");
                let schema = match self.relations.get(&source) {
                    Some(schema) => schema.clone(),
                    None => {
                        self.error(op, format!("unknown relation `{source}`"));
                        Schema::new()
                    }
                };
                self.record_schema(op, schema);
            }
            Some(Yzl::Alias) => {
                let alias = attribute_text(op, "alias");
                let schema = self
                    .input_schema(op)
                    .into_iter()
                    .map(|column| Column {
                        qualifier: Some(alias.clone()),
                        ..column
                    })
                    .collect();
                self.record_schema(op, schema);
            }
            Some(Yzl::Where | Yzl::Distinct | Yzl::Limit) => {
                let schema = self.input_schema(op);
                self.resolve_regions(op, &Ambient::Columns(&schema));
                self.record_schema(op, schema);
            }
            Some(Yzl::Select) => {
                let input = self.input_schema(op);
                self.resolve_regions(op, &Ambient::Columns(&input));
                let schema = unqualified(string_array(op, "names"));
                self.record_schema(op, schema);
            }
            Some(Yzl::Extend) => {
                let mut schema = self.input_schema(op);
                self.resolve_regions(op, &Ambient::Columns(&schema));
                schema.extend(unqualified(string_array(op, "names")));
                self.record_schema(op, schema);
            }
            Some(Yzl::Set) => {
                let schema = self.input_schema(op);
                let mut columns = Vec::new();
                for name in string_array(op, "names") {
                    match schema.iter().position(|column| column.matches(&name)) {
                        Some(index) => columns.push(index),
                        None => self.error(op, format!("unknown column `{name}`")),
                    }
                }
                self.set_index_array(op, "set_cols", &columns);
                self.resolve_regions(op, &Ambient::Columns(&schema));
                self.record_schema(op, schema);
            }
            Some(Yzl::Drop) => {
                let mut schema = self.input_schema(op);
                for name in string_array(op, "columns") {
                    match schema.iter().position(|column| column.matches(&name)) {
                        Some(index) => {
                            schema.remove(index);
                        }
                        None => self.error(op, format!("unknown column `{name}`")),
                    }
                }
                self.record_schema(op, schema);
            }
            Some(Yzl::Rename) => {
                let mut schema = self.input_schema(op);
                let to = string_array(op, "to");
                for (from, to) in string_array(op, "from").iter().zip(to) {
                    match schema.iter().position(|column| column.matches(from)) {
                        Some(index) => schema[index].name = to,
                        None => self.error(op, format!("unknown column `{from}`")),
                    }
                }
                self.record_schema(op, schema);
            }
            Some(Yzl::Aggregate) => {
                let input = self.input_schema(op);
                let mut keys = Vec::new();
                let mut schema = Schema::new();
                for name in string_array(op, "group_by") {
                    match self.find_column(op, &input, &name) {
                        Some(index) => {
                            keys.push(index);
                            schema.push(input[index].clone());
                        }
                        None => schema.push(Column {
                            qualifier: None,
                            name,
                        }),
                    }
                }
                self.set_index_array(op, "key_cols", &keys);
                schema.extend(unqualified(string_array(op, "names")));
                self.resolve_regions(op, &Ambient::Columns(&input));
                self.record_schema(op, schema);
            }
            Some(Yzl::Join) => {
                let mut schema = self.input_schema(op);
                let relation = symbol_text(op, "rhs");
                let mut rhs = match self.relations.get(&relation) {
                    Some(schema) => schema.clone(),
                    None => {
                        self.error(op, format!("unknown relation `{relation}`"));
                        Schema::new()
                    }
                };
                let alias = attribute_text(op, "rhs_alias");
                if !alias.is_empty() {
                    for column in &mut rhs {
                        column.qualifier = Some(alias.clone());
                    }
                }
                for name in string_array(op, "using_columns") {
                    for side in [&schema, &rhs] {
                        if !side.iter().any(|column| column.matches(&name)) {
                            self.error(op, format!("unknown column `{name}`"));
                        }
                    }
                }
                schema.extend(rhs);
                self.resolve_regions(op, &Ambient::Columns(&schema));
                self.record_schema(op, schema);
            }
            Some(Yzl::Output | Yzl::Yield | Yzl::Return | Yzl::List | Yzl::Struct | Yzl::Table) => {
            }
            None => self.resolve_regions(op, ambient),
        }
    }

    fn resolve_name(&mut self, op: &mut OperationRefMut<'c, '_>, ambient: &Ambient) {
        let reference = attribute_text(op, "name");
        match ambient {
            Ambient::Columns(schema) => {
                if let Some(index) = self.find_column(op, schema, &reference) {
                    self.set_index(op, "col", index);
                }
            }
            Ambient::Params(params) => match params.iter().position(|param| param == &reference) {
                Some(index) => self.set_index(op, "param", index),
                None => self.error(op, format!("unknown name `{reference}`")),
            },
            Ambient::None => self.error(op, format!("unknown name `{reference}`")),
        }
    }

    fn resolve_call(&mut self, op: &mut OperationRefMut<'c, '_>, ambient: &Ambient) {
        self.resolve_regions(op, ambient);
        let callee = symbol_text(op, "callee");
        let arguments = op.operand_count();
        let (kind, min, max) = match self.callables.get(&callee) {
            Some(callable) => (callable.kind, callable.min_args, callable.max_args),
            None => match self
                .registry
                .entries()
                .iter()
                .find(|entry| entry.name == callee)
            {
                Some(entry) => ("builtin", entry.min_args, entry.max_args),
                None => {
                    self.error(op, format!("unknown function `{callee}`"));
                    return;
                }
            },
        };
        if arguments < min || arguments > max {
            let expected = if min == max {
                format!("{min}")
            } else {
                format!("{min} to {max}")
            };
            self.error(
                op,
                format!("`{callee}` expects {expected} arguments, got {arguments}"),
            );
        }
        op.set_attribute(
            "callee_kind",
            StringAttribute::new(self.context, kind).into(),
        );
    }

    fn find_column<'a>(
        &mut self,
        op: &impl OperationLike<'c, 'a>,
        schema: &Schema,
        reference: &str,
    ) -> Option<usize>
    where
        'c: 'a,
    {
        let mut matches = schema
            .iter()
            .enumerate()
            .filter(|(_, column)| column.matches(reference));
        match (matches.next(), matches.next()) {
            (Some((index, _)), None) => Some(index),
            (Some(_), Some(_)) => {
                self.error(op, format!("`{reference}` is ambiguous"));
                None
            }
            (None, _) => {
                self.error(op, format!("unknown column `{reference}`"));
                None
            }
        }
    }

    fn resolve_regions<'a>(&mut self, op: &impl OperationLike<'c, 'a>, ambient: &Ambient)
    where
        'c: 'a,
    {
        for region in op.regions() {
            let mut block = region.first_block();
            while let Some(current) = block {
                self.resolve_block(current, ambient);
                block = current.next_in_region();
            }
        }
    }

    /// The schema flowing out of the op a region's yield hands back — the
    /// shape a `let`-bound query exposes to `from`.
    fn yielded_schema<'a>(&mut self, op: &impl OperationLike<'c, 'a>) -> Schema
    where
        'c: 'a,
    {
        op.regions()
            .next()
            .and_then(|region| region.first_block())
            .and_then(|block| {
                let mut last = block.first_operation()?;
                while let Some(next) = last.next_in_block() {
                    last = next;
                }
                let value = last.operand(0).ok()?;
                self.schemas.get(&value_id(value)).cloned()
            })
            .unwrap_or_default()
    }

    fn input_schema<'a>(&mut self, op: &impl OperationLike<'c, 'a>) -> Schema
    where
        'c: 'a,
    {
        op.operand(0)
            .ok()
            .and_then(|input| self.schemas.get(&value_id(input)).cloned())
            .unwrap_or_default()
    }

    fn record_schema<'a>(&mut self, op: &impl OperationLike<'c, 'a>, schema: Schema)
    where
        'c: 'a,
    {
        if let Ok(result) = op.result(0) {
            self.schemas.insert(value_id(result.into()), schema);
        }
    }

    fn set_index(&mut self, op: &mut OperationRefMut<'c, '_>, name: &str, index: usize) {
        op.set_attribute(
            name,
            IntegerAttribute::new(self.types.i64, index as i64).into(),
        );
    }

    fn set_index_array(&mut self, op: &mut OperationRefMut<'c, '_>, name: &str, indices: &[usize]) {
        let elements: Vec<Attribute> = indices
            .iter()
            .map(|&index| IntegerAttribute::new(self.types.i64, index as i64).into())
            .collect();
        op.set_attribute(name, ArrayAttribute::new(self.context, &elements).into());
    }

    fn error<'a>(&mut self, op: &impl OperationLike<'c, 'a>, message: String)
    where
        'c: 'a,
    {
        let span = self.source.span(op.location());
        self.diagnostics
            .emit(DiagnosticBuilder::error(span, message));
    }
}

fn symbol_name<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> Option<String> {
    let attribute = op.attribute("sym_name").ok()?;
    StringAttribute::try_from(attribute)
        .ok()
        .map(|name| name.value().to_string())
}

/// The referenced name, whether the attribute is a symbol reference or a
/// plain string.
fn symbol_text<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>, name: &str) -> String {
    let Ok(attribute) = op.attribute(name) else {
        return String::new();
    };
    FlatSymbolRefAttribute::try_from(attribute)
        .map(|symbol| symbol.value().to_string())
        .or_else(|_| StringAttribute::try_from(attribute).map(|string| string.value().to_string()))
        .unwrap_or_default()
}

fn attribute_text<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>, name: &str) -> String {
    op.attribute(name)
        .ok()
        .and_then(|attribute| StringAttribute::try_from(attribute).ok())
        .map(|string| string.value().to_string())
        .unwrap_or_default()
}

fn string_array<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>, name: &str) -> Vec<String> {
    let Ok(attribute) = op.attribute(name) else {
        return Vec::new();
    };
    yuzu_mlir::array_elements(attribute)
        .into_iter()
        .filter_map(|element| StringAttribute::try_from(element).ok())
        .map(|string| string.value().to_string())
        .collect()
}

/// The column names of the struct's `!yzr.rel` schema attribute.
fn schema_columns<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> Schema {
    let Ok(attribute) = op.attribute("schema") else {
        return Schema::new();
    };
    let Ok(schema) = TypeAttribute::try_from(attribute) else {
        return Schema::new();
    };
    let Some(rel) = yuzu_mlir::RelType::from_type(schema.value()) else {
        return Schema::new();
    };
    (0..rel.column_count())
        .map(|index| Column {
            qualifier: None,
            name: rel.column_name(index).to_string(),
        })
        .collect()
}

fn unqualified(names: Vec<String>) -> Schema {
    names
        .into_iter()
        .map(|name| Column {
            qualifier: None,
            name,
        })
        .collect()
}
