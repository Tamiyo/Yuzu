//! ResolveNames: binds every name in a yzl module — relations, columns,
//! parameters, callees — and records each answer as an attribute for the
//! conversions downstream. Column resolution needs only the *names* flowing
//! through the query, which the syntax fully determines; the types stay
//! unresolved for InferTypes.

use std::collections::HashMap;

use melior::ir::attribute::AttributeLike;
use melior::ir::attribute::{
    ArrayAttribute, FlatSymbolRefAttribute, IntegerAttribute, StringAttribute, TypeAttribute,
};

use melior::ir::operation::{OperationLike, OperationRef};
use melior::ir::{Attribute, BlockLike, BlockRef, Module, RegionLike, Value, ValueLike};
use yuzu_diagnostics::diagnostics::builder::DiagnosticBuilder;
use yuzu_diagnostics::diagnostics::engine::DiagnosticsEngine;
use yuzu_types::Registry;

use yuzu_mlir::DiagnosticsBridge;

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
    registry: &'a dyn Registry,
    source: &'a DiagnosticsBridge,
    diagnostics: &'e mut DiagnosticsEngine,
    structs: HashMap<String, Schema>,
    relations: HashMap<String, Schema>,
    callables: HashMap<String, Callable>,
    schemas: HashMap<usize, Schema>,
    types: yuzu_mlir::Types<'c>,
}

pub fn resolve_names(
    module: &Module,
    registry: &dyn Registry,
    source: &DiagnosticsBridge,
    diagnostics: &mut DiagnosticsEngine,
) {
    let context = module.context();
    let mut resolver = Resolver {
        registry,
        source,
        diagnostics,
        structs: HashMap::new(),
        relations: HashMap::new(),
        callables: HashMap::new(),
        schemas: HashMap::new(),
        types: yuzu_mlir::Types::new(unsafe { context.to_ref() }),
    };
    resolver.declare(module.body());
    resolver.resolve_block(module.body(), &Ambient::None);
}

impl<'c, 'a, 'e> Resolver<'c, 'a, 'e> {
    /// First walk: every declaration registers before anything resolves, so
    /// order between declarations does not matter.
    fn declare(&mut self, block: BlockRef<'c, '_>) {
        let mut operation = block.first_operation();
        while let Some(op) = operation {
            match op.name().as_string_ref().as_str().unwrap_or_default() {
                "yzl.struct" => {
                    if let Some(name) = symbol_name(op) {
                        let schema = schema_columns(op);
                        self.declare_named(op, "struct", &name);
                        self.structs.insert(name, schema);
                    }
                }
                "yzl.table" => {
                    if let Some(name) = symbol_name(op) {
                        let row = symbol_text(op, "row");
                        let Some(schema) = self.structs.get(&row).cloned() else {
                            self.error(op, format!("unknown struct `{row}`"));
                            operation = op.next_in_block();
                            continue;
                        };
                        self.declare_named(op, "relation", &name);
                        self.relations.insert(name, schema);
                    }
                }
                "yzl.fn" => {
                    if let Some(name) = symbol_name(op) {
                        let params = string_array(op, "params").len();
                        let kind = if op.attribute("external").is_ok() {
                            "external"
                        } else if op.attribute("agg").is_ok() {
                            "agg_fn"
                        } else {
                            "fn"
                        };
                        self.declare_named(op, "function", &name);
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

    fn declare_named(&mut self, op: OperationRef<'c, '_>, what: &str, name: &str) {
        if self.structs.contains_key(name)
            || self.relations.contains_key(name)
            || self.callables.contains_key(name)
        {
            self.error(op, format!("the {what} `{name}` is already defined"));
        }
    }

    fn resolve_block(&mut self, block: BlockRef<'c, '_>, ambient: &Ambient) {
        let mut operation = block.first_operation();
        while let Some(op) = operation {
            self.resolve_op(op, ambient);
            operation = op.next_in_block();
        }
    }

    fn resolve_op(&mut self, op: OperationRef<'c, '_>, ambient: &Ambient) {
        match op.name().as_string_ref().as_str().unwrap_or_default() {
            "yzl.name" => self.resolve_name(op, ambient),
            "yzl.call" => self.resolve_call(op, ambient),
            "yzl.fn" => {
                let params = string_array(op, "params");
                self.resolve_regions(op, &Ambient::Params(&params));
            }
            "yzl.let" => {
                self.resolve_regions(op, &Ambient::None);
                if let Some(name) = symbol_name(op) {
                    let schema = self.yielded_schema(op);
                    self.declare_named(op, "binding", &name);
                    self.relations.insert(name, schema);
                }
            }
            "yzl.from" => {
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
            "yzl.alias" => {
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
            "yzl.where" | "yzl.distinct" | "yzl.limit" => {
                let schema = self.input_schema(op);
                self.resolve_regions(op, &Ambient::Columns(&schema));
                self.record_schema(op, schema);
            }
            "yzl.select" => {
                let input = self.input_schema(op);
                self.resolve_regions(op, &Ambient::Columns(&input));
                self.record_schema(op, unqualified(string_array(op, "names")));
            }
            "yzl.extend" => {
                let mut schema = self.input_schema(op);
                self.resolve_regions(op, &Ambient::Columns(&schema));
                schema.extend(unqualified(string_array(op, "names")));
                self.record_schema(op, schema);
            }
            "yzl.set" => {
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
            "yzl.drop" => {
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
            "yzl.rename" => {
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
            "yzl.aggregate" => {
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
            "yzl.join" => {
                let mut schema = self.input_schema(op);
                let relation = symbol_text(op, "rhs");
                let mut rhs = match self.relations.get(&relation) {
                    Some(schema) => schema.clone(),
                    None => {
                        self.error(op, format!("unknown relation `{relation}`"));
                        Schema::new()
                    }
                };
                if let Ok(alias) = op.attribute("rhs_alias") {
                    let alias = StringAttribute::try_from(alias)
                        .map(|alias| alias.value().to_string())
                        .unwrap_or_default();
                    for column in &mut rhs {
                        column.qualifier = Some(alias.clone());
                    }
                }
                if op.attribute("using_columns").is_ok() {
                    for name in string_array(op, "using_columns") {
                        for side in [&schema, &rhs] {
                            if !side.iter().any(|column| column.matches(&name)) {
                                self.error(op, format!("unknown column `{name}`"));
                            }
                        }
                    }
                }
                schema.extend(rhs);
                self.resolve_regions(op, &Ambient::Columns(&schema));
                self.record_schema(op, schema);
            }
            "yzl.output" | "yzl.yield" | "yzl.return" | "yzl.list" | "yzl.struct" | "yzl.table" => {
            }
            _ => self.resolve_regions(op, ambient),
        }
    }

    fn resolve_name(&mut self, op: OperationRef<'c, '_>, ambient: &Ambient) {
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

    fn resolve_call(&mut self, op: OperationRef<'c, '_>, ambient: &Ambient) {
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
        let attribute = StringAttribute::new(unsafe { op.context().to_ref() }, kind);
        set_attribute(op, "callee_kind", attribute.into());
    }

    fn find_column(
        &mut self,
        op: OperationRef<'c, '_>,
        schema: &Schema,
        reference: &str,
    ) -> Option<usize> {
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

    fn resolve_regions(&mut self, op: OperationRef<'c, '_>, ambient: &Ambient) {
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
    fn yielded_schema(&mut self, op: OperationRef<'c, '_>) -> Schema {
        op.regions()
            .next()
            .and_then(|region| region.first_block())
            .and_then(|block| {
                let mut last = block.first_operation()?;
                while let Some(next) = last.next_in_block() {
                    last = next;
                }
                let value = last.operand(0).ok()?;
                self.schemas.get(&key(value)).cloned()
            })
            .unwrap_or_default()
    }

    fn input_schema(&mut self, op: OperationRef<'c, '_>) -> Schema {
        op.operand(0)
            .ok()
            .and_then(|input| self.schemas.get(&key(input)).cloned())
            .unwrap_or_default()
    }

    fn record_schema(&mut self, op: OperationRef<'c, '_>, schema: Schema) {
        if let Ok(result) = op.result(0) {
            self.schemas.insert(key(result.into()), schema);
        }
    }

    fn set_index(&mut self, op: OperationRef<'c, '_>, name: &str, index: usize) {
        let attribute = IntegerAttribute::new(self.types.i64, index as i64);
        set_attribute(op, name, attribute.into());
    }

    fn set_index_array(&mut self, op: OperationRef<'c, '_>, name: &str, indices: &[usize]) {
        let elements: Vec<Attribute> = indices
            .iter()
            .map(|&index| IntegerAttribute::new(self.types.i64, index as i64).into())
            .collect();
        let attribute = ArrayAttribute::new(unsafe { op.context().to_ref() }, &elements);
        set_attribute(op, name, attribute.into());
    }

    fn error(&mut self, op: OperationRef<'c, '_>, message: String) {
        let span = self.source.span(op.location());
        self.diagnostics
            .emit(DiagnosticBuilder::error(span, message));
    }
}

fn key(value: Value) -> usize {
    value.to_raw().ptr as usize
}

fn set_attribute(op: OperationRef, name: &str, attribute: Attribute) {
    unsafe {
        mlir_sys::mlirOperationSetAttributeByName(
            op.to_raw(),
            melior::StringRef::new(name).to_raw(),
            attribute.to_raw(),
        );
    }
}

fn symbol_name(op: OperationRef) -> Option<String> {
    let attribute = op.attribute("sym_name").ok()?;
    StringAttribute::try_from(attribute)
        .ok()
        .map(|name| name.value().to_string())
}

/// The referenced name, whether the attribute is a symbol reference or a
/// plain string.
fn symbol_text(op: OperationRef, name: &str) -> String {
    let Ok(attribute) = op.attribute(name) else {
        return String::new();
    };
    FlatSymbolRefAttribute::try_from(attribute)
        .map(|symbol| symbol.value().to_string())
        .or_else(|_| StringAttribute::try_from(attribute).map(|string| string.value().to_string()))
        .unwrap_or_default()
}

fn attribute_text(op: OperationRef, name: &str) -> String {
    op.attribute(name)
        .ok()
        .and_then(|attribute| StringAttribute::try_from(attribute).ok())
        .map(|string| string.value().to_string())
        .unwrap_or_default()
}

fn string_array(op: OperationRef, name: &str) -> Vec<String> {
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
fn schema_columns(op: OperationRef) -> Schema {
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
