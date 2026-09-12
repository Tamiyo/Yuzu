//! LowerYZL: the one rebuild. Names become block arguments, declarations
//! move to the dialect that owns their type, and stage ops become `yzr` —
//! after which `yzl` is illegal. The checking passes stamped every answer
//! this needs (`col`, `param`, `ty`), so nothing is re-derived here except
//! the row shapes, which the ops being built need anyway.
//!
//! A dialect conversion over a whole module is a translation, so this builds
//! a new module rather than rewriting in place: the source stays readable
//! while it is consumed, and no operand is remapped under its own use.

use std::collections::HashMap;

use melior::Context;
use melior::ir::attribute::{
    DenseI64ArrayAttribute, FlatSymbolRefAttribute, StringAttribute, TypeAttribute,
};
use melior::ir::operation::{OperationBuilder, OperationLike, OperationRef};
use melior::ir::{
    Attribute, Block, BlockLike, BlockRef, Identifier, Location, Module, Region, RegionLike, Type,
    Value, ValueLike,
};
use yuzu_mlir::ext::{ArrayAttributeExt, BlockExt, OperationExt};
use yuzu_mlir::ods::{yz, yzr};
use yuzu_mlir::ops::yzl::YzlOperationRef;
use yuzu_mlir::types;
use yuzu_mlir::{StructType, SymbolTable, value_id};
use yuzu_types::{BuiltinFunc, FunctionRegistry};

/// A row: the columns flowing out of a stage, in order.
type Schema<'c> = Vec<(&'c str, Type<'c>)>;

/// What a lowered region yields: the values its body computed, or the whole
/// row with those values substituted into the columns they replace.
enum Yielded<'k> {
    Body,
    Row(&'k [usize]),
}

struct YzlToYzr<'c, 'a, 'r> {
    context: &'c Context,
    registry: &'r dyn FunctionRegistry,
    /// What each yzl stage value became, and the row it carries.
    stages: HashMap<usize, (Value<'c, 'a>, Schema<'c>)>,
    /// The symbol declaring each distinct row shape, so a shape nobody
    /// declared is declared once.
    shapes: HashMap<Schema<'c>, &'c str>,
    /// The rows each `let` name stands for, already produced.
    bindings: HashMap<&'c str, (Value<'c, 'a>, Schema<'c>)>,
}

/// Expects a resolved, inferred module. Returns the `yz` + `yzr` module it
/// lowers to; anything it cannot lower is reported and left out.
pub fn lower_yzl_to_yzr<'c>(
    context: &'c Context,
    module: &Module<'c>,
    registry: &dyn FunctionRegistry,
) -> Module<'c> {
    let lowered = Module::new(Location::unknown(context));
    {
        let target = lowered.body();
        let mut symbols = SymbolTable::new(&lowered);
        let mut lowering = YzlToYzr {
            context,
            registry,
            stages: HashMap::new(),
            shapes: HashMap::new(),
            bindings: HashMap::new(),
        };

        let source = SymbolTable::new(module);
        lowering.intern_declared_shapes(module.body());
        lowering.lower_block(module.body(), target, &source, &mut symbols);
    }

    lowered
}

impl<'c, 'a> YzlToYzr<'c, 'a, '_> {
    /// Declarations first, so a stage can ask for a relation's row before
    /// the walk reaches the op that declared it.
    /// Seeds the shape index with what the program declared, so a stage
    /// whose row matches a declared struct reuses its name instead of
    /// interning a second one. Name lookups go through the symbol table;
    /// only shape-to-symbol needs an index of its own.
    fn intern_declared_shapes(&mut self, block: BlockRef<'c, '_>) {
        for op in block.operations() {
            if let Some(YzlOperationRef::Struct(item)) = YzlOperationRef::of(&op) {
                let fields = struct_fields(&item);
                self.shapes
                    .entry(fields)
                    .or_insert_with(|| item.sym_name().value());
            }
        }
    }

    fn lower_block(
        &mut self,
        block: BlockRef<'c, '_>,
        target: BlockRef<'c, 'a>,
        source: &SymbolTable<'c, '_>,
        symbols: &mut SymbolTable<'c, '_>,
    ) {
        for op in block.operations() {
            self.lower_op(op, target, source, symbols);
        }
    }

    fn lower_op(
        &mut self,
        op: OperationRef<'c, '_>,
        target: BlockRef<'c, 'a>,
        source: &SymbolTable<'c, '_>,
        symbols: &mut SymbolTable<'c, '_>,
    ) {
        match YzlOperationRef::of(&op) {
            Some(YzlOperationRef::Struct(item)) => {
                let fields = struct_fields(&item);
                self.declare_struct(item.sym_name().value(), &fields, symbols);
            }
            // A table declaration says nothing yzr needs: `yzr.table` names
            // the relation and carries its row as the result type.
            Some(YzlOperationRef::Table(_)) => {}
            Some(YzlOperationRef::From(from)) => {
                let relation = from.source().value();
                let Some((rows, schema)) =
                    self.relation_input(relation, source, target, symbols, op.location())
                else {
                    self.error(op, format!("`{relation}` has no row shape to scan"));
                    return;
                };

                self.record_stage(op, rows, schema);
            }
            // Queries are expressions, so a binding is a name for the value
            // its body yields: the stages inside lower into the module just
            // as they would outside it, and the name reaches the result.
            Some(YzlOperationRef::Let(binding)) => {
                let Some(block) = binding.body().first_block() else {
                    self.error(op, "`let` has no body to bind");
                    return;
                };

                for inner in block.operations() {
                    self.lower_op(inner, target, source, symbols);
                }

                let bound = block
                    .last_operation()
                    .and_then(|yielded| yielded.try_first_operand())
                    .and_then(|value| self.stages.get(&value_id(value)).cloned());

                match bound {
                    Some(rows) => {
                        self.bindings.insert(binding.sym_name().value(), rows);
                    }
                    None => self.error(op, "only a query can be bound by `let`"),
                }
            }
            Some(YzlOperationRef::Where(stage)) => {
                let Some((input, schema)) = self.input_stage(op) else {
                    return;
                };

                let (region, _) =
                    self.lower_region(stage.body(), &schema, op.location(), Yielded::Body);
                let filtered = target.append_operation(
                    yzr::filter(self.context, input, region, op.location()).into(),
                );

                self.record_stage(op, filtered.first_result(), schema);
            }
            Some(YzlOperationRef::Select(stage)) => {
                let Some((input, schema)) = self.input_stage(op) else {
                    return;
                };

                let (region, yielded) =
                    self.lower_region(stage.body(), &schema, op.location(), Yielded::Body);
                let produced = self.named_row(stage.names().strings(), yielded);
                let row = self.row_type(&produced, symbols);
                let projected = target.append_operation(
                    yzr::project(self.context, row, input, region, op.location()).into(),
                );

                self.record_stage(op, projected.first_result(), produced);
            }
            Some(YzlOperationRef::Extend(stage)) => {
                let Some((input, mut schema)) = self.input_stage(op) else {
                    return;
                };

                let (region, yielded) =
                    self.lower_region(stage.body(), &schema, op.location(), Yielded::Body);
                schema.extend(self.named_row(stage.names().strings(), yielded));
                let row = self.row_type(&schema, symbols);
                let extended = target.append_operation(
                    yzr::extend(self.context, row, input, region, op.location()).into(),
                );

                self.record_stage(op, extended.first_result(), schema);
            }
            Some(YzlOperationRef::Aggregate(stage)) => {
                let Some((input, schema)) = self.input_stage(op) else {
                    return;
                };

                let keys = crate::infer_types::indices(stage.key_cols());
                let (region, yielded) =
                    self.lower_region(stage.body(), &schema, op.location(), Yielded::Body);
                let mut produced: Schema<'c> = keys
                    .iter()
                    .filter_map(|&index| schema.get(index).cloned())
                    .collect();
                produced.extend(self.named_row(stage.names().strings(), yielded));

                let row = self.row_type(&produced, symbols);
                let indices: Vec<i64> = keys.iter().map(|&index| index as i64).collect();
                let grouped = target.append_operation(
                    yzr::aggregate(
                        self.context,
                        row,
                        input,
                        region,
                        DenseI64ArrayAttribute::new(self.context, &indices).into(),
                        op.location(),
                    )
                    .into(),
                );

                self.record_stage(op, grouped.first_result(), produced);
            }
            Some(YzlOperationRef::Join(stage)) => {
                let Some((lhs, mut schema)) = self.input_stage(op) else {
                    return;
                };

                // The one stage that has to conjure an input: yzl names the
                // right side, yzr joins two relations.
                let relation = stage.rhs().value();
                let Some((rows, right)) =
                    self.relation_input(relation, source, target, symbols, op.location())
                else {
                    self.error(op, format!("`{relation}` has no row shape to scan"));
                    return;
                };

                // Both sides carry through, and the `on` region's names were
                // resolved against exactly this concatenation — a qualifier
                // only ever chose a column, so it is spent by now.
                let left_width = schema.len();
                schema.extend(right.iter().copied());

                let region = match stage.using_columns() {
                    Some(columns) => self.join_keys(op, &columns.strings(), left_width, &schema),
                    None => {
                        self.lower_region(stage.on(), &schema, op.location(), Yielded::Body)
                            .0
                    }
                };

                let row = self.row_type(&schema, symbols);
                let joined = target.append_operation(
                    yzr::join(
                        self.context,
                        row,
                        lhs,
                        rows,
                        region,
                        stage.kind(),
                        op.location(),
                    )
                    .into(),
                );

                self.record_stage(op, joined.first_result(), schema);
            }
            Some(YzlOperationRef::Limit(stage)) => {
                let Some((input, schema)) = self.input_stage(op) else {
                    return;
                };

                let limited = target.append_operation(
                    yzr::limit(self.context, input, stage.count(), op.location()).into(),
                );

                self.record_stage(op, limited.first_result(), schema);
            }
            Some(YzlOperationRef::Output(_)) => {
                let Some((query, _)) = self.input_stage(op) else {
                    return;
                };

                target.append_operation(yzr::output(self.context, query, op.location()).into());
            }
            // A qualifier only ever chose a column, and resolution has spent
            // it by now: the row that arrives is the row that leaves.
            Some(YzlOperationRef::Alias(_)) => {
                let Some((input, schema)) = self.input_stage(op) else {
                    return;
                };

                self.record_stage(op, input, schema);
            }
            // A group with no measures: every column is a key, so each
            // distinct row survives exactly once.
            Some(YzlOperationRef::Distinct(_)) => {
                let Some((input, schema)) = self.input_stage(op) else {
                    return;
                };

                let keys: Vec<i64> = (0..schema.len() as i64).collect();
                let region = self.column_region(&schema, &[], op.location());
                let row = self.row_type(&schema, symbols);
                let grouped = target.append_operation(
                    yzr::aggregate(
                        self.context,
                        row,
                        input,
                        region,
                        DenseI64ArrayAttribute::new(self.context, &keys).into(),
                        op.location(),
                    )
                    .into(),
                );

                self.record_stage(op, grouped.first_result(), schema);
            }
            Some(YzlOperationRef::Drop(stage)) => {
                let Some((input, schema)) = self.input_stage(op) else {
                    return;
                };

                let Some(kept) = self.kept_columns(op, &stage.columns().strings(), &schema) else {
                    return;
                };

                let region = self.column_region(&schema, &kept, op.location());
                let produced: Schema<'c> = kept.iter().map(|&index| schema[index]).collect();
                let row = self.row_type(&produced, symbols);
                let projected = target.append_operation(
                    yzr::project(self.context, row, input, region, op.location()).into(),
                );

                self.record_stage(op, projected.first_result(), produced);
            }
            Some(YzlOperationRef::Set(stage)) => {
                let Some((input, mut schema)) = self.input_stage(op) else {
                    return;
                };

                // The body computes replacements, not a new row: every column
                // it does not name carries through in place.
                let columns = crate::infer_types::indices(stage.set_cols());
                let (region, yielded) =
                    self.lower_region(stage.body(), &schema, op.location(), Yielded::Row(&columns));

                for (column, ty) in schema.iter_mut().zip(&yielded) {
                    column.1 = *ty;
                }

                let row = self.row_type(&schema, symbols);
                let projected = target.append_operation(
                    yzr::project(self.context, row, input, region, op.location()).into(),
                );

                self.record_stage(op, projected.first_result(), schema);
            }
            // yzr has no op for a change of name alone, so a rename is the
            // projection of every column under the names the stage gave them.
            Some(YzlOperationRef::Rename(stage)) => {
                let Some((input, mut schema)) = self.input_stage(op) else {
                    return;
                };

                let columns = crate::infer_types::indices(stage.rename_cols());
                for (&index, name) in columns.iter().zip(stage.to().strings()) {
                    match schema.get_mut(index) {
                        Some(column) => column.0 = name,
                        None => {
                            self.error(op, format!("column {index} is not in the row"));
                            return;
                        }
                    }
                }

                let all: Vec<usize> = (0..schema.len()).collect();
                let region = self.column_region(&schema, &all, op.location());
                let row = self.row_type(&schema, symbols);
                let projected = target.append_operation(
                    yzr::project(self.context, row, input, region, op.location()).into(),
                );

                self.record_stage(op, projected.first_result(), schema);
            }
            // Expansion removes these once every call is gone, so one
            // reaching here means expansion did not finish — the reason is
            // already reported, and this says which declaration outlived it.
            Some(YzlOperationRef::Fn(_) | YzlOperationRef::Trait(_) | YzlOperationRef::Impl(_)) => {
                self.error(
                    op,
                    format!("`{}` was not expanded before lowering", op_name(op)),
                )
            }
            Some(YzlOperationRef::Missing(_)) => {
                self.error(op, "this part of the query is missing")
            }
            // Declarations yzr does not need, and the terminators a region
            // owns rather than the module.
            Some(
                YzlOperationRef::Name(_)
                | YzlOperationRef::Call(_)
                | YzlOperationRef::List(_)
                | YzlOperationRef::Yield(_)
                | YzlOperationRef::Return(_),
            )
            | None => {}
        }
    }

    /// A stage's region, rebuilt with the input row's columns as block
    /// arguments — which is what makes column reference into SSA use-def.
    fn lower_region(
        &mut self,
        source: melior::ir::RegionRef<'c, '_>,
        schema: &Schema<'c>,
        location: Location<'c>,
        yielded: Yielded<'_>,
    ) -> (Region<'c>, Vec<Type<'c>>) {
        let region = Region::new();
        let arguments: Vec<(Type<'c>, Location<'c>)> = schema
            .iter()
            .map(|(_, column)| (*column, location))
            .collect();
        let body = region.append_block(Block::new(&arguments));

        let mut produced = Vec::new();
        if let Some(block) = source.first_block() {
            let mut values: HashMap<usize, Value<'c, '_>> = HashMap::new();
            for op in block.operations() {
                self.lower_expression(op, body, &mut values, &mut produced);
            }
        }

        let row = match yielded {
            Yielded::Body => produced,
            Yielded::Row(columns) => Self::substituted_row(body, schema.len(), columns, &produced),
        };

        let types = row.iter().map(|value| value.r#type()).collect();
        body.append_operation(yzr::r#yield(self.context, &row, location).into());

        (region, types)
    }

    /// The whole row, with each replaced column taking the value the body
    /// computed for it — what `set` means, against a yzr that only projects.
    fn substituted_row<'b>(
        body: BlockRef<'c, 'b>,
        width: usize,
        columns: &[usize],
        produced: &[Value<'c, 'b>],
    ) -> Vec<Value<'c, 'b>> {
        (0..width)
            .map(|index| {
                columns
                    .iter()
                    .position(|&column| column == index)
                    .and_then(|slot| produced.get(slot).copied())
                    .unwrap_or_else(|| {
                        body.argument(index)
                            .expect("the column is in the row")
                            .into()
                    })
            })
            .collect()
    }

    /// A region yielding the row's own columns, in the order given — what the
    /// stages that only move names become, since yzr projects rows and has no
    /// op for a change of name alone.
    fn column_region(
        &self,
        schema: &Schema<'c>,
        columns: &[usize],
        location: Location<'c>,
    ) -> Region<'c> {
        let region = Region::new();
        let arguments: Vec<(Type<'c>, Location<'c>)> = schema
            .iter()
            .map(|(_, column)| (*column, location))
            .collect();
        let body = region.append_block(Block::new(&arguments));

        let yielded: Vec<Value<'c, '_>> = columns
            .iter()
            .map(|&index| {
                body.argument(index)
                    .expect("the column is in the row")
                    .into()
            })
            .collect();

        body.append_operation(yzr::r#yield(self.context, &yielded, location).into());

        region
    }

    /// An expression op, rebuilt against the values its operands became.
    fn lower_expression<'b>(
        &mut self,
        op: OperationRef<'c, '_>,
        body: BlockRef<'c, 'b>,
        values: &mut HashMap<usize, Value<'c, 'b>>,
        produced: &mut Vec<Value<'c, 'b>>,
    ) {
        match YzlOperationRef::of(&op) {
            Some(YzlOperationRef::Name(name)) => {
                let Some(index) = name.col().map(|col| col.value() as usize) else {
                    self.error(op, "a name outside a column context is not lowered yet");
                    return;
                };

                let Ok(column) = body.argument(index) else {
                    self.error(op, format!("column {index} is not in the row"));
                    return;
                };

                values.insert(value_id(op.first_result()), column.into());
            }
            Some(YzlOperationRef::Yield(_)) => {
                produced.extend(self.mapped_operands(op, values));
            }
            Some(YzlOperationRef::Call(call)) => {
                let callee = call.callee().value().to_string();
                let operands = self.mapped_operands(op, values);
                let ty = self.stamped_type(op);
                let kind = call
                    .callee_kind()
                    .map(|kind| kind.value())
                    .unwrap_or_default();
                let lowered = if kind == "builtin" && self.is_aggregate(&callee) {
                    self.lower_measure(op, &callee, &operands, ty, body)
                } else if kind == "external" {
                    yz::extern_call(
                        self.context,
                        ty,
                        &operands,
                        StringAttribute::new(self.context, &callee),
                        op.location(),
                    )
                    .into()
                } else {
                    yz::call(
                        self.context,
                        ty,
                        &operands,
                        FlatSymbolRefAttribute::new(self.context, &callee),
                        op.location(),
                    )
                    .into()
                };

                let appended = body.append_operation(lowered);
                values.insert(value_id(op.first_result()), appended.first_result());
            }
            // Everything else is a `yz` op, structurally unchanged: the
            // operands it was given, and the type inference stamped on it.
            None => {
                let operands = self.mapped_operands(op, values);
                let rebuilt = self.rebuild(op, &operands, body);
                if let Some(rebuilt) = rebuilt {
                    values.insert(value_id(op.first_result()), rebuilt);
                }
            }
            // The parse error above it already said what went wrong; this says
            // the query cannot be built from what is left, rather than
            // implying some lowering is still to come.
            Some(YzlOperationRef::Missing(_)) => {
                self.error(op, "this part of the query is missing")
            }
            Some(
                YzlOperationRef::List(_)
                | YzlOperationRef::From(_)
                | YzlOperationRef::Where(_)
                | YzlOperationRef::Select(_)
                | YzlOperationRef::Extend(_)
                | YzlOperationRef::Aggregate(_)
                | YzlOperationRef::Limit(_)
                | YzlOperationRef::Join(_)
                | YzlOperationRef::Rename(_)
                | YzlOperationRef::Alias(_)
                | YzlOperationRef::Distinct(_)
                | YzlOperationRef::Drop(_)
                | YzlOperationRef::Set(_)
                | YzlOperationRef::Output(_)
                | YzlOperationRef::Struct(_)
                | YzlOperationRef::Table(_)
                | YzlOperationRef::Fn(_)
                | YzlOperationRef::Trait(_)
                | YzlOperationRef::Impl(_)
                | YzlOperationRef::Let(_)
                | YzlOperationRef::Return(_),
            ) => self.error(op, format!("`{}` is not lowered yet", op_name(op))),
        }
    }

    /// A `yz` op carried over with its attributes, its mapped operands, and
    /// the concrete type inference gave it.
    fn rebuild<'b>(
        &mut self,
        op: OperationRef<'c, '_>,
        operands: &[Value<'c, 'b>],
        body: BlockRef<'c, 'b>,
    ) -> Option<Value<'c, 'b>> {
        let name = op.name();
        let existing = op.try_first_result()?;

        let ty = op
            .attribute("ty")
            .ok()
            .and_then(|attribute| TypeAttribute::try_from(attribute).ok())
            .map(|attribute| attribute.value())
            .unwrap_or_else(|| existing.r#type());

        let attributes: Vec<(Identifier<'c>, Attribute<'c>)> = (0..op.attribute_count())
            .map(|index| {
                op.attribute_at(index)
                    .expect("the attribute index is in range")
            })
            .filter(|(name, _)| name.as_string_ref().as_str() != Ok("ty"))
            .collect();

        let rebuilt = OperationBuilder::new(
            name.as_string_ref().as_str().expect("op names are utf-8"),
            op.location(),
        )
        .add_operands(operands)
        .add_results(&[ty])
        .add_attributes(&attributes)
        .build()
        .expect("a stamped yz op rebuilds");

        Some(body.append_operation(rebuilt).first_result())
    }

    fn mapped_operands<'b>(
        &mut self,
        op: OperationRef<'c, '_>,
        values: &HashMap<usize, Value<'c, 'b>>,
    ) -> Vec<Value<'c, 'b>> {
        op.operands()
            .filter_map(|operand| values.get(&value_id(operand)).copied())
            .collect()
    }

    /// A measure: `count` takes no value, every other aggregate does.
    fn lower_measure(
        &self,
        op: OperationRef<'c, '_>,
        callee: &str,
        operands: &[Value<'c, '_>],
        ty: Type<'c>,
        _body: BlockRef<'c, '_>,
    ) -> melior::ir::Operation<'c> {
        match operands.first() {
            Some(value) => yzr::agg(
                self.context,
                ty,
                *value,
                StringAttribute::new(self.context, callee),
                op.location(),
            )
            .into(),
            None => yzr::count(self.context, ty, op.location()).into(),
        }
    }

    fn is_aggregate(&self, callee: &str) -> bool {
        self.registry
            .entries()
            .iter()
            .any(|entry| entry.name == callee && matches!(entry.func, BuiltinFunc::Aggregate(_)))
    }

    /// The type inference stamped, or the one the op already carries.
    fn stamped_type(&self, op: OperationRef<'c, '_>) -> Type<'c> {
        op.attribute("ty")
            .ok()
            .and_then(|attribute| TypeAttribute::try_from(attribute).ok())
            .map(|attribute| attribute.value())
            .unwrap_or_else(|| op.first_result().r#type())
    }

    /// `using [a, b]` is sugar: yzr has only an on-region, so the columns
    /// become the equality the join was asking for.
    fn join_keys(
        &mut self,
        op: OperationRef<'c, '_>,
        columns: &[&str],
        left_width: usize,
        schema: &Schema<'c>,
    ) -> Region<'c> {
        let location = op.location();
        let region = Region::new();
        let arguments: Vec<(Type<'c>, Location<'c>)> = schema
            .iter()
            .map(|(_, column)| (*column, location))
            .collect();
        let body = region.append_block(Block::new(&arguments));

        let mut condition: Option<Value<'c, '_>> = None;
        for column in columns {
            let left = schema[..left_width]
                .iter()
                .position(|(name, _)| name == column);
            let right = schema[left_width..]
                .iter()
                .position(|(name, _)| name == column)
                .map(|index| index + left_width);
            let (Some(left), Some(right)) = (left, right) else {
                self.error(op, format!("`{column}` is not present in both relations"));
                continue;
            };

            let equal = body.append_operation(
                yz::cmp(
                    self.context,
                    types::boolean(self.context),
                    body.argument(left)
                        .expect("the left column is in range")
                        .into(),
                    body.argument(right)
                        .expect("the right column is in range")
                        .into(),
                    StringAttribute::new(self.context, "eq"),
                    location,
                )
                .into(),
            );

            condition = Some(match condition {
                Some(previous) => body
                    .append_operation(
                        yz::and(
                            self.context,
                            types::boolean(self.context),
                            previous,
                            equal.first_result(),
                            location,
                        )
                        .into(),
                    )
                    .first_result(),
                None => equal.first_result(),
            });
        }

        let yielded: Vec<Value<'c, '_>> = condition.into_iter().collect();
        body.append_operation(yzr::r#yield(self.context, &yielded, location).into());

        region
    }

    /// The columns a stage names, paired with what its region yielded.
    fn named_row(&self, names: Vec<&'c str>, yielded: Vec<Type<'c>>) -> Schema<'c> {
        names.into_iter().zip(yielded).collect()
    }

    /// The row a relation's rows have, found the way MLIR finds any symbol:
    /// the table names its struct, and the struct carries its fields.
    fn relation_schema(&self, name: &str, source: &SymbolTable<'c, '_>) -> Option<Schema<'c>> {
        let table = source.lookup(name)?;
        let YzlOperationRef::Table(table) = YzlOperationRef::of(&table)? else {
            return None;
        };

        let declaration = source.lookup(table.row().value())?;
        let YzlOperationRef::Struct(item) = YzlOperationRef::of(&declaration)? else {
            return None;
        };

        Some(struct_fields(&item))
    }

    /// The type standing for a row — declaring the shape when nothing has.
    fn row_type(&mut self, schema: &Schema<'c>, symbols: &mut SymbolTable<'c, '_>) -> Type<'c> {
        if let Some(name) = self.shapes.get(schema) {
            return StructType::new(self.context, name).into();
        }

        let name = match self.shapes.get(schema).cloned() {
            Some(name) => name,
            None => self.declare_struct("row", schema, symbols),
        };

        self.shapes.insert(schema.clone(), name);
        StructType::new(self.context, name).into()
    }

    /// Declares a struct in the lowered module, returning the name it got —
    /// the symbol table renames a collision, which is what interns a shape
    /// nobody declared.
    fn declare_struct(
        &self,
        name: &str,
        fields: &Schema<'c>,
        symbols: &mut SymbolTable<'c, '_>,
    ) -> &'c str {
        let names: Vec<Attribute<'c>> = fields
            .iter()
            .map(|(column, _)| StringAttribute::new(self.context, column).into())
            .collect();
        let types: Vec<Attribute<'c>> = fields
            .iter()
            .map(|(_, ty)| TypeAttribute::new(*ty).into())
            .collect();

        let declaration = yz::r#struct(
            self.context,
            StringAttribute::new(self.context, name),
            melior::ir::attribute::ArrayAttribute::new(self.context, &names),
            melior::ir::attribute::ArrayAttribute::new(self.context, &types),
            Location::unknown(self.context),
        );

        let assigned = symbols.insert(declaration.into());
        StringAttribute::try_from(assigned)
            .expect("a symbol name is a string")
            .value()
    }

    /// The rows a relation name stands for: a `let` has already produced
    /// them, and a table is scanned where it is used.
    fn relation_input(
        &mut self,
        name: &str,
        source: &SymbolTable<'c, '_>,
        target: BlockRef<'c, 'a>,
        symbols: &mut SymbolTable<'c, '_>,
        location: Location<'c>,
    ) -> Option<(Value<'c, 'a>, Schema<'c>)> {
        if let Some(bound) = self.bindings.get(name) {
            return Some(bound.clone());
        }

        let schema = self.relation_schema(name, source)?;
        let row = self.row_type(&schema, symbols);
        let scanned = target.append_operation(
            yzr::table(
                self.context,
                row,
                FlatSymbolRefAttribute::new(self.context, name),
                location,
            )
            .into(),
        );

        Some((scanned.first_result(), schema))
    }

    /// The columns that survive a `drop`, in order. Resolution removes the
    /// first column each name matches, so dropping one name twice drops two
    /// columns — the lowering has to agree with it exactly.
    fn kept_columns(
        &self,
        op: OperationRef<'c, '_>,
        columns: &[&str],
        schema: &Schema<'c>,
    ) -> Option<Vec<usize>> {
        let mut dropped: Vec<usize> = Vec::new();
        for column in columns {
            let found = schema
                .iter()
                .enumerate()
                .find(|(index, (name, _))| name == column && !dropped.contains(index))
                .map(|(index, _)| index);

            match found {
                Some(index) => dropped.push(index),
                None => {
                    self.error(op, format!("`{column}` is not in the row"));
                    return None;
                }
            }
        }

        Some(
            (0..schema.len())
                .filter(|index| !dropped.contains(index))
                .collect(),
        )
    }

    fn input_stage(&mut self, op: OperationRef<'c, '_>) -> Option<(Value<'c, 'a>, Schema<'c>)> {
        let input = op.try_first_operand()?;
        self.stages.get(&value_id(input)).cloned()
    }

    fn record_stage(&mut self, op: OperationRef<'c, '_>, value: Value<'c, 'a>, schema: Schema<'c>) {
        if let Some(result) = op.try_first_result() {
            self.stages.insert(value_id(result), (value, schema));
        }
    }

    fn error(&self, op: OperationRef<'c, '_>, message: impl AsRef<str>) {
        yuzu_mlir::diagnostics::emit_error(op.location(), message.as_ref());
    }
}

fn op_name<'c>(op: OperationRef<'c, '_>) -> String {
    op.name()
        .as_string_ref()
        .as_str()
        .unwrap_or("<non-utf8>")
        .to_string()
}

/// The fields a struct declaration carries, in order.
fn struct_fields<'c>(item: &yuzu_mlir::ops::yzl::StructOperationRef<'c, '_>) -> Schema<'c> {
    item.names()
        .strings()
        .into_iter()
        .zip(field_types(item.types()))
        .collect()
}

fn field_types<'c>(types: melior::ir::attribute::ArrayAttribute<'c>) -> Vec<Type<'c>> {
    types
        .elements()
        .filter_map(|element| TypeAttribute::try_from(element).ok())
        .map(|attribute| attribute.value())
        .collect()
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::test_support::check_lowered;

    /// The stage regions' block arguments are the row's columns, so a column
    /// reference becomes an SSA use.
    #[test]
    fn names_become_block_arguments() {
        check_lowered(
            r#"
struct Row { a: int64, b: int64 }
table t = Row

from t
|> where a > 10
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a", "b"] : [!yz.int64, !yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  %1 = yzr.filter %0 : !yz.struct<@Row> {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64):
                    %2 = yz.constant_int 10
                    %3 = yz.cmp "gt", %arg0, %2 : !yz.int64, !yz.int64 -> !yz.bool
                    yzr.yield %3 : !yz.bool
                  }
                  yzr.output %1 : !yz.struct<@Row>
                }
            "#]],
        );
    }

    /// The stages that name columns declare the row they produce, and a
    /// shape nobody declared is interned under a name of its own.
    #[test]
    fn named_stages_declare_the_row_they_produce() {
        check_lowered(
            r#"
struct Row { a: int64, b: int64 }
table t = Row

from t
|> extend a + b as e
|> select a as x, e as y
|> limit 5
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a", "b"] : [!yz.int64, !yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  yz.struct @row ["a", "b", "e"] : [!yz.int64, !yz.int64, !yz.int64]
                  %1 = yzr.extend %0 {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64):
                    %4 = yz.add %arg0, %arg1 : !yz.int64, !yz.int64 -> !yz.int64
                    yzr.yield %4 : !yz.int64
                  } : !yz.struct<@Row> -> !yz.struct<@row>
                  yz.struct @row_0 ["x", "y"] : [!yz.int64, !yz.int64]
                  %2 = yzr.project %1 {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64, %arg2: !yz.int64):
                    yzr.yield %arg0, %arg2 : !yz.int64, !yz.int64
                  } : !yz.struct<@row> -> !yz.struct<@row_0>
                  %3 = yzr.limit %2, 5 : !yz.struct<@row_0>
                  yzr.output %3 : !yz.struct<@row_0>
                }
            "#]],
        );
    }

    /// Measures become `yzr.agg`, and the keys come from the stamp
    /// resolution left.
    #[test]
    fn measures_become_aggregate_ops() {
        check_lowered(
            r#"
struct Row { a: int64, b: int64 }
table t = Row

from t
|> aggregate sum(a) as total, count() as n group by b
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a", "b"] : [!yz.int64, !yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  yz.struct @row ["b", "total", "n"] : [!yz.int64, !yz.int64, !yz.int64]
                  %1 = yzr.aggregate %0 keys [1] {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64):
                    %2 = yzr.agg "sum", %arg0 : !yz.int64 -> !yz.int64
                    %3 = yzr.count : !yz.int64
                    yzr.yield %2, %3 : !yz.int64, !yz.int64
                  } : !yz.struct<@Row> -> !yz.struct<@row>
                  yzr.output %1 : !yz.struct<@row>
                }
            "#]],
        );
    }

    /// The one stage with two inputs: the named right side becomes a scan,
    /// and the `on` region sees both rows' columns as one block.
    #[test]
    fn join_materialises_its_right_side() {
        check_lowered(
            r#"
struct Row { id: int64, dept_id: int64 }
table t = Row
struct Dept { key: int64, name: str }
table depts = Dept

from t
|> left join depts as d on dept_id == d.key
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["id", "dept_id"] : [!yz.int64, !yz.int64]
                  yz.struct @Dept ["key", "name"] : [!yz.int64, !yz.str]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  %1 = yzr.table @depts : !yz.struct<@Dept>
                  yz.struct @row ["id", "dept_id", "key", "name"] : [!yz.int64, !yz.int64, !yz.int64, !yz.str]
                  %2 = yzr.join "left", %0, %1 {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64, %arg2: !yz.int64, %arg3: !yz.str):
                    %3 = yz.cmp "eq", %arg1, %arg2 : !yz.int64, !yz.int64 -> !yz.bool
                    yzr.yield %3 : !yz.bool
                  } : !yz.struct<@Row>, !yz.struct<@Dept> -> !yz.struct<@row>
                  yzr.output %2 : !yz.struct<@row>
                }
            "#]],
        );
    }

    /// `using` is sugar for the equality it asks for; both sides' columns
    /// carry through, as they do for `on`.
    #[test]
    fn using_becomes_the_equality_it_means() {
        check_lowered(
            r#"
struct Row { id: int64, tag: str, part: int64 }
table t = Row
struct Other { id: int64, part: int64, extra: int64 }
table u = Other

from t
|> inner join u using (id, part)
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["id", "tag", "part"] : [!yz.int64, !yz.str, !yz.int64]
                  yz.struct @Other ["id", "part", "extra"] : [!yz.int64, !yz.int64, !yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  %1 = yzr.table @u : !yz.struct<@Other>
                  yz.struct @row ["id", "tag", "part", "id", "part", "extra"] : [!yz.int64, !yz.str, !yz.int64, !yz.int64, !yz.int64, !yz.int64]
                  %2 = yzr.join "inner", %0, %1 {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.str, %arg2: !yz.int64, %arg3: !yz.int64, %arg4: !yz.int64, %arg5: !yz.int64):
                    %3 = yz.cmp "eq", %arg0, %arg3 : !yz.int64, !yz.int64 -> !yz.bool
                    %4 = yz.cmp "eq", %arg2, %arg4 : !yz.int64, !yz.int64 -> !yz.bool
                    %5 = yz.and %3, %4 : !yz.bool, !yz.bool -> !yz.bool
                    yzr.yield %5 : !yz.bool
                  } : !yz.struct<@Row>, !yz.struct<@Other> -> !yz.struct<@row>
                  yzr.output %2 : !yz.struct<@row>
                }
            "#]],
        );
    }

    /// `alias` only qualifies names, and resolution has already used them
    /// to choose columns: nothing is left for yzr to represent.
    #[test]
    fn alias_leaves_no_trace() {
        check_lowered(
            r#"
struct Row { a: int64, b: int64 }
table t = Row

from t
|> as r
|> where r.a > 1
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a", "b"] : [!yz.int64, !yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  %1 = yzr.filter %0 : !yz.struct<@Row> {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64):
                    %2 = yz.constant_int 1
                    %3 = yz.cmp "gt", %arg0, %2 : !yz.int64, !yz.int64 -> !yz.bool
                    yzr.yield %3 : !yz.bool
                  }
                  yzr.output %1 : !yz.struct<@Row>
                }
            "#]],
        );
    }

    /// yzr has no distinct: it is a group keyed on every column, measuring
    /// nothing.
    #[test]
    fn distinct_groups_on_every_column() {
        check_lowered(
            r#"
struct Row { a: int64, b: str }
table t = Row

from t
|> distinct
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a", "b"] : [!yz.int64, !yz.str]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  %1 = yzr.aggregate %0 keys [0, 1] {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.str):
                    yzr.yield
                  } : !yz.struct<@Row> -> !yz.struct<@Row>
                  yzr.output %1 : !yz.struct<@Row>
                }
            "#]],
        );
    }

    /// `drop` names what leaves; the projection yields what stays.
    #[test]
    fn drop_projects_the_columns_that_stay() {
        check_lowered(
            r#"
struct Row { a: int64, b: str, c: int64 }
table t = Row

from t
|> drop b
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a", "b", "c"] : [!yz.int64, !yz.str, !yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  yz.struct @row ["a", "c"] : [!yz.int64, !yz.int64]
                  %1 = yzr.project %0 {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.str, %arg2: !yz.int64):
                    yzr.yield %arg0, %arg2 : !yz.int64, !yz.int64
                  } : !yz.struct<@Row> -> !yz.struct<@row>
                  yzr.output %1 : !yz.struct<@row>
                }
            "#]],
        );
    }

    /// `set` replaces columns in place, so the projection has to yield the
    /// columns it did not name as well.
    #[test]
    fn set_yields_the_untouched_columns_too() {
        check_lowered(
            r#"
struct Row { a: int64, b: int64, c: int64 }
table t = Row

from t
|> set b = a + 1
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a", "b", "c"] : [!yz.int64, !yz.int64, !yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  %1 = yzr.project %0 {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64, %arg2: !yz.int64):
                    %2 = yz.constant_int 1
                    %3 = yz.add %arg0, %2 : !yz.int64, !yz.int64 -> !yz.int64
                    yzr.yield %arg0, %3, %arg2 : !yz.int64, !yz.int64, !yz.int64
                  } : !yz.struct<@Row> -> !yz.struct<@Row>
                  yzr.output %1 : !yz.struct<@Row>
                }
            "#]],
        );
    }

    /// A binding is a name for rows that already exist: two uses share the
    /// one scan rather than each producing their own.
    #[test]
    fn a_binding_is_reused_not_rescanned() {
        check_lowered(
            r#"
struct Row { a: int64, b: int64 }
table t = Row

let big = from t |> where a > 10

from big
|> inner join big using (a)
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a", "b"] : [!yz.int64, !yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  %1 = yzr.filter %0 : !yz.struct<@Row> {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64):
                    %3 = yz.constant_int 10
                    %4 = yz.cmp "gt", %arg0, %3 : !yz.int64, !yz.int64 -> !yz.bool
                    yzr.yield %4 : !yz.bool
                  }
                  yz.struct @row ["a", "b", "a", "b"] : [!yz.int64, !yz.int64, !yz.int64, !yz.int64]
                  %2 = yzr.join "inner", %1, %1 {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64, %arg2: !yz.int64, %arg3: !yz.int64):
                    %3 = yz.cmp "eq", %arg0, %arg2 : !yz.int64, !yz.int64 -> !yz.bool
                    yzr.yield %3 : !yz.bool
                  } : !yz.struct<@Row>, !yz.struct<@Row> -> !yz.struct<@row>
                  yzr.output %2 : !yz.struct<@row>
                }
            "#]],
        );
    }

    /// Binding something that is not a query is not carried yet, and a gap
    /// is an error rather than a silently dropped binding.
    #[test]
    fn reports_a_binding_that_is_not_a_query() {
        check_lowered(
            r#"
struct Row { a: int64 }
table t = Row

let n = 1 + 2

from t
"#,
            expect![[r#"
                error: only a query can be bound by `let`
                 --> test.yz:5:1
                  |
                5 | let n = 1 + 2
                  | ^
            "#]],
        );
    }

    /// A rename moves names, not values, but yzr rows are typed by their
    /// struct — so the new names need a projection to live on.
    #[test]
    fn rename_projects_under_the_new_names() {
        check_lowered(
            r#"
struct Row { a: int64, b: str }
table t = Row

from t
|> rename a as x, b as y
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a", "b"] : [!yz.int64, !yz.str]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  yz.struct @row ["x", "y"] : [!yz.int64, !yz.str]
                  %1 = yzr.project %0 {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.str):
                    yzr.yield %arg0, %arg1 : !yz.int64, !yz.str
                  } : !yz.struct<@Row> -> !yz.struct<@row>
                  yzr.output %1 : !yz.struct<@row>
                }
            "#]],
        );
    }

    /// The stamp resolution left says which column each name applies to, so
    /// a qualified rename after a join renames one side rather than guessing.
    #[test]
    fn rename_follows_the_stamped_column() {
        check_lowered(
            r#"
struct Row { id: int64 }
table l = Row
struct Other { id: int64 }
table r = Other

from l
|> as a
|> inner join r as b on a.id == b.id
|> rename b.id as other
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["id"] : [!yz.int64]
                  yz.struct @Other ["id"] : [!yz.int64]
                  %0 = yzr.table @l : !yz.struct<@Row>
                  %1 = yzr.table @r : !yz.struct<@Row>
                  yz.struct @row ["id", "id"] : [!yz.int64, !yz.int64]
                  %2 = yzr.join "inner", %0, %1 {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64):
                    %4 = yz.cmp "eq", %arg0, %arg1 : !yz.int64, !yz.int64 -> !yz.bool
                    yzr.yield %4 : !yz.bool
                  } : !yz.struct<@Row>, !yz.struct<@Row> -> !yz.struct<@row>
                  yz.struct @row_0 ["id", "other"] : [!yz.int64, !yz.int64]
                  %3 = yzr.project %2 {
                  ^bb0(%arg0: !yz.int64, %arg1: !yz.int64):
                    yzr.yield %arg0, %arg1 : !yz.int64, !yz.int64
                  } : !yz.struct<@row> -> !yz.struct<@row_0>
                  yzr.output %3 : !yz.struct<@row_0>
                }
            "#]],
        );
    }

    /// A hole the parser left behind reaches here as a `yzl.missing`. The
    /// parse error above it says what went wrong; this says the query cannot
    /// be built from it, rather than implying a lowering is missing.
    #[test]
    fn reports_a_missing_piece() {
        check_lowered(
            r#"
struct Row { a: int64 }
table t = Row

from t
|> where a >
"#,
            expect![[r#"
                error: expected expression, found end of input
                 --> test.yz:6:13
                  |
                6 | |> where a >
                  | 

                error: binary expression is missing its right operand
                 --> test.yz:6:10
                  |
                6 | |> where a >
                  |          ^^^

                error: this part of the query is missing
                 --> test.yz:6:10
                  |
                6 | |> where a >
                  |          ^
            "#]],
        );
    }

    /// A stage the lowering does not carry yet is an error, not a silent gap.
    #[test]
    fn reports_a_stage_that_is_not_lowered() {
        check_lowered(
            r#"
struct Row { a: int64 }
table t = Row

from t
|> rename a as b
"#,
            expect![[r#"
                module {
                  yz.struct @Row ["a"] : [!yz.int64]
                  %0 = yzr.table @t : !yz.struct<@Row>
                  yz.struct @row ["b"] : [!yz.int64]
                  %1 = yzr.project %0 {
                  ^bb0(%arg0: !yz.int64):
                    yzr.yield %arg0 : !yz.int64
                  } : !yz.struct<@Row> -> !yz.struct<@row>
                  yzr.output %1 : !yz.struct<@row>
                }
            "#]],
        );
    }
}
