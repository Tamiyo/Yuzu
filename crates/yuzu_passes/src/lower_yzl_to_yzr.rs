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
use yuzu_mlir::ops::yzl::YzlOperationRef;
use yuzu_mlir::{StructType, SymbolTable, value_id};
use yuzu_types::{BuiltinFunc, FunctionRegistry};

/// A row: the columns flowing out of a stage, in order.
type Schema<'c> = Vec<(&'c str, Type<'c>)>;

struct YzlToYzr<'c, 'a, 'r> {
    context: &'c Context,
    registry: &'r dyn FunctionRegistry,
    /// What each yzl stage value became, and the row it carries.
    stages: HashMap<usize, (Value<'c, 'a>, Schema<'c>)>,
    /// The symbol declaring each distinct row shape, so a shape nobody
    /// declared is declared once.
    shapes: HashMap<Schema<'c>, &'c str>,
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
                let Some(schema) = self.relation_schema(relation, source) else {
                    self.error(op, format!("`{relation}` has no row shape to scan"));
                    return;
                };

                let row = self.row_type(&schema, symbols);
                let scanned = target.append_operation(
                    yuzu_mlir::ods::yzr::table(
                        self.context,
                        row,
                        FlatSymbolRefAttribute::new(self.context, relation),
                        op.location(),
                    )
                    .into(),
                );

                self.record_stage(op, scanned.first_result(), schema);
            }
            Some(YzlOperationRef::Where(stage)) => {
                let Some((input, schema)) = self.input_stage(op) else {
                    return;
                };

                let (region, _) = self.lower_region(stage.body(), &schema, op.location());
                let filtered = target.append_operation(
                    yuzu_mlir::ods::yzr::filter(self.context, input, region, op.location()).into(),
                );

                self.record_stage(op, filtered.first_result(), schema);
            }
            Some(YzlOperationRef::Select(stage)) => {
                let Some((input, schema)) = self.input_stage(op) else {
                    return;
                };

                let (region, yielded) = self.lower_region(stage.body(), &schema, op.location());
                let produced = self.named_row(stage.names().strings(), yielded);
                let row = self.row_type(&produced, symbols);
                let projected = target.append_operation(
                    yuzu_mlir::ods::yzr::project(self.context, row, input, region, op.location())
                        .into(),
                );

                self.record_stage(op, projected.first_result(), produced);
            }
            Some(YzlOperationRef::Extend(stage)) => {
                let Some((input, mut schema)) = self.input_stage(op) else {
                    return;
                };

                let (region, yielded) = self.lower_region(stage.body(), &schema, op.location());
                schema.extend(self.named_row(stage.names().strings(), yielded));
                let row = self.row_type(&schema, symbols);
                let extended = target.append_operation(
                    yuzu_mlir::ods::yzr::extend(self.context, row, input, region, op.location())
                        .into(),
                );

                self.record_stage(op, extended.first_result(), schema);
            }
            Some(YzlOperationRef::Aggregate(stage)) => {
                let Some((input, schema)) = self.input_stage(op) else {
                    return;
                };

                let keys = crate::infer_types::indices(stage.key_cols());
                let (region, yielded) = self.lower_region(stage.body(), &schema, op.location());
                let mut produced: Schema<'c> = keys
                    .iter()
                    .filter_map(|&index| schema.get(index).cloned())
                    .collect();
                produced.extend(self.named_row(stage.names().strings(), yielded));

                let row = self.row_type(&produced, symbols);
                let indices: Vec<i64> = keys.iter().map(|&index| index as i64).collect();
                let grouped = target.append_operation(
                    yuzu_mlir::ods::yzr::aggregate(
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
                let Some(right) = self.relation_schema(relation, source) else {
                    self.error(op, format!("`{relation}` has no row shape to scan"));
                    return;
                };

                let right_row = self.row_type(&right, symbols);
                let scanned = target.append_operation(
                    yuzu_mlir::ods::yzr::table(
                        self.context,
                        right_row,
                        FlatSymbolRefAttribute::new(self.context, relation),
                        op.location(),
                    )
                    .into(),
                );

                // Both sides carry through, and the `on` region's names were
                // resolved against exactly this concatenation — a qualifier
                // only ever chose a column, so it is spent by now.
                let left_width = schema.len();
                schema.extend(right.iter().copied());

                let region = match stage.using_columns() {
                    Some(columns) => self.join_keys(op, &columns.strings(), left_width, &schema),
                    None => self.lower_region(stage.on(), &schema, op.location()).0,
                };

                let row = self.row_type(&schema, symbols);
                let joined = target.append_operation(
                    yuzu_mlir::ods::yzr::join(
                        self.context,
                        row,
                        lhs,
                        scanned.first_result(),
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
                    yuzu_mlir::ods::yzr::limit(self.context, input, stage.count(), op.location())
                        .into(),
                );

                self.record_stage(op, limited.first_result(), schema);
            }
            Some(YzlOperationRef::Output(_)) => {
                let Some((query, _)) = self.input_stage(op) else {
                    return;
                };

                target.append_operation(
                    yuzu_mlir::ods::yzr::output(self.context, query, op.location()).into(),
                );
            }
            Some(
                YzlOperationRef::Rename(_)
                | YzlOperationRef::Alias(_)
                | YzlOperationRef::Distinct(_)
                | YzlOperationRef::Drop(_)
                | YzlOperationRef::Set(_)
                | YzlOperationRef::Let(_)
                | YzlOperationRef::Fn(_)
                | YzlOperationRef::Trait(_)
                | YzlOperationRef::Impl(_)
                | YzlOperationRef::Missing(_),
            ) => self.error(op, format!("`{}` is not lowered yet", op_name(op))),
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
    ) -> (Region<'c>, Vec<Type<'c>>) {
        let region = Region::new();
        let arguments: Vec<(Type<'c>, Location<'c>)> = schema
            .iter()
            .map(|(_, column)| (*column, location))
            .collect();
        let body = region.append_block(Block::new(&arguments));

        let Some(block) = source.first_block() else {
            return (region, Vec::new());
        };

        let mut values: HashMap<usize, Value<'c, '_>> = HashMap::new();
        let mut yielded = Vec::new();
        for op in block.operations() {
            self.lower_expression(op, body, &mut values, &mut yielded);
        }

        (region, yielded)
    }

    /// An expression op, rebuilt against the values its operands became.
    fn lower_expression<'b>(
        &mut self,
        op: OperationRef<'c, '_>,
        body: BlockRef<'c, 'b>,
        values: &mut HashMap<usize, Value<'c, 'b>>,
        yielded: &mut Vec<Type<'c>>,
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
                let operands = self.mapped_operands(op, values);
                yielded.extend(operands.iter().map(|value| value.r#type()));
                body.append_operation(
                    yuzu_mlir::ods::yzr::r#yield(self.context, &operands, op.location()).into(),
                );
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
                    yuzu_mlir::ods::yz::extern_call(
                        self.context,
                        ty,
                        &operands,
                        StringAttribute::new(self.context, &callee),
                        op.location(),
                    )
                    .into()
                } else {
                    yuzu_mlir::ods::yz::call(
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
            Some(
                YzlOperationRef::Missing(_)
                | YzlOperationRef::List(_)
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
            Some(value) => yuzu_mlir::ods::yzr::agg(
                self.context,
                ty,
                *value,
                StringAttribute::new(self.context, callee),
                op.location(),
            )
            .into(),
            None => yuzu_mlir::ods::yzr::count(self.context, ty, op.location()).into(),
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
                yuzu_mlir::ods::yz::cmp(
                    self.context,
                    yuzu_mlir::types::boolean(self.context),
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
                        yuzu_mlir::ods::yz::and(
                            self.context,
                            yuzu_mlir::types::boolean(self.context),
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
        body.append_operation(
            yuzu_mlir::ods::yzr::r#yield(self.context, &yielded, location).into(),
        );

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

        let declaration = yuzu_mlir::ods::yz::r#struct(
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

    /// A stage the lowering does not carry yet is an error, not a silent gap.
    #[test]
    fn reports_a_stage_that_is_not_lowered() {
        check_lowered(
            r#"
struct Row { a: int64 }
table t = Row

from t
|> distinct
"#,
            expect![[r#"
                error: `yzl.distinct` is not lowered yet
                 --> test.yz:5:1
                  |
                5 | from t
                  | ^
            "#]],
        );
    }
}
