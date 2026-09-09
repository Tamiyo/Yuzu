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
use melior::ir::attribute::{FlatSymbolRefAttribute, StringAttribute, TypeAttribute};
use melior::ir::operation::{OperationBuilder, OperationLike, OperationRef};
use melior::ir::{
    Attribute, Block, BlockLike, BlockRef, Identifier, Location, Module, Region, RegionLike, Type,
    Value, ValueLike,
};
use yuzu_mlir::ext::{ArrayAttributeExt, BlockExt, OperationExt};
use yuzu_mlir::ops::yzl::YzlOperationRef;
use yuzu_mlir::{StructType, SymbolTable, value_id};

/// A row: the columns flowing out of a stage, in order.
type Schema<'c> = Vec<(String, Type<'c>)>;

struct Lowering<'c, 'a> {
    context: &'c Context,
    /// The fields of each declared struct, by symbol.
    structs: HashMap<String, Schema<'c>>,
    /// The struct a relation's rows have, by relation name.
    relations: HashMap<String, String>,
    /// What each yzl stage value became, and the row it carries.
    stages: HashMap<usize, (Value<'c, 'a>, Schema<'c>)>,
    /// The symbol declaring each distinct row shape, so a shape nobody
    /// declared is declared once.
    shapes: HashMap<Schema<'c>, String>,
}

/// Expects a resolved, inferred module. Returns the `yz` + `yzr` module it
/// lowers to; anything it cannot lower is reported and left out.
pub fn lower_yzl<'c>(context: &'c Context, module: &Module<'c>) -> Module<'c> {
    let lowered = Module::new(Location::unknown(context));
    {
        let target = lowered.body();
        let mut symbols = SymbolTable::new(&lowered);
        let mut lowering = Lowering {
            context,
            structs: HashMap::new(),
            relations: HashMap::new(),
            stages: HashMap::new(),
            shapes: HashMap::new(),
        };

        lowering.declare(module.body());
        lowering.lower_block(module.body(), target, &mut symbols);
    }

    lowered
}

impl<'c, 'a> Lowering<'c, 'a> {
    /// Declarations first, so a stage can ask for a relation's row before
    /// the walk reaches the op that declared it.
    fn declare(&mut self, block: BlockRef<'c, '_>) {
        for op in block.operations() {
            match YzlOperationRef::of(&op) {
                Some(YzlOperationRef::Struct(item)) => {
                    let fields = item
                        .names()
                        .strings()
                        .into_iter()
                        .zip(field_types(item.types()))
                        .collect();
                    self.structs
                        .insert(item.sym_name().value().to_string(), fields);
                }
                Some(YzlOperationRef::Table(table)) => {
                    self.relations.insert(
                        table.sym_name().value().to_string(),
                        table.row().value().to_string(),
                    );
                }
                _ => {}
            }
        }
    }

    fn lower_block(
        &mut self,
        block: BlockRef<'c, '_>,
        target: BlockRef<'c, 'a>,
        symbols: &mut SymbolTable<'c, '_>,
    ) {
        for op in block.operations() {
            self.lower_op(op, target, symbols);
        }
    }

    fn lower_op(
        &mut self,
        op: OperationRef<'c, '_>,
        target: BlockRef<'c, 'a>,
        symbols: &mut SymbolTable<'c, '_>,
    ) {
        match YzlOperationRef::of(&op) {
            Some(YzlOperationRef::Struct(item)) => {
                let name = item.sym_name().value();
                let fields = self.structs[name].clone();
                self.declare_struct(name, &fields, symbols);
            }
            // A table declaration says nothing yzr needs: `yzr.table` names
            // the relation and carries its row as the result type.
            Some(YzlOperationRef::Table(_)) => {}
            Some(YzlOperationRef::From(from)) => {
                let source = from.source().value();
                let Some(schema) = self.relation_schema(source) else {
                    self.error(op, format!("`{source}` has no row shape to scan"));
                    return;
                };

                let row = self.row_type(&schema, symbols);
                let scanned = target.append_operation(
                    yuzu_mlir::ods::yzr::table(
                        self.context,
                        row,
                        FlatSymbolRefAttribute::new(self.context, source),
                        op.location(),
                    )
                    .into(),
                );

                self.record_stage(op, first_result(scanned), schema);
            }
            Some(YzlOperationRef::Where(stage)) => {
                let Some((input, schema)) = self.input_stage(op) else {
                    return;
                };

                let region = self.lower_region(stage.body(), &schema, op.location());
                let filtered = target.append_operation(
                    yuzu_mlir::ods::yzr::filter(self.context, input, region, op.location()).into(),
                );

                self.record_stage(op, first_result(filtered), schema);
            }
            Some(YzlOperationRef::Output(_)) => {
                let Some((query, _)) = self.input_stage(op) else {
                    return;
                };

                target.append_operation(
                    yuzu_mlir::ods::yzr::output(self.context, query, op.location()).into(),
                );
            }
            _ => self.error(op, format!("`{}` is not lowered yet", op_name(op))),
        }
    }

    /// A stage's region, rebuilt with the input row's columns as block
    /// arguments — which is what makes column reference into SSA use-def.
    fn lower_region(
        &mut self,
        source: melior::ir::RegionRef<'c, '_>,
        schema: &Schema<'c>,
        location: Location<'c>,
    ) -> Region<'c> {
        let region = Region::new();
        let arguments: Vec<(Type<'c>, Location<'c>)> = schema
            .iter()
            .map(|(_, column)| (*column, location))
            .collect();
        let body = region.append_block(Block::new(&arguments));

        let Some(block) = source.first_block() else {
            return region;
        };

        let mut values: HashMap<usize, Value<'c, '_>> = HashMap::new();
        for op in block.operations() {
            self.lower_expression(op, body, &mut values);
        }

        region
    }

    /// An expression op, rebuilt against the values its operands became.
    fn lower_expression<'b>(
        &mut self,
        op: OperationRef<'c, '_>,
        body: BlockRef<'c, 'b>,
        values: &mut HashMap<usize, Value<'c, 'b>>,
    ) {
        match YzlOperationRef::of(&op) {
            Some(YzlOperationRef::Name(_)) => {
                let Some(index) = op.index_attribute("col") else {
                    self.error(op, "a name outside a column context is not lowered yet");
                    return;
                };

                let Ok(column) = body.argument(index) else {
                    self.error(op, format!("column {index} is not in the row"));
                    return;
                };

                values.insert(value_id(result(op)), column.into());
            }
            Some(YzlOperationRef::Yield(_)) => {
                let operands = self.mapped_operands(op, values);
                body.append_operation(
                    yuzu_mlir::ods::yzr::r#yield(self.context, &operands, op.location()).into(),
                );
            }
            // Everything else is a `yz` op, structurally unchanged: the
            // operands it was given, and the type inference stamped on it.
            None => {
                let operands = self.mapped_operands(op, values);
                let rebuilt = self.rebuild(op, &operands, body);
                if let Some(rebuilt) = rebuilt {
                    values.insert(value_id(result(op)), rebuilt);
                }
            }
            Some(_) => self.error(op, format!("`{}` is not lowered yet", op_name(op))),
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
        let Ok(existing) = op.result(0) else {
            return None;
        };

        let ty = op
            .attribute("ty")
            .ok()
            .and_then(|attribute| TypeAttribute::try_from(attribute).ok())
            .map(|attribute| attribute.value())
            .unwrap_or_else(|| Value::from(existing).r#type());

        let attributes: Vec<(Identifier<'c>, Attribute<'c>)> = (0..op.attribute_count())
            .filter_map(|index| op.attribute_at(index).ok())
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

        Some(first_result(body.append_operation(rebuilt)))
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

    /// The row a relation's rows have: a table's declared struct, or a
    /// binding's computed shape.
    fn relation_schema(&self, name: &str) -> Option<Schema<'c>> {
        let row = self.relations.get(name)?;
        self.structs.get(row).cloned()
    }

    /// The type standing for a row — declaring the shape when nothing has.
    fn row_type(&mut self, schema: &Schema<'c>, symbols: &mut SymbolTable<'c, '_>) -> Type<'c> {
        if let Some(name) = self.shapes.get(schema) {
            return StructType::new(self.context, name).into();
        }

        let declared = self
            .structs
            .iter()
            .find(|(_, fields)| *fields == schema)
            .map(|(name, _)| name.clone());
        let name = match declared {
            Some(name) => name,
            None => self.declare_struct("row", schema, symbols),
        };

        self.shapes.insert(schema.clone(), name.clone());
        StructType::new(self.context, &name).into()
    }

    /// Declares a struct in the lowered module, returning the name it got —
    /// the symbol table renames a collision, which is what interns a shape
    /// nobody declared.
    fn declare_struct(
        &self,
        name: &str,
        fields: &Schema<'c>,
        symbols: &mut SymbolTable<'c, '_>,
    ) -> String {
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
            .to_string()
    }

    fn input_stage(&mut self, op: OperationRef<'c, '_>) -> Option<(Value<'c, 'a>, Schema<'c>)> {
        let input = op.operand(0).ok()?;
        self.stages.get(&value_id(input)).cloned()
    }

    fn record_stage(&mut self, op: OperationRef<'c, '_>, value: Value<'c, 'a>, schema: Schema<'c>) {
        if let Ok(result) = op.result(0) {
            self.stages.insert(value_id(result.into()), (value, schema));
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

fn field_types<'c>(types: melior::ir::attribute::ArrayAttribute<'c>) -> Vec<Type<'c>> {
    types
        .elements()
        .filter_map(|element| TypeAttribute::try_from(element).ok())
        .map(|attribute| attribute.value())
        .collect()
}

fn result<'c, 'a>(op: OperationRef<'c, 'a>) -> Value<'c, 'a> {
    op.result(0).expect("the op has a result").into()
}

fn first_result<'c, 'a>(op: OperationRef<'c, 'a>) -> Value<'c, 'a> {
    op.result(0).expect("the built op has a result").into()
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

    /// A stage the lowering does not carry yet is an error, not a silent gap.
    #[test]
    fn reports_a_stage_that_is_not_lowered() {
        check_lowered(
            r#"
struct Row { a: int64 }
table t = Row

from t
|> limit 5
"#,
            expect![[r#"
                error: `yzl.limit` is not lowered yet
                 --> test.yz:5:1
                  |
                5 | from t
                  | ^
            "#]],
        );
    }
}
