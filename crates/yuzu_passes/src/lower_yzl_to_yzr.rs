//! The yzl → yzr conversion, done as one rebuild. Names become block
//! arguments, declarations move to the dialect that owns their type, and
//! stage ops become `yzr`, after which `yzl` is illegal. A new module is
//! built rather than rewriting in place, so no operand is remapped under its
//! own use.

use std::collections::HashMap;

use melior::Context;
use melior::ir::attribute::StringAttribute;
use melior::ir::operation::{OperationLike, OperationRef};
use melior::ir::{BlockRef, Location, Module, Type, Value};
use yuzu_mlir::SymbolTable;
use yuzu_mlir::diagnostics::emit_error;
use yuzu_mlir::ext::{ArrayAttributeExt, BlockExt, OperationExt, ValueExt, ValueId};
use yuzu_mlir::ops::yzl::StructOp;

mod expr;
mod region;
mod rel;
mod row;

/// Expects a resolved, inferred module. Returns the `yz` + `yzr` module it
/// lowers to; anything it cannot lower is reported and left out.
pub fn lower_yzl_to_yzr<'c>(context: &'c Context, module: &Module<'c>) -> Module<'c> {
    let lowered = Module::new(Location::unknown(context));
    {
        let target = lowered.body();
        let mut symbols = SymbolTable::new(&lowered);
        let mut lowering = YzlToYzr {
            context,
            stages: HashMap::new(),
            shapes: HashMap::new(),
            bindings: HashMap::new(),
        };

        let source = SymbolTable::new(module);
        lowering.intern_declared_shapes(module.body());
        lowering.convert_block(module.body(), target, &source, &mut symbols);
    }

    lowered
}

/// A row: the columns flowing out of a stage, in order.
type Row<'c> = Vec<(&'c str, Type<'c>)>;

/// What a lowered region yields: the values its body computed, or the whole
/// row with those values substituted into the columns they replace.
enum Yielded<'k> {
    Body,
    Substituted(&'k [usize]),
}

struct YzlToYzr<'c, 'a> {
    context: &'c Context,
    /// What each yzl stage value became, and the row it carries.
    stages: HashMap<ValueId, (Value<'c, 'a>, Row<'c>)>,
    /// The symbol declaring each distinct row shape, so a shape nobody
    /// declared is declared once.
    shapes: HashMap<Row<'c>, &'c str>,
    /// The rows each `let` name stands for, already produced.
    bindings: HashMap<&'c str, (Value<'c, 'a>, Row<'c>)>,
}

impl<'c, 'a> YzlToYzr<'c, 'a> {
    fn convert_block(
        &mut self,
        block: BlockRef<'c, '_>,
        target: BlockRef<'c, 'a>,
        source: &SymbolTable<'c, '_>,
        symbols: &mut SymbolTable<'c, '_>,
    ) {
        for op in block.operations() {
            self.convert_op(op, target, source, symbols);
        }
    }

    fn input_stage(&mut self, op: OperationRef<'c, '_>) -> Option<(Value<'c, 'a>, Row<'c>)> {
        let input = op.try_first_operand()?;
        self.stages.get(&input.id()).cloned()
    }

    fn record_stage(&mut self, op: OperationRef<'c, '_>, value: Value<'c, 'a>, schema: Row<'c>) {
        if let Some(result) = op.try_first_result() {
            self.stages.insert(result.id(), (value, schema));
        }
    }

    fn intern(&self, name: &str) -> &'c str {
        StringAttribute::new(self.context, name).value()
    }

    fn report(&self, op: OperationRef<'c, '_>, message: &str) {
        emit_error(op.location(), message);
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
fn struct_fields<'c>(item: &StructOp<'c, '_>) -> Row<'c> {
    item.names()
        .strings()
        .into_iter()
        .zip(item.types().types())
        .collect()
}
