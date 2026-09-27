//! Expects a resolved, inferred module and builds the `yz` + `yzr` module
//! it lowers to; anything it cannot lower is reported and left out. A new
//! module is built rather than rewritten in place, so no operand is
//! remapped under its own use.

use melior::Context;
use melior::ir::attribute::StringAttribute;
use melior::ir::operation::{OperationLike, OperationRef};
use melior::ir::{BlockRef, Location, Module, Type, Value};
use rustc_hash::FxHashMap;
use yuzu_mlir::SymbolTable;
use yuzu_mlir::diagnostics::emit_error;
use yuzu_mlir::ir::attribute::array::ArrayAttributeExt;
use yuzu_mlir::ir::block::BlockExt;
use yuzu_mlir::ir::operation::OperationExt;
use yuzu_mlir::ir::value::{ValueExt, ValueId};
use yuzu_mlir::ops::yzl::StructOp;

mod expr;
mod region;
mod rel;
mod row;

pub fn lower_yzl_to_yzr<'c>(context: &'c Context, module: &Module<'c>) -> Module<'c> {
    let lowered = Module::new(Location::unknown(context));
    {
        let target = lowered.body();
        let mut symbols = SymbolTable::new(&lowered);
        let mut lowering = YzlToYzr {
            context,
            stages: FxHashMap::default(),
            shapes: FxHashMap::default(),
            bindings: FxHashMap::default(),
            externals: FxHashMap::default(),
        };

        let source = SymbolTable::new(module);
        lowering.intern_declared_shapes(module.body());
        lowering.record_externals(module.body());
        lowering.convert_block(module.body(), target, &source, &mut symbols);
    }

    lowered
}

type Row<'c> = Vec<(&'c str, Type<'c>)>;

/// What a region yields: what its body computed, or the whole row with
/// those values substituted in.
enum Yielded<'k> {
    Body,
    Substituted(&'k [usize]),
}

struct YzlToYzr<'c, 'a> {
    context: &'c Context,
    stages: FxHashMap<ValueId, (Value<'c, 'a>, Row<'c>)>,
    /// The struct declaring each row shape; one nobody declared is declared
    /// once.
    shapes: FxHashMap<Row<'c>, &'c str>,
    bindings: FxHashMap<&'c str, (Value<'c, 'a>, Row<'c>)>,
    /// The engine's name for each external function, by its symbol.
    externals: FxHashMap<&'c str, &'c str>,
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

fn struct_fields<'c>(item: &StructOp<'c, '_>) -> Row<'c> {
    item.names()
        .strings()
        .into_iter()
        .zip(item.types().types())
        .collect()
}
