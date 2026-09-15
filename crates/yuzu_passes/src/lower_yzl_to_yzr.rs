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
use melior::ir::attribute::TypeAttribute;
use melior::ir::operation::{OperationLike, OperationRef};
use melior::ir::{BlockRef, Location, Module, Type, Value};
use yuzu_mlir::SymbolTable;
use yuzu_mlir::ext::{ArrayAttributeExt, BlockExt, OperationExt, ValueExt};

/// A row: the columns flowing out of a stage, in order.
type Schema<'c> = Vec<(&'c str, Type<'c>)>;

/// What a lowered region yields: the values its body computed, or the whole
/// row with those values substituted into the columns they replace.
enum Yielded<'k> {
    Body,
    Row(&'k [usize]),
}

struct YzlToYzr<'c, 'a> {
    context: &'c Context,
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
        lowering.lower_block(module.body(), target, &source, &mut symbols);
    }

    lowered
}
impl<'c, 'a> YzlToYzr<'c, 'a> {
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

    fn input_stage(&mut self, op: OperationRef<'c, '_>) -> Option<(Value<'c, 'a>, Schema<'c>)> {
        let input = op.try_first_operand()?;
        self.stages.get(&input.id()).cloned()
    }

    fn record_stage(&mut self, op: OperationRef<'c, '_>, value: Value<'c, 'a>, schema: Schema<'c>) {
        if let Some(result) = op.try_first_result() {
            self.stages.insert(result.id(), (value, schema));
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
fn struct_fields<'c>(item: &yuzu_mlir::ops::yzl::StructOp<'c, '_>) -> Schema<'c> {
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

mod expr;
mod region;
mod row;
mod stage;
