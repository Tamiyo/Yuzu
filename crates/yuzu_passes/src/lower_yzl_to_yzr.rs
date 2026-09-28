//! Expects a resolved, inferred module and rewrites it in place into the
//! `yz` + `yzr` module it lowers to; anything it cannot lower is reported
//! and left out. Each yzr op is placed before the yzl op it comes from, a
//! stage's expressions move into its new region, and the yzl ops are
//! erased at the end, once nothing reads them.

use melior::ir::attribute::StringAttribute;
use melior::ir::operation::{Operation, OperationLike, OperationRef};
use melior::ir::{BlockLike, BlockRef, Location, Module, Type, Value};
use melior::{Context, IrRewriter};
use rustc_hash::FxHashMap;
use yuzu_mlir::SymbolTable;
use yuzu_mlir::diagnostics::emit_error;
use yuzu_mlir::ir::attribute::array::ArrayAttributeExt;
use yuzu_mlir::ir::block::BlockExt;
use yuzu_mlir::ir::operation::{OperationCast, OperationExt};
use yuzu_mlir::ir::value::{ValueExt, ValueId};
use yuzu_mlir::ops::yzl::StructOp;

mod expr;
mod region;
mod rel;
mod row;

pub fn lower_yzl_to_yzr<'c>(context: &'c Context, module: &mut Module<'c>) {
    let module = &*module;
    let body = module.body();
    let ops: Vec<OperationRef<'c, '_>> = body.operations().collect();
    let Some(&first) = ops.first() else {
        return;
    };

    let mut symbols = SymbolTable::new(module);
    let mut lowering = YzlToYzr {
        context,
        body,
        anchor: first,
        stages: FxHashMap::default(),
        shapes: FxHashMap::default(),
        declared: FxHashMap::default(),
        bindings: FxHashMap::default(),
        externals: FxHashMap::default(),
    };

    lowering.intern_declared_shapes(body);
    lowering.record_externals(body);
    for op in ops {
        lowering.anchor = op;
        lowering.convert_op(op, &mut symbols);
    }

    drop(symbols);
    erase_yzl(context, body);
}

/// The yzl ops left at the top level, last first, so each is erased after
/// the ops that use it.
fn erase_yzl<'c>(context: &'c Context, body: BlockRef<'c, '_>) {
    let rewriter = IrRewriter::new(context);
    let rewriter = rewriter.as_rewriter_base();
    let yzl: Vec<OperationRef<'c, '_>> = body
        .operations()
        .filter(|op| op.as_yzl().is_some())
        .collect();
    for op in yzl.into_iter().rev() {
        rewriter.erase_op(op);
    }
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
    body: BlockRef<'c, 'a>,
    /// The top-level yzl op being converted; what it becomes is placed
    /// before it.
    anchor: OperationRef<'c, 'a>,
    stages: FxHashMap<ValueId, (Value<'c, 'a>, Row<'c>)>,
    /// The struct declaring each row shape; one nobody declared is declared
    /// once.
    shapes: FxHashMap<Row<'c>, &'c str>,
    /// The fields of each struct the program declared, by its name.
    declared: FxHashMap<&'c str, Row<'c>>,
    bindings: FxHashMap<&'c str, (Value<'c, 'a>, Row<'c>)>,
    /// The engine's name for each external function, by its symbol.
    externals: FxHashMap<&'c str, &'c str>,
}

impl<'c, 'a> YzlToYzr<'c, 'a> {
    fn insert(&self, op: Operation<'c>) -> OperationRef<'c, 'a> {
        self.body.insert_operation_before(self.anchor, op)
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

    fn report_at(&self, location: Location<'c>, message: &str) {
        emit_error(location, message);
    }
}

fn op_name(op: OperationRef<'_, '_>) -> String {
    op.name()
        .as_string_ref()
        .as_str()
        .unwrap_or("<non-utf8>")
        .to_string()
}

fn struct_fields<'c>(item: StructOp<'c, '_>) -> Row<'c> {
    item.names().strings().zip(item.types().types()).collect()
}
