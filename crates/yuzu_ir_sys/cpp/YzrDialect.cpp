#include "YzrDialect.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "YzrDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "YzrTypes.cpp.inc"

#define GET_OP_CLASSES
#include "YzrOps.cpp.inc"

namespace yuzu::yzr {

void YzrDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "YzrTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "YzrOps.cpp.inc"
      >();
}

// The grouping rule. Two taints propagate forward through the straight-line
// region: `row` marks values derived from a non-key column that no aggregate
// has consumed yet, and `measure` marks values derived from an aggregate's
// result. A `yzr.agg` operand must not be measure-tainted (that would nest
// aggregations), and a yielded value must not be row-tainted (that would leak
// an ungrouped column) — no matter how many intermediate ops launder it.
mlir::LogicalResult AggregateOp::verify() {
  mlir::Block &block = getBody().front();
  llvm::DenseSet<mlir::Value> row;
  llvm::DenseSet<mlir::Value> measure;

  for (auto [index, argument] : llvm::enumerate(block.getArguments()))
    if (!llvm::is_contained(getKeys(), static_cast<int64_t>(index)))
      row.insert(argument);

  for (mlir::Operation &op : block.without_terminator()) {
    if (auto agg = llvm::dyn_cast<AggOp>(op)) {
      if (measure.contains(agg.getValue()))
        return agg.emitOpError("aggregates a value that is already a measure");
      measure.insert(agg.getResult());
      continue;
    }
    bool fromRow = llvm::any_of(op.getOperands(), [&](mlir::Value value) {
      return row.contains(value);
    });
    bool fromMeasure = llvm::any_of(op.getOperands(), [&](mlir::Value value) {
      return measure.contains(value);
    });
    for (mlir::Value result : op.getResults()) {
      if (fromRow)
        row.insert(result);
      if (fromMeasure)
        measure.insert(result);
    }
  }

  if (block.empty() || !llvm::isa<YieldOp>(&block.back()))
    return emitOpError("expects its body to end in a yzr.yield");
  auto yield = llvm::cast<YieldOp>(&block.back());
  for (mlir::Value value : yield.getValues())
    if (row.contains(value))
      return yield.emitOpError(
          "yields a value derived from a column that is neither grouped nor "
          "aggregated");
  return mlir::success();
}

} // namespace yuzu::yzr
