#include "YzrDialect.h"

#include "YzDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include "YzrDialect.cpp.inc"

#define GET_OP_CLASSES
#include "YzrOps.cpp.inc"

namespace yuzu::yzr {

void YzrDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "YzrOps.cpp.inc"
      >();
}

// A column that is not a key reaches the yield only through a `yzr.agg`:
// each path from its block argument ends at one before the yield.
mlir::LogicalResult AggregateOp::verify() {
  llvm::SmallDenseSet<int64_t> keys(getKeys().begin(), getKeys().end());
  for (mlir::BlockArgument column : getBody().front().getArguments()) {
    if (keys.contains(column.getArgNumber()))
      continue;
    llvm::SmallVector<mlir::Value> reached{column};
    llvm::SmallPtrSet<mlir::Operation *, 8> seen;
    while (!reached.empty()) {
      for (mlir::Operation *user : reached.pop_back_val().getUsers()) {
        if (llvm::isa<AggOp>(user))
          continue;
        if (llvm::isa<YieldOp>(user))
          return emitOpError("yields column ")
                 << column.getArgNumber()
                 << " outside an aggregate, and it is not a key";
        if (seen.insert(user).second)
          llvm::append_range(reached, user->getResults());
      }
    }
  }
  return mlir::success();
}

} // namespace yuzu::yzr
