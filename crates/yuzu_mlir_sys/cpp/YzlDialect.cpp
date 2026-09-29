#include "YzlDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "YzlDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "YzlTypes.cpp.inc"

#define GET_OP_CLASSES
#include "YzlOps.cpp.inc"

namespace yuzu::yzl {

void YzlDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "YzlTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "YzlOps.cpp.inc"
      >();
}

mlir::LogicalResult StructOp::verify() {
  if (getNames().size() != getTypes().size())
    return emitOpError("has ") << getNames().size() << " field names and "
                               << getTypes().size() << " field types";
  return mlir::success();
}

mlir::LogicalResult FnOp::verify() {
  size_t inputs = getSignature().getInputs().size();
  if (getParams().size() != inputs)
    return emitOpError("names ") << getParams().size()
                                 << " parameters, but its signature takes "
                                 << inputs;
  mlir::ArrayAttr boundParams = getBoundParamsAttr();
  mlir::ArrayAttr boundTraits = getBoundTraitsAttr();
  size_t bounded = boundParams ? boundParams.size() : 0;
  size_t traits = boundTraits ? boundTraits.size() : 0;
  if (bounded != traits)
    return emitOpError("bounds ") << bounded << " parameters with " << traits
                                  << " traits";
  return mlir::success();
}

} // namespace yuzu::yzl

// Promotion of places to values, for MLIR's `mem2reg`. This follows the
// upstream `memref.alloca`, `memref.load` and `memref.store`. A place holds a
// value of any type: inference has not run, so a load and the value stored
// need not have the same type yet.

namespace yuzu::yzl {

llvm::SmallVector<mlir::MemorySlot> LocalOp::getPromotableSlots() {
  return {mlir::MemorySlot{getPlace(), getPlace().getType().getElement()}};
}

// Only a load before any store reads this. The frontend stores every place
// where it declares it, so the value is removed again unused.
mlir::Value LocalOp::getDefaultValue(const mlir::MemorySlot &slot,
                                     mlir::OpBuilder &builder) {
  return MissingOp::create(builder, getLoc(), slot.elemType);
}

void LocalOp::handleBlockArgument(const mlir::MemorySlot &,
                                  mlir::BlockArgument, mlir::OpBuilder &) {}

std::optional<mlir::PromotableAllocationOpInterface>
LocalOp::handlePromotionComplete(const mlir::MemorySlot &,
                                 mlir::Value defaultValue, mlir::OpBuilder &) {
  if (defaultValue && defaultValue.use_empty())
    defaultValue.getDefiningOp()->erase();
  erase();
  return std::nullopt;
}

bool LoadOp::loadsFrom(const mlir::MemorySlot &slot) {
  return getPlace() == slot.ptr;
}

bool LoadOp::storesTo(const mlir::MemorySlot &) { return false; }

mlir::Value LoadOp::getStored(const mlir::MemorySlot &, mlir::OpBuilder &,
                              mlir::Value, const mlir::DataLayout &) {
  llvm_unreachable("a load does not store");
}

bool LoadOp::canUsesBeRemoved(
    const mlir::MemorySlot &slot,
    const llvm::SmallPtrSetImpl<mlir::OpOperand *> &blockingUses,
    llvm::SmallVectorImpl<mlir::OpOperand *> &, const mlir::DataLayout &) {
  return blockingUses.size() == 1 &&
         (*blockingUses.begin())->get() == slot.ptr && getPlace() == slot.ptr;
}

mlir::DeletionKind LoadOp::removeBlockingUses(
    const mlir::MemorySlot &, const llvm::SmallPtrSetImpl<mlir::OpOperand *> &,
    mlir::OpBuilder &, mlir::Value reachingDefinition,
    const mlir::DataLayout &) {
  getResult().replaceAllUsesWith(reachingDefinition);
  return mlir::DeletionKind::Delete;
}

bool StoreOp::loadsFrom(const mlir::MemorySlot &) { return false; }

bool StoreOp::storesTo(const mlir::MemorySlot &slot) {
  return getPlace() == slot.ptr;
}

mlir::Value StoreOp::getStored(const mlir::MemorySlot &, mlir::OpBuilder &,
                               mlir::Value, const mlir::DataLayout &) {
  return getValue();
}

bool StoreOp::canUsesBeRemoved(
    const mlir::MemorySlot &slot,
    const llvm::SmallPtrSetImpl<mlir::OpOperand *> &blockingUses,
    llvm::SmallVectorImpl<mlir::OpOperand *> &, const mlir::DataLayout &) {
  // A store of the place itself would let the place escape.
  return blockingUses.size() == 1 &&
         (*blockingUses.begin())->get() == slot.ptr && getPlace() == slot.ptr &&
         getValue() != slot.ptr;
}

mlir::DeletionKind StoreOp::removeBlockingUses(
    const mlir::MemorySlot &, const llvm::SmallPtrSetImpl<mlir::OpOperand *> &,
    mlir::OpBuilder &, mlir::Value, const mlir::DataLayout &) {
  return mlir::DeletionKind::Delete;
}

} // namespace yuzu::yzl
