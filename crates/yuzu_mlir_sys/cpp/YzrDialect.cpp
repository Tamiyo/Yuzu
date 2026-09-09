#include "YzrDialect.h"

#include "YzDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
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

} // namespace yuzu::yzr
