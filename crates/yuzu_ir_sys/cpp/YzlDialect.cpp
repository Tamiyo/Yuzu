#include "YzlDialect.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

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

} // namespace yuzu::yzl
