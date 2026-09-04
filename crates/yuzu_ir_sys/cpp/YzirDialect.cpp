#include "YzirDialect.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "YzirDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "YzirTypes.cpp.inc"

#define GET_OP_CLASSES
#include "YzirOps.cpp.inc"

namespace yuzu::yzir {

void YzirDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "YzirTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "YzirOps.cpp.inc"
      >();
}

} // namespace yuzu::yzir
