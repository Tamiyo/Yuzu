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

mlir::Operation *YzirDialect::materializeConstant(mlir::OpBuilder &builder,
                                                  mlir::Attribute value,
                                                  mlir::Type type,
                                                  mlir::Location loc) {
  auto integer = llvm::dyn_cast<mlir::IntegerAttr>(value);
  if (!integer || !llvm::isa<Int64Type>(type))
    return nullptr;
  return builder.create<ConstOp>(loc, type, integer);
}

mlir::OpFoldResult ConstOp::fold(FoldAdaptor) { return getValueAttr(); }

static mlir::OpFoldResult foldBinary(mlir::Attribute lhs, mlir::Attribute rhs,
                                     int64_t (*apply)(int64_t, int64_t)) {
  auto lhsInt = llvm::dyn_cast_if_present<mlir::IntegerAttr>(lhs);
  auto rhsInt = llvm::dyn_cast_if_present<mlir::IntegerAttr>(rhs);
  if (!lhsInt || !rhsInt)
    return {};
  return mlir::IntegerAttr::get(lhsInt.getType(),
                                apply(lhsInt.getInt(), rhsInt.getInt()));
}

mlir::OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  return foldBinary(adaptor.getLhs(), adaptor.getRhs(),
                    [](int64_t lhs, int64_t rhs) { return lhs + rhs; });
}

mlir::OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  return foldBinary(adaptor.getLhs(), adaptor.getRhs(),
                    [](int64_t lhs, int64_t rhs) { return lhs * rhs; });
}

} // namespace yuzu::yzir
