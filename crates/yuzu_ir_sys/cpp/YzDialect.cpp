#include "YzDialect.h"

#include <cmath>

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"

#include "YzDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "YzTypes.cpp.inc"

#define GET_OP_CLASSES
#include "YzOps.cpp.inc"

namespace yuzu::yz {

void YzDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "YzTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "YzOps.cpp.inc"
      >();
}

mlir::Operation *YzDialect::materializeConstant(mlir::OpBuilder &builder,
                                                mlir::Attribute value,
                                                mlir::Type type,
                                                mlir::Location loc) {
  if (llvm::isa<BoolType>(type))
    if (auto boolean = llvm::dyn_cast<mlir::BoolAttr>(value))
      return builder.create<ConstantBoolOp>(loc, type, boolean);
  if (llvm::isa<Int64Type>(type))
    if (auto integer = llvm::dyn_cast<mlir::IntegerAttr>(value))
      return builder.create<ConstantIntOp>(loc, type, integer);
  if (llvm::isa<Float64Type>(type))
    if (auto real = llvm::dyn_cast<mlir::FloatAttr>(value))
      return builder.create<ConstantFloatOp>(loc, type, real);
  if (llvm::isa<StrType>(type))
    if (auto text = llvm::dyn_cast<mlir::StringAttr>(value))
      return builder.create<ConstantStrOp>(loc, type, text);
  return nullptr;
}

mlir::OpFoldResult ConstantIntOp::fold(FoldAdaptor) { return getValueAttr(); }
mlir::OpFoldResult ConstantFloatOp::fold(FoldAdaptor) { return getValueAttr(); }
mlir::OpFoldResult ConstantBoolOp::fold(FoldAdaptor) { return getValueAttr(); }
mlir::OpFoldResult ConstantStrOp::fold(FoldAdaptor) { return getValueAttr(); }

static mlir::OpFoldResult foldNumericBinary(mlir::Attribute lhs,
                                            mlir::Attribute rhs,
                                            int64_t (*ints)(int64_t, int64_t),
                                            double (*floats)(double, double)) {
  if (auto lhsInt = llvm::dyn_cast_if_present<mlir::IntegerAttr>(lhs))
    if (auto rhsInt = llvm::dyn_cast_if_present<mlir::IntegerAttr>(rhs))
      return mlir::IntegerAttr::get(lhsInt.getType(),
                                    ints(lhsInt.getInt(), rhsInt.getInt()));
  if (auto lhsFloat = llvm::dyn_cast_if_present<mlir::FloatAttr>(lhs))
    if (auto rhsFloat = llvm::dyn_cast_if_present<mlir::FloatAttr>(rhs))
      return mlir::FloatAttr::get(
          lhsFloat.getType(),
          floats(lhsFloat.getValueAsDouble(), rhsFloat.getValueAsDouble()));
  return {};
}

// A zero divisor never folds, for either kind: the runtime owns that
// behavior, and folding it away would change what the query does.
static bool isZero(mlir::Attribute value) {
  if (auto integer = llvm::dyn_cast_if_present<mlir::IntegerAttr>(value))
    return integer.getInt() == 0;
  if (auto real = llvm::dyn_cast_if_present<mlir::FloatAttr>(value))
    return real.getValueAsDouble() == 0.0;
  return false;
}

mlir::OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  return foldNumericBinary(
      adaptor.getLhs(), adaptor.getRhs(),
      [](int64_t lhs, int64_t rhs) { return lhs + rhs; },
      [](double lhs, double rhs) { return lhs + rhs; });
}

mlir::OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
  return foldNumericBinary(
      adaptor.getLhs(), adaptor.getRhs(),
      [](int64_t lhs, int64_t rhs) { return lhs - rhs; },
      [](double lhs, double rhs) { return lhs - rhs; });
}

mlir::OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  return foldNumericBinary(
      adaptor.getLhs(), adaptor.getRhs(),
      [](int64_t lhs, int64_t rhs) { return lhs * rhs; },
      [](double lhs, double rhs) { return lhs * rhs; });
}

mlir::OpFoldResult DivOp::fold(FoldAdaptor adaptor) {
  if (isZero(adaptor.getRhs()))
    return {};
  return foldNumericBinary(
      adaptor.getLhs(), adaptor.getRhs(),
      [](int64_t lhs, int64_t rhs) { return lhs / rhs; },
      [](double lhs, double rhs) { return lhs / rhs; });
}

mlir::OpFoldResult RemOp::fold(FoldAdaptor adaptor) {
  if (isZero(adaptor.getRhs()))
    return {};
  return foldNumericBinary(
      adaptor.getLhs(), adaptor.getRhs(),
      [](int64_t lhs, int64_t rhs) { return lhs % rhs; },
      [](double lhs, double rhs) { return std::fmod(lhs, rhs); });
}

mlir::OpFoldResult NegOp::fold(FoldAdaptor adaptor) {
  if (auto integer =
          llvm::dyn_cast_if_present<mlir::IntegerAttr>(adaptor.getValue()))
    return mlir::IntegerAttr::get(integer.getType(), -integer.getInt());
  if (auto real =
          llvm::dyn_cast_if_present<mlir::FloatAttr>(adaptor.getValue()))
    return mlir::FloatAttr::get(real.getType(), -real.getValueAsDouble());
  return {};
}

mlir::OpFoldResult CmpOp::fold(FoldAdaptor adaptor) {
  double left;
  double right;
  if (auto lhs =
          llvm::dyn_cast_if_present<mlir::IntegerAttr>(adaptor.getLhs())) {
    auto rhs = llvm::dyn_cast_if_present<mlir::IntegerAttr>(adaptor.getRhs());
    if (!rhs)
      return {};
    left = static_cast<double>(lhs.getInt());
    right = static_cast<double>(rhs.getInt());
  } else if (auto lhs =
                 llvm::dyn_cast_if_present<mlir::FloatAttr>(adaptor.getLhs())) {
    auto rhs = llvm::dyn_cast_if_present<mlir::FloatAttr>(adaptor.getRhs());
    if (!rhs)
      return {};
    left = lhs.getValueAsDouble();
    right = rhs.getValueAsDouble();
  } else {
    return {};
  }
  auto value = llvm::StringSwitch<std::optional<bool>>(getPredicate())
                   .Case("eq", left == right)
                   .Case("ne", left != right)
                   .Case("lt", left < right)
                   .Case("le", left <= right)
                   .Case("gt", left > right)
                   .Case("ge", left >= right)
                   .Default(std::nullopt);
  if (!value)
    return {};
  return mlir::BoolAttr::get(getContext(), *value);
}

static mlir::OpFoldResult foldBoolBinary(mlir::MLIRContext *context,
                                         mlir::Attribute lhs,
                                         mlir::Attribute rhs,
                                         bool (*apply)(bool, bool)) {
  auto lhsBool = llvm::dyn_cast_if_present<mlir::BoolAttr>(lhs);
  auto rhsBool = llvm::dyn_cast_if_present<mlir::BoolAttr>(rhs);
  if (!lhsBool || !rhsBool)
    return {};
  return mlir::BoolAttr::get(context,
                             apply(lhsBool.getValue(), rhsBool.getValue()));
}

mlir::OpFoldResult AndOp::fold(FoldAdaptor adaptor) {
  return foldBoolBinary(getContext(), adaptor.getLhs(), adaptor.getRhs(),
                        [](bool lhs, bool rhs) { return lhs && rhs; });
}

mlir::OpFoldResult OrOp::fold(FoldAdaptor adaptor) {
  return foldBoolBinary(getContext(), adaptor.getLhs(), adaptor.getRhs(),
                        [](bool lhs, bool rhs) { return lhs || rhs; });
}

mlir::OpFoldResult NotOp::fold(FoldAdaptor adaptor) {
  auto value = llvm::dyn_cast_if_present<mlir::BoolAttr>(adaptor.getValue());
  if (!value)
    return {};
  return mlir::BoolAttr::get(getContext(), !value.getValue());
}

} // namespace yuzu::yz
