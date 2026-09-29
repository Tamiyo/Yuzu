#include "YzDialect.h"

#include <cmath>
#include <limits>
#include <optional>

#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

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

mlir::LogicalResult StructOp::verify() {
  if (getNames().size() != getTypes().size())
    return emitOpError("has ") << getNames().size() << " field names and "
                               << getTypes().size() << " field types";
  return mlir::success();
}

mlir::Operation *YzDialect::materializeConstant(mlir::OpBuilder &builder,
                                                mlir::Attribute value,
                                                mlir::Type type,
                                                mlir::Location loc) {
  if (llvm::isa<BoolType>(type))
    if (auto boolean = llvm::dyn_cast<mlir::BoolAttr>(value))
      return ConstantBoolOp::create(builder, loc, type, boolean);
  if (llvm::isa<Int64Type>(type))
    if (auto integer = llvm::dyn_cast<mlir::IntegerAttr>(value))
      return ConstantIntOp::create(builder, loc, type, integer);
  if (llvm::isa<Float64Type>(type))
    if (auto real = llvm::dyn_cast<mlir::FloatAttr>(value))
      return ConstantFloatOp::create(builder, loc, type, real);
  if (llvm::isa<StrType>(type))
    if (auto text = llvm::dyn_cast<mlir::StringAttr>(value))
      return ConstantStrOp::create(builder, loc, type, text);
  if (llvm::isa<ListType>(type))
    if (auto values = llvm::dyn_cast<mlir::ArrayAttr>(value))
      return ConstantListOp::create(builder, loc, type, values);
  return nullptr;
}

mlir::OpFoldResult ConstantIntOp::fold(FoldAdaptor) { return getValueAttr(); }
mlir::OpFoldResult ConstantFloatOp::fold(FoldAdaptor) { return getValueAttr(); }
mlir::OpFoldResult ConstantBoolOp::fold(FoldAdaptor) { return getValueAttr(); }
mlir::OpFoldResult ConstantStrOp::fold(FoldAdaptor) { return getValueAttr(); }
mlir::OpFoldResult ConstantListOp::fold(FoldAdaptor) { return getValuesAttr(); }

// Each value is what the element type's constant op would hold.
mlir::LogicalResult ConstantListOp::verify() {
  mlir::Type element = llvm::cast<ListType>(getType()).getInner();
  for (auto [index, value] : llvm::enumerate(getValues())) {
    bool fits = (llvm::isa<Int64Type>(element) && llvm::isa<mlir::IntegerAttr>(value)) ||
                (llvm::isa<Float64Type>(element) && llvm::isa<mlir::FloatAttr>(value)) ||
                (llvm::isa<BoolType>(element) && llvm::isa<mlir::BoolAttr>(value)) ||
                (llvm::isa<StrType>(element) && llvm::isa<mlir::StringAttr>(value));
    if (!fits)
      return emitOpError("value ") << index << " is not a constant of " << element;
  }
  return mlir::success();
}

// Folding answers at compile time what the engine would answer at run time,
// so a fold that cannot be carried out exactly declines instead of guessing.
// The engine owns overflow and division by zero; a constant that disagreed
// with it would quietly change the query rather than fail. `floats` is null
// for an operation whose float result the engine rounds its own way.
static mlir::OpFoldResult
foldNumericBinary(mlir::Attribute lhs, mlir::Attribute rhs,
                  std::optional<int64_t> (*ints)(int64_t, int64_t),
                  double (*floats)(double, double)) {
  if (auto lhsInt = llvm::dyn_cast_if_present<mlir::IntegerAttr>(lhs))
    if (auto rhsInt = llvm::dyn_cast_if_present<mlir::IntegerAttr>(rhs)) {
      std::optional<int64_t> folded = ints(lhsInt.getInt(), rhsInt.getInt());
      if (!folded)
        return {};
      return mlir::IntegerAttr::get(lhsInt.getType(), *folded);
    }
  if (!floats)
    return {};
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

// The one division the two's complement range cannot answer: its result is
// one past the largest representable integer.
static bool isOverflowingDivision(int64_t lhs, int64_t rhs) {
  return lhs == std::numeric_limits<int64_t>::min() && rhs == -1;
}

mlir::OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  return foldNumericBinary(
      adaptor.getLhs(), adaptor.getRhs(),
      [](int64_t lhs, int64_t rhs) -> std::optional<int64_t> {
        int64_t result;
        if (llvm::AddOverflow(lhs, rhs, result))
          return std::nullopt;
        return result;
      },
      [](double lhs, double rhs) { return lhs + rhs; });
}

mlir::OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
  return foldNumericBinary(
      adaptor.getLhs(), adaptor.getRhs(),
      [](int64_t lhs, int64_t rhs) -> std::optional<int64_t> {
        int64_t result;
        if (llvm::SubOverflow(lhs, rhs, result))
          return std::nullopt;
        return result;
      },
      [](double lhs, double rhs) { return lhs - rhs; });
}

mlir::OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  return foldNumericBinary(
      adaptor.getLhs(), adaptor.getRhs(),
      [](int64_t lhs, int64_t rhs) -> std::optional<int64_t> {
        int64_t result;
        if (llvm::MulOverflow(lhs, rhs, result))
          return std::nullopt;
        return result;
      },
      [](double lhs, double rhs) { return lhs * rhs; });
}

mlir::OpFoldResult DivOp::fold(FoldAdaptor adaptor) {
  if (isZero(adaptor.getRhs()))
    return {};
  return foldNumericBinary(
      adaptor.getLhs(), adaptor.getRhs(),
      [](int64_t lhs, int64_t rhs) -> std::optional<int64_t> {
        if (isOverflowingDivision(lhs, rhs))
          return std::nullopt;
        return lhs / rhs;
      },
      [](double lhs, double rhs) { return lhs / rhs; });
}

mlir::OpFoldResult RemOp::fold(FoldAdaptor adaptor) {
  if (isZero(adaptor.getRhs()))
    return {};
  return foldNumericBinary(
      adaptor.getLhs(), adaptor.getRhs(),
      [](int64_t lhs, int64_t rhs) -> std::optional<int64_t> {
        if (isOverflowingDivision(lhs, rhs))
          return std::nullopt;
        return lhs % rhs;
      },
      [](double lhs, double rhs) { return std::fmod(lhs, rhs); });
}

// By squaring, so an exponent of any size takes at most 64 steps. A float
// power does not fold: `pow` is not correctly rounded, so a folded constant
// could differ from the engine's answer in its last bit.
mlir::OpFoldResult PowOp::fold(FoldAdaptor adaptor) {
  return foldNumericBinary(
      adaptor.getLhs(), adaptor.getRhs(),
      [](int64_t base, int64_t exponent) -> std::optional<int64_t> {
        if (exponent < 0)
          return std::nullopt;
        int64_t result = 1;
        while (exponent > 0) {
          if ((exponent & 1) && llvm::MulOverflow(result, base, result))
            return std::nullopt;
          exponent >>= 1;
          if (exponent > 0 && llvm::MulOverflow(base, base, base))
            return std::nullopt;
        }
        return result;
      },
      nullptr);
}

// A shift by a negative count or by the width or more is not defined the
// same way everywhere, so only a count the engine agrees on folds.
static bool isShiftCount(int64_t count) { return count >= 0 && count < 64; }

static mlir::OpFoldResult
foldIntegerBinary(mlir::Attribute lhs, mlir::Attribute rhs,
                  std::optional<int64_t> (*ints)(int64_t, int64_t)) {
  auto lhsInt = llvm::dyn_cast_if_present<mlir::IntegerAttr>(lhs);
  auto rhsInt = llvm::dyn_cast_if_present<mlir::IntegerAttr>(rhs);
  if (!lhsInt || !rhsInt)
    return {};
  std::optional<int64_t> folded = ints(lhsInt.getInt(), rhsInt.getInt());
  if (!folded)
    return {};
  return mlir::IntegerAttr::get(lhsInt.getType(), *folded);
}

mlir::OpFoldResult ShlOp::fold(FoldAdaptor adaptor) {
  return foldIntegerBinary(
      adaptor.getLhs(), adaptor.getRhs(),
      [](int64_t value, int64_t count) -> std::optional<int64_t> {
        if (!isShiftCount(count))
          return std::nullopt;
        auto shifted = static_cast<int64_t>(static_cast<uint64_t>(value)
                                            << count);
        // A bit shifted out, or into the sign, is overflow.
        if ((shifted >> count) != value)
          return std::nullopt;
        return shifted;
      });
}

mlir::OpFoldResult ShrOp::fold(FoldAdaptor adaptor) {
  return foldIntegerBinary(
      adaptor.getLhs(), adaptor.getRhs(),
      [](int64_t value, int64_t count) -> std::optional<int64_t> {
        if (!isShiftCount(count))
          return std::nullopt;
        return value >> count;
      });
}

mlir::OpFoldResult NegOp::fold(FoldAdaptor adaptor) {
  if (auto integer =
          llvm::dyn_cast_if_present<mlir::IntegerAttr>(adaptor.getValue())) {
    // The range is asymmetric, so the least integer has no negation.
    if (integer.getInt() == std::numeric_limits<int64_t>::min())
      return {};
    return mlir::IntegerAttr::get(integer.getType(), -integer.getInt());
  }
  if (auto real =
          llvm::dyn_cast_if_present<mlir::FloatAttr>(adaptor.getValue()))
    return mlir::FloatAttr::get(real.getType(), -real.getValueAsDouble());
  return {};
}

template <typename T>
static std::optional<bool> comparePredicate(llvm::StringRef predicate, T lhs,
                                            T rhs) {
  return llvm::StringSwitch<std::optional<bool>>(predicate)
      .Case("eq", lhs == rhs)
      .Case("ne", lhs != rhs)
      .Case("lt", lhs < rhs)
      .Case("le", lhs <= rhs)
      .Case("gt", lhs > rhs)
      .Case("ge", lhs >= rhs)
      .Default(std::nullopt);
}

mlir::OpFoldResult CmpOp::fold(FoldAdaptor adaptor) {
  // A bool is an i1 integer attribute whose `getInt` sign-extends `true` to
  // -1, so bools are read first, as false before true. Integers compare as
  // integers: above 2^53 a double stands for more than one of them, so
  // comparing through one answers a different question than the engine will.
  std::optional<bool> value;
  if (auto lhs = llvm::dyn_cast_if_present<mlir::BoolAttr>(adaptor.getLhs())) {
    auto rhs = llvm::dyn_cast_if_present<mlir::BoolAttr>(adaptor.getRhs());
    if (!rhs)
      return {};
    value = comparePredicate(getPredicate(), static_cast<int>(lhs.getValue()),
                             static_cast<int>(rhs.getValue()));
  } else if (auto lhs = llvm::dyn_cast_if_present<mlir::IntegerAttr>(
                 adaptor.getLhs())) {
    auto rhs = llvm::dyn_cast_if_present<mlir::IntegerAttr>(adaptor.getRhs());
    if (!rhs)
      return {};
    value = comparePredicate(getPredicate(), lhs.getInt(), rhs.getInt());
  } else if (auto lhs = llvm::dyn_cast_if_present<mlir::FloatAttr>(
                 adaptor.getLhs())) {
    auto rhs = llvm::dyn_cast_if_present<mlir::FloatAttr>(adaptor.getRhs());
    if (!rhs)
      return {};
    value = comparePredicate(getPredicate(), lhs.getValueAsDouble(),
                             rhs.getValueAsDouble());
  } else {
    return {};
  }
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
