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

// !yz.struct<@Row> — the declared struct the symbol names.
mlir::Type StructType::parse(mlir::AsmParser &parser) {
  mlir::StringAttr name;
  if (parser.parseLess() || parser.parseSymbolName(name) ||
      parser.parseGreater())
    return {};
  return StructType::get(parser.getContext(),
                         mlir::FlatSymbolRefAttr::get(name));
}

void StructType::print(mlir::AsmPrinter &printer) const {
  printer << "<" << getName() << ">";
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

// Folding answers at compile time what the engine would answer at run time,
// so a fold that cannot be carried out exactly declines instead of guessing.
// The engine owns overflow and division by zero; a constant that disagreed
// with it would quietly change the query rather than fail.
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

namespace {

// `(x + c1) + c2` computes the same as `x + (c1 + c2)` only while no
// intermediate overflows differently, and this dialect leaves overflow to the
// engine rather than deciding it. Two constants of the same sign are safe:
// the intermediate sum lies between `x` and the final one, so it overflows
// only when the final one does. Mixed signs are not, since `x + 1 - 1` can
// overflow at the first step and not at all when reassociated.
struct ReassociateAdd : public mlir::OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(AddOp op, mlir::PatternRewriter &rewriter) const override {
    auto outer = op.getRhs().getDefiningOp<ConstantIntOp>();
    if (!outer)
      return mlir::failure();

    auto inner = op.getLhs().getDefiningOp<AddOp>();
    if (!inner)
      return mlir::failure();

    auto held = inner.getRhs().getDefiningOp<ConstantIntOp>();
    if (!held)
      return mlir::failure();

    int64_t left = static_cast<int64_t>(held.getValue());
    int64_t right = static_cast<int64_t>(outer.getValue());
    if ((left < 0) != (right < 0))
      return mlir::failure();

    int64_t total;
    if (llvm::AddOverflow(left, right, total))
      return mlir::failure();

    auto folded = rewriter.create<ConstantIntOp>(
        op.getLoc(), outer.getType(),
        mlir::IntegerAttr::get(held.getValueAttr().getType(), total));
    rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), inner.getLhs(),
                                       folded);
    return mlir::success();
  }
};

} // namespace

void AddOp::getCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
                                        mlir::MLIRContext *context) {
  patterns.add<ReassociateAdd>(context);
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

mlir::OpFoldResult PowOp::fold(FoldAdaptor adaptor) {
  return foldNumericBinary(
      adaptor.getLhs(), adaptor.getRhs(),
      [](int64_t base, int64_t exponent) -> std::optional<int64_t> {
        if (exponent < 0)
          return std::nullopt;
        int64_t result = 1;
        for (int64_t step = 0; step < exponent; ++step)
          if (llvm::MulOverflow(result, base, result))
            return std::nullopt;
        return result;
      },
      [](double base, double exponent) { return std::pow(base, exponent); });
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

namespace {

// Whether two constants hold the same value, as the engine compares them:
// -0.0 equals 0.0, and NaN equals nothing. None when the kinds differ.
std::optional<bool> sameValue(mlir::Attribute lhs, mlir::Attribute rhs) {
  if (auto left = llvm::dyn_cast<mlir::IntegerAttr>(lhs))
    if (auto right = llvm::dyn_cast<mlir::IntegerAttr>(rhs))
      return left.getInt() == right.getInt();
  if (auto left = llvm::dyn_cast<mlir::FloatAttr>(lhs))
    if (auto right = llvm::dyn_cast<mlir::FloatAttr>(rhs))
      return left.getValueAsDouble() == right.getValueAsDouble();
  if (auto left = llvm::dyn_cast<mlir::BoolAttr>(lhs))
    if (auto right = llvm::dyn_cast<mlir::BoolAttr>(rhs))
      return left.getValue() == right.getValue();
  if (auto left = llvm::dyn_cast<mlir::StringAttr>(lhs))
    if (auto right = llvm::dyn_cast<mlir::StringAttr>(rhs))
      return left.getValue() == right.getValue();
  return std::nullopt;
}

// `x in [a, b]` with a constant `x` is decided once an element equals it, or
// once every element is a constant that does not.
struct FoldMembership : public mlir::OpRewritePattern<InOp> {
  using OpRewritePattern<InOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(InOp op, mlir::PatternRewriter &rewriter) const override {
    mlir::Attribute value;
    if (!mlir::matchPattern(op.getValue(), mlir::m_Constant(&value)))
      return mlir::failure();

    auto list = op.getList().getDefiningOp<ListOp>();
    if (!list)
      return mlir::failure();

    bool decided = true;
    for (mlir::Value element : list.getElements()) {
      mlir::Attribute constant;
      if (!mlir::matchPattern(element, mlir::m_Constant(&constant))) {
        decided = false;
        continue;
      }

      std::optional<bool> same = sameValue(value, constant);
      if (!same) {
        decided = false;
        continue;
      }

      if (*same) {
        rewriter.replaceOpWithNewOp<ConstantBoolOp>(op, op.getType(),
                                                    rewriter.getBoolAttr(true));
        return mlir::success();
      }
    }

    if (!decided)
      return mlir::failure();
    rewriter.replaceOpWithNewOp<ConstantBoolOp>(op, op.getType(),
                                                rewriter.getBoolAttr(false));
    return mlir::success();
  }
};

} // namespace

void InOp::getCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
                                       mlir::MLIRContext *context) {
  patterns.add<FoldMembership>(context);
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
  // Integers compare as integers. Above 2^53 a double stands for more than
  // one of them, so comparing through one answers a different question than
  // the engine will.
  std::optional<bool> value;
  if (auto lhs =
          llvm::dyn_cast_if_present<mlir::IntegerAttr>(adaptor.getLhs())) {
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
