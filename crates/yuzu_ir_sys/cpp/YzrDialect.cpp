#include "YzrDialect.h"

#include "YzDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"

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

// !yzr.rel<a: !yz.int64, b: !yz.bool> — an empty schema prints as !yzr.rel<>.
mlir::Type RelType::parse(mlir::AsmParser &parser) {
  llvm::SmallVector<mlir::StringAttr> names;
  llvm::SmallVector<mlir::Type> types;
  if (parser.parseLess())
    return {};
  if (failed(parser.parseOptionalGreater())) {
    do {
      std::string name;
      mlir::Type type;
      if (parser.parseKeywordOrString(&name) || parser.parseColon() ||
          parser.parseType(type))
        return {};
      names.push_back(mlir::StringAttr::get(parser.getContext(), name));
      types.push_back(type);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseGreater())
      return {};
  }
  return RelType::get(parser.getContext(), names, types);
}

void RelType::print(mlir::AsmPrinter &printer) const {
  printer << "<";
  llvm::interleaveComma(llvm::zip(getNames(), getTypes()), printer,
                        [&](auto column) {
                          printer << std::get<0>(column).getValue() << ": "
                                  << std::get<1>(column);
                        });
  printer << ">";
}

// A stage region's block arguments are the input row's columns, one per
// schema entry, in order.
static mlir::LogicalResult verifyColumnsMatch(mlir::Operation *op, RelType rel,
                                              mlir::Block &block) {
  auto types = rel.getTypes();
  if (block.getNumArguments() != types.size())
    return op->emitOpError("expects one block argument per input column: ")
           << types.size() << " columns, " << block.getNumArguments()
           << " arguments";
  for (auto [index, pair] :
       llvm::enumerate(llvm::zip(block.getArgumentTypes(), types)))
    if (std::get<0>(pair) != std::get<1>(pair))
      return op->emitOpError("block argument ")
             << index << " has type " << std::get<0>(pair) << " but column `"
             << rel.getNames()[index].getValue() << "` has type "
             << std::get<1>(pair);
  return mlir::success();
}

static mlir::FailureOr<YieldOp> terminatingYield(mlir::Operation *op,
                                                 mlir::Block &block) {
  if (block.empty() || !llvm::isa<YieldOp>(&block.back()))
    return op->emitOpError("expects its body to end in a yzr.yield");
  return llvm::cast<YieldOp>(&block.back());
}

mlir::LogicalResult FilterOp::verify() {
  mlir::Block &block = getBody().front();
  auto rel = llvm::cast<RelType>(getInput().getType());
  if (failed(verifyColumnsMatch(*this, rel, block)))
    return mlir::failure();
  auto yield = terminatingYield(*this, block);
  if (failed(yield))
    return mlir::failure();
  if (yield->getValues().size() != 1 ||
      !llvm::isa<::yuzu::yz::BoolType>(yield->getValues().front().getType()))
    return emitOpError("expects its body to yield one !yz.bool predicate");
  return mlir::success();
}

mlir::LogicalResult ProjectOp::verify() {
  mlir::Block &block = getBody().front();
  auto input = llvm::cast<RelType>(getInput().getType());
  auto result = llvm::cast<RelType>(getResult().getType());
  if (failed(verifyColumnsMatch(*this, input, block)))
    return mlir::failure();
  auto yield = terminatingYield(*this, block);
  if (failed(yield))
    return mlir::failure();
  if (mlir::TypeRange(yield->getValues()) != mlir::TypeRange(result.getTypes()))
    return emitOpError(
        "expects the yielded types to match the result schema's column types");
  return mlir::success();
}

mlir::LogicalResult ExtendOp::verify() {
  mlir::Block &block = getBody().front();
  auto input = llvm::cast<RelType>(getInput().getType());
  auto result = llvm::cast<RelType>(getResult().getType());
  if (failed(verifyColumnsMatch(*this, input, block)))
    return mlir::failure();
  auto yield = terminatingYield(*this, block);
  if (failed(yield))
    return mlir::failure();
  size_t inputs = input.getTypes().size();
  if (result.getTypes().size() != inputs + yield->getValues().size())
    return emitOpError("expects the result schema to be the input columns "
                       "followed by one column per yielded value");
  if (result.getTypes().take_front(inputs) != input.getTypes() ||
      result.getNames().take_front(inputs) != input.getNames())
    return emitOpError("expects the result schema to keep the input columns");
  if (mlir::TypeRange(result.getTypes().drop_front(inputs)) !=
      mlir::TypeRange(yield->getValues()))
    return emitOpError("expects the appended column types to match the "
                       "yielded types");
  return mlir::success();
}

// The grouping rule. Two taints propagate forward through the straight-line
// region: `row` marks values derived from a non-key column that no aggregate
// has consumed yet, and `measure` marks values derived from an aggregate's
// result. A `yzr.agg` operand must not be measure-tainted (that would nest
// aggregations), and a yielded value must not be row-tainted (that would leak
// an ungrouped column) — no matter how many intermediate ops launder it.
mlir::LogicalResult AggregateOp::verify() {
  mlir::Block &block = getBody().front();
  if (failed(verifyColumnsMatch(
          *this, llvm::cast<RelType>(getInput().getType()), block)))
    return mlir::failure();
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

  auto yield = terminatingYield(*this, block);
  if (failed(yield))
    return mlir::failure();
  for (mlir::Value value : yield->getValues())
    if (row.contains(value))
      return yield->emitOpError(
          "yields a value derived from a column that is neither grouped nor "
          "aggregated");
  return mlir::success();
}

} // namespace yuzu::yzr
