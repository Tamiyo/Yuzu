#include "YzrDialect.h"

#include "YzDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
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

} // namespace yuzu::yzr
