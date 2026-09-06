#include "YzDialect.h"
#include "YzlDialect.h"
#include "YzrDialect.h"

#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

extern "C" void yzuRegisterAllDialects(MlirContext ctx) {
  mlir::DialectRegistry registry;
  registry.insert<yuzu::yz::YzDialect>();
  registry.insert<yuzu::yzl::YzlDialect>();
  registry.insert<yuzu::yzr::YzrDialect>();
  unwrap(ctx)->appendDialectRegistry(registry);
  unwrap(ctx)->loadDialect<yuzu::yz::YzDialect>();
  unwrap(ctx)->loadDialect<yuzu::yzl::YzlDialect>();
  unwrap(ctx)->loadDialect<yuzu::yzr::YzrDialect>();
}

// Typed construction and inspection of !yzr.rel from the Rust side, the way
// MLIR's own C API exposes its builtin types.
extern "C" MlirType yzuRelTypeGet(MlirContext ctx, intptr_t count,
                                  MlirStringRef const *names,
                                  MlirType const *types) {
  mlir::MLIRContext *context = unwrap(ctx);
  llvm::SmallVector<mlir::StringAttr> columnNames;
  llvm::SmallVector<mlir::Type> columnTypes;
  for (intptr_t index = 0; index < count; ++index) {
    columnNames.push_back(mlir::StringAttr::get(context, unwrap(names[index])));
    columnTypes.push_back(unwrap(types[index]));
  }
  return wrap(yuzu::yzr::RelType::get(context, columnNames, columnTypes));
}

extern "C" bool yzuTypeIsRelType(MlirType type) {
  return llvm::isa<yuzu::yzr::RelType>(unwrap(type));
}

extern "C" intptr_t yzuRelTypeColumnCount(MlirType type) {
  return llvm::cast<yuzu::yzr::RelType>(unwrap(type)).getNames().size();
}

extern "C" MlirStringRef yzuRelTypeColumnName(MlirType type, intptr_t index) {
  return wrap(llvm::cast<yuzu::yzr::RelType>(unwrap(type))
                  .getNames()[index]
                  .getValue());
}

extern "C" MlirType yzuRelTypeColumnType(MlirType type, intptr_t index) {
  return wrap(llvm::cast<yuzu::yzr::RelType>(unwrap(type)).getTypes()[index]);
}
