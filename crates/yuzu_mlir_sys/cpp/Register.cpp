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

// Typed construction and inspection of !yzl.param from the Rust side, the
// way MLIR's own C API exposes its builtin types.
extern "C" MlirType yzuParamTypeGet(MlirContext ctx, MlirStringRef name) {
  mlir::MLIRContext *context = unwrap(ctx);
  return wrap(yuzu::yzl::ParamType::get(
      context, mlir::StringAttr::get(context, unwrap(name))));
}

extern "C" bool yzuTypeIsParamType(MlirType type) {
  return llvm::isa<yuzu::yzl::ParamType>(unwrap(type));
}

extern "C" MlirStringRef yzuParamTypeName(MlirType type) {
  return wrap(
      llvm::cast<yuzu::yzl::ParamType>(unwrap(type)).getName().getValue());
}

// Typed construction and inspection of !yz.struct: the type is a symbol
// reference; the field list lives on the declaring op.
extern "C" MlirType yzuStructTypeGet(MlirContext ctx, MlirStringRef name) {
  mlir::MLIRContext *context = unwrap(ctx);
  return wrap(yuzu::yz::StructType::get(
      context, mlir::FlatSymbolRefAttr::get(context, unwrap(name))));
}

extern "C" bool yzuTypeIsStructType(MlirType type) {
  return llvm::isa<yuzu::yz::StructType>(unwrap(type));
}

extern "C" MlirStringRef yzuStructTypeName(MlirType type) {
  return wrap(
      llvm::cast<yuzu::yz::StructType>(unwrap(type)).getName().getValue());
}

// The types that carry no parameters. MLIR uniques types in the context, so
// each of these is a lookup returning the same pointer every time.
#define YZU_SINGLETON_TYPE(name, cls)                                          \
  extern "C" MlirType name(MlirContext ctx) {                                  \
    return wrap(cls::get(unwrap(ctx)));                                        \
  }

YZU_SINGLETON_TYPE(yzuInt64TypeGet, yuzu::yz::Int64Type)
YZU_SINGLETON_TYPE(yzuFloat64TypeGet, yuzu::yz::Float64Type)
YZU_SINGLETON_TYPE(yzuBoolTypeGet, yuzu::yz::BoolType)
YZU_SINGLETON_TYPE(yzuStrTypeGet, yuzu::yz::StrType)
YZU_SINGLETON_TYPE(yzuVarTypeGet, yuzu::yzl::VarType)
YZU_SINGLETON_TYPE(yzuQueryTypeGet, yuzu::yzl::QueryType)

#undef YZU_SINGLETON_TYPE
