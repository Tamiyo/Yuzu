#include "YzDialect.h"
#include "YzlDialect.h"
#include "YzrDialect.h"

#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

extern "C" void yzuRegisterAllDialects(MlirContext ctx) {
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

extern "C" MlirType yzuListTypeGet(MlirContext ctx, MlirType inner) {
  return wrap(yuzu::yz::ListType::get(unwrap(ctx), unwrap(inner)));
}

extern "C" bool yzuTypeIsListType(MlirType type) {
  return llvm::isa<yuzu::yz::ListType>(unwrap(type));
}

extern "C" MlirType yzuListTypeInner(MlirType type) {
  return wrap(llvm::cast<yuzu::yz::ListType>(unwrap(type)).getInner());
}

// The types that carry no parameters. MLIR uniques types in the context, so
// each of these is a lookup returning the same pointer every time.
#define YZU_SINGLETON_TYPE(getter, predicate, cls)                             \
  extern "C" MlirType getter(MlirContext ctx) {                                \
    return wrap(cls::get(unwrap(ctx)));                                        \
  }                                                                            \
  extern "C" bool predicate(MlirType type) {                                   \
    return llvm::isa<cls>(unwrap(type));                                       \
  }

YZU_SINGLETON_TYPE(yzuInt64TypeGet, yzuTypeIsInt64Type, yuzu::yz::Int64Type)
YZU_SINGLETON_TYPE(yzuFloat64TypeGet, yzuTypeIsFloat64Type, yuzu::yz::Float64Type)
YZU_SINGLETON_TYPE(yzuBoolTypeGet, yzuTypeIsBoolType, yuzu::yz::BoolType)
YZU_SINGLETON_TYPE(yzuStrTypeGet, yzuTypeIsStrType, yuzu::yz::StrType)
YZU_SINGLETON_TYPE(yzuUnresolvedTypeGet, yzuTypeIsUnresolvedType,
                   yuzu::yzl::UnresolvedType)
YZU_SINGLETON_TYPE(yzuErrorTypeGet, yzuTypeIsErrorType, yuzu::yzl::ErrorType)
YZU_SINGLETON_TYPE(yzuQueryTypeGet, yzuTypeIsQueryType, yuzu::yzl::QueryType)
YZU_SINGLETON_TYPE(yzuRefTypeGet, yzuTypeIsRefType, yuzu::yzl::RefType)

#undef YZU_SINGLETON_TYPE

// A location in a file whose name is already an attribute. A lowering makes
// one location for each op, and the overload taking a string hashes the file
// name again each time.
extern "C" MlirLocation yzuFileLineColRangeGet(MlirAttribute filename,
                                               unsigned startLine,
                                               unsigned startColumn,
                                               unsigned endLine,
                                               unsigned endColumn) {
  return wrap(mlir::Location(mlir::FileLineColRange::get(
      llvm::cast<mlir::StringAttr>(unwrap(filename)), startLine, startColumn,
      endLine, endColumn)));
}
