#include "YzDialect.h"
#include "YzlDialect.h"
#include "YzrDialect.h"

#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"

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
