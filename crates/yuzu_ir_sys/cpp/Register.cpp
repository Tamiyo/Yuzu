#include "YzirDialect.h"

#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"

// The one seam between the C++ dialect library and the Rust compiler: register
// every Yuzu dialect in a context created on the Rust side.
extern "C" void yzuRegisterAllDialects(MlirContext ctx) {
  mlir::DialectRegistry registry;
  registry.insert<yuzu::yzir::YzirDialect>();
  unwrap(ctx)->appendDialectRegistry(registry);
  unwrap(ctx)->loadDialect<yuzu::yzir::YzirDialect>();
}
