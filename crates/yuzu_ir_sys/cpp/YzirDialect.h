#pragma once

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "YzirDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "YzirTypes.h.inc"

#define GET_OP_CLASSES
#include "YzirOps.h.inc"
