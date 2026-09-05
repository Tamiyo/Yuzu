#pragma once

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "YzlDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "YzlTypes.h.inc"

#define GET_OP_CLASSES
#include "YzlOps.h.inc"
