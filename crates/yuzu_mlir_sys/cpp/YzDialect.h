#pragma once

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "YzlDialect.h"

#include "YzDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "YzTypes.h.inc"

#define GET_OP_CLASSES
#include "YzOps.h.inc"
