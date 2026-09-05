#ifndef YZIR_DIALECT_TD
#define YZIR_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Yzir_Dialect : Dialect {
  let name = "yzir";
  let summary = "Yuzu's core calculus: scalars, functions, folds";
  let cppNamespace = "::yuzu::yzir";
  let useDefaultTypePrinterParser = 1;
  let hasConstantMaterializer = 1;
}

class Yzir_Type<string name, string typeMnemonic> : TypeDef<Yzir_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Yzir_Int64Type : Yzir_Type<"Int64", "int64"> {
  let summary = "64-bit signed integer";
}

def Yzir_Float64Type : Yzir_Type<"Float64", "float64"> {
  let summary = "64-bit float";
}

def Yzir_BoolType : Yzir_Type<"Bool", "bool"> {
  let summary = "boolean";
}

def Yzir_StrType : Yzir_Type<"Str", "str"> {
  let summary = "string";
}

class Yzir_Op<string mnemonic, list<Trait> traits = []>
    : Op<Yzir_Dialect, mnemonic, traits>;

def Yzir_ConstOp : Yzir_Op<"const", [ConstantLike, Pure]> {
  let summary = "integer literal";
  let arguments = (ins I64Attr:$value);
  let results = (outs Yzir_Int64Type:$result);
  let assemblyFormat = "$value attr-dict";
  let hasFolder = 1;
}

def Yzir_AddOp : Yzir_Op<"add", [Pure]> {
  let summary = "integer addition";
  let arguments = (ins Yzir_Int64Type:$lhs, Yzir_Int64Type:$rhs);
  let results = (outs Yzir_Int64Type:$sum);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
  let hasFolder = 1;
}

def Yzir_MulOp : Yzir_Op<"mul", [Pure]> {
  let summary = "integer multiplication";
  let arguments = (ins Yzir_Int64Type:$lhs, Yzir_Int64Type:$rhs);
  let results = (outs Yzir_Int64Type:$product);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
  let hasFolder = 1;
}

#endif // YZIR_DIALECT_TD
