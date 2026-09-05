#ifndef YZIR_DIALECT_TD
#define YZIR_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Yz_Dialect : Dialect {
  let name = "yz";
  let summary = "Yuzu's core calculus: scalars, functions, folds";
  let cppNamespace = "::yuzu::yz";
  let useDefaultTypePrinterParser = 1;
  let hasConstantMaterializer = 1;
}

class Yz_Type<string name, string typeMnemonic> : TypeDef<Yz_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Yz_Int64Type : Yz_Type<"Int64", "int64"> {
  let summary = "64-bit signed integer";
}

def Yz_Float64Type : Yz_Type<"Float64", "float64"> {
  let summary = "64-bit float";
}

def Yz_BoolType : Yz_Type<"Bool", "bool"> {
  let summary = "boolean";
}

def Yz_StrType : Yz_Type<"Str", "str"> {
  let summary = "string";
}

class Yz_Op<string mnemonic, list<Trait> traits = []>
    : Op<Yz_Dialect, mnemonic, traits>;

def Yz_ConstOp : Yz_Op<"const", [ConstantLike, Pure]> {
  let summary = "integer literal";
  let arguments = (ins I64Attr:$value);
  let results = (outs Yz_Int64Type:$result);
  let assemblyFormat = "$value attr-dict";
  let hasFolder = 1;
}

def Yz_AddOp : Yz_Op<"add", [Pure]> {
  let summary = "integer addition";
  let arguments = (ins Yz_Int64Type:$lhs, Yz_Int64Type:$rhs);
  let results = (outs Yz_Int64Type:$sum);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
  let hasFolder = 1;
}

def Yz_MulOp : Yz_Op<"mul", [Pure]> {
  let summary = "integer multiplication";
  let arguments = (ins Yz_Int64Type:$lhs, Yz_Int64Type:$rhs);
  let results = (outs Yz_Int64Type:$product);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
  let hasFolder = 1;
}

#endif // YZIR_DIALECT_TD
