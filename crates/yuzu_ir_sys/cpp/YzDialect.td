#ifndef YZ_DIALECT_TD
#define YZ_DIALECT_TD

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

// Literals — one op per type, each constant-like and foldable so the stock
// canonicalizer can materialize fold results back into them.

def Yz_ConstOp : Yz_Op<"const", [ConstantLike, Pure]> {
  let summary = "integer literal";
  let arguments = (ins I64Attr:$value);
  let results = (outs Yz_Int64Type:$result);
  let assemblyFormat = "$value attr-dict";
  let hasFolder = 1;
}

def Yz_ConstFloatOp : Yz_Op<"const_float", [ConstantLike, Pure]> {
  let summary = "float literal";
  let arguments = (ins F64Attr:$value);
  let results = (outs Yz_Float64Type:$result);
  let assemblyFormat = "$value attr-dict";
  let hasFolder = 1;
}

def Yz_ConstBoolOp : Yz_Op<"const_bool", [ConstantLike, Pure]> {
  let summary = "boolean literal";
  let arguments = (ins BoolAttr:$value);
  let results = (outs Yz_BoolType:$result);
  let assemblyFormat = "$value attr-dict";
  let hasFolder = 1;
}

def Yz_ConstStrOp : Yz_Op<"const_str", [ConstantLike, Pure]> {
  let summary = "string literal";
  let arguments = (ins StrAttr:$value);
  let results = (outs Yz_StrType:$result);
  let assemblyFormat = "$value attr-dict";
  let hasFolder = 1;
}

// Integer arithmetic.

class Yz_IntBinaryOp<string mnemonic> : Yz_Op<mnemonic, [Pure]> {
  let arguments = (ins Yz_Int64Type:$lhs, Yz_Int64Type:$rhs);
  let results = (outs Yz_Int64Type:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
  let hasFolder = 1;
}

def Yz_AddOp : Yz_IntBinaryOp<"add"> {
  let summary = "integer addition";
}

def Yz_SubOp : Yz_IntBinaryOp<"sub"> {
  let summary = "integer subtraction";
}

def Yz_MulOp : Yz_IntBinaryOp<"mul"> {
  let summary = "integer multiplication";
}

def Yz_DivOp : Yz_IntBinaryOp<"div"> {
  let summary = "integer division; never folds a zero divisor";
}

def Yz_ModOp : Yz_IntBinaryOp<"mod"> {
  let summary = "integer remainder; never folds a zero divisor";
}

def Yz_NegOp : Yz_Op<"neg", [Pure]> {
  let summary = "integer negation";
  let arguments = (ins Yz_Int64Type:$value);
  let results = (outs Yz_Int64Type:$result);
  let assemblyFormat = "$value attr-dict";
  let hasFolder = 1;
}

// Comparison and logic.

def Yz_CmpOp : Yz_Op<"cmp", [Pure]> {
  let summary = "integer comparison";
  let arguments = (ins StrAttr:$predicate, Yz_Int64Type:$lhs, Yz_Int64Type:$rhs);
  let results = (outs Yz_BoolType:$result);
  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict";
  let hasFolder = 1;
}

class Yz_BoolBinaryOp<string mnemonic> : Yz_Op<mnemonic, [Pure]> {
  let arguments = (ins Yz_BoolType:$lhs, Yz_BoolType:$rhs);
  let results = (outs Yz_BoolType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
  let hasFolder = 1;
}

def Yz_AndOp : Yz_BoolBinaryOp<"and"> {
  let summary = "logical and";
}

def Yz_OrOp : Yz_BoolBinaryOp<"or"> {
  let summary = "logical or";
}

def Yz_NotOp : Yz_Op<"not", [Pure]> {
  let summary = "logical negation";
  let arguments = (ins Yz_BoolType:$value);
  let results = (outs Yz_BoolType:$result);
  let assemblyFormat = "$value attr-dict";
  let hasFolder = 1;
}

#endif // YZ_DIALECT_TD
