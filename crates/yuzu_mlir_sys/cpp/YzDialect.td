#ifndef YZ_DIALECT_TD
#define YZ_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
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

def Yz_ConstantIntOp : Yz_Op<"constant_int", [ConstantLike, Pure]> {
  let summary = "integer literal";
  let arguments = (ins I64Attr:$value);
  let results = (outs Yz_Int64Type:$result);
  let assemblyFormat = "$value attr-dict";
  let hasFolder = 1;
}

def Yz_ConstantFloatOp : Yz_Op<"constant_float", [ConstantLike, Pure]> {
  let summary = "float literal";
  let arguments = (ins F64Attr:$value);
  let results = (outs Yz_Float64Type:$result);
  let assemblyFormat = "$value attr-dict";
  let hasFolder = 1;
}

def Yz_ConstantBoolOp : Yz_Op<"constant_bool", [ConstantLike, Pure]> {
  let summary = "boolean literal";
  let arguments = (ins BoolAttr:$value);
  let results = (outs Yz_BoolType:$result);
  let assemblyFormat = "$value attr-dict";
  let hasFolder = 1;
}

def Yz_ConstantStrOp : Yz_Op<"constant_str", [ConstantLike, Pure]> {
  let summary = "string literal";
  let arguments = (ins StrAttr:$value);
  let results = (outs Yz_StrType:$result);
  let assemblyFormat = "$value attr-dict";
  let hasFolder = 1;
}

// Arithmetic. One op per surface operator, polymorphic over the numeric
// types the way the language's operators are; folds dispatch on the
// literal's kind.

def Yz_NumericType : AnyTypeOf<[Yz_Int64Type, Yz_Float64Type]>;

class Yz_ArithBinaryOp<string mnemonic, list<Trait> traits = []>
    : Yz_Op<mnemonic, traits # [Pure, SameOperandsAndResultType]> {
  let arguments = (ins Yz_NumericType:$lhs, Yz_NumericType:$rhs);
  let results = (outs Yz_NumericType:$result);
  let assemblyFormat = "$lhs `,` $rhs `:` type($result) attr-dict";
  let hasFolder = 1;
}

def Yz_AddOp : Yz_ArithBinaryOp<"add", [Commutative]> {
  let summary = "addition";
}

def Yz_SubOp : Yz_ArithBinaryOp<"sub"> {
  let summary = "subtraction";
}

def Yz_MulOp : Yz_ArithBinaryOp<"mul", [Commutative]> {
  let summary = "multiplication";
}

def Yz_DivOp : Yz_ArithBinaryOp<"div"> {
  let summary = "division; never folds a zero divisor";
}

def Yz_RemOp : Yz_ArithBinaryOp<"rem"> {
  let summary = "remainder; never folds a zero divisor";
}

def Yz_NegOp : Yz_Op<"neg", [Pure, SameOperandsAndResultType]> {
  let summary = "negation";
  let arguments = (ins Yz_NumericType:$value);
  let results = (outs Yz_NumericType:$result);
  let assemblyFormat = "$value `:` type($result) attr-dict";
  let hasFolder = 1;
}

// Comparison and logic.

def Yz_CmpOp : Yz_Op<"cmp", [Pure, AllTypesMatch<["lhs", "rhs"]>]> {
  let summary = "numeric comparison";
  let arguments = (ins StrAttr:$predicate, Yz_NumericType:$lhs,
                       Yz_NumericType:$rhs);
  let results = (outs Yz_BoolType:$result);
  let assemblyFormat = "$predicate `,` $lhs `,` $rhs `:` type($lhs) attr-dict";
  let hasFolder = 1;
}

def Yz_AndOp : Yz_Op<"and", [Commutative, Pure]> {
  let summary = "logical and";
  let arguments = (ins Yz_BoolType:$lhs, Yz_BoolType:$rhs);
  let results = (outs Yz_BoolType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
  let hasFolder = 1;
}

def Yz_OrOp : Yz_Op<"or", [Commutative, Pure]> {
  let summary = "logical or";
  let arguments = (ins Yz_BoolType:$lhs, Yz_BoolType:$rhs);
  let results = (outs Yz_BoolType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
  let hasFolder = 1;
}

def Yz_NotOp : Yz_Op<"not", [Pure]> {
  let summary = "logical negation";
  let arguments = (ins Yz_BoolType:$value);
  let results = (outs Yz_BoolType:$result);
  let assemblyFormat = "$value attr-dict";
  let hasFolder = 1;
}

// Functions.

def Yz_FuncOp : Yz_Op<"func", [Symbol, IsolatedFromAbove]> {
  let summary = "a concrete function; parameters are block arguments";
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name $function_type $body attr-dict";
}

def Yz_CallOp : Yz_Op<"call"> {
  let summary = "a call to a yz.func, gone after elaboration";
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs AnyType:$result);
  let assemblyFormat =
      "$callee `(` $operands `)` attr-dict `:` functional-type($operands, $result)";
}

def Yz_ExternCallOp : Yz_Op<"extern_call"> {
  let summary = "a call the target provides by name; never folds";
  let arguments = (ins StrAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs AnyType:$result);
  let assemblyFormat =
      "$callee `(` $operands `)` attr-dict `:` functional-type($operands, $result)";
}

def Yz_ReturnOp : Yz_Op<"return", [Terminator, HasParent<"FuncOp">]> {
  let summary = "a function body's result";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

#endif // YZ_DIALECT_TD
