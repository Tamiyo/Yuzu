#ifndef YZ_DIALECT_TD
#define YZ_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "YuzuAttrs.td"

def Yz_Dialect : Dialect {
  let name = "yz";
  let summary = "Yuzu's core calculus: scalars, functions, folds";
  let cppNamespace = "::yuzu::yz";
  let useDefaultTypePrinterParser = 1;
  let hasConstantMaterializer = 1;
  let dependentDialects = ["::yuzu::yzl::YzlDialect"];
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

def Yz_UnitType : Yz_Type<"Unit", "unit"> {
  let summary = "what a function returns when it declares no result";
  let description = [{
    It holds no value, so no column, parameter or field has it.
  }];
}

def Yz_StructType : Yz_Type<"Struct", "struct"> {
  let summary = "a declared struct, by name — the declaration is the identity";
  let description = [{
    Nominal, as in Mojo: `!yz.struct<@Row>` is the struct the symbol declares,
    and two declarations with identical fields are different types. The field
    list lives on the declaring op, not here.
  }];
  let parameters = (ins "::mlir::FlatSymbolRefAttr":$name);
  let assemblyFormat = "`<` $name `>`";
}

def Yz_ListType : Yz_Type<"List", "list"> {
  let summary = "a list of one inner type";
  let parameters = (ins "::mlir::Type":$inner);
  let assemblyFormat = "`<` $inner `>`";
}

// Declarations. The dialect that owns `!yz.struct` owns what declares it.

class Yz_Op<string mnemonic, list<Trait> traits = []>
    : Op<Yz_Dialect, mnemonic, traits>;

def Yz_StructOp : Yz_Op<"struct", [Symbol]> {
  let summary = "the fields of the struct `!yz.struct<@sym>` names";
  let arguments = (ins SymbolNameAttr:$sym_name, StrArrayAttr:$names,
      TypeArrayAttr:$types);
  let assemblyFormat = "$sym_name $names `:` $types attr-dict";
  let hasVerifier = 1;
}

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

def Yz_ConstantListOp : Yz_Op<"constant_list", [ConstantLike, Pure]> {
  let summary = "a list literal whose elements are all literals";
  let description = [{
    The elements are one attribute rather than one op each, so a long
    literal list costs a pass one op. Each value is the attribute of the
    element type's constant: an integer, a float, a bool or a string.
  }];
  let arguments = (ins ArrayAttr:$values);
  let results = (outs Yz_ListType:$result);
  let assemblyFormat = "$values attr-dict `:` type($result)";
  let hasFolder = 1;
  let hasVerifier = 1;
}

def Yz_ListOp : Yz_Op<"list", [Pure]> {
  let summary = "a list of values; the operands are the elements";
  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs Yz_ListType:$result);
  let assemblyFormat =
      "`[` $elements `]` attr-dict `:` functional-type($elements, $result)";
}

// Arithmetic. One op per surface operator, polymorphic over the numeric
// types the way the language's operators are; folds dispatch on the
// literal's kind. Operand types are spelled individually, because the
// constraints admit three more types. An operand has !yzl.unresolved until
// InferTypes stamps it concrete. An operand of a generic body has
// !yzl.param until InlineCalls substitutes the call's types. A value an
// error left behind has !yzl.error.

def Yz_NumericType : Type<
    CPred<"::llvm::isa<::yuzu::yz::Int64Type, ::yuzu::yz::Float64Type, "
          "::yuzu::yzl::UnresolvedType, ::yuzu::yzl::ParamType, ::yuzu::yzl::ErrorType>($_self)">,
    "numeric, !yzl.unresolved, !yzl.param or !yzl.error">;

def Yz_IntegerType : Type<
    CPred<"::llvm::isa<::yuzu::yz::Int64Type, ::yuzu::yzl::UnresolvedType, "
          "::yuzu::yzl::ParamType, ::yuzu::yzl::ErrorType>($_self)">,
    "!yz.int64, !yzl.unresolved, !yzl.param or !yzl.error">;

def Yz_ScalarOrUnresolvedType : Type<
    CPred<"::llvm::isa<::yuzu::yz::Int64Type, ::yuzu::yz::Float64Type, "
          "::yuzu::yz::BoolType, ::yuzu::yz::StrType, ::yuzu::yzl::UnresolvedType, "
          "::yuzu::yzl::ParamType, ::yuzu::yzl::ErrorType>($_self)">,
    "a scalar, !yzl.unresolved, !yzl.param or !yzl.error">;

def Yz_BoolOrUnresolvedType : Type<
    CPred<"::llvm::isa<::yuzu::yz::BoolType, ::yuzu::yzl::UnresolvedType, "
          "::yuzu::yzl::ParamType, ::yuzu::yzl::ErrorType>($_self)">,
    "!yz.bool, !yzl.unresolved, !yzl.param or !yzl.error">;

class Yz_ArithBinaryOp<string mnemonic, list<Trait> traits = []>
    : Yz_Op<mnemonic, traits # [Pure]> {
  let arguments = (ins Yz_NumericType:$lhs, Yz_NumericType:$rhs);
  let results = (outs Yz_NumericType:$result);
  let assemblyFormat =
      "$lhs `,` $rhs `:` type($lhs) `,` type($rhs) `->` type($result) attr-dict";
  let hasFolder = 1;
}

def Yz_AddOp : Yz_ArithBinaryOp<"add", [Commutative]> {
  let summary = "addition";
  let description = [{
    Canonicalizes `(x + c1) + c2` to `x + (c1 + c2)`, which is what lets one
    constant reach another through the value between them. Only when the two
    constants share a sign: mixed signs can make an intermediate overflow
    that the reassociated form does not, and overflow belongs to the engine.
    The pattern is `reassociate_add` in `yuzu_mlir::rewrite`.
  }];
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

def Yz_PowOp : Yz_ArithBinaryOp<"pow"> {
  let summary = "exponentiation; never folds a negative integer exponent";
}

class Yz_ShiftOp<string mnemonic> : Yz_Op<mnemonic, [Pure]> {
  let arguments = (ins Yz_IntegerType:$lhs, Yz_IntegerType:$rhs);
  let results = (outs Yz_IntegerType:$result);
  let assemblyFormat =
      "$lhs `,` $rhs `:` type($lhs) `,` type($rhs) `->` type($result) attr-dict";
  let hasFolder = 1;
}

def Yz_ShlOp : Yz_ShiftOp<"shl"> {
  let summary = "shift left; never folds a shift that loses bits";
}

def Yz_ShrOp : Yz_ShiftOp<"shr"> {
  let summary = "arithmetic shift right";
}

def Yz_NegOp : Yz_Op<"neg", [Pure]> {
  let summary = "negation";
  let arguments = (ins Yz_NumericType:$value);
  let results = (outs Yz_NumericType:$result);
  let assemblyFormat = "$value `:` type($value) `->` type($result) attr-dict";
  let hasFolder = 1;
}

// Comparison and logic.

def Yz_CmpOp : Yz_Op<"cmp", [Pure]> {
  let summary = "scalar comparison";
  let arguments = (ins Yuzu_StrEnumAttr<["eq", "ne", "lt", "le", "gt", "ge"]>:$predicate,
                       Yz_ScalarOrUnresolvedType:$lhs,
                       Yz_ScalarOrUnresolvedType:$rhs);
  let results = (outs Yz_BoolOrUnresolvedType:$result);
  let assemblyFormat =
      "$predicate `,` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `->` type($result) attr-dict";
  let hasFolder = 1;
}

def Yz_AndOp : Yz_Op<"and", [Commutative, Pure]> {
  let summary = "logical and";
  let arguments = (ins Yz_BoolOrUnresolvedType:$lhs, Yz_BoolOrUnresolvedType:$rhs);
  let results = (outs Yz_BoolOrUnresolvedType:$result);
  let assemblyFormat =
      "$lhs `,` $rhs `:` type($lhs) `,` type($rhs) `->` type($result) attr-dict";
  let hasFolder = 1;
}

def Yz_OrOp : Yz_Op<"or", [Commutative, Pure]> {
  let summary = "logical or";
  let arguments = (ins Yz_BoolOrUnresolvedType:$lhs, Yz_BoolOrUnresolvedType:$rhs);
  let results = (outs Yz_BoolOrUnresolvedType:$result);
  let assemblyFormat =
      "$lhs `,` $rhs `:` type($lhs) `,` type($rhs) `->` type($result) attr-dict";
  let hasFolder = 1;
}

def Yz_InOp : Yz_Op<"in", [Pure]> {
  let summary = "membership: whether the value is an element of the list";
  let description = [{
    The op canonicalizes to a constant when the value is a constant and the
    list is a `yz.constant_list` or a `yz.list`. The result is true when an
    element equals the value. It is false when every element is a constant
    that does not. The pattern is `fold_membership` in `yuzu_mlir::rewrite`.
  }];
  let arguments = (ins AnyType:$value, AnyType:$list);
  let results = (outs Yz_BoolOrUnresolvedType:$result);
  let assemblyFormat =
      "$value `,` $list `:` type($value) `,` type($list) `->` type($result) attr-dict";
}

def Yz_NotOp : Yz_Op<"not", [Pure]> {
  let summary = "logical negation";
  let arguments = (ins Yz_BoolOrUnresolvedType:$value);
  let results = (outs Yz_BoolOrUnresolvedType:$result);
  let assemblyFormat = "$value `:` type($value) `->` type($result) attr-dict";
  let hasFolder = 1;
}

def Yz_CoalesceOp : Yz_Op<"coalesce", [Pure]> {
  let summary = "the first operand that is not null";
  let description = [{
    A full join `using` a column gives the column from whichever side has
    the row, so the one column holds the key of every row.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat =
      "$lhs `,` $rhs `:` type($lhs) `,` type($rhs) `->` type($result) attr-dict";
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
