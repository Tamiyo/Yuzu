#ifndef YZL_DIALECT_TD
#define YZL_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Yzl_Dialect : Dialect {
  let name = "yzl";
  let summary = "Yuzu's source dialect: names and sugar, before resolution";
  let cppNamespace = "::yuzu::yzl";
  let useDefaultTypePrinterParser = 1;
}

class Yzl_Type<string name, string typeMnemonic> : TypeDef<Yzl_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Yzl_VarType : Yzl_Type<"Var", "var"> {
  let summary = "a unification variable: a type inference has not resolved yet";
}

def Yzl_QueryType : Yzl_Type<"Query", "query"> {
  let summary = "a relation whose schema resolution has not computed yet";
}

class Yzl_Op<string mnemonic, list<Trait> traits = []>
    : Op<Yzl_Dialect, mnemonic, traits>;

// Declarations.

def Yzl_StructOp : Yzl_Op<"struct", [Symbol]> {
  let summary = "a row type declaration; annotations are syntactically known";
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$schema);
  let assemblyFormat = "$sym_name $schema attr-dict";
}

def Yzl_TableOp : Yzl_Op<"table", [Symbol]> {
  let summary = "a table declaration naming its row struct";
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$row);
  let assemblyFormat = "$sym_name `of` $row attr-dict";
}

def Yzl_FnOp : Yzl_Op<"fn", [Symbol, IsolatedFromAbove]> {
  let summary = "a function declaration; parameters are referenced by name";
  let arguments = (ins SymbolNameAttr:$sym_name, StrArrayAttr:$params,
                       UnitAttr:$agg, UnitAttr:$external);
  let regions = (region AnyRegion:$body);
  let assemblyFormat =
      "$sym_name `params` $params (`agg` $agg^)? (`external` $external^)? "
      "$body attr-dict";
}

def Yzl_LetOp : Yzl_Op<"let", [Symbol]> {
  let summary = "a named binding of the single value its body yields";
  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name $body attr-dict";
}

def Yzl_ReturnOp : Yzl_Op<"return", [Terminator, HasParent<"FnOp">]> {
  let summary = "a function body's result";
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
}

// Query stages. 

def Yzl_FromOp : Yzl_Op<"from", [Pure]> {
  let summary = "start a query from a named table or binding";
  let arguments = (ins FlatSymbolRefAttr:$source);
  let results = (outs Yzl_QueryType:$result);
  let assemblyFormat = "$source attr-dict";
}

def Yzl_WhereOp : Yzl_Op<"where", [Pure]> {
  let summary = "keep the rows whose predicate holds";
  let arguments = (ins Yzl_QueryType:$input);
  let results = (outs Yzl_QueryType:$result);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$input $body attr-dict";
}

def Yzl_SelectOp : Yzl_Op<"select", [Pure]> {
  let summary = "replace the row with the yielded values, named by `as`";
  let arguments = (ins Yzl_QueryType:$input, StrArrayAttr:$names);
  let results = (outs Yzl_QueryType:$result);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$input `as` $names $body attr-dict";
}

def Yzl_ExtendOp : Yzl_Op<"extend", [Pure]> {
  let summary = "append the yielded values as new columns, named by `as`";
  let arguments = (ins Yzl_QueryType:$input, StrArrayAttr:$names);
  let results = (outs Yzl_QueryType:$result);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$input `as` $names $body attr-dict";
}

def Yzl_AggregateOp : Yzl_Op<"aggregate", [Pure]> {
  let summary = "group by named columns and compute the yielded measures";
  let arguments = (ins Yzl_QueryType:$input, StrArrayAttr:$group_by,
                       StrArrayAttr:$names);
  let results = (outs Yzl_QueryType:$result);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$input `group_by` $group_by `as` $names $body attr-dict";
}

def Yzl_LimitOp : Yzl_Op<"limit", [Pure]> {
  let summary = "keep at most the first count rows";
  let arguments = (ins Yzl_QueryType:$input, I64Attr:$count);
  let results = (outs Yzl_QueryType:$result);
  let assemblyFormat = "$input `,` $count attr-dict";
}

def Yzl_RenameOp : Yzl_Op<"rename", [Pure]> {
  let summary = "rename columns; a pure name effect";
  let arguments = (ins Yzl_QueryType:$input, StrArrayAttr:$from,
                       StrArrayAttr:$to);
  let results = (outs Yzl_QueryType:$result);
  let assemblyFormat = "$input `from` $from `to` $to attr-dict";
}

def Yzl_AliasOp : Yzl_Op<"alias", [Pure]> {
  let summary = "qualify the row's columns under a name";
  let arguments = (ins Yzl_QueryType:$input, StrAttr:$alias);
  let results = (outs Yzl_QueryType:$result);
  let assemblyFormat = "$input `as` $alias attr-dict";
}

// Expressions.

def Yzl_JoinOp : Yzl_Op<"join", [Pure]> {
  let summary = "join a named relation onto the query";
  let arguments = (ins Yzl_QueryType:$lhs, StrAttr:$kind, FlatSymbolRefAttr:$rhs,
                       OptionalAttr<StrAttr>:$rhs_alias,
                       OptionalAttr<StrArrayAttr>:$using_columns);
  let results = (outs Yzl_QueryType:$result);
  let regions = (region AnyRegion:$on);
  let assemblyFormat =
      "$kind `,` $lhs `,` $rhs (`as` $rhs_alias^)? (`using` $using_columns^)? "
      "$on attr-dict";
}

def Yzl_SetOp : Yzl_Op<"set", [Pure]> {
  let summary = "assign the yielded values to existing columns";
  let arguments = (ins Yzl_QueryType:$input, StrArrayAttr:$names);
  let results = (outs Yzl_QueryType:$result);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$input `as` $names $body attr-dict";
}

def Yzl_DistinctOp : Yzl_Op<"distinct", [Pure]> {
  let summary = "keep one row per distinct value";
  let arguments = (ins Yzl_QueryType:$input);
  let results = (outs Yzl_QueryType:$result);
  let assemblyFormat = "$input attr-dict";
}

def Yzl_DropOp : Yzl_Op<"drop", [Pure]> {
  let summary = "remove named columns; a pure name effect";
  let arguments = (ins Yzl_QueryType:$input, StrArrayAttr:$columns);
  let results = (outs Yzl_QueryType:$result);
  let assemblyFormat = "$input $columns attr-dict";
}

def Yzl_OutputOp : Yzl_Op<"output"> {
  let summary = "the program's query: what compilation exists to produce";
  let arguments = (ins Yzl_QueryType:$query);
  let assemblyFormat = "$query attr-dict";
}

def Yzl_NameOp : Yzl_Op<"name", [Pure]> {
  let summary = "an unresolved reference: column, binding, or parameter";
  let arguments = (ins StrAttr:$name);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$name `:` type($result) attr-dict";
}

def Yzl_ListOp : Yzl_Op<"list", [Pure]> {
  let summary = "a list literal";
  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs AnyType:$result);
  let assemblyFormat =
      "`[` $elements `]` attr-dict `:` functional-type($elements, $result)";
}

def Yzl_CallOp : Yzl_Op<"call", [Pure]> {
  let summary = "a call by name; registry and shadowing unresolved";
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs AnyType:$result);
  let assemblyFormat =
      "$callee `(` $operands `)` `:` functional-type($operands, $result) attr-dict";
}

def Yzl_YieldOp : Yzl_Op<"yield",
    [Terminator,
     ParentOneOf<["WhereOp", "SelectOp", "ExtendOp", "AggregateOp",
                  "LetOp", "JoinOp", "SetOp"]>]> {
  let summary = "the values a stage's region produces";
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
}

#endif // YZL_DIALECT_TD
