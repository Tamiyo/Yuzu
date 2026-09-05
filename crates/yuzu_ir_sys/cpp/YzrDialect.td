#ifndef YZR_DIALECT_TD
#define YZR_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def Yzr_Dialect : Dialect {
  let name = "yzr";
  let summary = "Yuzu's relational algebra: what the target runs";
  let cppNamespace = "::yuzu::yzr";
  let useDefaultTypePrinterParser = 1;
}

class Yzr_Type<string name, string typeMnemonic> : TypeDef<Yzr_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Yzr_RelType : Yzr_Type<"Rel", "rel"> {
  let summary = "a relation: named, typed columns";
  let parameters = (ins
      ArrayRefParameter<"::mlir::StringAttr">:$names,
      ArrayRefParameter<"::mlir::Type">:$types);
  let hasCustomAssemblyFormat = 1;
}

class Yzr_Op<string mnemonic, list<Trait> traits = []>
    : Op<Yzr_Dialect, mnemonic, traits>;

def Yzr_TableOp : Yzr_Op<"table"> {
  let summary = "scan a declared table";
  let arguments = (ins FlatSymbolRefAttr:$name);
  let results = (outs Yzr_RelType:$result);
  let assemblyFormat = "$name attr-dict `:` type($result)";
}

def Yzr_FilterOp : Yzr_Op<"filter", [SameOperandsAndResultType]> {
  let summary = "keep the rows whose predicate holds";
  let arguments = (ins Yzr_RelType:$input);
  let results = (outs Yzr_RelType:$result);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$input `:` type($input) $body attr-dict";
}

def Yzr_ProjectOp : Yzr_Op<"project"> {
  let summary = "replace the row with the yielded values";
  let arguments = (ins Yzr_RelType:$input);
  let results = (outs Yzr_RelType:$result);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$input $body `:` type($input) `->` type($result) attr-dict";
}

def Yzr_ExtendOp : Yzr_Op<"extend"> {
  let summary = "append the yielded values as new columns";
  let arguments = (ins Yzr_RelType:$input);
  let results = (outs Yzr_RelType:$result);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$input $body `:` type($input) `->` type($result) attr-dict";
}

def Yzr_LimitOp : Yzr_Op<"limit", [SameOperandsAndResultType]> {
  let summary = "keep at most the first count rows";
  let arguments = (ins Yzr_RelType:$input, I64Attr:$count);
  let results = (outs Yzr_RelType:$result);
  let assemblyFormat = "$input `,` $count `:` type($input) attr-dict";
}

def Yzr_AggregateOp : Yzr_Op<"aggregate"> {
  let summary = "group the input relation and compute measures";
  let description = [{
    The region's block arguments are the input row's columns. A block argument
    is used either through a `yzr.agg` op, or freely if its column index is in
    `keys` — the grouping rule, enforced by this op's verifier.
  }];
  let arguments = (ins Yzr_RelType:$input, DenseI64ArrayAttr:$keys);
  let results = (outs Yzr_RelType:$result);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$input `keys` $keys $body `:` type($input) `->` type($result) attr-dict";
}

def Yzr_AggOp : Yzr_Op<"agg"> {
  let summary = "a measure: an aggregate function over a row-level value";
  let arguments = (ins StrAttr:$fn, AnyType:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$fn `,` $value `:` type($value) `->` type($result) attr-dict";
}

def Yzr_YieldOp : Yzr_Op<"yield",
    [Terminator,
     ParentOneOf<["AggregateOp", "FilterOp", "ProjectOp", "ExtendOp"]>]> {
  let summary = "the values a stage's region produces";
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
}

#endif // YZR_DIALECT_TD
