#ifndef YZR_DIALECT_TD
#define YZR_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

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
  let summary = "a relation flowing between stages";
}

class Yzr_Op<string mnemonic, list<Trait> traits = []>
    : Op<Yzr_Dialect, mnemonic, traits>;

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
  let assemblyFormat = "$input `keys` $keys $body attr-dict";
  let hasVerifier = 1;
}

def Yzr_AggOp : Yzr_Op<"agg"> {
  let summary = "a measure: an aggregate function over a row-level value";
  let arguments = (ins StrAttr:$fn, AnyType:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$fn `,` $value `:` type($value) `->` type($result) attr-dict";
}

def Yzr_YieldOp : Yzr_Op<"yield", [Terminator, HasParent<"AggregateOp">]> {
  let summary = "the values a stage's region produces";
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
}

#endif // YZR_DIALECT_TD
