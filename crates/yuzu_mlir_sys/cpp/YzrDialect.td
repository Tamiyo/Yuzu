#ifndef YZR_DIALECT_TD
#define YZR_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Yzr_Dialect : Dialect {
  let name = "yzr";
  let summary = "Yuzu's relational algebra: what the target runs";
  let cppNamespace = "::yuzu::yzr";
}

// A relation is a name tied to a struct: relational values are typed by the
// row's `!yz.struct`, and relation-ness lives in the ops. Referenced by
// predicate so yzr's TableGen stays independent of yz's.
def Yzr_RowType : Type<CPred<"::llvm::isa<::yuzu::yz::StructType>($_self)">,
                       "a !yz.struct row type">;

class Yzr_Op<string mnemonic, list<Trait> traits = []>
    : Op<Yzr_Dialect, mnemonic, traits>;

def Yzr_TableOp : Yzr_Op<"table", [Pure]> {
  let summary = "scan a declared table";
  let arguments = (ins FlatSymbolRefAttr:$name);
  let results = (outs Yzr_RowType:$result);
  let assemblyFormat = "$name attr-dict `:` type($result)";
}

def Yzr_FilterOp
    : Yzr_Op<"filter", [IsolatedFromAbove, SameOperandsAndResultType]> {
  let summary = "keep the rows whose predicate holds";
  let arguments = (ins Yzr_RowType:$input);
  let results = (outs Yzr_RowType:$result);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$input `:` type($input) $body attr-dict";
}

def Yzr_ProjectOp : Yzr_Op<"project", [IsolatedFromAbove, Pure]> {
  let summary = "replace the row with the yielded values";
  let arguments = (ins Yzr_RowType:$input);
  let results = (outs Yzr_RowType:$result);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$input $body `:` type($input) `->` type($result) attr-dict";
}

def Yzr_ExtendOp : Yzr_Op<"extend", [IsolatedFromAbove, Pure]> {
  let summary = "append the yielded values as new columns";
  let arguments = (ins Yzr_RowType:$input);
  let results = (outs Yzr_RowType:$result);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$input $body `:` type($input) `->` type($result) attr-dict";
}

def Yzr_LimitOp : Yzr_Op<"limit", [Pure, SameOperandsAndResultType]> {
  let summary = "keep at most the first count rows";
  let arguments = (ins Yzr_RowType:$input, I64Attr:$count);
  let results = (outs Yzr_RowType:$result);
  let assemblyFormat = "$input `,` $count `:` type($input) attr-dict";
}

def Yzr_AggregateOp : Yzr_Op<"aggregate", [IsolatedFromAbove, Pure]> {
  let summary = "group the input relation and compute measures";
  let description = [{
    The region's block arguments are the input row's columns. A block argument
    is used either through a `yzr.agg` op, or freely if its column index is in
    `keys` — the grouping rule, enforced by this op's verifier.
  }];
  let arguments = (ins Yzr_RowType:$input, DenseI64ArrayAttr:$keys);
  let results = (outs Yzr_RowType:$result);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$input `keys` $keys $body `:` type($input) `->` type($result) attr-dict";
}

def Yzr_JoinOp : Yzr_Op<"join", [IsolatedFromAbove, Pure]> {
  let summary = "join two relations; the on-region sees both rows' columns";
  let arguments = (ins StrAttr:$kind, Yzr_RowType:$lhs, Yzr_RowType:$rhs);
  let results = (outs Yzr_RowType:$result);
  let regions = (region SizedRegion<1>:$on);
  let assemblyFormat =
      "$kind `,` $lhs `,` $rhs $on `:` type($lhs) `,` type($rhs) `->` type($result) attr-dict";
}

class Yzr_SetOp<string mnemonic>
    : Yzr_Op<mnemonic, [Pure, SameOperandsAndResultType]> {
  let arguments = (ins Yzr_RowType:$lhs, Yzr_RowType:$rhs);
  let results = (outs Yzr_RowType:$result);
  let assemblyFormat = "$lhs `,` $rhs `:` type($result) attr-dict";
}

def Yzr_UnionOp : Yzr_SetOp<"union"> {
  let summary = "all rows of both relations";
}

def Yzr_IntersectOp : Yzr_SetOp<"intersect"> {
  let summary = "the rows present in both relations";
}

def Yzr_ExceptOp : Yzr_SetOp<"except"> {
  let summary = "the left rows absent from the right";
}

def Yzr_AggOp : Yzr_Op<"agg", [Pure]> {
  let summary = "a measure: an aggregate function over a row-level value";
  let arguments = (ins StrAttr:$fn, AnyType:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$fn `,` $value `:` type($value) `->` type($result) attr-dict";
}

def Yzr_CountOp : Yzr_Op<"count", [Pure]> {
  let summary = "a measure counting the group's rows; takes no value";
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Yzr_OutputOp : Yzr_Op<"output"> {
  let summary = "the program's query: what emission serializes";
  let arguments = (ins Yzr_RowType:$query);
  let assemblyFormat = "$query `:` type($query) attr-dict";
}

def Yzr_YieldOp : Yzr_Op<"yield",
    [Terminator,
     ParentOneOf<["AggregateOp", "FilterOp", "ProjectOp", "ExtendOp",
                  "JoinOp"]>]> {
  let summary = "the values a stage's region produces";
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
}

#endif // YZR_DIALECT_TD
