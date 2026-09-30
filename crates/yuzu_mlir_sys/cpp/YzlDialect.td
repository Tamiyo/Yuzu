#ifndef YZL_DIALECT_TD
#define YZL_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "YuzuAttrs.td"

def Yzl_Dialect : Dialect {
  let name = "yzl";
  let summary = "Yuzu's source dialect: names and sugar, before resolution";
  let cppNamespace = "::yuzu::yzl";
  let useDefaultTypePrinterParser = 1;
}

class Yzl_Type<string name, string typeMnemonic> : TypeDef<Yzl_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Yzl_UnresolvedType : Yzl_Type<"Unresolved", "unresolved"> {
  let summary = "a unification variable: a type inference has not resolved yet";
}

def Yzl_ErrorType : Yzl_Type<"Error", "error"> {
  let summary = "the type of a value an error left behind";
  let description = [{
    What a type annotation the lowering could not resolve stands for, and
    what inference gives a value computed from a `yzl.missing` hole. It
    matches every type, so the error that made it is reported once and not
    again by each value that reads it. A module that holds it has reported
    an error, and never lowers to `yzr`.
  }];
}

def Yzl_QueryType : Yzl_Type<"Query", "query"> {
  let summary = "a relation whose schema resolution has not computed yet";
}

def Yzl_RefType : Yzl_Type<"Ref", "ref"> {
  let summary = "a place that holds the value of a local variable";
  let description = [{
    What `yzl.local` gives, and what `yzl.load` and `yzl.store` read and
    write. `element` is the type of the value the place holds: the local's
    annotation, or `!yzl.unresolved` until inference resolves it. `mem2reg`
    removes every place after inference.
  }];
  let parameters = (ins "::mlir::Type":$element);
  let assemblyFormat = "`<` $element `>`";
}

def Yzl_ParamType : Yzl_Type<"Param", "param"> {
  let summary = "a generic parameter in type position, by name";
  let description = [{
    The sibling of `!yzl.unresolved`: both stand for a type the checking passes
    resolve, but a parameter is named by the program and a variable is minted
    by inference. `Self` is the parameter a trait declares implicitly.
  }];
  let parameters = (ins "::mlir::StringAttr":$name);
  let assemblyFormat = "`<` $name `>`";
}

class Yzl_Op<string mnemonic, list<Trait> traits = []>
    : Op<Yzl_Dialect, mnemonic, traits>;

// Declarations.

def Yzl_MissingOp : Yzl_Op<"missing", [Pure]> {
  let summary = "a hole where a parse error removed the source";
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Yzl_StructOp : Yzl_Op<"struct", [Symbol]> {
  let summary = "a struct declaration: the identity `!yz.struct<@sym>` refers to";
  let arguments = (ins SymbolNameAttr:$sym_name, StrArrayAttr:$names,
      TypeArrayAttr:$types);
  let assemblyFormat = "$sym_name $names `:` $types attr-dict";
  let hasVerifier = 1;
}

def Yzl_TraitOp : Yzl_Op<"trait",
    [Symbol, SymbolTable, IsolatedFromAbove, NoTerminator]> {
  let summary = "a trait declaration: the signatures an implementation supplies";
  let description = [{
    The body holds one body-less `yzl.fn` per method, typed against the
    implicit `!yzl.param<"Self">`.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$sym_name $body attr-dict";
}

def Yzl_ImplOp : Yzl_Op<"impl",
    [SymbolTable, IsolatedFromAbove, NoTerminator]> {
  let summary = "an implementation of a trait for one type";
  let description = [{
    `Self` is the target. It is a name, not a type: whether `@Money` is a
    declared struct, a builtin scalar, or nothing at all is resolution's
    question, so a declared struct and a builtin are the same shape here.
  }];
  let arguments = (ins FlatSymbolRefAttr:$trait, FlatSymbolRefAttr:$target);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$trait `for` $target $body attr-dict";
}

def Yzl_TableOp : Yzl_Op<"table", [Symbol]> {
  let summary = "a table declaration naming its row struct";
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$row);
  let assemblyFormat = "$sym_name `of` $row attr-dict";
}

def Yzl_FnOp : Yzl_Op<"fn", [Symbol, IsolatedFromAbove]> {
  let summary = "a function declaration; parameters are referenced by name";
  let description = [{
    `external_name` marks a function the engine provides, and is the name the
    engine knows it by. The symbol is qualified by the module that declares
    it, so the two differ outside the entry file.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name, StrArrayAttr:$params,
                       TypeAttrOf<FunctionType>:$signature,
                       OptionalAttr<StrArrayAttr>:$type_params,
                       OptionalAttr<StrArrayAttr>:$bound_params,
                       OptionalAttr<ArrayAttr>:$bound_traits,
                       UnitAttr:$is_agg, OptionalAttr<StrAttr>:$external_name);
  let regions = (region AnyRegion:$body);
  let assemblyFormat =
      "$sym_name (`generics` $type_params^)? "
      "(`where` $bound_params^ `:` $bound_traits)? "
      "`params` $params $signature (`agg` $is_agg^)? "
      "(`external` $external_name^)? $body attr-dict";
  let hasVerifier = 1;
}

def Yzl_ConstOp : Yzl_Op<"const", [Symbol]> {
  let summary = "a named constant: the single value its body yields";
  let description = [{
    The annotation, when the program wrote one, is the type the body must
    yield — declared here the way a function's signature is, for inference
    to hold the body to.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name,
                       OptionalAttr<TypeAttr>:$annotation);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name (`:` $annotation^)? $body attr-dict";
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

def Yzl_WhereOp : Yzl_Op<"where", [IsolatedFromAbove, Pure]> {
  let summary = "keep the rows whose predicate holds";
  let arguments = (ins Yzl_QueryType:$input);
  let results = (outs Yzl_QueryType:$result);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$input $body attr-dict";
}

def Yzl_SelectOp : Yzl_Op<"select", [IsolatedFromAbove, Pure]> {
  let summary = "replace the row with the yielded values, named by `as`";
  let arguments = (ins Yzl_QueryType:$input, StrArrayAttr:$names);
  let results = (outs Yzl_QueryType:$result);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$input `as` $names $body attr-dict";
}

def Yzl_ExtendOp : Yzl_Op<"extend", [IsolatedFromAbove, Pure]> {
  let summary = "append the yielded values as new columns, named by `as`";
  let arguments = (ins Yzl_QueryType:$input, StrArrayAttr:$names);
  let results = (outs Yzl_QueryType:$result);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$input `as` $names $body attr-dict";
}

def Yzl_AggregateOp : Yzl_Op<"aggregate", [IsolatedFromAbove, Pure]> {
  let summary = "group by named columns and compute the yielded measures";
  let arguments = (ins Yzl_QueryType:$input, StrArrayAttr:$group_by,
                       StrArrayAttr:$names,
                       OptionalAttr<I64ArrayAttr>:$key_cols);
  let results = (outs Yzl_QueryType:$result);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$input `group_by` $group_by `as` $names $body attr-dict";
}

def Yzl_LimitOp : Yzl_Op<"limit", [Pure]> {
  let summary = "keep at most count rows, after skipping offset";
  let arguments = (ins Yzl_QueryType:$input, I64Attr:$count,
                       OptionalAttr<I64Attr>:$offset);
  let results = (outs Yzl_QueryType:$result);
  let assemblyFormat = "$input `,` $count (`offset` $offset^)? attr-dict";
}

def Yzl_RenameOp : Yzl_Op<"rename", [Pure]> {
  let summary = "rename columns; a pure name effect";
  let arguments = (ins Yzl_QueryType:$input, StrArrayAttr:$from,
                       StrArrayAttr:$to,
                       OptionalAttr<I64ArrayAttr>:$rename_cols);
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

def Yzl_JoinOp : Yzl_Op<"join", [IsolatedFromAbove, Pure]> {
  let summary = "join a named relation onto the query";
  let description = [{
    With `using`, `left_keys` and `right_keys` give the position of each
    named column in the left row and in the right row. The joined row starts
    with each named column once. The other left columns follow, then the
    other right columns. Last come the two sides' own copies of the named
    columns, which only a qualified name reaches.
  }];
  let arguments = (ins Yzl_QueryType:$lhs,
                       Yuzu_StrEnumAttr<["inner", "left", "right", "full"]>:$kind,
                       FlatSymbolRefAttr:$rhs,
                       OptionalAttr<StrAttr>:$rhs_alias,
                       OptionalAttr<StrArrayAttr>:$using_columns,
                       OptionalAttr<I64ArrayAttr>:$left_keys,
                       OptionalAttr<I64ArrayAttr>:$right_keys);
  let results = (outs Yzl_QueryType:$result);
  let regions = (region AnyRegion:$on);
  let assemblyFormat =
      "$kind `,` $lhs `,` $rhs (`as` $rhs_alias^)? (`using` $using_columns^)? "
      "$on attr-dict";
}

def Yzl_SetOp : Yzl_Op<"set", [IsolatedFromAbove, Pure]> {
  let summary = "assign the yielded values to existing columns";
  let arguments = (ins Yzl_QueryType:$input, StrArrayAttr:$names,
                       OptionalAttr<I64ArrayAttr>:$set_cols);
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
  let description = [{
    `drop_cols` gives the position of each dropped column in the input row.
  }];
  let arguments = (ins Yzl_QueryType:$input, StrArrayAttr:$columns,
                       I64ArrayAttr:$drop_cols);
  let results = (outs Yzl_QueryType:$result);
  let assemblyFormat = "$input $columns attr-dict";
}

// Local variables. A place is declared in its function's body where the
// variable is, and holds its value until `mem2reg` promotes it.

def Yzl_LocalOp : Yzl_Op<"local",
    [MemoryEffects<[MemAlloc<DefaultResource>]>,
     DeclareOpInterfaceMethods<PromotableAllocationOpInterface>]> {
  let summary = "a place for a local variable in a function body";
  let description = [{
    `var_name` is the variable as the program wrote it, for diagnostics. A place
    without `mut` has exactly one store, its initializer. `is_param` marks the
    place a parameter is copied into.
  }];
  let arguments = (ins StrAttr:$var_name, UnitAttr:$is_mut, UnitAttr:$is_param);
  let results = (outs Yzl_RefType:$place);
  let assemblyFormat =
      "$var_name (`mut` $is_mut^)? (`param` $is_param^)? attr-dict `:` qualified(type($place))";
}

def Yzl_LoadOp : Yzl_Op<"load",
    [MemoryEffects<[MemRead<DefaultResource>]>,
     DeclareOpInterfaceMethods<PromotableMemOpInterface>]> {
  let summary = "the value a place holds";
  let arguments = (ins Yzl_RefType:$place);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$place attr-dict `:` qualified(type($place)) `->` type($result)";
}

def Yzl_StoreOp : Yzl_Op<"store",
    [MemoryEffects<[MemWrite<DefaultResource>]>,
     DeclareOpInterfaceMethods<PromotableMemOpInterface>]> {
  let summary = "gives a place a new value";
  let arguments = (ins Yzl_RefType:$place, AnyType:$value);
  let assemblyFormat = "$place `,` $value attr-dict `:` qualified(type($place)) `,` type($value)";
}

def Yzl_OutputOp : Yzl_Op<"output"> {
  let summary = "the program's query: what compilation exists to produce";
  let arguments = (ins Yzl_QueryType:$query);
  let assemblyFormat = "$query attr-dict";
}

def Yzl_ListOp : Yzl_Op<"list", [Pure]> {
  let summary = "a list literal";
  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs AnyType:$result);
  let assemblyFormat =
      "`[` $elements `]` attr-dict `:` functional-type($elements, $result)";
}

def Yzl_CallOp : Yzl_Op<"call", [Pure]> {
  let summary = "a call to a function or a file-level constant, by symbol";
  let description = [{
    `is_agg` says the callee aggregates, whichever kind of callee it is: an
    `agg def` or an `external agg def`. Resolution knows which at the point
    the call is built, so nothing downstream asks again.
  }];
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands,
                       OptionalAttr<Yuzu_StrEnumAttr<["fn", "external", "const"]>>:$callee_source,
                       OptionalAttr<TypeArrayAttr>:$type_args, UnitAttr:$is_agg);
  let results = (outs AnyType:$result);
  let assemblyFormat =
      "$callee `(` $operands `)` `:` functional-type($operands, $result) attr-dict";
}

def Yzl_YieldOp : Yzl_Op<"yield",
    [Terminator,
     ParentOneOf<["WhereOp", "SelectOp", "ExtendOp", "AggregateOp",
                  "ConstOp", "JoinOp", "SetOp"]>]> {
  let summary = "the values a stage's region produces";
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
}

#endif // YZL_DIALECT_TD
