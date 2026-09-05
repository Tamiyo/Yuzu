#ifndef YZL_DIALECT_TD
#define YZL_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

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

class Yzl_Op<string mnemonic, list<Trait> traits = []>
    : Op<Yzl_Dialect, mnemonic, traits>;

def Yzl_NameOp : Yzl_Op<"name"> {
  let summary = "an unresolved reference: column, binding, or function";
  let arguments = (ins StrAttr:$name);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$name `:` type($result) attr-dict";
}

#endif // YZL_DIALECT_TD
