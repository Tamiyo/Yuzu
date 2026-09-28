#ifndef YUZU_ATTRS_TD
#define YUZU_ATTRS_TD

include "mlir/IR/OpBase.td"

// A string attribute that holds one of `cases`. The Rust enum read from it
// spells the same cases, and its tests check each against the verifier.
class Yuzu_StrEnumAttr<list<string> cases>
    : ConfinedAttr<StrAttr, [AttrConstraint<
          CPred<"::llvm::is_contained({"
                # !interleave(!foreach(c, cases, "::llvm::StringRef(\"" # c # "\")"), ", ")
                # "}, ::llvm::cast<::mlir::StringAttr>($_self).getValue())">,
          "one of " # !interleave(cases, ", ")>]>;

#endif // YUZU_ATTRS_TD
