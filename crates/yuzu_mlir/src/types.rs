//! The dialects' parameterless types. MLIR uniques types in the context, so
//! each of these is a lookup that returns the same type every time — hold
//! one in a field where a hot path wants it, rather than passing a bag of
//! them around.

use melior::Context;
use melior::ir::Type;

macro_rules! singleton {
    ($name:ident, $get:ident, $doc:literal) => {
        #[doc = $doc]
        pub fn $name(context: &Context) -> Type<'_> {
            unsafe { Type::from_raw(yuzu_mlir_sys::$get(context.to_raw())) }
        }
    };
}

singleton!(int64, yzuInt64TypeGet, "`!yz.int64`");
singleton!(float64, yzuFloat64TypeGet, "`!yz.float64`");
singleton!(boolean, yzuBoolTypeGet, "`!yz.bool`");
singleton!(str, yzuStrTypeGet, "`!yz.str`");
singleton!(var, yzuVarTypeGet, "`!yzl.var`, the unification variable");
singleton!(
    query,
    yzuQueryTypeGet,
    "`!yzl.query`, a relation before its schema is known"
);

/// How a type is written in source, for a diagnostic: a reader wrote
/// `int64` and `List[int64]`, not `!yz.int64` and `!yz.list<!yz.int64>`.
/// The MLIR spelling is the fallback, so a type with no source syntax still
/// prints as something.
pub fn name(context: &Context, ty: Type<'_>) -> String {
    if let Some(list) = crate::ListType::from_type(ty) {
        return format!("List[{}]", name(context, list.inner()));
    }

    if let Some(declaration) = crate::StructType::from_type(ty) {
        return declaration.name().to_string();
    }

    if let Some(param) = crate::ParamType::from_type(ty) {
        return param.name().to_string();
    }

    for (scalar, spelling) in [
        (int64(context), "int64"),
        (float64(context), "float64"),
        (boolean(context), "bool"),
        (str(context), "str"),
    ] {
        if ty == scalar {
            return spelling.to_string();
        }
    }

    ty.to_string()
}
