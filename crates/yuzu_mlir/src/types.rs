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

/// A scalar type's spelling in source, and the lookup returning it.
type Scalar = (&'static str, fn(&Context) -> Type<'_>);

const SCALARS: [Scalar; 4] = [
    ("int64", int64),
    ("float64", float64),
    ("bool", boolean),
    ("str", str),
];

/// The scalar type a name stands for, when it names one.
pub fn scalar<'c>(context: &'c Context, name: &str) -> Option<Type<'c>> {
    SCALARS
        .iter()
        .find(|(spelling, _)| *spelling == name)
        .map(|(_, get)| get(context))
}

/// How source spells a scalar type, when it is one.
pub fn scalar_name(context: &Context, ty: Type<'_>) -> Option<&'static str> {
    SCALARS
        .iter()
        .find(|(_, get)| get(context) == ty)
        .map(|(spelling, _)| *spelling)
}

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

    match scalar_name(context, ty) {
        Some(scalar) => scalar.to_string(),
        None => ty.to_string(),
    }
}
