//! The dialects' types. MLIR uniques a type in its context, so a
//! parameterless type is fetched with `get` and a parametrized one is a
//! typed view built with `new`; both return the same type every time.

mod yz;
mod yzl;

use melior::Context;
use melior::ir::Type;

pub use yz::{BoolType, Float64Type, Int64Type, ListType, StrType, StructType};
pub use yzl::{ErrorType, ParamType, QueryType, RefType, UnresolvedType};

/// A scalar type's spelling in source, and the lookup returning it.
type Scalar = (&'static str, fn(&Context) -> Type<'_>);

const SCALARS: [Scalar; 4] = [
    ("int64", Int64Type::get),
    ("float64", Float64Type::get),
    ("bool", BoolType::get),
    ("str", StrType::get),
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
    if let Some(list) = ListType::from_type(ty) {
        return format!("List[{}]", name(context, list.inner()));
    }

    if let Some(declaration) = StructType::from_type(ty) {
        return declaration.name().to_string();
    }

    if let Some(param) = ParamType::from_type(ty) {
        return param.name().to_string();
    }

    match scalar_name(context, ty) {
        Some(scalar) => scalar.to_string(),
        None => ty.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use melior::Context;
    use melior::ir::Type;

    use super::{BoolType, Float64Type, Int64Type, QueryType, RefType, StrType, UnresolvedType};

    type Singleton = (fn(&Context) -> Type<'_>, fn(Type<'_>) -> bool);

    const SINGLETONS: [Singleton; 7] = [
        (Int64Type::get, Int64Type::is),
        (Float64Type::get, Float64Type::is),
        (BoolType::get, BoolType::is),
        (StrType::get, StrType::is),
        (UnresolvedType::get, UnresolvedType::is),
        (QueryType::get, QueryType::is),
        (RefType::get, RefType::is),
    ];

    #[test]
    fn each_singleton_is_itself_and_no_other() {
        let context = crate::context();
        for (row, (get, _)) in SINGLETONS.iter().enumerate() {
            let ty = get(&context);
            for (column, (_, is)) in SINGLETONS.iter().enumerate() {
                assert_eq!(is(ty), row == column, "{ty} against predicate {column}");
            }
        }
    }
}
