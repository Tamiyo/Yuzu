//! The dialects' types. MLIR uniques a type in its context, so a
//! parameterless type is fetched with `get` and a parametrized one is a
//! typed view built with `new`; both return the same type every time.

mod yz;
mod yzl;

use melior::Context;
use melior::ir::Type;

pub use yz::{BoolType, Float64Type, Int64Type, ListType, StrType, StructType};
pub use yzl::{ErrorType, ParamType, QueryType, RefType, UnresolvedType};

/// A scalar type: how source spells it, the lookup returning it, and the
/// test for it.
struct Scalar {
    spelling: &'static str,
    get: fn(&Context) -> Type<'_>,
    is: fn(Type<'_>) -> bool,
}

const SCALARS: [Scalar; 4] = [
    Scalar {
        spelling: "int64",
        get: Int64Type::get,
        is: Int64Type::is,
    },
    Scalar {
        spelling: "float64",
        get: Float64Type::get,
        is: Float64Type::is,
    },
    Scalar {
        spelling: "bool",
        get: BoolType::get,
        is: BoolType::is,
    },
    Scalar {
        spelling: "str",
        get: StrType::get,
        is: StrType::is,
    },
];

/// The scalar type a name stands for, when it names one.
#[must_use]
pub fn scalar<'c>(context: &'c Context, name: &str) -> Option<Type<'c>> {
    SCALARS
        .iter()
        .find(|scalar| scalar.spelling == name)
        .map(|scalar| (scalar.get)(context))
}

/// How source spells a scalar type, when it is one.
fn scalar_name(ty: Type<'_>) -> Option<&'static str> {
    SCALARS
        .iter()
        .find(|scalar| (scalar.is)(ty))
        .map(|scalar| scalar.spelling)
}

/// How a type is written in source, for a diagnostic.
///
/// A reader wrote `int64` and `List[int64]`, not `!yz.int64` and
/// `!yz.list<!yz.int64>`. The MLIR spelling is the fallback, so a type with no source syntax still
/// prints as something.
#[must_use]
pub fn name(ty: Type<'_>) -> String {
    if let Some(list) = ListType::from_type(ty) {
        return format!("List[{}]", name(list.inner()));
    }

    if let Some(declaration) = StructType::from_type(ty) {
        return declaration.name().to_string();
    }

    if let Some(param) = ParamType::from_type(ty) {
        return param.name().to_string();
    }

    match scalar_name(ty) {
        Some(scalar) => scalar.to_string(),
        None => ty.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use melior::Context;
    use melior::ir::Type;

    use super::{
        BoolType, ErrorType, Float64Type, Int64Type, QueryType, RefType, StrType, UnresolvedType,
    };

    type Singleton = (fn(&Context) -> Type<'_>, fn(Type<'_>) -> bool);

    const SINGLETONS: [Singleton; 7] = [
        (Int64Type::get, Int64Type::is),
        (Float64Type::get, Float64Type::is),
        (BoolType::get, BoolType::is),
        (StrType::get, StrType::is),
        (UnresolvedType::get, UnresolvedType::is),
        (QueryType::get, QueryType::is),
        (ErrorType::get, ErrorType::is),
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

    #[test]
    fn a_place_holds_its_element_type() {
        let context = crate::context();
        let element = Int64Type::get(&context);
        let place: Type = RefType::new(&context, element).into();
        assert_eq!(place.to_string(), "!yzl.ref<!yz.int64>");
        let place = RefType::from_type(place).expect("a place is a ref");
        assert_eq!(place.element(), element);
        assert!(RefType::from_type(element).is_none());
    }
}
