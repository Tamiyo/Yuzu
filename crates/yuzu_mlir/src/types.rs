//! The dialects' types, each a typed view over the uniqued `Type`.
//!
//! `new` builds one in a context, and `from_type` tells one apart. MLIR
//! uniques a type in its context, so both return the same type every time.
//! `new` needs a context that has loaded the dialects, as [`crate::context`]
//! makes one. A debug build checks this, because MLIR does not.

mod yz;
mod yzl;

use melior::Context;
use melior::ir::Type;

pub use yz::{BoolType, Float64Type, Int64Type, ListType, StrType, StructType, UnitType};
pub use yzl::{ErrorType, ParamType, QueryType, RefType, UnresolvedType};

/// In a debug build, checks that `context` has loaded the dialect of `op`.
/// MLIR builds a type of a dialect that the context has not loaded, and
/// gives no error.
fn debug_assert_loaded(context: &Context, op: &str) {
    debug_assert!(
        context.is_registered_operation(op),
        "the context has not loaded the dialect of `{op}`; make it with `yuzu_mlir::context`"
    );
}

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
        get: |context| Int64Type::new(context).into(),
        is: |ty| Int64Type::from_type(ty).is_some(),
    },
    Scalar {
        spelling: "float64",
        get: |context| Float64Type::new(context).into(),
        is: |ty| Float64Type::from_type(ty).is_some(),
    },
    Scalar {
        spelling: "bool",
        get: |context| BoolType::new(context).into(),
        is: |ty| BoolType::from_type(ty).is_some(),
    },
    Scalar {
        spelling: "str",
        get: |context| StrType::new(context).into(),
        is: |ty| StrType::from_type(ty).is_some(),
    },
];

/// How source spells the list type, as in `List[int64]`.
pub const LIST: &str = "List";

/// How source spells each scalar type.
pub fn scalar_spellings() -> impl Iterator<Item = &'static str> {
    SCALARS.iter().map(|scalar| scalar.spelling)
}

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
/// The reader wrote `int64` and `List[int64]`, not `!yz.int64` and
/// `!yz.list<!yz.int64>`. A type with no source syntax falls back to its
/// MLIR spelling.
#[must_use]
pub fn name(ty: Type<'_>) -> String {
    if let Some(list) = ListType::from_type(ty) {
        return format!("{LIST}[{}]", name(list.inner()));
    }

    if let Some(declaration) = StructType::from_type(ty) {
        return declaration.name().to_string();
    }

    if let Some(param) = ParamType::from_type(ty) {
        return param.name().to_string();
    }

    if UnitType::from_type(ty).is_some() {
        return "unit".to_owned();
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
        BoolType, ErrorType, Float64Type, Int64Type, QueryType, RefType, StrType, UnitType,
        UnresolvedType,
    };

    type Singleton = (fn(&Context) -> Type<'_>, fn(Type<'_>) -> bool);

    const SINGLETONS: [Singleton; 8] = [
        (
            |context| Int64Type::new(context).into(),
            |ty| Int64Type::from_type(ty).is_some(),
        ),
        (
            |context| Float64Type::new(context).into(),
            |ty| Float64Type::from_type(ty).is_some(),
        ),
        (
            |context| BoolType::new(context).into(),
            |ty| BoolType::from_type(ty).is_some(),
        ),
        (
            |context| StrType::new(context).into(),
            |ty| StrType::from_type(ty).is_some(),
        ),
        (
            |context| UnitType::new(context).into(),
            |ty| UnitType::from_type(ty).is_some(),
        ),
        (
            |context| UnresolvedType::new(context).into(),
            |ty| UnresolvedType::from_type(ty).is_some(),
        ),
        (
            |context| QueryType::new(context).into(),
            |ty| QueryType::from_type(ty).is_some(),
        ),
        (
            |context| ErrorType::new(context).into(),
            |ty| ErrorType::from_type(ty).is_some(),
        ),
    ];

    #[test]
    #[should_panic(expected = "the context has not loaded the dialect")]
    fn a_type_needs_a_context_that_loaded_its_dialect() {
        let bare = Context::new();
        let _ = Int64Type::new(&bare);
    }

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
        let element = Int64Type::new(&context).into();
        let place: Type = RefType::new(&context, element).into();
        assert_eq!(place.to_string(), "!yzl.ref<!yz.int64>");
        let place = RefType::from_type(place).expect("a place is a ref");
        assert_eq!(place.element(), element);
        assert!(RefType::from_type(element).is_none());
    }
}
