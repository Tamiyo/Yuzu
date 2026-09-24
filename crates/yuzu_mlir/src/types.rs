//! The dialects' types. MLIR uniques a type in its context, so a
//! parameterless type is fetched with `get` and a parametrized one is a
//! typed view built with `new`; both return the same type every time.

use melior::Context;
use melior::StringRef;
use melior::ir::{Type, TypeLike};

/// `!yz.int64`.
pub struct Int64Type;

impl Int64Type {
    pub fn get(context: &Context) -> Type<'_> {
        // SAFETY: the context is live for the returned lifetime and the type is uniqued in it.
        unsafe { Type::from_raw(yuzu_mlir_sys::yzuInt64TypeGet(context.to_raw())) }
    }

    pub fn is(ty: Type<'_>) -> bool {
        // SAFETY: a type lives in its context, so the reference lives as long as the type.
        let context = unsafe { ty.context().to_ref() };
        ty == Self::get(context)
    }
}

/// `!yz.float64`.
pub struct Float64Type;

impl Float64Type {
    pub fn get(context: &Context) -> Type<'_> {
        // SAFETY: the context is live for the returned lifetime and the type is uniqued in it.
        unsafe { Type::from_raw(yuzu_mlir_sys::yzuFloat64TypeGet(context.to_raw())) }
    }

    pub fn is(ty: Type<'_>) -> bool {
        // SAFETY: a type lives in its context, so the reference lives as long as the type.
        let context = unsafe { ty.context().to_ref() };
        ty == Self::get(context)
    }
}

/// `!yz.bool`.
pub struct BoolType;

impl BoolType {
    pub fn get(context: &Context) -> Type<'_> {
        // SAFETY: the context is live for the returned lifetime and the type is uniqued in it.
        unsafe { Type::from_raw(yuzu_mlir_sys::yzuBoolTypeGet(context.to_raw())) }
    }

    pub fn is(ty: Type<'_>) -> bool {
        // SAFETY: a type lives in its context, so the reference lives as long as the type.
        let context = unsafe { ty.context().to_ref() };
        ty == Self::get(context)
    }
}

/// `!yz.str`.
pub struct StrType;

impl StrType {
    pub fn get(context: &Context) -> Type<'_> {
        // SAFETY: the context is live for the returned lifetime and the type is uniqued in it.
        unsafe { Type::from_raw(yuzu_mlir_sys::yzuStrTypeGet(context.to_raw())) }
    }

    pub fn is(ty: Type<'_>) -> bool {
        // SAFETY: a type lives in its context, so the reference lives as long as the type.
        let context = unsafe { ty.context().to_ref() };
        ty == Self::get(context)
    }
}

/// `!yzl.unresolved`, the unification variable.
pub struct UnresolvedType;

impl UnresolvedType {
    pub fn get(context: &Context) -> Type<'_> {
        // SAFETY: the context is live for the returned lifetime and the type is uniqued in it.
        unsafe { Type::from_raw(yuzu_mlir_sys::yzuUnresolvedTypeGet(context.to_raw())) }
    }

    pub fn is(ty: Type<'_>) -> bool {
        // SAFETY: a type lives in its context, so the reference lives as long as the type.
        let context = unsafe { ty.context().to_ref() };
        ty == Self::get(context)
    }
}

/// `!yzl.query`, a relation before its schema is known.
pub struct QueryType;

impl QueryType {
    pub fn get(context: &Context) -> Type<'_> {
        // SAFETY: the context is live for the returned lifetime and the type is uniqued in it.
        unsafe { Type::from_raw(yuzu_mlir_sys::yzuQueryTypeGet(context.to_raw())) }
    }

    pub fn is(ty: Type<'_>) -> bool {
        // SAFETY: a type lives in its context, so the reference lives as long as the type.
        let context = unsafe { ty.context().to_ref() };
        ty == Self::get(context)
    }
}

/// `!yz.list<inner>`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ListType<'c>(Type<'c>);

impl<'c> ListType<'c> {
    pub fn new(context: &'c Context, inner: Type<'c>) -> Self {
        // SAFETY: the context is live for `'c` and the type is uniqued in it, so the raw type lives as long as the context.
        unsafe {
            Self(Type::from_raw(yuzu_mlir_sys::yzuListTypeGet(
                context.to_raw(),
                inner.to_raw(),
            )))
        }
    }

    pub fn from_type(ty: Type<'c>) -> Option<Self> {
        // SAFETY: `ty` is a live type; the query only reads it.
        unsafe { yuzu_mlir_sys::yzuTypeIsListType(ty.to_raw()) }.then_some(Self(ty))
    }

    pub fn inner(&self) -> Type<'c> {
        // SAFETY: the inner type is a parameter of a uniqued type, so it lives as long as the context.
        unsafe { Type::from_raw(yuzu_mlir_sys::yzuListTypeInner(self.0.to_raw())) }
    }
}

impl<'c> From<ListType<'c>> for Type<'c> {
    fn from(ty: ListType<'c>) -> Self {
        ty.0
    }
}

/// `!yz.struct<@name>`, the row a struct declaration describes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct StructType<'c>(Type<'c>);

impl<'c> StructType<'c> {
    pub fn new(context: &'c Context, name: &str) -> Self {
        // SAFETY: the context is live for `'c` and the type is uniqued in it, so the raw type lives as long as the context.
        unsafe {
            Self(Type::from_raw(yuzu_mlir_sys::yzuStructTypeGet(
                context.to_raw(),
                StringRef::new(name).to_raw(),
            )))
        }
    }

    pub fn from_type(ty: Type<'c>) -> Option<Self> {
        // SAFETY: `ty` is a live type; the query only reads it.
        unsafe { yuzu_mlir_sys::yzuTypeIsStructType(ty.to_raw()) }.then_some(Self(ty))
    }

    /// The declaring symbol's name.
    pub fn name(&self) -> &'c str {
        // SAFETY: the name is stored beside the type in the context's uniquer, so it lives as long as `'c`.
        unsafe {
            StringRef::from_raw(yuzu_mlir_sys::yzuStructTypeName(self.0.to_raw()))
                .as_str()
                .expect("type names are utf-8")
        }
    }
}

impl<'c> From<StructType<'c>> for Type<'c> {
    fn from(ty: StructType<'c>) -> Self {
        ty.0
    }
}

/// `!yz.param<name>`, a type parameter inside a generic function.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ParamType<'c>(Type<'c>);

impl<'c> ParamType<'c> {
    pub fn new(context: &'c Context, name: &str) -> Self {
        // SAFETY: the context is live for `'c` and the type is uniqued in it, so the raw type lives as long as the context.
        unsafe {
            Self(Type::from_raw(yuzu_mlir_sys::yzuParamTypeGet(
                context.to_raw(),
                StringRef::new(name).to_raw(),
            )))
        }
    }

    pub fn from_type(ty: Type<'c>) -> Option<Self> {
        // SAFETY: `ty` is a live type; the query only reads it.
        unsafe { yuzu_mlir_sys::yzuTypeIsParamType(ty.to_raw()) }.then_some(Self(ty))
    }

    /// The parameter's name.
    pub fn name(&self) -> &'c str {
        // SAFETY: the name is stored beside the type in the context's uniquer, so it lives as long as `'c`.
        unsafe {
            StringRef::from_raw(yuzu_mlir_sys::yzuParamTypeName(self.0.to_raw()))
                .as_str()
                .expect("type names are utf-8")
        }
    }
}

impl<'c> From<ParamType<'c>> for Type<'c> {
    fn from(ty: ParamType<'c>) -> Self {
        ty.0
    }
}

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
