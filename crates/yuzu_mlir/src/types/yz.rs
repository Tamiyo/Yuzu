//! The `yz` dialect's types: the scalars, lists and structs a program's
//! values have.

use melior::Context;
use melior::StringRef;
use melior::ir::{Type, TypeLike};

/// `!yz.int64`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Int64Type<'c>(Type<'c>);

impl<'c> Int64Type<'c> {
    /// The type, uniqued in the context, which must have loaded the dialects.
    #[must_use]
    pub fn new(context: &'c Context) -> Self {
        super::debug_assert_loaded(context, "yz.constant_int");
        // SAFETY: the context is live for `'c` and the type is uniqued in it, so the raw type lives as long as the context.
        unsafe {
            Self(Type::from_raw(yuzu_mlir_sys::yzuInt64TypeGet(
                context.to_raw(),
            )))
        }
    }

    /// The view of a type, when it is this type.
    #[must_use]
    pub fn from_type(ty: Type<'c>) -> Option<Self> {
        // SAFETY: `ty` is a live type; the query only reads it.
        unsafe { yuzu_mlir_sys::yzuTypeIsInt64Type(ty.to_raw()) }.then_some(Self(ty))
    }
}

impl<'c> From<Int64Type<'c>> for Type<'c> {
    fn from(ty: Int64Type<'c>) -> Self {
        ty.0
    }
}

/// `!yz.float64`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Float64Type<'c>(Type<'c>);

impl<'c> Float64Type<'c> {
    /// The type, uniqued in the context, which must have loaded the dialects.
    #[must_use]
    pub fn new(context: &'c Context) -> Self {
        super::debug_assert_loaded(context, "yz.constant_int");
        // SAFETY: the context is live for `'c` and the type is uniqued in it, so the raw type lives as long as the context.
        unsafe {
            Self(Type::from_raw(yuzu_mlir_sys::yzuFloat64TypeGet(
                context.to_raw(),
            )))
        }
    }

    /// The view of a type, when it is this type.
    #[must_use]
    pub fn from_type(ty: Type<'c>) -> Option<Self> {
        // SAFETY: `ty` is a live type; the query only reads it.
        unsafe { yuzu_mlir_sys::yzuTypeIsFloat64Type(ty.to_raw()) }.then_some(Self(ty))
    }
}

impl<'c> From<Float64Type<'c>> for Type<'c> {
    fn from(ty: Float64Type<'c>) -> Self {
        ty.0
    }
}

/// `!yz.bool`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BoolType<'c>(Type<'c>);

impl<'c> BoolType<'c> {
    /// The type, uniqued in the context, which must have loaded the dialects.
    #[must_use]
    pub fn new(context: &'c Context) -> Self {
        super::debug_assert_loaded(context, "yz.constant_int");
        // SAFETY: the context is live for `'c` and the type is uniqued in it, so the raw type lives as long as the context.
        unsafe {
            Self(Type::from_raw(yuzu_mlir_sys::yzuBoolTypeGet(
                context.to_raw(),
            )))
        }
    }

    /// The view of a type, when it is this type.
    #[must_use]
    pub fn from_type(ty: Type<'c>) -> Option<Self> {
        // SAFETY: `ty` is a live type; the query only reads it.
        unsafe { yuzu_mlir_sys::yzuTypeIsBoolType(ty.to_raw()) }.then_some(Self(ty))
    }
}

impl<'c> From<BoolType<'c>> for Type<'c> {
    fn from(ty: BoolType<'c>) -> Self {
        ty.0
    }
}

/// `!yz.str`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct StrType<'c>(Type<'c>);

impl<'c> StrType<'c> {
    /// The type, uniqued in the context, which must have loaded the dialects.
    #[must_use]
    pub fn new(context: &'c Context) -> Self {
        super::debug_assert_loaded(context, "yz.constant_int");
        // SAFETY: the context is live for `'c` and the type is uniqued in it, so the raw type lives as long as the context.
        unsafe {
            Self(Type::from_raw(yuzu_mlir_sys::yzuStrTypeGet(
                context.to_raw(),
            )))
        }
    }

    /// The view of a type, when it is this type.
    #[must_use]
    pub fn from_type(ty: Type<'c>) -> Option<Self> {
        // SAFETY: `ty` is a live type; the query only reads it.
        unsafe { yuzu_mlir_sys::yzuTypeIsStrType(ty.to_raw()) }.then_some(Self(ty))
    }
}

impl<'c> From<StrType<'c>> for Type<'c> {
    fn from(ty: StrType<'c>) -> Self {
        ty.0
    }
}

/// `!yz.unit`: what a function returns when it declares no result.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct UnitType<'c>(Type<'c>);

impl<'c> UnitType<'c> {
    /// The type, uniqued in the context, which must have loaded the dialects.
    #[must_use]
    pub fn new(context: &'c Context) -> Self {
        super::debug_assert_loaded(context, "yz.constant_int");
        // SAFETY: the context is live for `'c` and the type is uniqued in it, so the raw type lives as long as the context.
        unsafe {
            Self(Type::from_raw(yuzu_mlir_sys::yzuUnitTypeGet(
                context.to_raw(),
            )))
        }
    }

    /// The view of a type, when it is this type.
    #[must_use]
    pub fn from_type(ty: Type<'c>) -> Option<Self> {
        // SAFETY: `ty` is a live type; the query only reads it.
        unsafe { yuzu_mlir_sys::yzuTypeIsUnitType(ty.to_raw()) }.then_some(Self(ty))
    }
}

impl<'c> From<UnitType<'c>> for Type<'c> {
    fn from(ty: UnitType<'c>) -> Self {
        ty.0
    }
}

/// `!yz.list<inner>`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ListType<'c>(Type<'c>);

impl<'c> ListType<'c> {
    /// The type, uniqued in the context, which must have loaded the dialects.
    #[must_use]
    pub fn new(context: &'c Context, inner: Type<'c>) -> Self {
        super::debug_assert_loaded(context, "yz.constant_int");
        // SAFETY: the context is live for `'c` and the type is uniqued in it, so the raw type lives as long as the context.
        unsafe {
            Self(Type::from_raw(yuzu_mlir_sys::yzuListTypeGet(
                context.to_raw(),
                inner.to_raw(),
            )))
        }
    }

    /// The view of a type, when it is this type.
    #[must_use]
    pub fn from_type(ty: Type<'c>) -> Option<Self> {
        // SAFETY: `ty` is a live type; the query only reads it.
        unsafe { yuzu_mlir_sys::yzuTypeIsListType(ty.to_raw()) }.then_some(Self(ty))
    }

    /// The type of each element.
    #[must_use]
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
    /// The type, uniqued in the context, which must have loaded the dialects.
    #[must_use]
    pub fn new(context: &'c Context, name: &str) -> Self {
        super::debug_assert_loaded(context, "yz.constant_int");
        // SAFETY: the context is live for `'c` and the type is uniqued in it, so the raw type lives as long as the context.
        unsafe {
            Self(Type::from_raw(yuzu_mlir_sys::yzuStructTypeGet(
                context.to_raw(),
                StringRef::new(name).to_raw(),
            )))
        }
    }

    /// The view of a type, when it is this type.
    #[must_use]
    pub fn from_type(ty: Type<'c>) -> Option<Self> {
        // SAFETY: `ty` is a live type; the query only reads it.
        unsafe { yuzu_mlir_sys::yzuTypeIsStructType(ty.to_raw()) }.then_some(Self(ty))
    }

    /// The declaring symbol's name.
    ///
    /// # Panics
    ///
    /// Panics if the name is not valid UTF-8.
    #[must_use]
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
