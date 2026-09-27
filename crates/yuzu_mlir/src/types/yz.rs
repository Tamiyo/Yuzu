//! The `yz` dialect's types: the scalars, lists and structs a program's
//! values have.

use melior::Context;
use melior::StringRef;
use melior::ir::{Type, TypeLike};

/// `!yz.int64`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Int64Type;

impl Int64Type {
    pub fn get(context: &Context) -> Type<'_> {
        // SAFETY: the context is live for the returned lifetime and the type is uniqued in it.
        unsafe { Type::from_raw(yuzu_mlir_sys::yzuInt64TypeGet(context.to_raw())) }
    }

    pub fn is(ty: Type<'_>) -> bool {
        // SAFETY: `ty` is a live type; the query only reads it.
        unsafe { yuzu_mlir_sys::yzuTypeIsInt64Type(ty.to_raw()) }
    }
}

/// `!yz.float64`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Float64Type;

impl Float64Type {
    pub fn get(context: &Context) -> Type<'_> {
        // SAFETY: the context is live for the returned lifetime and the type is uniqued in it.
        unsafe { Type::from_raw(yuzu_mlir_sys::yzuFloat64TypeGet(context.to_raw())) }
    }

    pub fn is(ty: Type<'_>) -> bool {
        // SAFETY: `ty` is a live type; the query only reads it.
        unsafe { yuzu_mlir_sys::yzuTypeIsFloat64Type(ty.to_raw()) }
    }
}

/// `!yz.bool`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BoolType;

impl BoolType {
    pub fn get(context: &Context) -> Type<'_> {
        // SAFETY: the context is live for the returned lifetime and the type is uniqued in it.
        unsafe { Type::from_raw(yuzu_mlir_sys::yzuBoolTypeGet(context.to_raw())) }
    }

    pub fn is(ty: Type<'_>) -> bool {
        // SAFETY: `ty` is a live type; the query only reads it.
        unsafe { yuzu_mlir_sys::yzuTypeIsBoolType(ty.to_raw()) }
    }
}

/// `!yz.str`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct StrType;

impl StrType {
    pub fn get(context: &Context) -> Type<'_> {
        // SAFETY: the context is live for the returned lifetime and the type is uniqued in it.
        unsafe { Type::from_raw(yuzu_mlir_sys::yzuStrTypeGet(context.to_raw())) }
    }

    pub fn is(ty: Type<'_>) -> bool {
        // SAFETY: `ty` is a live type; the query only reads it.
        unsafe { yuzu_mlir_sys::yzuTypeIsStrType(ty.to_raw()) }
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
