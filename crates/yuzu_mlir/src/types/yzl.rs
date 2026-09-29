//! The `yzl` dialect's types: what a value has before inference and
//! promotion finish.

use melior::Context;
use melior::StringRef;
use melior::ir::{Type, TypeLike};

/// `!yzl.unresolved`, the unification variable.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct UnresolvedType<'c>(Type<'c>);

impl<'c> UnresolvedType<'c> {
    #[must_use]
    pub fn new(context: &'c Context) -> Self {
        // SAFETY: the context is live for `'c` and the type is uniqued in it, so the raw type lives as long as the context.
        unsafe {
            Self(Type::from_raw(yuzu_mlir_sys::yzuUnresolvedTypeGet(
                context.to_raw(),
            )))
        }
    }

    #[must_use]
    pub fn from_type(ty: Type<'c>) -> Option<Self> {
        // SAFETY: `ty` is a live type; the query only reads it.
        unsafe { yuzu_mlir_sys::yzuTypeIsUnresolvedType(ty.to_raw()) }.then_some(Self(ty))
    }
}

impl<'c> From<UnresolvedType<'c>> for Type<'c> {
    fn from(ty: UnresolvedType<'c>) -> Self {
        ty.0
    }
}

/// `!yzl.error`, the type of a value an error left behind.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ErrorType<'c>(Type<'c>);

impl<'c> ErrorType<'c> {
    #[must_use]
    pub fn new(context: &'c Context) -> Self {
        // SAFETY: the context is live for `'c` and the type is uniqued in it, so the raw type lives as long as the context.
        unsafe {
            Self(Type::from_raw(yuzu_mlir_sys::yzuErrorTypeGet(
                context.to_raw(),
            )))
        }
    }

    #[must_use]
    pub fn from_type(ty: Type<'c>) -> Option<Self> {
        // SAFETY: `ty` is a live type; the query only reads it.
        unsafe { yuzu_mlir_sys::yzuTypeIsErrorType(ty.to_raw()) }.then_some(Self(ty))
    }
}

impl<'c> From<ErrorType<'c>> for Type<'c> {
    fn from(ty: ErrorType<'c>) -> Self {
        ty.0
    }
}

/// `!yzl.query`, a relation before its schema is known.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct QueryType<'c>(Type<'c>);

impl<'c> QueryType<'c> {
    #[must_use]
    pub fn new(context: &'c Context) -> Self {
        // SAFETY: the context is live for `'c` and the type is uniqued in it, so the raw type lives as long as the context.
        unsafe {
            Self(Type::from_raw(yuzu_mlir_sys::yzuQueryTypeGet(
                context.to_raw(),
            )))
        }
    }

    #[must_use]
    pub fn from_type(ty: Type<'c>) -> Option<Self> {
        // SAFETY: `ty` is a live type; the query only reads it.
        unsafe { yuzu_mlir_sys::yzuTypeIsQueryType(ty.to_raw()) }.then_some(Self(ty))
    }
}

impl<'c> From<QueryType<'c>> for Type<'c> {
    fn from(ty: QueryType<'c>) -> Self {
        ty.0
    }
}

/// `!yzl.ref<element>`, a place that holds the value of a local variable.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RefType<'c>(Type<'c>);

impl<'c> RefType<'c> {
    #[must_use]
    pub fn new(context: &'c Context, element: Type<'c>) -> Self {
        // SAFETY: the context is live for `'c` and the type is uniqued in it, so the raw type lives as long as the context.
        unsafe {
            Self(Type::from_raw(yuzu_mlir_sys::yzuRefTypeGet(
                context.to_raw(),
                element.to_raw(),
            )))
        }
    }

    #[must_use]
    pub fn from_type(ty: Type<'c>) -> Option<Self> {
        // SAFETY: `ty` is a live type; the query only reads it.
        unsafe { yuzu_mlir_sys::yzuTypeIsRefType(ty.to_raw()) }.then_some(Self(ty))
    }

    /// The type of the value the place holds.
    #[must_use]
    pub fn element(&self) -> Type<'c> {
        // SAFETY: the element is a parameter of a uniqued type, so it lives as long as the context.
        unsafe { Type::from_raw(yuzu_mlir_sys::yzuRefTypeElement(self.0.to_raw())) }
    }
}

impl<'c> From<RefType<'c>> for Type<'c> {
    fn from(ty: RefType<'c>) -> Self {
        ty.0
    }
}

/// `!yzl.param<name>`, a type parameter inside a generic function.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ParamType<'c>(Type<'c>);

impl<'c> ParamType<'c> {
    #[must_use]
    pub fn new(context: &'c Context, name: &str) -> Self {
        // SAFETY: the context is live for `'c` and the type is uniqued in it, so the raw type lives as long as the context.
        unsafe {
            Self(Type::from_raw(yuzu_mlir_sys::yzuParamTypeGet(
                context.to_raw(),
                StringRef::new(name).to_raw(),
            )))
        }
    }

    #[must_use]
    pub fn from_type(ty: Type<'c>) -> Option<Self> {
        // SAFETY: `ty` is a live type; the query only reads it.
        unsafe { yuzu_mlir_sys::yzuTypeIsParamType(ty.to_raw()) }.then_some(Self(ty))
    }

    /// The parameter's name.
    ///
    /// # Panics
    ///
    /// Panics if the name is not valid UTF-8.
    #[must_use]
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
