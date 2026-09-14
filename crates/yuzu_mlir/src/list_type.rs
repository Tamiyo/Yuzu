//! The typed view of `!yz.list`.

use melior::ir::{Type, TypeLike};

#[derive(Clone, Copy)]
pub struct ListType<'c>(Type<'c>);

impl<'c> ListType<'c> {
    pub fn new(context: &'c melior::Context, inner: Type<'c>) -> Self {
        unsafe {
            Self(Type::from_raw(yuzu_mlir_sys::yzuListTypeGet(
                context.to_raw(),
                inner.to_raw(),
            )))
        }
    }

    pub fn from_type(ty: Type<'c>) -> Option<Self> {
        unsafe { yuzu_mlir_sys::yzuTypeIsListType(ty.to_raw()) }.then_some(Self(ty))
    }

    pub fn inner(&self) -> Type<'c> {
        unsafe { Type::from_raw(yuzu_mlir_sys::yzuListTypeInner(self.0.to_raw())) }
    }
}

impl<'c> From<ListType<'c>> for Type<'c> {
    fn from(list: ListType<'c>) -> Self {
        list.0
    }
}
