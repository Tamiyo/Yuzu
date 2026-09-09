//! The typed view of `!yz.struct`: nominal, as in Mojo — the type carries
//! only the declaring symbol, and the field list lives on the declaring op.

use melior::StringRef;
use melior::ir::{Type, TypeLike};

#[derive(Clone, Copy)]
pub struct StructType<'c>(Type<'c>);

impl<'c> StructType<'c> {
    pub fn new(context: &'c melior::Context, name: &str) -> Self {
        unsafe {
            Self(Type::from_raw(yuzu_mlir_sys::yzuStructTypeGet(
                context.to_raw(),
                StringRef::new(name).to_raw(),
            )))
        }
    }

    pub fn from_type(ty: Type<'c>) -> Option<Self> {
        unsafe { yuzu_mlir_sys::yzuTypeIsStructType(ty.to_raw()) }.then_some(Self(ty))
    }

    /// The declaring symbol's name.
    pub fn name(&self) -> &'c str {
        unsafe {
            StringRef::from_raw(yuzu_mlir_sys::yzuStructTypeName(self.0.to_raw()))
                .as_str()
                .expect("struct names are utf-8")
        }
    }
}

impl<'c> From<StructType<'c>> for Type<'c> {
    fn from(declaration: StructType<'c>) -> Self {
        declaration.0
    }
}
