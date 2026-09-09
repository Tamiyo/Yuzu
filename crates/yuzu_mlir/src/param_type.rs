//! The typed view of `!yzl.param`: a generic parameter in type position,
//! named by the program rather than minted by inference.

use melior::StringRef;
use melior::ir::{Type, TypeLike};

#[derive(Clone, Copy)]
pub struct ParamType<'c>(Type<'c>);

impl<'c> ParamType<'c> {
    pub fn new(context: &'c melior::Context, name: &str) -> Self {
        unsafe {
            Self(Type::from_raw(yuzu_mlir_sys::yzuParamTypeGet(
                context.to_raw(),
                StringRef::new(name).to_raw(),
            )))
        }
    }

    pub fn from_type(ty: Type<'c>) -> Option<Self> {
        unsafe { yuzu_mlir_sys::yzuTypeIsParamType(ty.to_raw()) }.then_some(Self(ty))
    }

    /// The parameter's name.
    pub fn name(&self) -> &'c str {
        unsafe {
            StringRef::from_raw(yuzu_mlir_sys::yzuParamTypeName(self.0.to_raw()))
                .as_str()
                .expect("parameter names are utf-8")
        }
    }
}

impl<'c> From<ParamType<'c>> for Type<'c> {
    fn from(param: ParamType<'c>) -> Self {
        param.0
    }
}
