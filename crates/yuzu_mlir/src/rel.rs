//! The typed view of `!yzr.rel`: construction and inspection over the C seam,
//! with no textual round trip.

use melior::Context;
use melior::ir::{Type, TypeLike};

#[derive(Clone, Copy)]
pub struct RelType<'c>(Type<'c>);

impl<'c> RelType<'c> {
    pub fn new(context: &'c Context, columns: &[(&str, Type<'c>)]) -> Self {
        let names: Vec<mlir_sys::MlirStringRef> = columns
            .iter()
            .map(|(name, _)| melior::StringRef::new(name).to_raw())
            .collect();
        let types: Vec<mlir_sys::MlirType> = columns.iter().map(|(_, ty)| ty.to_raw()).collect();
        let raw = unsafe {
            yuzu_mlir_sys::yzuRelTypeGet(
                context.to_raw(),
                columns.len() as isize,
                names.as_ptr(),
                types.as_ptr(),
            )
        };
        Self(unsafe { Type::from_raw(raw) })
    }

    pub fn from_type(ty: Type<'c>) -> Option<Self> {
        unsafe { yuzu_mlir_sys::yzuTypeIsRelType(ty.to_raw()) }.then_some(Self(ty))
    }

    pub fn column_count(&self) -> usize {
        (unsafe { yuzu_mlir_sys::yzuRelTypeColumnCount(self.0.to_raw()) }) as usize
    }

    pub fn column_name(&self, index: usize) -> &'c str {
        let raw = unsafe { yuzu_mlir_sys::yzuRelTypeColumnName(self.0.to_raw(), index as isize) };
        let bytes = unsafe { std::slice::from_raw_parts(raw.data.cast(), raw.length) };
        std::str::from_utf8(bytes).expect("column names are utf-8")
    }

    pub fn column_type(&self, index: usize) -> Type<'c> {
        unsafe {
            Type::from_raw(yuzu_mlir_sys::yzuRelTypeColumnType(
                self.0.to_raw(),
                index as isize,
            ))
        }
    }
}

impl<'c> From<RelType<'c>> for Type<'c> {
    fn from(rel: RelType<'c>) -> Self {
        rel.0
    }
}
