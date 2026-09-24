use mlir_sys::{MlirContext, MlirStringRef, MlirType};

// SAFETY: these are the symbols the C++ leaf in `cpp/` exports, with the signatures it declares; `build.rs` links that leaf into this crate.
unsafe extern "C" {
    fn yzuRegisterAllDialects(ctx: MlirContext);
    pub fn yzuParamTypeGet(ctx: MlirContext, name: MlirStringRef) -> MlirType;
    pub fn yzuTypeIsParamType(ty: MlirType) -> bool;
    pub fn yzuParamTypeName(ty: MlirType) -> MlirStringRef;
    pub fn yzuStructTypeGet(ctx: MlirContext, name: MlirStringRef) -> MlirType;
    pub fn yzuTypeIsStructType(ty: MlirType) -> bool;
    pub fn yzuStructTypeName(ty: MlirType) -> MlirStringRef;
    pub fn yzuListTypeGet(ctx: MlirContext, inner: MlirType) -> MlirType;
    pub fn yzuTypeIsListType(ty: MlirType) -> bool;
    pub fn yzuListTypeInner(ty: MlirType) -> MlirType;
    pub fn yzuInt64TypeGet(ctx: MlirContext) -> MlirType;
    pub fn yzuFloat64TypeGet(ctx: MlirContext) -> MlirType;
    pub fn yzuBoolTypeGet(ctx: MlirContext) -> MlirType;
    pub fn yzuStrTypeGet(ctx: MlirContext) -> MlirType;
    pub fn yzuUnresolvedTypeGet(ctx: MlirContext) -> MlirType;
    pub fn yzuQueryTypeGet(ctx: MlirContext) -> MlirType;
    pub fn yzuRefTypeGet(ctx: MlirContext) -> MlirType;
}

pub fn register_all(ctx: MlirContext) {
    // SAFETY: the context is a live MLIR context; registration adds dialects to it and touches nothing else.
    unsafe { yzuRegisterAllDialects(ctx) }
}
