use mlir_sys::{MlirAttribute, MlirContext, MlirLocation, MlirStringRef, MlirType};

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
    pub fn yzuTypeIsInt64Type(ty: MlirType) -> bool;
    pub fn yzuFloat64TypeGet(ctx: MlirContext) -> MlirType;
    pub fn yzuTypeIsFloat64Type(ty: MlirType) -> bool;
    pub fn yzuBoolTypeGet(ctx: MlirContext) -> MlirType;
    pub fn yzuTypeIsBoolType(ty: MlirType) -> bool;
    pub fn yzuStrTypeGet(ctx: MlirContext) -> MlirType;
    pub fn yzuTypeIsStrType(ty: MlirType) -> bool;
    pub fn yzuUnresolvedTypeGet(ctx: MlirContext) -> MlirType;
    pub fn yzuTypeIsUnresolvedType(ty: MlirType) -> bool;
    pub fn yzuQueryTypeGet(ctx: MlirContext) -> MlirType;
    pub fn yzuTypeIsQueryType(ty: MlirType) -> bool;
    pub fn yzuRefTypeGet(ctx: MlirContext) -> MlirType;
    pub fn yzuTypeIsRefType(ty: MlirType) -> bool;

    pub fn yzuFileLineColRangeGet(
        filename: MlirAttribute,
        start_line: u32,
        start_column: u32,
        end_line: u32,
        end_column: u32,
    ) -> MlirLocation;
}

pub fn register_all(ctx: MlirContext) {
    // SAFETY: the context is a live MLIR context; registration adds dialects to it and touches nothing else.
    unsafe { yzuRegisterAllDialects(ctx) }
}
