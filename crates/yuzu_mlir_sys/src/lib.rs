use mlir_sys::{MlirContext, MlirStringRef, MlirType};

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
    pub fn yzuVarTypeGet(ctx: MlirContext) -> MlirType;
    pub fn yzuQueryTypeGet(ctx: MlirContext) -> MlirType;
}

pub fn register_all(ctx: MlirContext) {
    unsafe { yzuRegisterAllDialects(ctx) }
}
