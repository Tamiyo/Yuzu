use mlir_sys::{MlirContext, MlirStringRef, MlirType};

unsafe extern "C" {
    fn yzuRegisterAllDialects(ctx: MlirContext);
    pub fn yzuRelTypeGet(
        ctx: MlirContext,
        count: isize,
        names: *const MlirStringRef,
        types: *const MlirType,
    ) -> MlirType;
    pub fn yzuTypeIsRelType(ty: MlirType) -> bool;
    pub fn yzuRelTypeColumnCount(ty: MlirType) -> isize;
    pub fn yzuRelTypeColumnName(ty: MlirType, index: isize) -> MlirStringRef;
    pub fn yzuRelTypeColumnType(ty: MlirType, index: isize) -> MlirType;
}

pub fn register_all(ctx: MlirContext) {
    unsafe { yzuRegisterAllDialects(ctx) }
}
