use mlir_sys::MlirContext;

unsafe extern "C" {
    fn yzuRegisterAllDialects(ctx: MlirContext);
}

pub fn register_all(ctx: MlirContext) {
    unsafe { yzuRegisterAllDialects(ctx) }
}
