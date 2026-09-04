use mlir_sys::MlirContext;

unsafe extern "C" {
    fn yzuRegisterAllDialects(ctx: MlirContext);
}

/// Registers and loads every Yuzu dialect in the context.
pub fn register_all(ctx: MlirContext) {
    unsafe { yzuRegisterAllDialects(ctx) }
}
