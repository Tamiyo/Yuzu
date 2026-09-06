use melior::Context;

mod diagnostics_bridge;
pub mod ods;

pub use diagnostics_bridge::DiagnosticsBridge;

pub fn context() -> Context {
    let context = Context::new();
    yuzu_mlir_sys::register_all(context.to_raw());
    context
}
