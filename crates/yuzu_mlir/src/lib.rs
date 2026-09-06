use melior::Context;

pub mod ods;

/// A context with every Yuzu dialect registered and loaded.
pub fn context() -> Context {
    let context = Context::new();
    yuzu_mlir_sys::register_all(context.to_raw());
    context
}
