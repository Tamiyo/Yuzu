use melior::Context;

mod attr;
mod diagnostics_bridge;
pub mod ods;
pub mod ops;
mod rel;
mod types;
mod value;

pub use attr::array_elements;
pub use diagnostics_bridge::DiagnosticsBridge;
pub use rel::RelType;
pub use types::Types;
pub use value::value_id;

pub fn context() -> Context {
    let context = Context::new();
    yuzu_mlir_sys::register_all(context.to_raw());
    context
}
