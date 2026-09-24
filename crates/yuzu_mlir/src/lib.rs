use melior::Context;

pub mod attributes;
pub mod diagnostics;
pub mod ext;
pub mod ods;
pub mod ops;
mod symbol_table;
pub mod types;

pub use symbol_table::SymbolTable;
pub use types::{ListType, ParamType, StructType};

pub fn context() -> Context {
    let context = Context::new();
    yuzu_mlir_sys::register_all(context.to_raw());
    context
}
