use melior::Context;

pub mod attributes;
pub mod diagnostics;
pub mod ir;
pub mod ods;
pub mod ops;
pub mod types;

pub use ir::symbol_table::SymbolTable;
pub use types::{ListType, ParamType, StructType};

pub fn context() -> Context {
    let context = Context::new();
    yuzu_mlir_sys::register_all(context.to_raw());
    context
}
