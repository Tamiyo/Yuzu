use melior::Context;

pub mod attributes;
pub mod diagnostics;
pub mod ext;
mod list_type;
pub mod ods;
pub mod ops;
mod param_type;
mod struct_type;
mod symbol_table;
pub mod types;

pub use list_type::ListType;
pub use param_type::ParamType;
pub use struct_type::StructType;
pub use symbol_table::SymbolTable;

pub fn context() -> Context {
    let context = Context::new();
    yuzu_mlir_sys::register_all(context.to_raw());
    context
}
