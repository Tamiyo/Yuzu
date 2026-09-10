use melior::Context;

pub mod diagnostics;
pub mod ext;
pub mod ods;
pub mod ops;
mod param_type;
mod struct_type;
mod symbol_table;
pub mod types;
mod value;

pub use param_type::ParamType;
pub use struct_type::StructType;
pub use symbol_table::SymbolTable;
pub use value::value_id;

pub fn context() -> Context {
    let context = Context::new();
    yuzu_mlir_sys::register_all(context.to_raw());
    context
}
