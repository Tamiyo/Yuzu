use melior::Context;

pub mod attributes;
pub mod diagnostics;
pub mod ir;
pub mod ods;
pub mod ops;
pub mod types;

pub use ir::symbol_table::SymbolTable;
pub use types::{ListType, ParamType, StructType};

/// A context without a thread pool. A module holds one query and its
/// library, so handing the verifier's and the passes' work to other threads
/// costs more than the work: with the pool, a compile took 2.5 times as long.
pub fn context() -> Context {
    let context = Context::new_with_threading(false);
    yuzu_mlir_sys::register_all(context.to_raw());
    context
}
