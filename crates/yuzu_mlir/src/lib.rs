use melior::Context;

pub mod attributes;
pub mod diagnostics;
pub mod ir;
pub mod ods;
pub mod ops;
pub mod rewrite;
pub mod types;

/// A context without a thread pool.
///
/// A module holds one query and its library. For that little work, other
/// threads cost more than they save: a compile with the pool takes 2.5
/// times as long.
#[must_use]
pub fn context() -> Context {
    let context = Context::new_with_threading(false);
    // SAFETY: the context was just made and is live; loading the dialects touches nothing else.
    unsafe { yuzu_mlir_sys::yzuRegisterAllDialects(context.to_raw()) };
    context
}
