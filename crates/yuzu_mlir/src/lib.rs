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
/// A module holds one query and its library, so handing the verifier's and the passes' work to other threads
/// costs more than the work: with the pool, a compile took 2.5 times as long.
#[must_use]
pub fn context() -> Context {
    let context = Context::new_with_threading(false);
    // SAFETY: the context was just made and is live; loading the dialects touches nothing else.
    unsafe { yuzu_mlir_sys::yzuRegisterAllDialects(context.to_raw()) };
    context
}
