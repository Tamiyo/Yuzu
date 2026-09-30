mod check_aggregates;
mod check_mutability;
mod infer_types;
mod inline_calls;
mod legalize_operators;
mod lower_ast_to_yzl;
mod lower_yzl_to_yzr;
mod operators;
mod promote_locals;
mod remove_dead_symbols;
mod simplify_yzr;
#[cfg(test)]
mod test_support;

pub use check_aggregates::check_aggregates;
pub use check_mutability::check_mutability;
pub use infer_types::infer_types;
pub use inline_calls::inline_calls;
pub use legalize_operators::legalize_operators;
pub use lower_ast_to_yzl::{
    BoundLibrary, File, Lowering, NameKind, NameListener, NameTarget, NameUse, PRELUDE, ScopeEntry,
    bind_library, lower_ast_to_yzl, lower_ast_to_yzl_with_listener,
};
pub use lower_yzl_to_yzr::lower_yzl_to_yzr;
pub use promote_locals::promote_locals;
pub use remove_dead_symbols::remove_dead_symbols;
pub use simplify_yzr::simplify_yzr;

/// The name a declaration was written under, as a message shows it.
///
/// A symbol starts with its module's path, and an overloaded function's
/// symbol also ends in its parameter count. A name cannot start with a
/// digit, so a last piece that does is a count.
pub(crate) fn written_name(symbol: &str) -> &str {
    let mut pieces = symbol.rsplit('.');
    let last = pieces.next().unwrap_or(symbol);
    if last.starts_with(|c: char| c.is_ascii_digit()) {
        pieces.next().unwrap_or(last)
    } else {
        last
    }
}

/// A pass manager for one of the MLIR passes this crate runs. The driver
/// verifies the module once, after the lowering builds it; only a debug
/// build verifies it again after each pass, which is most of a pass's cost.
fn pass_manager(context: &melior::Context) -> melior::pass::PassManager<'_> {
    let passes = melior::pass::PassManager::new(context);
    passes.enable_verifier(cfg!(debug_assertions));
    passes
}
