mod check_aggregates;
mod check_mutability;
mod infer_types;
mod inline_calls;
mod lower_ast_to_yzl;
mod lower_yzl_to_yzr;
mod promote_locals;
mod remove_dead_symbols;
mod simplify_yzr;
#[cfg(test)]
mod test_support;

pub use check_aggregates::check_aggregates;
pub use check_mutability::check_mutability;
pub use infer_types::infer_types;
pub use inline_calls::inline_calls;
pub use lower_ast_to_yzl::{File, PRELUDE, lower_ast_to_yzl};
pub use lower_yzl_to_yzr::lower_yzl_to_yzr;
pub use promote_locals::promote_locals;
pub use remove_dead_symbols::remove_dead_symbols;
pub use simplify_yzr::simplify_yzr;
