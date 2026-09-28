//! The Yuzu language server: the LSP side of `yuzu_ide`. It turns protocol
//! positions into offsets, asks an `Analysis`, and turns the answer back.
//!
//! The layout follows rust-analyzer's `rust-analyzer` crate: `main_loop`
//! owns the state and dispatch, `handlers` answer one request each, and
//! `to_proto` and `from_proto` convert at the boundary.

mod capabilities;
mod checker;
mod diagnostics;
mod documents;
mod error;
mod from_proto;
mod global_state;
mod handlers;
mod line_index;
mod main_loop;
mod semantic_tokens;
mod to_proto;

pub use error::RunError;
pub use main_loop::run;
