//! What an enum-like attribute means, as a Rust type.
//!
//! ODS gives the op generator no enumerated attribute to make a type from, so
//! these are stored as plain strings. Rather than let each pass test string
//! literals, the spellings live here once and `build.rs` leaves the listed
//! accessor out of the generated view — the getter beside each type stands in
//! its place, under the name the attribute already had.
//!
//! The sibling of `types`: that says what a yz type is in Rust, this says
//! what an attribute's value is.

mod callee_kind;
mod cmp_predicate;
mod join_kind;

pub use callee_kind::CalleeKind;
pub use cmp_predicate::CmpPredicate;
pub use join_kind::JoinKind;
