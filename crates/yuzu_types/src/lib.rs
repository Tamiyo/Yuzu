//! The builtin functions and the registry that names them for the frontend.

mod functions;
mod registry;

pub use functions::{AggFunc, BuiltinFunc, Func};
pub use registry::{Builtins, Chain, FunctionRegistry, FunctionRegistryEntry, chain};
