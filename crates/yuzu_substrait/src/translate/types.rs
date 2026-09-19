//! Yuzu's types as Substrait's.

use melior::Context;
use melior::ir::Type;
use substrait::proto::{
    Type as SubstraitType,
    r#type::{self, Kind},
};
use yuzu_mlir::types;

use crate::proto::nullable;

/// The code a Substrait function signature names this type by, as in
/// `add:i64_i64`. `None` for a type no signature can carry.
pub(crate) fn type_code(context: &Context, ty: Type<'_>) -> Option<&'static str> {
    Some(match kind(context, ty)? {
        Kind::I64(_) => "i64",
        Kind::Fp64(_) => "fp64",
        Kind::Bool(_) => "bool",
        Kind::String(_) => "string",
        _ => return None,
    })
}

/// `None` for a type Substrait has no equivalent for; the caller reports it
/// against the op that produced it.
pub(crate) fn emit_type(context: &Context, ty: Type<'_>) -> Option<SubstraitType> {
    Some(SubstraitType {
        kind: Some(kind(context, ty)?),
    })
}

fn kind(context: &Context, ty: Type<'_>) -> Option<Kind> {
    let kind = if ty == types::int64(context) {
        Kind::I64(r#type::I64 {
            nullability: nullable(),
            ..Default::default()
        })
    } else if ty == types::float64(context) {
        Kind::Fp64(r#type::Fp64 {
            nullability: nullable(),
            ..Default::default()
        })
    } else if ty == types::boolean(context) {
        Kind::Bool(r#type::Boolean {
            nullability: nullable(),
            ..Default::default()
        })
    } else if ty == types::str(context) {
        Kind::String(r#type::String {
            nullability: nullable(),
            ..Default::default()
        })
    } else {
        return None;
    };

    Some(kind)
}
