//! Yuzu's types as Substrait's.

use melior::ir::Type;
use substrait::proto::{
    Type as SubstraitType,
    r#type::{self, Kind},
};
use yuzu_mlir::types::{BoolType, Float64Type, Int64Type, StrType};

use crate::proto::nullable;

/// The code a Substrait function signature names this type by, as in
/// `add:i64_i64`. `None` for a type no signature can carry.
pub(crate) fn type_code(ty: Type<'_>) -> Option<&'static str> {
    Some(match kind(ty)? {
        Kind::I64(_) => "i64",
        Kind::Fp64(_) => "fp64",
        Kind::Bool(_) => "bool",
        Kind::String(_) => "string",
        _ => return None,
    })
}

/// `None` for a type Substrait has no equivalent for; the caller reports it
/// against the op that produced it.
pub(crate) fn emit_type(ty: Type<'_>) -> Option<SubstraitType> {
    Some(SubstraitType {
        kind: Some(kind(ty)?),
    })
}

fn kind(ty: Type<'_>) -> Option<Kind> {
    let kind = if Int64Type::from_type(ty).is_some() {
        Kind::I64(r#type::I64 {
            nullability: nullable(),
            ..Default::default()
        })
    } else if Float64Type::from_type(ty).is_some() {
        Kind::Fp64(r#type::Fp64 {
            nullability: nullable(),
            ..Default::default()
        })
    } else if BoolType::from_type(ty).is_some() {
        Kind::Bool(r#type::Boolean {
            nullability: nullable(),
            ..Default::default()
        })
    } else if StrType::from_type(ty).is_some() {
        Kind::String(r#type::String {
            nullability: nullable(),
            ..Default::default()
        })
    } else {
        return None;
    };

    Some(kind)
}
