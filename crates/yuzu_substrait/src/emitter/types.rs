use substrait::proto::{
    Type as SubstraitType,
    r#type::{self, Kind},
};
use yuzu_types::{Column, Type, TypeCtx, TypeId};

use crate::proto::nullable;

pub(crate) fn type_code(types: &TypeCtx, ty: TypeId) -> &'static str {
    match types.ty(ty) {
        Type::Int8 | Type::UInt8 => "i8",
        Type::Int16 | Type::UInt16 => "i16",
        Type::Int32 | Type::UInt32 => "i32",
        Type::Int64 | Type::UInt64 => "i64",
        Type::Float32 => "fp32",
        Type::Float64 => "fp64",
        Type::Bool => "bool",
        Type::String => "string",
        _ => unreachable!("no Substrait type code for this yuzu type"),
    }
}

pub(crate) fn emit_type(types: &TypeCtx, ty: TypeId) -> SubstraitType {
    let kind = match types.ty(ty) {
        Type::Int8 | Type::UInt8 => Kind::I8(r#type::I8 {
            nullability: nullable(),
            ..Default::default()
        }),
        Type::Int16 | Type::UInt16 => Kind::I16(r#type::I16 {
            nullability: nullable(),
            ..Default::default()
        }),
        Type::Int32 | Type::UInt32 => Kind::I32(r#type::I32 {
            nullability: nullable(),
            ..Default::default()
        }),
        Type::Int64 | Type::UInt64 => Kind::I64(r#type::I64 {
            nullability: nullable(),
            ..Default::default()
        }),
        Type::Float32 => Kind::Fp32(r#type::Fp32 {
            nullability: nullable(),
            ..Default::default()
        }),
        Type::Float64 => Kind::Fp64(r#type::Fp64 {
            nullability: nullable(),
            ..Default::default()
        }),
        Type::Bool => Kind::Bool(r#type::Boolean {
            nullability: nullable(),
            ..Default::default()
        }),
        Type::String => Kind::String(r#type::String {
            nullability: nullable(),
            ..Default::default()
        }),
        _ => unreachable!("no Substrait type for this yuzu type"),
    };
    SubstraitType { kind: Some(kind) }
}

pub(crate) fn row_columns(types: &TypeCtx, rel_ty: TypeId) -> &[Column] {
    let Type::Relation(relation) = types.ty(rel_ty) else {
        unreachable!("a pipeline stage always has a relation type")
    };
    &relation.columns
}
