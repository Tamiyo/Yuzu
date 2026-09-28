//! Substrait shapes the translation builds in more than one place.

use substrait::proto::{
    Expression, RelCommon,
    expression::{
        FieldReference, Literal, ReferenceSegment, RexType,
        field_reference::{ReferenceType, RootReference, RootType},
        literal::LiteralType,
        reference_segment,
    },
    rel_common::{Emit, EmitKind},
    r#type::Nullability,
};

/// Everything is nullable: the language has no nullability yet, and a plan
/// that claimed otherwise would promise the engine something the compiler
/// has not checked.
pub(crate) fn nullable() -> i32 {
    Nullability::Nullable as i32
}

/// A column of the row, by position.
/// A column's position in a row, as Substrait counts fields.
pub(crate) fn field_index(index: usize) -> i32 {
    i32::try_from(index).expect("a row has fewer than 2^31 columns")
}

pub(crate) fn selection(index: i32) -> Expression {
    Expression {
        rex_type: Some(RexType::Selection(Box::new(FieldReference {
            reference_type: Some(ReferenceType::DirectReference(ReferenceSegment {
                reference_type: Some(reference_segment::ReferenceType::StructField(Box::new(
                    reference_segment::StructField {
                        field: index,
                        child: None,
                    },
                ))),
            })),
            root_type: Some(RootType::RootReference(RootReference {})),
        }))),
    }
}

pub(crate) fn literal(value: LiteralType) -> Expression {
    Expression {
        rex_type: Some(RexType::Literal(Literal {
            literal_type: Some(value),
            ..Default::default()
        })),
    }
}

/// The columns a projection keeps, by position over its input's columns
/// followed by its own expressions.
pub(crate) fn emit_common(output_mapping: Vec<i32>) -> RelCommon {
    RelCommon {
        emit_kind: Some(EmitKind::Emit(Emit { output_mapping })),
        ..Default::default()
    }
}
