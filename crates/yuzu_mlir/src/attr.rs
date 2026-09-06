//! Attribute helpers the melior wrappers do not cover.

use melior::ir::Attribute;
use melior::ir::attribute::AttributeLike;

/// The elements of an array attribute. melior 0.27's
/// `ArrayAttribute::try_from` tests the wrong predicate (dense i64 array), so
/// access goes through the C API directly.
pub fn array_elements<'c>(attribute: Attribute<'c>) -> Vec<Attribute<'c>> {
    if !attribute.is_array() {
        return Vec::new();
    }
    let raw = attribute.to_raw();
    let count = unsafe { mlir_sys::mlirArrayAttrGetNumElements(raw) };
    (0..count)
        .map(|index| unsafe { Attribute::from_raw(mlir_sys::mlirArrayAttrGetElement(raw, index)) })
        .collect()
}
