//! Integer attributes built from a Rust integer.

use melior::Context;
use melior::ir::attribute::IntegerAttribute;
use melior::ir::r#type::IntegerType;

pub trait IntegerAttributeExt<'c> {
    #[must_use]
    fn from_i64(context: &'c Context, value: i64) -> IntegerAttribute<'c> {
        IntegerAttribute::new(IntegerType::new(context, 64).into(), value)
    }
}

impl<'c> IntegerAttributeExt<'c> for IntegerAttribute<'c> {}
