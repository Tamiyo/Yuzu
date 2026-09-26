//! Array attributes built from a Rust list and read back as one.

use melior::Context;
use melior::ir::attribute::{
    ArrayAttribute, FlatSymbolRefAttribute, IntegerAttribute, StringAttribute, TypeAttribute,
};
use melior::ir::r#type::IntegerType;
use melior::ir::{Attribute, Type};

/// Array attributes, built from a Rust list and read back as one. Strings
/// read back are borrowed: attribute strings are context-uniqued, so they
/// outlive any pass reading them.
pub trait ArrayAttributeExt<'c> {
    fn from_strings(
        context: &'c Context,
        names: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> ArrayAttribute<'c> {
        let strings: Vec<Attribute<'c>> = names
            .into_iter()
            .map(|name| StringAttribute::new(context, name.as_ref()).into())
            .collect();

        ArrayAttribute::new(context, &strings)
    }

    fn from_symbols(
        context: &'c Context,
        symbols: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> ArrayAttribute<'c> {
        let symbols: Vec<Attribute<'c>> = symbols
            .into_iter()
            .map(|symbol| FlatSymbolRefAttribute::new(context, symbol.as_ref()).into())
            .collect();
        ArrayAttribute::new(context, &symbols)
    }

    fn from_types(
        context: &'c Context,
        types: impl IntoIterator<Item = Type<'c>>,
    ) -> ArrayAttribute<'c> {
        let types: Vec<Attribute<'c>> = types
            .into_iter()
            .map(|ty| TypeAttribute::new(ty).into())
            .collect();
        ArrayAttribute::new(context, &types)
    }

    fn from_indices(
        context: &'c Context,
        indices: impl IntoIterator<Item = usize>,
    ) -> ArrayAttribute<'c> {
        let i64 = IntegerType::new(context, 64).into();
        let indices: Vec<Attribute<'c>> = indices
            .into_iter()
            .map(|index| IntegerAttribute::new(i64, index as i64).into())
            .collect();
        ArrayAttribute::new(context, &indices)
    }

    fn elements(&self) -> impl Iterator<Item = Attribute<'c>>;
    fn strings(&self) -> Vec<&'c str>;
    fn symbols(&self) -> Vec<&'c str>;
    fn types(&self) -> Vec<Type<'c>>;
    fn indices(&self) -> Vec<usize>;
}

impl<'c> ArrayAttributeExt<'c> for ArrayAttribute<'c> {
    fn elements(&self) -> impl Iterator<Item = Attribute<'c>> {
        let array = *self;
        (0..array.len())
            .map(move |index| array.element(index).expect("the element index is in range"))
    }

    fn strings(&self) -> Vec<&'c str> {
        self.elements()
            .filter_map(|element| StringAttribute::try_from(element).ok())
            .map(|string| string.value())
            .collect()
    }

    fn symbols(&self) -> Vec<&'c str> {
        self.elements()
            .filter_map(|element| FlatSymbolRefAttribute::try_from(element).ok())
            .map(|symbol| symbol.value())
            .collect()
    }

    fn types(&self) -> Vec<Type<'c>> {
        self.elements()
            .filter_map(|element| TypeAttribute::try_from(element).ok())
            .map(|attribute| attribute.value())
            .collect()
    }

    fn indices(&self) -> Vec<usize> {
        self.elements()
            .filter_map(|element| IntegerAttribute::try_from(element).ok())
            .map(|index| index.value() as usize)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use melior::ir::Type;
    use melior::ir::attribute::ArrayAttribute;

    use super::ArrayAttributeExt;

    #[test]
    fn strings_round_trip() {
        let context = crate::context();
        let array = ArrayAttribute::from_strings(&context, ["a", "b"]);
        assert_eq!(array.strings(), ["a", "b"]);
        assert!(array.symbols().is_empty());
    }

    #[test]
    fn symbols_round_trip() {
        let context = crate::context();
        let array = ArrayAttribute::from_symbols(&context, [String::from("helpers.f")]);
        assert_eq!(array.symbols(), ["helpers.f"]);
        assert!(array.strings().is_empty());
    }

    #[test]
    fn types_round_trip() {
        let context = crate::context();
        let int64 = Type::index(&context);
        let array = ArrayAttribute::from_types(&context, [int64, int64]);
        assert_eq!(array.types(), [int64, int64]);
    }

    #[test]
    fn indices_round_trip() {
        let context = crate::context();
        let array = ArrayAttribute::from_indices(&context, [2, 0, 1]);
        assert_eq!(array.indices(), [2, 0, 1]);
    }
}
