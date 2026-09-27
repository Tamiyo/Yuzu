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
            .map(|index| {
                let index = i64::try_from(index).expect("an index fits in an i64");
                IntegerAttribute::new(i64, index).into()
            })
            .collect();
        ArrayAttribute::new(context, &indices)
    }

    fn elements(&self) -> impl Iterator<Item = Attribute<'c>> + use<'c, Self>;
    fn strings(&self) -> impl Iterator<Item = &'c str> + use<'c, Self>;
    fn symbols(&self) -> impl Iterator<Item = &'c str> + use<'c, Self>;
    fn types(&self) -> impl Iterator<Item = Type<'c>> + use<'c, Self>;
    fn indices(&self) -> impl Iterator<Item = usize> + use<'c, Self>;
}

impl<'c> ArrayAttributeExt<'c> for ArrayAttribute<'c> {
    fn elements(&self) -> impl Iterator<Item = Attribute<'c>> + use<'c> {
        let array = *self;
        (0..array.len())
            .map(move |index| array.element(index).expect("the element index is in range"))
    }

    fn strings(&self) -> impl Iterator<Item = &'c str> + use<'c> {
        self.elements().map(|element| {
            StringAttribute::try_from(element)
                .expect("the array holds only strings")
                .value()
        })
    }

    fn symbols(&self) -> impl Iterator<Item = &'c str> + use<'c> {
        self.elements().map(|element| {
            FlatSymbolRefAttribute::try_from(element)
                .expect("the array holds only symbols")
                .value()
        })
    }

    fn types(&self) -> impl Iterator<Item = Type<'c>> + use<'c> {
        self.elements().map(|element| {
            TypeAttribute::try_from(element)
                .expect("the array holds only types")
                .value()
        })
    }

    fn indices(&self) -> impl Iterator<Item = usize> + use<'c> {
        self.elements().map(|element| {
            let index = IntegerAttribute::try_from(element)
                .expect("the array holds only integers")
                .value();
            usize::try_from(index).expect("an index is not negative")
        })
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
        assert_eq!(array.strings().collect::<Vec<_>>(), ["a", "b"]);
    }

    #[test]
    fn symbols_round_trip() {
        let context = crate::context();
        let array = ArrayAttribute::from_symbols(&context, [String::from("helpers.f")]);
        assert_eq!(array.symbols().collect::<Vec<_>>(), ["helpers.f"]);
    }

    #[test]
    #[should_panic(expected = "the array holds only symbols")]
    fn an_element_of_another_kind_is_a_bug() {
        let context = crate::context();
        let array = ArrayAttribute::from_strings(&context, ["a"]);
        array.symbols().for_each(drop);
    }

    #[test]
    fn types_round_trip() {
        let context = crate::context();
        let int64 = Type::index(&context);
        let array = ArrayAttribute::from_types(&context, [int64, int64]);
        assert_eq!(array.types().collect::<Vec<_>>(), [int64, int64]);
    }

    #[test]
    fn indices_round_trip() {
        let context = crate::context();
        let array = ArrayAttribute::from_indices(&context, [2, 0, 1]);
        assert_eq!(array.indices().collect::<Vec<_>>(), [2, 0, 1]);
    }
}
