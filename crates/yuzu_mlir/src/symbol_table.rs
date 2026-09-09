use std::marker::PhantomData;

use melior::StringRef;
use melior::ir::operation::{Operation, OperationRef};
use melior::ir::{Attribute, Module};

pub struct SymbolTable<'c, 'a> {
    raw: mlir_sys::MlirSymbolTable,
    _module: PhantomData<&'a Module<'c>>,
}

impl<'c, 'a> SymbolTable<'c, 'a> {
    pub fn new(module: &'a Module<'c>) -> Self {
        Self {
            raw: unsafe { mlir_sys::mlirSymbolTableCreate(module.as_operation().to_raw()) },
            _module: PhantomData,
        }
    }

    pub fn lookup(&self, name: &str) -> Option<OperationRef<'c, 'a>> {
        unsafe {
            let operation =
                mlir_sys::mlirSymbolTableLookup(self.raw, StringRef::new(name).to_raw());

            if operation.ptr.is_null() {
                None
            } else {
                Some(OperationRef::from_raw(operation))
            }
        }
    }

    pub fn insert(&mut self, operation: Operation<'c>) -> Attribute<'c> {
        unsafe {
            Attribute::from_raw(mlir_sys::mlirSymbolTableInsert(
                self.raw,
                operation.into_raw(),
            ))
        }
    }
}

impl Drop for SymbolTable<'_, '_> {
    fn drop(&mut self) {
        unsafe { mlir_sys::mlirSymbolTableDestroy(self.raw) }
    }
}

#[cfg(test)]
mod tests {
    use melior::ir::attribute::{ArrayAttribute, StringAttribute};
    use melior::ir::{BlockLike, Location, Module};

    use super::SymbolTable;
    use crate::ext::OperationExt;

    #[test]
    fn looks_up_declared_symbols() {
        let context = crate::context();
        let module = Module::parse(
            &context,
            r#"
module {
  yzl.struct @Row ["a"] : [!yz.int64]
  yzl.table @t of @Row
}
"#,
        )
        .expect("the module parses");

        let table = SymbolTable::new(&module);
        let row = table.lookup("Row").expect("the struct is declared");
        assert_eq!(row.text_attribute("sym_name").as_deref(), Some("Row"));
        assert!(table.lookup("t").is_some());
        assert!(table.lookup("Ghost").is_none());
    }

    #[test]
    fn insert_renames_a_colliding_symbol() {
        let context = crate::context();
        let location = Location::unknown(&context);
        let module = Module::new(location);

        let declaration = |name: &str| {
            crate::ods::yzl::r#struct(
                &context,
                StringAttribute::new(&context, name),
                ArrayAttribute::new(&context, &[]),
                ArrayAttribute::new(&context, &[]),
                location,
            )
            .into()
        };

        let mut table = SymbolTable::new(&module);
        let first = table.insert(declaration("Row"));
        let second = table.insert(declaration("Row"));

        assert_eq!(
            StringAttribute::try_from(first)
                .expect("names are strings")
                .value(),
            "Row"
        );
        let renamed = StringAttribute::try_from(second)
            .expect("names are strings")
            .value();
        assert_ne!(renamed, "Row", "the collision renames the second symbol");
        assert!(table.lookup(renamed).is_some());
        assert!(
            module.body().first_operation().is_some(),
            "inserted declarations land in the module body"
        );
    }
}
