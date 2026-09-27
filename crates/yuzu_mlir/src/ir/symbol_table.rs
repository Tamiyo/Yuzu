//! MLIR's symbol table over a module: symbol lookup and insertion.

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
            // SAFETY: the module is a live op for `'a`, which the table borrows; MLIR returns an owned handle that `drop` frees once.
            raw: unsafe { mlir_sys::mlirSymbolTableCreate(module.as_operation().to_raw()) },
            _module: PhantomData,
        }
    }

    pub fn lookup(&self, name: &str) -> Option<OperationRef<'c, 'a>> {
        // SAFETY: the table and the name outlive the call. A null result is checked before it becomes a reference, and the op found belongs to the borrowed module.
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
        // SAFETY: `into_raw` hands the operation to MLIR, which owns it from here. The returned attribute is owned by the context.
        unsafe {
            Attribute::from_raw(mlir_sys::mlirSymbolTableInsert(
                self.raw,
                operation.into_raw(),
            ))
        }
    }

    /// Adds a symbol op that is already in the module where it should
    /// stand, renaming it when its name is taken. MLIR moves an op only
    /// when it has no parent, so this one stays in place.
    pub fn insert_placed(&mut self, operation: OperationRef<'c, 'a>) -> Attribute<'c> {
        // SAFETY: the op is live in the borrowed module; the returned attribute is owned by the context.
        unsafe {
            Attribute::from_raw(mlir_sys::mlirSymbolTableInsert(
                self.raw,
                operation.to_raw(),
            ))
        }
    }

    /// Removes a symbol op from the table and erases it.
    pub fn erase(&mut self, operation: OperationRef<'c, '_>) {
        // SAFETY: the op is a live symbol of the borrowed module and is not used after this call.
        unsafe { mlir_sys::mlirSymbolTableErase(self.raw, operation.to_raw()) }
    }
}

impl Drop for SymbolTable<'_, '_> {
    fn drop(&mut self) {
        // SAFETY: `raw` came from `mlirSymbolTableCreate` and is destroyed exactly once, here.
        unsafe { mlir_sys::mlirSymbolTableDestroy(self.raw) }
    }
}

#[cfg(test)]
mod tests {
    use melior::ir::attribute::{ArrayAttribute, StringAttribute};
    use melior::ir::{BlockLike, Location, Module};

    use super::SymbolTable;
    use crate::ir::operation::OperationExt;

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

    #[test]
    fn a_placed_symbol_keeps_its_place_and_an_erased_one_frees_its_name() {
        let context = crate::context();
        let module = Module::new(Location::unknown(&context));
        let declaration = |name: &str| {
            crate::ods::yzl::r#struct(
                &context,
                StringAttribute::new(&context, name),
                ArrayAttribute::new(&context, &[]),
                ArrayAttribute::new(&context, &[]),
                Location::unknown(&context),
            )
            .into()
        };
        let mut symbols = SymbolTable::new(&module);
        let first = module.body().append_operation(declaration("a"));
        symbols.insert_placed(first);
        let second = module.body().append_operation(declaration("b"));
        symbols.insert_placed(second);

        symbols.erase(first);
        let third = module.body().append_operation(declaration("a"));
        let name = symbols.insert_placed(third);

        assert_eq!(StringAttribute::try_from(name).unwrap().value(), "a");
        assert!(symbols.lookup("a").is_some() && symbols.lookup("b").is_some());
    }
}
