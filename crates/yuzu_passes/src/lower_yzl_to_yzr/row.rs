//! Rows: what a relation's columns are, and the struct symbol standing
//! for a shape. A shape nobody declared is declared once, and the symbol
//! table's renaming of a collision is what interns it.
use melior::ir::attribute::{FlatSymbolRefAttribute, StringAttribute, TypeAttribute};
use melior::ir::{Attribute, BlockLike, BlockRef, Location, Type, Value};
use yuzu_mlir::ext::{BlockExt, OperationCast, OperationExt};
use yuzu_mlir::ods::{yz, yzr};
use yuzu_mlir::ops::yzl::YzlOp;
use yuzu_mlir::{StructType, SymbolTable};

use crate::lower_yzl_to_yzr::{Schema, YzlToYzr, struct_fields};

impl<'c, 'a> YzlToYzr<'c, 'a> {
    /// Declarations first, so a stage can ask for a relation's row before
    /// the walk reaches the op that declared it.
    /// Seeds the shape index with what the program declared, so a stage
    /// whose row matches a declared struct reuses its name instead of
    /// interning a second one. Name lookups go through the symbol table;
    /// only shape-to-symbol needs an index of its own.
    pub(super) fn intern_declared_shapes(&mut self, block: BlockRef<'c, '_>) {
        for op in block.operations() {
            if let Some(YzlOp::Struct(item)) = op.as_yzl() {
                let fields = struct_fields(&item);
                self.shapes
                    .entry(fields)
                    .or_insert_with(|| item.sym_name().value());
            }
        }
    }

    /// The columns a stage names, paired with what its region yielded.
    pub(super) fn named_row(&self, names: Vec<&'c str>, yielded: Vec<Type<'c>>) -> Schema<'c> {
        names.into_iter().zip(yielded).collect()
    }

    /// The row a relation's rows have, found the way MLIR finds any symbol:
    /// the table names its struct, and the struct carries its fields.
    fn relation_schema(&self, name: &str, source: &SymbolTable<'c, '_>) -> Option<Schema<'c>> {
        let table = source.lookup(name)?;
        let YzlOp::Table(table) = table.as_yzl()? else {
            return None;
        };

        let declaration = source.lookup(table.row().value())?;
        let YzlOp::Struct(item) = declaration.as_yzl()? else {
            return None;
        };

        Some(struct_fields(&item))
    }

    /// The type standing for a row — declaring the shape when nothing has.
    pub(super) fn row_type(
        &mut self,
        schema: &Schema<'c>,
        symbols: &mut SymbolTable<'c, '_>,
    ) -> Type<'c> {
        if let Some(name) = self.shapes.get(schema) {
            return StructType::new(self.context, name).into();
        }

        let name = match self.shapes.get(schema).cloned() {
            Some(name) => name,
            None => self.declare_struct("row", schema, symbols),
        };

        self.shapes.insert(schema.clone(), name);
        StructType::new(self.context, name).into()
    }

    /// Declares a struct in the lowered module, returning the name it got —
    /// the symbol table renames a collision, which is what interns a shape
    /// nobody declared.
    pub(super) fn declare_struct(
        &self,
        name: &str,
        fields: &Schema<'c>,
        symbols: &mut SymbolTable<'c, '_>,
    ) -> &'c str {
        let names: Vec<Attribute<'c>> = fields
            .iter()
            .map(|(column, _)| StringAttribute::new(self.context, column).into())
            .collect();
        let types: Vec<Attribute<'c>> = fields
            .iter()
            .map(|(_, ty)| TypeAttribute::new(*ty).into())
            .collect();

        let declaration = yz::r#struct(
            self.context,
            StringAttribute::new(self.context, name),
            melior::ir::attribute::ArrayAttribute::new(self.context, &names),
            melior::ir::attribute::ArrayAttribute::new(self.context, &types),
            Location::unknown(self.context),
        );

        let assigned = symbols.insert(declaration.into());
        StringAttribute::try_from(assigned)
            .expect("a symbol name is a string")
            .value()
    }

    /// The rows a relation name stands for: a `let` has already produced
    /// them, and a table is scanned where it is used.
    pub(super) fn relation_input(
        &mut self,
        name: &str,
        source: &SymbolTable<'c, '_>,
        target: BlockRef<'c, 'a>,
        symbols: &mut SymbolTable<'c, '_>,
        location: Location<'c>,
    ) -> Option<(Value<'c, 'a>, Schema<'c>)> {
        if let Some(bound) = self.bindings.get(name) {
            return Some(bound.clone());
        }

        let schema = self.relation_schema(name, source)?;
        let row = self.row_type(&schema, symbols);
        let scanned = target.append_operation(
            yzr::table(
                self.context,
                row,
                FlatSymbolRefAttribute::new(self.context, name),
                location,
            )
            .into(),
        );

        Some((scanned.first_result(), schema))
    }
}
