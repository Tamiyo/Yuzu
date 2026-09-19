//! Rows: what a relation's columns are, and the struct symbol standing for
//! a shape. A shape nobody declared is declared once, and the symbol
//! table's renaming of a collision is what interns it.

use melior::ir::attribute::{
    ArrayAttribute, FlatSymbolRefAttribute, StringAttribute, TypeAttribute,
};
use melior::ir::{Attribute, BlockLike, BlockRef, Location, Type, Value};
use yuzu_mlir::diagnostics::emit_error;
use yuzu_mlir::ext::{BlockExt, OperationCast, OperationExt};
use yuzu_mlir::ods::{yz, yzr};
use yuzu_mlir::ops::yzl::YzlOp;
use yuzu_mlir::{StructType, SymbolTable};

use crate::lower_yzl_to_yzr::{Row, YzlToYzr, struct_fields};

impl<'c, 'a> YzlToYzr<'c, 'a> {
    /// Seeds the shape index with what the program declared, so a stage
    /// whose row matches a declared struct reuses its name.
    pub(super) fn intern_declared_shapes(&mut self, block: BlockRef<'c, '_>) {
        for op in block.operations() {
            if let Some(YzlOp::Struct(item)) = op.as_yzl() {
                self.shapes
                    .entry(struct_fields(&item))
                    .or_insert_with(|| item.sym_name().value());
            }
        }
    }

    /// The row a relation's rows have: the table names its struct, and the
    /// struct carries its fields.
    fn relation_row(&self, name: &str, source: &SymbolTable<'c, '_>) -> Option<Row<'c>> {
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

    /// The type standing for a row, declaring the shape when nothing has.
    pub(super) fn row_type(
        &mut self,
        row: &Row<'c>,
        symbols: &mut SymbolTable<'c, '_>,
    ) -> Type<'c> {
        if let Some(&name) = self.shapes.get(row) {
            return StructType::new(self.context, name).into();
        }

        let name = self.declare_struct("row", row, symbols);
        self.shapes.insert(row.clone(), name);
        StructType::new(self.context, name).into()
    }

    /// Declares a struct in the lowered module, returning the name it got.
    pub(super) fn declare_struct(
        &self,
        name: &str,
        fields: &Row<'c>,
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
            ArrayAttribute::new(self.context, &names),
            ArrayAttribute::new(self.context, &types),
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
    ) -> Option<(Value<'c, 'a>, Row<'c>)> {
        if let Some(bound) = self.bindings.get(name) {
            return Some(bound.clone());
        }

        let Some(row) = self.relation_row(name, source) else {
            emit_error(location, &format!("`{name}` has no row shape to scan"));
            return None;
        };

        let ty = self.row_type(&row, symbols);
        let scanned = target.append_operation(
            yzr::table(
                self.context,
                ty,
                FlatSymbolRefAttribute::new(self.context, name),
                location,
            )
            .into(),
        );

        Some((scanned.first_result(), row))
    }
}
