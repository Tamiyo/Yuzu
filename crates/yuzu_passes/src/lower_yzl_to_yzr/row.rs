use melior::Context;
use melior::ir::attribute::{
    ArrayAttribute, FlatSymbolRefAttribute, StringAttribute, TypeAttribute,
};
use melior::ir::operation::Operation;
use melior::ir::{Attribute, BlockRef, Location, Type, Value};
use yuzu_mlir::diagnostics::emit_error;
use yuzu_mlir::ir::block::BlockExt;
use yuzu_mlir::ir::operation::{OperationCast, OperationExt};
use yuzu_mlir::ods::{yz, yzr};
use yuzu_mlir::ops::yzl::YzlOp;
use yuzu_mlir::{StructType, SymbolTable};

use crate::lower_yzl_to_yzr::{Row, YzlToYzr, struct_fields};

impl<'c, 'a> YzlToYzr<'c, 'a> {
    /// A stage whose row matches a declared struct reuses its name. The
    /// fields are kept by name too: the struct is converted in place, so a
    /// table read later finds its row here.
    pub(super) fn intern_declared_shapes(&mut self, block: BlockRef<'c, '_>) {
        for op in block.operations() {
            if let Some(YzlOp::Struct(item)) = op.as_yzl() {
                let name = item.sym_name().value();
                let fields = struct_fields(&item);
                self.shapes.entry(fields.clone()).or_insert(name);
                self.declared.insert(name, fields);
            }
        }
    }

    fn relation_row(&self, name: &str, symbols: &SymbolTable<'c, 'a>) -> Option<Row<'c>> {
        let table = symbols.lookup(name)?;
        let YzlOp::Table(table) = table.as_yzl()? else {
            return None;
        };

        self.declared.get(table.row().value()).cloned()
    }

    pub(super) fn row_type(
        &mut self,
        row: &Row<'c>,
        symbols: &mut SymbolTable<'c, 'a>,
        location: Location<'c>,
    ) -> Type<'c> {
        if let Some(&name) = self.shapes.get(row) {
            return StructType::new(self.context, name).into();
        }

        let name = self.declare_struct("row", row, symbols, location);
        self.shapes.insert(row.clone(), name);
        StructType::new(self.context, name).into()
    }

    pub(super) fn declare_struct(
        &self,
        name: &str,
        fields: &Row<'c>,
        symbols: &mut SymbolTable<'c, 'a>,
        location: Location<'c>,
    ) -> &'c str {
        let placed = self.insert(struct_declaration(self.context, name, fields, location));
        let assigned = symbols.insert_placed(placed);
        StringAttribute::try_from(assigned)
            .expect("a symbol name is a string")
            .value()
    }

    /// A `let` has already produced its rows; a table is scanned where it is
    /// used.
    pub(super) fn relation_input(
        &mut self,
        name: &str,
        symbols: &mut SymbolTable<'c, 'a>,
        location: Location<'c>,
    ) -> Option<(Value<'c, 'a>, Row<'c>)> {
        if let Some(bound) = self.bindings.get(name) {
            return Some(bound.clone());
        }

        let Some(row) = self.relation_row(name, symbols) else {
            emit_error(location, &format!("`{name}` has no row shape to scan"));
            return None;
        };

        let ty = self.row_type(&row, symbols, location);
        let scanned = self.insert(
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

pub(super) fn struct_declaration<'c>(
    context: &'c Context,
    name: &str,
    fields: &Row<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    let names: Vec<Attribute<'c>> = fields
        .iter()
        .map(|(column, _)| StringAttribute::new(context, column).into())
        .collect();
    let types: Vec<Attribute<'c>> = fields
        .iter()
        .map(|(_, ty)| TypeAttribute::new(*ty).into())
        .collect();

    yz::r#struct(
        context,
        StringAttribute::new(context, name),
        ArrayAttribute::new(context, &names),
        ArrayAttribute::new(context, &types),
        location,
    )
    .into()
}
