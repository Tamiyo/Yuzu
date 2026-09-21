use melior::ir::operation::OperationLike;
use melior::ir::{BlockLike, BlockRef, Location, Module, ValueLike};
use yuzu_ast::ast;
use yuzu_diagnostics::source_map::SourceId;
use yuzu_mlir::SymbolTable as MlirSymbolTable;
use yuzu_mlir::ext::{BlockExt, OperationExt};
use yuzu_mlir::ods::yzl;
use yuzu_mlir::types;

use crate::lower_ast_to_yzl::AstToYzl;
use crate::lower_ast_to_yzl::symbols::ModulePath;

/// One file of the program; the entry file has no module.
#[derive(Clone)]
pub struct File {
    pub source_id: SourceId,
    pub module: Option<String>,
    pub root: ast::Root,
}

impl File {
    pub fn entry(source_id: SourceId, root: ast::Root) -> Self {
        Self {
            source_id,
            module: None,
            root,
        }
    }
}

impl<'c, 'd> AstToYzl<'c, 'd> {
    pub(super) fn lower(&mut self, files: &[File], entry: &File) -> Module<'c> {
        let module = Module::new(Location::new(
            self.context,
            self.sources.name(entry.source_id),
            1,
            1,
        ));

        // Names first, for every file: a reference may point forward.
        for file in files {
            self.set_file(file);
            self.bind_imports(&file.root);
            self.hoist_declarations(&file.root);
        }

        let body = module.body();
        {
            // The module's own symbol table renames a name taken twice, so a
            // second `let` under one gets a symbol without us minting it.
            let mut mlir_symbols = MlirSymbolTable::new(&module);
            for file in files {
                self.set_file(file);
                for stmt in file.root.stmts() {
                    self.convert_stmt(body, &stmt, &mut mlir_symbols);
                }
            }
        }

        self.convert_output(body);
        module
    }

    fn set_file(&mut self, file: &File) {
        self.source_id = file.source_id;
        self.symbols.set_module(match file.module.as_deref() {
            Some(module) => ModulePath::of(self.intern(module)),
            None => ModulePath::entry(),
        });
    }

    /// The program's result is its last top-level query.
    fn convert_output<'a>(&mut self, top: BlockRef<'c, 'a>) {
        let query = top
            .operations()
            .filter_map(|op| {
                let value = op.try_first_result()?;
                (value.r#type() == types::query(self.context)).then(|| (value, op.location()))
            })
            .last();

        if let Some((value, loc)) = query {
            top.append_operation(yzl::output(self.context, value, loc).into());
        }
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::test_support::{lowered_program, reported_program};

    #[test]
    fn a_program_of_several_files_becomes_one_module() {
        expect![[r#"
            module {
              yzl.struct @helpers.Row ["a"] : [!yz.int64]
              yzl.fn @helpers.double params ["x"] (!yz.int64) -> !yz.int64 {
              ^bb0(%arg0: !yzl.var):
                %2 = yz.constant_int 2
                %3 = yz.mul %arg0, %2 : !yzl.var, !yz.int64 -> !yzl.var
                yzl.return %3 : !yzl.var
              }
              yzl.table @h of @helpers.Row {sym_visibility = "private"}
              yzl.struct @Row ["b"] : [!yz.int64] {sym_visibility = "private"}
              yzl.table @t of @Row {sym_visibility = "private"}
              %0 = yzl.from @t
              %1 = yzl.select %0 as ["v"] {
              ^bb0(%arg0: !yzl.var):
                %2 = yzl.call @helpers.double(%arg0) : (!yzl.var) -> !yzl.var {callee_source = "fn"}
                yzl.yield %2 : !yzl.var
              }
              yzl.output %1
            }
        "#]].assert_eq(&lowered_program(&[
            (
                "helpers.yz",
                Some("helpers"),
                "pub struct Row { a: int64 }\npub def double(x: int64) -> int64 { return x * 2 }\n",
            ),
            (
                "main.yz",
                None,
                "from helpers import double, Row as Shape\n\ntable h = Shape\nstruct Row { b: int64 }\ntable t = Row\n\nfrom t |> select double(b) as v\n",
            ),
        ]));
    }

    #[test]
    fn a_module_cannot_hold_a_query() {
        expect![[r#"
            error: a module cannot hold a query
             --> helpers.yz:4:1
              |
            4 | from t |> select a as v
              | ^^^^^^^^^^^^^^^^^^^^^^^
        "#]]
        .assert_eq(&reported_program(&[
            (
                "helpers.yz",
                Some("helpers"),
                "struct Row { a: int64 }\ntable t = Row\n\nfrom t |> select a as v\n",
            ),
            (
                "main.yz",
                None,
                "struct M { b: int64 }\ntable m = M\n\nfrom m |> select b as w\n",
            ),
        ]));
    }
}
