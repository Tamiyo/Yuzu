use std::collections::HashSet;

use melior::ir::operation::OperationLike;
use melior::ir::{BlockLike, BlockRef, Location, Module, ValueLike};
use yuzu_ast::{AstNode, ast};
use yuzu_diagnostics::source_map::SourceId;
use yuzu_mlir::SymbolTable as MlirSymbolTable;
use yuzu_mlir::ext::{BlockExt, OperationExt};
use yuzu_mlir::ods::yzl;
use yuzu_mlir::types;
use yuzu_syntax::SyntaxKind;

use crate::lower_ast_to_yzl::AstToYzl;
use crate::lower_ast_to_yzl::symbols::SymbolTable;

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
            self.enter(file);
            self.bind_imports(&file.root);
            self.hoist(&file.root);
            self.record_declarations(&file.root);
            self.leave();
        }

        let reached = self.reached(files);
        let top = module.body();
        {
            // The module's own symbol table renames a name taken twice, so a
            // second `let` under one gets a symbol without us minting it.
            let mut symbols = MlirSymbolTable::new(&module);
            for file in files {
                self.enter(file);
                for stmt in file.root.stmts() {
                    if file.module.is_none() || self.is_reached(&reached, &stmt) {
                        self.convert_stmt(top, &stmt, &mut symbols);
                    }
                }

                self.leave();
            }
        }

        self.convert_output(top);
        module
    }

    /// Moves what the file declared out of `exports` and into the table the
    /// walk resolves against. `leave` moves it back, so one of the two holds
    /// it at a time.
    fn enter(&mut self, file: &File) {
        self.source_id = file.source_id;
        self.module = file.module.as_deref().map(|module| self.intern(module));
        let declarations = self.exports.remove(&self.module).unwrap_or_default();
        self.symbols = SymbolTable::over(declarations);
    }

    fn leave(&mut self) {
        self.exports.insert(self.module, self.symbols.take_module());
    }

    fn record_declarations(&mut self, root: &ast::Root) {
        for stmt in root.stmts() {
            let Some(symbol) = self
                .ident(declared_name(&stmt))
                .and_then(|name| self.symbols.binding(name))
                .and_then(|binding| binding.kind.symbol())
            else {
                continue;
            };

            self.declarations.insert(symbol, (self.module, stmt));
        }
    }

    /// The symbols the program reaches. Every identifier in a declaration
    /// counts as a reference, which says yes too often and never too seldom.
    fn reached(&self, files: &[File]) -> HashSet<&'c str> {
        let mut reached = HashSet::new();
        let mut pending: Vec<(Option<&'c str>, ast::Stmt)> = files
            .iter()
            .filter(|file| file.module.is_none())
            .flat_map(|file| file.root.stmts().map(|stmt| (None, stmt)))
            .collect();

        while let Some((scope, stmt)) = pending.pop() {
            let identifiers = stmt
                .syntax()
                .descendants_with_tokens()
                .filter_map(|element| element.into_token())
                .filter(|token| token.kind() == SyntaxKind::Identifier);
            for token in identifiers {
                let Some(symbol) = self
                    .exports
                    .get(&scope)
                    .and_then(|bindings| bindings.get(token.text()))
                    .and_then(|binding| binding.kind.symbol())
                else {
                    continue;
                };

                if reached.insert(symbol)
                    && let Some(declaration) = self.declarations.get(symbol)
                {
                    pending.push(declaration.clone());
                }
            }
        }

        reached
    }

    fn is_reached(&self, reached: &HashSet<&'c str>, stmt: &ast::Stmt) -> bool {
        let Some(name) = self.ident(declared_name(stmt)) else {
            return true;
        };

        self.symbols
            .binding(name)
            .and_then(|binding| binding.kind.symbol())
            .is_some_and(|symbol| reached.contains(symbol))
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

fn declared_name(stmt: &ast::Stmt) -> Option<ast::Ident> {
    match stmt {
        ast::Stmt::StructStmt(decl) => decl.name(),
        ast::Stmt::TableStmt(decl) => decl.name(),
        ast::Stmt::FuncStmt(decl) => decl.name(),
        ast::Stmt::TraitStmt(decl) => decl.name(),
        ast::Stmt::LetStmt(decl) => decl.name(),
        _ => None,
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
              yzl.table @h of @helpers.Row
              yzl.struct @Row ["b"] : [!yz.int64]
              yzl.table @t of @Row
              %0 = yzl.from @t
              %1 = yzl.select %0 as ["v"] {
              ^bb0(%arg0: !yzl.var):
                %2 = yzl.call @helpers.double(%arg0) : (!yzl.var) -> !yzl.var {callee_kind = "fn"}
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
    fn a_module_builds_only_what_the_program_reaches() {
        let module = lowered_program(&[
            (
                "helpers.yz",
                Some("helpers"),
                "pub def used(x: int64) -> int64 { return x * 2 }\npub def unused(x: int64) -> int64 { return x + 99 }\n",
            ),
            (
                "main.yz",
                None,
                "from helpers import used\n\nstruct Row { a: int64 }\ntable t = Row\n\nfrom t |> select used(a) as v\n",
            ),
        ]);
        assert!(
            module.contains("@helpers.used") && !module.contains("@helpers.unused"),
            "the reached one is built and the other is not:\n{module}"
        );
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
