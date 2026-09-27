use melior::ir::attribute::StringAttribute;
use melior::ir::operation::OperationLike;
use melior::ir::{BlockLike, BlockRef, Location, Module, ValueLike};
use yuzu_ast::ast;
use yuzu_diagnostics::source_map::SourceId;
use yuzu_mlir::ir::block::BlockExt;
use yuzu_mlir::ir::operation::OperationExt;
use yuzu_mlir::ods::yzl;
use yuzu_mlir::types::QueryType;

use crate::lower_ast_to_yzl::symbols::ModulePath;
use crate::lower_ast_to_yzl::{AstToYzl, Locals};

/// One file of the program; the entry file has no module.
#[derive(Clone)]
pub struct File {
    pub(crate) source_id: SourceId,
    pub(crate) module: Option<String>,
    pub(crate) root: ast::Root,
}

impl File {
    pub fn new(source_id: SourceId, module: Option<String>, root: ast::Root) -> Self {
        Self {
            source_id,
            module,
            root,
        }
    }

    pub fn entry(source_id: SourceId, root: ast::Root) -> Self {
        Self::new(source_id, None, root)
    }

    /// The module this file is, or `None` for the entry file.
    pub fn module(&self) -> Option<&str> {
        self.module.as_deref()
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
        for file in files {
            self.set_file(file);
            let mut locals = Locals::new();
            for stmt in file.root.stmts() {
                self.convert_stmt(body, &mut locals, &stmt);
            }
        }

        self.convert_output(body);
        module
    }

    fn set_file(&mut self, file: &File) {
        self.source_id = file.source_id;
        self.file = StringAttribute::new(self.context, self.sources.name(file.source_id));
        self.symbols.set_module(match file.module.as_deref() {
            Some(module) => ModulePath::from_path(self.symbols.intern(module)),
            None => ModulePath::entry(),
        });
    }

    /// The program's result is its last top-level query.
    fn convert_output<'a>(&mut self, top: BlockRef<'c, 'a>) {
        let query = top
            .operations()
            .filter_map(|op| {
                let value = op.try_first_result()?;
                (value.r#type() == QueryType::get(self.context)).then(|| (value, op.location()))
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
              ^bb0(%arg0: !yzl.unresolved):
                %2 = yzl.local "x" param
                yzl.store %2, %arg0 : !yzl.unresolved
                %3 = yzl.load %2 : !yzl.unresolved
                %4 = yz.constant_int 2
                %5 = yz.mul %3, %4 : !yzl.unresolved, !yz.int64 -> !yzl.unresolved
                yzl.return %5 : !yzl.unresolved
              }
              yzl.table @h of @helpers.Row {sym_visibility = "private"}
              yzl.struct @Row ["b"] : [!yz.int64] {sym_visibility = "private"}
              yzl.table @t of @Row {sym_visibility = "private"}
              %0 = yzl.from @t
              %1 = yzl.select %0 as ["v"] {
              ^bb0(%arg0: !yzl.unresolved):
                %2 = yzl.call @helpers.double(%arg0) : (!yzl.unresolved) -> !yzl.unresolved {callee_source = "fn"}
                yzl.yield %2 : !yzl.unresolved
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
    fn a_binding_a_module_exports_can_be_imported() {
        expect![[r#"
            module {
              yzl.const @helpers.cap {
                %2 = yz.constant_int 40
                %3 = yz.constant_int 2
                %4 = yz.add %2, %3 : !yz.int64, !yz.int64 -> !yzl.unresolved
                yzl.yield %4 : !yzl.unresolved
              }
              yzl.struct @Row ["a"] : [!yz.int64] {sym_visibility = "private"}
              yzl.table @t of @Row {sym_visibility = "private"}
              %0 = yzl.from @t
              %1 = yzl.select %0 as ["v"] {
              ^bb0(%arg0: !yzl.unresolved):
                %2 = yzl.call @helpers.cap() : () -> !yzl.unresolved {callee_source = "const"}
                %3 = yz.add %arg0, %2 : !yzl.unresolved, !yzl.unresolved -> !yzl.unresolved
                yzl.yield %3 : !yzl.unresolved
              }
              yzl.output %1
            }
        "#]]
        .assert_eq(&lowered_program(&[
            ("helpers.yz", Some("helpers"), "pub let cap = 40 + 2\n"),
            (
                "main.yz",
                None,
                "from helpers import cap\n\nstruct Row { a: int64 }\ntable t = Row\n\nfrom t |> select a + cap as v\n",
            ),
        ]));
    }

    /// A named query is a view, so importing one reads from it by name.
    #[test]
    fn a_query_a_module_binds_is_a_relation_where_it_is_imported() {
        let module = lowered_program(&[
            (
                "helpers.yz",
                Some("helpers"),
                "pub struct Row { a: int64 }\npub table t = Row\npub let small = from t |> where a < 10\n",
            ),
            (
                "main.yz",
                None,
                "from helpers import small\n\nfrom small |> select a as v\n",
            ),
        ]);
        assert!(
            module.contains("yzl.const @helpers.small")
                && module.contains("yzl.from @helpers.small"),
            "the imported query is built once and read by name:\n{module}"
        );
    }

    #[test]
    fn a_prelude_name_is_found_without_an_import() {
        let module = lowered_program(&[
            (
                "prelude.yz",
                Some("yuzu.prelude"),
                "pub def two() -> int64 { return 2 }\n",
            ),
            (
                "main.yz",
                None,
                "struct Row { a: int64 }\ntable t = Row\n\nfrom t |> select a + two() as v\n",
            ),
        ]);
        assert!(module.contains("yzl.call @yuzu.prelude.two()"), "{module}");
    }

    #[test]
    fn a_file_declaration_comes_before_a_prelude_name() {
        let module = lowered_program(&[
            (
                "prelude.yz",
                Some("yuzu.prelude"),
                "pub def two() -> int64 { return 2 }\n",
            ),
            (
                "main.yz",
                None,
                "def two() -> int64 { return 20 }\nstruct Row { a: int64 }\ntable t = Row\n\nfrom t |> select a + two() as v\n",
            ),
        ]);
        assert!(
            module.contains("yzl.call @two()") && !module.contains("yzl.call @yuzu.prelude.two()"),
            "{module}"
        );
    }

    #[test]
    fn a_private_prelude_name_is_not_found() {
        expect![[r#"
            error: unresolved identifier `two`
             --> main.yz:4:22
              |
            4 | from t |> select a + two() as v
              |                      ^^^^^
        "#]]
        .assert_eq(&reported_program(&[
            (
                "prelude.yz",
                Some("yuzu.prelude"),
                "def two() -> int64 { return 2 }\n",
            ),
            (
                "main.yz",
                None,
                "struct Row { a: int64 }\ntable t = Row\n\nfrom t |> select a + two() as v\n",
            ),
        ]));
    }

    #[test]
    fn a_binding_a_module_keeps_to_itself_cannot_be_imported() {
        expect![[r#"
            error: `cap` is not public; `helpers` keeps it to itself
             --> main.yz:1:21
              |
            1 | from helpers import cap
              |                     ^^^
        "#]]
        .assert_eq(&reported_program(&[
            ("helpers.yz", Some("helpers"), "let cap = 42\n"),
            (
                "main.yz",
                None,
                "from helpers import cap\n\nstruct Row { a: int64 }\ntable t = Row\n\nfrom t |> select a as v\n",
            ),
        ]));
    }

    #[test]
    fn a_name_a_binding_and_a_declaration_both_take_is_reported() {
        expect![[r#"
            error: the binding `f` is already defined
             --> test.yz:2:1
              |
            2 | let f = 1
              | ^^^^^^^^^
              = note: also declared at test.yz:1:1
        "#]]
        .assert_eq(&reported_program(&[(
            "test.yz",
            None,
            "def f(x: int64) -> int64 { return x }\nlet f = 1\n",
        )]));
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
