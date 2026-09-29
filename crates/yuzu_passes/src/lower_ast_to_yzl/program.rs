use melior::ir::attribute::StringAttribute;
use melior::ir::operation::OperationLike;
use melior::ir::{BlockLike, BlockRef, Location, Module, ValueLike};
use rustc_hash::FxHashMap;
use yuzu_ast::ast;
use yuzu_diagnostics::SourceId;
use yuzu_mlir::ir::block::BlockExt;
use yuzu_mlir::ir::operation::OperationExt;
use yuzu_mlir::ods::yzl;
use yuzu_mlir::types::QueryType;

use crate::lower_ast_to_yzl::symbols::ModulePath;
use crate::lower_ast_to_yzl::{AstToYzl, Locals};

/// One file of the program; the entry file has no module.
#[derive(Clone, Debug)]
pub struct File {
    pub(crate) source_id: SourceId,
    pub(crate) module: Option<String>,
    pub(crate) root: ast::Root,
    pub(crate) lowering: Lowering,
}

/// When a file's declarations are lowered.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Lowering {
    /// All of them, so an error in one that nothing uses is still reported.
    Eager,
    /// Each function, struct, table and trait only when a reference names
    /// it. The library is lowered this way: a program uses a few of its
    /// functions, and a declaration no one names would only be removed again.
    OnDemand,
}

impl File {
    #[must_use]
    pub fn new(source_id: SourceId, module: Option<String>, root: ast::Root) -> Self {
        Self {
            source_id,
            module,
            root,
            lowering: Lowering::Eager,
        }
    }

    #[must_use]
    pub fn entry(source_id: SourceId, root: ast::Root) -> Self {
        Self::new(source_id, None, root)
    }

    /// The module this file is, or `None` for the entry file.
    #[must_use]
    pub fn module(&self) -> Option<&str> {
        self.module.as_deref()
    }

    pub fn set_lowering(&mut self, lowering: Lowering) {
        self.lowering = lowering;
    }
}

impl<'c> AstToYzl<'c, '_> {
    pub(super) fn lower(&mut self, files: &[File], entry: &File) -> Module<'c> {
        let module = Module::new(Location::new(
            self.context,
            self.sources.name(entry.source_id),
            1,
            1,
        ));

        self.bind_names(files);

        let body = module.body();
        // Each declaration waits under its name, with its parameter count when
        // it is a function, for a reference to name it.
        let mut on_demand: FxHashMap<_, Vec<_>> = FxHashMap::default();
        for file in files {
            self.in_file(file, |this| {
                let mut locals = Locals::new();
                for stmt in file.root.stmts() {
                    if file.lowering == Lowering::OnDemand
                        && let Some(name) = this.read_ident(on_demand_name(&stmt))
                    {
                        let arity = match &stmt {
                            ast::Stmt::FuncStmt(decl) => Some(decl.params().count()),
                            _ => None,
                        };
                        on_demand
                            .entry(this.symbols.module().declares(name))
                            .or_default()
                            .push((file, stmt, arity));
                        continue;
                    }

                    this.convert_stmt(body, &mut locals, &stmt);
                }
            });
        }

        // A declaration lowered on demand may name others in turn.
        loop {
            let used = self.symbols.take_used();
            if used.is_empty() {
                break;
            }

            // A call lowers only the overload it calls; any other reference
            // lowers everything under the name.
            for used in used {
                let Some(waiting) = on_demand.get_mut(&used.at) else {
                    continue;
                };
                let named: Vec<_> = waiting
                    .extract_if(.., |(_, _, arity)| {
                        used.arity.is_none() || arity.is_none() || *arity == used.arity
                    })
                    .collect();
                for (file, stmt, _) in named {
                    self.in_file(file, |this| {
                        this.convert_stmt(body, &mut Locals::new(), &stmt);
                    });
                }
            }
        }

        self.convert_output(body);
        module
    }

    /// The program's result is its last top-level query.
    fn convert_output<'a>(&mut self, top: BlockRef<'c, 'a>) {
        let query = top
            .operations()
            .filter_map(|op| {
                let value = op.try_first_result()?;
                QueryType::from_type(value.r#type()).map(|_| (value, op.location()))
            })
            .last();

        if let Some((value, loc)) = query {
            top.append_operation(yzl::output(self.context, value, loc).into());
        }
    }

    /// Binds every file's names before any file is lowered.
    ///
    /// A reference may name a declaration further on, or in another file. A
    /// file the bound library already holds is not bound again.
    pub(super) fn bind_names(&mut self, files: &[File]) {
        for file in files {
            let module = self.file_module(file);
            if self.symbols.is_library_module(module) {
                continue;
            }

            self.in_file(file, |this| {
                this.bind_imports(&file.root);
                this.hoist_declarations(&file.root);
            });
        }
    }

    /// Runs a walk over one file. This is the only place the current file
    /// changes: spans, locations and name lookups all follow it. The walk
    /// starts with no scope open, and closes every scope it opens.
    fn in_file<T>(&mut self, file: &File, walk: impl FnOnce(&mut Self) -> T) -> T {
        self.source_id = file.source_id;
        self.file = StringAttribute::new(self.context, self.sources.name(file.source_id));
        let module = self.file_module(file);
        self.symbols.enter_module(module);

        let result = walk(self);
        debug_assert!(
            !self.symbols.has_open_scope(),
            "the walk over `{}` left a scope open",
            self.sources.name(file.source_id)
        );
        result
    }

    fn file_module(&self, file: &File) -> ModulePath<'c> {
        match file.module.as_deref() {
            Some(module) => ModulePath::from_path(self.symbols.intern(module)),
            None => ModulePath::entry(),
        }
    }
}

/// The name of a declaration that can wait until a reference names it. A
/// `let` cannot: the hoist leaves it pending, and a lookup of a pending
/// `let` is reported until its body is lowered.
fn on_demand_name(stmt: &ast::Stmt) -> Option<ast::Ident> {
    match stmt {
        ast::Stmt::StructStmt(decl) => decl.name(),
        ast::Stmt::TableStmt(decl) => decl.name(),
        ast::Stmt::FuncStmt(decl) => decl.name(),
        ast::Stmt::TraitStmt(decl) => decl.name(),
        ast::Stmt::LetStmt(_)
        | ast::Stmt::ImplStmt(_)
        | ast::Stmt::ExprStmt(_)
        | ast::Stmt::BlockStmt(_)
        | ast::Stmt::AssignStmt(_)
        | ast::Stmt::ReturnStmt(_)
        | ast::Stmt::ImportStmt(_)
        | ast::Stmt::FromImportStmt(_)
        | ast::Stmt::ModStmt(_) => None,
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
                %2 = yzl.local "x" param : !yzl.ref<!yz.int64>
                yzl.store %2, %arg0 : !yzl.ref<!yz.int64>, !yzl.unresolved
                %3 = yzl.load %2 : !yzl.ref<!yz.int64> -> !yzl.unresolved
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
    fn an_import_brings_the_public_overloads() {
        expect![[r"
            error: `f` expects 1 argument(s), found 0
             --> main.yz:6:31
              |
            6 | from t |> select f(a) as one, f() as zero
              |                               ^^^
        "]].assert_eq(&reported_program(&[
            (
                "helpers.yz",
                Some("helpers"),
                "pub def f(x: int64) -> int64 { return x }\ndef f() -> int64 { return 0 }\n",
            ),
            (
                "main.yz",
                None,
                "from helpers import f\n\nstruct Row { a: int64 }\ntable t = Row\n\nfrom t |> select f(a) as one, f() as zero\n",
            ),
        ]));
    }

    #[test]
    fn a_file_function_hides_the_prelude_overloads() {
        expect![[r"
            error: `two` expects 1 argument(s), found 0
             --> main.yz:5:31
              |
            5 | from t |> select two(a) as v, two() as w
              |                               ^^^^^
        "]].assert_eq(&reported_program(&[
            (
                "prelude.yz",
                Some("yuzu.prelude"),
                "pub def two() -> int64 { return 2 }\npub def two(x: int64) -> int64 { return x }\n",
            ),
            (
                "main.yz",
                None,
                "def two(x: int64) -> int64 { return 20 }\nstruct Row { a: int64 }\ntable t = Row\n\nfrom t |> select two(a) as v, two() as w\n",
            ),
        ]));
    }

    #[test]
    fn a_function_cannot_overload_an_import() {
        expect![[r"
            error: the function `f` is already defined
             --> main.yz:3:1
              |
            3 | def f() -> int64 { return 0 }
              | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              = note: also declared at main.yz:1:21
        "]]
        .assert_eq(&reported_program(&[
            (
                "helpers.yz",
                Some("helpers"),
                "pub def f(x: int64) -> int64 { return x }\n",
            ),
            (
                "main.yz",
                None,
                "from helpers import f\n\ndef f() -> int64 { return 0 }\n",
            ),
        ]));
    }

    #[test]
    fn a_private_prelude_name_is_not_found() {
        expect![[r"
            error: unresolved identifier `two`
             --> main.yz:4:22
              |
            4 | from t |> select a + two() as v
              |                      ^^^^^
        "]]
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
        expect![[r"
            error: `cap` is not public; `helpers` keeps it to itself
             --> main.yz:1:21
              |
            1 | from helpers import cap
              |                     ^^^
        "]]
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
        expect![[r"
            error: the binding `f` is already defined
             --> test.yz:2:1
              |
            2 | let f = 1
              | ^^^^^^^^^
              = note: also declared at test.yz:1:1
        "]]
        .assert_eq(&reported_program(&[(
            "test.yz",
            None,
            "def f(x: int64) -> int64 { return x }\nlet f = 1\n",
        )]));
    }

    #[test]
    fn a_module_cannot_hold_a_query() {
        expect![[r"
            error: a module cannot hold a query
             --> helpers.yz:4:1
              |
            4 | from t |> select a as v
              | ^^^^^^^^^^^^^^^^^^^^^^^
        "]]
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

    #[test]
    fn a_private_import_is_not_exported_again() {
        expect![[r"
            error: `f` is not public; `b` keeps it to itself
             --> main.yz:1:15
              |
            1 | from b import f
              |               ^
              = note: `b` imports `f`; `pub from` would export it again
        "]]
        .assert_eq(&reported_program(&[
            (
                "a.yz",
                Some("a"),
                "pub def f(x: int64) -> int64 { return x }\n",
            ),
            ("b.yz", Some("b"), "from a import f\n"),
            ("main.yz", None, "from b import f\n"),
        ]));
    }

    #[test]
    fn a_public_import_is_exported_again() {
        expect![[r""]].assert_eq(&reported_program(&[
            ("a.yz", Some("a"), "pub def f(x: int64) -> int64 { return x }\n"),
            ("b.yz", Some("b"), "pub from a import f\n"),
            (
                "main.yz",
                None,
                "from b import f\nstruct Row { a: int64 }\ntable t = Row\nfrom t |> select f(a) as v\n",
            ),
        ]));
    }

    #[test]
    fn a_call_lowers_only_the_overload_it_calls() {
        let module = crate::test_support::lowered(
            "struct Row { a: int64 }\ntable t = Row\n\nfrom t |> aggregate count() as n\n",
        );
        assert!(module.contains("@yuzu.prelude.count.0"), "{module}");
        assert!(!module.contains("@yuzu.prelude.count.1"), "{module}");
    }
}
