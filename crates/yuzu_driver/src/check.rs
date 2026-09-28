//! The front of the pipeline for an editor: a file's program loaded, lowered
//! and checked through `check_aggregates`, with every diagnostic kept and no
//! plan made.

use yuzu_diagnostics::diagnostics::Diagnostic;
use yuzu_diagnostics::diagnostics::engine::DiagnosticsEngine;
use yuzu_diagnostics::source_map::{SourceId, SourceMap};
use yuzu_syntax::GreenNode;

use crate::index::{Index, IndexReader};
use crate::modules::{self, ModuleResolver};
use crate::stdlib::Engine;
use crate::{CompileOptions, in_thread_context, lower_and_check};

/// The file a check asks about.
#[derive(Clone, Copy, Debug)]
pub enum Focus<'a> {
    /// A file that is no module: the program starts there. `syntax` is a
    /// tree parsed from `source` before, without errors, when there is one.
    Entry {
        name: &'a str,
        source: &'a str,
        syntax: Option<&'a GreenNode>,
    },
    /// A module, by its path. It is loaded whether or not anything imports
    /// it, and each of its declarations is lowered, even one nothing uses.
    Module(&'a str),
}

/// What a check found: the sources it read, what the stages reported, and
/// the index of names and types. A span names its source by its id in
/// `sources`.
#[derive(Debug)]
pub struct Checked {
    pub sources: SourceMap,
    pub diagnostics: Vec<Diagnostic>,
    pub index: Index,
    /// The tree each source was lowered from, by the source's id.
    pub syntax: Vec<(SourceId, GreenNode)>,
}

/// Checks the program `focus` belongs to, for the `DataFusion` engine.
///
/// The library is read from its files rather than from the cache a compile
/// uses, so a copy the resolver holds of a library module is the one read.
pub fn check(focus: Focus<'_>, resolver: &dyn ModuleResolver) -> Checked {
    let (name, source, syntax, module) = match focus {
        Focus::Entry {
            name,
            source,
            syntax,
        } => (name, source, syntax.cloned(), None),
        Focus::Module(path) => ("<check>", "", None, Some(path)),
    };

    let mut sources = SourceMap::new();
    let mut diagnostics = DiagnosticsEngine::new();
    let entry = sources.add(name.to_owned(), source.to_owned());
    let modules::Loaded { files, trees, .. } = modules::load_program(
        modules::EntryFile {
            source: entry,
            syntax,
        },
        &mut sources,
        &mut diagnostics,
        resolver,
        Engine::DataFusion,
        None,
        module,
    );

    let mut reader = IndexReader::new(&sources);
    in_thread_context(|context| {
        lower_and_check(
            context,
            &sources,
            &files,
            &mut diagnostics,
            None,
            &CompileOptions::default(),
            Some(&mut reader),
        );
    });
    let index = reader.finish();

    Checked {
        sources,
        diagnostics: diagnostics.into_diagnostics(),
        index,
        syntax: trees,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use yuzu_diagnostics::diagnostics::printer::DiagnosticPrinter;

    use super::{Checked, Focus, check};
    use crate::modules::{MapResolver, ModuleResolver, ModuleSource};

    struct LibraryCopy {
        path: &'static str,
        source: &'static str,
    }

    impl ModuleResolver for LibraryCopy {
        fn resolve(&self, _path: &str) -> Option<ModuleSource> {
            None
        }

        fn resolve_library(&self, path: &str) -> Option<ModuleSource> {
            (path == self.path).then(|| ModuleSource {
                name: format!("{path}.yz"),
                source: self.source.to_owned(),
                syntax: None,
            })
        }
    }

    fn rendered(checked: &Checked) -> String {
        let printer = DiagnosticPrinter::new(&checked.sources);
        checked
            .diagnostics
            .iter()
            .map(|diagnostic| printer.print(diagnostic))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn an_entry_reports_what_the_checks_find() {
        let resolver = MapResolver(HashMap::new());
        let checked = check(
            Focus::Entry {
                name: "main.yz",
                source: "def f(x: i64) -> int64 { return 1 }\n",
                syntax: None,
            },
            &resolver,
        );
        expect_test::expect![[r"
            error: unknown type `i64`
             --> main.yz:1:10
              |
            1 | def f(x: i64) -> int64 { return 1 }
              |          ^^^
        "]]
        .assert_eq(&rendered(&checked));
    }

    #[test]
    fn a_module_lowers_a_declaration_nothing_uses() {
        let resolver = MapResolver(HashMap::from([(
            "helpers".to_owned(),
            "pub def unused(x: i64) -> int64 { return 1 }\n".to_owned(),
        )]));
        let checked = check(Focus::Module("helpers"), &resolver);
        expect_test::expect![[r"
            error: unknown type `i64`
             --> helpers.yz:1:19
              |
            1 | pub def unused(x: i64) -> int64 { return 1 }
              |                   ^^^
        "]]
        .assert_eq(&rendered(&checked));
    }

    #[test]
    fn a_library_copy_replaces_the_built_in_one() {
        let resolver = LibraryCopy {
            path: "yuzu.target.datafusion",
            source: "pub external def power(a: i64, b: int64) -> int64\n",
        };
        let checked = check(Focus::Module("yuzu.target.datafusion"), &resolver);
        expect_test::expect![[r"
            error: unknown type `i64`
             --> yuzu.target.datafusion.yz:1:27
              |
            1 | pub external def power(a: i64, b: int64) -> int64
              |                           ^^^
        "]]
        .assert_eq(&rendered(&checked));
    }

    fn check_entry(source: &str) -> String {
        let resolver = MapResolver(HashMap::new());
        let checked = check(
            Focus::Entry {
                name: "main.yz",
                source,
                syntax: None,
            },
            &resolver,
        );
        checked
            .diagnostics
            .iter()
            .map(|diagnostic| diagnostic.message.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn an_unknown_type_is_reported_once() {
        expect_test::expect![[r"
            unknown type `i64`
            unknown type `i64`"]]
        .assert_eq(&check_entry(
            "def f(x: i64) -> int64 { return x + 1 }\nlet q: i64 = 1\n",
        ));
    }

    #[test]
    fn a_value_computed_from_an_error_is_not_reported() {
        expect_test::expect!["unresolved identifier `nosuch`"]
            .assert_eq(&check_entry("let y = 1 + nosuch\nlet z: str = y\n"));
    }

    #[test]
    fn a_syntax_error_is_reported_by_the_parser_alone() {
        expect_test::expect!["expected expression, found end of input"]
            .assert_eq(&check_entry("let a = 1 +\n"));
    }

    #[test]
    fn a_column_of_an_unknown_relation_is_not_reported() {
        expect_test::expect!["`nosuch` is not a relation"]
            .assert_eq(&check_entry("from nosuch |> where a > 1 |> select a\n"));
    }

    #[test]
    fn independent_errors_are_each_reported() {
        expect_test::expect![[r"
            expected expression, found `def`
            expected `str`, found `int64`
            expected `int64`, found `str`"]]
        .assert_eq(&check_entry(
            "let a = 1 +\ndef f() -> int64 { return \"s\" }\ndef g() -> str { return 1 }\n",
        ));
    }
}
