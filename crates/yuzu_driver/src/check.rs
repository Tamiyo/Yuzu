//! The front of the pipeline for an editor: a file's program loaded, lowered
//! and checked through `check_aggregates`, with every diagnostic kept and no
//! plan made.

use yuzu_diagnostics::{Diagnostic, DiagnosticsEngine, SourceId, SourceMap};
use yuzu_syntax::GreenNode;

use crate::compile::{in_thread_context, lower_and_check};
use crate::index::{Index, IndexReader};
use crate::modules::{self, LibrarySource, ModuleResolver, Origin};
use crate::stdlib::{self, Engine, Library};

/// The file a check asks about.
#[derive(Clone, Copy, Debug)]
pub enum Focus<'a> {
    /// A file that is no module: the program starts there. `syntax` is a
    /// tree parsed from `source` before, without errors, when there is one.
    Entry {
        origin: &'a Origin,
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

/// Checks the program `focus` belongs to, for the default engine.
///
/// The resolver's [`LibrarySource`] tells where the library comes from. It is
/// the cache a compile uses, a cache of the installed files, or the
/// resolver's own copy. The check reads that copy again each time.
pub fn check(focus: Focus<'_>, resolver: &dyn ModuleResolver) -> Checked {
    let engine = Engine::default();
    let made_up;
    let (origin, source, syntax, module) = match focus {
        Focus::Entry {
            origin,
            source,
            syntax,
        } => (origin, source, syntax.cloned(), None),
        Focus::Module(path) => {
            made_up = Origin::Named("<check>".to_owned());
            (&made_up, "", None, Some(path))
        }
    };

    // When the resolver has no copy of its own, the check uses the library of
    // a compile. It does not read, parse or bind the library again.
    let library = match resolver.library_source() {
        LibrarySource::BuiltIn => Some(Library::for_thread(engine)),
        LibrarySource::Installed(root) => Some(Library::for_thread_under(engine, root)),
        LibrarySource::Own => None,
    };

    let mut sources = SourceMap::new();
    let mut diagnostics = DiagnosticsEngine::new();
    let entry = origin.add_to(&mut sources, source.into());
    let modules::Loaded { files, trees, .. } = modules::load_program(
        modules::EntryFile {
            source: entry,
            syntax,
        },
        &mut sources,
        &mut diagnostics,
        resolver,
        engine,
        library.as_deref(),
        module,
    );

    let mut reader = IndexReader::new(&sources);
    let bound = library.is_some().then(|| stdlib::bound_library(engine));
    in_thread_context(|context| {
        lower_and_check(
            context,
            &sources,
            &files,
            &mut diagnostics,
            bound,
            None,
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

    use yuzu_diagnostics::DiagnosticPrinter;

    use super::{Checked, Focus, check};
    use crate::modules::{
        LibrarySource, MapResolver, ModuleResolver, ModuleSource, Origin, Unreadable,
    };

    struct LibraryCopy {
        path: &'static str,
        source: &'static str,
    }

    impl ModuleResolver for LibraryCopy {
        fn resolve(&self, _path: &str) -> Result<Option<ModuleSource>, Unreadable> {
            Ok(None)
        }

        fn resolve_library(&self, path: &str) -> Result<Option<ModuleSource>, Unreadable> {
            Ok((path == self.path).then(|| ModuleSource {
                origin: Origin::Named(format!("{path}.yz")),
                source: self.source.into(),
                syntax: None,
            }))
        }

        fn library_source(&self) -> LibrarySource<'_> {
            LibrarySource::Own
        }
    }

    fn rendered(checked: &Checked) -> String {
        DiagnosticPrinter::new(&checked.sources).render_all(&checked.diagnostics)
    }

    #[test]
    fn an_entry_reports_what_the_checks_find() {
        let resolver = MapResolver(HashMap::new());
        let checked = check(
            Focus::Entry {
                origin: &Origin::Named("main.yz".to_owned()),
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
                origin: &Origin::Named("main.yz".to_owned()),
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
            expected `int64`, found `str`
            expected `str`, found `int64`"]]
        .assert_eq(&check_entry(
            "let a = 1 +\ndef f() -> int64 { return \"s\" }\ndef g() -> str { return 1 }\n",
        ));
    }
}
