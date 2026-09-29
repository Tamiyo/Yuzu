//! Go to definition, find references, and the uses to highlight in a file.

use std::path::PathBuf;

use text_size::{TextRange, TextSize};
use yuzu_diagnostics::SourceId;

use crate::Checked;
use crate::names::{Name, Resolution};

/// A range in a file on disk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileRange {
    pub path: PathBuf,
    pub range: TextRange,
}

/// A name's declaration and its uses. A declaration or a use in a file with
/// no path, such as a library module built into the compiler, is left out.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct References {
    pub declaration: Option<FileRange>,
    pub uses: Vec<FileRange>,
}

pub(crate) fn goto_definition(
    checked: &Checked,
    source: SourceId,
    offset: TextSize,
) -> Option<FileRange> {
    let resolution = resolution_at(checked.resolutions(), source, offset)?;
    file_range(checked, resolution.declared)
}

pub(crate) fn references(checked: &Checked, source: SourceId, offset: TextSize) -> References {
    let Some((declared, uses)) = names(checked, source, offset) else {
        return References::default();
    };
    References {
        declaration: file_range(checked, declared),
        uses: uses
            .into_iter()
            .filter_map(|name| file_range(checked, name))
            .collect(),
    }
}

/// The declaration and uses in the file the position is in.
pub(crate) fn highlight_related(
    checked: &Checked,
    source: SourceId,
    offset: TextSize,
) -> Vec<TextRange> {
    let Some((declared, uses)) = names(checked, source, offset) else {
        return Vec::new();
    };
    std::iter::once(declared)
        .chain(uses)
        .filter(|name| name.source == source)
        .map(|name| name.range)
        .collect()
}

/// The resolution a position is on: a use first, then a declaration one
/// of the uses names.
pub(crate) fn resolution_at(
    resolutions: &[Resolution],
    source: SourceId,
    offset: TextSize,
) -> Option<&Resolution> {
    let on = |name: Name| name.source == source && name.range.contains_inclusive(offset);
    resolutions
        .iter()
        .find(|resolution| on(resolution.used))
        .or_else(|| {
            resolutions
                .iter()
                .find(|resolution| on(resolution.declared))
        })
}

/// The declaration of the name at a position, and each of its uses.
fn names(checked: &Checked, source: SourceId, offset: TextSize) -> Option<(Name, Vec<Name>)> {
    let resolutions = checked.resolutions();
    let at = resolution_at(resolutions, source, offset)?;

    let mut uses: Vec<Name> = Vec::new();
    for resolution in resolutions {
        if resolution.declared == at.declared && !uses.contains(&resolution.used) {
            uses.push(resolution.used);
        }
    }
    Some((at.declared, uses))
}

fn file_range(checked: &Checked, name: Name) -> Option<FileRange> {
    Some(FileRange {
        path: checked.path(name.source)?.to_path_buf(),
        range: name.range,
    })
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};

    use text_size::TextRange;

    use crate::test_support::{FILE, Tree, at, checked, cursor, render};
    use crate::{AnalysisHost, Change};

    const HELPERS: (&str, &str) = ("helpers.yz", "pub def two() -> int64 { return 2 }\n");

    fn check_definition(fixture: &str, expected: &Expect) {
        let (text, offset) = cursor(fixture);
        let (_tree, checked) = checked(&[HELPERS], &text);
        let rendered = checked
            .goto_definition(at(offset))
            .map(|target| render(&checked, &target.path, target.range));
        expected.assert_debug_eq(&rendered);
    }

    fn check_references(fixture: &str, expected: &Expect) {
        let (text, offset) = cursor(fixture);
        let (_tree, checked) = checked(&[HELPERS], &text);
        let references = checked.references(at(offset));
        let rendered: Vec<String> = references
            .declaration
            .iter()
            .chain(&references.uses)
            .map(|found| {
                format!(
                    "{} {:?}",
                    render(&checked, &found.path, found.range),
                    found.range
                )
            })
            .collect();
        expected.assert_eq(&rendered.join("\n"));
    }

    const PROGRAM: &str = r"from helpers import two
table t = { a: int64 }
let cap = 10
def double(x: int64) -> int64 {
    let y = x * 2
    return y
}
from t |> select double(a) + cap + two() as v
";

    #[test]
    fn a_local_goes_to_its_let() {
        check_definition(
            &PROGRAM.replacen("return y", "return $0y", 1),
            &expect![[r#"
                Some(
                    "main.yz:y",
                )
            "#]],
        );
    }

    #[test]
    fn a_parameter_goes_to_the_parameter() {
        check_definition(
            &PROGRAM.replacen("= x * 2", "= $0x * 2", 1),
            &expect![[r#"
                Some(
                    "main.yz:x",
                )
            "#]],
        );
    }

    #[test]
    fn a_call_goes_to_the_function() {
        check_definition(
            &PROGRAM.replacen("select double", "select dou$0ble", 1),
            &expect![[r#"
                Some(
                    "main.yz:double",
                )
            "#]],
        );
    }

    #[test]
    fn a_module_let_goes_to_its_let() {
        check_definition(
            &PROGRAM.replacen("+ cap", "+ $0cap", 1),
            &expect![[r#"
                Some(
                    "main.yz:cap",
                )
            "#]],
        );
    }

    #[test]
    fn a_relation_goes_to_its_table() {
        check_definition(
            &PROGRAM.replacen("from t", "from $0t", 1),
            &expect![[r#"
                Some(
                    "main.yz:t",
                )
            "#]],
        );
    }

    #[test]
    fn an_import_goes_to_the_other_file() {
        check_definition(
            &PROGRAM.replacen("+ two()", "+ t$0wo()", 1),
            &expect![[r#"
                Some(
                    "helpers.yz:two",
                )
            "#]],
        );
    }

    #[test]
    fn a_qualified_call_goes_to_the_function() {
        check_definition(
            &PROGRAM
                .replacen("from helpers import two", "import helpers", 1)
                .replacen("+ two()", "+ helpers.t$0wo()", 1),
            &expect![[r#"
                Some(
                    "helpers.yz:two",
                )
            "#]],
        );
    }

    #[test]
    fn references_include_a_qualified_call() {
        check_references(
            &PROGRAM
                .replacen("from helpers import two", "import helpers", 1)
                .replacen("+ two()", "+ helpers.t$0wo()", 1),
            &expect![[r"
                helpers.yz:two 8..11
                main.yz:two 159..162"]],
        );
    }

    #[test]
    fn a_declaration_in_the_built_in_library_is_left_out() {
        let (text, offset) = cursor("table t = { a: int64 }\nfrom t |> aggregate s$0um(a) as s\n");
        let (_tree, checked) = checked(&[], &text);
        let references = checked.references(at(offset));
        assert_eq!(references.declaration, None);
        let uses: Vec<&str> = references
            .uses
            .iter()
            .map(|found| &text[found.range])
            .collect();
        assert_eq!(uses, ["sum"]);
    }

    #[test]
    fn references_start_at_the_declaration() {
        check_references(
            &PROGRAM.replacen("let y = x", "let $0y = x", 1),
            &expect![[r"
                main.yz:y 100..101
                main.yz:y 121..122"]],
        );
    }

    #[test]
    fn a_parameter_named_like_its_function_goes_to_the_parameter() {
        let (text, offset) = cursor("def f(f: int64) -> int64 {\n    return $0f\n}\n");
        let (_tree, checked) = checked(&[], &text);
        let target = checked
            .goto_definition(at(offset))
            .map(|target| target.range);
        assert_eq!(target, Some(TextRange::new(6.into(), 7.into())));
    }

    #[test]
    fn highlight_stays_in_the_file() {
        let (text, offset) = cursor(&PROGRAM.replacen("+ cap", "+ $0cap", 1));
        let (_tree, checked) = checked(&[HELPERS], &text);
        let rendered: Vec<&str> = checked
            .highlight_related(at(offset))
            .iter()
            .map(|&range| &text[range])
            .collect();
        assert_eq!(rendered, ["cap", "cap"]);
    }

    #[test]
    fn references_reach_the_declaration_in_another_file() {
        check_references(
            &PROGRAM.replacen("+ two()", "+ t$0wo()", 1),
            &expect![[r"
                helpers.yz:two 8..11
                main.yz:two 20..23
                main.yz:two 160..163"]],
        );
    }

    #[test]
    fn each_overload_has_its_own_references() {
        check_references(
            r"table t = { a: int64 }
def f(x: int64) -> int64 { return x }
def f(x: int64, y: int64) -> int64 { return y }
from t |> select f(a) + f$0(a, a) + f(a, 1) as v
",
            &expect![[r"
                main.yz:f 65..66
                main.yz:f 133..134
                main.yz:f 143..144"]],
        );
    }

    #[test]
    fn a_library_function_goes_to_its_installed_file() {
        let (text, offset) = cursor("table t = { a: int64 }\nfrom t |> select $0sum(a) as v\n");
        let tree = Tree::new(&[]);
        let root =
            yuzu_driver::stdlib::install(&tree.0.join("cache")).expect("the library installs");
        let mut change = Change::default();
        change.set_file(FILE, Some(text.into()));
        change.set_path(FILE, Some(tree.0.join("main.yz")));
        let mut host = AnalysisHost::default();
        host.set_library_root(Some(&root));
        host.apply_change(change);
        let checked = host.analysis().check(FILE).expect("main.yz has a path");

        let target = checked
            .goto_definition(at(offset))
            .expect("`sum` has a declaration");
        assert!(target.path.starts_with(&root));
        expect!["aggregates.yz:sum"].assert_eq(&render(&checked, &target.path, target.range));
    }

    #[test]
    fn an_alias_goes_to_what_it_names() {
        check_definition(
            "from helpers import two as deux\ntable t = { a: int64 }\nfrom t |> select $0deux() as v\n",
            &expect![[r#"
                Some(
                    "helpers.yz:two",
                )
            "#]],
        );
    }

    #[test]
    fn an_import_item_goes_to_its_declaration() {
        check_definition(
            "from helpers import $0two\ntable t = { a: int64 }\nfrom t |> select two() as v\n",
            &expect![[r#"
                Some(
                    "helpers.yz:two",
                )
            "#]],
        );
    }

    #[test]
    fn a_module_qualifier_goes_to_its_file() {
        check_definition(
            "import helpers as h\ntable t = { a: int64 }\nfrom t |> select $0h.two() as v\n",
            &expect![[r#"
                Some(
                    "helpers.yz:",
                )
            "#]],
        );
    }

    #[test]
    fn a_trait_in_a_bound_goes_to_the_trait() {
        check_definition(
            "trait Numeric {\n    def zero(x: Self) -> Self\n}\ndef id[T](x: T) -> T where T: $0Numeric { return x }\n",
            &expect![[r#"
                Some(
                    "main.yz:Numeric",
                )
            "#]],
        );
    }

    #[test]
    fn both_names_in_an_impl_header_go_to_their_declarations() {
        let program = "trait Show {\n    def show(x: Self) -> str\n}\nstruct Row { a: int64 }\nimpl Show for Row {\n    def show(x: Row) -> str { return \"row\" }\n}\n";
        check_definition(
            &program.replacen("impl Show", "impl $0Show", 1),
            &expect![[r#"
                Some(
                    "main.yz:Show",
                )
            "#]],
        );
        check_definition(
            &program.replacen("for Row", "for $0Row", 1),
            &expect![[r#"
                Some(
                    "main.yz:Row",
                )
            "#]],
        );
    }

    #[test]
    fn a_column_goes_to_its_field() {
        check_definition(
            "struct Row { a: int64 }\ntable t = Row\nfrom t |> where $0a > 1\n",
            &expect![[r#"
                Some(
                    "main.yz:a",
                )
            "#]],
        );
    }

    #[test]
    fn a_column_an_item_named_goes_to_the_item() {
        check_definition(
            "table t = { a: int64 }\nfrom t |> select a + 1 as v |> where $0v > 1\n",
            &expect![[r#"
                Some(
                    "main.yz:v",
                )
            "#]],
        );
    }

    #[test]
    fn a_renamed_column_goes_to_the_rename() {
        check_definition(
            "table t = { a: int64 }\nfrom t |> rename a as b |> where $0b > 1\n",
            &expect![[r#"
                Some(
                    "main.yz:b",
                )
            "#]],
        );
    }

    #[test]
    fn a_group_key_alias_goes_to_the_key() {
        check_definition(
            "table t = { a: int64, b: int64 }\nfrom t |> aggregate sum(a) as total group by b as k |> where $0k > 1\n",
            &expect![[r#"
                Some(
                    "main.yz:k",
                )
            "#]],
        );
    }

    #[test]
    fn a_column_a_select_passes_on_keeps_its_field() {
        check_definition(
            "table t = { a: int64 }\nfrom t |> select a |> where $0a > 1\n",
            &expect![[r#"
                Some(
                    "main.yz:a",
                )
            "#]],
        );
    }
}
