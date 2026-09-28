//! Go to definition, find references, and the uses to highlight in a file.

use std::path::PathBuf;

use text_size::{TextRange, TextSize};
use yuzu_diagnostics::source_map::SourceId;

use crate::Checked;
use crate::names::{Name, Resolution};

/// A range in a file on disk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileRange {
    pub path: PathBuf,
    pub range: TextRange,
}

pub(crate) fn goto_definition(
    checked: &Checked,
    source: SourceId,
    offset: TextSize,
) -> Option<FileRange> {
    let resolution = resolution_at(checked.resolutions(), source, offset)?;
    file_range(checked, resolution.declared)
}

/// The declaration and each use, the declaration first. A use in a file
/// with no path, such as a library module built into the compiler, is left
/// out.
pub(crate) fn references(checked: &Checked, source: SourceId, offset: TextSize) -> Vec<FileRange> {
    names(checked, source, offset)
        .into_iter()
        .filter_map(|name| file_range(checked, name))
        .collect()
}

/// The declaration and uses in the file the position is in.
pub(crate) fn highlight(checked: &Checked, source: SourceId, offset: TextSize) -> Vec<TextRange> {
    names(checked, source, offset)
        .into_iter()
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

fn names(checked: &Checked, source: SourceId, offset: TextSize) -> Vec<Name> {
    let resolutions = checked.resolutions();
    let Some(at) = resolution_at(resolutions, source, offset) else {
        return Vec::new();
    };

    let mut names = vec![at.declared];
    for resolution in resolutions {
        if resolution.declared == at.declared && !names.contains(&resolution.used) {
            names.push(resolution.used);
        }
    }
    names
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

    use crate::test_support::{checked, cursor, render};

    const HELPERS: (&str, &str) = ("helpers.yz", "pub def two() -> int64 { return 2 }\n");

    fn check_definition(fixture: &str, expected: &Expect) {
        let (text, offset) = cursor(fixture);
        let (_tree, main, checked) = checked(&[HELPERS], &text);
        let rendered = checked
            .goto_definition(&main, offset)
            .map(|target| render(&checked, &target.path, target.range));
        expected.assert_debug_eq(&rendered);
    }

    fn check_references(fixture: &str, expected: &Expect) {
        let (text, offset) = cursor(fixture);
        let (_tree, main, checked) = checked(&[HELPERS], &text);
        let rendered: Vec<String> = checked
            .references(&main, offset)
            .iter()
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
        let (_tree, main, checked) = checked(&[], &text);
        let target = checked
            .goto_definition(&main, offset)
            .map(|target| target.range);
        assert_eq!(target, Some(TextRange::new(6.into(), 7.into())));
    }

    #[test]
    fn highlight_stays_in_the_file() {
        let (text, offset) = cursor(&PROGRAM.replacen("+ cap", "+ $0cap", 1));
        let (_tree, main, checked) = checked(&[HELPERS], &text);
        let rendered: Vec<&str> = checked
            .highlight(&main, offset)
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
                main.yz:two 160..163"]],
        );
    }
}
