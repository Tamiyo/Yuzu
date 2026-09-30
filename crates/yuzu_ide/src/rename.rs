//! Renames a name, with its declaration and its uses.
//!
//! The lowering tells which uses go through an import alias. A rename of
//! the declaration does not change the alias. A rename of the alias changes
//! only the alias and the uses that go through it.

use std::fmt;
use std::path::PathBuf;

use rustc_hash::FxHashMap;
use text_size::{TextRange, TextSize};
use yuzu_diagnostics::SourceId;
use yuzu_lexer::lexer::Lexer;
use yuzu_lexer::token_kind::TokenKind;

use crate::Checked;
use crate::names::{DeclarationKind, Name};
use crate::navigation::{FileRange, resolution_at, uses_of};

/// Why a name cannot be renamed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RenameError {
    /// The position is on no name the check resolved.
    NoName,
    /// A module is named by its file.
    Module,
    /// The new name is not one identifier.
    NotAnIdentifier(String),
    /// A declaration or a use is in a file that cannot be written, such as a
    /// library file.
    ReadOnly(PathBuf),
    /// A declaration or a use is in a source with no file.
    NoFile,
}

impl fmt::Display for RenameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RenameError::NoName => f.write_str("there is no name here to rename"),
            RenameError::Module => f.write_str("a module is renamed by renaming its file"),
            RenameError::NotAnIdentifier(name) => write!(f, "`{name}` is not a name"),
            RenameError::ReadOnly(path) => {
                write!(f, "`{}` cannot be changed", path.display())
            }
            RenameError::NoFile => {
                f.write_str("the name is declared or used in the compiler's own copy of a file")
            }
        }
    }
}

impl std::error::Error for RenameError {}

/// The name a rename at a position would change.
pub(crate) fn prepare_rename(
    checked: &Checked,
    source: SourceId,
    offset: TextSize,
) -> Result<TextRange, RenameError> {
    let (name, _) = renamed(checked, source, offset)?;
    Ok(name.range)
}

/// Each range to write the new name over.
pub(crate) fn rename(
    checked: &Checked,
    source: SourceId,
    offset: TextSize,
    new_name: &str,
) -> Result<Vec<FileRange>, RenameError> {
    if !is_identifier(new_name) {
        return Err(RenameError::NotAnIdentifier(new_name.to_owned()));
    }

    let (_, names) = renamed(checked, source, offset)?;
    // The check for a writable file occurs one time for each file.
    let mut writable: FxHashMap<SourceId, bool> = FxHashMap::default();
    names
        .into_iter()
        .map(|name| {
            let path = checked.path(name.source).ok_or(RenameError::NoFile)?;
            let is_writable = *writable.entry(name.source).or_insert_with(|| {
                !std::fs::metadata(path).is_ok_and(|metadata| metadata.permissions().readonly())
            });
            if !is_writable {
                return Err(RenameError::ReadOnly(path.to_path_buf()));
            }
            Ok(FileRange {
                path: path.to_path_buf(),
                range: name.range,
            })
        })
        .collect()
}

/// The name at a position, and each name a rename of it changes.
fn renamed(
    checked: &Checked,
    source: SourceId,
    offset: TextSize,
) -> Result<(Name, Vec<Name>), RenameError> {
    let at = resolution_at(checked, source, offset).ok_or(RenameError::NoName)?;
    if at.kind == DeclarationKind::Module {
        return Err(RenameError::Module);
    }

    let on_use = at.used.source == source && at.used.range.contains_inclusive(offset);
    let here = if on_use { at.used } else { at.declared };

    // An alias renames only itself and the names that go through it.
    if let Some(alias) = at.alias {
        let mut names = vec![alias];
        for resolution in checked.resolutions() {
            if resolution.alias == Some(alias) && !names.contains(&resolution.used) {
                names.push(resolution.used);
            }
        }
        return Ok((here, names));
    }

    // A declaration renames each declaration linked to it, and each use that
    // does not go through an alias.
    let mut names = linked(checked, at.declared);
    for declaration in 0..names.len() {
        for resolution in checked.resolutions() {
            if resolution.declared == names[declaration]
                && resolution.alias.is_none()
                && !names.contains(&resolution.used)
            {
                names.push(resolution.used);
            }
        }
    }
    Ok((here, names))
}

/// A declaration and each declaration that shares a use with it. For
/// example, the `a` of `using (a)` names a column on each side of a join.
fn linked(checked: &Checked, declared: Name) -> Vec<Name> {
    let mut linked = vec![declared];
    let mut next = 0;
    while let Some(&at) = linked.get(next) {
        next += 1;
        for used in uses_of(checked, at) {
            for resolution in checked.resolutions() {
                if resolution.used == used && !linked.contains(&resolution.declared) {
                    linked.push(resolution.declared);
                }
            }
        }
    }
    linked
}

/// Whether a text is one identifier, and so can name a declaration.
fn is_identifier(text: &str) -> bool {
    let mut tokens = Lexer::new(text);
    matches!(
        (tokens.next(), tokens.next()),
        (Some(token), None) if token.kind == TokenKind::Identifier
    )
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};

    use crate::test_support::{at, checked, cursor};

    const HELPERS: (&str, &str) = ("helpers.yz", "pub def two() -> int64 { return 2 }\n");

    fn check(fixture: &str, new_name: &str, expected: &Expect) {
        let (text, offset) = cursor(fixture);
        let (_tree, checked) = checked(&[HELPERS], &text);
        let rendered = match checked.rename(at(offset), new_name) {
            Ok(edits) => {
                let mut lines: Vec<String> = edits
                    .iter()
                    .map(|edit| {
                        let file = edit.path.file_name().unwrap().to_string_lossy();
                        let text = checked.path_text(&edit.path).unwrap();
                        format!("{file} {:?} {}", edit.range, &text[edit.range])
                    })
                    .collect();
                lines.sort();
                lines.join("\n")
            }
            Err(error) => error.to_string(),
        };
        expected.assert_eq(&rendered);
    }

    const PROGRAM: &str = "from helpers import two, two as deux\ntable t = { a: int64 }\ndef double(x: int64) -> int64 {\n    let y = x * 2\n    return y\n}\nfrom t |> select double(a) + deux() + two() as v\n";

    #[test]
    fn a_local_renames_its_let_and_its_uses() {
        check(
            &PROGRAM.replacen("return y", "return $0y", 1),
            "z",
            &expect![[r"
                main.yz 100..101 y
                main.yz 121..122 y"]],
        );
    }

    #[test]
    fn a_let_no_one_reads_renames() {
        check(
            &PROGRAM.replacen("return y", "let $0w = 1\n    return y", 1),
            "z",
            &expect!["main.yz 118..119 w"],
        );
    }

    #[test]
    fn a_using_column_renames_both_sides() {
        check(
            "table t = { id: int64, x: int64 }\ntable u = { id: int64, y: int64 }\nfrom t |> join u using (id) |> select $0id\n",
            "key",
            &expect![[r"
                main.yz 106..108 id
                main.yz 12..14 id
                main.yz 46..48 id
                main.yz 92..94 id"]],
        );
    }

    #[test]
    fn a_function_renames_across_files() {
        check(
            &PROGRAM.replacen("+ two()", "+ $0two()", 1),
            "three",
            &expect![[r"
                helpers.yz 8..11 two
                main.yz 163..166 two
                main.yz 20..23 two
                main.yz 25..28 two"]],
        );
    }

    #[test]
    fn an_alias_renames_only_its_own_spellings() {
        check(
            &PROGRAM.replacen("+ deux()", "+ $0deux()", 1),
            "zwei",
            &expect![[r"
                main.yz 154..158 deux
                main.yz 32..36 deux"]],
        );
    }

    #[test]
    fn a_column_renames_its_field_and_its_reads() {
        check(
            &PROGRAM.replacen("double(a)", "double($0a)", 1),
            "b",
            &expect![[r"
                main.yz 149..150 a
                main.yz 49..50 a"]],
        );
    }

    #[test]
    fn a_new_name_must_be_one_identifier() {
        check(
            &PROGRAM.replacen("return y", "return $0y", 1),
            "let",
            &expect!["`let` is not a name"],
        );
        check(
            &PROGRAM.replacen("return y", "return $0y", 1),
            "a b",
            &expect!["`a b` is not a name"],
        );
    }

    #[test]
    fn a_module_is_not_renamed() {
        check(
            "import $0helpers\n",
            "other",
            &expect!["a module is renamed by renaming its file"],
        );
    }
}
