//! Renaming a name: its declaration and each use that spells it.
//!
//! An alias spells the name another way, so renaming the declaration leaves
//! the alias as it is, and renaming the alias renames it in its file alone.

use std::fmt;
use std::path::PathBuf;

use text_size::{TextRange, TextSize};
use yuzu_diagnostics::SourceId;
use yuzu_lexer::lexer::Lexer;
use yuzu_lexer::token_kind::TokenKind;

use crate::Checked;
use crate::names::{DeclarationKind, Name};
use crate::navigation::{FileRange, resolution_at};

/// Why a name cannot be renamed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RenameError {
    /// The position is on no name the check resolved.
    NoName,
    /// A module is named by its file.
    Module,
    /// The new name is not one identifier.
    NotAnIdentifier(String),
    /// A declaration or a use is in a file that cannot be written, as the
    /// library's files are.
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
            RenameError::NoFile => f.write_str("a use of this name is in no file"),
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
    names
        .into_iter()
        .map(|name| {
            let path = checked.path(name.source).ok_or(RenameError::NoFile)?;
            if std::fs::metadata(path).is_ok_and(|metadata| metadata.permissions().readonly()) {
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
    let at = resolution_at(checked.resolutions(), source, offset).ok_or(RenameError::NoName)?;
    if at.kind == DeclarationKind::Module {
        return Err(RenameError::Module);
    }

    let on_use = at.used.source == source && at.used.range.contains_inclusive(offset);
    let here = if on_use { at.used } else { at.declared };
    let spelling = text(checked, here);
    let declared = text(checked, at.declared);

    // A declaration's name renames every use that spells it; an alias's,
    // only its own spellings in its file.
    let is_alias = spelling != declared;
    let mut names: Vec<Name> = Vec::new();
    if !is_alias {
        names.push(at.declared);
    }
    for resolution in checked.resolutions() {
        let used = resolution.used;
        if resolution.declared == at.declared
            && text(checked, used) == spelling
            && (!is_alias || used.source == source)
            && !names.contains(&used)
        {
            names.push(used);
        }
    }
    Ok((here, names))
}

fn text(checked: &Checked, name: Name) -> &str {
    &checked.text(name.source)[name.range]
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
