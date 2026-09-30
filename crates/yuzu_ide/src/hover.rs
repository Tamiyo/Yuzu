//! What hovering shows: a name's declaration, or an expression's type.

use text_size::{TextRange, TextSize};
use yuzu_ast::ast::{self, AstNode, Mutability};
use yuzu_diagnostics::SourceId;
use yuzu_syntax::SyntaxKind;

use crate::Checked;
use crate::file_structure;
use crate::names::{DeclarationKind, Resolution, declaring, node_at};
use crate::navigation::resolution_at;

/// What a hover shows, and the range it is about.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HoverResult {
    /// What the hover is about, in the file hovered.
    pub range: TextRange,
    /// Markdown.
    pub markup: String,
}

pub(crate) fn hover(checked: &Checked, source: SourceId, offset: TextSize) -> Option<HoverResult> {
    if let Some(resolution) = resolution_at(checked, source, offset) {
        let range = if resolution.used.source == source
            && resolution.used.range.contains_inclusive(offset)
        {
            resolution.used.range
        } else {
            resolution.declared.range
        };
        let text = describe(checked, &resolution)?;
        return Some(HoverResult {
            range,
            markup: code(&text),
        });
    }

    let typed = checked
        .index()
        .types
        .iter()
        .filter(|typed| typed.at.source_id == source && typed.at.range.contains_inclusive(offset))
        .min_by_key(|typed| typed.at.range.len())?;
    Some(HoverResult {
        range: typed.at.range,
        markup: code(&typed.ty),
    })
}

/// A declaration as a reader would write it: a function's signature, a
/// parameter or a `let` with its type, a table or struct in full, or a
/// module by its path.
fn describe(checked: &Checked, resolution: &Resolution) -> Option<String> {
    if resolution.kind == DeclarationKind::Module {
        return Some(format!("module {}", checked.name(resolution)));
    }

    let root = checked.syntax(resolution.declared.source)?;
    let declared = declaring(&root, resolution.declared.range)?;
    let declaration = node_at(&root, resolution.declaration.range)?;
    let ty = checked.type_at(resolution.declaration);

    let text = match declared.kind() {
        SyntaxKind::FuncParam => declared.text().to_string(),
        SyntaxKind::LetStmt => {
            let binding = ast::LetStmt::cast(declared)?;
            let written = binding
                .type_annotation()
                .map(|annotation| annotation.syntax().text().to_string());
            let mutable = match binding.mutability() {
                Mutability::Mutable => "mut ",
                Mutability::Immutable => "",
            };
            let name = checked.name(resolution);
            match written.as_deref().or(ty) {
                Some(ty) => format!("let {mutable}{name}: {ty}"),
                None => format!("let {mutable}{name}"),
            }
        }
        SyntaxKind::FuncStmt => file_structure::header(&ast::FuncStmt::cast(declaration)?),
        _ => declaration.text().to_string(),
    };
    Some(file_structure::compact(&text))
}

fn code(text: &str) -> String {
    format!("```yuzu\n{text}\n```")
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};

    use crate::test_support::{at, checked, cursor};

    fn check(fixture: &str, expected: &Expect) {
        let (text, offset) = cursor(fixture);
        let (_tree, checked) = checked(&[], &text);
        let hover = checked.hover(at(offset));
        let rendered = hover.map(|hover| format!("{} {}", &text[hover.range], hover.markup));
        expected.assert_debug_eq(&rendered);
    }

    const PROGRAM: &str = r"table t = { a: int64 }
let cap = 10
def double(x: int64) -> int64 {
    let mut y = x * 2
    return y
}
from t |> select double(1) + cap as v
";

    #[test]
    fn a_call_shows_the_signature() {
        check(
            &PROGRAM.replacen("select double", "select dou$0ble", 1),
            &expect![[r#"
                Some(
                    "double ```yuzu\ndef double(x: int64) -> int64\n```",
                )
            "#]],
        );
    }

    #[test]
    fn a_struct_no_one_names_shows_itself() {
        check(
            &format!("{PROGRAM}struct $0Unused {{ a: int64 }}\n"),
            &expect![[r#"
                Some(
                    "Unused ```yuzu\nstruct Unused { a: int64 }\n```",
                )
            "#]],
        );
    }

    #[test]
    fn a_local_shows_its_type() {
        check(
            &PROGRAM.replacen("return y", "return $0y", 1),
            &expect![[r#"
                Some(
                    "y ```yuzu\nlet mut y: int64\n```",
                )
            "#]],
        );
    }

    #[test]
    fn a_parameter_shows_its_annotation() {
        check(
            &PROGRAM.replacen("= x * 2", "= $0x * 2", 1),
            &expect![[r#"
                Some(
                    "x ```yuzu\nx: int64\n```",
                )
            "#]],
        );
    }

    #[test]
    fn a_literal_shows_its_type() {
        check(
            &PROGRAM.replacen("* 2", "* $02", 1),
            &expect![[r#"
                Some(
                    "2 ```yuzu\nint64\n```",
                )
            "#]],
        );
    }

    #[test]
    fn a_declaration_shows_itself() {
        check(
            &PROGRAM.replacen("let cap", "let c$0ap", 1),
            &expect![[r#"
                Some(
                    "cap ```yuzu\nlet cap: int64\n```",
                )
            "#]],
        );
    }

    #[test]
    fn a_module_shows_its_path() {
        let (text, offset) = cursor(
            "import helpers as h\ntable t = { a: int64 }\nfrom t |> select $0h.two() as v\n",
        );
        let (_tree, checked) = checked(
            &[("helpers.yz", "pub def two() -> int64 { return 2 }\n")],
            &text,
        );
        let hover = checked.hover(at(offset));
        let rendered = hover.map(|hover| format!("{} {}", &text[hover.range], hover.markup));
        expect![[r#"
            Some(
                "h ```yuzu\nmodule helpers\n```",
            )
        "#]]
        .assert_debug_eq(&rendered);
    }

    #[test]
    fn a_column_shows_its_field() {
        check(
            "table t = { a: int64 }\nfrom t |> where $0a > 1\n",
            &expect![[r#"
                Some(
                    "a ```yuzu\na: int64\n```",
                )
            "#]],
        );
    }
}
