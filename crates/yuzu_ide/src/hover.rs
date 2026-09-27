//! What hovering shows: a name's declaration, or an expression's type.

use text_size::{TextRange, TextSize};
use yuzu_ast::{self as ast, AstNode, Mutability};
use yuzu_diagnostics::source_map::SourceId;
use yuzu_syntax::SyntaxKind;

use crate::Checked;
use crate::names::{Resolution, Trees, node_at, resolutions};
use crate::navigation::resolution_at;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HoverResult {
    /// What the hover is about, in the file hovered.
    pub range: TextRange,
    /// Markdown.
    pub markup: String,
}

pub(crate) fn hover(checked: &Checked, source: SourceId, offset: TextSize) -> Option<HoverResult> {
    let mut trees = Trees::new(checked);
    let resolutions = resolutions(checked, &mut trees);
    if let Some(resolution) = resolution_at(&resolutions, source, offset) {
        let range = if resolution.used.source == source
            && resolution.used.range.contains_inclusive(offset)
        {
            resolution.used.range
        } else {
            resolution.declared.range
        };
        let text = describe(checked, &mut trees, resolution)?;
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
/// parameter or a `let` with its type, or a table or struct in full.
fn describe(checked: &Checked, trees: &mut Trees<'_>, resolution: &Resolution) -> Option<String> {
    let root = trees.get(resolution.declared.source);
    let declared = root
        .covering_element(resolution.declared.range)
        .parent()?
        .parent()?;
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
            match written.or(ty) {
                Some(ty) => format!("let {mutable}{}: {ty}", resolution.name),
                None => format!("let {mutable}{}", resolution.name),
            }
        }
        SyntaxKind::FuncStmt => {
            let function = ast::FuncStmt::cast(declaration)?;
            let end = function
                .body()
                .map_or(function.syntax().text_range().end(), |body| {
                    body.syntax().text_range().start()
                });
            let start = function.syntax().text_range().start();
            function
                .syntax()
                .text()
                .slice(TextRange::new(start, end) - start)
                .to_string()
        }
        _ => declaration.text().to_string(),
    };
    Some(text.split_whitespace().collect::<Vec<_>>().join(" "))
}

fn code(text: &str) -> String {
    format!("```yuzu\n{text}\n```")
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};

    use crate::test_support::{checked, cursor};

    fn check(fixture: &str, expected: Expect) {
        let (text, offset) = cursor(fixture);
        let (_tree, main, checked) = checked(&[], &text);
        let hover = checked.hover(&main, offset);
        let rendered = hover.map(|hover| format!("{} {}", &text[hover.range], hover.markup));
        expected.assert_debug_eq(&rendered);
    }

    const PROGRAM: &str = r#"let cap = 10
def double(x: int64) -> int64 {
    let mut y = x * 2
    return y
}
from t |> select double(1) + cap as v
"#;

    #[test]
    fn a_call_shows_the_signature() {
        check(
            &PROGRAM.replacen("select double", "select dou$0ble", 1),
            expect![[r#"
                Some(
                    "double ```yuzu\ndef double(x: int64) -> int64\n```",
                )
            "#]],
        );
    }

    #[test]
    fn a_local_shows_its_type() {
        check(
            &PROGRAM.replacen("return y", "return $0y", 1),
            expect![[r#"
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
            expect![[r#"
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
            expect![[r#"
            Some(
                "2 ```yuzu\nint64\n```",
            )
        "#]],
        );
    }
}
