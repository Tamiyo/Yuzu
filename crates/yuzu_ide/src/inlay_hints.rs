//! The type inference gave each `let` the program wrote without one.

use text_size::TextSize;
use yuzu_ast::{self as ast, AstNode};
use yuzu_diagnostics::diagnostics::Span;
use yuzu_diagnostics::source_map::SourceId;

use crate::Checked;
use crate::names::Trees;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InlayHint {
    /// Where the hint goes: right after the name.
    pub offset: TextSize,
    pub label: String,
}

pub(crate) fn inlay_hints(checked: &Checked, source: SourceId) -> Vec<InlayHint> {
    let root = Trees::new(checked).get(source);
    root.descendants()
        .filter_map(ast::LetStmt::cast)
        .filter(|binding| binding.type_annotation().is_none())
        .filter_map(|binding| {
            let name = binding.name()?;
            let ty = checked.type_at(Span {
                source_id: source,
                range: binding.syntax().text_range(),
            })?;
            Some(InlayHint {
                offset: name.syntax().text_range().end(),
                label: format!(": {ty}"),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::test_support::checked;

    #[test]
    fn a_let_without_a_type_gets_one() {
        let text = "let cap = 10\nlet named: int64 = 1\ndef f() -> float64 {\n    let half = 0.5\n    return half\n}\n";
        let (_tree, main, checked) = checked(&[], text);
        let rendered: Vec<String> = checked
            .inlay_hints(&main)
            .iter()
            .map(|hint| {
                format!(
                    "{}{}",
                    text[..usize::from(hint.offset)].rsplit(' ').next().unwrap(),
                    hint.label
                )
            })
            .collect();
        expect![[r#"
            cap: int64
            half: float64"#]]
        .assert_eq(&rendered.join("\n"));
    }
}
