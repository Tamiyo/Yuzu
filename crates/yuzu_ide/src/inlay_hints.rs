//! The type inference gave each `let` the program wrote without one.

use rowan::WalkEvent;
use text_size::{TextRange, TextSize};
use yuzu_ast::ast::{self, AstNode};
use yuzu_diagnostics::{SourceId, Span};

use crate::Checked;

/// A label an editor shows inside the text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InlayHint {
    /// Where the hint goes: right after the name.
    pub offset: TextSize,
    pub label: String,
}

pub(crate) fn inlay_hints(checked: &Checked, source: SourceId, range: TextRange) -> Vec<InlayHint> {
    let Some(root) = checked.syntax(source) else {
        return Vec::new();
    };
    // A subtree outside the range is skipped whole.
    let mut hints = Vec::new();
    let mut walk = root.preorder();
    while let Some(event) = walk.next() {
        let WalkEvent::Enter(node) = event else {
            continue;
        };
        if node.text_range().intersect(range).is_none() {
            walk.skip_subtree();
            continue;
        }
        let Some(binding) = ast::LetStmt::cast(node) else {
            continue;
        };
        if binding.type_annotation().is_some() {
            continue;
        }
        let Some(name) = binding.name() else {
            continue;
        };
        let Some(ty) = checked.type_at(Span {
            source_id: source,
            range: binding.syntax().text_range(),
        }) else {
            continue;
        };
        hints.push(InlayHint {
            offset: name.syntax().text_range().end(),
            label: format!(": {ty}"),
        });
    }
    hints
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use text_size::{TextRange, TextSize};

    use crate::test_support::{FILE, checked};

    #[test]
    fn a_let_without_a_type_gets_one() {
        let text = "let cap = 10\nlet named: int64 = 1\ndef f() -> float64 {\n    let half = 0.5\n    return half\n}\n";
        let (_tree, checked) = checked(&[], text);
        let whole = TextRange::up_to(TextSize::of(text));
        let rendered: Vec<String> = checked
            .inlay_hints(FILE, whole)
            .iter()
            .map(|hint| {
                format!(
                    "{}{}",
                    text[..usize::from(hint.offset)].rsplit(' ').next().unwrap(),
                    hint.label
                )
            })
            .collect();
        expect![[r"
            cap: int64
            half: float64"]]
        .assert_eq(&rendered.join("\n"));
    }
}
