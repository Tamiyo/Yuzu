//! The ranges an editor can fold: bodies in braces, lists that span lines,
//! whole queries, and runs of comments or imports. A range on one line is
//! left out, since there is nothing to fold.

use text_size::TextRange;
use yuzu_syntax::{SyntaxKind, SyntaxNode, SyntaxToken};

/// A range an editor can fold, and what it holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fold {
    pub range: TextRange,
    pub kind: FoldKind,
}

/// What a fold holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FoldKind {
    Block,
    Query,
    Comment,
    Imports,
}

pub(crate) fn folding_ranges(root: &SyntaxNode) -> Vec<Fold> {
    let mut folds = Vec::new();
    for node in root.descendants() {
        if let Some(fold) = fold_node(&node) {
            folds.push(fold);
        }
    }
    fold_comments(root, &mut folds);
    fold_imports(root, &mut folds);

    // A slice of the root's `SyntaxText` walks the file's tokens from the
    // start, so the text is taken once and sliced as a string.
    let text = root.text().to_string();
    folds.retain(|fold| text[fold.range].contains('\n'));
    folds.sort_by_key(|fold| fold.range.start());
    folds
}

fn fold_node(node: &SyntaxNode) -> Option<Fold> {
    let (range, kind) = match node.kind() {
        SyntaxKind::BlockStmt
        | SyntaxKind::StructStmt
        | SyntaxKind::TableStmt
        | SyntaxKind::TraitStmt
        | SyntaxKind::ImplStmt
        | SyntaxKind::StructExpr => (braces(node)?, FoldKind::Block),
        SyntaxKind::ListExpr | SyntaxKind::ArgList => (node.text_range(), FoldKind::Block),
        SyntaxKind::Pipeline => (node.text_range(), FoldKind::Query),
        _ => return None,
    };
    Some(Fold { range, kind })
}

/// From a node's own `{` to its own `}`.
fn braces(node: &SyntaxNode) -> Option<TextRange> {
    let tokens: Vec<SyntaxToken> = node
        .children_with_tokens()
        .filter_map(rowan::NodeOrToken::into_token)
        .collect();
    let open = tokens
        .iter()
        .find(|token| token.kind() == SyntaxKind::LeftCurly)?;
    let close = tokens
        .iter()
        .rfind(|token| token.kind() == SyntaxKind::RightCurly)?;
    Some(TextRange::new(
        open.text_range().start(),
        close.text_range().end(),
    ))
}

/// Comments on consecutive lines fold as one.
fn fold_comments(root: &SyntaxNode, folds: &mut Vec<Fold>) {
    let mut run: Option<TextRange> = None;
    let mut newlines = 0;
    for token in root
        .descendants_with_tokens()
        .filter_map(rowan::NodeOrToken::into_token)
    {
        match token.kind() {
            SyntaxKind::Comment => {
                run = Some(match run {
                    Some(range) if newlines <= 1 => range.cover(token.text_range()),
                    Some(range) => {
                        folds.push(Fold {
                            range,
                            kind: FoldKind::Comment,
                        });
                        token.text_range()
                    }
                    None => token.text_range(),
                });
                newlines = 0;
            }
            SyntaxKind::Newline => newlines += 1,
            SyntaxKind::Space => {}
            _ => {
                if let Some(range) = run.take() {
                    folds.push(Fold {
                        range,
                        kind: FoldKind::Comment,
                    });
                }
            }
        }
    }
    if let Some(range) = run {
        folds.push(Fold {
            range,
            kind: FoldKind::Comment,
        });
    }
}

/// Imports on consecutive statements fold as one.
fn fold_imports(root: &SyntaxNode, folds: &mut Vec<Fold>) {
    let mut run: Option<TextRange> = None;
    for child in root.children() {
        if matches!(
            child.kind(),
            SyntaxKind::ImportStmt | SyntaxKind::FromImportStmt
        ) {
            run = Some(run.map_or(child.text_range(), |range| range.cover(child.text_range())));
        } else if let Some(range) = run.take() {
            folds.push(Fold {
                range,
                kind: FoldKind::Imports,
            });
        }
    }
    if let Some(range) = run {
        folds.push(Fold {
            range,
            kind: FoldKind::Imports,
        });
    }
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};

    use crate::test_support::{FILE, analysis};

    fn check(text: &str, expected: Expect) {
        let rendered: Vec<String> = analysis(text)
            .folding_ranges(FILE)
            .unwrap()
            .iter()
            .map(|fold| format!("{:?} {:?}", fold.kind, &text[fold.range]))
            .collect();
        expected.assert_eq(&rendered.join("\n"));
    }

    #[test]
    fn multiline_constructs() {
        check(
            r"// one
// two
import a
import b
struct Point {
    x: float64,
}
def f(x: int64) -> int64 { return x }
def g(x: int64) -> int64 {
    return [
        x,
    ]
}
from employees
|> select id
from staff |> select id
",
            expect![[r#"
                Comment "// one\n// two"
                Imports "import a\nimport b"
                Block "{\n    x: float64,\n}"
                Block "{\n    return [\n        x,\n    ]\n}"
                Block "[\n        x,\n    ]"
                Query "from employees\n|> select id""#]],
        );
    }
}
