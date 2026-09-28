//! The references in a check's index, narrowed from the syntax each side
//! was lowered from to the name inside it. The index says what the name is
//! and what kind of declaration it names, so a use of a parameter finds the
//! parameter and not the function around it.

use text_size::TextRange;
use yuzu_ast::{self as ast, AstNode};
use yuzu_diagnostics::diagnostics::Span;
use yuzu_diagnostics::source_map::SourceId;
use yuzu_driver::index::TargetKind;
use yuzu_syntax::{SyntaxKind, SyntaxNode};

/// A name's place in one of a check's sources.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Name {
    pub(crate) source: SourceId,
    pub(crate) range: TextRange,
}

/// A use of a name, and the declaration it names.
#[derive(Clone, Debug)]
pub(crate) struct Resolution {
    pub(crate) name: String,
    pub(crate) used: Name,
    pub(crate) declared: Name,
    /// The whole declaration, as the index gives it.
    pub(crate) declaration: Span,
}

/// The tree a check lowered a source from.
pub(crate) fn tree(checked: &yuzu_driver::Checked, source: SourceId) -> Option<SyntaxNode> {
    checked
        .syntax
        .iter()
        .find(|(id, _)| *id == source)
        .map(|(_, green)| SyntaxNode::new_root(green.clone()))
}

pub(crate) fn resolutions(checked: &yuzu_driver::Checked) -> Vec<Resolution> {
    checked
        .index
        .references
        .iter()
        .filter_map(|reference| {
            let used = used_name(
                &tree(checked, reference.at.source_id)?,
                reference.at.range,
                &reference.name,
            )?;
            let declared = declared_name(
                &tree(checked, reference.target.source_id)?,
                reference.target.range,
                &reference.name,
                reference.kind,
            )?;
            Some(Resolution {
                name: reference.name.clone(),
                used: Name {
                    source: reference.at.source_id,
                    range: used,
                },
                declared: Name {
                    source: reference.target.source_id,
                    range: declared,
                },
                declaration: reference.target,
            })
        })
        .collect()
}

/// The node that declares the name at `name`: the node around its `Ident`.
pub(crate) fn declaring(root: &SyntaxNode, name: TextRange) -> Option<SyntaxNode> {
    root.covering_element(name).parent()?.parent()
}

/// The node an op was lowered from: the outermost whose range is the op's.
pub(crate) fn node_at(root: &SyntaxNode, range: TextRange) -> Option<SyntaxNode> {
    nodes_at(root, range).last()
}

/// Each node whose range is exactly `range`, innermost first. A name and
/// the expression that reads it cover the same text.
fn nodes_at(root: &SyntaxNode, range: TextRange) -> impl Iterator<Item = SyntaxNode> {
    let covering = match root.covering_element(range) {
        rowan::NodeOrToken::Node(node) => Some(node),
        rowan::NodeOrToken::Token(token) => token.parent(),
    };
    covering
        .into_iter()
        .flat_map(|node| node.ancestors())
        .skip_while(move |node| node.text_range() != range)
        .take_while(move |node| node.text_range() == range)
}

/// Where a use spells `name`: a name read, a callee, or the relation of a
/// `from` or a `join`.
fn used_name(root: &SyntaxNode, range: TextRange, name: &str) -> Option<TextRange> {
    let ident = nodes_at(root, range).find_map(|node| match node.kind() {
        SyntaxKind::IdentExpr => ast::IdentExpr::cast(node)?.name(),
        SyntaxKind::CallExpr => match ast::CallExpr::cast(node)?.callee()? {
            ast::Expr::IdentExpr(callee) => callee.name(),
            _ => None,
        },
        SyntaxKind::FromExpr => ast::FromExpr::cast(node)?.relation(),
        SyntaxKind::JoinExpr => ast::JoinExpr::cast(node)?.relation(),
        _ => None,
    })?;
    let token = ident.token()?;
    (token.text() == name).then(|| token.text_range())
}

/// Where a declaration spells `name`. A local or a symbol is named by the
/// declaration's own `Ident`; a parameter by one of the function's.
fn declared_name(
    root: &SyntaxNode,
    range: TextRange,
    name: &str,
    kind: TargetKind,
) -> Option<TextRange> {
    let declaration = node_at(root, range)?;
    let ident = match kind {
        TargetKind::Local | TargetKind::Symbol => declaration.children().find_map(ast::Ident::cast),
        TargetKind::Parameter => ast::FuncStmt::cast(declaration)?
            .params()
            .filter_map(|param| param.name())
            .find(|ident| ident.token().is_some_and(|token| token.text() == name)),
    }?;
    let token = ident.token()?;
    (token.text() == name).then(|| token.text_range())
}
