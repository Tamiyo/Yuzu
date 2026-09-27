//! The references in a check's index, narrowed from the syntax each side
//! was lowered from to the name inside it. A use gives the name it spells;
//! the declaration it names is then searched for that name, so a use of a
//! parameter finds the parameter and not the function around it.

use text_size::TextRange;
use yuzu_ast::{self as ast, AstNode};
use yuzu_diagnostics::diagnostics::Span;
use yuzu_diagnostics::source_map::SourceId;
use yuzu_syntax::{SyntaxKind, SyntaxNode, SyntaxToken};

use crate::Checked;

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

/// The syntax tree of each source a check read, parsed once.
pub(crate) struct Trees<'k> {
    checked: &'k Checked,
    trees: Vec<(SourceId, SyntaxNode)>,
}

impl<'k> Trees<'k> {
    pub(crate) fn new(checked: &'k Checked) -> Self {
        Trees {
            checked,
            trees: Vec::new(),
        }
    }

    pub(crate) fn get(&mut self, source: SourceId) -> SyntaxNode {
        if let Some((_, tree)) = self.trees.iter().find(|(id, _)| *id == source) {
            return tree.clone();
        }
        let (tree, _) = crate::parse(self.checked.text(source));
        self.trees.push((source, tree.clone()));
        tree
    }
}

pub(crate) fn resolutions(checked: &Checked, trees: &mut Trees<'_>) -> Vec<Resolution> {
    checked
        .index()
        .references
        .iter()
        .filter_map(|reference| {
            let (name, used) = used_name(&trees.get(reference.at.source_id), reference.at.range)?;
            let declared = declared_name(
                &trees.get(reference.target.source_id),
                reference.target.range,
                &name,
            )?;
            Some(Resolution {
                name,
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

/// The name a use spells: a name read, a callee, or the relation of a
/// `from` or a `join`.
fn used_name(root: &SyntaxNode, range: TextRange) -> Option<(String, TextRange)> {
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
    let token = identifier(ident)?;
    Some((token.text().to_owned(), token.text_range()))
}

/// Where a declaration spells `name`: its own name, or one of its
/// parameters.
fn declared_name(root: &SyntaxNode, range: TextRange, name: &str) -> Option<TextRange> {
    node_at(root, range)?
        .descendants()
        .filter(|node| {
            node.kind() == SyntaxKind::Ident
                && node.parent().is_some_and(|parent| declares(parent.kind()))
        })
        .filter_map(|node| identifier(ast::Ident::cast(node)?))
        .find(|token| token.text() == name)
        .map(|token| token.text_range())
}

fn declares(kind: SyntaxKind) -> bool {
    matches!(
        kind,
        SyntaxKind::FuncStmt
            | SyntaxKind::FuncParam
            | SyntaxKind::LetStmt
            | SyntaxKind::TableStmt
            | SyntaxKind::StructStmt
            | SyntaxKind::TraitStmt
    )
}

fn identifier(ident: ast::Ident) -> Option<SyntaxToken> {
    ident
        .token()
        .filter(|token| token.kind() == SyntaxKind::Identifier)
}
