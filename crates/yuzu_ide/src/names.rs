//! The references in a check's index, narrowed from the syntax each side
//! was lowered from to the name inside it. The index says what the name is
//! and what kind of declaration it names, so a use of a parameter finds the
//! parameter and not the function around it.

use rustc_hash::FxHashMap;
use text_size::TextRange;
use yuzu_ast::ast::{self, AstNode, Mutability};
use yuzu_diagnostics::{SourceId, Span};
use yuzu_driver::index::{Reference, TargetKind};
use yuzu_syntax::{GreenNode, SyntaxKind, SyntaxNode};

/// A name's place in one of a check's sources.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Name {
    pub(crate) source: SourceId,
    pub(crate) range: TextRange,
}

/// A use of a name, and the declaration it names.
#[derive(Clone, Debug)]
pub(crate) struct Resolution {
    /// The reference in the index this resolution narrows.
    pub(crate) reference: usize,
    pub(crate) used: Name,
    pub(crate) declared: Name,
    /// The whole declaration, as the index gives it.
    pub(crate) declaration: Span,
    pub(crate) kind: DeclarationKind,
}

/// What a resolved name is declared as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DeclarationKind {
    Parameter,
    Let(Mutability),
    Function,
    Table,
    Struct,
    Trait,
}

/// The tree of each source a check lowered.
pub(crate) type Trees = FxHashMap<SourceId, GreenNode>;

pub(crate) fn resolutions(references: &[Reference], trees: &Trees) -> Vec<Resolution> {
    let root = |source: SourceId| trees.get(&source).cloned().map(SyntaxNode::new_root);
    references
        .iter()
        .enumerate()
        .filter_map(|(at, reference)| {
            let used = used_name(
                &root(reference.at.source_id)?,
                reference.at.range,
                &reference.name,
            )?;
            let declared_root = root(reference.target.source_id)?;
            let declared = declared_name(
                &declared_root,
                reference.target.range,
                &reference.name,
                reference.kind,
            )?;
            Some(Resolution {
                reference: at,
                used: Name {
                    source: reference.at.source_id,
                    range: used,
                },
                declared: Name {
                    source: reference.target.source_id,
                    range: declared,
                },
                declaration: reference.target,
                kind: declaration_kind(&declaring(&declared_root, declared)?)?,
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

/// Where a use spells `name`: a name read, a callee bare or qualified by its
/// module, or the relation of a `from` or a `join`.
fn used_name(root: &SyntaxNode, range: TextRange, name: &str) -> Option<TextRange> {
    let ident = nodes_at(root, range).find_map(|node| match node.kind() {
        SyntaxKind::IdentExpr => ast::IdentExpr::cast(node)?.name(),
        SyntaxKind::CallExpr => match ast::CallExpr::cast(node)?.callee()? {
            ast::Expr::IdentExpr(callee) => callee.name(),
            ast::Expr::FieldAccessExpr(callee) => callee.field(),
            ast::Expr::CallExpr(_)
            | ast::Expr::StructExpr(_)
            | ast::Expr::ListExpr(_)
            | ast::Expr::BinaryExpr(_)
            | ast::Expr::UnaryExpr(_)
            | ast::Expr::ParenExpr(_)
            | ast::Expr::Literal(_)
            | ast::Expr::Pipeline(_) => None,
        },
        SyntaxKind::FromSource => ast::FromSource::cast(node)?.relation(),
        SyntaxKind::JoinStage => ast::JoinStage::cast(node)?.relation(),
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

fn declaration_kind(declaring: &SyntaxNode) -> Option<DeclarationKind> {
    let kind = match declaring.kind() {
        SyntaxKind::FuncParam => DeclarationKind::Parameter,
        SyntaxKind::LetStmt => {
            DeclarationKind::Let(ast::LetStmt::cast(declaring.clone())?.mutability())
        }
        SyntaxKind::FuncStmt => DeclarationKind::Function,
        SyntaxKind::TableStmt => DeclarationKind::Table,
        SyntaxKind::StructStmt => DeclarationKind::Struct,
        SyntaxKind::TraitStmt => DeclarationKind::Trait,
        other => {
            debug_assert!(false, "the index names a declaration in a {other:?}");
            return None;
        }
    };
    Some(kind)
}
