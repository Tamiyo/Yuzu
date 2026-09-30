//! The references in a check's index, with each declaration narrowed to
//! the name it declares. The lowering gives the name each use wrote, and
//! the whole declaration it names.

use rustc_hash::FxHashMap;
use text_size::TextRange;
use yuzu_ast::ast::{self, AstNode, Mutability};
use yuzu_diagnostics::{SourceId, Span};
use yuzu_driver::index::{Declaration, Reference, TargetKind};
use yuzu_syntax::{GreenNode, SyntaxKind, SyntaxNode};

/// A name's place in one of a check's sources.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Name {
    pub(crate) source: SourceId,
    pub(crate) range: TextRange,
}

/// A use of a name, and the declaration it names. A declaration no use
/// names is its own use.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Resolution {
    /// The reference in the index this resolution narrows, when a use
    /// names the declaration.
    pub(crate) reference: Option<usize>,
    pub(crate) used: Name,
    pub(crate) declared: Name,
    /// The whole declaration, as the index gives it.
    pub(crate) declaration: Span,
    pub(crate) kind: DeclarationKind,
    /// The name an import gave the declaration, which the use goes
    /// through: the `y` of `x as y`.
    pub(crate) alias: Option<Name>,
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
    /// A module, whose declaration is its file.
    Module,
    /// A column: a struct's field, or a stage's item that names one.
    Column,
}

/// The tree of each source a check lowered.
pub(crate) type Trees = FxHashMap<SourceId, GreenNode>;

pub(crate) fn resolutions(references: &[Reference], trees: &Trees) -> Vec<Resolution> {
    let root = |source: SourceId| trees.get(&source).cloned().map(SyntaxNode::new_root);
    references
        .iter()
        .enumerate()
        .filter_map(|(at, reference)| {
            let (declared, kind) = match reference.kind {
                TargetKind::Module => (reference.target.range, DeclarationKind::Module),
                TargetKind::Declaration => {
                    let root = root(reference.target.source_id)?;
                    let declared = declared_name(&root, reference.target.range, &reference.name)?;
                    (declared, declaration_kind(&declaring(&root, declared)?)?)
                }
            };
            let alias = match reference.alias {
                Some(item) => Some(alias_name(&root(item.source_id)?, item)?),
                None => None,
            };
            Some(Resolution {
                reference: Some(at),
                used: Name {
                    source: reference.at.source_id,
                    range: reference.at.range,
                },
                declared: Name {
                    source: reference.target.source_id,
                    range: declared,
                },
                declaration: reference.target,
                kind,
                alias,
            })
        })
        .collect()
}

/// Each declaration in the index, as a use of itself, so that a
/// declaration no name uses is still found.
pub(crate) fn declarations(declarations: &[Declaration], trees: &Trees) -> Vec<Resolution> {
    declarations
        .iter()
        .filter_map(|declaration| {
            let root = SyntaxNode::new_root(trees.get(&declaration.at.source_id)?.clone());
            let range = declared_name(&root, declaration.at.range, &declaration.name)?;
            let declared = Name {
                source: declaration.at.source_id,
                range,
            };
            Some(Resolution {
                reference: None,
                used: declared,
                declared,
                declaration: declaration.at,
                kind: declaration_kind(&declaring(&root, range)?)?,
                alias: None,
            })
        })
        .collect()
}

/// The new name an import item gives: the `y` of `x as y`.
fn alias_name(root: &SyntaxNode, item: Span) -> Option<Name> {
    let alias = ast::ImportItem::cast(node_at(root, item.range)?)?
        .alias()?
        .token()?;
    Some(Name {
        source: item.source_id,
        range: alias.text_range(),
    })
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

/// Where a declaration spells `name`: the one of its own `Ident`s that
/// does, as `group by b as c` holds two.
fn declared_name(root: &SyntaxNode, range: TextRange, name: &str) -> Option<TextRange> {
    node_at(root, range)?
        .children()
        .filter_map(ast::Ident::cast)
        .filter_map(|ident| ident.token())
        .find(|token| token.text() == name)
        .map(|token| token.text_range())
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
        SyntaxKind::StructField
        | SyntaxKind::SelectItem
        | SyntaxKind::AggregateItem
        | SyntaxKind::GroupByItem
        | SyntaxKind::RenameItem => DeclarationKind::Column,
        other => {
            debug_assert!(false, "the index names a declaration in a {other:?}");
            return None;
        }
    };
    Some(kind)
}
