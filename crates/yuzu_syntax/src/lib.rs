use num_traits::{FromPrimitive, ToPrimitive};

mod syntax_kind;

pub use syntax_kind::*;

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub enum YuzuLanguage {}

impl rowan::Language for YuzuLanguage {
    type Kind = syntax_kind::SyntaxKind;

    fn kind_from_raw(raw: rowan::SyntaxKind) -> Self::Kind {
        Self::Kind::from_u16(raw.0).unwrap()
    }

    fn kind_to_raw(kind: Self::Kind) -> rowan::SyntaxKind {
        rowan::SyntaxKind(kind.to_u16().unwrap())
    }
}

pub type GreenNode = rowan::GreenNode;
pub type SyntaxNode = rowan::SyntaxNode<YuzuLanguage>;
pub type SyntaxNodePtr = rowan::ast::SyntaxNodePtr<YuzuLanguage>;
pub type SyntaxElement = rowan::SyntaxElement<YuzuLanguage>;
pub type SyntaxToken = rowan::SyntaxToken<YuzuLanguage>;
