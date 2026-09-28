use num_derive::{FromPrimitive, ToPrimitive};
use yuzu_lexer::token_kind::TokenKind;

#[repr(u16)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, FromPrimitive, ToPrimitive)]
pub enum SyntaxKind {
    // Tokens
    Plus,
    Minus,
    Star,
    StarStar,
    Slash,
    Percent,
    Eq,
    EqEq,
    Neq,
    Lt,
    Lte,
    Gt,
    Gte,
    Shl,
    Shr,
    Arrow,
    Pipe,
    Dot,
    LeftParen,
    RightParen,
    LeftCurly,
    RightCurly,
    LeftSquare,
    RightSquare,
    Comma,
    Colon,
    AggKw,
    AggregateKw,
    AndKw,
    AsKw,
    ByKw,
    DefKw,
    DistinctKw,
    DropKw,
    ExtendKw,
    ExternalKw,
    ForKw,
    FromKw,
    FullKw,
    GroupKw,
    ImplKw,
    ImportKw,
    InKw,
    InnerKw,
    JoinKw,
    LeftKw,
    LetKw,
    LimitKw,
    ModKw,
    MutKw,
    NotKw,
    OffsetKw,
    OnKw,
    OrKw,
    PubKw,
    RenameKw,
    ReturnKw,
    RightKw,
    SelectKw,
    SetKw,
    StructKw,
    TableKw,
    TraitKw,
    UsingKw,
    WhereKw,
    Identifier,
    BoolLit,
    IntLit,
    FloatLit,
    HexLit,
    BinaryLit,
    StringLit,
    RawStringLit,
    Comment,
    Newline,
    Space,

    // Nodes
    Root,
    Ident,

    TypeAnnotation,
    NamedTypeAnnotation,
    FuncTypeAnnotation,
    FuncTypeAnnotationParams,

    TypeParam,
    TypeBound,
    TraitRef,

    Stmt,
    StructStmt,
    StructField,
    TraitStmt,
    ImplStmt,
    FuncStmt,
    FuncParam,
    TableStmt,
    BlockStmt,
    LetStmt,
    AssignStmt,
    ReturnStmt,
    ExprStmt,
    ModStmt,
    ImportStmt,
    FromImportStmt,
    ModulePath,
    ImportItem,

    Expr,
    IdentExpr,
    CallExpr,
    ArgList,
    FieldAccessExpr,
    StructExpr,
    StructFieldInit,
    ListExpr,
    BinaryExpr,
    UnaryExpr,
    ParenExpr,

    Pipeline,
    FromSource,
    SelectStage,
    SelectItem,
    WhereStage,
    DistinctStage,
    DropStage,
    RenameStage,
    RenameItem,
    ExtendStage,
    SetStage,
    SetItem,
    LimitStage,
    AliasStage,
    AggregateStage,
    AggregateItem,
    GroupBy,
    GroupByItem,
    JoinStage,
    JoinOn,
    JoinUsing,

    Literal,
    BoolLiteral,
    IntLiteral,
    FloatLiteral,
    StringLiteral,

    // Other
    Error,
}

impl SyntaxKind {
    /// Whether the kind is trivia: the tokens a reader sees and the grammar
    /// does not. `TokenKind::is_trivia` answers the same question a layer
    /// down, and the two must agree.
    #[must_use]
    pub fn is_trivia(self) -> bool {
        matches!(
            self,
            SyntaxKind::Comment | SyntaxKind::Space | SyntaxKind::Newline
        )
    }

    #[must_use]
    pub fn is_keyword(self) -> bool {
        matches!(
            self,
            SyntaxKind::AggKw
                | SyntaxKind::AggregateKw
                | SyntaxKind::AndKw
                | SyntaxKind::AsKw
                | SyntaxKind::ByKw
                | SyntaxKind::DefKw
                | SyntaxKind::DistinctKw
                | SyntaxKind::DropKw
                | SyntaxKind::ExtendKw
                | SyntaxKind::ExternalKw
                | SyntaxKind::ForKw
                | SyntaxKind::FromKw
                | SyntaxKind::FullKw
                | SyntaxKind::GroupKw
                | SyntaxKind::ImplKw
                | SyntaxKind::ImportKw
                | SyntaxKind::InKw
                | SyntaxKind::InnerKw
                | SyntaxKind::JoinKw
                | SyntaxKind::LeftKw
                | SyntaxKind::LetKw
                | SyntaxKind::LimitKw
                | SyntaxKind::ModKw
                | SyntaxKind::MutKw
                | SyntaxKind::NotKw
                | SyntaxKind::OffsetKw
                | SyntaxKind::OnKw
                | SyntaxKind::OrKw
                | SyntaxKind::PubKw
                | SyntaxKind::RenameKw
                | SyntaxKind::ReturnKw
                | SyntaxKind::RightKw
                | SyntaxKind::SelectKw
                | SyntaxKind::SetKw
                | SyntaxKind::StructKw
                | SyntaxKind::TableKw
                | SyntaxKind::TraitKw
                | SyntaxKind::UsingKw
                | SyntaxKind::WhereKw
        )
    }
}

impl From<TokenKind> for SyntaxKind {
    fn from(token_kind: TokenKind) -> Self {
        match token_kind {
            TokenKind::Plus => Self::Plus,
            TokenKind::Minus => Self::Minus,
            TokenKind::Star => Self::Star,
            TokenKind::StarStar => Self::StarStar,
            TokenKind::Slash => Self::Slash,
            TokenKind::Percent => Self::Percent,
            TokenKind::Eq => Self::Eq,
            TokenKind::EqEq => Self::EqEq,
            TokenKind::Neq => Self::Neq,
            TokenKind::Lt => Self::Lt,
            TokenKind::Lte => Self::Lte,
            TokenKind::Gt => Self::Gt,
            TokenKind::Gte => Self::Gte,
            TokenKind::Shl => Self::Shl,
            TokenKind::Shr => Self::Shr,
            TokenKind::Arrow => Self::Arrow,
            TokenKind::Pipe => Self::Pipe,
            TokenKind::Dot => Self::Dot,
            TokenKind::LeftParen => Self::LeftParen,
            TokenKind::RightParen => Self::RightParen,
            TokenKind::LeftCurly => Self::LeftCurly,
            TokenKind::RightCurly => Self::RightCurly,
            TokenKind::LeftSquare => Self::LeftSquare,
            TokenKind::RightSquare => Self::RightSquare,
            TokenKind::Comma => Self::Comma,
            TokenKind::Colon => Self::Colon,
            TokenKind::AggKw => Self::AggKw,
            TokenKind::AggregateKw => Self::AggregateKw,
            TokenKind::AndKw => Self::AndKw,
            TokenKind::AsKw => Self::AsKw,
            TokenKind::ByKw => Self::ByKw,
            TokenKind::DefKw => Self::DefKw,
            TokenKind::DistinctKw => Self::DistinctKw,
            TokenKind::DropKw => Self::DropKw,
            TokenKind::ExtendKw => Self::ExtendKw,
            TokenKind::ExternalKw => Self::ExternalKw,
            TokenKind::ForKw => Self::ForKw,
            TokenKind::FromKw => Self::FromKw,
            TokenKind::FullKw => Self::FullKw,
            TokenKind::GroupKw => Self::GroupKw,
            TokenKind::ImplKw => Self::ImplKw,
            TokenKind::ImportKw => Self::ImportKw,
            TokenKind::InKw => Self::InKw,
            TokenKind::InnerKw => Self::InnerKw,
            TokenKind::JoinKw => Self::JoinKw,
            TokenKind::LeftKw => Self::LeftKw,
            TokenKind::LetKw => Self::LetKw,
            TokenKind::LimitKw => Self::LimitKw,
            TokenKind::ModKw => Self::ModKw,
            TokenKind::MutKw => Self::MutKw,
            TokenKind::NotKw => Self::NotKw,
            TokenKind::OffsetKw => Self::OffsetKw,
            TokenKind::OnKw => Self::OnKw,
            TokenKind::OrKw => Self::OrKw,
            TokenKind::PubKw => Self::PubKw,
            TokenKind::RenameKw => Self::RenameKw,
            TokenKind::ReturnKw => Self::ReturnKw,
            TokenKind::RightKw => Self::RightKw,
            TokenKind::SelectKw => Self::SelectKw,
            TokenKind::SetKw => Self::SetKw,
            TokenKind::StructKw => Self::StructKw,
            TokenKind::TableKw => Self::TableKw,
            TokenKind::TraitKw => Self::TraitKw,
            TokenKind::UsingKw => Self::UsingKw,
            TokenKind::WhereKw => Self::WhereKw,
            TokenKind::Identifier => Self::Identifier,
            TokenKind::BoolLit => Self::BoolLit,
            TokenKind::IntLit => Self::IntLit,
            TokenKind::FloatLit => Self::FloatLit,
            TokenKind::HexLit => Self::HexLit,
            TokenKind::BinaryLit => Self::BinaryLit,
            TokenKind::StringLit => Self::StringLit,
            TokenKind::RawStringLit => Self::RawStringLit,
            TokenKind::Comment => Self::Comment,
            TokenKind::Newline => Self::Newline,
            TokenKind::Space => Self::Space,
            TokenKind::Error => Self::Error,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_tokens_map_to_their_own_syntax_kinds() {
        let cases = [
            (TokenKind::AggregateKw, SyntaxKind::AggregateKw),
            (TokenKind::GroupKw, SyntaxKind::GroupKw),
            (TokenKind::ByKw, SyntaxKind::ByKw),
        ];

        for (token, expected) in cases {
            assert_eq!(SyntaxKind::from(token), expected, "{token:?}");
        }
    }

    #[test]
    fn join_tokens_map_to_their_own_syntax_kinds() {
        let cases = [
            (TokenKind::JoinKw, SyntaxKind::JoinKw),
            (TokenKind::OnKw, SyntaxKind::OnKw),
            (TokenKind::UsingKw, SyntaxKind::UsingKw),
            (TokenKind::InnerKw, SyntaxKind::InnerKw),
            (TokenKind::LeftKw, SyntaxKind::LeftKw),
            (TokenKind::RightKw, SyntaxKind::RightKw),
            (TokenKind::FullKw, SyntaxKind::FullKw),
        ];

        for (token, expected) in cases {
            assert_eq!(SyntaxKind::from(token), expected, "{token:?}");
        }
    }

    #[test]
    fn keyword_tokens_do_not_share_a_syntax_kind() {
        let tokens = [
            TokenKind::JoinKw,
            TokenKind::OnKw,
            TokenKind::UsingKw,
            TokenKind::InnerKw,
            TokenKind::LeftKw,
            TokenKind::RightKw,
            TokenKind::FullKw,
            TokenKind::SelectKw,
            TokenKind::FromKw,
            TokenKind::WhereKw,
            TokenKind::AsKw,
            TokenKind::InKw,
            TokenKind::AggregateKw,
            TokenKind::GroupKw,
            TokenKind::ByKw,
            TokenKind::AggKw,
            TokenKind::ExternalKw,
        ];

        let mut seen = Vec::new();
        for token in tokens {
            let kind = SyntaxKind::from(token);
            assert!(!seen.contains(&kind), "{token:?} reuses {kind:?}");
            seen.push(kind);
        }
    }
}
