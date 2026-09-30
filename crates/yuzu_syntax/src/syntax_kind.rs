use yuzu_lexer::token_kind::TokenKind;

/// Declares the syntax kinds: one for each token kind, by the same name, then
/// the nodes, then `Error`. The token list is checked against [`TokenKind`]
/// both ways, so the two cannot drift apart.
macro_rules! syntax_kinds {
    (tokens: [$($token:ident,)*] nodes: [$($node:ident,)*]) => {
        #[repr(u16)]
        #[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub enum SyntaxKind {
            $($token,)*
            $($node,)*
            Error,
        }

        impl SyntaxKind {
            /// Every kind, in declaration order, which is also the order of
            /// their raw values.
            pub const ALL: &[SyntaxKind] = &[$(Self::$token,)* $(Self::$node,)* Self::Error];

            /// The token kind this kind stands for, when it is a token's kind.
            #[must_use]
            pub fn token_kind(self) -> Option<TokenKind> {
                match self {
                    $(Self::$token => Some(TokenKind::$token),)*
                    $(Self::$node)|* | Self::Error => None,
                }
            }
        }

        impl From<TokenKind> for SyntaxKind {
            fn from(token_kind: TokenKind) -> Self {
                match token_kind {
                    $(TokenKind::$token => Self::$token,)*
                    TokenKind::Error => Self::Error,
                }
            }
        }
    };
}

syntax_kinds! {
    tokens: [
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
        Whitespace,
    ]
    nodes: [
        Root,
        Ident,
        NamedTypeAnnotation,
        FuncTypeAnnotation,
        FuncTypeAnnotationParams,
        TypeParam,
        TypeBound,
        TraitRef,
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
        BoolLiteral,
        IntLiteral,
        FloatLiteral,
        StringLiteral,
    ]
}

impl SyntaxKind {
    /// Whether the kind is trivia: the tokens a reader sees and the grammar
    /// does not.
    #[must_use]
    pub fn is_trivia(self) -> bool {
        self.token_kind().is_some_and(TokenKind::is_trivia)
    }

    /// Whether the kind is a keyword's.
    #[must_use]
    pub fn is_keyword(self) -> bool {
        self.token_kind().is_some_and(TokenKind::is_keyword)
    }

    /// The kind a raw value stands for.
    ///
    /// # Panics
    ///
    /// Panics if the raw value is no kind's: a tree only holds the values
    /// this crate gave it.
    #[must_use]
    pub fn from_raw(raw: u16) -> Self {
        *Self::ALL
            .get(usize::from(raw))
            .unwrap_or_else(|| panic!("{raw} is not a syntax kind"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_token_kind_has_its_own_syntax_kind_and_back() {
        for &token in TokenKind::ALL {
            let kind = SyntaxKind::from(token);
            match token {
                TokenKind::Error => assert_eq!(kind, SyntaxKind::Error),
                _ => assert_eq!(kind.token_kind(), Some(token), "{token:?}"),
            }
        }
    }

    #[test]
    fn a_raw_value_is_its_kind() {
        for &kind in SyntaxKind::ALL {
            assert_eq!(SyntaxKind::from_raw(kind as u16), kind);
        }
    }
}
