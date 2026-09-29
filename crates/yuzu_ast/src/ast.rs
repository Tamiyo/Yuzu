use yuzu_syntax::{SyntaxElement, SyntaxKind, SyntaxNode, SyntaxToken};

pub trait AstNode: Sized {
    fn can_cast(kind: SyntaxKind) -> bool;
    fn cast(node: SyntaxNode) -> Option<Self>;
    fn syntax(&self) -> &SyntaxNode;
}

macro_rules! ast_node {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct $name(SyntaxNode);

        impl AstNode for $name {
            fn can_cast(kind: SyntaxKind) -> bool {
                kind == SyntaxKind::$name
            }

            fn cast(node: SyntaxNode) -> Option<Self> {
                Self::can_cast(node.kind()).then(|| Self(node))
            }

            fn syntax(&self) -> &SyntaxNode {
                &self.0
            }
        }
    };
}

macro_rules! ast_enum {
    ($(#[$attr:meta])* $name:ident, { $($variant:ident),+ $(,)? }) => {
        $(#[$attr])*
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum $name {
            $($variant($variant)),+
        }

        impl AstNode for $name {
            fn can_cast(kind: SyntaxKind) -> bool {
                $($variant::can_cast(kind))||+
            }

            fn cast(node: SyntaxNode) -> Option<Self> {
                $(
                    if $variant::can_cast(node.kind()) {
                        return $variant::cast(node).map(Self::$variant);
                    }
                )+
                None
            }

            fn syntax(&self) -> &SyntaxNode {
                match self {
                    $(Self::$variant(it) => it.syntax()),+
                }
            }
        }
    };
}

mod support {
    use super::AstNode;
    use yuzu_syntax::SyntaxNode;

    pub(super) fn child<N: AstNode>(parent: &SyntaxNode) -> Option<N> {
        parent.children().find_map(N::cast)
    }

    pub(super) fn nth_child<N: AstNode>(parent: &SyntaxNode, n: usize) -> Option<N> {
        parent.children().filter_map(N::cast).nth(n)
    }

    pub(super) fn children<N: AstNode>(parent: &SyntaxNode) -> impl Iterator<Item = N> + use<N> {
        parent.children().filter_map(N::cast)
    }
}

/// The visibility a node's own tokens declare. What narrows a `pub` is the
/// parenthesis right after it, not the `mod` inside: a module declaration
/// carries a `mod` of its own, and looking for that alone read every
/// `pub mod` as narrowed.
fn visibility_of(syntax: &SyntaxNode) -> Visibility {
    let mut tokens = syntax
        .children_with_tokens()
        .filter_map(SyntaxElement::into_token)
        .filter(|token| !token.kind().is_trivia());

    if tokens
        .find(|token| token.kind() == SyntaxKind::PubKw)
        .is_none()
    {
        return Visibility::Private;
    }

    match tokens.next().map(|token| token.kind()) {
        Some(SyntaxKind::LeftParen) => Visibility::Module,
        _ => Visibility::Public,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Mutability {
    Mutable,
    Immutable,
}

/// How far a declaration's name reaches.
///
/// `Module` is what `pub(mod)` asks for: the files of the enclosing module and no further, which is how a
/// module keeps a helper its own while its siblings still use it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Visibility {
    Public,
    Module,
    Private,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOp {
    And,
    Or,
    In,
    NotIn,
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
    Eq,
    Neq,
    Lt,
    Lte,
    Gt,
    Gte,
    ShiftLeft,
    ShiftRight,
}

impl BinOp {
    fn from_kind(kind: SyntaxKind) -> Option<Self> {
        Some(match kind {
            SyntaxKind::Plus => BinOp::Add,
            SyntaxKind::Minus => BinOp::Sub,
            SyntaxKind::Star => BinOp::Mul,
            SyntaxKind::Slash => BinOp::Div,
            SyntaxKind::Percent => BinOp::Rem,
            SyntaxKind::StarStar => BinOp::Pow,
            SyntaxKind::EqEq => BinOp::Eq,
            SyntaxKind::Neq => BinOp::Neq,
            SyntaxKind::Lt => BinOp::Lt,
            SyntaxKind::Lte => BinOp::Lte,
            SyntaxKind::Gt => BinOp::Gt,
            SyntaxKind::Gte => BinOp::Gte,
            SyntaxKind::Shl => BinOp::ShiftLeft,
            SyntaxKind::Shr => BinOp::ShiftRight,
            SyntaxKind::AndKw => BinOp::And,
            SyntaxKind::OrKw => BinOp::Or,
            SyntaxKind::InKw => BinOp::In,
            SyntaxKind::NotKw => BinOp::NotIn,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Neg,
    Pos,
    Not,
}

impl UnaryOp {
    fn from_kind(kind: SyntaxKind) -> Option<Self> {
        Some(match kind {
            SyntaxKind::Plus => UnaryOp::Pos,
            SyntaxKind::Minus => UnaryOp::Neg,
            SyntaxKind::NotKw => UnaryOp::Not,
            _ => return None,
        })
    }
}

ast_node!(Root);
impl Root {
    pub fn stmts(&self) -> impl Iterator<Item = Stmt> + use<> {
        support::children(self.syntax())
    }
}

ast_node!(Ident);
impl Ident {
    /// The identifier, the node's only token. Its text is read without a
    /// copy while the caller holds the token.
    #[must_use]
    pub fn token(&self) -> Option<SyntaxToken> {
        self.0.first_token()
    }
}

ast_enum!(TypeAnnotation, {
    NamedTypeAnnotation,
    FuncTypeAnnotation,
});

ast_node!(NamedTypeAnnotation);
impl NamedTypeAnnotation {
    #[must_use]
    pub fn name(&self) -> Option<Ident> {
        support::child(self.syntax())
    }

    pub fn args(&self) -> impl Iterator<Item = TypeAnnotation> + use<> {
        support::children(self.syntax())
    }
}

ast_node!(FuncTypeAnnotation);
impl FuncTypeAnnotation {
    #[must_use]
    pub fn params(&self) -> Option<FuncTypeAnnotationParams> {
        support::child(self.syntax())
    }
    #[must_use]
    pub fn result(&self) -> Option<TypeAnnotation> {
        support::child(self.syntax())
    }
}

ast_node!(FuncTypeAnnotationParams);
impl FuncTypeAnnotationParams {
    pub fn params(&self) -> impl Iterator<Item = TypeAnnotation> + use<> {
        support::children(self.syntax())
    }
}

ast_node!(TypeParam);
impl TypeParam {
    #[must_use]
    pub fn name(&self) -> Option<Ident> {
        support::child(self.syntax())
    }
}

ast_node!(TypeBound);
impl TypeBound {
    #[must_use]
    pub fn subject(&self) -> Option<Ident> {
        support::child(self.syntax())
    }

    pub fn traits(&self) -> impl Iterator<Item = TraitRef> + use<> {
        support::children(self.syntax())
    }
}

ast_node!(TraitRef);
impl TraitRef {
    #[must_use]
    pub fn name(&self) -> Option<Ident> {
        support::child(self.syntax())
    }
}

ast_enum!(Stmt, {
    StructStmt,
    TraitStmt,
    ImplStmt,
    FuncStmt,
    TableStmt,
    BlockStmt,
    LetStmt,
    AssignStmt,
    ReturnStmt,
    ExprStmt,
    ImportStmt,
    FromImportStmt,
    ModStmt,
});

ast_node!(ModulePath);
impl ModulePath {
    /// The path as one string, `yuzu.std.math`: a new `String` for each call.
    #[must_use]
    pub fn to_dotted(&self) -> String {
        let mut dotted = String::new();
        for segment in self.segments().filter_map(|segment| segment.token()) {
            if !dotted.is_empty() {
                dotted.push('.');
            }
            dotted.push_str(segment.text());
        }
        dotted
    }

    /// The segments, outermost first. One segment names a module directly;
    /// more name the path through the modules holding it.
    pub fn segments(&self) -> impl Iterator<Item = Ident> + use<> {
        support::children(self.syntax())
    }
}

ast_node!(ModStmt);
impl ModStmt {
    /// Whether another file may name this. Private unless `pub` says so, so
    /// forgetting to export is a complaint from the importer rather than a
    /// name that quietly became API.
    #[must_use]
    pub fn visibility(&self) -> Visibility {
        visibility_of(self.syntax())
    }

    #[must_use]
    pub fn name(&self) -> Option<Ident> {
        support::child(self.syntax())
    }
}

ast_node!(ImportStmt);
impl ImportStmt {
    /// Whether a file that imports this module may name the module through
    /// it. An import is private unless `pub` exports it again.
    #[must_use]
    pub fn visibility(&self) -> Visibility {
        visibility_of(self.syntax())
    }

    #[must_use]
    pub fn path(&self) -> Option<ModulePath> {
        support::child(self.syntax())
    }

    /// The name this file calls the module, when `as` gave it one.
    #[must_use]
    pub fn alias(&self) -> Option<Ident> {
        support::child(self.syntax())
    }
}

ast_node!(FromImportStmt);
impl FromImportStmt {
    /// Whether a file that imports this module may name the items through
    /// it. An import is private unless `pub` exports it again.
    #[must_use]
    pub fn visibility(&self) -> Visibility {
        visibility_of(self.syntax())
    }

    #[must_use]
    pub fn path(&self) -> Option<ModulePath> {
        support::child(self.syntax())
    }

    pub fn items(&self) -> impl Iterator<Item = ImportItem> + use<> {
        support::children(self.syntax())
    }
}

ast_node!(ImportItem);
impl ImportItem {
    #[must_use]
    pub fn name(&self) -> Option<Ident> {
        support::child(self.syntax())
    }

    /// The name this file calls the import, when `as` gave it one.
    #[must_use]
    pub fn alias(&self) -> Option<Ident> {
        support::children(self.syntax()).nth(1)
    }
}

ast_node!(StructStmt);
impl StructStmt {
    /// Whether another file may name this. Private unless `pub` says so, so
    /// forgetting to export is a complaint from the importer rather than a
    /// name that quietly became API.
    #[must_use]
    pub fn visibility(&self) -> Visibility {
        visibility_of(self.syntax())
    }

    #[must_use]
    pub fn name(&self) -> Option<Ident> {
        support::child(self.syntax())
    }

    pub fn fields(&self) -> impl Iterator<Item = StructField> + use<> {
        support::children(self.syntax())
    }
}

ast_node!(StructField);
impl StructField {
    #[must_use]
    pub fn visibility(&self) -> Visibility {
        visibility_of(self.syntax())
    }

    #[must_use]
    pub fn name(&self) -> Option<Ident> {
        support::child(self.syntax())
    }

    #[must_use]
    pub fn ty(&self) -> Option<TypeAnnotation> {
        support::child(self.syntax())
    }
}

ast_node!(TraitStmt);
impl TraitStmt {
    #[must_use]
    pub fn visibility(&self) -> Visibility {
        visibility_of(self.syntax())
    }

    #[must_use]
    pub fn name(&self) -> Option<Ident> {
        support::child(self.syntax())
    }

    pub fn methods(&self) -> impl Iterator<Item = FuncStmt> + use<> {
        support::children(self.syntax())
    }
}

ast_node!(ImplStmt);
impl ImplStmt {
    #[must_use]
    pub fn trait_(&self) -> Option<TraitRef> {
        support::child(self.syntax())
    }

    #[must_use]
    pub fn ty(&self) -> Option<Ident> {
        support::child(self.syntax())
    }

    pub fn methods(&self) -> impl Iterator<Item = FuncStmt> + use<> {
        support::children(self.syntax())
    }
}

ast_node!(FuncStmt);
impl FuncStmt {
    /// Whether another file may name this. Private unless `pub` says so, so
    /// forgetting to export is a complaint from the importer rather than a
    /// name that quietly became API.
    #[must_use]
    pub fn visibility(&self) -> Visibility {
        visibility_of(self.syntax())
    }
    #[must_use]
    pub fn is_agg(&self) -> bool {
        self.has_marker(SyntaxKind::AggKw)
    }

    #[must_use]
    pub fn is_external(&self) -> bool {
        self.has_marker(SyntaxKind::ExternalKw)
    }

    fn has_marker(&self, marker: SyntaxKind) -> bool {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .take_while(|token| token.kind() != SyntaxKind::DefKw)
            .any(|token| token.kind() == marker)
    }

    #[must_use]
    pub fn name(&self) -> Option<Ident> {
        support::child(self.syntax())
    }

    pub fn type_params(&self) -> impl Iterator<Item = TypeParam> + use<> {
        support::children(self.syntax())
    }

    pub fn params(&self) -> impl Iterator<Item = FuncParam> + use<> {
        support::children(self.syntax())
    }

    #[must_use]
    pub fn result(&self) -> Option<TypeAnnotation> {
        support::child(self.syntax())
    }

    pub fn bounds(&self) -> impl Iterator<Item = TypeBound> + use<> {
        support::children(self.syntax())
    }

    #[must_use]
    pub fn body(&self) -> Option<BlockStmt> {
        support::child(self.syntax())
    }
}

ast_node!(FuncParam);
impl FuncParam {
    #[must_use]
    pub fn name(&self) -> Option<Ident> {
        support::child(self.syntax())
    }

    #[must_use]
    pub fn ty(&self) -> Option<TypeAnnotation> {
        support::child(self.syntax())
    }
}

ast_node!(TableStmt);
impl TableStmt {
    #[must_use]
    pub fn visibility(&self) -> Visibility {
        visibility_of(self.syntax())
    }

    #[must_use]
    pub fn name(&self) -> Option<Ident> {
        support::nth_child(self.syntax(), 0)
    }

    #[must_use]
    pub fn struct_name(&self) -> Option<Ident> {
        support::nth_child(self.syntax(), 1)
    }

    pub fn inline_fields(&self) -> impl Iterator<Item = StructField> + use<> {
        support::children(self.syntax())
    }
}

ast_node!(BlockStmt);
impl BlockStmt {
    pub fn stmts(&self) -> impl Iterator<Item = Stmt> + use<> {
        support::children(self.syntax())
    }
}

ast_node!(LetStmt);
impl LetStmt {
    #[must_use]
    pub fn visibility(&self) -> Visibility {
        visibility_of(self.syntax())
    }

    #[must_use]
    pub fn name(&self) -> Option<Ident> {
        support::child(self.syntax())
    }

    #[must_use]
    pub fn mutability(&self) -> Mutability {
        match self.mut_token() {
            Some(_) => Mutability::Mutable,
            None => Mutability::Immutable,
        }
    }

    pub fn mut_token(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == SyntaxKind::MutKw)
    }

    #[must_use]
    pub fn type_annotation(&self) -> Option<TypeAnnotation> {
        support::child(self.syntax())
    }

    #[must_use]
    pub fn expr(&self) -> Option<Expr> {
        support::child(self.syntax())
    }
}

ast_node!(AssignStmt);
impl AssignStmt {
    #[must_use]
    pub fn target(&self) -> Option<Expr> {
        support::nth_child(self.syntax(), 0)
    }

    #[must_use]
    pub fn value(&self) -> Option<Expr> {
        support::nth_child(self.syntax(), 1)
    }
}

ast_node!(ReturnStmt);
impl ReturnStmt {
    #[must_use]
    pub fn expr(&self) -> Option<Expr> {
        support::child(self.syntax())
    }
}

ast_node!(ExprStmt);
impl ExprStmt {
    #[must_use]
    pub fn expr(&self) -> Option<Expr> {
        support::child(self.syntax())
    }
}

ast_enum!(Expr, {
    IdentExpr,
    CallExpr,
    FieldAccessExpr,
    StructExpr,
    ListExpr,
    BinaryExpr,
    UnaryExpr,
    ParenExpr,
    Literal,
    Pipeline,
});

ast_node!(IdentExpr);
impl IdentExpr {
    #[must_use]
    pub fn name(&self) -> Option<Ident> {
        support::child(self.syntax())
    }
}

ast_node!(CallExpr);
impl CallExpr {
    #[must_use]
    pub fn callee(&self) -> Option<Expr> {
        support::child(self.syntax())
    }

    #[must_use]
    pub fn args(&self) -> Option<ArgList> {
        support::child(self.syntax())
    }
}

ast_node!(ArgList);
impl ArgList {
    pub fn args(&self) -> impl Iterator<Item = Expr> + use<> {
        support::children(self.syntax())
    }
}

ast_node!(FieldAccessExpr);
impl FieldAccessExpr {
    #[must_use]
    pub fn base(&self) -> Option<Expr> {
        support::child(self.syntax())
    }

    #[must_use]
    pub fn field(&self) -> Option<Ident> {
        support::child(self.syntax())
    }
}

ast_node!(StructExpr);
impl StructExpr {
    #[must_use]
    pub fn name(&self) -> Option<Ident> {
        support::child(self.syntax())
    }

    pub fn fields(&self) -> impl Iterator<Item = StructFieldInit> + use<> {
        support::children(self.syntax())
    }
}

ast_node!(StructFieldInit);
impl StructFieldInit {
    #[must_use]
    pub fn name(&self) -> Option<Ident> {
        support::child(self.syntax())
    }

    #[must_use]
    pub fn value(&self) -> Option<Expr> {
        support::child(self.syntax())
    }
}

ast_node!(ListExpr);
impl ListExpr {
    pub fn elements(&self) -> impl Iterator<Item = Expr> + use<> {
        support::children(self.syntax())
    }
}

ast_node!(BinaryExpr);
impl BinaryExpr {
    #[must_use]
    pub fn lhs(&self) -> Option<Expr> {
        support::nth_child(self.syntax(), 0)
    }

    pub fn op(&self) -> Option<BinOp> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find_map(|token| BinOp::from_kind(token.kind()))
    }

    #[must_use]
    pub fn rhs(&self) -> Option<Expr> {
        support::nth_child(self.syntax(), 1)
    }
}

ast_node!(UnaryExpr);
impl UnaryExpr {
    pub fn op(&self) -> Option<UnaryOp> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find_map(|token| UnaryOp::from_kind(token.kind()))
    }

    #[must_use]
    pub fn expr(&self) -> Option<Expr> {
        support::child(self.syntax())
    }
}

ast_node!(ParenExpr);
impl ParenExpr {
    #[must_use]
    pub fn expr(&self) -> Option<Expr> {
        support::child(self.syntax())
    }
}

ast_node!(Pipeline);
impl Pipeline {
    #[must_use]
    pub fn source(&self) -> Option<FromSource> {
        support::child(self.syntax())
    }

    pub fn stages(&self) -> impl Iterator<Item = Stage> + use<> {
        support::children(self.syntax())
    }
}

ast_enum!(
    /// One `|> …` of a pipeline. Its node holds only its own tokens; what it
    /// reads is the stage before it.
    Stage, {
        SetStage,
        LimitStage,
        AliasStage,
        AggregateStage,
        SelectStage,
        WhereStage,
        DistinctStage,
        DropStage,
        RenameStage,
        ExtendStage,
        JoinStage,
    }
);

ast_node!(FromSource);
impl FromSource {
    #[must_use]
    pub fn relation(&self) -> Option<Ident> {
        support::nth_child(self.syntax(), 0)
    }

    #[must_use]
    pub fn alias(&self) -> Option<Ident> {
        support::nth_child(self.syntax(), 1)
    }
}

ast_node!(SelectStage);
impl SelectStage {
    pub fn items(&self) -> impl Iterator<Item = SelectItem> + use<> {
        support::children(self.syntax())
    }
}

ast_node!(SelectItem);
impl SelectItem {
    #[must_use]
    pub fn expr(&self) -> Option<Expr> {
        support::child(self.syntax())
    }

    #[must_use]
    pub fn alias(&self) -> Option<Ident> {
        support::child(self.syntax())
    }
}

ast_node!(WhereStage);
impl WhereStage {
    #[must_use]
    pub fn predicate(&self) -> Option<Expr> {
        support::child(self.syntax())
    }
}

ast_node!(DistinctStage);

ast_node!(DropStage);
impl DropStage {
    pub fn columns(&self) -> impl Iterator<Item = Ident> + use<> {
        support::children(self.syntax())
    }
}

ast_node!(RenameStage);
impl RenameStage {
    pub fn items(&self) -> impl Iterator<Item = RenameItem> + use<> {
        support::children(self.syntax())
    }
}

ast_node!(RenameItem);
impl RenameItem {
    /// `e.id as eid` names the column of one input; a bare `id as eid` has to
    /// find it on its own.
    #[must_use]
    pub fn qualifier(&self) -> Option<Ident> {
        self.is_qualified()
            .then(|| support::nth_child(self.syntax(), 0))
            .flatten()
    }

    /// The column renamed.
    #[must_use]
    pub fn column(&self) -> Option<Ident> {
        support::nth_child(self.syntax(), usize::from(self.is_qualified()))
    }

    /// The column's new name.
    #[must_use]
    pub fn alias(&self) -> Option<Ident> {
        support::nth_child(self.syntax(), usize::from(self.is_qualified()) + 1)
    }

    fn is_qualified(&self) -> bool {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .any(|token| token.kind() == SyntaxKind::Dot)
    }
}

ast_node!(ExtendStage);
impl ExtendStage {
    pub fn items(&self) -> impl Iterator<Item = SelectItem> + use<> {
        support::children(self.syntax())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JoinKind {
    Inner,
    Left,
    Right,
    Full,
}

impl JoinKind {
    fn from_token(kind: SyntaxKind) -> Option<Self> {
        Some(match kind {
            SyntaxKind::InnerKw => JoinKind::Inner,
            SyntaxKind::LeftKw => JoinKind::Left,
            SyntaxKind::RightKw => JoinKind::Right,
            SyntaxKind::FullKw => JoinKind::Full,
            _ => return None,
        })
    }
}

ast_node!(JoinStage);
impl JoinStage {
    /// The join's kind: inner unless a keyword says otherwise.
    #[must_use]
    pub fn kind(&self) -> JoinKind {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find_map(|token| JoinKind::from_token(token.kind()))
            .unwrap_or(JoinKind::Inner)
    }

    #[must_use]
    pub fn relation(&self) -> Option<Ident> {
        support::nth_child(self.syntax(), 0)
    }

    #[must_use]
    pub fn alias(&self) -> Option<Ident> {
        support::nth_child(self.syntax(), 1)
    }

    #[must_use]
    pub fn on(&self) -> Option<JoinOn> {
        support::child(self.syntax())
    }

    #[must_use]
    pub fn using(&self) -> Option<JoinUsing> {
        support::child(self.syntax())
    }
}

ast_node!(JoinOn);
impl JoinOn {
    #[must_use]
    pub fn condition(&self) -> Option<Expr> {
        support::child(self.syntax())
    }
}

ast_node!(JoinUsing);
impl JoinUsing {
    pub fn columns(&self) -> impl Iterator<Item = Ident> + use<> {
        support::children(self.syntax())
    }
}

ast_node!(SetStage);
impl SetStage {
    pub fn items(&self) -> impl Iterator<Item = SetItem> + use<> {
        support::children(self.syntax())
    }
}

ast_node!(SetItem);
impl SetItem {
    #[must_use]
    pub fn column(&self) -> Option<Ident> {
        support::child(self.syntax())
    }

    #[must_use]
    pub fn value(&self) -> Option<Expr> {
        support::child(self.syntax())
    }
}

ast_node!(LimitStage);
impl LimitStage {
    #[must_use]
    pub fn count(&self) -> Option<Expr> {
        support::nth_child(self.syntax(), 0)
    }

    #[must_use]
    pub fn offset(&self) -> Option<Expr> {
        support::nth_child(self.syntax(), 1)
    }
}

ast_node!(AliasStage);
impl AliasStage {
    #[must_use]
    pub fn alias(&self) -> Option<Ident> {
        support::child(self.syntax())
    }
}

ast_node!(AggregateStage);
impl AggregateStage {
    pub fn items(&self) -> impl Iterator<Item = AggregateItem> + use<> {
        support::children(self.syntax())
    }

    #[must_use]
    pub fn group_by(&self) -> Option<GroupBy> {
        support::child(self.syntax())
    }
}

ast_node!(AggregateItem);
impl AggregateItem {
    #[must_use]
    pub fn expr(&self) -> Option<Expr> {
        support::child(self.syntax())
    }

    #[must_use]
    pub fn alias(&self) -> Option<Ident> {
        support::child(self.syntax())
    }
}

ast_node!(GroupBy);
impl GroupBy {
    pub fn items(&self) -> impl Iterator<Item = GroupByItem> + use<> {
        support::children(self.syntax())
    }
}

ast_node!(GroupByItem);
impl GroupByItem {
    #[must_use]
    pub fn qualifier(&self) -> Option<Ident> {
        self.is_qualified()
            .then(|| support::nth_child(self.syntax(), 0))
            .flatten()
    }

    #[must_use]
    pub fn column(&self) -> Option<Ident> {
        support::nth_child(self.syntax(), usize::from(self.is_qualified()))
    }

    #[must_use]
    pub fn alias(&self) -> Option<Ident> {
        self.is_aliased()
            .then(|| support::nth_child(self.syntax(), usize::from(self.is_qualified()) + 1))
            .flatten()
    }

    fn is_qualified(&self) -> bool {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .any(|token| token.kind() == SyntaxKind::Dot)
    }

    fn is_aliased(&self) -> bool {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .any(|token| token.kind() == SyntaxKind::AsKw)
    }
}

ast_enum!(Literal, {
    BoolLiteral,
    IntLiteral,
    FloatLiteral,
    StringLiteral,
});

ast_node!(BoolLiteral);
impl BoolLiteral {
    #[must_use]
    pub fn value(&self) -> Option<bool> {
        self.0.first_token()?.text().parse().ok()
    }
}

ast_node!(IntLiteral);
impl IntLiteral {
    /// The literal's value, when a `u64` holds it. `0x` and `0b` give the
    /// base, and `_` separates digits.
    #[must_use]
    pub fn value(&self) -> Option<u64> {
        let token = self.0.first_token()?;
        let text = token.text();
        let (digits, radix) = match text.get(..2) {
            Some("0x" | "0X") => (&text[2..], 16),
            Some("0b" | "0B") => (&text[2..], 2),
            _ => (text, 10),
        };
        digits
            .chars()
            .filter(|&c| c != '_')
            .try_fold(0_u64, |value, c| {
                value
                    .checked_mul(u64::from(radix))?
                    .checked_add(u64::from(c.to_digit(radix)?))
            })
    }
}

ast_node!(FloatLiteral);
impl FloatLiteral {
    /// The literal's value. `_` separates digits.
    #[must_use]
    pub fn value(&self) -> Option<f64> {
        let token = self.0.first_token()?;
        let text = token.text();
        if text.contains('_') {
            text.replace('_', "").parse().ok()
        } else {
            text.parse().ok()
        }
    }
}

ast_node!(StringLiteral);
impl StringLiteral {
    /// The literal's text, with its escapes replaced; a raw string's text as
    /// written. A new `String` for each call.
    #[must_use]
    pub fn value(&self) -> Option<String> {
        let token = self.0.first_token()?;
        let text = token.text();
        if token.kind() == SyntaxKind::RawStringLit {
            let inner = text.strip_prefix("r\"")?.strip_suffix('"')?;
            return Some(inner.to_owned());
        }
        let inner = text.strip_prefix('"')?.strip_suffix('"')?;
        Some(yuzu_lexer::escape::unescape(inner))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yuzu_diagnostics::{diagnostics::engine::DiagnosticsEngine, source_map::SourceMap};

    fn parsed(input: &str) -> SyntaxNode {
        let mut diagnostics = DiagnosticsEngine::new();
        let source_id = SourceMap::new().add("test".to_owned(), input.to_owned());
        yuzu_parser::parse_text(input, &mut diagnostics, source_id)
    }

    fn first<N: AstNode>(input: &str) -> N {
        parsed(input)
            .descendants()
            .find_map(N::cast)
            .expect("the input holds the node")
    }

    /// The outermost join stage, so a chained query yields its last stage.
    fn join(input: &str) -> JoinStage {
        first(input)
    }

    fn text(ident: Option<Ident>) -> Option<String> {
        Some(ident?.token()?.text().to_owned())
    }

    fn rename(input: &str) -> RenameItem {
        first(input)
    }

    fn aggregate(input: &str) -> AggregateStage {
        first(input)
    }

    /// Private unless `pub` says otherwise, and a field answers for itself
    /// rather than for the struct holding it.
    #[test]
    fn visibility_is_private_until_pub_says_otherwise() {
        let syntax = parsed("pub struct P { pub x: int, y: int }\nstruct Q { z: int }\n");
        let structs: Vec<StructStmt> = syntax.descendants().filter_map(StructStmt::cast).collect();
        assert_eq!(structs[0].visibility(), Visibility::Public);
        assert_eq!(structs[1].visibility(), Visibility::Private);

        let fields: Vec<StructField> = structs[0].fields().collect();
        assert_eq!(fields[0].visibility(), Visibility::Public);
        assert_eq!(fields[1].visibility(), Visibility::Private);
    }

    #[test]
    fn an_import_is_private_until_pub_exports_it() {
        let syntax = parsed("pub from a import x\nfrom a import y\npub import b\nimport c\n");
        let from: Vec<Visibility> = syntax
            .descendants()
            .filter_map(FromImportStmt::cast)
            .map(|import| import.visibility())
            .collect();
        assert_eq!(from, [Visibility::Public, Visibility::Private]);

        let plain: Vec<Visibility> = syntax
            .descendants()
            .filter_map(ImportStmt::cast)
            .map(|import| import.visibility())
            .collect();
        assert_eq!(plain, [Visibility::Public, Visibility::Private]);
    }

    /// A module declaration carries a `mod` of its own, so the narrowing has
    /// to be recognised by the parenthesis after `pub` rather than by the
    /// word inside it. Looking for the word alone read every `pub mod` as
    /// narrowed, which kept every public submodule out of reach.
    #[test]
    fn a_public_module_is_not_a_narrowed_one() {
        let syntax = parsed("mod internal\npub mod math\npub(mod) mod shared\n");
        let reaches: Vec<Visibility> = syntax
            .descendants()
            .filter_map(ModStmt::cast)
            .map(|decl| decl.visibility())
            .collect();
        assert_eq!(
            reaches,
            [Visibility::Private, Visibility::Public, Visibility::Module]
        );
    }

    #[test]
    fn import_reads_its_path_and_alias() {
        let syntax = parsed("import yuzu.std.math as m\n");
        let import = syntax
            .descendants()
            .find_map(ImportStmt::cast)
            .expect("an import");
        let segments: Vec<String> = import
            .path()
            .expect("a path")
            .segments()
            .filter_map(|s| text(Some(s)))
            .collect();
        assert_eq!(segments, ["yuzu", "std", "math"]);
        assert_eq!(text(import.alias()).as_deref(), Some("m"));
    }

    #[test]
    fn from_import_reads_its_items_and_their_renames() {
        let syntax = parsed("from helpers import spread, avg3 as mean\n");
        let import = syntax
            .descendants()
            .find_map(FromImportStmt::cast)
            .expect("a from-import");
        let segments: Vec<String> = import
            .path()
            .expect("a path")
            .segments()
            .filter_map(|s| text(Some(s)))
            .collect();
        assert_eq!(segments, ["helpers"]);

        let items: Vec<ImportItem> = import.items().collect();
        assert_eq!(items.len(), 2);
        assert_eq!(text(items[0].name()).as_deref(), Some("spread"));
        assert!(items[0].alias().is_none());
        assert_eq!(text(items[1].name()).as_deref(), Some("avg3"));
        assert_eq!(text(items[1].alias()).as_deref(), Some("mean"));
    }

    /// Three reaches, and `pub(mod)` is the middle: the files of the
    /// enclosing module see it and nobody beyond them does.
    #[test]
    fn pub_mod_reaches_the_enclosing_module_only() {
        let syntax = parsed(
            "pub struct A { pub x: int, pub(mod) y: int, z: int }\npub(mod) struct B { w: int }\n",
        );
        let structs: Vec<StructStmt> = syntax.descendants().filter_map(StructStmt::cast).collect();
        assert_eq!(structs[0].visibility(), Visibility::Public);
        assert_eq!(structs[1].visibility(), Visibility::Module);

        let fields: Vec<StructField> = structs[0].fields().collect();
        assert_eq!(fields[0].visibility(), Visibility::Public);
        assert_eq!(fields[1].visibility(), Visibility::Module);
        assert_eq!(fields[2].visibility(), Visibility::Private);
    }

    /// Every declaration that carries a name carries a visibility with it.
    #[test]
    fn every_named_declaration_can_be_public() {
        let syntax = parsed(
            "pub table t = P\npub def f(x: int) -> int { return x }\npub let c = 1\npub trait S { def s(x: int) -> int }\n",
        );
        assert_eq!(
            syntax
                .descendants()
                .find_map(TableStmt::cast)
                .expect("a table")
                .visibility(),
            Visibility::Public
        );
        assert_eq!(
            syntax
                .descendants()
                .find_map(FuncStmt::cast)
                .expect("a function")
                .visibility(),
            Visibility::Public
        );
        assert_eq!(
            syntax
                .descendants()
                .find_map(LetStmt::cast)
                .expect("a binding")
                .visibility(),
            Visibility::Public
        );
        assert_eq!(
            syntax
                .descendants()
                .find_map(TraitStmt::cast)
                .expect("a trait")
                .visibility(),
            Visibility::Public
        );
    }

    #[test]
    fn a_pipeline_reads_its_source_and_each_stage() {
        let pipeline = parsed("from t |> where a |> limit 5")
            .descendants()
            .find_map(Pipeline::cast)
            .expect("a pipeline");

        assert_eq!(
            text(pipeline.source().and_then(|from| from.relation())).as_deref(),
            Some("t")
        );
        let stages: Vec<Stage> = pipeline.stages().collect();
        assert!(matches!(
            stages[..],
            [Stage::WhereStage(_), Stage::LimitStage(_)]
        ));
        assert_eq!(stages[1].syntax().text().to_string(), "|> limit 5");
    }

    #[test]
    fn aggregate_reads_items_and_group_by() {
        let stage = aggregate("from t |> aggregate sum(a) as s, count() group by b, e.c as k");

        let items: Vec<AggregateItem> = stage.items().collect();
        assert_eq!(items.len(), 2);
        assert!(items[0].expr().is_some());
        assert_eq!(text(items[0].alias()).as_deref(), Some("s"));
        assert!(items[1].expr().is_some());
        assert!(items[1].alias().is_none());

        let keys: Vec<GroupByItem> = stage.group_by().expect("has group by").items().collect();
        assert_eq!(keys.len(), 2);
        assert!(keys[0].qualifier().is_none());
        assert_eq!(text(keys[0].column()).as_deref(), Some("b"));
        assert!(keys[0].alias().is_none());
        assert_eq!(text(keys[1].qualifier()).as_deref(), Some("e"));
        assert_eq!(text(keys[1].column()).as_deref(), Some("c"));
        assert_eq!(text(keys[1].alias()).as_deref(), Some("k"));
    }

    #[test]
    fn aggregate_without_group_by() {
        let stage = aggregate("from t |> aggregate count()");
        assert!(stage.group_by().is_none());
        assert_eq!(stage.items().count(), 1);
    }

    #[test]
    fn func_stmt_reads_the_agg_marker() {
        let func: FuncStmt = first("agg def agg_of(x: int64) -> int64 { return sum(x) }");
        assert!(func.is_agg());
        assert_eq!(text(func.name()).as_deref(), Some("agg_of"));

        let func: FuncStmt = first("def plain(x: int64) -> int64 { return x }");
        assert!(!func.is_agg());
    }

    #[test]
    fn rename_item_reads_a_qualifier() {
        let item = rename("from t |> rename e.id as eid");
        assert_eq!(text(item.qualifier()).as_deref(), Some("e"));
        assert_eq!(text(item.column()).as_deref(), Some("id"));
        assert_eq!(text(item.alias()).as_deref(), Some("eid"));
    }

    #[test]
    fn rename_item_without_a_qualifier() {
        let item = rename("from t |> rename id as eid");
        assert_eq!(text(item.qualifier()), None);
        assert_eq!(text(item.column()).as_deref(), Some("id"));
        assert_eq!(text(item.alias()).as_deref(), Some("eid"));
    }

    #[test]
    fn join_reads_its_relation_and_alias() {
        let join = join("from t |> join u as d on a == d.b");
        assert_eq!(text(join.relation()).as_deref(), Some("u"));
        assert_eq!(text(join.alias()).as_deref(), Some("d"));
        assert_eq!(join.kind(), JoinKind::Inner);
        assert!(join.on().is_some());
        assert!(join.using().is_none());
    }

    #[test]
    fn join_alias_is_optional_and_may_omit_as() {
        assert_eq!(
            text(join("from t |> join u d on a == d.b").alias()).as_deref(),
            Some("d")
        );
        assert_eq!(text(join("from t |> join u on a == b").alias()), None);
    }

    #[test]
    fn join_reads_its_kind() {
        assert_eq!(
            join("from t |> left join u on a == b").kind(),
            JoinKind::Left
        );
        assert_eq!(
            join("from t |> right join u on a == b").kind(),
            JoinKind::Right
        );
        assert_eq!(
            join("from t |> full join u on a == b").kind(),
            JoinKind::Full
        );
        assert_eq!(
            join("from t |> inner join u on a == b").kind(),
            JoinKind::Inner
        );
    }

    #[test]
    fn join_reads_its_using_columns() {
        let join = join("from t |> join u using (a, b)");
        let columns: Vec<String> = join
            .using()
            .expect("a using clause")
            .columns()
            .filter_map(|column| text(Some(column)))
            .collect();
        assert_eq!(columns, ["a", "b"]);
        assert!(join.on().is_none());
    }

    #[test]
    fn join_ignores_the_idents_of_a_previous_stage() {
        let join = join("from t |> drop a, b |> join u as d on c == d.e");
        assert_eq!(text(join.relation()).as_deref(), Some("u"));
        assert_eq!(text(join.alias()).as_deref(), Some("d"));
    }

    #[test]
    fn each_join_reads_only_its_own_relation_and_kind() {
        let joins: Vec<JoinStage> =
            parsed("from t |> left join u on a == b |> join v as x on c == x.d")
                .descendants()
                .filter_map(JoinStage::cast)
                .collect();

        assert_eq!(text(joins[0].relation()).as_deref(), Some("u"));
        assert_eq!(joins[0].kind(), JoinKind::Left);
        assert_eq!(text(joins[1].relation()).as_deref(), Some("v"));
        assert_eq!(text(joins[1].alias()).as_deref(), Some("x"));
        assert_eq!(joins[1].kind(), JoinKind::Inner);
    }
}
