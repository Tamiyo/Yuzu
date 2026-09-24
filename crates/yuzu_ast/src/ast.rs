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
    ($name:ident, { $($variant:ident),+ $(,)? }) => {
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

/// How far a declaration's name reaches. `Module` is what `pub(mod)` asks
/// for: the files of the enclosing module and no further, which is how a
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
    fn from_token(token: SyntaxToken) -> Option<Self> {
        Some(match token.kind() {
            SyntaxKind::Plus => BinOp::Add,
            SyntaxKind::Minus => BinOp::Sub,
            SyntaxKind::Star => BinOp::Mul,
            SyntaxKind::Slash => BinOp::Div,
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
    fn from_token(token: SyntaxToken) -> Option<Self> {
        Some(match token.kind() {
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
    pub fn text(&self) -> Option<String> {
        Some(self.0.first_token()?.text().to_string())
    }
}

ast_enum!(TypeAnnotation, {
    NamedTypeAnnotation,
    FuncTypeAnnotation,
});

ast_node!(NamedTypeAnnotation);
impl NamedTypeAnnotation {
    pub fn name(&self) -> Option<Ident> {
        support::child(self.syntax())
    }

    pub fn name_text(&self) -> Option<String> {
        self.name().and_then(|ident| ident.text())
    }
    pub fn args(&self) -> impl Iterator<Item = TypeAnnotation> + use<> {
        support::children(self.syntax())
    }
}

ast_node!(FuncTypeAnnotation);
impl FuncTypeAnnotation {
    pub fn params(&self) -> Option<FuncTypeAnnotationParams> {
        support::child(self.syntax())
    }
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
    pub fn name(&self) -> Option<Ident> {
        support::child(self.syntax())
    }

    pub fn name_text(&self) -> Option<String> {
        self.name().and_then(|ident| ident.text())
    }
}

ast_node!(TypeBound);
impl TypeBound {
    pub fn subject(&self) -> Option<Ident> {
        support::child(self.syntax())
    }

    pub fn subject_text(&self) -> Option<String> {
        self.subject().and_then(|ident| ident.text())
    }

    pub fn traits(&self) -> impl Iterator<Item = TraitRef> + use<> {
        support::children(self.syntax())
    }
}

ast_node!(TraitRef);
impl TraitRef {
    pub fn name(&self) -> Option<Ident> {
        support::child(self.syntax())
    }

    pub fn name_text(&self) -> Option<String> {
        self.name().and_then(|ident| ident.text())
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
    /// The segments, outermost first. One segment names a module directly;
    /// more name the path through the modules holding it.
    pub fn segments(&self) -> impl Iterator<Item = Ident> + use<> {
        support::children(self.syntax())
    }

    pub fn segments_text(&self) -> impl Iterator<Item = String> + use<> {
        self.segments().filter_map(|ident| ident.text())
    }
}

ast_node!(ModStmt);
impl ModStmt {
    /// Whether another file may name this. Private unless `pub` says so, so
    /// forgetting to export is a complaint from the importer rather than a
    /// name that quietly became API.
    pub fn visibility(&self) -> Visibility {
        visibility_of(self.syntax())
    }

    pub fn name(&self) -> Option<Ident> {
        support::child(self.syntax())
    }

    pub fn name_text(&self) -> Option<String> {
        self.name().and_then(|ident| ident.text())
    }
}

ast_node!(ImportStmt);
impl ImportStmt {
    pub fn path(&self) -> Option<ModulePath> {
        support::child(self.syntax())
    }

    /// The name this file calls the module, when `as` gave it one.
    pub fn alias(&self) -> Option<Ident> {
        support::child(self.syntax())
    }

    pub fn alias_text(&self) -> Option<String> {
        self.alias().and_then(|ident| ident.text())
    }
}

ast_node!(FromImportStmt);
impl FromImportStmt {
    pub fn path(&self) -> Option<ModulePath> {
        support::child(self.syntax())
    }

    pub fn items(&self) -> impl Iterator<Item = ImportItem> + use<> {
        support::children(self.syntax())
    }
}

ast_node!(ImportItem);
impl ImportItem {
    pub fn name(&self) -> Option<Ident> {
        support::child(self.syntax())
    }

    pub fn name_text(&self) -> Option<String> {
        self.name().and_then(|ident| ident.text())
    }

    /// The name this file calls the import, when `as` gave it one.
    pub fn alias(&self) -> Option<Ident> {
        support::children(self.syntax()).nth(1)
    }

    pub fn alias_text(&self) -> Option<String> {
        self.alias().and_then(|ident| ident.text())
    }
}

ast_node!(StructStmt);
impl StructStmt {
    /// Whether another file may name this. Private unless `pub` says so, so
    /// forgetting to export is a complaint from the importer rather than a
    /// name that quietly became API.
    pub fn visibility(&self) -> Visibility {
        visibility_of(self.syntax())
    }

    pub fn name(&self) -> Option<Ident> {
        support::child(self.syntax())
    }

    pub fn name_text(&self) -> Option<String> {
        self.name().and_then(|name| name.text())
    }

    pub fn fields(&self) -> impl Iterator<Item = StructField> + use<> {
        support::children(self.syntax())
    }
}

ast_node!(StructField);
impl StructField {
    pub fn visibility(&self) -> Visibility {
        visibility_of(self.syntax())
    }

    pub fn name(&self) -> Option<Ident> {
        support::child(self.syntax())
    }

    pub fn name_text(&self) -> Option<String> {
        self.name().and_then(|name| name.text())
    }

    pub fn mutability(&self) -> Mutability {
        let is_mut = self
            .syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .any(|token| token.kind() == SyntaxKind::MutKw);

        if is_mut {
            Mutability::Mutable
        } else {
            Mutability::Immutable
        }
    }

    pub fn ty(&self) -> Option<TypeAnnotation> {
        support::child(self.syntax())
    }
}

ast_node!(TraitStmt);
impl TraitStmt {
    pub fn visibility(&self) -> Visibility {
        visibility_of(self.syntax())
    }

    pub fn name(&self) -> Option<Ident> {
        support::child(self.syntax())
    }

    pub fn name_text(&self) -> Option<String> {
        self.name().and_then(|name| name.text())
    }

    pub fn methods(&self) -> impl Iterator<Item = FuncStmt> + use<> {
        support::children(self.syntax())
    }
}

ast_node!(ImplStmt);
impl ImplStmt {
    pub fn trait_(&self) -> Option<TraitRef> {
        support::child(self.syntax())
    }

    pub fn ty(&self) -> Option<Ident> {
        support::child(self.syntax())
    }

    pub fn ty_text(&self) -> Option<String> {
        self.ty().and_then(|ident| ident.text())
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
    pub fn visibility(&self) -> Visibility {
        visibility_of(self.syntax())
    }
    pub fn is_agg(&self) -> bool {
        self.has_marker(SyntaxKind::AggKw)
    }

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

    pub fn name(&self) -> Option<Ident> {
        support::child(self.syntax())
    }

    pub fn name_text(&self) -> Option<String> {
        self.name().and_then(|name| name.text())
    }

    pub fn type_params(&self) -> impl Iterator<Item = TypeParam> + use<> {
        support::children(self.syntax())
    }

    pub fn params(&self) -> impl Iterator<Item = FuncParam> + use<> {
        support::children(self.syntax())
    }

    pub fn param_count(&self) -> usize {
        self.params().count()
    }

    pub fn result(&self) -> Option<TypeAnnotation> {
        support::child(self.syntax())
    }

    pub fn bounds(&self) -> impl Iterator<Item = TypeBound> + use<> {
        support::children(self.syntax())
    }

    pub fn body(&self) -> Option<BlockStmt> {
        support::child(self.syntax())
    }
}

ast_node!(FuncParam);
impl FuncParam {
    pub fn name(&self) -> Option<Ident> {
        support::child(self.syntax())
    }

    pub fn name_text(&self) -> Option<String> {
        self.name().and_then(|ident| ident.text())
    }

    pub fn ty(&self) -> Option<TypeAnnotation> {
        support::child(self.syntax())
    }
}

ast_node!(TableStmt);
impl TableStmt {
    pub fn visibility(&self) -> Visibility {
        visibility_of(self.syntax())
    }

    pub fn name(&self) -> Option<Ident> {
        support::nth_child(self.syntax(), 0)
    }

    pub fn name_text(&self) -> Option<String> {
        self.name().and_then(|name| name.text())
    }

    pub fn struct_name(&self) -> Option<Ident> {
        support::nth_child(self.syntax(), 1)
    }

    pub fn struct_name_text(&self) -> Option<String> {
        self.struct_name().and_then(|ident| ident.text())
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
    pub fn visibility(&self) -> Visibility {
        visibility_of(self.syntax())
    }

    pub fn name(&self) -> Option<Ident> {
        support::child(self.syntax())
    }

    pub fn name_text(&self) -> Option<String> {
        self.name().and_then(|name| name.text())
    }

    pub fn mutability(&self) -> Mutability {
        let is_mut = self
            .syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .any(|token| token.kind() == SyntaxKind::MutKw);

        if is_mut {
            Mutability::Mutable
        } else {
            Mutability::Immutable
        }
    }

    pub fn type_annotation(&self) -> Option<TypeAnnotation> {
        support::child(self.syntax())
    }

    pub fn expr(&self) -> Option<Expr> {
        support::child(self.syntax())
    }
}

ast_node!(AssignStmt);
impl AssignStmt {
    pub fn target(&self) -> Option<Expr> {
        support::nth_child(self.syntax(), 0)
    }

    pub fn value(&self) -> Option<Expr> {
        support::nth_child(self.syntax(), 1)
    }
}

ast_node!(ReturnStmt);
impl ReturnStmt {
    pub fn expr(&self) -> Option<Expr> {
        support::child(self.syntax())
    }
}

ast_node!(ExprStmt);
impl ExprStmt {
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
    Rel,
});

ast_node!(IdentExpr);
impl IdentExpr {
    pub fn name(&self) -> Option<Ident> {
        support::child(self.syntax())
    }

    pub fn name_text(&self) -> Option<String> {
        self.name().and_then(|ident| ident.text())
    }
}

ast_node!(CallExpr);
impl CallExpr {
    pub fn callee(&self) -> Option<Expr> {
        support::child(self.syntax())
    }

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
    pub fn base(&self) -> Option<Expr> {
        support::child(self.syntax())
    }

    pub fn field(&self) -> Option<Ident> {
        support::child(self.syntax())
    }

    pub fn field_text(&self) -> Option<String> {
        self.field().and_then(|ident| ident.text())
    }
}

ast_node!(StructExpr);
impl StructExpr {
    pub fn name(&self) -> Option<Ident> {
        support::child(self.syntax())
    }

    pub fn name_text(&self) -> Option<String> {
        self.name().and_then(|ident| ident.text())
    }

    pub fn fields(&self) -> impl Iterator<Item = StructFieldInit> + use<> {
        support::children(self.syntax())
    }
}

ast_node!(StructFieldInit);
impl StructFieldInit {
    pub fn name(&self) -> Option<Ident> {
        support::child(self.syntax())
    }

    pub fn name_text(&self) -> Option<String> {
        self.name().and_then(|ident| ident.text())
    }

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
    pub fn lhs(&self) -> Option<Expr> {
        support::nth_child(self.syntax(), 0)
    }

    pub fn op(&self) -> Option<BinOp> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find_map(BinOp::from_token)
    }

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
            .find_map(UnaryOp::from_token)
    }

    pub fn expr(&self) -> Option<Expr> {
        support::child(self.syntax())
    }
}

ast_node!(ParenExpr);
impl ParenExpr {
    pub fn expr(&self) -> Option<Expr> {
        support::child(self.syntax())
    }
}

ast_enum!(Rel, {
    SetExpr,
    LimitExpr,
    AliasExpr,
    AggregateExpr,
    FromExpr,
    SelectExpr,
    WhereExpr,
    DistinctExpr,
    DropExpr,
    RenameExpr,
    ExtendExpr,
    JoinExpr,
});

ast_node!(FromExpr);
impl FromExpr {
    pub fn relation(&self) -> Option<Ident> {
        support::nth_child(self.syntax(), 0)
    }

    pub fn relation_text(&self) -> Option<String> {
        self.relation().and_then(|ident| ident.text())
    }

    pub fn alias(&self) -> Option<Ident> {
        support::nth_child(self.syntax(), 1)
    }

    pub fn alias_text(&self) -> Option<String> {
        self.alias().and_then(|ident| ident.text())
    }
}

ast_node!(SelectExpr);
impl SelectExpr {
    pub fn input(&self) -> Option<Expr> {
        support::child(self.syntax())
    }

    pub fn items(&self) -> impl Iterator<Item = SelectItem> + use<> {
        support::children(self.syntax())
    }
}

ast_node!(SelectItem);
impl SelectItem {
    pub fn expr(&self) -> Option<Expr> {
        support::child(self.syntax())
    }

    pub fn alias(&self) -> Option<Ident> {
        support::child(self.syntax())
    }

    pub fn alias_text(&self) -> Option<String> {
        self.alias().and_then(|ident| ident.text())
    }
}

ast_node!(WhereExpr);
impl WhereExpr {
    pub fn input(&self) -> Option<Expr> {
        support::nth_child(self.syntax(), 0)
    }

    pub fn predicate(&self) -> Option<Expr> {
        support::nth_child(self.syntax(), 1)
    }
}

ast_node!(DistinctExpr);
impl DistinctExpr {
    pub fn input(&self) -> Option<Expr> {
        support::child(self.syntax())
    }
}

ast_node!(DropExpr);
impl DropExpr {
    pub fn input(&self) -> Option<Expr> {
        support::child(self.syntax())
    }

    pub fn columns(&self) -> impl Iterator<Item = Ident> + use<> {
        support::children(self.syntax())
    }

    pub fn columns_text(&self) -> impl Iterator<Item = String> + use<> {
        self.columns().filter_map(|ident| ident.text())
    }
}

ast_node!(RenameExpr);
impl RenameExpr {
    pub fn input(&self) -> Option<Expr> {
        support::child(self.syntax())
    }

    pub fn items(&self) -> impl Iterator<Item = RenameItem> + use<> {
        support::children(self.syntax())
    }
}

ast_node!(RenameItem);
impl RenameItem {
    /// `e.id as eid` names the column of one input; a bare `id as eid` has to
    /// find it on its own.
    pub fn qualifier(&self) -> Option<Ident> {
        self.is_qualified()
            .then(|| support::nth_child(self.syntax(), 0))
            .flatten()
    }

    pub fn qualifier_text(&self) -> Option<String> {
        self.qualifier().and_then(|ident| ident.text())
    }

    pub fn from(&self) -> Option<Ident> {
        support::nth_child(self.syntax(), self.is_qualified() as usize)
    }

    pub fn from_text(&self) -> Option<String> {
        self.from().and_then(|ident| ident.text())
    }

    pub fn to(&self) -> Option<Ident> {
        support::nth_child(self.syntax(), self.is_qualified() as usize + 1)
    }

    pub fn to_text(&self) -> Option<String> {
        self.to().and_then(|ident| ident.text())
    }

    fn is_qualified(&self) -> bool {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .any(|token| token.kind() == SyntaxKind::Dot)
    }
}

ast_node!(ExtendExpr);
impl ExtendExpr {
    pub fn input(&self) -> Option<Expr> {
        support::child(self.syntax())
    }

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
    fn from_token(token: SyntaxToken) -> Option<Self> {
        Some(match token.kind() {
            SyntaxKind::InnerKw => JoinKind::Inner,
            SyntaxKind::LeftKw => JoinKind::Left,
            SyntaxKind::RightKw => JoinKind::Right,
            SyntaxKind::FullKw => JoinKind::Full,
            _ => return None,
        })
    }
}

ast_node!(JoinExpr);
impl JoinExpr {
    pub fn input(&self) -> Option<Expr> {
        support::child(self.syntax())
    }

    pub fn kind(&self) -> Option<JoinKind> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find_map(JoinKind::from_token)
    }

    pub fn relation(&self) -> Option<Ident> {
        support::nth_child(self.syntax(), 0)
    }

    pub fn relation_text(&self) -> Option<String> {
        self.relation().and_then(|ident| ident.text())
    }

    pub fn alias(&self) -> Option<Ident> {
        support::nth_child(self.syntax(), 1)
    }

    pub fn alias_text(&self) -> Option<String> {
        self.alias().and_then(|ident| ident.text())
    }

    pub fn on(&self) -> Option<JoinOn> {
        support::child(self.syntax())
    }

    pub fn using(&self) -> Option<JoinUsing> {
        support::child(self.syntax())
    }
}

ast_node!(JoinOn);
impl JoinOn {
    pub fn condition(&self) -> Option<Expr> {
        support::child(self.syntax())
    }
}

ast_node!(JoinUsing);
impl JoinUsing {
    pub fn columns(&self) -> impl Iterator<Item = Ident> + use<> {
        support::children(self.syntax())
    }

    pub fn columns_text(&self) -> impl Iterator<Item = String> + use<> {
        self.columns().filter_map(|ident| ident.text())
    }
}

ast_node!(SetExpr);
impl SetExpr {
    pub fn input(&self) -> Option<Expr> {
        support::child(self.syntax())
    }

    pub fn items(&self) -> impl Iterator<Item = SetItem> + use<> {
        support::children(self.syntax())
    }
}

ast_node!(SetItem);
impl SetItem {
    pub fn column(&self) -> Option<Ident> {
        support::child(self.syntax())
    }

    pub fn column_text(&self) -> Option<String> {
        self.column().and_then(|ident| ident.text())
    }

    pub fn value(&self) -> Option<Expr> {
        support::child(self.syntax())
    }
}

ast_node!(LimitExpr);
impl LimitExpr {
    pub fn input(&self) -> Option<Expr> {
        support::nth_child(self.syntax(), 0)
    }

    pub fn count(&self) -> Option<Expr> {
        support::nth_child(self.syntax(), 1)
    }

    pub fn offset(&self) -> Option<Expr> {
        support::nth_child(self.syntax(), 2)
    }
}

ast_node!(AliasExpr);
impl AliasExpr {
    pub fn input(&self) -> Option<Expr> {
        support::child(self.syntax())
    }

    pub fn alias(&self) -> Option<Ident> {
        support::child(self.syntax())
    }

    pub fn alias_text(&self) -> Option<String> {
        self.alias().and_then(|ident| ident.text())
    }
}

ast_node!(AggregateExpr);
impl AggregateExpr {
    pub fn input(&self) -> Option<Expr> {
        support::child(self.syntax())
    }

    pub fn items(&self) -> impl Iterator<Item = AggregateItem> + use<> {
        support::children(self.syntax())
    }

    pub fn group_by(&self) -> Option<GroupBy> {
        support::child(self.syntax())
    }
}

ast_node!(AggregateItem);
impl AggregateItem {
    pub fn expr(&self) -> Option<Expr> {
        support::child(self.syntax())
    }

    pub fn alias(&self) -> Option<Ident> {
        support::child(self.syntax())
    }

    pub fn alias_text(&self) -> Option<String> {
        self.alias().and_then(|ident| ident.text())
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
    pub fn qualifier(&self) -> Option<Ident> {
        self.is_qualified()
            .then(|| support::nth_child(self.syntax(), 0))
            .flatten()
    }

    pub fn qualifier_text(&self) -> Option<String> {
        self.qualifier().and_then(|ident| ident.text())
    }

    pub fn column(&self) -> Option<Ident> {
        support::nth_child(self.syntax(), self.is_qualified() as usize)
    }

    pub fn column_text(&self) -> Option<String> {
        self.column().and_then(|ident| ident.text())
    }

    pub fn alias(&self) -> Option<Ident> {
        self.is_aliased()
            .then(|| support::nth_child(self.syntax(), self.is_qualified() as usize + 1))
            .flatten()
    }

    pub fn alias_text(&self) -> Option<String> {
        self.alias().and_then(|ident| ident.text())
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
    pub fn value(&self) -> Option<bool> {
        self.0.first_token()?.text().parse().ok()
    }
}

ast_node!(IntLiteral);
impl IntLiteral {
    pub fn value(&self) -> Option<u64> {
        self.0.first_token()?.text().parse().ok()
    }
}

ast_node!(FloatLiteral);
impl FloatLiteral {
    pub fn value(&self) -> Option<f64> {
        self.0.first_token()?.text().parse().ok()
    }
}

ast_node!(StringLiteral);
impl StringLiteral {
    pub fn value(&self) -> Option<String> {
        let text = self.0.first_token()?.text().to_string();
        let unquoted = text
            .strip_prefix('"')
            .and_then(|inner| inner.strip_suffix('"'))
            .unwrap_or(text.as_str());
        Some(unquoted.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yuzu_diagnostics::{diagnostics::engine::DiagnosticsEngine, source_map::SourceMap};
    use yuzu_lexer::lexer::{Lexer, Token};

    /// The outermost join stage, so a chained query yields its last stage.
    fn join(input: &str) -> JoinExpr {
        let tokens: Vec<Token> = Lexer::new(input).collect();
        let mut diagnostics = DiagnosticsEngine::new();
        let mut sources = SourceMap::new();
        let source_id = sources.add("test".to_string(), input.to_string());

        let syntax = yuzu_parser::parse(&tokens, &mut diagnostics, source_id);
        syntax
            .descendants()
            .find_map(JoinExpr::cast)
            .expect("input has a join stage")
    }

    fn text(ident: Option<Ident>) -> Option<String> {
        ident.and_then(|ident| ident.text())
    }

    fn rename(input: &str) -> RenameItem {
        let tokens: Vec<Token> = Lexer::new(input).collect();
        let mut diagnostics = DiagnosticsEngine::new();
        let mut sources = SourceMap::new();
        let source_id = sources.add("test".to_string(), input.to_string());

        let syntax = yuzu_parser::parse(&tokens, &mut diagnostics, source_id);
        syntax
            .descendants()
            .find_map(RenameItem::cast)
            .expect("input has a rename item")
    }

    fn aggregate(input: &str) -> AggregateExpr {
        let tokens: Vec<Token> = Lexer::new(input).collect();
        let mut diagnostics = DiagnosticsEngine::new();
        let mut sources = SourceMap::new();
        let source_id = sources.add("test".to_string(), input.to_string());

        let syntax = yuzu_parser::parse(&tokens, &mut diagnostics, source_id);
        syntax
            .descendants()
            .find_map(AggregateExpr::cast)
            .expect("input has an aggregate stage")
    }

    fn parsed(input: &str) -> SyntaxNode {
        let tokens: Vec<Token> = Lexer::new(input).collect();
        let mut diagnostics = DiagnosticsEngine::new();
        let mut sources = SourceMap::new();
        let source_id = sources.add("test".to_string(), input.to_string());
        yuzu_parser::parse(&tokens, &mut diagnostics, source_id)
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
            .filter_map(|s| s.text())
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
            .filter_map(|s| s.text())
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
    fn aggregate_reads_items_and_group_by() {
        let stage = aggregate("from t |> aggregate sum(a) as s, count() group by b, e.c as k");
        assert!(stage.input().is_some());

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
        let tokens: Vec<Token> =
            Lexer::new("agg def agg_of(x: int64) -> int64 { return sum(x) }").collect();
        let mut diagnostics = DiagnosticsEngine::new();
        let mut sources = SourceMap::new();
        let source_id = sources.add("test".to_string(), "x".to_string());
        let syntax = yuzu_parser::parse(&tokens, &mut diagnostics, source_id);
        let func = syntax
            .descendants()
            .find_map(FuncStmt::cast)
            .expect("input has a function");
        assert!(func.is_agg());
        assert_eq!(text(func.name()).as_deref(), Some("agg_of"));

        let tokens: Vec<Token> = Lexer::new("def plain(x: int64) -> int64 { return x }").collect();
        let syntax = yuzu_parser::parse(&tokens, &mut diagnostics, source_id);
        let func = syntax
            .descendants()
            .find_map(FuncStmt::cast)
            .expect("input has a function");
        assert!(!func.is_agg());
    }

    #[test]
    fn rename_item_reads_a_qualifier() {
        let item = rename("from t |> rename e.id as eid");
        assert_eq!(text(item.qualifier()).as_deref(), Some("e"));
        assert_eq!(text(item.from()).as_deref(), Some("id"));
        assert_eq!(text(item.to()).as_deref(), Some("eid"));
    }

    #[test]
    fn rename_item_without_a_qualifier() {
        let item = rename("from t |> rename id as eid");
        assert_eq!(text(item.qualifier()), None);
        assert_eq!(text(item.from()).as_deref(), Some("id"));
        assert_eq!(text(item.to()).as_deref(), Some("eid"));
    }

    #[test]
    fn join_reads_its_relation_and_alias() {
        let join = join("from t |> join u as d on a == d.b");
        assert_eq!(text(join.relation()).as_deref(), Some("u"));
        assert_eq!(text(join.alias()).as_deref(), Some("d"));
        assert_eq!(join.kind(), None);
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
            Some(JoinKind::Left)
        );
        assert_eq!(
            join("from t |> right join u on a == b").kind(),
            Some(JoinKind::Right)
        );
        assert_eq!(
            join("from t |> full join u on a == b").kind(),
            Some(JoinKind::Full)
        );
        assert_eq!(
            join("from t |> inner join u on a == b").kind(),
            Some(JoinKind::Inner)
        );
    }

    #[test]
    fn join_reads_its_using_columns() {
        let join = join("from t |> join u using (a, b)");
        let columns: Vec<String> = join
            .using()
            .expect("a using clause")
            .columns()
            .filter_map(|column| column.text())
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
    fn a_nested_join_keeps_its_own_relation_and_kind() {
        let outer = join("from t |> left join u on a == b |> join v as x on c == x.d");
        assert_eq!(text(outer.relation()).as_deref(), Some("v"));
        assert_eq!(text(outer.alias()).as_deref(), Some("x"));
        assert_eq!(outer.kind(), None);
    }
}
