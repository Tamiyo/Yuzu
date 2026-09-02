use id_arena::Id;
use yuzu_anf_derive::TreeCopy;
use yuzu_core::adt::{Float, Int, SymbolId};
use yuzu_types::{AggFunc, TypeId};

pub type StmtId = Id<Stmt>;
pub type ExprId = Id<Expr>;
pub type AtomId = Id<Atom>;
pub type BindingId = Id<Binding>;
pub type RelId = Id<Rel>;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    And,
    Or,
    In,
    NotIn,
    Eq,
    Neq,
    Lt,
    Lte,
    Gt,
    Gte,
    ShiftLeft,
    ShiftRight,
    UnaryPos,
    UnaryNeg,
    UnaryNot,
}

impl Op {
    pub fn symbol(self) -> &'static str {
        match self {
            Op::Add | Op::UnaryPos => "+",
            Op::Sub | Op::UnaryNeg => "-",
            Op::Mul => "*",
            Op::Div => "/",
            Op::Pow => "**",
            Op::And => "and",
            Op::Or => "or",
            Op::In => "in",
            Op::NotIn => "not in",
            Op::Eq => "==",
            Op::Neq => "!=",
            Op::Lt => "<",
            Op::Lte => "<=",
            Op::Gt => ">",
            Op::Gte => ">=",
            Op::ShiftLeft => "<<",
            Op::ShiftRight => ">>",
            Op::UnaryNot => "not",
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct Root {
    pub stmts: Box<[StmtId]>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ident {
    pub name: SymbolId,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Binding {
    pub name: Ident,
    pub ty: TypeId,
}

#[derive(Clone, PartialEq, Eq, TreeCopy)]
pub struct StructField {
    pub name: Ident,
    pub ty: TypeId,
}

#[derive(Clone, PartialEq, Eq, TreeCopy)]
pub enum Stmt {
    Struct {
        name: Ident,
        fields: Box<[StructField]>,
    },
    Table {
        name: Ident,
        row: TypeId,
    },
    Func {
        binding: BindingId,
        name: Ident,
        params: Box<[BindingId]>,
        ret: TypeId,
        body: Option<StmtId>,
    },
    Block {
        stmts: Box<[StmtId]>,
    },
    Let {
        binding: BindingId,
        expr: ExprId,
    },
    Assign {
        target: AtomId,
        value: ExprId,
    },
    Return {
        value: Option<AtomId>,
    },
    Expr {
        value: ExprId,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, TreeCopy)]
pub struct StructFieldInit {
    pub name: Ident,
    pub value: AtomId,
}

#[derive(Clone, PartialEq, Eq, TreeCopy)]
pub enum Expr {
    Call {
        op: Op,
        args: Box<[AtomId]>,
        ty: TypeId,
    },
    FuncCall {
        callee: AtomId,
        args: Box<[AtomId]>,
        ty: TypeId,
    },
    MethodCall {
        receiver: AtomId,
        method: Ident,
        args: Box<[AtomId]>,
        ty: TypeId,
    },
    StructInit {
        name: SymbolId,
        fields: Box<[StructFieldInit]>,
        ty: TypeId,
    },
    ListInit {
        elements: Box<[AtomId]>,
        ty: TypeId,
    },
    Rel(RelId),
    AggCall {
        func: AggFunc,
        args: Box<[AtomId]>,
        ty: TypeId,
    },
    /// A call to an `external fn`: the target provides the body, so the call
    /// reduces its arguments and survives to the plan by name.
    ExternCall {
        name: Ident,
        args: Box<[AtomId]>,
        ty: TypeId,
    },
    Atom {
        value: AtomId,
    },
}

/// A per-row computation (a `select` column or `where` predicate): the temporary
/// bindings it needs, ending in the `value` atom it yields.
#[derive(Clone, PartialEq, Eq, Hash, TreeCopy)]
pub struct Thunk {
    pub stmts: Box<[StmtId]>,
    pub value: AtomId,
}

#[derive(Clone, PartialEq, Eq, Hash, TreeCopy)]
pub struct SelectItem {
    pub body: Thunk,
    pub alias: Option<Ident>,
}

#[derive(Clone, PartialEq, Eq, Hash, TreeCopy)]
pub struct SetItem {
    pub column: Ident,
    pub value: Thunk,
}

#[derive(Clone, PartialEq, Eq, Hash, TreeCopy)]
pub struct AggregateItem {
    pub body: Thunk,
    pub alias: Option<Ident>,
}

/// A group key, resolved to its input-row position by inference; `name` is the
/// output column's name, kept for printing.
#[derive(Clone, Copy, PartialEq, Eq, Hash, TreeCopy)]
pub struct GroupKey {
    pub name: Ident,
    pub column: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, TreeCopy)]
pub struct RenameItem {
    pub from: Ident,
    pub to: Ident,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum JoinKind {
    Inner,
    Left,
    Right,
    Full,
}

impl JoinKind {
    pub fn keyword(self) -> &'static str {
        match self {
            JoinKind::Inner => "inner",
            JoinKind::Left => "left",
            JoinKind::Right => "right",
            JoinKind::Full => "full",
        }
    }
}

/// Not desugared to an equality: both sides spell the column the same way, so
/// only the plan's field indices can tell them apart.
#[derive(Clone, PartialEq, Eq, Hash, TreeCopy)]
pub enum JoinCondition {
    On(Thunk),
    Using(Box<[Ident]>),
}

/// A relational pipeline stage. Inputs nest, so a `RelId` chain mirrors the
/// `|>` pipeline. Each carries its own `Relation[row]` type.
#[derive(Clone, PartialEq, Eq, Hash, TreeCopy)]
pub enum Rel {
    From {
        relation: Ident,
        alias: Option<Ident>,
        ty: TypeId,
    },
    Join {
        left: RelId,
        right: RelId,
        kind: JoinKind,
        condition: JoinCondition,
        ty: TypeId,
    },
    Select {
        input: RelId,
        items: Box<[SelectItem]>,
        ty: TypeId,
    },
    Where {
        input: RelId,
        predicate: Thunk,
        ty: TypeId,
    },
    Distinct {
        input: RelId,
        ty: TypeId,
    },
    Drop {
        input: RelId,
        columns: Box<[Ident]>,
        ty: TypeId,
    },
    Rename {
        input: RelId,
        items: Box<[RenameItem]>,
        ty: TypeId,
    },
    Extend {
        input: RelId,
        items: Box<[SelectItem]>,
        ty: TypeId,
    },
    Set {
        input: RelId,
        items: Box<[SetItem]>,
        ty: TypeId,
    },
    /// The counts stay thunks so the reducer folds them; a plan needs constants
    /// and emission reports anything that did not reduce to one.
    Limit {
        input: RelId,
        count: Thunk,
        offset: Option<Thunk>,
        ty: TypeId,
    },
    /// Renames the whole row: the columns are unchanged, but from here they are
    /// named through `alias`.
    Alias {
        input: RelId,
        alias: Ident,
        ty: TypeId,
    },
    Aggregate {
        input: RelId,
        items: Box<[AggregateItem]>,
        groups: Box<[GroupKey]>,
        ty: TypeId,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Atom {
    Var {
        binding: BindingId,
    },
    FuncRef {
        binding: BindingId,
        ty: TypeId,
    },
    Field {
        base: AtomId,
        field: Ident,
        ty: TypeId,
    },
    /// A column of the row flowing through the query, resolved to its position
    /// by inference — a join concatenates rows, so a name alone no longer says
    /// which column is meant. `name` is kept for printing.
    Column {
        name: Ident,
        column: u32,
        ty: TypeId,
    },
    Const(Const),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Const {
    Int { value: Int },
    Float { value: Float },
    Bool { value: bool },
    String { value: SymbolId },
}

/// Rebuilds a node against a copier: tree-id fields (`StmtId`, `ExprId`,
/// `RelId`) route through the copier, which translates them into the
/// destination arenas; pool ids (atoms, bindings) and plain data pass through
/// unchanged. Derived per node — each variant rebuilds by calling `copy_tree`
/// on every field, and the field's type decides what that means.
pub trait TreeCopy {
    fn copy_tree(&self, copier: &mut impl TreeCopier) -> Self;
}

/// Translates tree ids from one program's trees into another's.
pub trait TreeCopier {
    fn copy_stmt(&mut self, id: StmtId) -> StmtId;
    fn copy_expr(&mut self, id: ExprId) -> ExprId;
    fn copy_rel(&mut self, id: RelId) -> RelId;
}

impl TreeCopy for StmtId {
    fn copy_tree(&self, copier: &mut impl TreeCopier) -> Self {
        copier.copy_stmt(*self)
    }
}

impl TreeCopy for ExprId {
    fn copy_tree(&self, copier: &mut impl TreeCopier) -> Self {
        copier.copy_expr(*self)
    }
}

impl TreeCopy for RelId {
    fn copy_tree(&self, copier: &mut impl TreeCopier) -> Self {
        copier.copy_rel(*self)
    }
}

/// Leaves: pool ids and plain data, valid in any program.
macro_rules! leaf_copy {
    ($($ty:ty),* $(,)?) => {$(
        impl TreeCopy for $ty {
            fn copy_tree(&self, _copier: &mut impl TreeCopier) -> Self {
                *self
            }
        }
    )*};
}

leaf_copy!(
    AtomId, BindingId, Ident, SymbolId, TypeId, Op, JoinKind, AggFunc, u32
);

impl<T: TreeCopy> TreeCopy for Option<T> {
    fn copy_tree(&self, copier: &mut impl TreeCopier) -> Self {
        self.as_ref().map(|value| value.copy_tree(copier))
    }
}

impl<T: TreeCopy> TreeCopy for Box<[T]> {
    fn copy_tree(&self, copier: &mut impl TreeCopier) -> Self {
        self.iter().map(|value| value.copy_tree(copier)).collect()
    }
}
