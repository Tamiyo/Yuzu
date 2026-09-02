use id_arena::Id;

use crate::hir::{ExprId, Ident, TraitRef, TypeAnnotationId, TypeBound};

pub type StmtId = Id<Stmt>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mutability {
    Immutable,
    Mutable,
}

#[derive(Clone, PartialEq, Eq)]
pub struct StructField {
    pub name: Ident,
    pub mutability: Mutability,
    pub type_annotation: TypeAnnotationId,
}

#[derive(Clone, PartialEq, Eq)]
pub struct FuncParam {
    pub name: Ident,
    pub type_annotation: TypeAnnotationId,
}

#[derive(Clone, PartialEq, Eq)]
pub enum Stmt {
    Struct {
        name: Ident,
        fields: Box<[StructField]>,
    },
    Impl {
        trait_ref: Option<TraitRef>,
        name: Ident,
        methods: Box<[StmtId]>,
    },
    Trait {
        name: Ident,
        methods: Box<[StmtId]>,
    },
    Func {
        name: Ident,
        type_params: Box<[Ident]>,
        params: Box<[FuncParam]>,
        type_bounds: Box<[TypeBound]>,
        ret_type_annotation: TypeAnnotationId,
        body: Option<StmtId>,
        is_agg: bool,
        is_external: bool,
    },
    Block {
        stmts: Box<[StmtId]>,
    },
    Table {
        name: Ident,
        row: Ident,
    },
    InlineTable {
        name: Ident,
        fields: Box<[StructField]>,
    },
    Let {
        name: Ident,
        mutability: Mutability,
        type_annotation: Option<TypeAnnotationId>,
        expr: ExprId,
    },
    Assign {
        target: ExprId,
        value: ExprId,
    },
    Return {
        expr: Option<ExprId>,
    },
    Expr {
        expr: ExprId,
    },

    /// Error-recovery node for input that failed to lower.
    Missing,
}
