use id_arena::Id;
use string_interner::symbol::SymbolU32;

pub type SymbolId = SymbolU32;
pub type TypeId = Id<Type>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    Bool,
    String,
    Unit,
    Relation(Relation),
    List(List),
    Struct(Struct),
    Func(FuncType),
    TypeParam(TypeParam),
    Error,

    // A special type that should not be used outside of type unification.
    TypeVar(TypeVariable),
}

impl Type {
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            Type::Int8
                | Type::Int16
                | Type::Int32
                | Type::Int64
                | Type::UInt8
                | Type::UInt16
                | Type::UInt32
                | Type::UInt64
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Relation {
    pub columns: Vec<Column>,
}

/// One column of a relation's row. A join concatenates rows, so a name alone
/// need not be unique; the qualifier is the relation it can be named through —
/// the `from`/`join` alias it came from, or none once a stage computes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Column {
    pub qualifier: Option<SymbolId>,
    pub name: SymbolId,
    pub ty: TypeId,
}

impl Column {
    pub fn new(qualifier: Option<SymbolId>, name: SymbolId, ty: TypeId) -> Self {
        Self {
            qualifier,
            name,
            ty,
        }
    }

    pub fn named_by(&self, qualifier: SymbolId) -> bool {
        self.qualifier == Some(qualifier)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct List {
    pub inner: TypeId,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Struct {
    pub name: SymbolId,
    pub fields: Vec<(SymbolId, TypeId)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FuncType {
    pub args: Vec<TypeId>,
    pub ret_type: TypeId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeParam {
    pub name: SymbolId,
    pub index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeVariable {
    pub index: usize,
    pub kind: InferKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InferKind {
    General,
    Int,
    Float,
}

impl Type {
    pub fn is_int(&self) -> bool {
        matches!(
            self,
            Type::Int8
                | Type::Int16
                | Type::Int32
                | Type::Int64
                | Type::UInt8
                | Type::UInt16
                | Type::UInt32
                | Type::UInt64
        )
    }

    pub fn is_float(&self) -> bool {
        matches!(self, Type::Float32 | Type::Float64)
    }

    pub fn is_numeric(&self) -> bool {
        self.is_int() || self.is_float()
    }

    pub fn is_unsigned(&self) -> bool {
        matches!(
            self,
            Type::UInt8 | Type::UInt16 | Type::UInt32 | Type::UInt64
        )
    }

    pub fn is_hole(&self) -> bool {
        matches!(self, Type::TypeVar { .. })
    }
}

/// A builtin aggregate function. The identity every stage shares: inference
/// types it, the ANF carries it, the plan holds it as a measure, and each
/// emitter maps it onto its dialect at the boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggFunc {
    Count,
    CountDistinct,
    Sum,
    Min,
    Max,
    Avg,
    /// A user-declared `external agg fn`: the target provides it, so it has no
    /// static name here — the symbol names it through the interner.
    External(SymbolId),
}

impl AggFunc {
    pub fn name(self) -> &'static str {
        match self {
            AggFunc::Count => "count",
            AggFunc::CountDistinct => "count_distinct",
            AggFunc::Sum => "sum",
            AggFunc::Min => "min",
            AggFunc::Max => "max",
            AggFunc::Avg => "avg",
            AggFunc::External(_) => unreachable!("an external's name lives in the interner"),
        }
    }
}

/// A builtin scalar function.
///
/// The model's own vocabulary rather than any interchange format's: evaluating
/// a call means knowing what the function does, and two plans calling one
/// function have to compare equal however each spelled it. Emitters map these
/// onto their dialect at the boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Func {
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Negate,
    ShiftLeft,
    ShiftRight,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    And,
    Or,
    Not,
    In,
    /// A user-declared `external fn`: the target provides it, so it has no
    /// static name here — the symbol names it through the interner.
    External(SymbolId),
}

impl Func {
    /// How the function is spelled in source, for diagnostics.
    pub fn symbol(self) -> &'static str {
        match self {
            Func::Add => "+",
            Func::Subtract => "-",
            Func::Multiply => "*",
            Func::Divide => "/",
            Func::Power => "**",
            Func::Negate => "-",
            Func::ShiftLeft => "<<",
            Func::ShiftRight => ">>",
            Func::Equal => "==",
            Func::NotEqual => "!=",
            Func::Less => "<",
            Func::LessEqual => "<=",
            Func::Greater => ">",
            Func::GreaterEqual => ">=",
            Func::And => "and",
            Func::Or => "or",
            Func::Not => "not",
            Func::In => "in",
            Func::External(_) => unreachable!("an external's name lives in the interner"),
        }
    }
}

/// A function the target dialect guarantees. Aggregates are the first kind;
/// builtin scalars join as a sibling variant, so everything that handles a
/// builtin dispatches on the kind rather than assuming aggregates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinFunc {
    Aggregate(AggFunc),
}

impl BuiltinFunc {
    pub fn name(self) -> &'static str {
        match self {
            BuiltinFunc::Aggregate(func) => func.name(),
        }
    }
}
