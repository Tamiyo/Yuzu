/// A builtin aggregate function. Inference types it, and the Substrait
/// translation maps it onto an extension function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggFunc {
    Count,
    CountDistinct,
}

impl AggFunc {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            AggFunc::Count => "count",
            AggFunc::CountDistinct => "count_distinct",
        }
    }
}

/// A builtin scalar function.
///
/// The model's own vocabulary rather than any interchange format's: evaluating
/// a call means knowing what the function does, and two plans calling one
/// function have to compare equal however each spelled it. The Substrait
/// translation maps these onto its extension functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Func {
    Add,
    Subtract,
    Multiply,
    Divide,
    Negate,
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
}

impl Func {
    /// How the function is spelled in source, for diagnostics.
    #[must_use]
    pub fn symbol(self) -> &'static str {
        match self {
            Func::Add => "+",
            Func::Subtract | Func::Negate => "-",
            Func::Multiply => "*",
            Func::Divide => "/",
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
        }
    }
}

/// A function the target dialect guarantees.
///
/// Aggregates are the first kind; builtin scalars join as a sibling variant,
/// so everything that handles a builtin dispatches on the kind rather than
/// assuming aggregates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinFunc {
    Aggregate(AggFunc),
    /// A scalar the language spells as an operator — the parser lowers `**`,
    /// `<<`, `>>` and `in` to calls, so they resolve like any other builtin.
    Scalar(Func),
}

impl BuiltinFunc {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            BuiltinFunc::Aggregate(func) => func.name(),
            BuiltinFunc::Scalar(func) => func.symbol(),
        }
    }
}
