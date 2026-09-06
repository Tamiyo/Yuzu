//! The dialect types, parsed once per context. MLIR types are uniqued inside
//! a context, so these are handles, not constants — every crate that would
//! otherwise re-parse "!yz.int64" takes them from here.

use melior::Context;
use melior::ir::Type;
use melior::ir::r#type::IntegerType;

pub struct Types<'c> {
    /// The unification variable: a type inference has not resolved yet.
    pub var: Type<'c>,
    /// A relation whose schema resolution has not computed yet.
    pub query: Type<'c>,
    pub int64: Type<'c>,
    pub float64: Type<'c>,
    pub boolean: Type<'c>,
    pub str: Type<'c>,
    /// The builtin i64, which integer attributes are typed with.
    pub i64: Type<'c>,
}

impl<'c> Types<'c> {
    pub fn new(context: &'c Context) -> Self {
        let parse = |text| Type::parse(context, text).expect("the dialect types parse");
        Self {
            var: parse("!yzl.var"),
            query: parse("!yzl.query"),
            int64: parse("!yz.int64"),
            float64: parse("!yz.float64"),
            boolean: parse("!yz.bool"),
            str: parse("!yz.str"),
            i64: IntegerType::new(context, 64).into(),
        }
    }
}
