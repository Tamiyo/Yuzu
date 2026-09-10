//! The dialects' parameterless types. MLIR uniques types in the context, so
//! each of these is a lookup that returns the same type every time — hold
//! one in a field where a hot path wants it, rather than passing a bag of
//! them around.

use melior::Context;
use melior::ir::Type;

macro_rules! singleton {
    ($name:ident, $get:ident, $doc:literal) => {
        #[doc = $doc]
        pub fn $name(context: &Context) -> Type<'_> {
            unsafe { Type::from_raw(yuzu_mlir_sys::$get(context.to_raw())) }
        }
    };
}

singleton!(int64, yzuInt64TypeGet, "`!yz.int64`");
singleton!(float64, yzuFloat64TypeGet, "`!yz.float64`");
singleton!(boolean, yzuBoolTypeGet, "`!yz.bool`");
singleton!(str, yzuStrTypeGet, "`!yz.str`");
singleton!(var, yzuVarTypeGet, "`!yzl.var`, the unification variable");
singleton!(
    query,
    yzuQueryTypeGet,
    "`!yzl.query`, a relation before its schema is known"
);
