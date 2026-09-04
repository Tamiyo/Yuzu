use yuzu_types::{Type, TypeId};

use crate::{ExprId, infer::InferCtx};

impl InferCtx<'_> {
    pub(crate) fn coerce(&mut self, expr: ExprId, source: TypeId, target: TypeId) -> bool {
        let source = self.resolve(source);
        let target = self.resolve(target);
        if source == target {
            return true;
        }

        let source_ty: &Type = self.types.ty(source);
        let target_ty = self.types.ty(target);
        if !source_ty.is_numeric() || !target_ty.is_numeric() {
            return false;
        }

        let (Some(source_rank), Some(target_rank)) =
            (numeric_rank(source_ty), numeric_rank(target_ty))
        else {
            return false;
        };

        if source_rank > target_rank {
            return false;
        }

        if !target_ty.is_float() && source_ty.is_unsigned() != target_ty.is_unsigned() {
            return false;
        }

        self.record_adjustment(expr, target);
        true
    }

    fn record_adjustment(&mut self, expr: ExprId, target: TypeId) {
        self.result.adjustments.set(expr.index(), target);
    }
}

fn numeric_rank(ty: &Type) -> Option<u8> {
    let rank = match ty {
        Type::Int8 | Type::UInt8 => 1,
        Type::Int16 | Type::UInt16 => 2,
        Type::Int32 | Type::UInt32 => 3,
        Type::Int64 | Type::UInt64 => 4,
        Type::Float32 => 5,
        Type::Float64 => 6,
        _ => return None,
    };
    Some(rank)
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};

    use crate::infer::test_support::check_src;

    fn coerces(from: &str, to: &str, expected: Expect) {
        check_src(&format!("let a: {from} = 0\nlet b: {to} = a"), expected);
    }

    #[test]
    fn integer_widening_coerces() {
        coerces("int32", "int64", expect![""]);
    }

    #[test]
    fn float_widening_coerces() {
        coerces("float32", "float64", expect![""]);
    }

    #[test]
    fn integer_widens_into_float() {
        coerces("int32", "float64", expect![""]);
    }

    #[test]
    fn narrowing_is_rejected() {
        coerces(
            "int64",
            "int32",
            expect!["value of type `Int64` is not assignable to `Int32`"],
        );
    }

    #[test]
    fn signedness_mismatch_is_rejected() {
        coerces(
            "int32",
            "uint32",
            expect!["value of type `Int32` is not assignable to `UInt32`"],
        );
    }

    #[test]
    fn unsigned_widening_coerces() {
        coerces("uint8", "uint16", expect![""]);
    }

    #[test]
    fn unsigned_widens_into_float() {
        coerces("uint32", "float64", expect![""]);
    }

    #[test]
    fn float_narrowing_is_rejected() {
        coerces(
            "float64",
            "float32",
            expect!["value of type `Float64` is not assignable to `Float32`"],
        );
    }

    #[test]
    fn same_type_coerces() {
        coerces("int64", "int64", expect![""]);
    }
}
