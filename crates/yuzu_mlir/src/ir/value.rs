//! Value identity, for maps a pass keys by value, and the op that
//! defines a value.

use melior::ir::operation::OperationResult;
use melior::ir::{Value, ValueLike};

/// The identity of a value while the op that defines it is alive.
///
/// A value is not uniqued, so a new op can reuse the memory of an erased
/// one. A map keyed by `ValueId` must not outlive an erasure in the IR it
/// reads.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ValueId(usize);

pub trait ValueExt<'c>: ValueLike<'c> {
    fn id(&self) -> ValueId {
        ValueId(self.to_raw().ptr as usize)
    }

    /// Whether exactly one operand uses the value.
    fn has_one_use(&self) -> bool {
        // SAFETY: the value is live while `self` borrows it, and its use list
        // is read, not changed.
        unsafe {
            let first = mlir_sys::mlirValueGetFirstUse(self.to_raw());
            !mlir_sys::mlirOpOperandIsNull(first)
                && mlir_sys::mlirOpOperandIsNull(mlir_sys::mlirOpOperandGetNextUse(first))
        }
    }
}

impl<'c, T: ValueLike<'c>> ValueExt<'c> for T {}

/// The value as an op's result, or `None` for a block argument.
///
/// melior's `OperationResult::try_from` prints the whole value into its
/// error, so a failed conversion costs a full print even when the caller
/// drops the error. This checks the kind first.
#[must_use]
pub fn op_result<'c, 'a>(value: Value<'c, 'a>) -> Option<OperationResult<'c, 'a>> {
    value
        .is_operation_result()
        .then(|| OperationResult::try_from(value).ok())
        .flatten()
}

#[cfg(test)]
mod tests {
    use melior::ir::operation::OperationLike;
    use melior::ir::{BlockLike, Module, RegionLike};

    use crate::ir::block::BlockExt;

    use super::op_result;

    #[test]
    fn a_block_argument_is_no_op_result() {
        let context = crate::context();
        let module = Module::parse(
            &context,
            r#"
module {
  yz.struct @row ["a"] : [!yz.int64]
  %t = yzr.table @t : !yz.struct<@row>
  %p = yzr.project %t {
  ^bb0(%a: !yz.int64):
    %one = yz.constant_int 1
    yzr.yield %one : !yz.int64
  } : !yz.struct<@row> -> !yz.struct<@row>
}
"#,
        )
        .expect("the module parses");

        let project = module
            .body()
            .operations()
            .nth(2)
            .expect("the module holds a project");
        let block = project
            .region(0)
            .ok()
            .and_then(|region| region.first_block())
            .expect("the project has a block");
        let argument = block.argument(0).expect("the block takes the row's column");
        let constant = block.first_operation().expect("the block holds a constant");

        assert!(op_result(argument.into()).is_none());
        assert!(op_result(constant.result(0).expect("a constant has a result").into()).is_some());
    }
}
