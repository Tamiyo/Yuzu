//! The dialects' rewrite patterns, and the greedy driver that applies them
//! together with the ops' folders.
//!
//! A pattern is a safe function of the op it matches, the rewriter and the
//! context. The folders stay on the ops in C++, where MLIR looks for them.

use melior::ir::attribute::{BoolAttribute, FloatAttribute, IntegerAttribute, StringAttribute};
use melior::ir::operation::{OperationLike, OperationRef};
use melior::ir::{Attribute, Module, Value, ValueLike};
use melior::{
    Context, Error, GreedyRewriteDriverConfig, PatternRewriter, RewritePattern, RewritePatternSet,
    RewriterBase, apply_patterns_and_fold_greedily, create_op_rewrite_pattern,
};

use crate::ir::attribute::array::ArrayAttributeExt;
use crate::ir::attribute::integer::IntegerAttributeExt;
use crate::ir::operation::OperationCast;
use crate::ir::value::op_result;
use crate::ods::yz;
use crate::ops::yz::YzOp;

/// Applies the patterns and the folders until nothing changes, the way
/// MLIR's canonicalizer does.
///
/// # Errors
///
/// Returns an error when the rewrite does not settle within the driver's
/// iteration limit.
pub fn canonicalize(context: &Context, module: &Module) -> Result<(), Error> {
    let patterns = RewritePatternSet::new(context);
    patterns.add(pattern(context, "yz.add", reassociate_add));
    patterns.add(pattern(context, "yz.in", fold_membership));

    let config = GreedyRewriteDriverConfig::new();
    config.set_use_top_down_traversal(true);
    apply_patterns_and_fold_greedily(module, patterns.freeze(), &config)
}

/// What a pattern does to one op it matched; `true` when it changed the IR.
type Rewrite = for<'c, 'a> fn(&'c Context, OperationRef<'c, 'a>, RewriterBase<'c, 'a>) -> bool;

fn pattern(context: &Context, root: &str, rewrite: Rewrite) -> RewritePattern {
    create_op_rewrite_pattern(
        root,
        1,
        context,
        move |_, op, rewriter| {
            // SAFETY: the driver calls a pattern with a live op and the live
            // rewriter of the rewrite in progress, for this call only. The
            // context owns both and outlives the rewrite.
            unsafe {
                let rewriter = PatternRewriter::from_raw(rewriter);
                let base = rewriter.as_rewriter_base();
                let context = base.context().to_ref();
                rewrite(context, OperationRef::from_raw(op), base)
            }
        },
        &[],
    )
}

/// `(x + c1) + c2` becomes `x + (c1 + c2)`, which lets one constant reach
/// another through the value between them.
///
/// The two forms compute the same only while no intermediate overflows
/// differently, and the dialect leaves overflow to the engine. Two constants
/// of the same sign are safe: the intermediate sum lies between `x` and the
/// final one, so it overflows only when the final one does. Mixed signs are
/// not, since `x + 1 - 1` can overflow at the first step and not at all when
/// reassociated.
fn reassociate_add<'c>(
    context: &'c Context,
    op: OperationRef<'c, '_>,
    rewriter: RewriterBase<'c, '_>,
) -> bool {
    let Some(YzOp::Add(add)) = op.as_yz() else {
        return false;
    };
    let Some((outer, ty)) = constant_int(add.rhs()) else {
        return false;
    };
    let Some(inner) = op_result(add.lhs()).map(|result| result.owner()) else {
        return false;
    };
    let Some(YzOp::Add(inner)) = inner.as_yz() else {
        return false;
    };
    let Some((held, _)) = constant_int(inner.rhs()) else {
        return false;
    };
    if (held < 0) != (outer < 0) {
        return false;
    }
    let Some(total) = held.checked_add(outer) else {
        return false;
    };

    let location = op.location();
    rewriter.set_insertion_point_before(op);
    let folded = rewriter.insert(
        yz::constant_int(
            context,
            ty,
            IntegerAttribute::from_i64(context, total),
            location,
        )
        .into(),
    );
    let sum = rewriter.insert(
        yz::add(
            context,
            add.result().r#type(),
            inner.lhs(),
            folded.result(0).expect("a constant has a result").into(),
            location,
        )
        .into(),
    );
    rewriter.replace_op_with_operation(op, sum);
    true
}

/// `x in [a, b]` with a constant `x` is decided once an element equals it,
/// or once every element is a constant that does not.
fn fold_membership<'c>(
    context: &'c Context,
    op: OperationRef<'c, '_>,
    rewriter: RewriterBase<'c, '_>,
) -> bool {
    let Some(YzOp::In(membership)) = op.as_yz() else {
        return false;
    };
    let Some(value) = constant(membership.value()) else {
        return false;
    };
    let Some(list) = op_result(membership.list()).map(|result| result.owner()) else {
        return false;
    };

    let found = match list.as_yz() {
        Some(YzOp::ConstantList(constants)) => constants
            .values()
            .elements()
            .any(|element| same_value(value, element) == Some(true)),
        Some(YzOp::List(elements)) => {
            let mut decided = true;
            let mut found = false;
            for element in elements.elements() {
                match constant(element).and_then(|element| same_value(value, element)) {
                    Some(true) => {
                        found = true;
                        break;
                    }
                    Some(false) => {}
                    None => decided = false,
                }
            }
            if !found && !decided {
                return false;
            }
            found
        }
        _ => return false,
    };

    rewriter.set_insertion_point_before(op);
    let decided = rewriter.insert(
        yz::constant_bool(
            context,
            membership.result().r#type(),
            BoolAttribute::new(context, found),
            op.location(),
        )
        .into(),
    );
    rewriter.replace_op_with_operation(op, decided);
    true
}

/// The attribute of the constant op a value comes from.
fn constant<'c>(value: Value<'c, '_>) -> Option<Attribute<'c>> {
    let owner = op_result(value)?.owner();
    Some(match owner.as_yz()? {
        YzOp::ConstantInt(constant) => constant.value().into(),
        YzOp::ConstantFloat(constant) => constant.value().into(),
        YzOp::ConstantBool(constant) => constant.value().into(),
        YzOp::ConstantStr(constant) => constant.value().into(),
        _ => return None,
    })
}

/// The integer a value holds when a `yz.constant_int` makes it, with its type.
fn constant_int<'c>(value: Value<'c, '_>) -> Option<(i64, melior::ir::Type<'c>)> {
    let owner = op_result(value)?.owner();
    let Some(YzOp::ConstantInt(constant)) = owner.as_yz() else {
        return None;
    };
    Some((constant.value().value(), value.r#type()))
}

/// Whether two constants are equal, the way the engine compares them: `-0.0`
/// equals `0.0`, and NaN equals nothing. `None` for two of different kinds.
fn same_value(lhs: Attribute<'_>, rhs: Attribute<'_>) -> Option<bool> {
    // A bool is an integer attribute of one bit, so it is asked first.
    if let (Ok(lhs), Ok(rhs)) = (BoolAttribute::try_from(lhs), BoolAttribute::try_from(rhs)) {
        return Some(lhs.value() == rhs.value());
    }
    if let (Ok(lhs), Ok(rhs)) = (
        IntegerAttribute::try_from(lhs),
        IntegerAttribute::try_from(rhs),
    ) {
        return Some(lhs.value() == rhs.value());
    }
    if let (Ok(lhs), Ok(rhs)) = (FloatAttribute::try_from(lhs), FloatAttribute::try_from(rhs)) {
        #[expect(clippy::float_cmp, reason = "the engine compares exactly")]
        return Some(lhs.value() == rhs.value());
    }
    if let (Ok(lhs), Ok(rhs)) = (
        StringAttribute::try_from(lhs),
        StringAttribute::try_from(rhs),
    ) {
        return Some(lhs.value() == rhs.value());
    }
    None
}
