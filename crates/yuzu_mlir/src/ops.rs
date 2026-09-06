//! Op classes for matching during walks. melior's generated dialect enums
//! (`YzlDialectOperation` and friends) exist but consume owned operations;
//! a pass walks borrowed refs, so classification happens here instead.

use melior::ir::operation::OperationLike;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Yzl {
    Struct,
    Table,
    Fn,
    Let,
    Return,
    From,
    Where,
    Select,
    Extend,
    Aggregate,
    Limit,
    Rename,
    Alias,
    Join,
    Set,
    Distinct,
    Drop,
    Output,
    Name,
    Call,
    List,
    Yield,
}

impl Yzl {
    pub fn of<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> Option<Self> {
        Some(match op.name().as_string_ref().as_str().ok()? {
            "yzl.struct" => Self::Struct,
            "yzl.table" => Self::Table,
            "yzl.fn" => Self::Fn,
            "yzl.let" => Self::Let,
            "yzl.return" => Self::Return,
            "yzl.from" => Self::From,
            "yzl.where" => Self::Where,
            "yzl.select" => Self::Select,
            "yzl.extend" => Self::Extend,
            "yzl.aggregate" => Self::Aggregate,
            "yzl.limit" => Self::Limit,
            "yzl.rename" => Self::Rename,
            "yzl.alias" => Self::Alias,
            "yzl.join" => Self::Join,
            "yzl.set" => Self::Set,
            "yzl.distinct" => Self::Distinct,
            "yzl.drop" => Self::Drop,
            "yzl.output" => Self::Output,
            "yzl.name" => Self::Name,
            "yzl.call" => Self::Call,
            "yzl.list" => Self::List,
            "yzl.yield" => Self::Yield,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Yz {
    ConstantInt,
    ConstantFloat,
    ConstantBool,
    ConstantStr,
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Neg,
    Cmp,
    And,
    Or,
    Not,
    Func,
    Call,
    ExternCall,
    Return,
}

impl Yz {
    pub fn of<'c: 'a, 'a>(op: &impl OperationLike<'c, 'a>) -> Option<Self> {
        Some(match op.name().as_string_ref().as_str().ok()? {
            "yz.constant_int" => Self::ConstantInt,
            "yz.constant_float" => Self::ConstantFloat,
            "yz.constant_bool" => Self::ConstantBool,
            "yz.constant_str" => Self::ConstantStr,
            "yz.add" => Self::Add,
            "yz.sub" => Self::Sub,
            "yz.mul" => Self::Mul,
            "yz.div" => Self::Div,
            "yz.rem" => Self::Rem,
            "yz.neg" => Self::Neg,
            "yz.cmp" => Self::Cmp,
            "yz.and" => Self::And,
            "yz.or" => Self::Or,
            "yz.not" => Self::Not,
            "yz.func" => Self::Func,
            "yz.call" => Self::Call,
            "yz.extern_call" => Self::ExternCall,
            "yz.return" => Self::Return,
            _ => return None,
        })
    }
}
