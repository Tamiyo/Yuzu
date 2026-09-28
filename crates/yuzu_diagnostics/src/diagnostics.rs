use text_size::TextRange;

use crate::source_map::SourceId;

pub mod builder;
pub mod engine;
pub mod printer;

#[derive(Debug)]
pub struct Diagnostic {
    pub severity: Severity,
    pub code: String,
    pub message: String,
    pub labels: Vec<Label>,
    pub notes: Vec<String>,
}

#[derive(Debug, PartialEq)]
pub enum Severity {
    Error,
    Warning,
    Remark,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Span {
    pub source_id: SourceId,
    pub range: TextRange,
}

#[derive(Debug)]
pub struct Label {
    pub style: LabelStyle,
    pub span: Span,
    pub message: String,
}

#[derive(Debug)]
pub enum LabelStyle {
    Primary,
    Secondary,
}
