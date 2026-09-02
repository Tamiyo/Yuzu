use yuzu_core::adt::StringInterner;

use crate::graph::RelGraph;
use crate::{Const, Expr, ExprId, Func, JoinCondition, JoinKind, Measure, Rel, RelId, SelectItem};

pub fn dump(graph: &RelGraph, interner: &StringInterner) -> String {
    let printer = PlanPrinter { graph, interner };
    let mut out = String::new();
    if let Some(root) = graph.root() {
        printer.fmt_rel(root, 0, &mut out);
    }
    out
}

struct PlanPrinter<'p> {
    graph: &'p RelGraph,
    interner: &'p StringInterner,
}

impl PlanPrinter<'_> {
    fn fmt_rel(&self, id: RelId, depth: usize, out: &mut String) {
        out.push_str(&"  ".repeat(depth));
        match self.graph.plan().rel(id) {
            Rel::From {
                relation, alias, ..
            } => {
                out.push_str("from ");
                out.push_str(self.text(*relation));
                if let Some(alias) = alias {
                    out.push_str(" as ");
                    out.push_str(self.text(*alias));
                }
            }
            Rel::Join {
                kind, condition, ..
            } => {
                out.push_str("join ");
                out.push_str(kind_name(*kind));
                match condition {
                    Some(JoinCondition::On(predicate)) => {
                        out.push_str(" on ");
                        self.fmt_expr(*predicate, out);
                    }
                    Some(JoinCondition::Using(keys)) => {
                        out.push_str(" using (");
                        for (index, key) in keys.iter().enumerate() {
                            if index > 0 {
                                out.push_str(", ");
                            }
                            out.push_str(&format!("#{} = #{}", key.left, key.right));
                        }
                        out.push(')');
                    }
                    None => {}
                }
            }
            Rel::Select { items, .. } => {
                out.push_str("select ");
                self.fmt_items(items, out);
            }
            Rel::Where { predicate, .. } => {
                out.push_str("where ");
                self.fmt_expr(*predicate, out);
            }
            Rel::Distinct { .. } => out.push_str("distinct"),
            Rel::Drop { columns, .. } => {
                out.push_str("drop [");
                for (index, column) in columns.iter().enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(&format!("#{column}"));
                }
                out.push(']');
            }
            Rel::Rename { items, .. } => {
                out.push_str("rename [");
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(&format!("#{} as {}", item.column, self.text(item.to)));
                }
                out.push(']');
            }
            Rel::Extend { items, .. } => {
                out.push_str("extend ");
                self.fmt_items(items, out);
            }
            Rel::Set { items, .. } => {
                out.push_str("set [");
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(&format!("#{} = ", item.column));
                    self.fmt_expr(item.value, out);
                }
                out.push(']');
            }
            Rel::Limit { count, offset, .. } => {
                out.push_str(&format!("limit {count}"));
                if let Some(offset) = offset {
                    out.push_str(&format!(" offset {offset}"));
                }
            }
            Rel::Alias { alias, .. } => {
                out.push_str("as ");
                out.push_str(self.text(*alias));
            }
            Rel::Aggregate {
                groupings,
                measures,
                ..
            } => {
                out.push_str("aggregate [");
                for (index, measure) in measures.iter().enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    self.fmt_measure(measure, out);
                }
                out.push(']');
                if !groupings.is_empty() {
                    out.push_str(" group [");
                    for (index, key) in groupings.iter().enumerate() {
                        if index > 0 {
                            out.push_str(", ");
                        }
                        out.push_str(&format!("#{key}"));
                    }
                    out.push(']');
                }
            }
        }
        out.push('\n');
        for input in self.graph.inputs(id) {
            self.fmt_rel(input, depth + 1, out);
        }
    }

    fn fmt_measure(&self, measure: &Measure, out: &mut String) {
        match measure.func {
            yuzu_types::AggFunc::External(symbol) => out.push_str(self.text(symbol)),
            _ => out.push_str(measure.func.name()),
        }
        out.push('(');
        for (index, &arg) in measure.args.iter().enumerate() {
            if index > 0 {
                out.push_str(", ");
            }
            self.fmt_expr(arg, out);
        }
        out.push(')');
    }

    fn fmt_items(&self, items: &[SelectItem], out: &mut String) {
        out.push('[');
        for (index, item) in items.iter().enumerate() {
            if index > 0 {
                out.push_str(", ");
            }
            self.fmt_expr(item.body, out);
            if let Some(alias) = item.alias {
                out.push_str(" as ");
                out.push_str(self.text(alias));
            }
        }
        out.push(']');
    }

    fn fmt_expr(&self, id: ExprId, out: &mut String) {
        match self.graph.plan().expr(id) {
            Expr::Column { column, .. } => out.push_str(&format!("#{column}")),
            Expr::Literal { value, .. } => match value {
                Const::Int { value } => out.push_str(&value.to_string()),
                Const::Float { value } => out.push_str(&value.to_string()),
                Const::Bool { value } => out.push_str(&value.to_string()),
                Const::String { value } => out.push_str(&format!("{:?}", self.text(*value))),
            },
            Expr::Call { func, args, .. } => {
                match func {
                    Func::External(symbol) => out.push_str(self.text(*symbol)),
                    _ => out.push_str(func_name(*func)),
                }
                out.push('(');
                for (index, &arg) in args.iter().enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    self.fmt_expr(arg, out);
                }
                out.push(')');
            }
        }
    }

    fn text(&self, symbol: yuzu_core::adt::SymbolId) -> &str {
        self.interner.text(symbol)
    }
}

fn kind_name(kind: JoinKind) -> &'static str {
    match kind {
        JoinKind::Inner => "inner",
        JoinKind::Left => "left",
        JoinKind::Right => "right",
        JoinKind::Full => "full",
        JoinKind::Cross => "cross",
    }
}

fn func_name(func: Func) -> &'static str {
    match func {
        Func::Add => "add",
        Func::Subtract => "subtract",
        Func::Multiply => "multiply",
        Func::Divide => "divide",
        Func::Power => "power",
        Func::Negate => "negate",
        Func::ShiftLeft => "shift_left",
        Func::ShiftRight => "shift_right",
        Func::Equal => "equal",
        Func::NotEqual => "not_equal",
        Func::Less => "less",
        Func::LessEqual => "less_equal",
        Func::Greater => "greater",
        Func::GreaterEqual => "greater_equal",
        Func::And => "and",
        Func::Or => "or",
        Func::Not => "not",
        Func::In => "in",
        Func::External(_) => unreachable!("an external's name lives in the interner"),
    }
}
