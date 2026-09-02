use yuzu_core::adt::{StringInterner, SymbolId};

use crate::AnfCtx;
use yuzu_types::AggFunc;

use crate::{
    AggregateItem, Atom, AtomId, Const, Expr, ExprId, JoinCondition, Op, Rel, RelId, Root,
    SelectItem, Stmt, StmtId, Thunk,
};

struct AnfPrinter<'a> {
    anf: &'a AnfCtx,
    interner: &'a StringInterner,
}

pub fn dump(anf: &AnfCtx, interner: &StringInterner, root: &Root) -> String {
    let printer = AnfPrinter { anf, interner };
    let mut out = String::new();
    for &stmt in root.stmts.iter() {
        printer.fmt_stmt(stmt, 0, &mut out);
    }
    out
}

fn indent(out: &mut String, depth: usize) {
    for _ in 0..depth {
        out.push_str("  ");
    }
}

impl AnfPrinter<'_> {
    fn text(&self, name: SymbolId) -> &str {
        self.interner.text(name)
    }

    fn binding_name(&self, binding: crate::BindingId) -> &str {
        self.text(self.anf.binding(binding).name.name)
    }

    fn fmt_stmt(&self, id: StmtId, depth: usize, out: &mut String) {
        indent(out, depth);
        match self.anf.stmt(id) {
            Stmt::Let { binding, expr } => {
                out.push_str("let ");
                out.push_str(self.binding_name(*binding));
                out.push_str(" = ");
                self.fmt_expr(*expr, depth, out);
                out.push('\n');
            }
            Stmt::Return { value } => {
                out.push_str("return");
                if let Some(atom) = value {
                    out.push(' ');
                    self.fmt_atom(*atom, out);
                }
                out.push('\n');
            }
            Stmt::Expr { value } => {
                self.fmt_expr(*value, depth, out);
                out.push('\n');
            }
            Stmt::Block { stmts } => {
                out.push_str("{\n");
                for &stmt in stmts.iter() {
                    self.fmt_stmt(stmt, depth + 1, out);
                }
                indent(out, depth);
                out.push_str("}\n");
            }
            Stmt::Struct { name, fields } => {
                out.push_str("struct ");
                out.push_str(self.text(name.name));
                out.push_str(" { ");
                for (i, field) in fields.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(self.text(field.name.name));
                }
                out.push_str(" }\n");
            }
            Stmt::Table { name, .. } => {
                out.push_str("table ");
                out.push_str(self.text(name.name));
                out.push('\n');
            }
            Stmt::Func {
                name, params, body, ..
            } => {
                out.push_str("fn ");
                out.push_str(self.text(name.name));
                out.push('(');
                for (i, &param) in params.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(self.binding_name(param));
                }
                out.push(')');
                match body {
                    Some(body) => {
                        out.push(' ');
                        self.fmt_stmt(*body, depth, out);
                    }
                    None => out.push('\n'),
                }
            }
            Stmt::Assign { target, value } => {
                self.fmt_atom(*target, out);
                out.push_str(" = ");
                self.fmt_expr(*value, depth, out);
                out.push('\n');
            }
        }
    }

    fn fmt_expr(&self, id: ExprId, depth: usize, out: &mut String) {
        match self.anf.expr(id) {
            Expr::Call { op, args, .. } => {
                out.push_str(op_name(*op));
                self.fmt_args(args, out);
            }
            Expr::AggCall { func, args, .. } => {
                match func {
                    AggFunc::External(symbol) => out.push_str(self.text(*symbol)),
                    _ => out.push_str(func.name()),
                }
                self.fmt_args(args, out);
            }
            Expr::ExternCall { name, args, .. } => {
                out.push_str(self.text(name.name));
                self.fmt_args(args, out);
            }
            Expr::FuncCall { callee, args, .. } => {
                self.fmt_atom(*callee, out);
                self.fmt_args(args, out);
            }
            Expr::MethodCall {
                receiver,
                method,
                args,
                ..
            } => {
                self.fmt_atom(*receiver, out);
                out.push('.');
                out.push_str(self.text(method.name));
                self.fmt_args(args, out);
            }
            Expr::StructInit { name, fields, .. } => {
                out.push_str(self.text(*name));
                out.push_str(" { ");
                for (i, field) in fields.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(self.text(field.name.name));
                    out.push_str(": ");
                    self.fmt_atom(field.value, out);
                }
                out.push_str(" }");
            }
            Expr::ListInit { elements, .. } => {
                out.push('[');
                self.fmt_atoms(elements, out);
                out.push(']');
            }
            Expr::Rel(rel) => self.fmt_rel(*rel, depth, out),
            Expr::Atom { value } => self.fmt_atom(*value, out),
        }
    }

    /// Prints a pipeline one stage per line: the `from` source, then each `|>`
    /// stage on its own line indented under it.
    fn fmt_rel(&self, id: RelId, depth: usize, out: &mut String) {
        match self.anf.rel(id) {
            Rel::From {
                relation, alias, ..
            } => {
                out.push_str("from ");
                out.push_str(self.text(relation.name));
                if let Some(alias) = alias {
                    out.push_str(" as ");
                    out.push_str(self.text(alias.name));
                }
            }
            Rel::Join {
                left,
                right,
                kind,
                condition,
                ..
            } => {
                self.fmt_rel(*left, depth, out);
                self.fmt_stage(depth, kind.keyword(), out);
                out.push_str(" join ");
                self.fmt_rel(*right, depth, out);
                match condition {
                    JoinCondition::On(thunk) => {
                        out.push_str(" on ");
                        self.fmt_thunk(thunk, depth, out);
                    }
                    JoinCondition::Using(columns) => {
                        out.push_str(" using ");
                        for (i, column) in columns.iter().enumerate() {
                            if i > 0 {
                                out.push_str(", ");
                            }
                            out.push_str(self.text(column.name));
                        }
                    }
                }
            }
            Rel::Select { input, items, .. } => {
                self.fmt_rel(*input, depth, out);
                self.fmt_stage(depth, "select ", out);
                self.fmt_select_items(items, depth, out);
            }
            Rel::Where {
                input, predicate, ..
            } => {
                self.fmt_rel(*input, depth, out);
                self.fmt_stage(depth, "where ", out);
                self.fmt_thunk(predicate, depth, out);
            }
            Rel::Distinct { input, .. } => {
                self.fmt_rel(*input, depth, out);
                self.fmt_stage(depth, "distinct", out);
            }
            Rel::Drop { input, columns, .. } => {
                self.fmt_rel(*input, depth, out);
                self.fmt_stage(depth, "drop ", out);
                for (i, column) in columns.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(self.text(column.name));
                }
            }
            Rel::Rename { input, items, .. } => {
                self.fmt_rel(*input, depth, out);
                self.fmt_stage(depth, "rename ", out);
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(self.text(item.from.name));
                    out.push_str(" as ");
                    out.push_str(self.text(item.to.name));
                }
            }
            Rel::Set { input, items, .. } => {
                self.fmt_rel(*input, depth, out);
                self.fmt_stage(depth, "set ", out);
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(self.text(item.column.name));
                    out.push_str(" = ");
                    self.fmt_thunk(&item.value, depth, out);
                }
            }
            Rel::Limit {
                input,
                count,
                offset,
                ..
            } => {
                self.fmt_rel(*input, depth, out);
                self.fmt_stage(depth, "limit ", out);
                self.fmt_thunk(count, depth, out);
                if let Some(offset) = offset {
                    out.push_str(" offset ");
                    self.fmt_thunk(offset, depth, out);
                }
            }
            Rel::Alias { input, alias, .. } => {
                self.fmt_rel(*input, depth, out);
                self.fmt_stage(depth, "as ", out);
                out.push_str(self.text(alias.name));
            }
            Rel::Extend { input, items, .. } => {
                self.fmt_rel(*input, depth, out);
                self.fmt_stage(depth, "extend ", out);
                self.fmt_select_items(items, depth, out);
            }
            Rel::Aggregate {
                input,
                items,
                groups,
                ..
            } => {
                self.fmt_rel(*input, depth, out);
                self.fmt_stage(depth, "aggregate ", out);
                self.fmt_aggregate_items(items, depth, out);
                for (i, key) in groups.iter().enumerate() {
                    out.push_str(if i == 0 { " group by " } else { ", " });
                    out.push_str(self.text(key.name.name));
                }
            }
        }
    }

    fn fmt_stage(&self, depth: usize, keyword: &str, out: &mut String) {
        out.push('\n');
        indent(out, depth + 1);
        out.push_str("|> ");
        out.push_str(keyword);
    }

    fn fmt_select_items(&self, items: &[SelectItem], depth: usize, out: &mut String) {
        for (i, item) in items.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            self.fmt_thunk(&item.body, depth, out);
            if let Some(alias) = item.alias {
                out.push_str(" as ");
                out.push_str(self.text(alias.name));
            }
        }
    }

    fn fmt_aggregate_items(&self, items: &[AggregateItem], depth: usize, out: &mut String) {
        for (i, item) in items.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            self.fmt_thunk(&item.body, depth, out);
            if let Some(alias) = item.alias {
                out.push_str(" as ");
                out.push_str(self.text(alias.name));
            }
        }
    }

    /// A single-computation thunk prints as just its expression; otherwise its
    /// temporaries lead its value (`%t0 = add(a, b); %t0`), or just `a` for a
    /// bare column.
    fn fmt_thunk(&self, thunk: &Thunk, depth: usize, out: &mut String) {
        if let [stmt] = &*thunk.stmts
            && let (Stmt::Let { binding, expr }, Atom::Var { binding: value }) =
                (self.anf.stmt(*stmt), self.anf.atom(thunk.value))
            && binding == value
        {
            self.fmt_expr(*expr, depth, out);
            return;
        }
        for &stmt in thunk.stmts.iter() {
            if let Stmt::Let { binding, expr } = self.anf.stmt(stmt) {
                out.push_str(self.binding_name(*binding));
                out.push_str(" = ");
                self.fmt_expr(*expr, depth, out);
                out.push_str("; ");
            }
        }
        self.fmt_atom(thunk.value, out);
    }

    fn fmt_args(&self, args: &[AtomId], out: &mut String) {
        out.push('(');
        self.fmt_atoms(args, out);
        out.push(')');
    }

    fn fmt_atoms(&self, atoms: &[AtomId], out: &mut String) {
        for (i, &atom) in atoms.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            self.fmt_atom(atom, out);
        }
    }

    fn fmt_atom(&self, id: AtomId, out: &mut String) {
        match self.anf.atom(id) {
            crate::Atom::Var { binding } => out.push_str(self.binding_name(*binding)),
            crate::Atom::FuncRef { binding, .. } => out.push_str(self.binding_name(*binding)),
            crate::Atom::Field { base, field, .. } => {
                self.fmt_atom(*base, out);
                out.push('.');
                out.push_str(self.text(field.name));
            }
            crate::Atom::Column { name, .. } => out.push_str(self.text(name.name)),
            crate::Atom::Const(constant) => self.fmt_const(constant, out),
        }
    }

    fn fmt_const(&self, constant: &Const, out: &mut String) {
        match constant {
            Const::Int { value } => out.push_str(&value.to_string()),
            Const::Float { value } => out.push_str(&value.to_string()),
            Const::Bool { value } => out.push_str(&value.to_string()),
            Const::String { value } => {
                out.push_str(&format!("{:?}", self.text(*value)));
            }
        }
    }
}

fn op_name(op: Op) -> &'static str {
    match op {
        Op::Add => "add",
        Op::Sub => "sub",
        Op::Mul => "mul",
        Op::Div => "div",
        Op::Pow => "pow",
        Op::And => "and",
        Op::Or => "or",
        Op::In => "in",
        Op::NotIn => "not_in",
        Op::Eq => "eq",
        Op::Neq => "neq",
        Op::Lt => "lt",
        Op::Lte => "lte",
        Op::Gt => "gt",
        Op::Gte => "gte",
        Op::ShiftLeft => "shl",
        Op::ShiftRight => "shr",
        Op::UnaryPos => "pos",
        Op::UnaryNeg => "neg",
        Op::UnaryNot => "not",
    }
}
