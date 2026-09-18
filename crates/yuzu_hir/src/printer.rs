use yuzu_core::adt::StringInterner;

use crate::{HirCtx, hir::*};

struct HirPrinter<'a> {
    hir: &'a HirCtx,
    interner: &'a StringInterner,
}

pub fn dump(hir: &HirCtx, interner: &StringInterner, root: &Root) -> String {
    let printer = HirPrinter { hir, interner };
    let mut out = String::new();
    for &stmt in root.stmts.iter() {
        printer.fmt_stmt(stmt, 0, &mut out);
    }
    out
}

fn line(out: &mut String, depth: usize, text: impl AsRef<str>) {
    for _ in 0..depth {
        out.push_str("  ");
    }
    out.push_str(text.as_ref());
    out.push('\n');
}

impl HirPrinter<'_> {
    fn text(&self, ident: &Ident) -> String {
        self.interner.text(ident.symbol).to_string()
    }

    fn fmt_stmt(&self, id: StmtId, depth: usize, out: &mut String) {
        match self.hir.stmt(id) {
            Stmt::Struct { name, fields } => {
                line(out, depth, format!("Struct {:?}", self.text(name)));
                for field in fields.iter() {
                    self.fmt_struct_field(field, depth + 1, out);
                }
            }
            Stmt::Impl {
                trait_ref,
                name,
                methods,
            } => {
                match trait_ref {
                    Some(trait_ref) => line(
                        out,
                        depth,
                        format!(
                            "Impl {:?} for {:?}",
                            self.text(&trait_ref.name),
                            self.text(name)
                        ),
                    ),
                    None => line(out, depth, format!("Impl {:?}", self.text(name))),
                }
                for &method in methods.iter() {
                    self.fmt_stmt(method, depth + 1, out);
                }
            }
            Stmt::Trait { name, methods } => {
                line(out, depth, format!("Trait {:?}", self.text(name)));
                for &method in methods.iter() {
                    self.fmt_stmt(method, depth + 1, out);
                }
            }
            Stmt::Func {
                name,
                type_params,
                params,
                type_bounds,
                body,
                ret_type_annotation,
                is_agg,
                is_external,
            } => {
                let mut kind = String::new();
                if *is_external {
                    kind.push_str("External ");
                }
                if *is_agg {
                    kind.push_str("Agg ");
                }
                kind.push_str("Func");
                line(out, depth, format!("{kind} {:?}", self.text(name)));
                for type_param in type_params.iter() {
                    line(
                        out,
                        depth + 1,
                        format!("type_param {:?}", self.text(type_param)),
                    );
                }
                for param in params.iter() {
                    self.fmt_func_param(param, depth + 1, out);
                }
                for bound in type_bounds.iter() {
                    line(
                        out,
                        depth + 1,
                        format!("bound {:?}", self.text(&bound.subject)),
                    );
                    for trait_ref in bound.traits.iter() {
                        line(
                            out,
                            depth + 2,
                            format!("trait {:?}", self.text(&trait_ref.name)),
                        );
                    }
                }
                line(out, depth + 1, "ret:");
                self.fmt_annotation(*ret_type_annotation, depth + 2, out);
                line(out, depth + 1, "body:");

                if let Some(body) = body {
                    self.fmt_stmt(*body, depth + 2, out);
                }
            }
            Stmt::Block { stmts } => {
                line(out, depth, "Block");
                for &stmt in stmts.iter() {
                    self.fmt_stmt(stmt, depth + 1, out);
                }
            }
            Stmt::Table { name, row } => {
                line(
                    out,
                    depth,
                    format!("Table {:?} row {:?}", self.text(name), self.text(row)),
                );
            }
            Stmt::InlineTable { name, fields } => {
                line(out, depth, format!("InlineTable {:?}", self.text(name)));
                for field in fields.iter() {
                    self.fmt_struct_field(field, depth + 1, out);
                }
            }
            Stmt::Let {
                name,
                mutability,
                type_annotation,
                expr,
            } => {
                line(
                    out,
                    depth,
                    format!("Let {:?} {:?}", self.text(name), mutability),
                );
                if let Some(type_annotation) = type_annotation {
                    line(out, depth + 1, "type:");
                    self.fmt_annotation(*type_annotation, depth + 2, out);
                }
                line(out, depth + 1, "expr:");
                self.fmt_expr(*expr, depth + 2, out);
            }
            Stmt::Assign { target, value } => {
                line(out, depth, "Assign");
                self.fmt_expr(*target, depth + 1, out);
                self.fmt_expr(*value, depth + 1, out);
            }
            Stmt::Return { expr } => {
                line(out, depth, "Return");
                if let Some(expr) = expr {
                    self.fmt_expr(*expr, depth + 1, out);
                }
            }
            Stmt::Expr { expr } => {
                line(out, depth, "Expr");
                self.fmt_expr(*expr, depth + 1, out);
            }
            Stmt::Missing => line(out, depth, "Missing"),
        }
    }

    fn fmt_struct_field(&self, field: &StructField, depth: usize, out: &mut String) {
        line(out, depth, format!("field {:?}:", self.text(&field.name)));
        self.fmt_annotation(field.type_annotation, depth + 1, out);
    }

    fn fmt_func_param(&self, param: &FuncParam, depth: usize, out: &mut String) {
        line(out, depth, format!("param {:?}:", self.text(&param.name)));
        self.fmt_annotation(param.type_annotation, depth + 1, out);
    }

    fn fmt_annotation(&self, id: TypeAnnotationId, depth: usize, out: &mut String) {
        match self.hir.annotation(id) {
            TypeAnnotation::Named { name, args } => {
                line(out, depth, format!("Named {:?}", self.text(name)));
                for &arg in args.iter() {
                    self.fmt_annotation(arg, depth + 1, out);
                }
            }
            TypeAnnotation::Func { params, ret } => {
                line(out, depth, "Func");
                for &param in params.iter() {
                    self.fmt_annotation(param, depth + 1, out);
                }
                line(out, depth + 1, "ret:");
                self.fmt_annotation(*ret, depth + 2, out);
            }
            TypeAnnotation::Missing => line(out, depth, "Missing"),
            TypeAnnotation::Self_ => line(out, depth, "self"),
        }
    }

    fn fmt_expr(&self, id: ExprId, depth: usize, out: &mut String) {
        match self.hir.expr(id) {
            Expr::Ident { value } => line(out, depth, format!("Ident {:?}", self.text(value))),
            Expr::Call { op, args } => {
                line(out, depth, format!("Call {:?}", op));
                for &arg in args.iter() {
                    self.fmt_expr(arg, depth + 1, out);
                }
            }
            Expr::FuncCall { callee, args } => {
                line(out, depth, "FuncCall");
                self.fmt_expr(*callee, depth + 1, out);
                for &arg in args.iter() {
                    self.fmt_expr(arg, depth + 1, out);
                }
            }
            Expr::MethodCall {
                receiver,
                method,
                args,
            } => {
                line(out, depth, format!("MethodCall {:?}", self.text(method)));
                self.fmt_expr(*receiver, depth + 1, out);
                for &arg in args.iter() {
                    self.fmt_expr(arg, depth + 1, out);
                }
            }
            Expr::FieldAccess { base, field } => {
                line(out, depth, format!("FieldAccess {:?}", self.text(field)));
                self.fmt_expr(*base, depth + 1, out);
            }
            Expr::StructInit { name, fields } => {
                line(
                    out,
                    depth,
                    format!("StructInit {:?}", self.interner.text(*name)),
                );
                for field in fields.iter() {
                    line(
                        out,
                        depth + 1,
                        format!("field {:?}:", self.text(&field.name)),
                    );
                    self.fmt_expr(field.value, depth + 2, out);
                }
            }
            Expr::ListInit { elements } => {
                line(out, depth, "ListInit");
                for &element in elements.iter() {
                    self.fmt_expr(element, depth + 1, out);
                }
            }
            Expr::Literal(literal) => {
                line(out, depth, format!("Literal {}", self.fmt_literal(literal)))
            }
            Expr::Rel(rel) => {
                line(out, depth, "Rel");
                self.fmt_rel(*rel, depth + 1, out);
            }
            Expr::Missing => line(out, depth, "Missing"),
        }
    }

    fn fmt_literal(&self, literal: &Literal) -> String {
        match literal {
            Literal::Bool { value } => format!("Bool {value}"),
            Literal::Int { value } => format!("Int {value}"),
            Literal::Float { value } => format!("Float {value}"),
            Literal::String { value } => format!("String {:?}", self.interner.text(*value)),
            Literal::Missing => "Missing".to_string(),
        }
    }

    fn fmt_rel(&self, id: RelId, depth: usize, out: &mut String) {
        match self.hir.rel(id) {
            Rel::From { relation, alias } => match alias {
                Some(alias) => line(
                    out,
                    depth,
                    format!("From {:?} as {:?}", self.text(relation), self.text(alias)),
                ),
                None => line(out, depth, format!("From {:?}", self.text(relation))),
            },
            Rel::Join {
                left,
                right,
                kind,
                condition,
            } => {
                line(out, depth, format!("Join {}", kind.keyword()));
                self.fmt_rel(*left, depth + 1, out);
                self.fmt_rel(*right, depth + 1, out);
                match condition {
                    JoinCondition::On(expr) => {
                        line(out, depth + 1, "on:");
                        self.fmt_expr(*expr, depth + 2, out);
                    }
                    JoinCondition::Using(columns) => {
                        for column in columns.iter() {
                            line(out, depth + 1, format!("using {:?}", self.text(column)));
                        }
                    }
                }
            }
            Rel::Select { input, items } => {
                line(out, depth, "Select");
                self.fmt_rel(*input, depth + 1, out);
                for item in items.iter() {
                    self.fmt_select_item(item, depth + 1, out);
                }
            }
            Rel::Where { input, predicate } => {
                line(out, depth, "Where");
                self.fmt_rel(*input, depth + 1, out);
                self.fmt_expr(*predicate, depth + 1, out);
            }
            Rel::Distinct { input } => {
                line(out, depth, "Distinct");
                self.fmt_rel(*input, depth + 1, out);
            }
            Rel::Drop { input, items } => {
                line(out, depth, "Drop");
                self.fmt_rel(*input, depth + 1, out);
                for column in items.iter() {
                    line(out, depth + 1, format!("column {:?}", self.text(column)));
                }
            }
            Rel::Rename { input, items } => {
                line(out, depth, "Rename");
                self.fmt_rel(*input, depth + 1, out);
                for item in items.iter() {
                    line(
                        out,
                        depth + 1,
                        match &item.qualifier {
                            Some(alias) => format!(
                                "rename {:?}.{:?} -> {:?}",
                                self.text(alias),
                                self.text(&item.from),
                                self.text(&item.to)
                            ),
                            None => format!(
                                "rename {:?} -> {:?}",
                                self.text(&item.from),
                                self.text(&item.to)
                            ),
                        },
                    );
                }
            }
            Rel::Extend { input, items } => {
                line(out, depth, "Extend");
                self.fmt_rel(*input, depth + 1, out);
                for item in items.iter() {
                    self.fmt_select_item(item, depth + 1, out);
                }
            }
            Rel::Set { input, items } => {
                line(out, depth, "Set");
                self.fmt_rel(*input, depth + 1, out);
                for item in items.iter() {
                    line(
                        out,
                        depth + 1,
                        format!("set {:?}:", self.text(&item.column)),
                    );
                    self.fmt_expr(item.value, depth + 2, out);
                }
            }
            Rel::Limit {
                input,
                count,
                offset,
            } => {
                line(out, depth, "Limit");
                self.fmt_rel(*input, depth + 1, out);
                line(out, depth + 1, "count:");
                self.fmt_expr(*count, depth + 2, out);
                if let Some(offset) = offset {
                    line(out, depth + 1, "offset:");
                    self.fmt_expr(*offset, depth + 2, out);
                }
            }
            Rel::Alias { input, alias } => {
                line(out, depth, format!("Alias {:?}", self.text(alias)));
                self.fmt_rel(*input, depth + 1, out);
            }
            Rel::Aggregate {
                input,
                items,
                groups,
            } => {
                line(out, depth, "Aggregate");
                self.fmt_rel(*input, depth + 1, out);
                for item in items.iter() {
                    self.fmt_aggregate_item(item, depth + 1, out);
                }
                for key in groups.iter() {
                    let qualifier = key
                        .qualifier
                        .as_ref()
                        .map(|qualifier| format!("{:?}.", self.text(qualifier)))
                        .unwrap_or_default();
                    let alias = key
                        .alias
                        .as_ref()
                        .map(|alias| format!(" as {:?}", self.text(alias)))
                        .unwrap_or_default();
                    line(
                        out,
                        depth + 1,
                        format!("group by {qualifier}{:?}{alias}", self.text(&key.column)),
                    );
                }
            }
            Rel::Missing => line(out, depth, "Missing"),
        }
    }

    fn fmt_select_item(&self, item: &SelectItem, depth: usize, out: &mut String) {
        match &item.alias {
            Some(alias) => line(out, depth, format!("item as {:?}:", self.text(alias))),
            None => line(out, depth, "item:"),
        }
        self.fmt_expr(item.expr, depth + 1, out);
    }

    fn fmt_aggregate_item(&self, item: &AggregateItem, depth: usize, out: &mut String) {
        match &item.alias {
            Some(alias) => line(out, depth, format!("item as {:?}:", self.text(alias))),
            None => line(out, depth, "item:"),
        }
        self.fmt_expr(item.expr, depth + 1, out);
    }
}
