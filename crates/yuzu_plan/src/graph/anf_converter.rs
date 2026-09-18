use std::collections::HashMap;

use yuzu_core::adt::{Signedness, StringInterner, SymbolId};
use yuzu_diagnostics::diagnostics::{Span, builder::DiagnosticBuilder, engine::DiagnosticsEngine};
use yuzu_types::{Column, Type, TypeCtx, TypeId};

use crate::graph::{RelGraph, RelGraphConverter};
use crate::{
    Const, Expr, ExprId, Func, JoinCondition, JoinKey, JoinKind, Measure, Rel, RelId, RenameItem,
    SelectItem, SetItem,
};

/// The values a thunk's temporaries stand for. A plan expression is a tree, so
/// a `%rN` is not translated where it is bound but where it is used, splicing
/// its definition into each use site; interning collapses the copies.
type Defs = HashMap<yuzu_anf::BindingId, yuzu_anf::ExprId>;

/// The reduced program contained something a plan cannot express; a diagnostic
/// has already been reported at its source position.
struct Unsupported;

pub struct AnfToRelGraphConverter<'g> {
    anf: &'g yuzu_anf::AnfCtx,
    types: &'g mut TypeCtx,
    interner: &'g StringInterner,
    source_map: &'g yuzu_anf::AnfSourceMap,
    diagnostics: &'g mut DiagnosticsEngine,
    query_stmt: yuzu_anf::StmtId,
    graph: RelGraph,
    agg: Option<AggState>,
}

/// Conversion state for one `Aggregate` node's items. An aggregate call
/// becomes a measure (deduplicated structurally) and reads back as an output
/// column past the keys; a column outside a measure's arguments sits at group
/// level, where only the keys survive, so it remaps to its key position.
struct AggState {
    keys: Vec<u32>,
    measures: Vec<Measure>,
    names: Vec<yuzu_core::adt::SymbolId>,
    current_name: yuzu_core::adt::SymbolId,
    in_args: bool,
}

impl<'g> AnfToRelGraphConverter<'g> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        anf: &'g yuzu_anf::AnfCtx,
        types: &'g mut TypeCtx,
        interner: &'g StringInterner,
        source_map: &'g yuzu_anf::AnfSourceMap,
        diagnostics: &'g mut DiagnosticsEngine,
        query_stmt: yuzu_anf::StmtId,
    ) -> Self {
        Self {
            anf,
            types,
            interner,
            source_map,
            diagnostics,
            query_stmt,
            graph: RelGraph::new(),
            agg: None,
        }
    }

    fn convert_rel(&mut self, rel: yuzu_anf::RelId) -> Result<RelId, Unsupported> {
        let (rel, inputs) = match self.anf.rel(rel).clone() {
            yuzu_anf::Rel::From {
                relation,
                alias,
                ty,
            } => (
                Rel::From {
                    relation: relation.name,
                    alias: alias.map(|alias| alias.name),
                    ty,
                },
                Vec::new(),
            ),
            yuzu_anf::Rel::Join {
                left,
                right,
                kind,
                condition,
                ty,
            } => {
                let condition = match condition {
                    yuzu_anf::JoinCondition::On(thunk) => {
                        JoinCondition::On(self.convert_thunk(&thunk)?)
                    }
                    yuzu_anf::JoinCondition::Using(columns) => {
                        let left_ty = self.rel_ty(left);
                        let right_ty = self.rel_ty(right);
                        JoinCondition::Using(
                            columns
                                .iter()
                                .map(|column| JoinKey {
                                    left: self.position(left_ty, column.name),
                                    right: self.position(right_ty, column.name),
                                })
                                .collect(),
                        )
                    }
                };
                (
                    Rel::Join {
                        kind: convert_kind(kind),
                        condition: Some(condition),
                        ty,
                    },
                    vec![self.convert_rel(left)?, self.convert_rel(right)?],
                )
            }
            yuzu_anf::Rel::Select { input, items, ty } => (
                Rel::Select {
                    items: self.convert_select_items(&items)?,
                    ty,
                },
                vec![self.convert_rel(input)?],
            ),
            yuzu_anf::Rel::Where {
                input,
                predicate,
                ty,
            } => (
                Rel::Where {
                    predicate: self.convert_thunk(&predicate)?,
                    ty,
                },
                vec![self.convert_rel(input)?],
            ),
            yuzu_anf::Rel::Distinct { input, ty } => {
                (Rel::Distinct { ty }, vec![self.convert_rel(input)?])
            }
            yuzu_anf::Rel::Drop { input, columns, ty } => {
                let input_ty = self.rel_ty(input);
                (
                    Rel::Drop {
                        columns: columns
                            .iter()
                            .map(|column| self.position(input_ty, column.name))
                            .collect(),
                        ty,
                    },
                    vec![self.convert_rel(input)?],
                )
            }
            yuzu_anf::Rel::Rename { input, items, ty } => {
                let input_ty = self.rel_ty(input);
                (
                    Rel::Rename {
                        items: items
                            .iter()
                            .map(|item| RenameItem {
                                column: self.position(input_ty, item.from.name),
                                to: item.to.name,
                            })
                            .collect(),
                        ty,
                    },
                    vec![self.convert_rel(input)?],
                )
            }
            yuzu_anf::Rel::Extend { input, items, ty } => (
                Rel::Extend {
                    items: self.convert_select_items(&items)?,
                    ty,
                },
                vec![self.convert_rel(input)?],
            ),
            yuzu_anf::Rel::Set { input, items, ty } => {
                let input_ty = self.rel_ty(input);
                let items = items
                    .iter()
                    .map(|item| {
                        Ok(SetItem {
                            column: self.position(input_ty, item.column.name),
                            value: self.convert_thunk(&item.value)?,
                        })
                    })
                    .collect::<Result<_, Unsupported>>()?;
                (Rel::Set { items, ty }, vec![self.convert_rel(input)?])
            }
            yuzu_anf::Rel::Limit {
                input,
                count,
                offset,
                ty,
            } => (
                Rel::Limit {
                    count: self.convert_count(&count)?,
                    offset: match offset {
                        Some(offset) => Some(self.convert_count(&offset)?),
                        None => None,
                    },
                    ty,
                },
                vec![self.convert_rel(input)?],
            ),
            yuzu_anf::Rel::Alias { input, alias, ty } => (
                Rel::Alias {
                    alias: alias.name,
                    ty,
                },
                vec![self.convert_rel(input)?],
            ),
            yuzu_anf::Rel::Aggregate {
                input,
                items,
                groups,
                ty,
            } => return self.convert_aggregate_rel(input, &items, &groups, ty),
        };
        Ok(self.graph.add(rel, &inputs))
    }

    fn convert_select_items(
        &mut self,
        items: &[yuzu_anf::SelectItem],
    ) -> Result<Box<[SelectItem]>, Unsupported> {
        items
            .iter()
            .map(|item| {
                Ok(SelectItem {
                    body: self.convert_thunk(&item.body)?,
                    alias: item.alias.map(|alias| alias.name),
                })
            })
            .collect()
    }

    /// The aggregate computes keys ++ measures; when an item is more than a
    /// bare aggregate call, a `Select` over that row computes the surface
    /// output, so however many calls the items contain, there is one
    /// `Aggregate` and at most one projection.
    fn convert_aggregate_rel(
        &mut self,
        input: yuzu_anf::RelId,
        items: &[yuzu_anf::AggregateItem],
        groups: &[yuzu_anf::GroupKey],
        ty: TypeId,
    ) -> Result<RelId, Unsupported> {
        let input = self.convert_rel(input)?;
        let output_columns = self.columns(ty).to_vec();
        let key_count = groups.len();
        let groupings: Vec<u32> = groups.iter().map(|key| key.column).collect();

        self.agg = Some(AggState {
            keys: groupings.clone(),
            measures: Vec::new(),
            names: Vec::new(),
            current_name: output_columns
                .first()
                .map(|column| column.name)
                .unwrap_or_else(|| groups[0].name.name),
            in_args: false,
        });
        let mut item_exprs = Vec::new();
        for (index, item) in items.iter().enumerate() {
            let state = self.agg.as_mut().expect("aggregate state is set");
            state.current_name = output_columns[key_count + index].name;
            let body = self.convert_thunk(&item.body);
            if body.is_err() {
                self.agg = None;
            }
            item_exprs.push((body?, item.alias.map(|alias| alias.name)));
        }
        let state = self.agg.take().expect("aggregate state is set");

        let mut aggregate_columns = output_columns[..key_count].to_vec();
        for (measure, &name) in state.measures.iter().zip(&state.names) {
            aggregate_columns.push(Column::new(None, name, measure.ty));
        }
        let aggregate_ty = self.types.relation_ty(aggregate_columns);
        let node = self.graph.add(
            Rel::Aggregate {
                groupings: groupings.into(),
                measures: state.measures.clone().into(),
                ty: aggregate_ty,
            },
            &[input],
        );

        let mut bare = state.measures.len() == items.len();
        if bare {
            for (index, (expr, _)) in item_exprs.iter().enumerate() {
                let measure_column = self.graph.intern_expr(Expr::Column {
                    column: (key_count + index) as u32,
                    ty: state.measures[index].ty,
                });
                if *expr != measure_column {
                    bare = false;
                    break;
                }
            }
        }
        if bare {
            return Ok(node);
        }

        let mut select_items = Vec::with_capacity(key_count + item_exprs.len());
        for (index, column) in output_columns[..key_count].iter().enumerate() {
            let body = self.graph.intern_expr(Expr::Column {
                column: index as u32,
                ty: column.ty,
            });
            select_items.push(SelectItem { body, alias: None });
        }
        for (body, alias) in item_exprs {
            select_items.push(SelectItem { body, alias });
        }
        Ok(self.graph.add(
            Rel::Select {
                items: select_items.into(),
                ty,
            },
            &[node],
        ))
    }

    fn convert_thunk(&mut self, thunk: &yuzu_anf::Thunk) -> Result<ExprId, Unsupported> {
        let mut defs = Defs::new();
        for &stmt in thunk.stmts.iter() {
            if let yuzu_anf::Stmt::Let { binding, expr } = self.anf.stmt(stmt) {
                defs.insert(*binding, *expr);
            }
        }
        self.convert_atom(thunk.value, &defs)
    }

    fn convert_expr(&mut self, expr: yuzu_anf::ExprId, defs: &Defs) -> Result<ExprId, Unsupported> {
        match self.anf.expr(expr).clone() {
            yuzu_anf::Expr::Atom { value } => self.convert_atom(value, defs),
            yuzu_anf::Expr::Call { op, args, ty } => match op {
                yuzu_anf::Op::UnaryPos => self.convert_atom(args[0], defs),
                yuzu_anf::Op::In | yuzu_anf::Op::NotIn => {
                    self.convert_membership(expr, op, &args, ty, defs)
                }
                _ => {
                    let args = args
                        .iter()
                        .map(|&arg| self.convert_atom(arg, defs))
                        .collect::<Result<_, _>>()?;
                    Ok(self.graph.intern_expr(Expr::Call {
                        func: convert_op(op),
                        args,
                        ty,
                    }))
                }
            },
            yuzu_anf::Expr::AggCall { func, args, ty } => {
                let state = self
                    .agg
                    .as_mut()
                    .expect("an aggregate call sits in an aggregate item");
                state.in_args = true;
                let args = args
                    .iter()
                    .map(|&arg| self.convert_atom(arg, defs))
                    .collect::<Result<_, _>>()?;
                let state = self
                    .agg
                    .as_mut()
                    .expect("an aggregate call sits in an aggregate item");
                state.in_args = false;

                let measure = Measure { func, args, ty };
                let index = match state.measures.iter().position(|seen| *seen == measure) {
                    Some(index) => index,
                    None => {
                        state.measures.push(measure);
                        state.names.push(state.current_name);
                        state.measures.len() - 1
                    }
                };
                let column = (state.keys.len() + index) as u32;
                Ok(self.graph.intern_expr(Expr::Column { column, ty }))
            }
            yuzu_anf::Expr::ExternCall { name, args, ty } => {
                let args = args
                    .iter()
                    .map(|&arg| self.convert_atom(arg, defs))
                    .collect::<Result<_, _>>()?;
                Ok(self.graph.intern_expr(Expr::Call {
                    func: Func::External(name.name),
                    args,
                    ty,
                }))
            }
            yuzu_anf::Expr::FuncCall { callee, .. } => {
                let message = match *self.anf.atom(callee) {
                    yuzu_anf::Atom::FuncRef { binding, .. } => format!(
                        "call to `{}` could not be fully reduced",
                        self.interner.text(self.anf.binding(binding).name.name)
                    ),
                    _ => "a call could not be fully reduced".to_string(),
                };
                Err(self.unsupported(expr, message))
            }
            yuzu_anf::Expr::MethodCall { .. } => {
                Err(self.unsupported(expr, "method calls are not yet supported in query plans"))
            }
            yuzu_anf::Expr::StructInit { .. } => {
                Err(self.unsupported(expr, "a struct value cannot be a query column"))
            }
            yuzu_anf::Expr::ListInit { .. } => {
                Err(self.unsupported(expr, "a list can only be tested for membership"))
            }
            yuzu_anf::Expr::Rel(_) => {
                Err(self.unsupported(expr, "a query cannot be used as a column"))
            }
        }
    }

    fn convert_atom(&mut self, atom: yuzu_anf::AtomId, defs: &Defs) -> Result<ExprId, Unsupported> {
        let expr = match *self.anf.atom(atom) {
            yuzu_anf::Atom::Const(constant) => Expr::Literal {
                value: convert_const(constant),
                ty: self.const_ty(constant),
            },
            yuzu_anf::Atom::Column { column, ty, .. } => {
                let column = match &self.agg {
                    Some(state) if !state.in_args => state
                        .keys
                        .iter()
                        .position(|&key| key == column)
                        .expect("inference admits only group keys at group level")
                        as u32,
                    _ => column,
                };
                Expr::Column { column, ty }
            }
            yuzu_anf::Atom::Var { binding } => match defs.get(&binding) {
                Some(&expr) => return self.convert_expr(expr, defs),
                None => {
                    let name = self.interner.text(self.anf.binding(binding).name.name);
                    let message = if name.starts_with('%') {
                        "query column depends on a value that could not be fully reduced"
                            .to_string()
                    } else {
                        format!(
                            "query column depends on `{name}`, which could not be fully reduced"
                        )
                    };
                    return Err(self.unsupported_query(message));
                }
            },
            yuzu_anf::Atom::Field { field, .. } => {
                let message = format!(
                    "`{}` is a field of a value, not a query column",
                    self.interner.text(field.name)
                );
                return Err(self.unsupported_query(message));
            }
            yuzu_anf::Atom::FuncRef { .. } => {
                unreachable!("a function reference cannot be a column")
            }
        };
        Ok(self.graph.intern_expr(expr))
    }

    /// `x in xs` becomes a variadic call over the list's elements, and
    /// `not in` wraps it in `not`, so a membership test has one spelling and
    /// no list value survives into the plan.
    fn convert_membership(
        &mut self,
        expr: yuzu_anf::ExprId,
        op: yuzu_anf::Op,
        args: &[yuzu_anf::AtomId],
        ty: TypeId,
        defs: &Defs,
    ) -> Result<ExprId, Unsupported> {
        let [value, list] = args else {
            unreachable!("a membership test has a value and a list")
        };
        let Some(elements) = self.list_elements(*list, defs) else {
            let message = "a membership test needs a list that is known at compile time";
            return Err(self.unsupported(expr, message));
        };

        let mut options = vec![self.convert_atom(*value, defs)?];
        for element in elements.iter() {
            options.push(self.convert_atom(*element, defs)?);
        }
        let test = self.graph.intern_expr(Expr::Call {
            func: Func::In,
            args: options.into(),
            ty,
        });

        Ok(match op {
            yuzu_anf::Op::In => test,
            yuzu_anf::Op::NotIn => self.graph.intern_expr(Expr::Call {
                func: Func::Not,
                args: Box::new([test]),
                ty,
            }),
            _ => unreachable!("membership is `in` or `not in`"),
        })
    }

    fn list_elements(
        &self,
        list: yuzu_anf::AtomId,
        defs: &Defs,
    ) -> Option<Box<[yuzu_anf::AtomId]>> {
        let yuzu_anf::Atom::Var { binding } = *self.anf.atom(list) else {
            return None;
        };
        let expr = *defs.get(&binding)?;
        let yuzu_anf::Expr::ListInit { elements, .. } = self.anf.expr(expr) else {
            return None;
        };
        Some(elements.clone())
    }

    /// A `limit` count or offset. A plan carries a number, so the thunk has to
    /// have reduced to a single non-negative integer.
    fn convert_count(&mut self, thunk: &yuzu_anf::Thunk) -> Result<u64, Unsupported> {
        match *self.anf.atom(thunk.value) {
            yuzu_anf::Atom::Const(yuzu_anf::Const::Int { value }) => value
                .as_i64()
                .try_into()
                .map_err(|_| self.unsupported_query("`limit` needs a non-negative row count")),
            _ => {
                Err(self
                    .unsupported_query("`limit` needs a row count that is known at compile time"))
            }
        }
    }

    fn rel_ty(&self, rel: yuzu_anf::RelId) -> TypeId {
        match self.anf.rel(rel) {
            yuzu_anf::Rel::From { ty, .. }
            | yuzu_anf::Rel::Join { ty, .. }
            | yuzu_anf::Rel::Select { ty, .. }
            | yuzu_anf::Rel::Where { ty, .. }
            | yuzu_anf::Rel::Distinct { ty, .. }
            | yuzu_anf::Rel::Drop { ty, .. }
            | yuzu_anf::Rel::Rename { ty, .. }
            | yuzu_anf::Rel::Extend { ty, .. }
            | yuzu_anf::Rel::Set { ty, .. }
            | yuzu_anf::Rel::Limit { ty, .. }
            | yuzu_anf::Rel::Alias { ty, .. }
            | yuzu_anf::Rel::Aggregate { ty, .. } => *ty,
        }
    }

    fn columns(&self, rel_ty: TypeId) -> &[Column] {
        let Type::Relation(relation) = self.types.ty(rel_ty) else {
            unreachable!("a pipeline stage always has a relation type")
        };
        &relation.columns
    }

    fn position(&self, rel_ty: TypeId, name: SymbolId) -> u32 {
        self.columns(rel_ty)
            .iter()
            .position(|column| column.name == name)
            .expect("inference resolved every column it admitted") as u32
    }

    /// A transcription of the type inference gave the value, which `Int` and
    /// `Float` carry in full — no inference of our own.
    fn const_ty(&mut self, constant: yuzu_anf::Const) -> TypeId {
        match constant {
            yuzu_anf::Const::Int { value } => {
                let ty = match (value.signedness(), value.num_bits()) {
                    (Signedness::Signed, 8) => Type::Int8,
                    (Signedness::Signed, 16) => Type::Int16,
                    (Signedness::Signed, 32) => Type::Int32,
                    (Signedness::Signed, _) => Type::Int64,
                    (Signedness::Unsigned, 8) => Type::UInt8,
                    (Signedness::Unsigned, 16) => Type::UInt16,
                    (Signedness::Unsigned, 32) => Type::UInt32,
                    (Signedness::Unsigned, _) => Type::UInt64,
                };
                self.types.intern_ty(ty)
            }
            yuzu_anf::Const::Float { value } => match value.num_bits() {
                32 => self.types.intern_ty(Type::Float32),
                _ => self.types.float64_ty(),
            },
            yuzu_anf::Const::Bool { .. } => self.types.bool_ty(),
            yuzu_anf::Const::String { .. } => self.types.str_ty(),
        }
    }

    fn unsupported(&mut self, expr: yuzu_anf::ExprId, message: impl Into<String>) -> Unsupported {
        let span = self
            .source_map
            .expr(expr)
            .expect("a reported node is always in the source map");
        self.report(span, message)
    }

    fn unsupported_query(&mut self, message: impl Into<String>) -> Unsupported {
        let span = self
            .source_map
            .stmt(self.query_stmt)
            .expect("a reported node is always in the source map");
        self.report(span, message)
    }

    fn report(&mut self, span: Span, message: impl Into<String>) -> Unsupported {
        let diagnostic = DiagnosticBuilder::error(span, message)
            .note("expressions must be evaluatable at compile time");
        self.diagnostics.emit(diagnostic);
        Unsupported
    }
}

impl RelGraphConverter<yuzu_anf::RelId> for AnfToRelGraphConverter<'_> {
    fn convert(&mut self, root: yuzu_anf::RelId) -> Option<RelGraph> {
        let root = self.convert_rel(root).ok()?;
        self.graph.set_root(root);
        Some(std::mem::take(&mut self.graph))
    }
}

fn convert_kind(kind: yuzu_anf::JoinKind) -> JoinKind {
    match kind {
        yuzu_anf::JoinKind::Inner => JoinKind::Inner,
        yuzu_anf::JoinKind::Left => JoinKind::Left,
        yuzu_anf::JoinKind::Right => JoinKind::Right,
        yuzu_anf::JoinKind::Full => JoinKind::Full,
    }
}

fn convert_const(constant: yuzu_anf::Const) -> Const {
    match constant {
        yuzu_anf::Const::Int { value } => Const::Int { value },
        yuzu_anf::Const::Float { value } => Const::Float { value },
        yuzu_anf::Const::Bool { value } => Const::Bool { value },
        yuzu_anf::Const::String { value } => Const::String { value },
    }
}

fn convert_op(op: yuzu_anf::Op) -> Func {
    match op {
        yuzu_anf::Op::Add => Func::Add,
        yuzu_anf::Op::Sub => Func::Subtract,
        yuzu_anf::Op::Mul => Func::Multiply,
        yuzu_anf::Op::Div => Func::Divide,
        yuzu_anf::Op::Pow => Func::Power,
        yuzu_anf::Op::ShiftLeft => Func::ShiftLeft,
        yuzu_anf::Op::ShiftRight => Func::ShiftRight,
        yuzu_anf::Op::Eq => Func::Equal,
        yuzu_anf::Op::Neq => Func::NotEqual,
        yuzu_anf::Op::Lt => Func::Less,
        yuzu_anf::Op::Lte => Func::LessEqual,
        yuzu_anf::Op::Gt => Func::Greater,
        yuzu_anf::Op::Gte => Func::GreaterEqual,
        yuzu_anf::Op::And => Func::And,
        yuzu_anf::Op::Or => Func::Or,
        yuzu_anf::Op::UnaryNeg => Func::Negate,
        yuzu_anf::Op::UnaryNot => Func::Not,
        yuzu_anf::Op::In | yuzu_anf::Op::NotIn | yuzu_anf::Op::UnaryPos => {
            unreachable!("handled by the membership and identity paths")
        }
    }
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};
    use yuzu_ast::ast::{AstNode, Root as AstRoot};
    use yuzu_core::adt::StringInterner;
    use yuzu_diagnostics::{diagnostics::engine::DiagnosticsEngine, source_map::SourceMap};
    use yuzu_hir::HirCtx;
    use yuzu_lexer::lexer::{Lexer, Token};
    use yuzu_types::TypeCtx;

    use super::AnfToRelGraphConverter;
    use crate::graph::{RelGraph, RelGraphConverter};
    use crate::printer::dump;

    const TABLES: &str = "struct R { a: int32, b: int32 }\ntable t = R\nstruct S { a: int32, c: int32 }\ntable u = S\n";

    /// Runs the pipeline through reduction, converts the tail query, and
    /// returns the graph alongside anything the conversion reported.
    fn convert(input: &str) -> (Option<RelGraph>, StringInterner, Vec<String>) {
        let mut interner = StringInterner::new();
        let mut diagnostics = DiagnosticsEngine::new();
        let mut sources = SourceMap::new();
        let source_id = sources.add("test".to_string(), input.to_string());

        let tokens: Vec<Token> = Lexer::new(input).collect();
        let syntax = yuzu_parser::parse(&tokens, &mut diagnostics, source_id);
        let ast_root = AstRoot::cast(syntax).expect("root node");

        let mut hir = HirCtx::new();
        let (hir_root, hir_source_map) = yuzu_hir::lower(
            ast_root,
            &mut hir,
            &mut interner,
            &mut diagnostics,
            source_id,
        );

        let mut types = TypeCtx::new();
        let inference = yuzu_hir::infer(
            &hir_root,
            &hir,
            &yuzu_types::Builtins,
            &mut interner,
            &mut types,
            &mut diagnostics,
            &hir_source_map,
        );

        let messages: Vec<&str> = diagnostics
            .diagnostics()
            .iter()
            .map(|diagnostic| diagnostic.message.as_str())
            .collect();
        assert!(
            messages.is_empty(),
            "program should type-check cleanly, got: {messages:?}"
        );

        let mut anf = yuzu_anf::AnfCtx::new();
        let (anf_root, mut anf_source_map) = yuzu_anf::lower(
            &hir_root,
            &hir,
            &inference,
            &types,
            &mut anf,
            &mut interner,
            &mut diagnostics,
            &hir_source_map,
        );
        let reduced = yuzu_anf::reduce(&anf_root, &mut anf, &mut interner, &mut anf_source_map);

        let (query, query_stmt) = reduced
            .stmts
            .iter()
            .rev()
            .find_map(|&id| match anf.stmt(id) {
                yuzu_anf::Stmt::Expr { value } => match anf.expr(*value) {
                    yuzu_anf::Expr::Rel(rel) => Some((*rel, id)),
                    _ => None,
                },
                _ => None,
            })
            .expect("test program ends in a query");

        let mut converter = AnfToRelGraphConverter::new(
            &anf,
            &mut types,
            &interner,
            &anf_source_map,
            &mut diagnostics,
            query_stmt,
        );
        let graph = converter.convert(query);

        let errors = diagnostics
            .diagnostics()
            .iter()
            .map(|diagnostic| diagnostic.message.clone())
            .collect();
        (graph, interner, errors)
    }

    fn check(input: &str, expected: Expect) {
        let (graph, interner, errors) = convert(input);
        assert!(
            errors.is_empty(),
            "conversion should succeed, got: {errors:?}"
        );
        let graph = graph.expect("a clean conversion produces a graph");
        expected.assert_eq(&dump(&graph, &interner));
    }

    fn check_error(input: &str, expected_message: &str) {
        let (graph, _, errors) = convert(input);
        assert!(graph.is_none(), "conversion should fail");
        assert!(
            errors.iter().any(|error| error.contains(expected_message)),
            "expected an error containing {expected_message:?}, got: {errors:?}"
        );
    }

    #[test]
    fn select_becomes_a_tree() {
        check(
            &format!("{TABLES}from t |> select a + 1 as x, b"),
            expect![[r#"
                select [add(#0, 1i32) as x, #1]
                  from t
            "#]],
        );
    }

    #[test]
    fn membership_becomes_a_variadic_call() {
        check(
            &format!("{TABLES}from t |> where a in [1, 3]"),
            expect![[r#"
                where in(#0, 1i32, 3i32)
                  from t
            "#]],
        );
    }

    #[test]
    fn negated_membership_wraps_in_not() {
        check(
            &format!("{TABLES}from t |> where a not in [1, 3]"),
            expect![[r#"
                where not(in(#0, 1i32, 3i32))
                  from t
            "#]],
        );
    }

    #[test]
    fn join_condition_uses_concatenated_positions() {
        check(
            &format!("{TABLES}from t x |> join u y on x.a == y.a"),
            expect![[r#"
                join inner on equal(#0, #2)
                  from t as x
                  from u as y
            "#]],
        );
    }

    #[test]
    fn using_resolves_each_side_to_a_position() {
        check(
            &format!("{TABLES}from t x |> join u y using (a)"),
            expect![[r#"
                join inner using (#0 = #0)
                  from t as x
                  from u as y
            "#]],
        );
    }

    #[test]
    fn drop_and_rename_become_positional() {
        check(
            &format!("{TABLES}from t |> drop b"),
            expect![[r#"
                drop [#1]
                  from t
            "#]],
        );
        check(
            &format!("{TABLES}from t |> rename a as z"),
            expect![[r#"
                rename [#0 as z]
                  from t
            "#]],
        );
    }

    #[test]
    fn set_becomes_positional() {
        check(
            &format!("{TABLES}from t |> set a = 5"),
            expect![[r#"
                set [#0 = 5i64]
                  from t
            "#]],
        );
    }

    #[test]
    fn limit_count_is_a_folded_constant() {
        check(
            &format!("{TABLES}from t |> limit 2 + 3"),
            expect![[r#"
                limit 5
                  from t
            "#]],
        );
    }

    #[test]
    fn alias_is_kept() {
        check(
            &format!("{TABLES}from t |> as x |> select x.a"),
            expect![[r#"
                select [#0]
                  as x
                    from t
            "#]],
        );
    }

    #[test]
    fn distinct_converts() {
        check(
            &format!("{TABLES}from t |> distinct"),
            expect![[r#"
            distinct
              from t
        "#]],
        );
    }

    #[test]
    fn extend_converts() {
        check(
            &format!("{TABLES}from t |> extend a * 2 as d"),
            expect![[r#"
            extend [multiply(#0, 2i32) as d]
              from t
        "#]],
        );
    }

    #[test]
    fn limit_offset_is_a_folded_constant() {
        check(
            &format!("{TABLES}from t |> limit 5 offset 1 + 1"),
            expect![[r#"
            limit 5 offset 2
              from t
        "#]],
        );
    }

    #[test]
    fn every_join_kind_converts() {
        check(
            &format!("{TABLES}from t x |> left join u y on x.a == y.a"),
            expect![[r#"
                join left on equal(#0, #2)
                  from t as x
                  from u as y
            "#]],
        );
        check(
            &format!("{TABLES}from t x |> right join u y on x.a == y.a"),
            expect![[r#"
                join right on equal(#0, #2)
                  from t as x
                  from u as y
            "#]],
        );
        check(
            &format!("{TABLES}from t x |> full join u y on x.a == y.a"),
            expect![[r#"
                join full on equal(#0, #2)
                  from t as x
                  from u as y
            "#]],
        );
    }

    #[test]
    fn unary_operators_convert() {
        check(
            &format!("{TABLES}from t |> where not (a == 1) |> select -a as n, +b as p"),
            expect![[r#"
                select [negate(#0) as n, #1 as p]
                  where not(equal(#0, 1i32))
                    from t
            "#]],
        );
    }

    #[test]
    fn every_literal_kind_converts() {
        check(
            &format!("{TABLES}from t |> select 1.5 as f, true as g, \"s\" as h"),
            expect![[r#"
                select [1.5f64 as f, true as g, "s" as h]
                  from t
            "#]],
        );
    }

    #[test]
    fn unreduced_call_in_a_column_is_reported() {
        check_error(
            &format!(
                "{TABLES}def f(n: int64) -> int64 {{ return n * f(n - 1) }}\nfrom t |> select f(3) as v"
            ),
            "call to `f` could not be fully reduced",
        );
    }

    #[test]
    fn join_edges_point_at_both_inputs_in_order() {
        let (graph, interner, errors) =
            convert(&format!("{TABLES}from t x |> join u y on x.a == y.a"));
        assert!(
            errors.is_empty(),
            "conversion should succeed, got: {errors:?}"
        );
        let graph = graph.expect("a clean conversion produces a graph");
        let root = graph.root().expect("the graph has a root");
        expect![[r#"
            join inner on equal(#0, #2)
              from t as x
              from u as y
        "#]]
        .assert_eq(&dump(&graph, &interner));
        assert_eq!(graph.node_count(), 3);

        // The join's inputs come back in operand order: left, then right.
        let inputs = graph.inputs(root);
        let names: Vec<&str> = inputs
            .iter()
            .map(|&input| {
                let crate::Rel::From { relation, .. } = graph.plan().rel(input) else {
                    panic!("a join input is a read")
                };
                interner.text(*relation)
            })
            .collect();
        assert_eq!(names, ["t", "u"]);

        for &input in &inputs {
            assert_eq!(graph.parents(input).collect::<Vec<_>>(), [root]);
        }
        assert_eq!(graph.parents(root).count(), 0);
        assert!(graph.inputs(inputs[0]).is_empty());
    }

    #[test]
    fn identical_column_expressions_intern_to_one_tree() {
        let (graph, interner, errors) =
            convert(&format!("{TABLES}from t |> select a + 1 as x, a + 1 as y"));
        assert!(
            errors.is_empty(),
            "conversion should succeed, got: {errors:?}"
        );
        let graph = graph.expect("a clean conversion produces a graph");
        expect![[r#"
            select [add(#0, 1i32) as x, add(#0, 1i32) as y]
              from t
        "#]]
        .assert_eq(&dump(&graph, &interner));
        // The column, the literal, and one `add` shared by both items.
        assert_eq!(graph.plan().expr_count(), 3);
    }

    #[test]
    fn self_join_shares_one_read() {
        let (graph, interner, errors) = convert(&format!("{TABLES}from t |> join t using (a)"));
        assert!(
            errors.is_empty(),
            "conversion should succeed, got: {errors:?}"
        );
        let graph = graph.expect("a clean conversion produces a graph");
        let root = graph.root().expect("the graph has a root");

        // The dump re-expands the shared read; the node count is what shows
        // both sides of the join are one relation — a DAG, not a tree.
        expect![[r#"
            join inner using (#0 = #0)
              from t
              from t
        "#]]
        .assert_eq(&dump(&graph, &interner));
        assert_eq!(graph.node_count(), 2);
        let inputs = graph.inputs(root);
        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0], inputs[1]);
        assert_eq!(graph.parents(inputs[0]).collect::<Vec<_>>(), [root]);
        assert_eq!(graph.parents(root).count(), 0);
    }

    #[test]
    fn bare_aggregate_is_one_node() {
        check(
            &format!("{TABLES}from t |> aggregate sum(a) as s, count() as n group by b"),
            expect![[r#"
                aggregate [sum(#0), count()] group [#1]
                  from t
            "#]],
        );
    }

    #[test]
    fn full_table_aggregate_has_no_groupings() {
        check(
            &format!("{TABLES}from t |> aggregate count() as n"),
            expect![[r#"
            aggregate [count()]
              from t
        "#]],
        );
    }

    #[test]
    fn composite_item_projects_over_the_measures() {
        check(
            &format!("{TABLES}from t |> aggregate max(a) - min(a) as spread group by b"),
            expect![[r#"
                select [#0, subtract(#1, #2) as spread]
                  aggregate [max(#0), min(#0)] group [#1]
                    from t
            "#]],
        );
    }

    #[test]
    fn group_key_reads_back_as_its_output_position() {
        check(
            &format!("{TABLES}from t |> aggregate b + min(a) as v group by b"),
            expect![[r#"
                select [#0, add(#0, #1) as v]
                  aggregate [min(#0)] group [#1]
                    from t
            "#]],
        );
    }

    #[test]
    fn count_and_count_distinct_stay_apart() {
        let (graph, _, errors) = convert(&format!(
            "{TABLES}from t |> aggregate count(a) + count_distinct(a) as v group by b"
        ));
        assert!(
            errors.is_empty(),
            "conversion should succeed, got: {errors:?}"
        );
        let graph = graph.expect("a clean conversion produces a graph");
        let measures = graph
            .nodes()
            .find_map(|node| match graph.plan().rel(node) {
                crate::Rel::Aggregate { measures, .. } => Some(measures.len()),
                _ => None,
            })
            .expect("the graph holds an aggregate");
        assert_eq!(measures, 2);
    }

    #[test]
    fn group_level_expressions_read_key_output_positions() {
        check(
            &format!("{TABLES}from t |> aggregate sum(a + b) as s, a + b as x group by a, b"),
            expect![[r#"
                select [#0, #1, #2 as s, add(#0, #1) as x]
                  aggregate [sum(add(#0, #1))] group [#0, #1]
                    from t
            "#]],
        );
    }

    #[test]
    fn repeated_measures_intern_to_one() {
        let (graph, _, errors) = convert(&format!(
            "{TABLES}from t |> aggregate max(a) - min(a) + max(a) as v group by b"
        ));
        assert!(
            errors.is_empty(),
            "conversion should succeed, got: {errors:?}"
        );
        let graph = graph.expect("a clean conversion produces a graph");
        let aggregate = graph
            .nodes()
            .find_map(|node| match graph.plan().rel(node) {
                crate::Rel::Aggregate { measures, .. } => Some(measures.len()),
                _ => None,
            })
            .expect("the graph holds an aggregate");
        // `max(a)` twice and `min(a)` once: two measures, not three.
        assert_eq!(aggregate, 2);
    }

    fn validated(input: &str, target: &str) -> Vec<String> {
        let (graph, _, errors) = convert(input);
        assert!(errors.is_empty(), "conversion should succeed: {errors:?}");
        let graph = graph.expect("a clean conversion produces a graph");

        let mut diagnostics = DiagnosticsEngine::new();
        let span = yuzu_diagnostics::diagnostics::Span {
            source_id: SourceMap::new().add("test".into(), String::new()),
            range: Default::default(),
        };
        crate::validate(
            &graph,
            &target.parse().expect("a known target"),
            &mut diagnostics,
            span,
        );
        diagnostics
            .diagnostics()
            .iter()
            .map(|diagnostic| diagnostic.message.clone())
            .collect()
    }

    #[test]
    fn datafusion_target_rejects_shifts() {
        let errors = validated(
            &format!("{TABLES}from t |> select a << 1 as x, b >> 1 as y"),
            "datafusion",
        );
        assert_eq!(
            errors,
            [
                "`<<` is not supported by the datafusion target",
                "`>>` is not supported by the datafusion target"
            ]
        );
    }

    #[test]
    fn postgres_target_accepts_shifts() {
        let errors = validated(&format!("{TABLES}from t |> select a << 1 as x"), "postgres");
        assert!(errors.is_empty(), "{errors:?}");
    }

    #[test]
    fn aggregates_validate_against_the_target() {
        let errors = validated(
            &format!("{TABLES}from t |> aggregate count_distinct(a) as n group by b"),
            "datafusion",
        );
        assert!(errors.is_empty(), "{errors:?}");
    }

    #[test]
    fn extern_calls_become_external_plan_calls() {
        check(
            &format!(
                "{TABLES}external def clamp(x: int32, low: int32) -> int32\nexternal agg def median(x: int32) -> float64\nfrom t |> aggregate median(a) as m group by b |> select m, clamp(b, 1) as c"
            ),
            expect![[r#"
                select [#1, clamp(#0, 1i32) as c]
                  aggregate [median(#0)] group [#1]
                    from t
            "#]],
        );
    }

    #[test]
    fn limit_needs_a_constant_count() {
        check_error(
            &format!(
                "{TABLES}def f(n: int64) -> int64 {{ return n * f(n - 1) }}\nfrom t |> limit f(3)"
            ),
            "`limit` needs a row count that is known at compile time",
        );
    }
}
