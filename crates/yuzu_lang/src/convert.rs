//! The AST → yzl conversion. Names come out as `yzl.name`, unresolved
//! types as `!yzl.var`, sugar intact — the checking passes resolve them.

use std::collections::HashMap;

use melior::Context;
use melior::ir::attribute::{
    ArrayAttribute, FlatSymbolRefAttribute, FloatAttribute, IntegerAttribute, StringAttribute,
    TypeAttribute,
};
use melior::ir::operation::OperationBuilder;
use melior::ir::{
    Attribute, Block, BlockLike, BlockRef, Identifier, Location, Module, Region, RegionLike, Type,
    Value,
};
use yuzu_ast::ast::{self, AstNode, BinOp, UnaryOp};
use yuzu_diagnostics::diagnostics::engine::DiagnosticsEngine;
use yuzu_diagnostics::diagnostics::printer::DiagnosticPrinter;
use yuzu_diagnostics::source_map::SourceMap;
use yuzu_lexer::lexer::{Lexer, Token};
use yuzu_mlir::ods::{yz, yzl};

/// A parsed and converted module, along with every construct the
/// conversion does not carry yet.
pub struct Conversion<'c> {
    pub module: Module<'c>,
    pub unsupported: Vec<String>,
}

/// Parses the source and converts it to a yzl module, printing any parse
/// diagnostics to stderr. Returns `None` when the source has no root.
pub fn convert_source<'c>(
    context: &'c Context,
    name: &str,
    source: &str,
) -> Option<Conversion<'c>> {
    let mut diagnostics = DiagnosticsEngine::new();
    let mut sources = SourceMap::new();
    let source_id = sources.add(name.to_string(), source.to_string());
    let tokens: Vec<Token> = Lexer::new(source).collect();
    let syntax = yuzu_parser::parse(&tokens, &mut diagnostics, source_id);
    let printer = DiagnosticPrinter::new(&sources);
    for diagnostic in diagnostics.diagnostics() {
        eprintln!("{}", printer.print(diagnostic));
    }
    let root = ast::Root::cast(syntax)?;
    let converter = AstToYzl::new(context, name, source);
    let module = converter.convert(&root);
    Some(Conversion {
        module,
        unsupported: converter.unsupported.into_inner(),
    })
}

struct AstToYzl<'c> {
    context: &'c Context,
    unsupported: std::cell::RefCell<Vec<String>>,
    name: String,
    line_starts: Vec<usize>,
    types: yuzu_mlir::Types<'c>,
}

type Locals<'c, 'a> = HashMap<String, Value<'c, 'a>>;

impl<'c> AstToYzl<'c> {
    fn new(context: &'c Context, name: &str, source: &str) -> Self {
        let mut line_starts = vec![0];
        line_starts.extend(source.match_indices('\n').map(|(at, _)| at + 1));
        Self {
            context,
            unsupported: std::cell::RefCell::new(Vec::new()),
            name: name.to_string(),
            line_starts,
            types: yuzu_mlir::Types::new(context),
        }
    }

    fn note(&self, what: impl Into<String>) {
        self.unsupported.borrow_mut().push(what.into());
    }

    fn location(&self, node: &impl AstNode) -> Location<'c> {
        let offset: usize = node.syntax().text_range().start().into();
        let line = self.line_starts.partition_point(|&start| start <= offset);
        let column = offset - self.line_starts[line - 1] + 1;
        Location::new(self.context, &self.name, line, column)
    }

    fn convert(&self, root: &ast::Root) -> Module<'c> {
        let module = Module::new(Location::new(self.context, &self.name, 1, 1));
        let top = module.body();
        let mut query = None;
        for stmt in root.stmts() {
            match &stmt {
                ast::Stmt::ExprStmt(expr_stmt) => {
                    if let Some(expr) = expr_stmt.expr() {
                        let value = self.convert_expr(top, &Locals::new(), &expr);
                        if matches!(expr, ast::Expr::Rel(_)) {
                            query = Some((value, self.location(expr_stmt)));
                        }
                    }
                }
                _ => self.convert_stmt(top, &stmt),
            }
        }
        if let Some((value, loc)) = query {
            top.append_operation(yzl::output(self.context, value, loc).into());
        }
        module
    }

    fn convert_stmt<'a>(&self, block: BlockRef<'c, 'a>, stmt: &ast::Stmt) {
        match stmt {
            ast::Stmt::StructStmt(decl) => self.convert_struct(block, decl),
            ast::Stmt::TableStmt(decl) => self.convert_table(block, decl),
            ast::Stmt::FuncStmt(decl) => self.convert_fn(block, decl),
            ast::Stmt::LetStmt(decl) => self.convert_let(block, decl),
            ast::Stmt::ExprStmt(stmt) => {
                if let Some(expr) = stmt.expr() {
                    self.convert_expr(block, &Locals::new(), &expr);
                }
            }
            unsupported => self.note(format!("unsupported statement {unsupported:?}")),
        }
    }

    fn convert_struct<'a>(&self, block: BlockRef<'c, 'a>, decl: &ast::StructStmt) {
        let Some(name) = ident_text(decl.name()) else {
            return;
        };
        let schema = self.schema_type(decl.fields());
        block.append_operation(
            yzl::r#struct(
                self.context,
                StringAttribute::new(self.context, &name),
                TypeAttribute::new(schema),
                self.location(decl),
            )
            .into(),
        );
    }

    fn convert_table<'a>(&self, block: BlockRef<'c, 'a>, decl: &ast::TableStmt) {
        let Some(name) = ident_text(decl.name()) else {
            return;
        };
        let row = match ident_text(decl.row_struct()) {
            Some(row) => row,
            // An inline table declares its row shape in place; give the shape
            // a struct of its own so the table can point at it.
            None => {
                let row = format!("{name}_row");
                let schema = self.schema_type(decl.inline_fields());
                block.append_operation(
                    yzl::r#struct(
                        self.context,
                        StringAttribute::new(self.context, &row),
                        TypeAttribute::new(schema),
                        self.location(decl),
                    )
                    .into(),
                );
                row
            }
        };
        block.append_operation(
            yzl::table(
                self.context,
                StringAttribute::new(self.context, &name),
                FlatSymbolRefAttribute::new(self.context, &row),
                self.location(decl),
            )
            .into(),
        );
    }

    fn convert_fn<'a>(&self, block: BlockRef<'c, 'a>, decl: &ast::FuncStmt) {
        let Some(name) = ident_text(decl.name()) else {
            return;
        };
        let params: Vec<Attribute> = decl
            .params()
            .filter_map(|param| ident_text(param.name()))
            .map(|name| StringAttribute::new(self.context, &name).into())
            .collect();
        let signature = {
            let params: Vec<&str> = decl
                .params()
                .map(|param| self.annotation_type(param.ty()))
                .collect();
            let result = self.annotation_type(decl.result());
            Type::parse(
                self.context,
                &format!("({}) -> {result}", params.join(", ")),
            )
            .expect("a signature type parses")
        };

        let region = Region::new();
        if let Some(body) = decl.body() {
            let entry = region.append_block(Block::new(&[]));
            let mut locals = Locals::new();
            for stmt in body.stmts() {
                match stmt {
                    ast::Stmt::LetStmt(binding) => {
                        let (Some(name), Some(expr)) = (ident_text(binding.name()), binding.expr())
                        else {
                            continue;
                        };
                        let value = self.convert_expr(entry, &locals, &expr);
                        locals.insert(name, value);
                    }
                    ast::Stmt::AssignStmt(assign) => {
                        let target = assign.target().and_then(|target| match target {
                            ast::Expr::IdentExpr(ident) => ident_text(ident.name()),
                            _ => None,
                        });
                        let (Some(name), Some(value)) = (target, assign.value()) else {
                            self.note(format!("unsupported assignment {assign:?}"));
                            continue;
                        };
                        let value = self.convert_expr(entry, &locals, &value);
                        locals.insert(name, value);
                    }
                    ast::Stmt::ReturnStmt(ret) => {
                        let values: Vec<Value> = ret
                            .expr()
                            .map(|expr| self.convert_expr(entry, &locals, &expr))
                            .into_iter()
                            .collect();
                        entry.append_operation(
                            yzl::r#return(self.context, &values, self.location(&ret)).into(),
                        );
                    }
                    unsupported => {
                        self.note(format!("unsupported function statement {unsupported:?}"))
                    }
                }
            }
        }

        let mut builder = OperationBuilder::new("yzl.fn", self.location(decl))
            .add_attributes(&[
                (
                    Identifier::new(self.context, "sym_name"),
                    StringAttribute::new(self.context, &name).into(),
                ),
                (
                    Identifier::new(self.context, "params"),
                    ArrayAttribute::new(self.context, &params).into(),
                ),
                (
                    Identifier::new(self.context, "signature"),
                    TypeAttribute::new(signature).into(),
                ),
            ])
            .add_regions([region]);
        for (marker, set) in [("agg", decl.is_agg()), ("external", decl.is_external())] {
            if set {
                builder = builder.add_attributes(&[(
                    Identifier::new(self.context, marker),
                    Attribute::unit(self.context),
                )]);
            }
        }
        block.append_operation(builder.build().expect("yzl.fn builds"));
    }

    fn convert_let<'a>(&self, block: BlockRef<'c, 'a>, decl: &ast::LetStmt) {
        let (Some(name), Some(expr)) = (ident_text(decl.name()), decl.expr()) else {
            return;
        };
        let region = Region::new();
        let body = region.append_block(Block::new(&[]));
        let value = self.convert_expr(body, &Locals::new(), &expr);
        body.append_operation(yzl::r#yield(self.context, &[value], self.location(decl)).into());
        block.append_operation(
            yzl::r#let(
                self.context,
                region,
                StringAttribute::new(self.context, &name),
                self.location(decl),
            )
            .into(),
        );
    }

    fn convert_expr<'a>(
        &self,
        block: BlockRef<'c, 'a>,
        locals: &Locals<'c, 'a>,
        expr: &ast::Expr,
    ) -> Value<'c, 'a> {
        let loc = self.location(expr);
        match expr {
            ast::Expr::Literal(literal) => self.convert_literal(block, literal),
            ast::Expr::IdentExpr(ident) => {
                let name = ident_text(ident.name()).unwrap_or_default();
                if let Some(&value) = locals.get(&name) {
                    return value;
                }
                self.name_ref(block, &name, loc)
            }
            ast::Expr::FieldAccessExpr(access) => {
                // `t.a` is one qualified reference, not a load: keep the
                // dotted name whole for resolution.
                let base = access
                    .base()
                    .and_then(|base| match base {
                        ast::Expr::IdentExpr(ident) => ident_text(ident.name()),
                        _ => None,
                    })
                    .unwrap_or_default();
                let field = ident_text(access.field()).unwrap_or_default();
                self.name_ref(block, &format!("{base}.{field}"), loc)
            }
            ast::Expr::BinaryExpr(binary) => self.convert_binary(block, locals, binary),
            ast::Expr::UnaryExpr(unary) => {
                let value = match unary.expr() {
                    Some(expr) => self.convert_expr(block, locals, &expr),
                    None => self.name_ref(block, "", loc),
                };
                let result = match unary.op() {
                    Some(UnaryOp::Neg) => yz::neg(self.context, self.types.var, value, loc).into(),
                    Some(UnaryOp::Not) => yz::not(self.context, self.types.var, value, loc).into(),
                    _ => return value,
                };
                first_result(block.append_operation(result))
            }
            ast::Expr::CallExpr(call) => {
                let callee = call
                    .callee()
                    .and_then(|callee| match callee {
                        ast::Expr::IdentExpr(ident) => ident_text(ident.name()),
                        _ => None,
                    })
                    .unwrap_or_default();
                let args: Vec<ast::Expr> = call
                    .args()
                    .into_iter()
                    .flat_map(|args| args.args().collect::<Vec<_>>())
                    .collect();
                let operands: Vec<Value> = args
                    .iter()
                    .map(|arg| self.convert_expr(block, locals, arg))
                    .collect();
                first_result(
                    block.append_operation(
                        yzl::call(
                            self.context,
                            self.types.var,
                            &operands,
                            FlatSymbolRefAttribute::new(self.context, &callee),
                            loc,
                        )
                        .into(),
                    ),
                )
            }
            ast::Expr::ListExpr(list) => {
                let elements: Vec<ast::Expr> = list.elements().collect();
                let values: Vec<Value> = elements
                    .iter()
                    .map(|element| self.convert_expr(block, locals, element))
                    .collect();
                first_result(
                    block.append_operation(
                        yzl::list(self.context, self.types.var, &values, loc).into(),
                    ),
                )
            }
            ast::Expr::ParenExpr(paren) => match paren.expr() {
                Some(inner) => self.convert_expr(block, locals, &inner),
                None => self.name_ref(block, "", loc),
            },
            ast::Expr::Rel(rel) => self.convert_rel(block, rel),
            unsupported => {
                self.note(format!("unsupported expression {unsupported:?}"));
                self.name_ref(block, "", loc)
            }
        }
    }

    fn convert_binary<'a>(
        &self,
        block: BlockRef<'c, 'a>,
        locals: &Locals<'c, 'a>,
        binary: &ast::BinaryExpr,
    ) -> Value<'c, 'a> {
        let loc = self.location(binary);
        let lhs = match binary.lhs() {
            Some(expr) => self.convert_expr(block, locals, &expr),
            None => self.name_ref(block, "", loc),
        };
        let rhs = match binary.rhs() {
            Some(expr) => self.convert_expr(block, locals, &expr),
            None => self.name_ref(block, "", loc),
        };
        let context = self.context;
        let cmp = |predicate| {
            yz::cmp(
                context,
                self.types.var,
                lhs,
                rhs,
                StringAttribute::new(context, predicate),
                loc,
            )
            .into()
        };
        // Operators without a yz op yet lower as calls by name, which is what
        // they are before the registry resolves them.
        let call = |callee| {
            yzl::call(
                context,
                self.types.var,
                &[lhs, rhs],
                FlatSymbolRefAttribute::new(context, callee),
                loc,
            )
            .into()
        };
        let operation = match binary.op() {
            Some(BinOp::Add) => yz::add(context, self.types.var, lhs, rhs, loc).into(),
            Some(BinOp::Sub) => yz::sub(context, self.types.var, lhs, rhs, loc).into(),
            Some(BinOp::Mul) => yz::mul(context, self.types.var, lhs, rhs, loc).into(),
            Some(BinOp::Div) => yz::div(context, self.types.var, lhs, rhs, loc).into(),
            Some(BinOp::And) => yz::and(context, self.types.var, lhs, rhs, loc).into(),
            Some(BinOp::Or) => yz::or(context, self.types.var, lhs, rhs, loc).into(),
            Some(BinOp::Eq) => cmp("eq"),
            Some(BinOp::Neq) => cmp("ne"),
            Some(BinOp::Lt) => cmp("lt"),
            Some(BinOp::Lte) => cmp("le"),
            Some(BinOp::Gt) => cmp("gt"),
            Some(BinOp::Gte) => cmp("ge"),
            Some(BinOp::Pow) => call("pow"),
            Some(BinOp::ShiftLeft) => call("shift_left"),
            Some(BinOp::ShiftRight) => call("shift_right"),
            Some(BinOp::In) => call("in"),
            Some(BinOp::NotIn) => {
                let contains = first_result(block.append_operation(call("in")));
                yz::not(context, self.types.var, contains, loc).into()
            }
            None => {
                self.note(format!("operator missing in {binary:?}"));
                return lhs;
            }
        };
        first_result(block.append_operation(operation))
    }

    fn convert_literal<'a>(
        &self,
        block: BlockRef<'c, 'a>,
        literal: &ast::Literal,
    ) -> Value<'c, 'a> {
        let loc = self.location(literal);
        let operation = match literal {
            ast::Literal::IntLiteral(int) => yz::constant_int(
                self.context,
                self.types.int64,
                IntegerAttribute::new(self.types.i64, int.value().unwrap_or_default() as i64),
                loc,
            )
            .into(),
            ast::Literal::FloatLiteral(float) => yz::constant_float(
                self.context,
                self.types.float64,
                FloatAttribute::new(
                    self.context,
                    Type::float64(self.context),
                    float.value().unwrap_or_default(),
                ),
                loc,
            )
            .into(),
            ast::Literal::BoolLiteral(boolean) => yz::constant_bool(
                self.context,
                self.types.boolean,
                Attribute::parse(
                    self.context,
                    if boolean.value().unwrap_or_default() {
                        "true"
                    } else {
                        "false"
                    },
                )
                .expect("a bool attribute parses"),
                loc,
            )
            .into(),
            ast::Literal::StringLiteral(string) => yz::constant_str(
                self.context,
                self.types.str,
                StringAttribute::new(self.context, &string.value().unwrap_or_default()),
                loc,
            )
            .into(),
        };
        first_result(block.append_operation(operation))
    }

    fn convert_rel<'a>(&self, block: BlockRef<'c, 'a>, rel: &ast::Rel) -> Value<'c, 'a> {
        let loc = self.location(rel);
        match rel {
            ast::Rel::FromExpr(from) => {
                let source = ident_text(from.relation()).unwrap_or_default();
                let value = first_result(
                    block.append_operation(
                        yzl::from(
                            self.context,
                            self.types.query,
                            FlatSymbolRefAttribute::new(self.context, &source),
                            loc,
                        )
                        .into(),
                    ),
                );
                match ident_text(from.alias()) {
                    Some(alias) => first_result(
                        block.append_operation(
                            yzl::alias(
                                self.context,
                                self.types.query,
                                value,
                                StringAttribute::new(self.context, &alias),
                                loc,
                            )
                            .into(),
                        ),
                    ),
                    None => value,
                }
            }
            ast::Rel::WhereExpr(stage) => {
                let input = self.convert_input(block, stage.input());
                let region = Region::new();
                let body = region.append_block(Block::new(&[]));
                let predicate = match stage.predicate() {
                    Some(expr) => self.convert_expr(body, &Locals::new(), &expr),
                    None => self.name_ref(body, "", loc),
                };
                body.append_operation(yzl::r#yield(self.context, &[predicate], loc).into());
                first_result(block.append_operation(
                    yzl::r#where(self.context, self.types.query, input, region, loc).into(),
                ))
            }
            ast::Rel::SelectExpr(stage) => {
                let input = self.convert_input(block, stage.input());
                let items = stage
                    .items()
                    .map(|item| (item.alias(), item.expr()))
                    .collect();
                let (names, region) = self.convert_items(items, loc);
                first_result(block.append_operation(
                    yzl::select(self.context, self.types.query, input, region, names, loc).into(),
                ))
            }
            ast::Rel::ExtendExpr(stage) => {
                let input = self.convert_input(block, stage.input());
                let items = stage
                    .items()
                    .map(|item| (item.alias(), item.expr()))
                    .collect();
                let (names, region) = self.convert_items(items, loc);
                first_result(block.append_operation(
                    yzl::extend(self.context, self.types.query, input, region, names, loc).into(),
                ))
            }
            ast::Rel::AggregateExpr(stage) => {
                let input = self.convert_input(block, stage.input());
                let group_by: Vec<Attribute> = stage
                    .group_by()
                    .into_iter()
                    .flat_map(|group| group.items().collect::<Vec<_>>())
                    .filter_map(|item| ident_text(item.column()))
                    .map(|name| StringAttribute::new(self.context, &name).into())
                    .collect();
                let items = stage
                    .items()
                    .map(|item| (item.alias(), item.expr()))
                    .collect();
                let (names, region) = self.convert_items(items, loc);
                first_result(
                    block.append_operation(
                        yzl::aggregate(
                            self.context,
                            self.types.query,
                            input,
                            region,
                            ArrayAttribute::new(self.context, &group_by),
                            names,
                            loc,
                        )
                        .into(),
                    ),
                )
            }
            ast::Rel::LimitExpr(stage) => {
                let input = self.convert_input(block, stage.input());
                let count = match stage.count() {
                    Some(ast::Expr::Literal(ast::Literal::IntLiteral(int))) => {
                        int.value().unwrap_or_default() as i64
                    }
                    other => {
                        self.note(format!("unsupported limit count {other:?}"));
                        0
                    }
                };
                first_result(
                    block.append_operation(
                        yzl::limit(
                            self.context,
                            self.types.query,
                            input,
                            IntegerAttribute::new(self.types.i64, count),
                            loc,
                        )
                        .into(),
                    ),
                )
            }
            ast::Rel::RenameExpr(stage) => {
                let input = self.convert_input(block, stage.input());
                let mut from = Vec::new();
                let mut to = Vec::new();
                for item in stage.items() {
                    let (Some(old), Some(new)) = (ident_text(item.from()), ident_text(item.to()))
                    else {
                        continue;
                    };
                    from.push(StringAttribute::new(self.context, &old).into());
                    to.push(StringAttribute::new(self.context, &new).into());
                }
                first_result(
                    block.append_operation(
                        yzl::rename(
                            self.context,
                            self.types.query,
                            input,
                            ArrayAttribute::new(self.context, &from),
                            ArrayAttribute::new(self.context, &to),
                            loc,
                        )
                        .into(),
                    ),
                )
            }
            ast::Rel::AliasExpr(stage) => {
                let input = self.convert_input(block, stage.input());
                let alias = ident_text(stage.alias()).unwrap_or_default();
                first_result(
                    block.append_operation(
                        yzl::alias(
                            self.context,
                            self.types.query,
                            input,
                            StringAttribute::new(self.context, &alias),
                            loc,
                        )
                        .into(),
                    ),
                )
            }
            ast::Rel::JoinExpr(stage) => {
                let lhs = self.convert_input(block, stage.input());
                let kind = match stage.kind() {
                    Some(ast::JoinKind::Left) => "left",
                    Some(ast::JoinKind::Right) => "right",
                    Some(ast::JoinKind::Full) => "full",
                    _ => "inner",
                };
                let rhs = ident_text(stage.relation()).unwrap_or_default();
                let on = Region::new();
                if let Some(condition) = stage.on().and_then(|on| on.condition()) {
                    let body = on.append_block(Block::new(&[]));
                    let value = self.convert_expr(body, &Locals::new(), &condition);
                    body.append_operation(yzl::r#yield(self.context, &[value], loc).into());
                }
                let mut builder = OperationBuilder::new("yzl.join", loc)
                    .add_operands(&[lhs])
                    .add_results(&[self.types.query])
                    .add_regions([on])
                    .add_attributes(&[
                        (
                            Identifier::new(self.context, "kind"),
                            StringAttribute::new(self.context, kind).into(),
                        ),
                        (
                            Identifier::new(self.context, "rhs"),
                            FlatSymbolRefAttribute::new(self.context, &rhs).into(),
                        ),
                    ]);
                if let Some(alias) = ident_text(stage.alias()) {
                    builder = builder.add_attributes(&[(
                        Identifier::new(self.context, "rhs_alias"),
                        StringAttribute::new(self.context, &alias).into(),
                    )]);
                }
                if let Some(using) = stage.using() {
                    let columns: Vec<Attribute> = using
                        .columns()
                        .filter_map(|column| column.text())
                        .map(|name| StringAttribute::new(self.context, &name).into())
                        .collect();
                    builder = builder.add_attributes(&[(
                        Identifier::new(self.context, "using_columns"),
                        ArrayAttribute::new(self.context, &columns).into(),
                    )]);
                }
                first_result(block.append_operation(builder.build().expect("yzl.join builds")))
            }
            ast::Rel::SetExpr(stage) => {
                let input = self.convert_input(block, stage.input());
                let items = stage
                    .items()
                    .map(|item| (item.column(), item.value()))
                    .collect();
                let (names, region) = self.convert_items(items, loc);
                first_result(block.append_operation(
                    yzl::set(self.context, self.types.query, input, region, names, loc).into(),
                ))
            }
            ast::Rel::DistinctExpr(stage) => {
                let input = self.convert_input(block, stage.input());
                first_result(block.append_operation(
                    yzl::distinct(self.context, self.types.query, input, loc).into(),
                ))
            }
            ast::Rel::DropExpr(stage) => {
                let input = self.convert_input(block, stage.input());
                let columns: Vec<Attribute> = stage
                    .columns()
                    .filter_map(|column| column.text())
                    .map(|name| StringAttribute::new(self.context, &name).into())
                    .collect();
                first_result(
                    block.append_operation(
                        yzl::drop(
                            self.context,
                            self.types.query,
                            input,
                            ArrayAttribute::new(self.context, &columns),
                            loc,
                        )
                        .into(),
                    ),
                )
            }
        }
    }

    /// A stage's input is another stage, or a bare name referring to a bound
    /// relation — which `from` covers until resolution decides what it was.
    fn convert_input<'a>(
        &self,
        block: BlockRef<'c, 'a>,
        input: Option<ast::Expr>,
    ) -> Value<'c, 'a> {
        match input {
            Some(ast::Expr::Rel(rel)) => self.convert_rel(block, &rel),
            Some(ast::Expr::IdentExpr(ident)) => {
                let name = ident_text(ident.name()).unwrap_or_default();
                let loc = self.location(&ident);
                first_result(
                    block.append_operation(
                        yzl::from(
                            self.context,
                            self.types.query,
                            FlatSymbolRefAttribute::new(self.context, &name),
                            loc,
                        )
                        .into(),
                    ),
                )
            }
            other => {
                self.note(format!("unsupported stage input {other:?}"));
                first_result(
                    block.append_operation(
                        yzl::from(
                            self.context,
                            self.types.query,
                            FlatSymbolRefAttribute::new(self.context, ""),
                            Location::unknown(self.context),
                        )
                        .into(),
                    ),
                )
            }
        }
    }

    fn convert_items(
        &self,
        items: Vec<(Option<ast::Ident>, Option<ast::Expr>)>,
        loc: Location<'c>,
    ) -> (ArrayAttribute<'c>, Region<'c>) {
        let region = Region::new();
        let body = region.append_block(Block::new(&[]));
        let mut names = Vec::new();
        let mut values = Vec::new();
        for (index, (alias, expr)) in items.into_iter().enumerate() {
            let name = ident_text(alias)
                .or_else(|| match &expr {
                    Some(ast::Expr::IdentExpr(ident)) => ident_text(ident.name()),
                    _ => None,
                })
                .unwrap_or_else(|| format!("column{index}"));
            names.push(StringAttribute::new(self.context, &name).into());
            let value = match &expr {
                Some(expr) => self.convert_expr(body, &Locals::new(), expr),
                None => self.name_ref(body, "", loc),
            };
            values.push(value);
        }
        body.append_operation(yzl::r#yield(self.context, &values, loc).into());
        (ArrayAttribute::new(self.context, &names), region)
    }

    fn name_ref<'a>(
        &self,
        block: BlockRef<'c, 'a>,
        name: &str,
        loc: Location<'c>,
    ) -> Value<'c, 'a> {
        first_result(
            block.append_operation(
                yzl::_name(
                    self.context,
                    self.types.var,
                    StringAttribute::new(self.context, name),
                    loc,
                )
                .into(),
            ),
        )
    }

    fn schema_type(&self, fields: impl Iterator<Item = ast::StructField>) -> Type<'c> {
        let columns: Vec<String> = fields
            .filter_map(|field| {
                let name = ident_text(field.name())?;
                Some(format!("{name}: {}", self.annotation_type(field.ty())))
            })
            .collect();
        Type::parse(self.context, &format!("!yzr.rel<{}>", columns.join(", ")))
            .expect("a schema type parses")
    }

    fn annotation_type(&self, annotation: Option<ast::TypeAnnotation>) -> &'static str {
        let name = annotation
            .and_then(|annotation| match annotation {
                ast::TypeAnnotation::NamedTypeAnnotation(named) => ident_text(named.name()),
                _ => None,
            })
            .unwrap_or_default();
        match name.as_str() {
            "int64" => "!yz.int64",
            "float64" => "!yz.float64",
            "bool" => "!yz.bool",
            "str" => "!yz.str",
            _ => "!yzl.var",
        }
    }
}

fn ident_text(ident: Option<ast::Ident>) -> Option<String> {
    ident.and_then(|ident| ident.text())
}

fn first_result<'c, 'a>(operation: melior::ir::operation::OperationRef<'c, 'a>) -> Value<'c, 'a> {
    operation
        .result(0)
        .expect("every converted op has one result")
        .into()
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use melior::ir::operation::OperationLike;

    fn converted(source: &str) -> String {
        let context = yuzu_mlir::context();
        let emission =
            super::convert_source(&context, "test.yz", source).expect("the source converts");
        assert!(
            emission.unsupported.is_empty(),
            "unsupported constructs: {:?}",
            emission.unsupported
        );
        assert!(
            emission.module.as_operation().verify(),
            "the converted module verifies"
        );
        emission.module.as_operation().to_string()
    }

    /// Every query the existing end-to-end suites compile must convert cleanly:
    /// no unsupported constructs, and a module that verifies.
    #[test]
    fn the_correctness_corpus_converts() {
        let corpus = concat!(env!("CARGO_MANIFEST_DIR"), "/../../python/tests");
        let mut sources = vec![std::fs::read_to_string(format!("{corpus}/support.py")).unwrap()];
        for entry in std::fs::read_dir(format!("{corpus}/correctness")).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|extension| extension == "py") {
                sources.push(std::fs::read_to_string(path).unwrap());
            }
        }

        let context = yuzu_mlir::context();
        let mut queries = 0;
        let mut failures = Vec::new();
        for source in &sources {
            for (index, chunk) in source.split(r#"""""#).enumerate() {
                // Odd chunks are the contents of triple-quoted strings; the
                // ones holding Yuzu source mention a pipe or a declaration.
                if index % 2 == 0 || !(chunk.contains("|>") || chunk.contains("struct ")) {
                    continue;
                }
                queries += 1;
                match super::convert_source(&context, "corpus.yz", chunk) {
                    Some(emission)
                        if emission.unsupported.is_empty()
                            && emission.module.as_operation().verify() => {}
                    Some(emission) => {
                        failures.push(format!("{chunk}\n  -> {:?}", emission.unsupported))
                    }
                    None => failures.push(format!("{chunk}\n  -> no root")),
                }
            }
        }
        assert!(
            queries > 30,
            "the corpus extraction found only {queries} queries"
        );
        assert!(
            failures.is_empty(),
            "{} of {queries} corpus queries failed to convert:\n{}",
            failures.len(),
            failures.join("\n---\n")
        );
    }

    #[test]
    fn converts_the_canonical_pipeline() {
        expect![[r#"
            module {
              yzl.struct @Row !yzr.rel<a: !yz.int64, b: !yz.int64>
              yzl.table @t of @Row
              %0 = yzl.from @t
              %1 = yzl.where %0 {
                %5 = yzl.name "a" : !yzl.var
                %6 = yz.constant_int 10
                %7 = yz.cmp "gt", %5, %6 : !yzl.var, !yz.int64 -> !yzl.var
                yzl.yield %7 : !yzl.var
              }
              %2 = yzl.extend %1 as ["e"] {
                %5 = yzl.name "a" : !yzl.var
                %6 = yzl.call @f(%5) : (!yzl.var) -> !yzl.var
                %7 = yzl.name "b" : !yzl.var
                %8 = yz.add %6, %7 : !yzl.var, !yzl.var -> !yzl.var
                yzl.yield %8 : !yzl.var
              }
              %3 = yzl.aggregate %2 group_by ["b"] as ["s"] {
                %5 = yzl.name "e" : !yzl.var
                %6 = yzl.call @sum(%5) : (!yzl.var) -> !yzl.var
                yzl.yield %6 : !yzl.var
              }
              %4 = yzl.limit %3, 10
              yzl.output %4
            }
        "#]]
        .assert_eq(&converted(
            r#"
struct Row { a: int64, b: int64 }
table t = Row

from t
|> where a > 10
|> extend f(a) + b as e
|> aggregate sum(e) as s group by b
|> limit 10
"#,
        ));
    }

    #[test]
    fn converts_declarations() {
        expect![[r#"
            module {
              yzl.fn @f params ["x"] (!yz.int64) -> !yz.int64 {
                %0 = yzl.name "x" : !yzl.var
                %1 = yz.constant_int 2
                %2 = yz.mul %0, %1 : !yzl.var, !yz.int64 -> !yzl.var
                %3 = yz.constant_int 1
                %4 = yz.add %2, %3 : !yzl.var, !yz.int64 -> !yzl.var
                yzl.return %4 : !yzl.var
              }
              yzl.fn @spread params ["x"] (!yz.int64) -> !yz.int64 agg {
                %0 = yzl.name "x" : !yzl.var
                %1 = yzl.call @max(%0) : (!yzl.var) -> !yzl.var
                %2 = yzl.name "x" : !yzl.var
                %3 = yzl.call @min(%2) : (!yzl.var) -> !yzl.var
                %4 = yz.sub %1, %3 : !yzl.var, !yzl.var -> !yzl.var
                yzl.return %4 : !yzl.var
              }
              yzl.fn @upper params ["s"] (!yz.str) -> !yz.str external {
              }
            }
        "#]]
        .assert_eq(&converted(
            r#"
fn f(x: int64) -> int64 {
    let doubled = x * 2
    return doubled + 1
}

agg fn spread(x: int64) -> int64 {
    return max(x) - min(x)
}

external fn upper(s: str) -> str
"#,
        ));
    }

    #[test]
    fn converts_joins_sets_and_membership() {
        expect![[r#"
            module {
              %0 = yzl.from @employees
              %1 = yzl.join "inner", %0, @departments as "d" {
                %6 = yzl.name "dept_id" : !yzl.var
                %7 = yzl.name "d.id" : !yzl.var
                %8 = yz.cmp "eq", %6, %7 : !yzl.var, !yzl.var -> !yzl.var
                yzl.yield %8 : !yzl.var
              }
              %2 = yzl.set %1 as ["level"] {
                %6 = yzl.name "level" : !yzl.var
                %7 = yz.constant_int 1
                %8 = yz.add %6, %7 : !yzl.var, !yz.int64 -> !yzl.var
                yzl.yield %8 : !yzl.var
              }
              %3 = yzl.where %2 {
                %6 = yzl.name "level" : !yzl.var
                %7 = yz.constant_int 1
                %8 = yz.constant_int 3
                %9 = yzl.list[%7, %8] : (!yz.int64, !yz.int64) -> !yzl.var
                %10 = yzl.call @in(%6, %9) : (!yzl.var, !yzl.var) -> !yzl.var
                yzl.yield %10 : !yzl.var
              }
              %4 = yzl.drop %3 ["rating"]
              %5 = yzl.distinct %4
              yzl.output %5
            }
        "#]]
        .assert_eq(&converted(
            r#"
from employees
|> inner join departments as d on dept_id == d.id
|> set level = level + 1
|> where level in [1, 3]
|> drop rating
|> distinct
"#,
        ));
    }

    #[test]
    fn converts_sugar_and_bindings() {
        expect![[r#"
            module {
              yzl.let @base {
                %4 = yzl.from @t
                %5 = yzl.where %4 {
                  %6 = yzl.name "active" : !yzl.var
                  yzl.yield %6 : !yzl.var
                }
                yzl.yield %5 : !yzl.query
              }
              %0 = yzl.from @base
              %1 = yzl.rename %0 from ["a"] to ["renamed"]
              %2 = yzl.alias %1 as "q"
              %3 = yzl.select %2 as ["out"] {
                %4 = yzl.name "q.renamed" : !yzl.var
                yzl.yield %4 : !yzl.var
              }
              yzl.output %3
            }
        "#]]
        .assert_eq(&converted(
            r#"
let base = from t |> where active

from base
|> rename a as renamed
|> as q
|> select q.renamed as out
"#,
        ));
    }
}
