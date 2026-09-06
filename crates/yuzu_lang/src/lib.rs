//! The AST → yzl emitter: the new pipeline's front door. Everything comes out
//! at the source level — names as `yzl.name`, unresolved types as `!yzl.var`,
//! sugar intact — for the checking passes to resolve.

use std::collections::HashMap;

use melior::Context;
use melior::ir::attribute::{
    ArrayAttribute, FlatSymbolRefAttribute, FloatAttribute, IntegerAttribute, StringAttribute,
    TypeAttribute,
};
use melior::ir::operation::OperationBuilder;
use melior::ir::r#type::IntegerType;
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

/// Parses the source and emits it as a yzl module, printing any parse
/// diagnostics to stderr. Returns `None` when the source has no root.
pub fn emit_source<'c>(context: &'c Context, name: &str, source: &str) -> Option<Module<'c>> {
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
    Some(Emitter::new(context, name, source).emit(&root))
}

struct Emitter<'c> {
    context: &'c Context,
    name: String,
    line_starts: Vec<usize>,
    var: Type<'c>,
    query: Type<'c>,
    int64: Type<'c>,
    float64: Type<'c>,
    boolean: Type<'c>,
    str: Type<'c>,
    i64: Type<'c>,
}

type Locals<'c, 'a> = HashMap<String, Value<'c, 'a>>;

impl<'c> Emitter<'c> {
    fn new(context: &'c Context, name: &str, source: &str) -> Self {
        let mut line_starts = vec![0];
        line_starts.extend(source.match_indices('\n').map(|(at, _)| at + 1));
        let parse = |text| Type::parse(context, text).expect("the dialect types parse");
        Self {
            context,
            name: name.to_string(),
            line_starts,
            var: parse("!yzl.var"),
            query: parse("!yzl.query"),
            int64: parse("!yz.int64"),
            float64: parse("!yz.float64"),
            boolean: parse("!yz.bool"),
            str: parse("!yz.str"),
            i64: IntegerType::new(context, 64).into(),
        }
    }

    fn location(&self, node: &impl AstNode) -> Location<'c> {
        let offset: usize = node.syntax().text_range().start().into();
        let line = self.line_starts.partition_point(|&start| start <= offset);
        let column = offset - self.line_starts[line - 1] + 1;
        Location::new(self.context, &self.name, line, column)
    }

    fn emit(&self, root: &ast::Root) -> Module<'c> {
        let module = Module::new(Location::new(self.context, &self.name, 1, 1));
        let top = module.body();
        for stmt in root.stmts() {
            self.emit_stmt(top, &stmt);
        }
        module
    }

    fn emit_stmt<'a>(&self, block: BlockRef<'c, 'a>, stmt: &ast::Stmt) {
        match stmt {
            ast::Stmt::StructStmt(decl) => self.emit_struct(block, decl),
            ast::Stmt::TableStmt(decl) => self.emit_table(block, decl),
            ast::Stmt::FuncStmt(decl) => self.emit_fn(block, decl),
            ast::Stmt::LetStmt(decl) => self.emit_let(block, decl),
            ast::Stmt::ExprStmt(stmt) => {
                if let Some(expr) = stmt.expr() {
                    self.emit_expr(block, &Locals::new(), &expr);
                }
            }
            unsupported => eprintln!("yuzu_lang: unsupported statement {unsupported:?}"),
        }
    }

    fn emit_struct<'a>(&self, block: BlockRef<'c, 'a>, decl: &ast::StructStmt) {
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

    fn emit_table<'a>(&self, block: BlockRef<'c, 'a>, decl: &ast::TableStmt) {
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

    fn emit_fn<'a>(&self, block: BlockRef<'c, 'a>, decl: &ast::FuncStmt) {
        let Some(name) = ident_text(decl.name()) else {
            return;
        };
        let params: Vec<Attribute> = decl
            .params()
            .filter_map(|param| ident_text(param.name()))
            .map(|name| StringAttribute::new(self.context, &name).into())
            .collect();

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
                        let value = self.emit_expr(entry, &locals, &expr);
                        locals.insert(name, value);
                    }
                    ast::Stmt::ReturnStmt(ret) => {
                        let values: Vec<Value> = ret
                            .expr()
                            .map(|expr| self.emit_expr(entry, &locals, &expr))
                            .into_iter()
                            .collect();
                        entry.append_operation(
                            yzl::r#return(self.context, &values, self.location(&ret)).into(),
                        );
                    }
                    unsupported => {
                        eprintln!("yuzu_lang: unsupported function statement {unsupported:?}")
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

    fn emit_let<'a>(&self, block: BlockRef<'c, 'a>, decl: &ast::LetStmt) {
        let (Some(name), Some(expr)) = (ident_text(decl.name()), decl.expr()) else {
            return;
        };
        let region = Region::new();
        let body = region.append_block(Block::new(&[]));
        let value = self.emit_expr(body, &Locals::new(), &expr);
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

    fn emit_expr<'a>(
        &self,
        block: BlockRef<'c, 'a>,
        locals: &Locals<'c, 'a>,
        expr: &ast::Expr,
    ) -> Value<'c, 'a> {
        let loc = self.location(expr);
        match expr {
            ast::Expr::Literal(literal) => self.emit_literal(block, literal),
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
            ast::Expr::BinaryExpr(binary) => self.emit_binary(block, locals, binary),
            ast::Expr::UnaryExpr(unary) => {
                let value = match unary.expr() {
                    Some(expr) => self.emit_expr(block, locals, &expr),
                    None => self.name_ref(block, "", loc),
                };
                let result = match unary.op() {
                    Some(UnaryOp::Neg) => yz::neg(self.context, self.var, value, loc).into(),
                    Some(UnaryOp::Not) => yz::not(self.context, self.var, value, loc).into(),
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
                    .map(|arg| self.emit_expr(block, locals, arg))
                    .collect();
                first_result(
                    block.append_operation(
                        yzl::call(
                            self.context,
                            self.var,
                            &operands,
                            FlatSymbolRefAttribute::new(self.context, &callee),
                            loc,
                        )
                        .into(),
                    ),
                )
            }
            ast::Expr::ParenExpr(paren) => match paren.expr() {
                Some(inner) => self.emit_expr(block, locals, &inner),
                None => self.name_ref(block, "", loc),
            },
            ast::Expr::Rel(rel) => self.emit_rel(block, rel),
            unsupported => {
                eprintln!("yuzu_lang: unsupported expression {unsupported:?}");
                self.name_ref(block, "", loc)
            }
        }
    }

    fn emit_binary<'a>(
        &self,
        block: BlockRef<'c, 'a>,
        locals: &Locals<'c, 'a>,
        binary: &ast::BinaryExpr,
    ) -> Value<'c, 'a> {
        let loc = self.location(binary);
        let lhs = match binary.lhs() {
            Some(expr) => self.emit_expr(block, locals, &expr),
            None => self.name_ref(block, "", loc),
        };
        let rhs = match binary.rhs() {
            Some(expr) => self.emit_expr(block, locals, &expr),
            None => self.name_ref(block, "", loc),
        };
        let context = self.context;
        let cmp = |predicate| {
            yz::cmp(
                context,
                self.var,
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
                self.var,
                &[lhs, rhs],
                FlatSymbolRefAttribute::new(context, callee),
                loc,
            )
            .into()
        };
        let operation = match binary.op() {
            Some(BinOp::Add) => yz::add(context, self.var, lhs, rhs, loc).into(),
            Some(BinOp::Sub) => yz::sub(context, self.var, lhs, rhs, loc).into(),
            Some(BinOp::Mul) => yz::mul(context, self.var, lhs, rhs, loc).into(),
            Some(BinOp::Div) => yz::div(context, self.var, lhs, rhs, loc).into(),
            Some(BinOp::And) => yz::and(context, self.var, lhs, rhs, loc).into(),
            Some(BinOp::Or) => yz::or(context, self.var, lhs, rhs, loc).into(),
            Some(BinOp::Eq) => cmp("eq"),
            Some(BinOp::Neq) => cmp("ne"),
            Some(BinOp::Lt) => cmp("lt"),
            Some(BinOp::Lte) => cmp("le"),
            Some(BinOp::Gt) => cmp("gt"),
            Some(BinOp::Gte) => cmp("ge"),
            Some(BinOp::Pow) => call("pow"),
            Some(BinOp::ShiftLeft) => call("shift_left"),
            Some(BinOp::ShiftRight) => call("shift_right"),
            Some(BinOp::In) | Some(BinOp::NotIn) | None => {
                eprintln!("yuzu_lang: unsupported operator in {binary:?}");
                return lhs;
            }
        };
        first_result(block.append_operation(operation))
    }

    fn emit_literal<'a>(&self, block: BlockRef<'c, 'a>, literal: &ast::Literal) -> Value<'c, 'a> {
        let loc = self.location(literal);
        let operation = match literal {
            ast::Literal::IntLiteral(int) => yz::constant_int(
                self.context,
                self.int64,
                IntegerAttribute::new(self.i64, int.value().unwrap_or_default() as i64),
                loc,
            )
            .into(),
            ast::Literal::FloatLiteral(float) => yz::constant_float(
                self.context,
                self.float64,
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
                self.boolean,
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
                self.str,
                StringAttribute::new(self.context, &string.value().unwrap_or_default()),
                loc,
            )
            .into(),
        };
        first_result(block.append_operation(operation))
    }

    fn emit_rel<'a>(&self, block: BlockRef<'c, 'a>, rel: &ast::Rel) -> Value<'c, 'a> {
        let loc = self.location(rel);
        match rel {
            ast::Rel::FromExpr(from) => {
                let source = ident_text(from.relation()).unwrap_or_default();
                let value = first_result(
                    block.append_operation(
                        yzl::from(
                            self.context,
                            self.query,
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
                                self.query,
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
                let input = self.emit_input(block, stage.input());
                let region = Region::new();
                let body = region.append_block(Block::new(&[]));
                let predicate = match stage.predicate() {
                    Some(expr) => self.emit_expr(body, &Locals::new(), &expr),
                    None => self.name_ref(body, "", loc),
                };
                body.append_operation(yzl::r#yield(self.context, &[predicate], loc).into());
                first_result(block.append_operation(
                    yzl::r#where(self.context, self.query, input, region, loc).into(),
                ))
            }
            ast::Rel::SelectExpr(stage) => {
                let input = self.emit_input(block, stage.input());
                let items = stage
                    .items()
                    .map(|item| (item.alias(), item.expr()))
                    .collect();
                let (names, region) = self.emit_items(items, loc);
                first_result(block.append_operation(
                    yzl::select(self.context, self.query, input, region, names, loc).into(),
                ))
            }
            ast::Rel::ExtendExpr(stage) => {
                let input = self.emit_input(block, stage.input());
                let items = stage
                    .items()
                    .map(|item| (item.alias(), item.expr()))
                    .collect();
                let (names, region) = self.emit_items(items, loc);
                first_result(block.append_operation(
                    yzl::extend(self.context, self.query, input, region, names, loc).into(),
                ))
            }
            ast::Rel::AggregateExpr(stage) => {
                let input = self.emit_input(block, stage.input());
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
                let (names, region) = self.emit_items(items, loc);
                first_result(
                    block.append_operation(
                        yzl::aggregate(
                            self.context,
                            self.query,
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
                let input = self.emit_input(block, stage.input());
                let count = match stage.count() {
                    Some(ast::Expr::Literal(ast::Literal::IntLiteral(int))) => {
                        int.value().unwrap_or_default() as i64
                    }
                    other => {
                        eprintln!("yuzu_lang: unsupported limit count {other:?}");
                        0
                    }
                };
                first_result(
                    block.append_operation(
                        yzl::limit(
                            self.context,
                            self.query,
                            input,
                            IntegerAttribute::new(self.i64, count),
                            loc,
                        )
                        .into(),
                    ),
                )
            }
            ast::Rel::RenameExpr(stage) => {
                let input = self.emit_input(block, stage.input());
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
                            self.query,
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
                let input = self.emit_input(block, stage.input());
                let alias = ident_text(stage.alias()).unwrap_or_default();
                first_result(
                    block.append_operation(
                        yzl::alias(
                            self.context,
                            self.query,
                            input,
                            StringAttribute::new(self.context, &alias),
                            loc,
                        )
                        .into(),
                    ),
                )
            }
            unsupported => {
                eprintln!("yuzu_lang: unsupported stage {unsupported:?}");
                first_result(
                    block.append_operation(
                        yzl::from(
                            self.context,
                            self.query,
                            FlatSymbolRefAttribute::new(self.context, ""),
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
    fn emit_input<'a>(&self, block: BlockRef<'c, 'a>, input: Option<ast::Expr>) -> Value<'c, 'a> {
        match input {
            Some(ast::Expr::Rel(rel)) => self.emit_rel(block, &rel),
            Some(ast::Expr::IdentExpr(ident)) => {
                let name = ident_text(ident.name()).unwrap_or_default();
                let loc = self.location(&ident);
                first_result(
                    block.append_operation(
                        yzl::from(
                            self.context,
                            self.query,
                            FlatSymbolRefAttribute::new(self.context, &name),
                            loc,
                        )
                        .into(),
                    ),
                )
            }
            other => {
                eprintln!("yuzu_lang: unsupported stage input {other:?}");
                first_result(
                    block.append_operation(
                        yzl::from(
                            self.context,
                            self.query,
                            FlatSymbolRefAttribute::new(self.context, ""),
                            Location::unknown(self.context),
                        )
                        .into(),
                    ),
                )
            }
        }
    }

    fn emit_items(
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
                Some(expr) => self.emit_expr(body, &Locals::new(), expr),
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
                    self.var,
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
        .expect("every emitted op has one result")
        .into()
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use melior::ir::operation::OperationLike;

    fn emitted(source: &str) -> String {
        let context = yuzu_mlir::context();
        let module = super::emit_source(&context, "test.yz", source).expect("the source emits");
        assert!(
            module.as_operation().verify(),
            "the emitted module verifies"
        );
        module.as_operation().to_string()
    }

    #[test]
    fn emits_the_canonical_pipeline() {
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
            }
        "#]]
        .assert_eq(&emitted(
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
    fn emits_declarations() {
        expect![[r#"
            module {
              yzl.fn @f params ["x"] {
                %0 = yzl.name "x" : !yzl.var
                %1 = yz.constant_int 2
                %2 = yz.mul %0, %1 : !yzl.var, !yz.int64 -> !yzl.var
                %3 = yz.constant_int 1
                %4 = yz.add %2, %3 : !yzl.var, !yz.int64 -> !yzl.var
                yzl.return %4 : !yzl.var
              }
              yzl.fn @spread params ["x"] agg {
                %0 = yzl.name "x" : !yzl.var
                %1 = yzl.call @max(%0) : (!yzl.var) -> !yzl.var
                %2 = yzl.name "x" : !yzl.var
                %3 = yzl.call @min(%2) : (!yzl.var) -> !yzl.var
                %4 = yz.sub %1, %3 : !yzl.var, !yzl.var -> !yzl.var
                yzl.return %4 : !yzl.var
              }
              yzl.fn @upper params ["s"] external {
              }
            }
        "#]]
        .assert_eq(&emitted(
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
    fn emits_sugar_and_bindings() {
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
            }
        "#]]
        .assert_eq(&emitted(
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
