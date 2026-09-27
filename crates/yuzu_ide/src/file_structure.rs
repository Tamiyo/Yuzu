//! The declarations of a file as a tree, for an outline. A node names its
//! parent by index, so the list is flat and the parent comes first.

use text_size::TextRange;
use yuzu_ast::{self as ast, AstNode};
use yuzu_syntax::{SyntaxKind, SyntaxNode};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructureNode {
    pub parent: Option<usize>,
    pub label: String,
    pub navigation_range: TextRange,
    pub node_range: TextRange,
    pub kind: StructureNodeKind,
    pub detail: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructureNodeKind {
    Module,
    Function,
    Struct,
    Table,
    Trait,
    Impl,
    Field,
    Constant,
    Query,
}

pub(crate) fn file_structure(root: &ast::Root) -> Vec<StructureNode> {
    let mut structure = Structure { nodes: Vec::new() };
    for stmt in root.stmts() {
        structure.read_stmt(&stmt, None);
    }
    structure.nodes
}

struct Structure {
    nodes: Vec<StructureNode>,
}

impl Structure {
    fn read_stmt(&mut self, stmt: &ast::Stmt, parent: Option<usize>) {
        match stmt {
            ast::Stmt::FuncStmt(func) => self.read_func(func, parent),
            ast::Stmt::StructStmt(struct_) => {
                let at = self.push(parent, struct_, struct_.name(), StructureNodeKind::Struct);
                self.read_fields(struct_.fields(), at);
            }
            ast::Stmt::TableStmt(table) => {
                let at = self.push(parent, table, table.name(), StructureNodeKind::Table);
                self.read_fields(table.inline_fields(), at);
            }
            ast::Stmt::TraitStmt(trait_) => {
                let at = self.push(parent, trait_, trait_.name(), StructureNodeKind::Trait);
                for method in trait_.methods() {
                    self.read_func(&method, at);
                }
            }
            ast::Stmt::ImplStmt(impl_) => self.read_impl(impl_, parent),
            ast::Stmt::LetStmt(let_) => {
                self.push(parent, let_, let_.name(), StructureNodeKind::Constant);
            }
            ast::Stmt::ModStmt(mod_) => {
                self.push(parent, mod_, mod_.name(), StructureNodeKind::Module);
            }
            ast::Stmt::ExprStmt(expr) => self.read_query(expr, parent),
            ast::Stmt::BlockStmt(_)
            | ast::Stmt::AssignStmt(_)
            | ast::Stmt::ReturnStmt(_)
            | ast::Stmt::ImportStmt(_)
            | ast::Stmt::FromImportStmt(_) => {}
        }
    }

    fn read_func(&mut self, func: &ast::FuncStmt, parent: Option<usize>) {
        let Some(at) = self.push(parent, func, func.name(), StructureNodeKind::Function) else {
            return;
        };
        self.nodes[at].detail = signature(func);
    }

    fn read_fields(
        &mut self,
        fields: impl Iterator<Item = ast::StructField>,
        parent: Option<usize>,
    ) {
        if parent.is_none() {
            return;
        }
        for field in fields {
            let Some(at) = self.push(parent, &field, field.name(), StructureNodeKind::Field) else {
                continue;
            };
            self.nodes[at].detail = field
                .ty()
                .map(|ty| compact(&ty.syntax().text().to_string()));
        }
    }

    fn read_impl(&mut self, impl_: &ast::ImplStmt, parent: Option<usize>) {
        let Some((ty, navigation_range)) = read_name(impl_.ty()) else {
            return;
        };
        let trait_ = impl_.trait_().and_then(|trait_| read_name(trait_.name()));
        let label = match trait_ {
            Some((trait_, _)) => format!("impl {trait_} for {ty}"),
            None => format!("impl {ty}"),
        };
        let at = self.push_labeled(
            parent,
            impl_.syntax(),
            label,
            navigation_range,
            StructureNodeKind::Impl,
        );
        for method in impl_.methods() {
            self.read_func(&method, Some(at));
        }
    }

    /// A query statement is named by the relation it starts from.
    fn read_query(&mut self, expr: &ast::ExprStmt, parent: Option<usize>) {
        let Some(from) = expr.syntax().descendants().find_map(ast::FromExpr::cast) else {
            return;
        };
        let Some((relation, navigation_range)) = read_name(from.relation()) else {
            return;
        };
        let label = format!("from {relation}");
        self.push_labeled(
            parent,
            expr.syntax(),
            label,
            navigation_range,
            StructureNodeKind::Query,
        );
    }

    /// A declaration without a name has nothing to show, so it is left out.
    fn push(
        &mut self,
        parent: Option<usize>,
        node: &impl AstNode,
        name: Option<ast::Ident>,
        kind: StructureNodeKind,
    ) -> Option<usize> {
        let (label, navigation_range) = read_name(name)?;
        Some(self.push_labeled(parent, node.syntax(), label, navigation_range, kind))
    }

    fn push_labeled(
        &mut self,
        parent: Option<usize>,
        node: &SyntaxNode,
        label: String,
        navigation_range: TextRange,
        kind: StructureNodeKind,
    ) -> usize {
        self.nodes.push(StructureNode {
            parent,
            label,
            navigation_range,
            node_range: node.text_range(),
            kind,
            detail: None,
        });
        self.nodes.len() - 1
    }
}

/// A name and its range. Parse recovery can leave an `Ident` node that
/// holds an error token instead of an identifier, and that is no name.
fn read_name(ident: Option<ast::Ident>) -> Option<(String, TextRange)> {
    let token = ident?.token()?;
    (token.kind() == SyntaxKind::Identifier).then(|| (token.text().to_owned(), token.text_range()))
}

/// The text from the parameters to the body: `(n: int64) -> int64`.
fn signature(func: &ast::FuncStmt) -> Option<String> {
    let syntax = func.syntax();
    let start = syntax
        .children_with_tokens()
        .find(|element| element.kind() == SyntaxKind::LeftParen)?
        .text_range()
        .start();
    let end = func.body().map_or(syntax.text_range().end(), |body| {
        body.syntax().text_range().start()
    });
    let range = TextRange::new(start, end.max(start)) - syntax.text_range().start();
    let text = syntax.text().to_string();
    Some(compact(&text[range]))
}

fn compact(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};

    use crate::test_support::{FILE, analysis};

    fn check(text: &str, expected: Expect) {
        let nodes = analysis(text).file_structure(FILE).unwrap();
        let rendered: Vec<String> = nodes
            .iter()
            .map(|node| {
                let depth = std::iter::successors(node.parent, |&at| nodes[at].parent).count();
                format!(
                    "{}{:?} {} {}",
                    "  ".repeat(depth),
                    node.kind,
                    node.label,
                    node.detail.as_deref().unwrap_or("")
                )
                .trim_end()
                .to_string()
            })
            .collect();
        expected.assert_eq(&rendered.join("\n"));
    }

    #[test]
    fn declarations_and_queries() {
        check(
            r#"
mod helpers
struct Point { x: float64, y: List[int64] }
table employees = { id: str }
trait Shape { def area(p: Point) -> float64 }
impl Shape for Point {
    def area(p: Point) -> float64 { return p.x }
}
agg def spread(x: int64) -> int64 { return max(x) - min(x) }
let cap = 10
from employees e |> select e.id
"#,
            expect![[r#"
                Module helpers
                Struct Point
                  Field x float64
                  Field y List[int64]
                Table employees
                  Field id str
                Trait Shape
                  Function area (p: Point) -> float64
                Impl impl Shape for Point
                  Function area (p: Point) -> float64
                Function spread (x: int64) -> int64
                Constant cap
                Query from employees"#]],
        );
    }

    #[test]
    fn a_name_lost_to_parse_recovery_is_left_out() {
        check("let = 1\n", expect![[r#""#]]);
    }
}
