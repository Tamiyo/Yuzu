//! The names that fit where the cursor is.
//!
//! Where the cursor is, is read from the text as it is now, since the check
//! may be older than what was just typed. The check's names are found by
//! what was written before the edit: a file's names by the file, a stage's
//! columns by where the stage starts.

use text_size::TextSize;
use yuzu_ast::ast::{self, AstNode};
use yuzu_diagnostics::SourceId;
use yuzu_driver::index::ScopeKind;
use yuzu_syntax::{SyntaxKind, SyntaxNode, SyntaxToken};

use crate::Checked;

/// Where the cursor is, as the text around it says.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletionSite {
    /// Where the stage the cursor is in starts, when it is in one.
    pub stage: Option<TextSize>,
    /// The locals and parameters written before the cursor that it can read.
    pub locals: Vec<(String, CompletionKind)>,
    pub keywords: &'static [&'static str],
    /// Whether a function or a module-level `let` fits here.
    pub takes_value: bool,
}

/// A name that fits, and what it is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletionItem {
    pub label: String,
    pub kind: CompletionKind,
}

/// What a completion names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompletionKind {
    Keyword,
    Column,
    Local,
    Parameter,
    Function,
    /// A module-level `let`.
    Binding,
    Module,
}

/// What a stage starts with after its `|>`.
const STAGE_KEYWORDS: &[&str] = &[
    "where",
    "select",
    "extend",
    "aggregate",
    "limit",
    "rename",
    "as",
    "join",
    "left",
    "right",
    "full",
    "set",
    "distinct",
    "drop",
];

/// What a statement at a file's top level starts with.
const FILE_KEYWORDS: &[&str] = &[
    "struct", "table", "def", "agg", "external", "let", "trait", "impl", "import", "from", "mod",
    "pub",
];

/// What a statement in a function body starts with.
const BODY_KEYWORDS: &[&str] = &["let", "return"];

pub(crate) fn completion_site(root: &SyntaxNode, offset: TextSize) -> CompletionSite {
    let token = root.token_at_offset(offset).left_biased();
    let typed = token
        .clone()
        .filter(|token| token.kind() == SyntaxKind::Identifier);
    let before = match &typed {
        Some(typed) => previous_significant(typed),
        None => token.and_then(|token| {
            if token.kind().is_trivia() {
                previous_significant(&token)
            } else {
                Some(token)
            }
        }),
    };

    if before
        .as_ref()
        .is_some_and(|before| before.kind() == SyntaxKind::Pipe)
    {
        return CompletionSite {
            stage: None,
            locals: Vec::new(),
            keywords: STAGE_KEYWORDS,
            takes_value: false,
        };
    }

    let anchor = typed
        .as_ref()
        .and_then(SyntaxToken::parent)
        .or_else(|| before.as_ref().and_then(SyntaxToken::parent))
        .unwrap_or_else(|| root.clone());
    let stage = anchor
        .ancestors()
        .find_map(ast::Stage::cast)
        .map(|stage| stage.syntax().text_range().start());
    let body = anchor.ancestors().find_map(ast::FuncStmt::cast);

    let starts_statement = typed.as_ref().is_some_and(starts_statement);
    let keywords = match (stage, &body, starts_statement) {
        (None, Some(_), true) => BODY_KEYWORDS,
        (None, None, true) => FILE_KEYWORDS,
        _ => &[],
    };
    // A stage's expressions are isolated from the function around them.
    let locals = match (stage, &body) {
        (None, Some(body)) => locals_before(body, offset),
        _ => Vec::new(),
    };
    CompletionSite {
        stage,
        locals,
        keywords,
        takes_value: stage.is_some() || body.is_some() || !starts_statement,
    }
}

/// What fits at `site`, in `source`, from the names `checked` found.
pub(crate) fn completions(
    checked: &Checked,
    source: SourceId,
    site: &CompletionSite,
) -> Vec<CompletionItem> {
    let mut items: Vec<CompletionItem> = Vec::new();
    let mut add = |label: &str, kind: CompletionKind| {
        if !items.iter().any(|item| item.label == label) {
            items.push(CompletionItem {
                label: label.to_owned(),
                kind,
            });
        }
    };

    for keyword in site.keywords {
        add(keyword, CompletionKind::Keyword);
    }
    if let Some(stage) = site.stage
        && let Some(row) = checked
            .index()
            .rows
            .iter()
            .find(|row| row.stage.source_id == source && row.stage.range.start() == stage)
    {
        for column in &row.columns {
            add(column, CompletionKind::Column);
        }
    }
    for (name, kind) in site.locals.iter().rev() {
        add(name, *kind);
    }
    if site.takes_value
        && let Some(scope) = checked.index().scopes.get(&source)
    {
        for name in scope {
            let kind = match name.kind {
                ScopeKind::Function => CompletionKind::Function,
                ScopeKind::Binding => CompletionKind::Binding,
                ScopeKind::Module => CompletionKind::Module,
                ScopeKind::Struct | ScopeKind::Relation | ScopeKind::Trait => continue,
            };
            add(&name.name, kind);
        }
    }
    items
}

fn previous_significant(token: &SyntaxToken) -> Option<SyntaxToken> {
    std::iter::successors(token.prev_token(), SyntaxToken::prev_token)
        .find(|token| !token.kind().is_trivia())
}

/// Whether the name being typed is the first thing in its statement.
fn starts_statement(typed: &SyntaxToken) -> bool {
    typed
        .parent_ancestors()
        .find(|node| {
            node.parent().is_some_and(|parent| {
                matches!(parent.kind(), SyntaxKind::Root | SyntaxKind::BlockStmt)
            })
        })
        .is_some_and(|statement| statement.text_range().start() == typed.text_range().start())
}

/// The function's parameters, and each `let` written before `offset` in a
/// block around it.
fn locals_before(func: &ast::FuncStmt, offset: TextSize) -> Vec<(String, CompletionKind)> {
    let mut locals: Vec<(String, CompletionKind)> = func
        .params()
        .filter_map(|param| {
            Some((
                param.name()?.token()?.text().to_owned(),
                CompletionKind::Parameter,
            ))
        })
        .collect();
    let Some(body) = func.body() else {
        return locals;
    };
    let blocks = body
        .syntax()
        .descendants()
        .filter_map(ast::BlockStmt::cast)
        .filter(|block| block.syntax().text_range().contains_inclusive(offset));
    for block in blocks {
        for stmt in block.stmts() {
            if let ast::Stmt::LetStmt(binding) = stmt
                && binding.syntax().text_range().end() <= offset
                && let Some(name) = binding.name().and_then(|name| name.token())
            {
                locals.push((name.text().to_owned(), CompletionKind::Local));
            }
        }
    }
    locals
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};

    use crate::FilePosition;
    use crate::test_support::{FILE, analysis, checked, cursor};

    fn check(fixture: &str, expected: &Expect) {
        let (text, offset) = cursor(fixture);
        let (_tree, checked) = checked(&[], &text);
        let site = analysis(&text)
            .completion_site(FilePosition {
                file_id: FILE,
                offset,
            })
            .expect("the file is open");
        let rendered: Vec<String> = checked
            .completions(FILE, &site)
            .iter()
            .map(|item| format!("{:?} {}", item.kind, item.label))
            .collect();
        expected.assert_eq(&rendered.join("\n"));
    }

    const PROGRAM: &str = "table t = { a: int64, b: int64 }\nlet cap = 10\ndef double(x: int64) -> int64 {\n    let y = x * 2\n    return y\n}\n";

    #[test]
    fn a_stage_offers_its_columns_and_the_module_values() {
        check(
            &format!("{PROGRAM}from t |> where $0a > 1\n"),
            &expect![[r"
                Column a
                Column b
                Binding cap
                Function double
                Binding ENGINE
                Function avg
                Function count
                Function count_distinct
                Function max
                Function min
                Function pow
                Function shift_left
                Function shift_right
                Function sum"]],
        );
    }

    #[test]
    fn after_a_pipe_the_stages() {
        check(
            &format!("{PROGRAM}from t |> $0\n"),
            &expect![[r"
            Keyword where
            Keyword select
            Keyword extend
            Keyword aggregate
            Keyword limit
            Keyword rename
            Keyword as
            Keyword join
            Keyword left
            Keyword right
            Keyword full
            Keyword set
            Keyword distinct
            Keyword drop"]],
        );
    }

    #[test]
    fn a_body_offers_its_locals_before_the_cursor() {
        check(
            &PROGRAM.replacen("return y", "return $0y", 1),
            &expect![[r"
                Local y
                Parameter x
                Binding cap
                Function double
                Binding ENGINE
                Function avg
                Function count
                Function count_distinct
                Function max
                Function min
                Function pow
                Function shift_left
                Function shift_right
                Function sum"]],
        );
    }

    #[test]
    fn a_statement_at_the_top_offers_the_declarations() {
        check(
            &format!("{PROGRAM}s$0\n"),
            &expect![[r"
            Keyword struct
            Keyword table
            Keyword def
            Keyword agg
            Keyword external
            Keyword let
            Keyword trait
            Keyword impl
            Keyword import
            Keyword from
            Keyword mod
            Keyword pub"]],
        );
    }
}
