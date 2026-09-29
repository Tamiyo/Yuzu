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
    /// The name before the `.` the cursor is after: a module, whose
    /// exported names fit.
    pub member_of: Option<String>,
    /// The module a `from .. import` names, whose exported names fit.
    pub import_from: Option<String>,
    /// Whether a relation fits: after `from` or `join`.
    pub takes_relation: bool,
    /// Whether a type fits: after `:`, `->` or `[`.
    pub takes_type: bool,
}

impl CompletionSite {
    fn with_keywords(keywords: &'static [&'static str]) -> Self {
        CompletionSite {
            stage: None,
            locals: Vec::new(),
            keywords,
            takes_value: false,
            member_of: None,
            import_from: None,
            takes_relation: false,
            takes_type: false,
        }
    }
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
    Relation,
    Struct,
    Trait,
    /// A type the language has: a scalar, or `List`.
    Type,
}

/// The types every program has.
const BUILTIN_TYPES: &[&str] = &["int64", "float64", "bool", "str", "List"];

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

    let before_kind = before.as_ref().map(SyntaxToken::kind);
    match before_kind {
        Some(SyntaxKind::Pipe) => return CompletionSite::with_keywords(STAGE_KEYWORDS),
        Some(SyntaxKind::Dot) => {
            let base = before
                .as_ref()
                .and_then(previous_significant)
                .filter(|base| base.kind() == SyntaxKind::Identifier);
            return CompletionSite {
                member_of: base.map(|base| base.text().to_owned()),
                ..CompletionSite::with_keywords(&[])
            };
        }
        Some(SyntaxKind::FromKw | SyntaxKind::JoinKw) => {
            return CompletionSite {
                takes_relation: true,
                ..CompletionSite::with_keywords(&[])
            };
        }
        Some(SyntaxKind::Colon | SyntaxKind::Arrow | SyntaxKind::LeftSquare) => {
            return CompletionSite {
                takes_type: true,
                ..CompletionSite::with_keywords(&[])
            };
        }
        _ => {}
    }

    let anchor = typed
        .as_ref()
        .and_then(SyntaxToken::parent)
        .or_else(|| before.as_ref().and_then(SyntaxToken::parent))
        .unwrap_or_else(|| root.clone());
    if let Some(import) = anchor.ancestors().find_map(ast::FromImportStmt::cast)
        && let Some(keyword) = import
            .syntax()
            .children_with_tokens()
            .find(|element| element.kind() == SyntaxKind::ImportKw)
        && keyword.text_range().end() <= offset
    {
        return CompletionSite {
            import_from: import.path().map(|path| path.to_dotted()),
            ..CompletionSite::with_keywords(&[])
        };
    }

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
        takes_value: stage.is_some() || body.is_some() || !starts_statement,
        ..CompletionSite::with_keywords(keywords)
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

    let index = checked.index();
    // A module's names: through a name that names it, or by its path.
    let module_file = site
        .member_of
        .as_deref()
        .and_then(|base| {
            index
                .scopes
                .get(&source)?
                .iter()
                .find(|name| name.name == base && name.kind == ScopeKind::Module)?
                .module_file
        })
        .or_else(|| index.modules.get(site.import_from.as_deref()?).copied());
    if site.member_of.is_some() || site.import_from.is_some() {
        for name in module_file
            .and_then(|file| index.scopes.get(&file))
            .into_iter()
            .flatten()
            .filter(|name| name.is_exported)
        {
            add(&name.name, scope_kind(name.kind));
        }
        return items;
    }

    for keyword in site.keywords {
        add(keyword, CompletionKind::Keyword);
    }
    if site.takes_type {
        for ty in BUILTIN_TYPES {
            add(ty, CompletionKind::Type);
        }
    }
    let wanted = |kind: ScopeKind| match kind {
        ScopeKind::Function | ScopeKind::Binding | ScopeKind::Module => site.takes_value,
        ScopeKind::Relation => site.takes_relation,
        ScopeKind::Struct => site.takes_type,
        ScopeKind::Trait => false,
    };
    if let Some(stage) = site.stage
        && let Some(row) = index
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
    for name in index.scopes.get(&source).into_iter().flatten() {
        if wanted(name.kind) {
            add(&name.name, scope_kind(name.kind));
        }
    }
    items
}

fn scope_kind(kind: ScopeKind) -> CompletionKind {
    match kind {
        ScopeKind::Function => CompletionKind::Function,
        ScopeKind::Binding => CompletionKind::Binding,
        ScopeKind::Module => CompletionKind::Module,
        ScopeKind::Relation => CompletionKind::Relation,
        ScopeKind::Struct => CompletionKind::Struct,
        ScopeKind::Trait => CompletionKind::Trait,
    }
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
        check_with(&[], fixture, expected);
    }

    fn check_with(files: &[(&str, &str)], fixture: &str, expected: &Expect) {
        let (text, offset) = cursor(fixture);
        let (_tree, checked) = checked(files, &text);
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

    #[test]
    fn after_a_module_and_a_dot_its_exported_names() {
        check_with(
            &[(
                "helpers.yz",
                "pub def two() -> int64 { return 2 }\ndef hidden() -> int64 { return 0 }\n",
            )],
            "import helpers as h\ntable t = { a: int64 }\nfrom t |> select h.$0 as v\n",
            &expect!["Function two"],
        );
    }

    #[test]
    fn an_import_item_takes_the_module_names() {
        check_with(
            &[(
                "helpers.yz",
                "pub def two() -> int64 { return 2 }\ndef hidden() -> int64 { return 0 }\n",
            )],
            "from helpers import $0\n",
            &expect!["Function two"],
        );
    }

    #[test]
    fn after_from_the_relations() {
        check(&format!("{PROGRAM}from $0\n"), &expect!["Relation t"]);
    }

    #[test]
    fn after_a_colon_the_types() {
        check(
            &format!("struct Row {{ a: int64 }}\n{PROGRAM}def f(x: $0) -> int64 {{ return 1 }}\n"),
            &expect![[r"
                Type int64
                Type float64
                Type bool
                Type str
                Type List
                Struct Row"]],
        );
    }
}
