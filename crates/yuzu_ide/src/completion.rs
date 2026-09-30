//! The names that fit where the cursor is.
//!
//! Where the cursor is, is read from the text as it is now, since the check
//! may be older than what was just typed. The check's names are found by
//! what was written before the edit: a file's names by the file, a stage's
//! columns by where the stage starts.

use text_size::TextSize;
use yuzu_ast::ast::{self, AstNode, Visibility};
use yuzu_diagnostics::SourceId;
use yuzu_driver::index::ScopeKind;
use yuzu_lexer::token_kind::TokenKind;
use yuzu_mlir::types;
use yuzu_syntax::{SyntaxKind, SyntaxNode, SyntaxToken};

use crate::Checked;

/// Where the cursor is, as the text around it says.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletionSite {
    /// Where the stage the cursor is in starts, when it is in one.
    pub stage: Option<TextSize>,
    /// The locals and parameters written before the cursor that it can read.
    pub locals: Vec<(String, CompletionKind)>,
    /// The keywords that can start what is written here.
    pub keywords: &'static [TokenKind],
    /// The declared names that fit here.
    pub expected: ExpectedNames,
}

/// The declared names that fit where the cursor is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExpectedNames {
    /// None: only keywords fit.
    Nothing,
    /// A function, a module-level `let` or a module.
    Value,
    /// A relation: after `from` or `join`.
    Relation,
    /// A type: after `->`, or after `:` or `[` in a type.
    Type,
    /// A trait: after the `:` of a bound.
    Trait,
    /// The names a module exports, after the name before the `.`.
    MemberOf(String),
    /// The names a module exports, in a `from .. import` of its path.
    ImportFrom(String),
}

impl CompletionSite {
    fn keywords_only(keywords: &'static [TokenKind]) -> Self {
        CompletionSite {
            stage: None,
            locals: Vec::new(),
            keywords,
            expected: ExpectedNames::Nothing,
        }
    }

    fn expecting(expected: ExpectedNames) -> Self {
        CompletionSite {
            expected,
            ..CompletionSite::keywords_only(&[])
        }
    }

    /// What fits here from the text alone: its keywords, the locals written
    /// before it, and the types every program has. A file not checked yet
    /// gets these.
    #[must_use]
    pub fn syntax_completions(&self) -> Vec<CompletionItem> {
        let mut items = Vec::new();
        self.add_syntax_items(&mut |label, kind| push_item(&mut items, label, kind));
        items
    }

    fn add_syntax_items(&self, add: &mut dyn FnMut(&str, CompletionKind)) {
        for keyword in self.keywords {
            add(&keyword.to_string(), CompletionKind::Keyword);
        }
        if self.expected == ExpectedNames::Type {
            for ty in types::scalar_spellings().chain([types::LIST]) {
                add(ty, CompletionKind::Type);
            }
        }
        for (name, kind) in self.locals.iter().rev() {
            add(name, *kind);
        }
    }
}

/// Adds an item unless one of its name is there already.
fn push_item(items: &mut Vec<CompletionItem>, label: &str, kind: CompletionKind) {
    if !items.iter().any(|item| item.label == label) {
        items.push(CompletionItem {
            label: label.to_owned(),
            kind,
        });
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

/// What a stage starts with after its `|>`.
const STAGE_KEYWORDS: &[TokenKind] = &[
    TokenKind::WhereKw,
    TokenKind::SelectKw,
    TokenKind::ExtendKw,
    TokenKind::AggregateKw,
    TokenKind::LimitKw,
    TokenKind::RenameKw,
    TokenKind::AsKw,
    TokenKind::JoinKw,
    TokenKind::LeftKw,
    TokenKind::RightKw,
    TokenKind::FullKw,
    TokenKind::SetKw,
    TokenKind::DistinctKw,
    TokenKind::DropKw,
];

/// What a statement at a file's top level starts with.
const FILE_KEYWORDS: &[TokenKind] = &[
    TokenKind::StructKw,
    TokenKind::TableKw,
    TokenKind::DefKw,
    TokenKind::AggKw,
    TokenKind::ExternalKw,
    TokenKind::LetKw,
    TokenKind::TraitKw,
    TokenKind::ImplKw,
    TokenKind::ImportKw,
    TokenKind::FromKw,
    TokenKind::ModKw,
    TokenKind::PubKw,
];

/// What a statement in a function body starts with.
const BODY_KEYWORDS: &[TokenKind] = &[TokenKind::LetKw, TokenKind::ReturnKw];

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

    if let Some(site) = before.as_ref().and_then(site_after) {
        return site;
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
        return CompletionSite::expecting(import.path().map_or(ExpectedNames::Nothing, |path| {
            ExpectedNames::ImportFrom(path.to_dotted())
        }));
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
    let expected = if stage.is_some() || body.is_some() || !starts_statement {
        ExpectedNames::Value
    } else {
        ExpectedNames::Nothing
    };
    CompletionSite {
        stage,
        locals,
        keywords,
        expected,
    }
}

/// The site right after `before`, when that token alone decides it.
fn site_after(before: &SyntaxToken) -> Option<CompletionSite> {
    match before.kind() {
        SyntaxKind::Pipe => Some(CompletionSite::keywords_only(STAGE_KEYWORDS)),
        SyntaxKind::Dot => {
            let base =
                previous_significant(before).filter(|base| base.kind() == SyntaxKind::Identifier);
            Some(CompletionSite::expecting(
                base.map_or(ExpectedNames::Nothing, |base| {
                    ExpectedNames::MemberOf(base.text().to_owned())
                }),
            ))
        }
        SyntaxKind::FromKw | SyntaxKind::JoinKw => {
            Some(CompletionSite::expecting(ExpectedNames::Relation))
        }
        SyntaxKind::Arrow => Some(CompletionSite::expecting(ExpectedNames::Type)),
        // A `:` or a `[` opens a type only in a type's syntax; in a struct
        // literal or a list it opens a value.
        SyntaxKind::Colon | SyntaxKind::LeftSquare => {
            match before.parent().map(|node| node.kind()) {
                Some(
                    SyntaxKind::FuncParam
                    | SyntaxKind::LetStmt
                    | SyntaxKind::StructField
                    | SyntaxKind::NamedTypeAnnotation,
                ) => Some(CompletionSite::expecting(ExpectedNames::Type)),
                Some(SyntaxKind::TypeBound) => {
                    Some(CompletionSite::expecting(ExpectedNames::Trait))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// What fits at `site`, in `source`, from the names `checked` found.
pub(crate) fn completions(
    checked: &Checked,
    source: SourceId,
    site: &CompletionSite,
) -> Vec<CompletionItem> {
    let mut items: Vec<CompletionItem> = Vec::new();
    let mut add = |label: &str, kind: CompletionKind| push_item(&mut items, label, kind);

    let index = checked.index();
    // A module's names: through a name that names it, or by its path.
    let module_file = match &site.expected {
        ExpectedNames::MemberOf(base) => index
            .scopes
            .get(&source)
            .into_iter()
            .flatten()
            .find(|name| name.name == *base && name.kind == ScopeKind::Module)
            .and_then(|name| name.module_file),
        ExpectedNames::ImportFrom(path) => index.modules.get(path).copied(),
        ExpectedNames::Nothing
        | ExpectedNames::Value
        | ExpectedNames::Relation
        | ExpectedNames::Type
        | ExpectedNames::Trait => None,
    };
    if matches!(
        site.expected,
        ExpectedNames::MemberOf(_) | ExpectedNames::ImportFrom(_)
    ) {
        for name in module_file
            .and_then(|file| index.scopes.get(&file))
            .into_iter()
            .flatten()
            .filter(|name| name.visibility == Visibility::Public)
        {
            add(&name.name, scope_kind(name.kind));
        }
        return items;
    }

    site.add_syntax_items(&mut add);
    let wanted = |kind: ScopeKind| {
        let expected = match kind {
            ScopeKind::Function | ScopeKind::Binding | ScopeKind::Module => ExpectedNames::Value,
            ScopeKind::Relation => ExpectedNames::Relation,
            ScopeKind::Struct => ExpectedNames::Type,
            ScopeKind::Trait => ExpectedNames::Trait,
        };
        site.expected == expected
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

    use crate::test_support::{FILE, analysis, at, checked, cursor};

    fn check(fixture: &str, expected: &Expect) {
        check_with(&[], fixture, expected);
    }

    fn check_with(files: &[(&str, &str)], fixture: &str, expected: &Expect) {
        let (text, offset) = cursor(fixture);
        let (_tree, checked) = checked(files, &text);
        let site = analysis(&text)
            .completion_site(at(offset))
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

    #[test]
    fn a_list_takes_values_not_types() {
        check(
            &PROGRAM.replacen("let y = x * 2", "let y = [$0x]", 1),
            &expect![[r"
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
    fn a_struct_literal_field_takes_a_value() {
        check(
            &format!("struct Row {{ a: int64 }}\n{PROGRAM}").replacen(
                "let y = x * 2",
                "let y = Row { a: $0x }",
                1,
            ),
            &expect![[r"
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
    fn a_bound_takes_the_traits() {
        check(
            "trait Numeric {\n    def zero(x: Self) -> Self\n}\ndef id[T](x: T) -> T where T: $0 { return x }\n",
            &expect!["Trait Numeric"],
        );
    }

    #[test]
    fn a_file_not_checked_yet_still_has_its_keywords() {
        let (text, offset) = cursor(&format!("{PROGRAM}from t |> $0\n"));
        let site = analysis(&text)
            .completion_site(at(offset))
            .expect("the file is open");
        let labels: Vec<String> = site
            .syntax_completions()
            .into_iter()
            .map(|item| item.label)
            .collect();
        assert!(labels.contains(&"where".to_owned()), "{labels:?}");
    }
}
