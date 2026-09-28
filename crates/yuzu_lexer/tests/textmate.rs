//! Generates the keyword, operator and punctuation patterns of the VS Code
//! grammar from `TokenKind`. `UPDATE_EXPECT=1` writes them into the grammar.

use expect_test::expect_file;
use serde_json::{Value, json};
use yuzu_lexer::token_kind::TokenKind;

const GRAMMAR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../editors/vscode/syntaxes/yuzu.tmLanguage.json"
);

#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    Declaration,
    Modifier,
    Import,
    Control,
    Query,
    Logical,
    Pipe,
    Arrow,
    Bitwise,
    Arithmetic,
    Comparison,
    Assignment,
    Round,
    Curly,
    Square,
    Comma,
    Colon,
    Dot,
    HandWritten,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Entry {
    Keywords,
    Operators,
    Punctuation,
}

const KEYWORD_ROLES: [Role; 6] = [
    Role::Declaration,
    Role::Modifier,
    Role::Import,
    Role::Control,
    Role::Query,
    Role::Logical,
];

macro_rules! roles {
    ($($kind:ident => $role:ident,)*) => {
        const KINDS: &[TokenKind] = &[$(TokenKind::$kind),*];

        fn role(kind: TokenKind) -> Role {
            match kind {
                $(TokenKind::$kind => Role::$role,)*
            }
        }
    };
}

roles! {
    Plus => Arithmetic,
    Minus => Arithmetic,
    Star => Arithmetic,
    StarStar => Arithmetic,
    Slash => Arithmetic,
    Percent => Arithmetic,
    Eq => Assignment,
    EqEq => Comparison,
    Neq => Comparison,
    Lt => Comparison,
    Lte => Comparison,
    Gt => Comparison,
    Gte => Comparison,
    Shl => Bitwise,
    Shr => Bitwise,
    Arrow => Arrow,
    Pipe => Pipe,
    Dot => Dot,
    LeftParen => Round,
    RightParen => Round,
    LeftCurly => Curly,
    RightCurly => Curly,
    LeftSquare => Square,
    RightSquare => Square,
    Comma => Comma,
    Colon => Colon,
    AggKw => Modifier,
    AggregateKw => Query,
    AndKw => Logical,
    AsKw => Query,
    ByKw => Query,
    DefKw => Declaration,
    DistinctKw => Query,
    DropKw => Query,
    ExtendKw => Query,
    ExternalKw => Modifier,
    ForKw => Control,
    FromKw => Query,
    FullKw => Query,
    GroupKw => Query,
    ImplKw => Declaration,
    ImportKw => Import,
    InKw => Logical,
    InnerKw => Query,
    JoinKw => Query,
    LeftKw => Query,
    LetKw => Declaration,
    LimitKw => Query,
    ModKw => Declaration,
    MutKw => Modifier,
    NotKw => Logical,
    OffsetKw => Query,
    OnKw => Query,
    OrKw => Logical,
    PubKw => Modifier,
    RenameKw => Query,
    ReturnKw => Control,
    RightKw => Query,
    SelectKw => Query,
    SetKw => Query,
    StructKw => Declaration,
    TableKw => Declaration,
    TraitKw => Declaration,
    UsingKw => Query,
    WhereKw => Query,
    Identifier => HandWritten,
    BoolLit => HandWritten,
    IntLit => HandWritten,
    FloatLit => HandWritten,
    HexLit => HandWritten,
    BinaryLit => HandWritten,
    StringLit => HandWritten,
    RawStringLit => HandWritten,
    Comment => HandWritten,
    Newline => HandWritten,
    Whitespace => HandWritten,
    Error => HandWritten,
}

impl Role {
    fn entry(self) -> Option<Entry> {
        match self {
            Role::Declaration
            | Role::Modifier
            | Role::Import
            | Role::Control
            | Role::Query
            | Role::Logical => Some(Entry::Keywords),
            Role::Pipe
            | Role::Arrow
            | Role::Bitwise
            | Role::Arithmetic
            | Role::Comparison
            | Role::Assignment => Some(Entry::Operators),
            Role::Round | Role::Curly | Role::Square | Role::Comma | Role::Colon | Role::Dot => {
                Some(Entry::Punctuation)
            }
            Role::HandWritten => None,
        }
    }

    fn scope(self) -> &'static str {
        match self {
            Role::Declaration => "storage.type.yuzu",
            Role::Modifier => "storage.modifier.yuzu",
            Role::Import => "keyword.control.import.yuzu",
            Role::Control => "keyword.control.yuzu",
            Role::Query => "keyword.control.query.yuzu",
            Role::Logical => "keyword.operator.logical.yuzu",
            Role::Pipe => "keyword.operator.pipe.yuzu",
            Role::Arrow => "keyword.operator.arrow.yuzu",
            Role::Bitwise => "keyword.operator.bitwise.shift.yuzu",
            Role::Arithmetic => "keyword.operator.arithmetic.yuzu",
            Role::Comparison => "keyword.operator.comparison.yuzu",
            Role::Assignment => "keyword.operator.assignment.yuzu",
            Role::Round => "punctuation.brackets.round.yuzu",
            Role::Curly => "punctuation.brackets.curly.yuzu",
            Role::Square => "punctuation.brackets.square.yuzu",
            Role::Comma => "punctuation.separator.comma.yuzu",
            Role::Colon => "punctuation.separator.colon.yuzu",
            Role::Dot => "punctuation.accessor.dot.yuzu",
            Role::HandWritten => unreachable!("a hand-written token has no generated scope"),
        }
    }
}

impl Entry {
    fn key(self) -> &'static str {
        match self {
            Entry::Keywords => "keywords",
            Entry::Operators => "operators",
            Entry::Punctuation => "punctuation",
        }
    }
}

fn spellings(entry: Entry) -> Vec<(String, Role)> {
    KINDS
        .iter()
        .filter(|&&kind| role(kind).entry() == Some(entry))
        .map(|&kind| (kind.to_string(), role(kind)))
        .collect()
}

fn keyword_patterns() -> Vec<Value> {
    let keywords = spellings(Entry::Keywords);
    KEYWORD_ROLES
        .iter()
        .map(|&role| {
            let mut words: Vec<&str> = keywords
                .iter()
                .filter(|(_, r)| *r == role)
                .map(|(word, _)| word.as_str())
                .collect();
            words.sort_unstable();
            json!({
                "match": format!("\\b(?:{})\\b", words.join("|")),
                "name": role.scope(),
            })
        })
        .collect()
}

// A TextMate engine takes the first pattern that matches at a position, so
// the longer spellings come first: `<<` before `<`, `->` before `-`.
fn symbol_patterns(entry: Entry) -> Vec<Value> {
    let mut symbols = spellings(entry);
    symbols.sort_by(|(a, _), (b, _)| b.len().cmp(&a.len()).then(a.cmp(b)));
    symbols
        .iter()
        .map(|(symbol, role)| json!({ "match": escape(symbol), "name": role.scope() }))
        .collect()
}

fn escape(symbol: &str) -> String {
    symbol
        .chars()
        .flat_map(|c| {
            let special = "\\^$.|?*+()[]{}".contains(c);
            special.then_some('\\').into_iter().chain([c])
        })
        .collect()
}

#[test]
fn grammar_patterns_match_the_lexer() {
    let text = std::fs::read_to_string(GRAMMAR).unwrap();
    let mut grammar: Value = serde_json::from_str(&text).unwrap();

    let repository = &mut grammar["repository"];
    repository[Entry::Keywords.key()] = json!({ "patterns": keyword_patterns() });
    for entry in [Entry::Operators, Entry::Punctuation] {
        repository[entry.key()] = json!({ "patterns": symbol_patterns(entry) });
    }

    let mut generated = serde_json::to_string_pretty(&grammar).unwrap();
    generated.push('\n');
    expect_file!["../../../editors/vscode/syntaxes/yuzu.tmLanguage.json"].assert_eq(&generated);
}
