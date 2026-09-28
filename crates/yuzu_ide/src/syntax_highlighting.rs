//! Highlighting. What a name is follows from the node that holds it, so the
//! syntax tree answers for most names at once. A bare name in an expression
//! needs name resolution, so it waits for a check and takes the highlight of
//! the declaration it resolves to. A type name is a `Type` whether it is
//! built in or declared.

use std::ops::BitOr;

use text_size::TextRange;
use yuzu_ast::{self as ast, AstNode, Ident, Mutability};
use yuzu_diagnostics::source_map::SourceId;
use yuzu_syntax::{SyntaxKind, SyntaxNode, SyntaxToken};

use crate::Checked;
use crate::names::DeclarationKind;

/// A range of text, and its highlight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HlRange {
    pub range: TextRange,
    pub highlight: Highlight,
}

/// What a range is, and the details of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Highlight {
    pub tag: HlTag,
    pub mods: HlMods,
}

/// What a highlighted range is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HlTag {
    Keyword,
    Comment,
    StringLiteral,
    NumericLiteral,
    BoolLiteral,
    Module,
    Table,
    Struct,
    Trait,
    Function,
    Parameter,
    TypeParam,
    Type,
    Local,
    Field,
}

/// A detail of a highlight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HlMod {
    Declaration,
    Mutable,
}

/// A set of [`HlMod`]s.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HlMods(u32);

impl HlMod {
    /// Each modifier, in the order of its bit in [`HlMods`].
    pub const ALL: [HlMod; 2] = [HlMod::Declaration, HlMod::Mutable];

    fn mask(self) -> u32 {
        1 << self as u32
    }
}

impl HlMods {
    /// Whether the set holds `m`.
    #[must_use]
    pub fn contains(self, m: HlMod) -> bool {
        self.0 & m.mask() != 0
    }

    /// The modifiers in the set, in the order of [`HlMod::ALL`].
    pub fn iter(self) -> impl Iterator<Item = HlMod> {
        HlMod::ALL.into_iter().filter(move |&m| self.contains(m))
    }
}

impl From<HlTag> for Highlight {
    fn from(tag: HlTag) -> Self {
        Highlight {
            tag,
            mods: HlMods::default(),
        }
    }
}

impl BitOr<HlMod> for HlTag {
    type Output = Highlight;

    fn bitor(self, rhs: HlMod) -> Highlight {
        Highlight::from(self) | rhs
    }
}

impl BitOr<HlMod> for Highlight {
    type Output = Highlight;

    fn bitor(self, rhs: HlMod) -> Highlight {
        Highlight {
            tag: self.tag,
            mods: HlMods(self.mods.0 | rhs.mask()),
        }
    }
}

pub(crate) fn highlight(root: &SyntaxNode) -> Vec<HlRange> {
    root.descendants_with_tokens()
        .filter_map(rowan::NodeOrToken::into_token)
        .filter_map(|token| {
            Some(HlRange {
                range: token.text_range(),
                highlight: highlight_token(&token)?,
            })
        })
        .collect()
}

/// Each use a check resolved in a source, highlighted as its declaration.
pub(crate) fn highlight_uses(checked: &Checked, source: SourceId) -> Vec<HlRange> {
    checked
        .resolutions()
        .iter()
        .filter(|resolution| resolution.used.source == source)
        .map(|resolution| {
            let highlight = match resolution.kind {
                DeclarationKind::Parameter => HlTag::Parameter.into(),
                DeclarationKind::Let(Mutability::Mutable) => HlTag::Local | HlMod::Mutable,
                DeclarationKind::Let(Mutability::Immutable) => HlTag::Local.into(),
                DeclarationKind::Function => HlTag::Function.into(),
                DeclarationKind::Table => HlTag::Table.into(),
                DeclarationKind::Struct => HlTag::Struct.into(),
                DeclarationKind::Trait => HlTag::Trait.into(),
            };
            HlRange {
                range: resolution.used.range,
                highlight,
            }
        })
        .collect()
}

fn highlight_token(token: &SyntaxToken) -> Option<Highlight> {
    let kind = token.kind();
    if kind.is_keyword() {
        return Some(HlTag::Keyword.into());
    }

    let tag = match kind {
        SyntaxKind::Comment => HlTag::Comment,
        SyntaxKind::StringLit | SyntaxKind::RawStringLit => HlTag::StringLiteral,
        SyntaxKind::IntLit | SyntaxKind::FloatLit | SyntaxKind::HexLit | SyntaxKind::BinaryLit => {
            HlTag::NumericLiteral
        }
        SyntaxKind::BoolLit => HlTag::BoolLiteral,
        SyntaxKind::Identifier => return highlight_name(token),
        _ => return None,
    };
    Some(tag.into())
}

fn highlight_name(token: &SyntaxToken) -> Option<Highlight> {
    let ident = Ident::cast(token.parent()?)?;
    let parent = ident.syntax().parent()?;

    let highlight = match parent.kind() {
        SyntaxKind::FuncStmt => HlTag::Function | HlMod::Declaration,
        SyntaxKind::FuncParam => HlTag::Parameter | HlMod::Declaration,
        SyntaxKind::TypeParam => HlTag::TypeParam | HlMod::Declaration,
        SyntaxKind::TypeBound => HlTag::TypeParam.into(),
        SyntaxKind::TraitStmt => HlTag::Trait | HlMod::Declaration,
        SyntaxKind::TraitRef => HlTag::Trait.into(),
        SyntaxKind::StructStmt => HlTag::Struct | HlMod::Declaration,
        SyntaxKind::ImplStmt | SyntaxKind::NamedTypeAnnotation => HlTag::Type.into(),
        SyntaxKind::TableStmt if is_named_by(&parent, &ident, ast::TableStmt::name) => {
            HlTag::Table | HlMod::Declaration
        }
        SyntaxKind::StructExpr | SyntaxKind::TableStmt => HlTag::Struct.into(),
        SyntaxKind::LetStmt => highlight_let(&parent),
        SyntaxKind::ModStmt | SyntaxKind::ImportStmt => HlTag::Module | HlMod::Declaration,
        SyntaxKind::ModulePath => HlTag::Module.into(),
        SyntaxKind::IdentExpr if is_callee(&parent) => HlTag::Function.into(),
        SyntaxKind::FromSource if is_named_by(&parent, &ident, ast::FromSource::relation) => {
            HlTag::Table.into()
        }
        SyntaxKind::JoinStage if is_named_by(&parent, &ident, ast::JoinStage::relation) => {
            HlTag::Table.into()
        }
        SyntaxKind::FromSource | SyntaxKind::JoinStage | SyntaxKind::AliasStage => {
            HlTag::Local | HlMod::Declaration
        }
        SyntaxKind::RenameItem if is_named_by(&parent, &ident, ast::RenameItem::qualifier) => {
            HlTag::Local.into()
        }
        SyntaxKind::GroupByItem if is_named_by(&parent, &ident, ast::GroupByItem::qualifier) => {
            HlTag::Local.into()
        }
        SyntaxKind::RenameItem if is_named_by(&parent, &ident, ast::RenameItem::to) => {
            HlTag::Field | HlMod::Declaration
        }
        SyntaxKind::GroupByItem if is_named_by(&parent, &ident, ast::GroupByItem::alias) => {
            HlTag::Field | HlMod::Declaration
        }
        SyntaxKind::StructField | SyntaxKind::SelectItem | SyntaxKind::AggregateItem => {
            HlTag::Field | HlMod::Declaration
        }
        SyntaxKind::StructFieldInit
        | SyntaxKind::FieldAccessExpr
        | SyntaxKind::RenameItem
        | SyntaxKind::GroupByItem
        | SyntaxKind::DropStage
        | SyntaxKind::SetItem
        | SyntaxKind::JoinUsing => HlTag::Field.into(),
        _ => return None,
    };
    Some(highlight)
}

fn highlight_let(node: &SyntaxNode) -> Highlight {
    let declaration = HlTag::Local | HlMod::Declaration;
    match ast::LetStmt::cast(node.clone()).map(|stmt| stmt.mutability()) {
        Some(Mutability::Mutable) => declaration | HlMod::Mutable,
        Some(Mutability::Immutable) | None => declaration,
    }
}

fn is_callee(ident_expr: &SyntaxNode) -> bool {
    ident_expr
        .parent()
        .and_then(ast::CallExpr::cast)
        .and_then(|call| call.callee())
        .is_some_and(|callee| callee.syntax() == ident_expr)
}

/// Whether `accessor` names this identifier in the node that holds it.
fn is_named_by<N: AstNode>(
    parent: &SyntaxNode,
    ident: &Ident,
    accessor: impl Fn(&N) -> Option<Ident>,
) -> bool {
    N::cast(parent.clone())
        .and_then(|node| accessor(&node))
        .as_ref()
        == Some(ident)
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};

    use crate::HlTag;
    use crate::test_support::{FILE, analysis};

    fn check(text: &str, expected: &Expect) {
        let rendered: Vec<String> = analysis(text)
            .highlight(FILE)
            .unwrap()
            .iter()
            .filter(|range| range.highlight.tag != HlTag::Keyword)
            .map(|range| {
                let mods: Vec<String> = range
                    .highlight
                    .mods
                    .iter()
                    .map(|m| format!("{m:?}"))
                    .collect();
                let line = format!(
                    "{} {:?} {}",
                    &text[range.range],
                    range.highlight.tag,
                    mods.join(",")
                );
                line.trim_end().to_string()
            })
            .collect();
        expected.assert_eq(&rendered.join("\n"));
    }

    #[test]
    fn declarations() {
        check(
            r"
pub struct Point { x: float64 }
table employees = { id: str }
table staff = Point
trait Shape { def area(p: Point) -> float64 }
impl Shape for Point { def area(p: Point) -> float64 { return p.x } }
def classify[T](n: int64) -> List[int64] where T: Shape { let mut total = 0 }
mod helpers
import yuzu.std.math as m
",
            &expect![[r"
                Point Struct Declaration
                x Field Declaration
                float64 Type
                employees Table Declaration
                id Field Declaration
                str Type
                staff Table Declaration
                Point Struct
                Shape Trait Declaration
                area Function Declaration
                p Parameter Declaration
                Point Type
                float64 Type
                Shape Trait
                Point Type
                area Function Declaration
                p Parameter Declaration
                Point Type
                float64 Type
                x Field
                classify Function Declaration
                T TypeParam Declaration
                n Parameter Declaration
                int64 Type
                List Type
                int64 Type
                T TypeParam
                Shape Trait
                total Local Declaration,Mutable
                0 NumericLiteral
                helpers Module Declaration
                yuzu Module
                std Module
                math Module
                m Module Declaration"]],
        );
    }

    #[test]
    fn queries() {
        check(
            r"
from employees e
|> join departments d using (dept_id)
|> where e.salary >= max(1) // high
|> rename e.name as employee
|> aggregate count(e.id) as n group by d.name as dept
|> drop n
|> as summary
",
            &expect![[r"
                employees Table
                e Local Declaration
                departments Table
                d Local Declaration
                dept_id Field
                salary Field
                max Function
                1 NumericLiteral
                // high Comment
                e Local
                name Field
                employee Field Declaration
                count Function
                id Field
                n Field Declaration
                d Local
                name Field
                dept Field Declaration
                n Field
                summary Local Declaration"]],
        );
    }

    #[test]
    fn resolved_uses_take_their_declarations_highlight() {
        let text = "table t = { a: int64 }\nlet cap = 10\ndef f(x: int64) -> int64 {\n    let mut k = x\n    k = k + cap\n    return k\n}\nfrom t |> select f(a) as v\n";
        let (_tree, checked) = crate::test_support::checked(&[], text);
        let rendered: Vec<String> = checked
            .highlight_uses(crate::test_support::FILE)
            .iter()
            .map(|range| {
                let mods: Vec<String> = range
                    .highlight
                    .mods
                    .iter()
                    .map(|m| format!("{m:?}"))
                    .collect();
                format!(
                    "{} {:?} {}",
                    &text[range.range],
                    range.highlight.tag,
                    mods.join(",")
                )
                .trim_end()
                .to_string()
            })
            .collect();
        expect![[r"
            x Parameter
            k Local Mutable
            cap Local
            k Local Mutable
            t Table
            f Function"]]
        .assert_eq(&rendered.join("\n"));
    }
}
