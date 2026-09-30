//! What a call's function takes, while its arguments are being written.
//!
//! The call is found in the text as it is now, since the check that
//! resolves its function may be older than the `(` just typed; the
//! function's name was written before, so the check has it.

use text_size::{TextRange, TextSize};
use yuzu_ast::ast::{self, AstNode};
use yuzu_diagnostics::SourceId;
use yuzu_syntax::{SyntaxKind, SyntaxNode};

use crate::Checked;
use crate::file_structure;
use crate::names::{DeclarationKind, node_at};

/// A call around a position: where its function's name is written, and
/// which of its arguments the position is in, counted from zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CallSite {
    pub callee: TextRange,
    pub argument: usize,
}

/// The overloads a call may mean, and the one and the parameter the
/// position is at.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignatureHelp {
    pub signatures: Vec<Signature>,
    pub active_signature: usize,
    pub active_parameter: usize,
}

/// One overload: its header, and where each parameter is in it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signature {
    pub label: String,
    pub parameters: Vec<TextRange>,
}

/// The innermost call whose argument list holds `offset`: after its `(`,
/// and not after its `)`.
pub(crate) fn call_at(root: &SyntaxNode, offset: TextSize) -> Option<CallSite> {
    let token = root.token_at_offset(offset).left_biased()?;
    token.parent_ancestors().find_map(|node| {
        let call = ast::CallExpr::cast(node)?;
        let args = call.args()?;
        let open = args
            .syntax()
            .children_with_tokens()
            .find(|element| element.kind() == SyntaxKind::LeftParen)?;
        let close = args
            .syntax()
            .children_with_tokens()
            .find(|element| element.kind() == SyntaxKind::RightParen);
        let inside = offset >= open.text_range().end()
            && close.is_none_or(|close| offset <= close.text_range().start());
        if !inside {
            return None;
        }

        let callee = match call.callee()? {
            ast::Expr::IdentExpr(ident) => ident.syntax().text_range(),
            ast::Expr::FieldAccessExpr(access) => access.field()?.syntax().text_range(),
            ast::Expr::CallExpr(_)
            | ast::Expr::StructExpr(_)
            | ast::Expr::ListExpr(_)
            | ast::Expr::BinaryExpr(_)
            | ast::Expr::UnaryExpr(_)
            | ast::Expr::ParenExpr(_)
            | ast::Expr::Literal(_)
            | ast::Expr::Pipeline(_) => return None,
        };
        let argument = args
            .syntax()
            .children_with_tokens()
            .filter(|element| {
                element.kind() == SyntaxKind::Comma && element.text_range().end() <= offset
            })
            .count();
        Some(CallSite { callee, argument })
    })
}

/// The overloads of the function `site` calls in `source`, as the check
/// resolved its name.
pub(crate) fn signature_help(
    checked: &Checked,
    source: SourceId,
    site: CallSite,
) -> Option<SignatureHelp> {
    let resolution = checked.resolutions().iter().find(|resolution| {
        resolution.used.source == source
            && resolution.used.range == site.callee
            && resolution.kind == DeclarationKind::Function
    })?;
    let root = checked.syntax(resolution.declared.source)?;
    let named = ast::FuncStmt::cast(node_at(&root, resolution.declaration.range)?)?;
    let name = named.name()?.token()?.text().to_owned();

    // An overload is a function of the same name beside it.
    let overloads: Vec<ast::FuncStmt> = named
        .syntax()
        .parent()?
        .children()
        .filter_map(ast::FuncStmt::cast)
        .filter(|func| {
            func.name()
                .and_then(|ident| ident.token())
                .is_some_and(|token| token.text() == name)
        })
        .collect();

    let signatures: Vec<Signature> = overloads.iter().map(signature).collect();
    let resolved = overloads
        .iter()
        .position(|func| func.syntax() == named.syntax())
        .expect("a function is one of its own overloads");
    let takes_argument = |at: usize| signatures[at].parameters.len() > site.argument;
    let active_signature = if takes_argument(resolved) {
        resolved
    } else {
        (0..signatures.len())
            .find(|&at| takes_argument(at))
            .unwrap_or(resolved)
    };
    Some(SignatureHelp {
        signatures,
        active_signature,
        active_parameter: site.argument,
    })
}

/// A function's header, rebuilt from its parts so each parameter's place in
/// it is known: `def f(x: int64, y: int64) -> int64`.
fn signature(func: &ast::FuncStmt) -> Signature {
    let header = file_structure::header(func);
    let before_params = header.find('(').map_or(header.as_str(), |at| &header[..at]);
    let mut label = format!("{before_params}(");
    let mut parameters = Vec::new();
    for (index, param) in func.params().enumerate() {
        if index > 0 {
            label.push_str(", ");
        }
        let start = TextSize::of(label.as_str());
        label.push_str(&file_structure::compact(&param.syntax().text().to_string()));
        parameters.push(TextRange::new(start, TextSize::of(label.as_str())));
    }
    label.push(')');
    if let Some(result) = func.result() {
        label.push_str(" -> ");
        label.push_str(&file_structure::compact(
            &result.syntax().text().to_string(),
        ));
    }
    Signature { label, parameters }
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};

    use crate::test_support::{FILE, at, checked, cursor};

    fn check(fixture: &str, expected: &Expect) {
        let (text, offset) = cursor(fixture);
        let (_tree, checked) = checked(&[], &text);
        let help = crate::test_support::analysis(&text)
            .call_at(at(offset))
            .and_then(|site| checked.signature_help(FILE, site));
        let rendered = help.map(|help| {
            let lines: Vec<String> = help
                .signatures
                .iter()
                .enumerate()
                .map(|(at, signature)| {
                    let active = if at == help.active_signature {
                        "*"
                    } else {
                        " "
                    };
                    let parameter = signature
                        .parameters
                        .get(help.active_parameter)
                        .map_or("", |range| &signature.label[*range]);
                    format!("{active} {} [{parameter}]", signature.label)
                })
                .collect();
            lines.join("\n")
        });
        expected.assert_eq(rendered.as_deref().unwrap_or("none"));
    }

    const OVERLOADS: &str = "def f(x: int64) -> int64 { return x }\ndef f(x: int64, y: int64) -> int64 { return x + y }\ntable t = { a: int64 }\n";

    #[test]
    fn a_call_shows_its_overloads() {
        check(
            &format!("{OVERLOADS}from t |> select f(a, $0) as v\n"),
            &expect![[r"
                  def f(x: int64) -> int64 []
                * def f(x: int64, y: int64) -> int64 [y: int64]"]],
        );
    }

    #[test]
    fn the_first_argument_picks_the_overload_the_call_resolved() {
        check(
            &format!("{OVERLOADS}from t |> select f($0a) as v\n"),
            &expect![[r"
                * def f(x: int64) -> int64 [x: int64]
                  def f(x: int64, y: int64) -> int64 [x: int64]"]],
        );
    }

    #[test]
    fn outside_the_parentheses_there_is_none() {
        check(
            &format!("{OVERLOADS}from t |> select f(a)$0 as v\n"),
            &expect!["none"],
        );
    }

    #[test]
    fn a_call_past_every_overload_still_shows_them() {
        check(
            &format!("{OVERLOADS}from t |> select f(a, a, $0) as v\n"),
            &expect![[r"
                  def f(x: int64) -> int64 []
                * def f(x: int64, y: int64) -> int64 []"]],
        );
    }

    #[test]
    fn a_name_the_check_read_before_its_call_was_typed() {
        let read = format!("{OVERLOADS}from t |> select f as v\n");
        let (_tree, checked) = checked(&[], &read);
        let (now, offset) = cursor(&read.replace("select f as v", "select f($0 as v"));
        let site = crate::test_support::analysis(&now)
            .call_at(at(offset))
            .expect("the position is in a call");
        let help = checked
            .signature_help(FILE, site)
            .expect("the check resolved `f`");
        expect!["def f(x: int64) -> int64"]
            .assert_eq(&help.signatures[help.active_signature].label);
    }
}
