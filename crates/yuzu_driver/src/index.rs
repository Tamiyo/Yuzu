//! What the IR knows about a program's names and types, kept as plain data
//! so it outlives the MLIR context it was read from.
//!
//! References are read after lowering, while each read of a local is still
//! a load of its place; types are read after inference. An op's location is
//! the range the lowering made it from, so both are keyed by source range.

use melior::Context;
use melior::ir::operation::{OperationLike, OperationRef, OperationResult};
use melior::ir::{Module, RegionLike, Type, Value, ValueLike};
use rustc_hash::FxHashMap;
use yuzu_diagnostics::diagnostics::Span;
use yuzu_diagnostics::source_map::SourceMap;
use yuzu_mlir::attributes::CalleeSource;
use yuzu_mlir::diagnostics::span;
use yuzu_mlir::ir::block::BlockExt;
use yuzu_mlir::ir::operation::{OperationCast, OperationExt};
use yuzu_mlir::ir::region::RegionExt;
use yuzu_mlir::ops::yzl::YzlOp;
use yuzu_mlir::types::{self, ErrorType, QueryType, RefType, UnresolvedType};

/// A name the program uses, and the declaration it names. `at` covers the
/// syntax the use was lowered from, which holds the name; `target` covers
/// the whole declaration.
#[derive(Clone, Debug)]
pub struct Reference {
    pub at: Span,
    pub target: Span,
    /// The name as the declaration spells it.
    pub name: String,
    pub kind: TargetKind,
}

/// What a reference's target declares, which says where in its syntax the
/// name is: a function's parameter lies inside the function it belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetKind {
    Local,
    Parameter,
    /// A function, a module-level `let`, a table, or a struct.
    Symbol,
}

/// A type inference settled, and the syntax it belongs to: an expression,
/// or the declaration of a local or a module-level `let`.
#[derive(Clone, Debug)]
pub struct Typed {
    pub at: Span,
    pub ty: String,
}

/// The references and types a check read from the IR.
#[derive(Debug, Default)]
pub struct Index {
    pub references: Vec<Reference>,
    pub types: Vec<Typed>,
}

/// Reads an index from the module at the two points a check passes.
pub(crate) struct IndexReader<'s> {
    sources: &'s SourceMap,
    index: Index,
    /// Each local's declaration, and the value it first stores.
    initializers: Vec<(Span, Span)>,
}

impl<'s> IndexReader<'s> {
    pub(crate) fn new(sources: &'s SourceMap) -> Self {
        IndexReader {
            sources,
            index: Index::default(),
            initializers: Vec::new(),
        }
    }

    pub(crate) fn finish(self) -> Index {
        self.index
    }

    /// References, from the module as the lowering left it.
    pub(crate) fn read_lowered(&mut self, module: &Module<'_>) {
        let body = module.body();
        let mut declarations: FxHashMap<&str, Span> = FxHashMap::default();
        for op in body.operations() {
            if let Some(name) = op.text_attribute("sym_name")
                && let Some(at) = span(self.sources, op.location())
            {
                declarations.insert(name, at);
            }
        }

        for op in body.operations() {
            self.read_references(op, &declarations);
        }
    }

    /// Types, from the module as inference left it.
    pub(crate) fn read_inferred(&mut self, context: &Context, module: &Module<'_>) {
        for op in module.body().operations() {
            self.read_types(context, op);
        }

        for (declaration, initializer) in std::mem::take(&mut self.initializers) {
            let ty = self
                .index
                .types
                .iter()
                .find(|typed| same(typed.at, initializer))
                .map(|typed| typed.ty.clone());
            if let Some(ty) = ty {
                self.index.types.push(Typed {
                    at: declaration,
                    ty,
                });
            }
        }
    }

    fn read_references(&mut self, op: OperationRef<'_, '_>, declarations: &FxHashMap<&str, Span>) {
        let target = match op.as_yzl() {
            Some(YzlOp::Load(_)) => op
                .try_first_operand()
                .and_then(|place| self.read_local(place)),
            Some(YzlOp::Store(_)) => {
                self.read_initializer(op);
                None
            }
            Some(YzlOp::Call(call)) => match call.callee_source() {
                Some(CalleeSource::Builtin) => None,
                Some(CalleeSource::Fn | CalleeSource::External | CalleeSource::Const) | None => {
                    symbol(declarations, call.callee().value())
                }
            },
            Some(YzlOp::From(from)) => symbol(declarations, from.source().value()),
            Some(YzlOp::Join(join)) => symbol(declarations, join.rhs().value()),
            _ => None,
        };

        if let Some((target, name, kind)) = target
            && let Some(at) = span(self.sources, op.location())
        {
            self.index.references.push(Reference {
                at,
                target,
                name,
                kind,
            });
        }

        for region in op.regions() {
            for block in region.blocks() {
                for inner in block.operations() {
                    self.read_references(inner, declarations);
                }
            }
        }
    }

    /// The first store to a place is the value its declaration gives it.
    fn read_initializer(&mut self, store: OperationRef<'_, '_>) {
        let (Some(place), Some(value)) = (store.operand(0).ok(), store.operand(1).ok()) else {
            return;
        };
        let (Some(declaration), Some(initializer)) =
            (self.defined_at(place), self.defined_at(value))
        else {
            return;
        };
        if !self
            .initializers
            .iter()
            .any(|(seen, _)| same(*seen, declaration))
        {
            self.initializers.push((declaration, initializer));
        }
    }

    fn read_types(&mut self, context: &Context, op: OperationRef<'_, '_>) {
        let ty = match op.as_yzl() {
            Some(YzlOp::Const(_)) => last_yield(op),
            _ => op.try_first_result().map(|result| result.r#type()),
        };

        if let Some(ty) = ty.filter(|&ty| is_shown(context, ty))
            && let Some(at) = span(self.sources, op.location())
        {
            self.index.types.push(Typed {
                at,
                ty: types::name(context, ty),
            });
        }

        for region in op.regions() {
            for block in region.blocks() {
                for inner in block.operations() {
                    self.read_types(context, inner);
                }
            }
        }
    }

    /// The local a place is: its declaration, name and kind.
    fn read_local(&self, place: Value<'_, '_>) -> Option<(Span, String, TargetKind)> {
        let result = OperationResult::try_from(place).ok()?;
        let owner = result.owner();
        let Some(YzlOp::Local(local)) = owner.as_yzl() else {
            return None;
        };
        let kind = if local.is_param() {
            TargetKind::Parameter
        } else {
            TargetKind::Local
        };
        let name = local.var_name().value().to_owned();
        Some((span(self.sources, owner.location())?, name, kind))
    }

    fn defined_at(&self, value: Value<'_, '_>) -> Option<Span> {
        let result = OperationResult::try_from(value).ok()?;
        span(self.sources, result.owner().location())
    }
}

/// A symbol's declaration, and its name without the module path that
/// qualifies it: `helpers.two` is declared as `two`.
fn symbol(
    declarations: &FxHashMap<&str, Span>,
    symbol: &str,
) -> Option<(Span, String, TargetKind)> {
    let target = *declarations.get(symbol)?;
    let name = symbol.rsplit('.').next().unwrap_or(symbol).to_owned();
    Some((target, name, TargetKind::Symbol))
}

/// A type worth showing a reader: one inference settled, of a value the
/// program wrote. A place, a relation and an error are none of these.
fn is_shown(context: &Context, ty: Type<'_>) -> bool {
    ty != UnresolvedType::get(context)
        && ty != QueryType::get(context)
        && ty != RefType::get(context)
        && !ErrorType::is(ty)
}

/// The type a `let` yields: what the last op of its region returns.
fn last_yield<'c>(op: OperationRef<'c, '_>) -> Option<Type<'c>> {
    let terminator = op.regions().next()?.first_block()?.last_operation()?;
    terminator.try_first_operand().map(|value| value.r#type())
}

fn same(a: Span, b: Span) -> bool {
    a.source_id == b.source_id && a.range == b.range
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};
    use yuzu_diagnostics::diagnostics::Span;

    use crate::modules::MapResolver;
    use crate::{Focus, check};

    fn check_index(source: &str, expected: Expect) {
        check_program(&[], source, expected);
    }

    fn check_program(modules: &[(&str, &str)], source: &str, expected: Expect) {
        let resolver = MapResolver(
            modules
                .iter()
                .map(|(path, text)| ((*path).to_owned(), (*text).to_owned()))
                .collect(),
        );
        let checked = check(
            Focus::Entry {
                name: "main.yz",
                source,
                syntax: None,
            },
            &resolver,
        );
        let text = |span: Span| {
            let text = &checked.sources.text(span.source_id)[span.range];
            text.lines().next().unwrap_or_default().to_owned()
        };

        let mut lines: Vec<String> = checked
            .index
            .references
            .iter()
            .filter(|reference| checked.sources.name(reference.at.source_id) == "main.yz")
            .map(|reference| {
                let file = checked.sources.name(reference.target.source_id);
                let target = match file {
                    "main.yz" => format!("{:?}", text(reference.target)),
                    _ => format!("{file}: {:?}", text(reference.target)),
                };
                format!("use {:?} -> {target}", text(reference.at))
            })
            .collect();
        lines.extend(
            checked
                .index
                .types
                .iter()
                .filter(|typed| checked.sources.name(typed.at.source_id) == "main.yz")
                .map(|typed| format!("type {:?}: {}", text(typed.at), typed.ty)),
        );
        expected.assert_eq(&lines.join("\n"));
    }

    #[test]
    fn names_and_types_of_a_program() {
        check_index(
            r#"table t = { a: int64 }
let cap = 10
def double(x: int64) -> int64 {
    let y = x * 2
    return y
}
from t |> select double(a) + cap as v
"#,
            expect![[r#"
                use "x" -> "def double(x: int64) -> int64 {"
                use "y" -> "let y = x * 2"
                use "from t" -> "table t = { a: int64 }"
                use "double(a)" -> "def double(x: int64) -> int64 {"
                use "cap" -> "let cap = 10"
                type "let cap = 10": int64
                type "10": int64
                type "2": int64
                type "x * 2": int64
                type "double(a)": int64
                type "cap": int64
                type "double(a) + cap": int64
                type "let y = x * 2": int64"#]],
        );
    }

    #[test]
    fn a_join_and_an_import_name_their_declarations() {
        check_program(
            &[("helpers", "pub def two() -> int64 { return 2 }\n")],
            r#"from helpers import two
table t = { a: int64 }
table u = { b: int64 }
from t |> join u on a == b |> select a + two() as v
"#,
            expect![[r#"
                use "from t" -> "table t = { a: int64 }"
                use "|> join u on a == b" -> "table u = { b: int64 }"
                use "two()" -> helpers.yz: "pub def two() -> int64 { return 2 }"
                type "a == b": bool
                type "two()": int64
                type "a + two()": int64"#]],
        );
    }
}
