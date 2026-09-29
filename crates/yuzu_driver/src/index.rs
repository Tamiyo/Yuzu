//! What the IR knows about a program's names and types, kept as plain data
//! so it outlives the MLIR context it was read from.
//!
//! References are what the lowering told its [`NameListener`] as it
//! resolved each name, since the IR keeps no import or alias. Types are
//! read from the IR after inference; an op's location is the range the
//! lowering made it from, so both are keyed by source range.

use melior::ir::operation::{OperationLike, OperationRef};
use melior::ir::{Module, Type, Value, ValueLike};
use rustc_hash::{FxHashMap, FxHashSet};
use text_size::TextRange;
use yuzu_diagnostics::{SourceId, SourceMap, Span};
use yuzu_mlir::diagnostics::span;
use yuzu_mlir::ir::block::BlockExt;
use yuzu_mlir::ir::operation::{OperationCast, OperationExt};
use yuzu_mlir::ir::region::RegionExt;
use yuzu_mlir::ir::value::op_result;
use yuzu_mlir::ops::yzl::YzlOp;
use yuzu_mlir::types::{self, ErrorType, QueryType, RefType, UnresolvedType};
use yuzu_passes::{NameKind, NameListener, NameTarget, NameUse};

/// A name the program uses, and what it names. `at` is the name as
/// written; `target` covers the whole declaration, or the start of a
/// module's file.
#[derive(Clone, Debug)]
pub struct Reference {
    pub at: Span,
    pub target: Span,
    /// The name as the declaration spells it, which an alias does not; a
    /// module's path.
    pub name: String,
    pub kind: TargetKind,
}

/// What a reference names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetKind {
    /// A declaration, whose own name is inside `target`.
    Declaration,
    /// A module, whose file `target` points at.
    Module,
}

/// A type inference settled, and the syntax it belongs to: an expression,
/// or the declaration of a local or a module-level `let`.
#[derive(Clone, Debug)]
pub struct Typed {
    pub at: Span,
    pub ty: String,
}

/// The columns the expressions of a stage can read.
#[derive(Clone, Debug)]
pub struct StageRow {
    pub stage: Span,
    pub columns: Vec<String>,
}

/// A name the top level of a file can use, and what it declares.
#[derive(Clone, Debug)]
pub struct ScopeName {
    pub name: String,
    pub kind: ScopeKind,
}

/// What a name in scope declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScopeKind {
    Struct,
    Relation,
    Function,
    Trait,
    /// A module-level `let`.
    Binding,
    Module,
}

/// The references, types and scopes a check read.
#[derive(Debug, Default)]
pub struct Index {
    pub references: Vec<Reference>,
    pub types: Vec<Typed>,
    pub rows: Vec<StageRow>,
    /// The names each file's top level can use, by the file's source.
    pub scopes: FxHashMap<SourceId, Vec<ScopeName>>,
}

/// Reads an index from the module at the two points a check passes.
pub(crate) struct IndexReader<'s> {
    sources: &'s SourceMap,
    index: Index,
    /// Each local's declaration, and the value it first stores.
    initializers: FxHashMap<Span, Span>,
    /// The names recorded so far. The lowering may resolve a name twice, as
    /// the hoist and the walk both read a signature.
    recorded: FxHashSet<Span>,
}

impl NameListener for IndexReader<'_> {
    fn on_name(&mut self, name: NameUse<'_>) {
        if !self.recorded.insert(name.used) {
            return;
        }

        let (target, declared, kind) = match name.target {
            NameTarget::Declaration { at, name } => (at, name, TargetKind::Declaration),
            NameTarget::Module { file, path } => (
                Span {
                    source_id: file,
                    range: TextRange::empty(0.into()),
                },
                path,
                TargetKind::Module,
            ),
        };
        self.index.references.push(Reference {
            at: name.used,
            target,
            name: declared.to_owned(),
            kind,
        });
    }

    fn on_row(&mut self, stage: Span, columns: &mut dyn Iterator<Item = &str>) {
        self.index.rows.push(StageRow {
            stage,
            columns: columns.map(str::to_owned).collect(),
        });
    }

    fn on_file(&mut self, file: SourceId, names: &mut dyn Iterator<Item = (&str, NameKind)>) {
        let names = names
            .map(|(name, kind)| ScopeName {
                name: name.to_owned(),
                kind: match kind {
                    NameKind::Struct => ScopeKind::Struct,
                    NameKind::Relation => ScopeKind::Relation,
                    NameKind::Function => ScopeKind::Function,
                    NameKind::Trait => ScopeKind::Trait,
                    NameKind::Binding => ScopeKind::Binding,
                    NameKind::Module => ScopeKind::Module,
                },
            })
            .collect();
        self.index.scopes.insert(file, names);
    }
}

impl<'s> IndexReader<'s> {
    pub(crate) fn new(sources: &'s SourceMap) -> Self {
        IndexReader {
            sources,
            index: Index::default(),
            initializers: FxHashMap::default(),
            recorded: FxHashSet::default(),
        }
    }

    pub(crate) fn finish(self) -> Index {
        self.index
    }

    /// Each local's initializer, from the module as the lowering left it,
    /// while each local is still a place that its first store fills.
    pub(crate) fn read_lowered(&mut self, module: &Module<'_>) {
        for op in module.body().operations() {
            self.read_initializers(op);
        }
    }

    /// Types, from the module as inference left it.
    pub(crate) fn read_inferred(&mut self, module: &Module<'_>) {
        for op in module.body().operations() {
            self.read_types(op);
        }

        let types: FxHashMap<Span, &str> = self
            .index
            .types
            .iter()
            .map(|typed| (typed.at, typed.ty.as_str()))
            .collect();
        let initialized: Vec<Typed> = std::mem::take(&mut self.initializers)
            .into_iter()
            .filter_map(|(declaration, initializer)| {
                Some(Typed {
                    at: declaration,
                    ty: types.get(&initializer)?.to_string(),
                })
            })
            .collect();
        self.index.types.extend(initialized);
    }

    fn read_initializers(&mut self, op: OperationRef<'_, '_>) {
        if let Some(YzlOp::Store(_)) = op.as_yzl() {
            self.read_initializer(op);
        }

        for region in op.regions() {
            for block in region.blocks() {
                for inner in block.operations() {
                    self.read_initializers(inner);
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
        self.initializers.entry(declaration).or_insert(initializer);
    }

    fn read_types(&mut self, op: OperationRef<'_, '_>) {
        let ty = match op.as_yzl() {
            Some(YzlOp::Const(_)) => last_yield(op),
            _ => op.try_first_result().map(|result| result.r#type()),
        };

        if let Some(ty) = ty.filter(|&ty| is_shown(ty))
            && let Some(at) = span(self.sources, op.location())
        {
            self.index.types.push(Typed {
                at,
                ty: types::name(ty),
            });
        }

        for region in op.regions() {
            for block in region.blocks() {
                for inner in block.operations() {
                    self.read_types(inner);
                }
            }
        }
    }

    fn defined_at(&self, value: Value<'_, '_>) -> Option<Span> {
        let result = op_result(value)?;
        span(self.sources, result.owner().location())
    }
}

/// A type worth showing a reader: one inference settled, of a value the
/// program wrote. A place, a relation and an error are none of these.
fn is_shown(ty: Type<'_>) -> bool {
    UnresolvedType::from_type(ty).is_none()
        && QueryType::from_type(ty).is_none()
        && RefType::from_type(ty).is_none()
        && ErrorType::from_type(ty).is_none()
}

/// The type a `let` yields: what the last op of its region returns.
fn last_yield<'c>(op: OperationRef<'c, '_>) -> Option<Type<'c>> {
    let terminator = op.body_terminator()?;
    terminator.try_first_operand().map(|value| value.r#type())
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};
    use yuzu_diagnostics::Span;

    use crate::modules::{MapResolver, Origin};
    use crate::{Focus, check};

    fn check_index(source: &str, expected: &Expect) {
        check_program(&[], source, expected);
    }

    fn check_program(modules: &[(&str, &str)], source: &str, expected: &Expect) {
        let resolver = MapResolver(
            modules
                .iter()
                .map(|(path, text)| ((*path).to_owned(), (*text).to_owned()))
                .collect(),
        );
        let checked = check(
            Focus::Entry {
                origin: &Origin::Named("main.yz".to_owned()),
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
            r"table t = { a: int64 }
let cap = 10
def double(x: int64) -> int64 {
    let y = x * 2
    return y
}
from t |> select double(a) + cap as v
",
            &expect![[r#"
                use "x" -> "x: int64"
                use "y" -> "let y = x * 2"
                use "t" -> "table t = { a: int64 }"
                use "a" -> "a: int64"
                use "double" -> "def double(x: int64) -> int64 {"
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
            r"from helpers import two
table t = { a: int64 }
table u = { b: int64 }
from t |> join u on a == b |> select a + two() as v
",
            &expect![[r#"
                use "helpers" -> helpers.yz: ""
                use "two" -> helpers.yz: "pub def two() -> int64 { return 2 }"
                use "t" -> "table t = { a: int64 }"
                use "u" -> "table u = { b: int64 }"
                use "a" -> "a: int64"
                use "b" -> "b: int64"
                use "a" -> "a: int64"
                use "two" -> helpers.yz: "pub def two() -> int64 { return 2 }"
                type "a == b": bool
                type "two()": int64
                type "a + two()": int64"#]],
        );
    }
}
