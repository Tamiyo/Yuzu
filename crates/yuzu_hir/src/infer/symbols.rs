use std::collections::HashMap;

use yuzu_types::{BuiltinFunc, TypeId};

use crate::{Mutability, StmtId, SymbolId};

#[derive(Clone, Copy)]
pub(crate) enum Binding {
    LetStmt {
        mutability: Mutability,
        ty: TypeId,
    },
    Param {
        ty: TypeId,
    },
    /// A `from`/`join` alias standing for a row of the query, so `alias.column`
    /// is a column reference rather than a struct field access.
    Relation {
        ty: TypeId,
    },
    FuncStmt {
        stmt: StmtId,
        ty: TypeId,
        is_agg: bool,
        is_external: bool,
    },
    /// A registered builtin. Not a value — call sites resolve it through the
    /// registry, and a bare reference is an error.
    Builtin {
        func: BuiltinFunc,
    },
}

impl Binding {
    pub(crate) fn ty(&self) -> TypeId {
        match self {
            Binding::LetStmt { ty, .. }
            | Binding::Param { ty, .. }
            | Binding::Relation { ty, .. }
            | Binding::FuncStmt { ty, .. } => *ty,
            Binding::Builtin { .. } => unreachable!("a builtin has no value type"),
        }
    }
}

pub(crate) enum ScopeKind {
    Func { return_ty: TypeId },
    Block,
}

struct Scope {
    kind: ScopeKind,
    bindings: HashMap<SymbolId, Binding>,
    types: HashMap<SymbolId, TypeId>,
    structs: HashMap<SymbolId, StmtId>,
    relations: HashMap<SymbolId, TypeId>,
    row: Option<TypeId>,
}

impl Scope {
    fn new(kind: ScopeKind) -> Self {
        Self {
            kind,
            bindings: HashMap::new(),
            types: HashMap::new(),
            structs: HashMap::new(),
            relations: HashMap::new(),
            row: None,
        }
    }
}

pub(crate) struct SymbolTable {
    scopes: Vec<Scope>,
}

impl SymbolTable {
    pub(crate) fn new() -> Self {
        Self {
            scopes: vec![Scope::new(ScopeKind::Block)],
        }
    }

    pub(crate) fn push_scope(&mut self, kind: ScopeKind) {
        self.scopes.push(Scope::new(kind));
    }

    pub(crate) fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    pub(crate) fn row(&self) -> Option<TypeId> {
        self.scopes.last()?.row
    }

    /// Puts a stage's output row in scope and hands it back, so each stage
    /// leaves the row its successor resolves against.
    pub(crate) fn replace_current_row(&mut self, row: TypeId) -> TypeId {
        let scope = self.scopes.last_mut().expect("there is always a scope");
        scope.row = Some(row);
        row
    }

    pub(crate) fn return_ty(&self) -> Option<TypeId> {
        self.scopes.iter().rev().find_map(|scope| match scope.kind {
            ScopeKind::Func { return_ty } => Some(return_ty),
            ScopeKind::Block => None,
        })
    }

    pub(crate) fn bind_symbol(&mut self, name: SymbolId, binding: Binding) {
        self.current().bindings.insert(name, binding);
    }

    pub(crate) fn lookup_symbol(&self, name: SymbolId) -> Option<&Binding> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.bindings.get(&name))
    }

    pub(crate) fn bind_type(&mut self, name: SymbolId, ty: TypeId) {
        self.current().types.insert(name, ty);
    }

    pub(crate) fn lookup_type(&self, name: SymbolId) -> Option<TypeId> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.types.get(&name).copied())
    }

    pub(crate) fn bind_struct(&mut self, name: SymbolId, decl: StmtId) {
        self.current().structs.insert(name, decl);
    }

    pub(crate) fn lookup_struct(&self, name: SymbolId) -> Option<StmtId> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.structs.get(&name).copied())
    }

    pub(crate) fn bind_relation(&mut self, name: SymbolId, relation: TypeId) {
        self.current().relations.insert(name, relation);
    }

    pub(crate) fn lookup_relation(&self, name: SymbolId) -> Option<TypeId> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.relations.get(&name).copied())
    }

    fn current(&mut self) -> &mut Scope {
        self.scopes.last_mut().expect("symbol table has no scope")
    }
}
