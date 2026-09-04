use rustc_hash::FxHashMap;

use yuzu_core::adt::SymbolId;
use yuzu_types::TypeId;

use crate::BindingId;

/// What a name resolves to in the ANF: a local value (let or parameter) or a
/// reference to a function declaration.
#[derive(Clone, Copy)]
pub(crate) enum Symbol {
    Value(BindingId),
    Func(BindingId, TypeId),
}

pub(crate) struct SymbolTable {
    scopes: Vec<Scope>,
}

#[derive(Default)]
struct Scope {
    symbols: FxHashMap<SymbolId, Symbol>,
}

impl SymbolTable {
    pub(crate) fn new() -> Self {
        Self {
            scopes: vec![Scope::default()],
        }
    }

    pub(crate) fn push_scope(&mut self) {
        self.scopes.push(Scope::default());
    }

    pub(crate) fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    pub(crate) fn bind_value(&mut self, name: SymbolId, binding: BindingId) {
        self.insert(name, Symbol::Value(binding));
    }

    pub(crate) fn bind_func(&mut self, name: SymbolId, binding: BindingId, ty: TypeId) {
        self.insert(name, Symbol::Func(binding, ty));
    }

    pub(crate) fn lookup(&self, name: SymbolId) -> Option<Symbol> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.symbols.get(&name).copied())
    }

    fn insert(&mut self, name: SymbolId, symbol: Symbol) {
        self.scopes
            .last_mut()
            .expect("expected a scope but found none")
            .symbols
            .insert(name, symbol);
    }
}
