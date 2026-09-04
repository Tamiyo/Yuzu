use std::collections::HashMap;

use yuzu_types::{SymbolId, Type, TypeId};

use crate::infer::InferCtx;

impl InferCtx<'_> {
    pub(crate) fn concretize(&mut self) {
        let mut memo = HashMap::new();
        for index in 0..self.result.expr_types.0.len() {
            if let Some(ty) = self.result.expr_types.0[index] {
                self.result.expr_types.0[index] = Some(self.concretize_ty(ty, &mut memo));
            }
        }
        for index in 0..self.result.rel_types.0.len() {
            if let Some(ty) = self.result.rel_types.0[index] {
                self.result.rel_types.0[index] = Some(self.concretize_ty(ty, &mut memo));
            }
        }
    }

    fn concretize_ty(&mut self, ty: TypeId, memo: &mut HashMap<TypeId, TypeId>) -> TypeId {
        if let Some(&concretized) = memo.get(&ty) {
            return concretized;
        }
        let resolved_ty = self.resolve(ty);
        let concretized = match self.types.ty(resolved_ty).clone() {
            Type::List(l) => {
                let inner = self.concretize_ty(l.inner, memo);
                self.types.list_ty(inner)
            }
            Type::Relation(r) => {
                let columns = r
                    .columns
                    .clone()
                    .into_iter()
                    .map(|column| yuzu_types::Column {
                        ty: self.concretize_ty(column.ty, memo),
                        ..column
                    })
                    .collect();
                self.types.relation_ty(columns)
            }
            Type::Func(f) => {
                let args: Vec<TypeId> = f
                    .args
                    .iter()
                    .map(|&a| self.concretize_ty(a, memo))
                    .collect();
                let ret = self.concretize_ty(f.ret_type, memo);
                self.types.func_ty(args, ret)
            }
            Type::Struct(s) => {
                let fields: Vec<(SymbolId, TypeId)> = s
                    .fields
                    .iter()
                    .map(|&(n, t)| (n, self.concretize_ty(t, memo)))
                    .collect();
                self.types.struct_ty(s.name, fields)
            }
            _ => resolved_ty,
        };
        memo.insert(ty, concretized);
        concretized
    }
}
