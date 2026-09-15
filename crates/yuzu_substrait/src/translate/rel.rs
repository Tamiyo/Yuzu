//! The stages. yzr is already the shape a plan wants — the lowering turned
//! `drop`, `rename`, `set` and `distinct` into projections and groupings —
//! so each op here is one Substrait relation.

use std::collections::HashMap;

use melior::ir::operation::{OperationLike, OperationRef};
use melior::ir::{RegionLike, Value, ValueLike};
use substrait::proto::{
    AggregateFunction, AggregateRel, AggregationPhase, Expression, FetchRel, FilterRel,
    FunctionArgument, JoinRel, NamedStruct, ProjectRel, ReadRel, Rel, RelCommon,
    aggregate_function::AggregationInvocation,
    aggregate_rel::{Grouping, Measure},
    expression::literal::LiteralType,
    fetch_rel::{CountMode, OffsetMode},
    function_argument::ArgType,
    join_rel::JoinType,
    read_rel::{NamedTable, ReadType},
    rel::RelType,
    rel_common::{Emit, EmitKind},
    r#type,
};
use yuzu_mlir::attributes::JoinKind;
use yuzu_mlir::ext::{BlockExt, OperationCast, OperationExt, ValueExt};
use yuzu_mlir::ops::yzr::YzrOp;
use yuzu_types::AggFunc;

use crate::extensions::{EXTERNAL_URN, aggregate_target};
use crate::translate::Translator;
use crate::translate::expr::{literal, selection};
use crate::translate::functions;
use crate::translate::types::{emit_type, nullable, type_code};

impl<'c, 'a> Translator<'c, 'a, '_> {
    /// The relation a value stands for. A value two stages read is walked
    /// once and written out at each use, since Substrait nests.
    pub(crate) fn translate_rel(&mut self, value: Value<'c, 'a>) -> Option<Rel> {
        if let Some(translated) = self.translated.get(&value.id()) {
            return Some(translated.clone());
        }

        let op = Self::producer(value)?;
        let rel_type = match op.as_yzr()? {
            YzrOp::Table(table) => self.translate_table(op, table.name().value())?,
            YzrOp::Filter(_) => self.translate_filter(op)?,
            YzrOp::Project(_) => self.translate_project(op, Projection::Replace)?,
            YzrOp::Extend(_) => self.translate_project(op, Projection::Append)?,
            YzrOp::Limit(limit) => {
                let count = limit.count().value();
                let offset = limit.offset().map(|offset| offset.value());
                self.translate_fetch(op, count, offset)?
            }
            YzrOp::Aggregate(grouping) => {
                let dense = grouping.keys();
                let keys: Vec<i32> = (0..dense.len())
                    .filter_map(|index| dense.element(index).ok())
                    .map(|key| key as i32)
                    .collect();
                self.translate_aggregate(op, &keys)?
            }
            YzrOp::Join(join) => {
                let kind = join_type(join.kind());
                self.translate_join(op, kind)?
            }
            YzrOp::Union(_) | YzrOp::Intersect(_) | YzrOp::Except(_) => {
                self.unsupported(op, "set operations are not lowered yet");
                return None;
            }
            YzrOp::Output(_) | YzrOp::Agg(_) | YzrOp::Count(_) | YzrOp::Yield(_) => {
                self.unsupported(op, "this is not a relation");
                return None;
            }
        };

        let rel = Rel {
            rel_type: Some(rel_type),
        };
        self.translated.insert(value.id(), rel.clone());
        Some(rel)
    }

    /// The relation a stage reads, and how many columns it carries.
    fn translate_input(&mut self, op: OperationRef<'c, 'a>) -> Option<(Rel, usize)> {
        let value = op.try_first_operand()?;
        let width = self.width(value)?;
        Some((self.translate_rel(value)?, width))
    }

    fn translate_table(&mut self, op: OperationRef<'c, 'a>, name: &str) -> Option<RelType> {
        let (names, types) = self.row(op.first_result().r#type())?;
        let mut fields = Vec::with_capacity(types.len());
        for ty in types {
            let Some(field) = emit_type(self.context, ty) else {
                self.unsupported(op, "this column has no Substrait type");
                return None;
            };

            fields.push(field);
        }

        Some(RelType::Read(Box::new(ReadRel {
            base_schema: Some(NamedStruct {
                names: names.iter().map(|name| name.to_string()).collect(),
                r#struct: Some(r#type::Struct {
                    types: fields,
                    nullability: nullable(),
                    ..Default::default()
                }),
            }),
            read_type: Some(ReadType::NamedTable(NamedTable {
                names: vec![name.to_string()],
                ..Default::default()
            })),
            ..Default::default()
        })))
    }

    fn translate_filter(&mut self, op: OperationRef<'c, 'a>) -> Option<RelType> {
        let (input, _) = self.translate_input(op)?;
        let region = self.translate_region(op)?;
        let Some(condition) = self.yielded(op, &region)?.into_iter().next() else {
            self.unsupported(op, "`where` has no predicate to filter on");
            return None;
        };

        Some(RelType::Filter(Box::new(FilterRel {
            input: Some(Box::new(input)),
            condition: Some(Box::new(condition)),
            ..Default::default()
        })))
    }

    /// Substrait appends a projection's expressions to its input's columns
    /// and the emit mapping picks what survives: a `project` keeps only what
    /// it computed, an `extend` keeps the input's columns as well.
    fn translate_project(&mut self, op: OperationRef<'c, 'a>, kind: Projection) -> Option<RelType> {
        let (input, width) = self.translate_input(op)?;
        let region = self.translate_region(op)?;
        let expressions = self.yielded(op, &region)?;
        let computed = width..width + expressions.len();
        let output_mapping = match kind {
            Projection::Replace => computed.map(|index| index as i32).collect(),
            Projection::Append => (0..width)
                .chain(computed)
                .map(|index| index as i32)
                .collect(),
        };

        Some(RelType::Project(Box::new(ProjectRel {
            common: emit(output_mapping),
            input: Some(Box::new(input)),
            expressions,
            ..Default::default()
        })))
    }

    fn translate_fetch(
        &mut self,
        op: OperationRef<'c, 'a>,
        count: i64,
        offset: Option<i64>,
    ) -> Option<RelType> {
        let (input, _) = self.translate_input(op)?;
        Some(RelType::Fetch(Box::new(FetchRel {
            input: Some(Box::new(input)),
            offset_mode: Some(OffsetMode::OffsetExpr(Box::new(literal(LiteralType::I64(
                offset.unwrap_or(0),
            ))))),
            count_mode: Some(CountMode::CountExpr(Box::new(literal(LiteralType::I64(
                count,
            ))))),
            ..Default::default()
        })))
    }

    /// The keys are columns of the input row, by position, and the measures
    /// are the region's `yzr.agg` and `yzr.count` ops. A grouping with no
    /// measures is what `distinct` became.
    fn translate_aggregate(&mut self, op: OperationRef<'c, 'a>, keys: &[i32]) -> Option<RelType> {
        let (input, _) = self.translate_input(op)?;
        let region = self.translate_region(op)?;
        let measures = self.translate_measures(op, &region.values)?;

        Some(RelType::Aggregate(Box::new(AggregateRel {
            input: Some(Box::new(input)),
            grouping_expressions: keys.iter().map(|&key| selection(key)).collect(),
            groupings: vec![Grouping {
                expression_references: (0..keys.len() as u32).collect(),
                ..Default::default()
            }],
            measures,
            ..Default::default()
        })))
    }

    fn translate_measures(
        &mut self,
        op: OperationRef<'c, 'a>,
        values: &HashMap<usize, Expression>,
    ) -> Option<Vec<Measure>> {
        let Some(block) = op.regions().next().and_then(|region| region.first_block()) else {
            return Some(Vec::new());
        };

        let mut measures = Vec::new();
        for inner in block.operations() {
            let (func, arguments) = match inner.as_yzr() {
                Some(YzrOp::Agg(measure)) => {
                    let name = measure.r#fn().value();
                    let Some(func) = functions::of_aggregate(name) else {
                        self.unsupported(inner, format!("`{name}` is not an aggregate function"));
                        return None;
                    };

                    (func, vec![measure.value()])
                }
                Some(YzrOp::Count(_)) => (AggFunc::Count, Vec::new()),
                _ => continue,
            };

            measures.push(self.translate_measure(inner, func, &arguments, values)?);
        }

        Some(measures)
    }

    fn translate_measure(
        &mut self,
        op: OperationRef<'c, '_>,
        func: AggFunc,
        arguments: &[Value<'c, '_>],
        values: &HashMap<usize, Expression>,
    ) -> Option<Measure> {
        let (urn, base) = match func {
            AggFunc::External(_) => (EXTERNAL_URN, String::new()),
            func => {
                let (urn, base) = aggregate_target(func);
                (urn, base.to_string())
            }
        };

        let mut signature = Vec::new();
        let mut emitted = Vec::new();
        for &argument in arguments {
            let Some(code) = type_code(self.context, argument.r#type()) else {
                self.unsupported(op, "this measure's argument has no Substrait type");
                return None;
            };

            signature.push(code);
            emitted.push(FunctionArgument {
                arg_type: Some(ArgType::Value(self.expression_of(op, argument, values)?)),
            });
        }

        let Some(output) = emit_type(self.context, op.first_result().r#type()) else {
            self.unsupported(op, "this measure has no Substrait type");
            return None;
        };

        let anchor = self
            .extensions
            .register(urn, format!("{base}:{}", signature.join("_")));
        Some(Measure {
            measure: Some(AggregateFunction {
                function_reference: anchor,
                arguments: emitted,
                output_type: Some(output),
                phase: AggregationPhase::InitialToResult as i32,
                invocation: quantifier(func) as i32,
                ..Default::default()
            }),
            filter: None,
        })
    }

    fn translate_join(&mut self, op: OperationRef<'c, 'a>, kind: JoinType) -> Option<RelType> {
        let left = self.translate_rel(op.operand(0).ok()?)?;
        let right = self.translate_rel(op.operand(1).ok()?)?;
        let region = self.translate_region(op)?;
        let Some(expression) = self.yielded(op, &region)?.into_iter().next() else {
            self.unsupported(op, "a join needs a condition to match its rows on");
            return None;
        };

        Some(RelType::Join(Box::new(JoinRel {
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
            expression: Some(Box::new(expression)),
            r#type: kind as i32,
            ..Default::default()
        })))
    }
}

enum Projection {
    /// `select`: the row becomes what the region computed.
    Replace,
    /// `extend`: what the region computed joins the row.
    Append,
}

fn emit(output_mapping: Vec<i32>) -> Option<RelCommon> {
    Some(RelCommon {
        emit_kind: Some(EmitKind::Emit(Emit { output_mapping })),
        ..Default::default()
    })
}

fn join_type(kind: JoinKind) -> JoinType {
    match kind {
        JoinKind::Inner => JoinType::Inner,
        JoinKind::Left => JoinType::Left,
        JoinKind::Right => JoinType::Right,
        JoinKind::Full => JoinType::Outer,
    }
}

/// `count_distinct` counts the distinct values it is given; every other
/// aggregate takes them all.
fn quantifier(func: AggFunc) -> AggregationInvocation {
    match func {
        AggFunc::CountDistinct => AggregationInvocation::Distinct,
        AggFunc::Count
        | AggFunc::Sum
        | AggFunc::Min
        | AggFunc::Max
        | AggFunc::Avg
        | AggFunc::External(_) => AggregationInvocation::All,
    }
}
