//! The stages. yzr is already the shape a plan wants — the lowering turned
//! `drop`, `rename`, `set` and `distinct` into projections and groupings —
//! so each op here is one Substrait relation.

use melior::ir::operation::{OperationLike, OperationRef};
use melior::ir::{Value, ValueLike};
use rustc_hash::FxHashMap;
use substrait::proto::{
    AggregateFunction, AggregateRel, AggregationPhase, Expression, FetchRel, FilterRel, JoinRel,
    NamedStruct, ProjectRel, ReadRel, Rel,
    aggregate_rel::{Grouping, Measure},
    expression::literal::LiteralType,
    fetch_rel::{CountMode, OffsetMode},
    join_rel::JoinType,
    read_rel::{NamedTable, ReadType},
    rel::RelType,
    r#type,
};
use yuzu_mlir::attributes::JoinKind;
use yuzu_mlir::ir::operation::{OperationCast, OperationExt};
use yuzu_mlir::ir::value::{ValueExt, ValueId};
use yuzu_mlir::ops::yzr::YzrOp;

use crate::proto::{emit_common, field_index, literal, nullable, selection};
use crate::translate::expr::{Application, Region, yielded};
use crate::translate::functions;
use crate::translate::types::emit_type;
use crate::translate::{Translator, report};

/// What a projection keeps of its input's columns.
#[derive(Clone, Copy)]
enum Projection {
    /// `select`: the row becomes what the region computed.
    Replace,
    /// `extend`: what the region computed joins the row.
    Append,
}

impl<'c, 'a> Translator<'c, 'a, '_> {
    /// The relation a value stands for. A value two stages read is walked
    /// once and written out at each use, since Substrait nests.
    pub(crate) fn translate_rel(&mut self, value: Value<'c, 'a>) -> Option<Rel> {
        if let Some(translated) = self.translated.get(&value.id()) {
            return Some(translated.clone());
        }

        let op = Self::producer(value).expect("a relation comes out of a stage");
        let Some(stage) = op.as_yzr() else {
            report(op, "this is not a relation");
            return None;
        };
        let rel_type = match stage {
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
                    .map(|index| {
                        let key = dense
                            .element(index)
                            .expect("an index below the length reads");
                        i32::try_from(key).expect("a grouping key is a field index")
                    })
                    .collect();
                self.translate_aggregate(op, &keys)?
            }
            YzrOp::Join(join) => {
                let kind = join_type(join.kind());
                self.translate_join(op, kind)?
            }
            YzrOp::Union(_) | YzrOp::Intersect(_) | YzrOp::Except(_) => {
                report(op, "set operations are not lowered yet");
                return None;
            }
            YzrOp::Output(_) | YzrOp::Agg(_) | YzrOp::Yield(_) => {
                report(op, "this is not a relation");
                return None;
            }
        };

        let rel = Rel {
            rel_type: Some(rel_type),
        };
        if self.shared.contains(&value.id()) {
            self.translated.insert(value.id(), rel.clone());
        }
        Some(rel)
    }

    fn translate_table(&mut self, op: OperationRef<'c, 'a>, name: &str) -> Option<RelType> {
        let Some((names, types)) = self.row(op.first_result().r#type()) else {
            report(op, "the table's row is not declared");
            return None;
        };
        let mut fields = Vec::with_capacity(types.len());
        for ty in types {
            let Some(field) = emit_type(ty) else {
                report(op, "this column has no Substrait type");
                return None;
            };

            fields.push(field);
        }

        Some(RelType::Read(Box::new(ReadRel {
            base_schema: Some(NamedStruct {
                names: names.iter().map(std::string::ToString::to_string).collect(),
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
        let Some(condition) = yielded(op, &region)?.into_iter().next() else {
            report(op, "`where` has no predicate to filter on");
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
        let expressions = yielded(op, &region)?;
        let computed = width..width + expressions.len();
        let output_mapping = match kind {
            Projection::Replace => computed.map(field_index).collect(),
            Projection::Append => (0..width).chain(computed).map(field_index).collect(),
        };

        Some(RelType::Project(Box::new(ProjectRel {
            common: Some(emit_common(output_mapping)),
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
    /// are the region's `yzr.agg` ops. A grouping with no measures is what
    /// `distinct` became.
    fn translate_aggregate(&mut self, op: OperationRef<'c, 'a>, keys: &[i32]) -> Option<RelType> {
        let (input, _) = self.translate_input(op)?;
        let region = self.translate_region(op)?;
        let measures = self.translate_measures(&region)?;

        Some(RelType::Aggregate(Box::new(AggregateRel {
            input: Some(Box::new(input)),
            grouping_expressions: keys.iter().map(|&key| selection(key)).collect(),
            groupings: vec![Grouping {
                expression_references: (0..u32::try_from(keys.len())
                    .expect("a grouping has fewer than 2^32 keys"))
                    .collect(),
                ..Default::default()
            }],
            measures,
            ..Default::default()
        })))
    }

    /// The measures are what the region yields, not which operations it
    /// holds: two items may name the same measure, and common subexpression
    /// elimination leaves one operation yielded twice.
    fn translate_measures(&mut self, region: &Region<'c, 'a>) -> Option<Vec<Measure>> {
        let mut measures = Vec::with_capacity(region.yielded.len());
        for &value in &region.yielded {
            let Some(op) = Self::producer(value) else {
                report(
                    region.op,
                    "a grouping yields measures, and a column is not one",
                );
                return None;
            };
            let Some(YzrOp::Agg(measure)) = op.as_yzr() else {
                report(op, "a grouping yields measures, and this is not one");
                return None;
            };

            let func = functions::of_aggregate(measure.r#fn().value());
            let arguments: Vec<Value<'c, '_>> = measure.values().collect();
            measures.push(self.translate_measure(op, &func, &arguments, &region.values)?);
        }

        Some(measures)
    }

    fn translate_measure(
        &mut self,
        op: OperationRef<'c, '_>,
        func: &functions::Aggregate,
        arguments: &[Value<'c, '_>],
        values: &FxHashMap<ValueId, Expression>,
    ) -> Option<Measure> {
        let Application {
            anchor,
            arguments,
            output,
        } = self.translate_arguments(op, func.urn, &func.base, arguments, values)?;
        Some(Measure {
            measure: Some(AggregateFunction {
                function_reference: anchor,
                arguments,
                output_type: Some(output),
                phase: AggregationPhase::InitialToResult as i32,
                invocation: func.invocation as i32,
                ..Default::default()
            }),
            filter: None,
        })
    }

    fn translate_join(&mut self, op: OperationRef<'c, 'a>, kind: JoinType) -> Option<RelType> {
        let left = self.translate_rel(op.operand(0).expect("a verified join has a left input"))?;
        let right =
            self.translate_rel(op.operand(1).expect("a verified join has a right input"))?;
        let region = self.translate_region(op)?;
        let Some(expression) = yielded(op, &region)?.into_iter().next() else {
            report(op, "a join needs a condition to match its rows on");
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

    /// The relation a stage reads, and how many columns it carries.
    fn translate_input(&mut self, op: OperationRef<'c, 'a>) -> Option<(Rel, usize)> {
        let value = op
            .try_first_operand()
            .expect("a verified stage has its input");
        let Some(width) = self.width(value) else {
            report(op, "the input's row is not declared");
            return None;
        };
        Some((self.translate_rel(value)?, width))
    }
}

fn join_type(kind: JoinKind) -> JoinType {
    match kind {
        JoinKind::Inner => JoinType::Inner,
        JoinKind::Left => JoinType::Left,
        JoinKind::Right => JoinType::Right,
        JoinKind::Full => JoinType::Outer,
    }
}
