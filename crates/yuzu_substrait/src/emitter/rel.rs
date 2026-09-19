use substrait::proto::{
    AggregateFunction, AggregateRel, AggregationPhase, Expression, FetchRel, FilterRel,
    FunctionArgument, JoinRel, NamedStruct, ProjectRel, ReadRel, Rel,
    aggregate_function::AggregationInvocation,
    aggregate_rel::{Grouping, Measure},
    expression::{RexType, ScalarFunction, literal::LiteralType},
    fetch_rel::{CountMode, OffsetMode},
    function_argument::ArgType,
    join_rel::JoinType,
    read_rel::{NamedTable, ReadType},
    rel::RelType,
    r#type,
};
use yuzu_core::adt::SymbolId;
use yuzu_plan::{
    JoinCondition, JoinKey, JoinKind, Measure as PlanMeasure, Rel as PlanRel, RelId, SelectItem,
    SetItem,
};
use yuzu_types::TypeId;

use crate::emitter::types::{emit_type, row_columns, type_code};
use crate::emitter::{GraphEmitter, Unsupported};
use crate::extensions::{BOOLEAN_URN, COMPARISON_URN, aggregate_target};
use crate::proto::{emit_common, literal, nullable, selection};

impl GraphEmitter<'_> {
    pub(crate) fn emit_rel(&mut self, id: RelId) -> Result<Rel, Unsupported> {
        let inputs = self.graph.inputs(id);
        let rel_type = match self.graph.plan().rel(id).clone() {
            PlanRel::From { relation, ty, .. } => self.emit_from(relation, ty),
            PlanRel::Join {
                kind, condition, ..
            } => self.emit_join(inputs[0], inputs[1], kind, condition)?,
            PlanRel::Select { items, .. } => self.emit_select(inputs[0], &items)?,
            PlanRel::Where { predicate, .. } => self.emit_where(inputs[0], predicate)?,
            PlanRel::Distinct { ty, .. } => self.emit_distinct(inputs[0], ty)?,
            PlanRel::Drop { columns, .. } => self.emit_drop(inputs[0], &columns)?,
            // Rename is positionally a no-op: the new names live in the output
            // row type and surface as the plan's output names.
            PlanRel::Rename { .. } => return self.emit_rel(inputs[0]),
            PlanRel::Extend { items, .. } => self.emit_extend(inputs[0], &items)?,
            PlanRel::Set { items, .. } => self.emit_set(inputs[0], &items)?,
            PlanRel::Limit { count, offset, .. } => self.emit_limit(inputs[0], count, offset)?,
            // `as` renames the row, which lives in the type; the plan is its
            // input unchanged.
            PlanRel::Alias { .. } => return self.emit_rel(inputs[0]),
            PlanRel::Aggregate {
                groupings,
                measures,
                ..
            } => self.emit_aggregate(inputs[0], &groupings, &measures)?,
        };
        Ok(Rel {
            rel_type: Some(rel_type),
        })
    }

    fn emit_from(&mut self, relation: SymbolId, ty: TypeId) -> RelType {
        let fields = row_columns(self.types, ty);
        let names = fields
            .iter()
            .map(|column| self.interner.text(column.name).to_string())
            .collect();
        let types = fields
            .iter()
            .map(|column| emit_type(self.types, column.ty))
            .collect();
        RelType::Read(Box::new(ReadRel {
            base_schema: Some(NamedStruct {
                names,
                r#struct: Some(r#type::Struct {
                    types,
                    nullability: nullable(),
                    ..Default::default()
                }),
            }),
            read_type: Some(ReadType::NamedTable(NamedTable {
                names: vec![self.interner.text(relation).to_string()],
                ..Default::default()
            })),
            ..Default::default()
        }))
    }

    fn emit_join(
        &mut self,
        left: RelId,
        right: RelId,
        kind: JoinKind,
        condition: Option<JoinCondition>,
    ) -> Result<RelType, Unsupported> {
        let left_ty = self.graph.plan().rel(left).ty();
        let left_width = row_columns(self.types, left_ty).len() as i32;

        let left_rel = self.emit_rel(left)?;
        let right_rel = self.emit_rel(right)?;

        let expression = match condition {
            // An `on` condition indexes the concatenated row already.
            Some(JoinCondition::On(predicate)) => self.emit_expr(predicate)?,
            // `using` only constrains the rows; both sides' columns carry
            // through, so the join emits its inputs concatenated.
            Some(JoinCondition::Using(keys)) => self.emit_using(left_ty, left_width, &keys),
            None => unreachable!("no source produces a cross join yet"),
        };

        Ok(RelType::Join(Box::new(JoinRel {
            left: Some(Box::new(left_rel)),
            right: Some(Box::new(right_rel)),
            expression: Some(Box::new(expression)),
            r#type: join_type(kind) as i32,
            ..Default::default()
        })))
    }

    fn emit_using(&mut self, left_ty: TypeId, left_width: i32, keys: &[JoinKey]) -> Expression {
        let mut condition: Option<Expression> = None;
        for key in keys {
            let equality = self.emit_using_equality(left_ty, left_width, *key);
            condition = Some(match condition {
                Some(left) => self.emit_and(left, equality),
                None => equality,
            });
        }
        condition.expect("a `using` clause always has a column")
    }

    fn emit_using_equality(
        &mut self,
        left_ty: TypeId,
        left_width: i32,
        key: JoinKey,
    ) -> Expression {
        let code = type_code(
            self.types,
            row_columns(self.types, left_ty)[key.left as usize].ty,
        );
        let anchor = self
            .extensions
            .register(COMPARISON_URN, format!("equal:{code}_{code}"));
        self.emit_bool_function(
            anchor,
            vec![
                selection(key.left as i32),
                selection(left_width + key.right as i32),
            ],
        )
    }

    fn emit_and(&mut self, left: Expression, right: Expression) -> Expression {
        let anchor = self
            .extensions
            .register(BOOLEAN_URN, "and:bool_bool".to_string());
        self.emit_bool_function(anchor, vec![left, right])
    }

    fn emit_bool_function(&mut self, anchor: u32, arguments: Vec<Expression>) -> Expression {
        Expression {
            rex_type: Some(RexType::ScalarFunction(ScalarFunction {
                function_reference: anchor,
                output_type: Some(emit_type(self.types, self.types.bool_ty())),
                arguments: arguments
                    .into_iter()
                    .map(|value| FunctionArgument {
                        arg_type: Some(ArgType::Value(value)),
                    })
                    .collect(),
                ..Default::default()
            })),
        }
    }

    fn emit_select(&mut self, input: RelId, items: &[SelectItem]) -> Result<RelType, Unsupported> {
        let input_columns = self.width(input);
        let input = self.emit_rel(input)?;
        let expressions = self.emit_select_items(items)?;
        let output_mapping = (input_columns..input_columns + expressions.len() as i32).collect();
        Ok(RelType::Project(Box::new(ProjectRel {
            common: emit_common(output_mapping),
            input: Some(Box::new(input)),
            expressions,
            ..Default::default()
        })))
    }

    fn emit_where(
        &mut self,
        input: RelId,
        predicate: yuzu_plan::ExprId,
    ) -> Result<RelType, Unsupported> {
        let input = self.emit_rel(input)?;
        let condition = self.emit_expr(predicate)?;
        Ok(RelType::Filter(Box::new(FilterRel {
            input: Some(Box::new(input)),
            condition: Some(Box::new(condition)),
            ..Default::default()
        })))
    }

    fn emit_distinct(&mut self, input: RelId, ty: TypeId) -> Result<RelType, Unsupported> {
        let columns = row_columns(self.types, ty).len() as i32;
        let input = self.emit_rel(input)?;
        Ok(RelType::Aggregate(Box::new(AggregateRel {
            input: Some(Box::new(input)),
            grouping_expressions: (0..columns).map(selection).collect(),
            groupings: vec![Grouping {
                expression_references: (0..columns as u32).collect(),
                ..Default::default()
            }],
            measures: Vec::new(),
            ..Default::default()
        })))
    }

    fn emit_aggregate(
        &mut self,
        input: RelId,
        groupings: &[u32],
        measures: &[PlanMeasure],
    ) -> Result<RelType, Unsupported> {
        let input = self.emit_rel(input)?;
        let measures = measures
            .iter()
            .map(|measure| self.emit_measure(measure))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(RelType::Aggregate(Box::new(AggregateRel {
            input: Some(Box::new(input)),
            grouping_expressions: groupings.iter().map(|&key| selection(key as i32)).collect(),
            groupings: vec![Grouping {
                expression_references: (0..groupings.len() as u32).collect(),
                ..Default::default()
            }],
            measures,
            ..Default::default()
        })))
    }

    fn emit_measure(&mut self, measure: &PlanMeasure) -> Result<Measure, Unsupported> {
        let (urn, base) = if let yuzu_types::AggFunc::External(symbol) = measure.func {
            (
                crate::extensions::EXTERNAL_URN,
                self.interner.text(symbol).to_string(),
            )
        } else {
            let (urn, base) = aggregate_target(measure.func);
            (urn, base.to_string())
        };
        let signature: Vec<&str> = measure
            .args
            .iter()
            .map(|&arg| type_code(self.types, self.graph.plan().expr(arg).ty()))
            .collect();
        let name = format!("{base}:{}", signature.join("_"));

        let mut arguments = Vec::new();
        for &arg in measure.args.iter() {
            arguments.push(FunctionArgument {
                arg_type: Some(ArgType::Value(self.emit_expr(arg)?)),
            });
        }

        let anchor = self.extensions.register(urn, name);
        Ok(Measure {
            measure: Some(AggregateFunction {
                function_reference: anchor,
                arguments,
                output_type: Some(emit_type(self.types, measure.ty)),
                phase: AggregationPhase::InitialToResult as i32,
                invocation: quantifier(measure.func) as i32,
                ..Default::default()
            }),
            filter: None,
        })
    }

    fn emit_drop(&mut self, input: RelId, columns: &[u32]) -> Result<RelType, Unsupported> {
        let output_mapping = (0..self.width(input))
            .filter(|index| !columns.contains(&(*index as u32)))
            .collect();
        let input = self.emit_rel(input)?;
        Ok(RelType::Project(Box::new(ProjectRel {
            common: emit_common(output_mapping),
            input: Some(Box::new(input)),
            expressions: Vec::new(),
            ..Default::default()
        })))
    }

    fn emit_extend(&mut self, input: RelId, items: &[SelectItem]) -> Result<RelType, Unsupported> {
        let input_columns = self.width(input);
        let input = self.emit_rel(input)?;
        let expressions = self.emit_select_items(items)?;
        let output_mapping = (0..input_columns)
            .chain(input_columns..input_columns + expressions.len() as i32)
            .collect();
        Ok(RelType::Project(Box::new(ProjectRel {
            common: emit_common(output_mapping),
            input: Some(Box::new(input)),
            expressions,
            ..Default::default()
        })))
    }

    /// `set` keeps the row's shape: every column is emitted, with the ones it
    /// names taking a computed expression in place of the original.
    fn emit_set(&mut self, input: RelId, items: &[SetItem]) -> Result<RelType, Unsupported> {
        let width = self.width(input);
        let input = self.emit_rel(input)?;
        let expressions = items
            .iter()
            .map(|item| self.emit_expr(item.value))
            .collect::<Result<Vec<_>, _>>()?;

        // A replacement lands past the input's columns, so the mapping picks it
        // up where the original stood.
        let output_mapping = (0..width)
            .map(
                |column| match items.iter().position(|item| item.column == column as u32) {
                    Some(item) => width + item as i32,
                    None => column,
                },
            )
            .collect();

        Ok(RelType::Project(Box::new(ProjectRel {
            common: emit_common(output_mapping),
            input: Some(Box::new(input)),
            expressions,
            ..Default::default()
        })))
    }

    fn emit_limit(
        &mut self,
        input: RelId,
        count: u64,
        offset: Option<u64>,
    ) -> Result<RelType, Unsupported> {
        let input = self.emit_rel(input)?;
        Ok(RelType::Fetch(Box::new(FetchRel {
            input: Some(Box::new(input)),
            offset_mode: Some(OffsetMode::OffsetExpr(Box::new(literal(LiteralType::I64(
                offset.unwrap_or(0) as i64,
            ))))),
            count_mode: Some(CountMode::CountExpr(Box::new(literal(LiteralType::I64(
                count as i64,
            ))))),
            ..Default::default()
        })))
    }

    fn emit_select_items(&mut self, items: &[SelectItem]) -> Result<Vec<Expression>, Unsupported> {
        items.iter().map(|item| self.emit_expr(item.body)).collect()
    }

    fn width(&self, input: RelId) -> i32 {
        row_columns(self.types, self.graph.plan().rel(input).ty()).len() as i32
    }
}

/// A `_distinct` builtin is its base function under Substrait's DISTINCT
/// invocation; the dialect spells it as a separate function instead of a
/// keyword.
fn quantifier(func: yuzu_types::AggFunc) -> AggregationInvocation {
    match func {
        yuzu_types::AggFunc::CountDistinct => AggregationInvocation::Distinct,
        _ => AggregationInvocation::All,
    }
}

fn join_type(kind: JoinKind) -> JoinType {
    match kind {
        JoinKind::Inner => JoinType::Inner,
        JoinKind::Left => JoinType::Left,
        JoinKind::Right => JoinType::Right,
        JoinKind::Full => JoinType::Outer,
        JoinKind::Cross => unreachable!("no source produces a cross join yet"),
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::emitter::test_support::{TABLE, check};

    #[test]
    fn emits_external_functions_by_name() {
        check(
            &format!(
                "{TABLE}external def clamp(x: int32, low: int32) -> int32\nexternal agg def median(x: int32) -> float64\nfrom t |> aggregate median(a) as m group by b |> extend clamp(b, 1) as c"
            ),
            expect![[r#"
                {
                  "version": {
                    "minorNumber": 85,
                    "producer": "yuzu"
                  },
                  "extensionUrns": [
                    {
                      "extensionUrnAnchor": 1,
                      "urn": "extension:io.yuzu:external"
                    }
                  ],
                  "extensions": [
                    {
                      "extensionFunction": {
                        "extensionUrnReference": 1,
                        "functionAnchor": 1,
                        "name": "median:i32"
                      }
                    },
                    {
                      "extensionFunction": {
                        "extensionUrnReference": 1,
                        "functionAnchor": 2,
                        "name": "clamp:i32_i32"
                      }
                    }
                  ],
                  "relations": [
                    {
                      "root": {
                        "input": {
                          "project": {
                            "common": {
                              "emit": {
                                "outputMapping": [
                                  0,
                                  1,
                                  2
                                ]
                              }
                            },
                            "input": {
                              "aggregate": {
                                "input": {
                                  "read": {
                                    "baseSchema": {
                                      "names": [
                                        "a",
                                        "b"
                                      ],
                                      "struct": {
                                        "types": [
                                          {
                                            "i32": {
                                              "nullability": "NULLABILITY_NULLABLE"
                                            }
                                          },
                                          {
                                            "i32": {
                                              "nullability": "NULLABILITY_NULLABLE"
                                            }
                                          }
                                        ],
                                        "nullability": "NULLABILITY_NULLABLE"
                                      }
                                    },
                                    "namedTable": {
                                      "names": [
                                        "t"
                                      ]
                                    }
                                  }
                                },
                                "groupings": [
                                  {
                                    "expressionReferences": [
                                      0
                                    ]
                                  }
                                ],
                                "measures": [
                                  {
                                    "measure": {
                                      "functionReference": 1,
                                      "arguments": [
                                        {
                                          "value": {
                                            "selection": {
                                              "directReference": {
                                                "structField": {}
                                              },
                                              "rootReference": {}
                                            }
                                          }
                                        }
                                      ],
                                      "outputType": {
                                        "fp64": {
                                          "nullability": "NULLABILITY_NULLABLE"
                                        }
                                      },
                                      "phase": "AGGREGATION_PHASE_INITIAL_TO_RESULT",
                                      "invocation": "AGGREGATION_INVOCATION_ALL"
                                    }
                                  }
                                ],
                                "groupingExpressions": [
                                  {
                                    "selection": {
                                      "directReference": {
                                        "structField": {
                                          "field": 1
                                        }
                                      },
                                      "rootReference": {}
                                    }
                                  }
                                ]
                              }
                            },
                            "expressions": [
                              {
                                "scalarFunction": {
                                  "functionReference": 2,
                                  "arguments": [
                                    {
                                      "value": {
                                        "selection": {
                                          "directReference": {
                                            "structField": {}
                                          },
                                          "rootReference": {}
                                        }
                                      }
                                    },
                                    {
                                      "value": {
                                        "literal": {
                                          "i32": 1
                                        }
                                      }
                                    }
                                  ],
                                  "outputType": {
                                    "i32": {
                                      "nullability": "NULLABILITY_NULLABLE"
                                    }
                                  }
                                }
                              }
                            ]
                          }
                        },
                        "names": [
                          "b",
                          "m",
                          "c"
                        ]
                      }
                    }
                  ]
                }"#]],
        );
    }

    #[test]
    fn emits_grouped_aggregate() {
        check(
            &format!("{TABLE}from t |> aggregate sum(a) as s group by b"),
            expect![[r#"
                {
                  "version": {
                    "minorNumber": 85,
                    "producer": "yuzu"
                  },
                  "extensionUrns": [
                    {
                      "extensionUrnAnchor": 1,
                      "urn": "extension:io.substrait:functions_arithmetic"
                    }
                  ],
                  "extensions": [
                    {
                      "extensionFunction": {
                        "extensionUrnReference": 1,
                        "functionAnchor": 1,
                        "name": "sum:i32"
                      }
                    }
                  ],
                  "relations": [
                    {
                      "root": {
                        "input": {
                          "aggregate": {
                            "input": {
                              "read": {
                                "baseSchema": {
                                  "names": [
                                    "a",
                                    "b"
                                  ],
                                  "struct": {
                                    "types": [
                                      {
                                        "i32": {
                                          "nullability": "NULLABILITY_NULLABLE"
                                        }
                                      },
                                      {
                                        "i32": {
                                          "nullability": "NULLABILITY_NULLABLE"
                                        }
                                      }
                                    ],
                                    "nullability": "NULLABILITY_NULLABLE"
                                  }
                                },
                                "namedTable": {
                                  "names": [
                                    "t"
                                  ]
                                }
                              }
                            },
                            "groupings": [
                              {
                                "expressionReferences": [
                                  0
                                ]
                              }
                            ],
                            "measures": [
                              {
                                "measure": {
                                  "functionReference": 1,
                                  "arguments": [
                                    {
                                      "value": {
                                        "selection": {
                                          "directReference": {
                                            "structField": {}
                                          },
                                          "rootReference": {}
                                        }
                                      }
                                    }
                                  ],
                                  "outputType": {
                                    "i64": {
                                      "nullability": "NULLABILITY_NULLABLE"
                                    }
                                  },
                                  "phase": "AGGREGATION_PHASE_INITIAL_TO_RESULT",
                                  "invocation": "AGGREGATION_INVOCATION_ALL"
                                }
                              }
                            ],
                            "groupingExpressions": [
                              {
                                "selection": {
                                  "directReference": {
                                    "structField": {
                                      "field": 1
                                    }
                                  },
                                  "rootReference": {}
                                }
                              }
                            ]
                          }
                        },
                        "names": [
                          "b",
                          "s"
                        ]
                      }
                    }
                  ]
                }"#]],
        );
    }

    #[test]
    fn emits_count_distinct_as_distinct_invocation() {
        check(
            &format!("{TABLE}from t |> aggregate count_distinct(a) as kinds, count(a) as values"),
            expect![[r#"
                {
                  "version": {
                    "minorNumber": 85,
                    "producer": "yuzu"
                  },
                  "extensionUrns": [
                    {
                      "extensionUrnAnchor": 1,
                      "urn": "extension:io.substrait:functions_aggregate_generic"
                    }
                  ],
                  "extensions": [
                    {
                      "extensionFunction": {
                        "extensionUrnReference": 1,
                        "functionAnchor": 1,
                        "name": "count:i32"
                      }
                    },
                    {
                      "extensionFunction": {
                        "extensionUrnReference": 1,
                        "functionAnchor": 2,
                        "name": "count:i32"
                      }
                    }
                  ],
                  "relations": [
                    {
                      "root": {
                        "input": {
                          "aggregate": {
                            "input": {
                              "read": {
                                "baseSchema": {
                                  "names": [
                                    "a",
                                    "b"
                                  ],
                                  "struct": {
                                    "types": [
                                      {
                                        "i32": {
                                          "nullability": "NULLABILITY_NULLABLE"
                                        }
                                      },
                                      {
                                        "i32": {
                                          "nullability": "NULLABILITY_NULLABLE"
                                        }
                                      }
                                    ],
                                    "nullability": "NULLABILITY_NULLABLE"
                                  }
                                },
                                "namedTable": {
                                  "names": [
                                    "t"
                                  ]
                                }
                              }
                            },
                            "groupings": [
                              {}
                            ],
                            "measures": [
                              {
                                "measure": {
                                  "functionReference": 1,
                                  "arguments": [
                                    {
                                      "value": {
                                        "selection": {
                                          "directReference": {
                                            "structField": {}
                                          },
                                          "rootReference": {}
                                        }
                                      }
                                    }
                                  ],
                                  "outputType": {
                                    "i64": {
                                      "nullability": "NULLABILITY_NULLABLE"
                                    }
                                  },
                                  "phase": "AGGREGATION_PHASE_INITIAL_TO_RESULT",
                                  "invocation": "AGGREGATION_INVOCATION_DISTINCT"
                                }
                              },
                              {
                                "measure": {
                                  "functionReference": 2,
                                  "arguments": [
                                    {
                                      "value": {
                                        "selection": {
                                          "directReference": {
                                            "structField": {}
                                          },
                                          "rootReference": {}
                                        }
                                      }
                                    }
                                  ],
                                  "outputType": {
                                    "i64": {
                                      "nullability": "NULLABILITY_NULLABLE"
                                    }
                                  },
                                  "phase": "AGGREGATION_PHASE_INITIAL_TO_RESULT",
                                  "invocation": "AGGREGATION_INVOCATION_ALL"
                                }
                              }
                            ]
                          }
                        },
                        "names": [
                          "kinds",
                          "values"
                        ]
                      }
                    }
                  ]
                }"#]],
        );
    }

    #[test]
    fn emits_full_table_count() {
        check(
            &format!("{TABLE}from t |> aggregate count() as n"),
            expect![[r#"
            {
              "version": {
                "minorNumber": 85,
                "producer": "yuzu"
              },
              "extensionUrns": [
                {
                  "extensionUrnAnchor": 1,
                  "urn": "extension:io.substrait:functions_aggregate_generic"
                }
              ],
              "extensions": [
                {
                  "extensionFunction": {
                    "extensionUrnReference": 1,
                    "functionAnchor": 1,
                    "name": "count:"
                  }
                }
              ],
              "relations": [
                {
                  "root": {
                    "input": {
                      "aggregate": {
                        "input": {
                          "read": {
                            "baseSchema": {
                              "names": [
                                "a",
                                "b"
                              ],
                              "struct": {
                                "types": [
                                  {
                                    "i32": {
                                      "nullability": "NULLABILITY_NULLABLE"
                                    }
                                  },
                                  {
                                    "i32": {
                                      "nullability": "NULLABILITY_NULLABLE"
                                    }
                                  }
                                ],
                                "nullability": "NULLABILITY_NULLABLE"
                              }
                            },
                            "namedTable": {
                              "names": [
                                "t"
                              ]
                            }
                          }
                        },
                        "groupings": [
                          {}
                        ],
                        "measures": [
                          {
                            "measure": {
                              "functionReference": 1,
                              "outputType": {
                                "i64": {
                                  "nullability": "NULLABILITY_NULLABLE"
                                }
                              },
                              "phase": "AGGREGATION_PHASE_INITIAL_TO_RESULT",
                              "invocation": "AGGREGATION_INVOCATION_ALL"
                            }
                          }
                        ]
                      }
                    },
                    "names": [
                      "n"
                    ]
                  }
                }
              ]
            }"#]],
        );
    }

    #[test]
    fn emits_composite_aggregate_as_aggregate_then_project() {
        check(
            &format!("{TABLE}from t |> aggregate max(a) - min(a) as spread group by b"),
            expect![[r#"
                {
                  "version": {
                    "minorNumber": 85,
                    "producer": "yuzu"
                  },
                  "extensionUrns": [
                    {
                      "extensionUrnAnchor": 1,
                      "urn": "extension:io.substrait:functions_arithmetic"
                    }
                  ],
                  "extensions": [
                    {
                      "extensionFunction": {
                        "extensionUrnReference": 1,
                        "functionAnchor": 1,
                        "name": "max:i32"
                      }
                    },
                    {
                      "extensionFunction": {
                        "extensionUrnReference": 1,
                        "functionAnchor": 2,
                        "name": "min:i32"
                      }
                    },
                    {
                      "extensionFunction": {
                        "extensionUrnReference": 1,
                        "functionAnchor": 3,
                        "name": "subtract:i32_i32"
                      }
                    }
                  ],
                  "relations": [
                    {
                      "root": {
                        "input": {
                          "project": {
                            "common": {
                              "emit": {
                                "outputMapping": [
                                  3,
                                  4
                                ]
                              }
                            },
                            "input": {
                              "aggregate": {
                                "input": {
                                  "read": {
                                    "baseSchema": {
                                      "names": [
                                        "a",
                                        "b"
                                      ],
                                      "struct": {
                                        "types": [
                                          {
                                            "i32": {
                                              "nullability": "NULLABILITY_NULLABLE"
                                            }
                                          },
                                          {
                                            "i32": {
                                              "nullability": "NULLABILITY_NULLABLE"
                                            }
                                          }
                                        ],
                                        "nullability": "NULLABILITY_NULLABLE"
                                      }
                                    },
                                    "namedTable": {
                                      "names": [
                                        "t"
                                      ]
                                    }
                                  }
                                },
                                "groupings": [
                                  {
                                    "expressionReferences": [
                                      0
                                    ]
                                  }
                                ],
                                "measures": [
                                  {
                                    "measure": {
                                      "functionReference": 1,
                                      "arguments": [
                                        {
                                          "value": {
                                            "selection": {
                                              "directReference": {
                                                "structField": {}
                                              },
                                              "rootReference": {}
                                            }
                                          }
                                        }
                                      ],
                                      "outputType": {
                                        "i32": {
                                          "nullability": "NULLABILITY_NULLABLE"
                                        }
                                      },
                                      "phase": "AGGREGATION_PHASE_INITIAL_TO_RESULT",
                                      "invocation": "AGGREGATION_INVOCATION_ALL"
                                    }
                                  },
                                  {
                                    "measure": {
                                      "functionReference": 2,
                                      "arguments": [
                                        {
                                          "value": {
                                            "selection": {
                                              "directReference": {
                                                "structField": {}
                                              },
                                              "rootReference": {}
                                            }
                                          }
                                        }
                                      ],
                                      "outputType": {
                                        "i32": {
                                          "nullability": "NULLABILITY_NULLABLE"
                                        }
                                      },
                                      "phase": "AGGREGATION_PHASE_INITIAL_TO_RESULT",
                                      "invocation": "AGGREGATION_INVOCATION_ALL"
                                    }
                                  }
                                ],
                                "groupingExpressions": [
                                  {
                                    "selection": {
                                      "directReference": {
                                        "structField": {
                                          "field": 1
                                        }
                                      },
                                      "rootReference": {}
                                    }
                                  }
                                ]
                              }
                            },
                            "expressions": [
                              {
                                "selection": {
                                  "directReference": {
                                    "structField": {}
                                  },
                                  "rootReference": {}
                                }
                              },
                              {
                                "scalarFunction": {
                                  "functionReference": 3,
                                  "arguments": [
                                    {
                                      "value": {
                                        "selection": {
                                          "directReference": {
                                            "structField": {
                                              "field": 1
                                            }
                                          },
                                          "rootReference": {}
                                        }
                                      }
                                    },
                                    {
                                      "value": {
                                        "selection": {
                                          "directReference": {
                                            "structField": {
                                              "field": 2
                                            }
                                          },
                                          "rootReference": {}
                                        }
                                      }
                                    }
                                  ],
                                  "outputType": {
                                    "i32": {
                                      "nullability": "NULLABILITY_NULLABLE"
                                    }
                                  }
                                }
                              }
                            ]
                          }
                        },
                        "names": [
                          "b",
                          "spread"
                        ]
                      }
                    }
                  ]
                }"#]],
        );
    }

    #[test]
    fn emits_read_and_project() {
        check(
            &format!("{TABLE}from t |> select a, a + b as s"),
            expect![[r#"
            {
              "version": {
                "minorNumber": 85,
                "producer": "yuzu"
              },
              "extensionUrns": [
                {
                  "extensionUrnAnchor": 1,
                  "urn": "extension:io.substrait:functions_arithmetic"
                }
              ],
              "extensions": [
                {
                  "extensionFunction": {
                    "extensionUrnReference": 1,
                    "functionAnchor": 1,
                    "name": "add:i32_i32"
                  }
                }
              ],
              "relations": [
                {
                  "root": {
                    "input": {
                      "project": {
                        "common": {
                          "emit": {
                            "outputMapping": [
                              2,
                              3
                            ]
                          }
                        },
                        "input": {
                          "read": {
                            "baseSchema": {
                              "names": [
                                "a",
                                "b"
                              ],
                              "struct": {
                                "types": [
                                  {
                                    "i32": {
                                      "nullability": "NULLABILITY_NULLABLE"
                                    }
                                  },
                                  {
                                    "i32": {
                                      "nullability": "NULLABILITY_NULLABLE"
                                    }
                                  }
                                ],
                                "nullability": "NULLABILITY_NULLABLE"
                              }
                            },
                            "namedTable": {
                              "names": [
                                "t"
                              ]
                            }
                          }
                        },
                        "expressions": [
                          {
                            "selection": {
                              "directReference": {
                                "structField": {}
                              },
                              "rootReference": {}
                            }
                          },
                          {
                            "scalarFunction": {
                              "functionReference": 1,
                              "arguments": [
                                {
                                  "value": {
                                    "selection": {
                                      "directReference": {
                                        "structField": {}
                                      },
                                      "rootReference": {}
                                    }
                                  }
                                },
                                {
                                  "value": {
                                    "selection": {
                                      "directReference": {
                                        "structField": {
                                          "field": 1
                                        }
                                      },
                                      "rootReference": {}
                                    }
                                  }
                                }
                              ],
                              "outputType": {
                                "i32": {
                                  "nullability": "NULLABILITY_NULLABLE"
                                }
                              }
                            }
                          }
                        ]
                      }
                    },
                    "names": [
                      "a",
                      "s"
                    ]
                  }
                }
              ]
            }"#]],
        );
    }

    const JOIN_TABLES: &str = "struct Other { a: int32, c: int32 }\ntable u = Other\nstruct Codes { c: int32, d: int32 }\ntable v = Codes\n";

    #[test]
    fn emits_inner_join_on_condition() {
        check(
            &format!("{TABLE}{JOIN_TABLES}from t e |> join v x on e.b == x.c"),
            expect![[r#"
                {
                  "version": {
                    "minorNumber": 85,
                    "producer": "yuzu"
                  },
                  "extensionUrns": [
                    {
                      "extensionUrnAnchor": 1,
                      "urn": "extension:io.substrait:functions_comparison"
                    }
                  ],
                  "extensions": [
                    {
                      "extensionFunction": {
                        "extensionUrnReference": 1,
                        "functionAnchor": 1,
                        "name": "equal:i32_i32"
                      }
                    }
                  ],
                  "relations": [
                    {
                      "root": {
                        "input": {
                          "join": {
                            "left": {
                              "read": {
                                "baseSchema": {
                                  "names": [
                                    "a",
                                    "b"
                                  ],
                                  "struct": {
                                    "types": [
                                      {
                                        "i32": {
                                          "nullability": "NULLABILITY_NULLABLE"
                                        }
                                      },
                                      {
                                        "i32": {
                                          "nullability": "NULLABILITY_NULLABLE"
                                        }
                                      }
                                    ],
                                    "nullability": "NULLABILITY_NULLABLE"
                                  }
                                },
                                "namedTable": {
                                  "names": [
                                    "t"
                                  ]
                                }
                              }
                            },
                            "right": {
                              "read": {
                                "baseSchema": {
                                  "names": [
                                    "c",
                                    "d"
                                  ],
                                  "struct": {
                                    "types": [
                                      {
                                        "i32": {
                                          "nullability": "NULLABILITY_NULLABLE"
                                        }
                                      },
                                      {
                                        "i32": {
                                          "nullability": "NULLABILITY_NULLABLE"
                                        }
                                      }
                                    ],
                                    "nullability": "NULLABILITY_NULLABLE"
                                  }
                                },
                                "namedTable": {
                                  "names": [
                                    "v"
                                  ]
                                }
                              }
                            },
                            "expression": {
                              "scalarFunction": {
                                "functionReference": 1,
                                "arguments": [
                                  {
                                    "value": {
                                      "selection": {
                                        "directReference": {
                                          "structField": {
                                            "field": 1
                                          }
                                        },
                                        "rootReference": {}
                                      }
                                    }
                                  },
                                  {
                                    "value": {
                                      "selection": {
                                        "directReference": {
                                          "structField": {
                                            "field": 2
                                          }
                                        },
                                        "rootReference": {}
                                      }
                                    }
                                  }
                                ],
                                "outputType": {
                                  "bool": {
                                    "nullability": "NULLABILITY_NULLABLE"
                                  }
                                }
                              }
                            },
                            "type": "JOIN_TYPE_INNER"
                          }
                        },
                        "names": [
                          "a",
                          "b",
                          "c",
                          "d"
                        ]
                      }
                    }
                  ]
                }"#]],
        );
    }

    #[test]
    fn emits_left_join_using_one_key() {
        check(
            &format!("{TABLE}{JOIN_TABLES}from t |> left join u using (a)"),
            expect![[r#"
                {
                  "version": {
                    "minorNumber": 85,
                    "producer": "yuzu"
                  },
                  "extensionUrns": [
                    {
                      "extensionUrnAnchor": 1,
                      "urn": "extension:io.substrait:functions_comparison"
                    }
                  ],
                  "extensions": [
                    {
                      "extensionFunction": {
                        "extensionUrnReference": 1,
                        "functionAnchor": 1,
                        "name": "equal:i32_i32"
                      }
                    }
                  ],
                  "relations": [
                    {
                      "root": {
                        "input": {
                          "join": {
                            "left": {
                              "read": {
                                "baseSchema": {
                                  "names": [
                                    "a",
                                    "b"
                                  ],
                                  "struct": {
                                    "types": [
                                      {
                                        "i32": {
                                          "nullability": "NULLABILITY_NULLABLE"
                                        }
                                      },
                                      {
                                        "i32": {
                                          "nullability": "NULLABILITY_NULLABLE"
                                        }
                                      }
                                    ],
                                    "nullability": "NULLABILITY_NULLABLE"
                                  }
                                },
                                "namedTable": {
                                  "names": [
                                    "t"
                                  ]
                                }
                              }
                            },
                            "right": {
                              "read": {
                                "baseSchema": {
                                  "names": [
                                    "a",
                                    "c"
                                  ],
                                  "struct": {
                                    "types": [
                                      {
                                        "i32": {
                                          "nullability": "NULLABILITY_NULLABLE"
                                        }
                                      },
                                      {
                                        "i32": {
                                          "nullability": "NULLABILITY_NULLABLE"
                                        }
                                      }
                                    ],
                                    "nullability": "NULLABILITY_NULLABLE"
                                  }
                                },
                                "namedTable": {
                                  "names": [
                                    "u"
                                  ]
                                }
                              }
                            },
                            "expression": {
                              "scalarFunction": {
                                "functionReference": 1,
                                "arguments": [
                                  {
                                    "value": {
                                      "selection": {
                                        "directReference": {
                                          "structField": {}
                                        },
                                        "rootReference": {}
                                      }
                                    }
                                  },
                                  {
                                    "value": {
                                      "selection": {
                                        "directReference": {
                                          "structField": {
                                            "field": 2
                                          }
                                        },
                                        "rootReference": {}
                                      }
                                    }
                                  }
                                ],
                                "outputType": {
                                  "bool": {
                                    "nullability": "NULLABILITY_NULLABLE"
                                  }
                                }
                              }
                            },
                            "type": "JOIN_TYPE_LEFT"
                          }
                        },
                        "names": [
                          "a",
                          "b",
                          "a",
                          "c"
                        ]
                      }
                    }
                  ]
                }"#]],
        );
    }

    #[test]
    fn places_a_using_joins_right_columns_after_its_dropped_keys() {
        check(
            &format!("{TABLE}{JOIN_TABLES}from t |> join u d using (a) |> select d.c"),
            expect![[r#"
                {
                  "version": {
                    "minorNumber": 85,
                    "producer": "yuzu"
                  },
                  "extensionUrns": [
                    {
                      "extensionUrnAnchor": 1,
                      "urn": "extension:io.substrait:functions_comparison"
                    }
                  ],
                  "extensions": [
                    {
                      "extensionFunction": {
                        "extensionUrnReference": 1,
                        "functionAnchor": 1,
                        "name": "equal:i32_i32"
                      }
                    }
                  ],
                  "relations": [
                    {
                      "root": {
                        "input": {
                          "project": {
                            "common": {
                              "emit": {
                                "outputMapping": [
                                  4
                                ]
                              }
                            },
                            "input": {
                              "join": {
                                "left": {
                                  "read": {
                                    "baseSchema": {
                                      "names": [
                                        "a",
                                        "b"
                                      ],
                                      "struct": {
                                        "types": [
                                          {
                                            "i32": {
                                              "nullability": "NULLABILITY_NULLABLE"
                                            }
                                          },
                                          {
                                            "i32": {
                                              "nullability": "NULLABILITY_NULLABLE"
                                            }
                                          }
                                        ],
                                        "nullability": "NULLABILITY_NULLABLE"
                                      }
                                    },
                                    "namedTable": {
                                      "names": [
                                        "t"
                                      ]
                                    }
                                  }
                                },
                                "right": {
                                  "read": {
                                    "baseSchema": {
                                      "names": [
                                        "a",
                                        "c"
                                      ],
                                      "struct": {
                                        "types": [
                                          {
                                            "i32": {
                                              "nullability": "NULLABILITY_NULLABLE"
                                            }
                                          },
                                          {
                                            "i32": {
                                              "nullability": "NULLABILITY_NULLABLE"
                                            }
                                          }
                                        ],
                                        "nullability": "NULLABILITY_NULLABLE"
                                      }
                                    },
                                    "namedTable": {
                                      "names": [
                                        "u"
                                      ]
                                    }
                                  }
                                },
                                "expression": {
                                  "scalarFunction": {
                                    "functionReference": 1,
                                    "arguments": [
                                      {
                                        "value": {
                                          "selection": {
                                            "directReference": {
                                              "structField": {}
                                            },
                                            "rootReference": {}
                                          }
                                        }
                                      },
                                      {
                                        "value": {
                                          "selection": {
                                            "directReference": {
                                              "structField": {
                                                "field": 2
                                              }
                                            },
                                            "rootReference": {}
                                          }
                                        }
                                      }
                                    ],
                                    "outputType": {
                                      "bool": {
                                        "nullability": "NULLABILITY_NULLABLE"
                                      }
                                    }
                                  }
                                },
                                "type": "JOIN_TYPE_INNER"
                              }
                            },
                            "expressions": [
                              {
                                "selection": {
                                  "directReference": {
                                    "structField": {
                                      "field": 3
                                    }
                                  },
                                  "rootReference": {}
                                }
                              }
                            ]
                          }
                        },
                        "names": [
                          "c"
                        ]
                      }
                    }
                  ]
                }"#]],
        );
    }

    #[test]
    fn resolves_the_carried_using_key_to_the_emitted_column() {
        check(
            &format!("{TABLE}{JOIN_TABLES}from t e |> join u d using (a) |> select e.a"),
            expect![[r#"
                {
                  "version": {
                    "minorNumber": 85,
                    "producer": "yuzu"
                  },
                  "extensionUrns": [
                    {
                      "extensionUrnAnchor": 1,
                      "urn": "extension:io.substrait:functions_comparison"
                    }
                  ],
                  "extensions": [
                    {
                      "extensionFunction": {
                        "extensionUrnReference": 1,
                        "functionAnchor": 1,
                        "name": "equal:i32_i32"
                      }
                    }
                  ],
                  "relations": [
                    {
                      "root": {
                        "input": {
                          "project": {
                            "common": {
                              "emit": {
                                "outputMapping": [
                                  4
                                ]
                              }
                            },
                            "input": {
                              "join": {
                                "left": {
                                  "read": {
                                    "baseSchema": {
                                      "names": [
                                        "a",
                                        "b"
                                      ],
                                      "struct": {
                                        "types": [
                                          {
                                            "i32": {
                                              "nullability": "NULLABILITY_NULLABLE"
                                            }
                                          },
                                          {
                                            "i32": {
                                              "nullability": "NULLABILITY_NULLABLE"
                                            }
                                          }
                                        ],
                                        "nullability": "NULLABILITY_NULLABLE"
                                      }
                                    },
                                    "namedTable": {
                                      "names": [
                                        "t"
                                      ]
                                    }
                                  }
                                },
                                "right": {
                                  "read": {
                                    "baseSchema": {
                                      "names": [
                                        "a",
                                        "c"
                                      ],
                                      "struct": {
                                        "types": [
                                          {
                                            "i32": {
                                              "nullability": "NULLABILITY_NULLABLE"
                                            }
                                          },
                                          {
                                            "i32": {
                                              "nullability": "NULLABILITY_NULLABLE"
                                            }
                                          }
                                        ],
                                        "nullability": "NULLABILITY_NULLABLE"
                                      }
                                    },
                                    "namedTable": {
                                      "names": [
                                        "u"
                                      ]
                                    }
                                  }
                                },
                                "expression": {
                                  "scalarFunction": {
                                    "functionReference": 1,
                                    "arguments": [
                                      {
                                        "value": {
                                          "selection": {
                                            "directReference": {
                                              "structField": {}
                                            },
                                            "rootReference": {}
                                          }
                                        }
                                      },
                                      {
                                        "value": {
                                          "selection": {
                                            "directReference": {
                                              "structField": {
                                                "field": 2
                                              }
                                            },
                                            "rootReference": {}
                                          }
                                        }
                                      }
                                    ],
                                    "outputType": {
                                      "bool": {
                                        "nullability": "NULLABILITY_NULLABLE"
                                      }
                                    }
                                  }
                                },
                                "type": "JOIN_TYPE_INNER"
                              }
                            },
                            "expressions": [
                              {
                                "selection": {
                                  "directReference": {
                                    "structField": {}
                                  },
                                  "rootReference": {}
                                }
                              }
                            ]
                          }
                        },
                        "names": [
                          "a"
                        ]
                      }
                    }
                  ]
                }"#]],
        );
    }

    #[test]
    fn emits_right_join_on_condition() {
        check(
            &format!("{TABLE}{JOIN_TABLES}from t |> right join v on b == c"),
            expect![[r#"
                {
                  "version": {
                    "minorNumber": 85,
                    "producer": "yuzu"
                  },
                  "extensionUrns": [
                    {
                      "extensionUrnAnchor": 1,
                      "urn": "extension:io.substrait:functions_comparison"
                    }
                  ],
                  "extensions": [
                    {
                      "extensionFunction": {
                        "extensionUrnReference": 1,
                        "functionAnchor": 1,
                        "name": "equal:i32_i32"
                      }
                    }
                  ],
                  "relations": [
                    {
                      "root": {
                        "input": {
                          "join": {
                            "left": {
                              "read": {
                                "baseSchema": {
                                  "names": [
                                    "a",
                                    "b"
                                  ],
                                  "struct": {
                                    "types": [
                                      {
                                        "i32": {
                                          "nullability": "NULLABILITY_NULLABLE"
                                        }
                                      },
                                      {
                                        "i32": {
                                          "nullability": "NULLABILITY_NULLABLE"
                                        }
                                      }
                                    ],
                                    "nullability": "NULLABILITY_NULLABLE"
                                  }
                                },
                                "namedTable": {
                                  "names": [
                                    "t"
                                  ]
                                }
                              }
                            },
                            "right": {
                              "read": {
                                "baseSchema": {
                                  "names": [
                                    "c",
                                    "d"
                                  ],
                                  "struct": {
                                    "types": [
                                      {
                                        "i32": {
                                          "nullability": "NULLABILITY_NULLABLE"
                                        }
                                      },
                                      {
                                        "i32": {
                                          "nullability": "NULLABILITY_NULLABLE"
                                        }
                                      }
                                    ],
                                    "nullability": "NULLABILITY_NULLABLE"
                                  }
                                },
                                "namedTable": {
                                  "names": [
                                    "v"
                                  ]
                                }
                              }
                            },
                            "expression": {
                              "scalarFunction": {
                                "functionReference": 1,
                                "arguments": [
                                  {
                                    "value": {
                                      "selection": {
                                        "directReference": {
                                          "structField": {
                                            "field": 1
                                          }
                                        },
                                        "rootReference": {}
                                      }
                                    }
                                  },
                                  {
                                    "value": {
                                      "selection": {
                                        "directReference": {
                                          "structField": {
                                            "field": 2
                                          }
                                        },
                                        "rootReference": {}
                                      }
                                    }
                                  }
                                ],
                                "outputType": {
                                  "bool": {
                                    "nullability": "NULLABILITY_NULLABLE"
                                  }
                                }
                              }
                            },
                            "type": "JOIN_TYPE_RIGHT"
                          }
                        },
                        "names": [
                          "a",
                          "b",
                          "c",
                          "d"
                        ]
                      }
                    }
                  ]
                }"#]],
        );
    }

    #[test]
    fn emits_full_join_over_a_joined_row() {
        check(
            &format!("{TABLE}{JOIN_TABLES}from t |> full join v on b == c |> select a, d"),
            expect![[r#"
                {
                  "version": {
                    "minorNumber": 85,
                    "producer": "yuzu"
                  },
                  "extensionUrns": [
                    {
                      "extensionUrnAnchor": 1,
                      "urn": "extension:io.substrait:functions_comparison"
                    }
                  ],
                  "extensions": [
                    {
                      "extensionFunction": {
                        "extensionUrnReference": 1,
                        "functionAnchor": 1,
                        "name": "equal:i32_i32"
                      }
                    }
                  ],
                  "relations": [
                    {
                      "root": {
                        "input": {
                          "project": {
                            "common": {
                              "emit": {
                                "outputMapping": [
                                  4,
                                  5
                                ]
                              }
                            },
                            "input": {
                              "join": {
                                "left": {
                                  "read": {
                                    "baseSchema": {
                                      "names": [
                                        "a",
                                        "b"
                                      ],
                                      "struct": {
                                        "types": [
                                          {
                                            "i32": {
                                              "nullability": "NULLABILITY_NULLABLE"
                                            }
                                          },
                                          {
                                            "i32": {
                                              "nullability": "NULLABILITY_NULLABLE"
                                            }
                                          }
                                        ],
                                        "nullability": "NULLABILITY_NULLABLE"
                                      }
                                    },
                                    "namedTable": {
                                      "names": [
                                        "t"
                                      ]
                                    }
                                  }
                                },
                                "right": {
                                  "read": {
                                    "baseSchema": {
                                      "names": [
                                        "c",
                                        "d"
                                      ],
                                      "struct": {
                                        "types": [
                                          {
                                            "i32": {
                                              "nullability": "NULLABILITY_NULLABLE"
                                            }
                                          },
                                          {
                                            "i32": {
                                              "nullability": "NULLABILITY_NULLABLE"
                                            }
                                          }
                                        ],
                                        "nullability": "NULLABILITY_NULLABLE"
                                      }
                                    },
                                    "namedTable": {
                                      "names": [
                                        "v"
                                      ]
                                    }
                                  }
                                },
                                "expression": {
                                  "scalarFunction": {
                                    "functionReference": 1,
                                    "arguments": [
                                      {
                                        "value": {
                                          "selection": {
                                            "directReference": {
                                              "structField": {
                                                "field": 1
                                              }
                                            },
                                            "rootReference": {}
                                          }
                                        }
                                      },
                                      {
                                        "value": {
                                          "selection": {
                                            "directReference": {
                                              "structField": {
                                                "field": 2
                                              }
                                            },
                                            "rootReference": {}
                                          }
                                        }
                                      }
                                    ],
                                    "outputType": {
                                      "bool": {
                                        "nullability": "NULLABILITY_NULLABLE"
                                      }
                                    }
                                  }
                                },
                                "type": "JOIN_TYPE_OUTER"
                              }
                            },
                            "expressions": [
                              {
                                "selection": {
                                  "directReference": {
                                    "structField": {}
                                  },
                                  "rootReference": {}
                                }
                              },
                              {
                                "selection": {
                                  "directReference": {
                                    "structField": {
                                      "field": 3
                                    }
                                  },
                                  "rootReference": {}
                                }
                              }
                            ]
                          }
                        },
                        "names": [
                          "a",
                          "d"
                        ]
                      }
                    }
                  ]
                }"#]],
        );
    }

    #[test]
    fn emits_pipeline_stages() {
        check(
            &format!("{TABLE}from t |> extend a * 2 as d |> drop b |> rename a as x |> distinct"),
            expect![[r#"
                {
                  "version": {
                    "minorNumber": 85,
                    "producer": "yuzu"
                  },
                  "extensionUrns": [
                    {
                      "extensionUrnAnchor": 1,
                      "urn": "extension:io.substrait:functions_arithmetic"
                    }
                  ],
                  "extensions": [
                    {
                      "extensionFunction": {
                        "extensionUrnReference": 1,
                        "functionAnchor": 1,
                        "name": "multiply:i32_i32"
                      }
                    }
                  ],
                  "relations": [
                    {
                      "root": {
                        "input": {
                          "aggregate": {
                            "input": {
                              "project": {
                                "common": {
                                  "emit": {
                                    "outputMapping": [
                                      0,
                                      2
                                    ]
                                  }
                                },
                                "input": {
                                  "project": {
                                    "common": {
                                      "emit": {
                                        "outputMapping": [
                                          0,
                                          1,
                                          2
                                        ]
                                      }
                                    },
                                    "input": {
                                      "read": {
                                        "baseSchema": {
                                          "names": [
                                            "a",
                                            "b"
                                          ],
                                          "struct": {
                                            "types": [
                                              {
                                                "i32": {
                                                  "nullability": "NULLABILITY_NULLABLE"
                                                }
                                              },
                                              {
                                                "i32": {
                                                  "nullability": "NULLABILITY_NULLABLE"
                                                }
                                              }
                                            ],
                                            "nullability": "NULLABILITY_NULLABLE"
                                          }
                                        },
                                        "namedTable": {
                                          "names": [
                                            "t"
                                          ]
                                        }
                                      }
                                    },
                                    "expressions": [
                                      {
                                        "scalarFunction": {
                                          "functionReference": 1,
                                          "arguments": [
                                            {
                                              "value": {
                                                "selection": {
                                                  "directReference": {
                                                    "structField": {}
                                                  },
                                                  "rootReference": {}
                                                }
                                              }
                                            },
                                            {
                                              "value": {
                                                "literal": {
                                                  "i32": 2
                                                }
                                              }
                                            }
                                          ],
                                          "outputType": {
                                            "i32": {
                                              "nullability": "NULLABILITY_NULLABLE"
                                            }
                                          }
                                        }
                                      }
                                    ]
                                  }
                                }
                              }
                            },
                            "groupings": [
                              {
                                "expressionReferences": [
                                  0,
                                  1
                                ]
                              }
                            ],
                            "groupingExpressions": [
                              {
                                "selection": {
                                  "directReference": {
                                    "structField": {}
                                  },
                                  "rootReference": {}
                                }
                              },
                              {
                                "selection": {
                                  "directReference": {
                                    "structField": {
                                      "field": 1
                                    }
                                  },
                                  "rootReference": {}
                                }
                              }
                            ]
                          }
                        },
                        "names": [
                          "x",
                          "d"
                        ]
                      }
                    }
                  ]
                }"#]],
        );
    }
}
