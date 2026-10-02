//! yzr → Substrait. MLIR calls a rewrite between dialects a conversion and
//! a rewrite out of MLIR a translation; this is the second kind, and the end
//! of the pipeline — prost structs out.
//!
//! The relational graph is the module's use-def chain: `yzr.output` names
//! the query, each stage names its input, and a value that two stages read
//! is one relation read twice. Substrait nests instead of sharing, so a
//! relation read twice is translated once and written out at both uses.

use melior::Context;
use melior::ir::operation::{OperationLike, OperationRef};
use melior::ir::{Location, Module, Type, Value, ValueLike};
use rustc_hash::{FxHashMap, FxHashSet};
use substrait::proto::{PlanRel, Rel, RelRoot, plan_rel};
use substrait::version;
use yuzu_mlir::diagnostics::emit_error;
use yuzu_mlir::ir::attribute::array::ArrayAttributeExt;
use yuzu_mlir::ir::block::BlockExt;
use yuzu_mlir::ir::operation::{OperationCast, OperationExt};
use yuzu_mlir::ir::symbol_table::SymbolTable;
use yuzu_mlir::ir::value::{ValueExt, ValueId, op_result};
use yuzu_mlir::ops::yz::{StructOp, YzOp};
use yuzu_mlir::ops::yzr::YzrOp;
use yuzu_mlir::types::StructType;

use crate::Plan;
use crate::extensions::Extensions;

mod expr;
mod functions;
mod rel;
mod types;

/// Translates the module's query. `None` once it has reported why it could
/// not, so run this inside `yuzu_mlir::diagnostics::capture`.
///
/// # Panics
///
/// Panics if the module does not verify. The translation trusts each stage
/// to hold what its verifier promises.
#[must_use]
pub fn translate<'c>(context: &'c Context, module: &Module<'c>) -> Option<Plan> {
    let symbols = SymbolTable::new(module);
    let body = module.body();
    let Some(output) = body
        .operations()
        .find(|op| matches!(op.as_yzr(), Some(YzrOp::Output(_))))
    else {
        // The program as a whole lacks it, so no one place is to blame.
        emit_error(Location::unknown(context), "the program has no query");
        return None;
    };

    let query = output
        .try_first_operand()
        .expect("a verified yzr.output has its query");
    let mut translator = Translator {
        symbols: &symbols,
        shared: shared_relations(module),
        translated: FxHashMap::default(),
        extensions: Extensions::default(),
    };

    // The relation first: translating it is what registers the functions the
    // plan has to declare.
    let relation = translator.translate_rel(query)?;
    let Some((columns, _)) = translator.row(query.r#type()) else {
        emit_error(output.location(), "the query has no row shape to name");
        return None;
    };

    let names = columns
        .iter()
        .map(std::string::ToString::to_string)
        .collect();

    Some(Plan(substrait::proto::Plan {
        version: Some(version::version_with_producer("yuzu")),
        extension_urns: translator.extensions.urns(),
        extensions: translator.extensions.declarations(),
        relations: vec![PlanRel {
            rel_type: Some(plan_rel::RelType::Root(RelRoot {
                input: Some(relation),
                names,
            })),
        }],
        ..Default::default()
    }))
}

/// The relational values more than one stage reads. Substrait nests
/// relations, so the plan writes each of these out at every use. Any other
/// relation moves into the one stage that reads it.
fn shared_relations(module: &Module<'_>) -> FxHashSet<ValueId> {
    let mut reads: FxHashMap<ValueId, usize> = FxHashMap::default();
    for op in module.body().operations() {
        for operand in op.operands() {
            *reads.entry(operand.id()).or_default() += 1;
        }
    }
    reads
        .into_iter()
        .filter_map(|(value, count)| (count > 1).then_some(value))
        .collect()
}

struct Translator<'c, 'a, 's> {
    symbols: &'s SymbolTable<'c, 'a>,
    shared: FxHashSet<ValueId>,
    /// What each shared relation translated to, so it is walked once.
    translated: FxHashMap<ValueId, Rel>,
    extensions: Extensions,
}

impl<'c, 'a> Translator<'c, 'a, '_> {
    /// The operation a value came out of. A block argument came out of no
    /// operation, which is how a column of the row is told from a computed
    /// value.
    fn producer<'v>(value: Value<'c, 'v>) -> Option<OperationRef<'c, 'v>> {
        Some(op_result(value)?.owner())
    }

    /// What `read` takes from the `yz.struct` that declares a row type.
    fn read_declaration<R>(
        &self,
        ty: Type<'c>,
        read: impl FnOnce(StructOp<'c, '_>) -> R,
    ) -> Option<R> {
        let declaration = self.symbols.lookup(StructType::from_type(ty)?.name())?;
        match declaration.as_yz()? {
            YzOp::Struct(item) => Some(read(item)),
            _ => None,
        }
    }

    fn row(&self, ty: Type<'c>) -> Option<(Vec<&'c str>, Vec<Type<'c>>)> {
        self.read_declaration(ty, |item| {
            (
                item.names().strings().collect(),
                item.types().types().collect(),
            )
        })
    }

    fn width(&self, value: Value<'c, 'a>) -> Option<usize> {
        self.read_declaration(value.r#type(), |item| item.names().len())
    }
}

/// Reports that Substrait cannot express `op`. The error goes on the
/// operation, so its location points into the source.
fn report(op: OperationRef<'_, '_>, message: &str) {
    emit_error(op.location(), message);
}

#[cfg(test)]
pub(crate) mod test_support {
    use expect_test::Expect;
    use melior::Context;
    use melior::ir::Module;
    use yuzu_ast::ast;
    use yuzu_ast::ast::AstNode;
    use yuzu_diagnostics::{DiagnosticPrinter, DiagnosticsEngine, SourceMap};

    pub(crate) const TABLE: &str = "struct Row { a: int64, b: int64 }\ntable t = Row\n";

    /// Runs the whole MLIR pipeline over the source and translates what
    /// comes out, so a test reads the plan the compiler would hand an
    /// engine — not one assembled by hand.
    fn compile(source: &str) -> (Option<String>, String) {
        type Group = for<'c> fn(&'c Context, &mut Module<'c>);

        let context = yuzu_mlir::context();
        let mut sources = SourceMap::new();
        let mut diagnostics = DiagnosticsEngine::new();
        let prelude = include_str!("../../yuzu_passes/tests/prelude.yz");
        let prelude_id = sources.add("<prelude>".to_string(), prelude.to_string());
        let prelude_root = ast::Root::cast(yuzu_parser::parse_text(
            prelude,
            &mut diagnostics,
            prelude_id,
        ))
        .expect("a source has a root");
        let mut prelude = yuzu_passes::File::new(
            prelude_id,
            Some(yuzu_passes::PRELUDE.to_string()),
            prelude_root,
        );
        prelude.set_decl_lowering(yuzu_passes::DeclarationLowering::OnDemand);
        let source_id = sources.add("test.yz".to_string(), source.to_string());
        let syntax = yuzu_parser::parse_text(source, &mut diagnostics, source_id);
        let root = ast::Root::cast(syntax).expect("a source has a root");
        let mut module = yuzu_passes::lower_ast_to_yzl(
            &context,
            &sources,
            &[prelude, yuzu_passes::File::entry(source_id, root)],
            &mut diagnostics,
            None,
        );

        // As in the driver, a group runs only when the group before it
        // reported no error. A pass may take the work before it as settled.
        let groups: [Group; 3] = [
            |context, module| {
                yuzu_passes::check_mutability(module);
                yuzu_passes::infer_types(context, module);
                yuzu_passes::promote_locals(context, module);
                yuzu_passes::check_aggregates(module);
            },
            |context, module| {
                yuzu_passes::inline_calls(context, module);
                yuzu_passes::remove_dead_symbols(context, module);
            },
            |context, module| {
                yuzu_passes::lower_yzl_to_yzr(context, module);
                yuzu_passes::simplify_yzr(context, module);
                yuzu_passes::legalize_operators(context, module);
            },
        ];
        for group in groups {
            if diagnostics.has_errors() {
                break;
            }
            yuzu_mlir::diagnostics::capture(&context, &sources, &mut diagnostics, || {
                group(&context, &mut module);
            });
        }
        let plan = if diagnostics.diagnostics().is_empty() {
            yuzu_mlir::diagnostics::capture(&context, &sources, &mut diagnostics, || {
                super::translate(&context, &module)
            })
        } else {
            None
        };

        let printer = DiagnosticPrinter::new(&sources);
        let reported: Vec<String> = diagnostics
            .diagnostics()
            .iter()
            .map(|diagnostic| printer.render(diagnostic))
            .collect();

        (plan.map(|plan| plan.to_json()), reported.join("\n"))
    }

    pub(crate) fn check(source: &str, expected: &Expect) {
        let (plan, reported) = compile(source);
        assert!(reported.is_empty(), "translation reported:\n{reported}");
        expected.assert_eq(&plan.expect("the program has a query"));
    }

    pub(crate) fn check_error(source: &str, expected: &Expect) {
        let (plan, reported) = compile(source);
        assert!(plan.is_none(), "an untranslatable program has no plan");
        expected.assert_eq(&reported);
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::translate::test_support::{TABLE, check, check_error};

    /// Every stage the pipeline can produce, in one plan: the scan, the
    /// filter, the projection that appends, the grouping and the fetch.
    #[test]
    #[expect(clippy::too_many_lines, reason = "the expected plan is long")]
    fn translates_the_canonical_pipeline() {
        check(
            &format!(
                "{TABLE}from t\n|> where a > 10\n|> extend a + b as e\n|> aggregate sum(e) as s group by b\n|> limit 5 offset 2"
            ),
            &expect![[r#"
                {
                  "version": {
                    "minorNumber": 85,
                    "producer": "yuzu"
                  },
                  "extensionUrns": [
                    {
                      "extensionUrnAnchor": 1,
                      "urn": "extension:io.substrait:functions_comparison"
                    },
                    {
                      "extensionUrnAnchor": 2,
                      "urn": "extension:io.substrait:functions_arithmetic"
                    }
                  ],
                  "extensions": [
                    {
                      "extensionFunction": {
                        "extensionUrnReference": 1,
                        "functionAnchor": 1,
                        "name": "gt:i64_i64"
                      }
                    },
                    {
                      "extensionFunction": {
                        "extensionUrnReference": 2,
                        "functionAnchor": 2,
                        "name": "add:i64_i64"
                      }
                    },
                    {
                      "extensionFunction": {
                        "extensionUrnReference": 2,
                        "functionAnchor": 3,
                        "name": "sum:i64"
                      }
                    }
                  ],
                  "relations": [
                    {
                      "root": {
                        "input": {
                          "fetch": {
                            "input": {
                              "aggregate": {
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
                                      "filter": {
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
                                                    "i64": {
                                                      "nullability": "NULLABILITY_NULLABLE"
                                                    }
                                                  },
                                                  {
                                                    "i64": {
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
                                        "condition": {
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
                                                    "i64": "10"
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
                                        }
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
                                            "i64": {
                                              "nullability": "NULLABILITY_NULLABLE"
                                            }
                                          }
                                        }
                                      }
                                    ]
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
                                      "functionReference": 3,
                                      "arguments": [
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
                            "offsetExpr": {
                              "literal": {
                                "i64": "2"
                              }
                            },
                            "countExpr": {
                              "literal": {
                                "i64": "5"
                              }
                            }
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

    /// `**` has no Substrait function, and the target is what says so.
    /// A file of declarations and no query produces nothing, which is worth
    /// saying. It is about the program rather than a place in it, so it
    /// prints as its message alone.
    #[test]
    fn reports_a_program_with_no_query() {
        check_error(
            "struct Row { a: int64 }\ntable t = Row\n",
            &expect![[r"
                error: the program has no query
            "]],
        );
    }

    #[test]
    fn reports_an_operator_the_target_does_not_have() {
        check_error(
            &format!("{TABLE}from t |> select a ** 2 as p"),
            &expect![[r"
                error: `**` has no implementation for this engine
                 --> test.yz:3:18
                  |
                3 | from t |> select a ** 2 as p
                  |                  ^^^^^^
            "]],
        );
    }
}
