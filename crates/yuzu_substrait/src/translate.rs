//! yzr → Substrait. MLIR calls a rewrite between dialects a conversion and
//! a rewrite out of MLIR a translation; this is the second kind, and the end
//! of the pipeline — prost structs out.
//!
//! The relational graph is the module's use-def chain: `yzr.output` names
//! the query, each stage names its input, and a value that two stages read
//! is one relation read twice. Substrait nests instead of sharing, so a
//! relation read twice is translated once and written out at both uses.

use melior::Context;
use melior::ir::operation::{OperationLike, OperationRef, OperationResult};
use melior::ir::{Location, Module, Type, Value, ValueLike};
use rustc_hash::FxHashMap;
use substrait::proto::{Plan, PlanRel, Rel, RelRoot, plan_rel};
use substrait::version;
use yuzu_mlir::StructType;
use yuzu_mlir::SymbolTable;
use yuzu_mlir::diagnostics::emit_error;
use yuzu_mlir::ir::attribute::array::ArrayAttributeExt;
use yuzu_mlir::ir::block::BlockExt;
use yuzu_mlir::ir::operation::{OperationCast, OperationExt};
use yuzu_mlir::ir::value::ValueId;
use yuzu_mlir::ops::yz::YzOp;
use yuzu_mlir::ops::yzr::YzrOp;

use crate::extensions::Extensions;

mod expr;
mod functions;
mod rel;
mod types;

/// Translates the module's query. `None` once it has reported why it could
/// not, so run this inside `yuzu_mlir::diagnostics::capture`.
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

    let query = output.try_first_operand()?;
    let mut translator = Translator {
        context,
        symbols: &symbols,
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

    let names = columns.iter().map(|name| name.to_string()).collect();

    Some(Plan {
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
    })
}

struct Translator<'c, 'a, 's> {
    context: &'c Context,
    symbols: &'s SymbolTable<'c, 'a>,
    /// What each relational value already translated to, so a relation two
    /// stages read is walked once.
    translated: FxHashMap<ValueId, Rel>,
    extensions: Extensions,
}

impl<'c, 'a> Translator<'c, 'a, '_> {
    /// The operation a value came out of. A block argument came out of no
    /// operation, which is how a column of the row is told from a computed
    /// value.
    fn producer<'v>(value: Value<'c, 'v>) -> Option<OperationRef<'c, 'v>> {
        Some(OperationResult::try_from(value).ok()?.owner())
    }

    /// The names and types of a row, from the `yz.struct` that declares it.
    fn row(&self, ty: Type<'c>) -> Option<(Vec<&'c str>, Vec<Type<'c>>)> {
        let declaration = self.symbols.lookup(StructType::from_type(ty)?.name())?;
        let YzOp::Struct(item) = declaration.as_yz()? else {
            return None;
        };

        Some((
            item.names().strings().collect(),
            item.types().types().collect(),
        ))
    }

    fn width(&self, value: Value<'c, 'a>) -> Option<usize> {
        Some(self.row(value.r#type())?.0.len())
    }

    /// Substrait has no way to say this. Reported against the operation, so
    /// the location it carries is the source the reader wrote.
    fn report(&self, op: OperationRef<'c, '_>, message: &str) {
        emit_error(op.location(), message);
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use expect_test::Expect;
    use yuzu_ast::AstNode;
    use yuzu_ast::ast;
    use yuzu_diagnostics::diagnostics::engine::DiagnosticsEngine;
    use yuzu_diagnostics::diagnostics::printer::DiagnosticPrinter;
    use yuzu_diagnostics::source_map::SourceMap;
    use yuzu_lexer::lexer::{Lexer, Token};

    use crate::to_json;

    pub(crate) const TABLE: &str = "struct Row { a: int64, b: int64 }\ntable t = Row\n";

    /// Runs the whole MLIR pipeline over the source and translates what
    /// comes out, so a test reads the plan the compiler would hand an
    /// engine — not one assembled by hand.
    fn compile(source: &str) -> (Option<String>, String) {
        let context = yuzu_mlir::context();
        let mut sources = SourceMap::new();
        let mut diagnostics = DiagnosticsEngine::new();
        let prelude = include_str!("../../yuzu_passes/tests/prelude.yz");
        let prelude_id = sources.add("<prelude>".to_string(), prelude.to_string());
        let tokens: Vec<Token> = Lexer::new(prelude).collect();
        let prelude_root =
            ast::Root::cast(yuzu_parser::parse(&tokens, &mut diagnostics, prelude_id))
                .expect("a source has a root");
        let mut prelude = yuzu_passes::File::new(
            prelude_id,
            Some(yuzu_passes::PRELUDE.to_string()),
            prelude_root,
        );
        prelude.set_lowering(yuzu_passes::Lowering::OnDemand);
        let source_id = sources.add("test.yz".to_string(), source.to_string());
        let tokens: Vec<Token> = Lexer::new(source).collect();
        let syntax = yuzu_parser::parse(&tokens, &mut diagnostics, source_id);
        let root = ast::Root::cast(syntax).expect("a source has a root");
        let mut module = yuzu_passes::lower_ast_to_yzl(
            &context,
            &sources,
            &[prelude, yuzu_passes::File::entry(source_id, root)],
            &mut diagnostics,
            &yuzu_types::Builtins,
            None,
        );

        yuzu_mlir::diagnostics::capture(&context, &sources, &mut diagnostics, || {
            yuzu_passes::check_mutability(&module);
            yuzu_passes::promote_locals(&context, &mut module);
            yuzu_passes::infer_types(&context, &mut module, &yuzu_types::Builtins);
            yuzu_passes::check_aggregates(&module);
            yuzu_passes::inline_calls(&context, &mut module);
            yuzu_passes::lower_yzl_to_yzr(&context, &mut module);
            yuzu_passes::simplify_yzr(&context, &mut module);
            yuzu_passes::legalize_operators(&context, &mut module);
        });
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
            .map(|diagnostic| printer.print(diagnostic))
            .collect();

        (plan.map(|plan| to_json(&plan)), reported.join("\n"))
    }

    pub(crate) fn check(source: &str, expected: Expect) {
        let (plan, reported) = compile(source);
        assert!(reported.is_empty(), "translation reported:\n{reported}");
        expected.assert_eq(&plan.expect("the program has a query"));
    }

    pub(crate) fn check_error(source: &str, expected: Expect) {
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
    fn translates_the_canonical_pipeline() {
        check(
            &format!(
                "{TABLE}from t\n|> where a > 10\n|> extend a + b as e\n|> aggregate sum(e) as s group by b\n|> limit 5 offset 2"
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
                      "urn": "extension:io.substrait:functions_comparison"
                    },
                    {
                      "extensionUrnAnchor": 2,
                      "urn": "extension:io.substrait:functions_arithmetic"
                    },
                    {
                      "extensionUrnAnchor": 3,
                      "urn": "extension:io.yuzu:external"
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
                        "extensionUrnReference": 3,
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
            expect![[r#"
                error: the program has no query
            "#]],
        );
    }

    #[test]
    fn reports_an_operator_the_target_does_not_have() {
        check_error(
            &format!("{TABLE}from t |> select a ** 2 as p"),
            expect![[r#"
                error: `**` has no implementation for this engine
                 --> test.yz:3:18
                  |
                3 | from t |> select a ** 2 as p
                  |                  ^^^^^^
            "#]],
        );
    }
}
