use substrait::proto::{
    Expression, FunctionArgument,
    expression::{RexType, ScalarFunction, SingularOrList, literal::LiteralType},
    function_argument::ArgType,
};
use yuzu_plan::{Const, Expr, ExprId, Func};

use crate::emitter::types::{emit_type, type_code};
use crate::emitter::{GraphEmitter, Unsupported};
use crate::extensions::function_target;
use crate::proto::{literal, selection};

impl GraphEmitter<'_> {
    pub(crate) fn emit_expr(&mut self, id: ExprId) -> Result<Expression, Unsupported> {
        let expression = match self.graph.plan().expr(id).clone() {
            Expr::Column { column, .. } => selection(column as i32),
            Expr::Literal { value, .. } => literal(self.literal_value(value)),
            Expr::Call { func, args, ty } => match func {
                Func::In => {
                    let [value, options @ ..] = &args[..] else {
                        unreachable!("a membership test has a value")
                    };
                    let value = self.emit_expr(*value)?;
                    let options = options
                        .iter()
                        .map(|&option| self.emit_expr(option))
                        .collect::<Result<_, _>>()?;
                    Expression {
                        rex_type: Some(RexType::SingularOrList(Box::new(SingularOrList {
                            value: Some(Box::new(value)),
                            options,
                        }))),
                    }
                }
                _ => return self.emit_scalar_function(func, &args, ty),
            },
        };
        Ok(expression)
    }

    fn emit_scalar_function(
        &mut self,
        func: Func,
        args: &[ExprId],
        ty: yuzu_types::TypeId,
    ) -> Result<Expression, Unsupported> {
        let (urn, base) = if let Func::External(symbol) = func {
            (
                crate::extensions::EXTERNAL_URN,
                self.interner.text(symbol).to_string(),
            )
        } else {
            let Some((urn, base)) = function_target(func) else {
                let message = format!("`{}` has no Substrait mapping yet", func.symbol());
                return Err(self.unsupported_query(message));
            };
            (urn, base.to_string())
        };

        let signature: Vec<&str> = args
            .iter()
            .map(|&arg| type_code(self.types, self.graph.plan().expr(arg).ty()))
            .collect();
        let name = format!("{base}:{}", signature.join("_"));

        let mut arguments = Vec::new();
        for &arg in args {
            arguments.push(FunctionArgument {
                arg_type: Some(ArgType::Value(self.emit_expr(arg)?)),
            });
        }

        let anchor = self.extensions.register(urn, name);
        Ok(Expression {
            rex_type: Some(RexType::ScalarFunction(ScalarFunction {
                function_reference: anchor,
                output_type: Some(emit_type(self.types, ty)),
                arguments,
                ..Default::default()
            })),
        })
    }

    fn literal_value(&self, constant: Const) -> LiteralType {
        match constant {
            Const::Int { value } => match value.num_bits() {
                8 => LiteralType::I8(value.as_i64() as i32),
                16 => LiteralType::I16(value.as_i64() as i32),
                32 => LiteralType::I32(value.as_i64() as i32),
                _ => LiteralType::I64(value.as_i64()),
            },
            Const::Float { value } => match value.num_bits() {
                32 => LiteralType::Fp32(value.as_f64() as f32),
                _ => LiteralType::Fp64(value.as_f64()),
            },
            Const::Bool { value } => LiteralType::Boolean(value),
            Const::String { value } => LiteralType::String(self.interner.text(value).to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::emitter::test_support::{TABLE, check, check_error};

    #[test]
    fn reports_a_call_that_could_not_be_reduced() {
        check_error(
            &format!(
                "{TABLE}def fact(n: int64) -> int64 {{ return n * fact(n - 1) }}\nfrom t |> select fact(3) as v"
            ),
            expect!["call to `fact` could not be fully reduced"],
        );
    }

    #[test]
    fn emits_literal_types() {
        check(
            &format!(
                "{TABLE}let big = 5000000000\nlet f = 1.5\nlet flag = true\nlet name = \"jon\"\nfrom t |> select big, f, flag, name"
            ),
            expect![[r#"
                {
                  "version": {
                    "minorNumber": 85,
                    "producer": "yuzu"
                  },
                  "relations": [
                    {
                      "root": {
                        "input": {
                          "project": {
                            "common": {
                              "emit": {
                                "outputMapping": [
                                  2,
                                  3,
                                  4,
                                  5
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
                                "literal": {
                                  "i64": "5000000000"
                                }
                              },
                              {
                                "literal": {
                                  "fp64": 1.5
                                }
                              },
                              {
                                "literal": {
                                  "boolean": true
                                }
                              },
                              {
                                "literal": {
                                  "string": "jon"
                                }
                              }
                            ]
                          }
                        },
                        "names": [
                          "big",
                          "f",
                          "flag",
                          "name"
                        ]
                      }
                    }
                  ]
                }"#]],
        );
    }

    #[test]
    fn a_list_column_is_reported() {
        check_error(
            &format!("{TABLE}let xs: List[int32] = [1, 2]\nfrom t |> select xs as l"),
            expect!["a list can only be tested for membership"],
        );
    }

    #[test]
    fn reports_an_unsupported_operator() {
        check_error(
            &format!("{TABLE}from t |> select a ** 2 as p"),
            expect!["`**` has no Substrait mapping yet"],
        );
    }

    #[test]
    fn reports_a_struct_column() {
        check_error(
            &format!(
                "{TABLE}struct P {{ v: int64 }}\nlet p = P {{ v: 1 }}\nfrom t |> select p as q"
            ),
            expect!["a struct value cannot be a query column"],
        );
    }

    #[test]
    fn reports_an_unreduced_column_dependency() {
        check_error(
            &format!(
                "{TABLE}def fact(n: int64) -> int64 {{ return n * fact(n - 1) }}\nlet v = fact(3)\nfrom t |> select v as w"
            ),
            expect!["query column depends on a value that could not be fully reduced"],
        );
    }

    #[test]
    fn emits_filter_with_literal() {
        check(
            &format!("{TABLE}from t |> where a > 1"),
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
                        "name": "gt:i32_i32"
                      }
                    }
                  ],
                  "relations": [
                    {
                      "root": {
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
                                        "i32": 1
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
                        "names": [
                          "a",
                          "b"
                        ]
                      }
                    }
                  ]
                }"#]],
        );
    }

    #[test]
    fn emits_membership_as_singular_or_list() {
        check(
            &format!("{TABLE}let xs: List[int32] = [1, 2]\nfrom t |> where a in xs"),
            expect![[r#"
                {
                  "version": {
                    "minorNumber": 85,
                    "producer": "yuzu"
                  },
                  "relations": [
                    {
                      "root": {
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
                            "condition": {
                              "singularOrList": {
                                "value": {
                                  "selection": {
                                    "directReference": {
                                      "structField": {}
                                    },
                                    "rootReference": {}
                                  }
                                },
                                "options": [
                                  {
                                    "literal": {
                                      "i32": 1
                                    }
                                  },
                                  {
                                    "literal": {
                                      "i32": 2
                                    }
                                  }
                                ]
                              }
                            }
                          }
                        },
                        "names": [
                          "a",
                          "b"
                        ]
                      }
                    }
                  ]
                }"#]],
        );
    }

    #[test]
    fn emits_not_in_wrapped_in_not() {
        check(
            &format!("{TABLE}let xs: List[int32] = [1, 2]\nfrom t |> where a not in xs"),
            expect![[r#"
                {
                  "version": {
                    "minorNumber": 85,
                    "producer": "yuzu"
                  },
                  "extensionUrns": [
                    {
                      "extensionUrnAnchor": 1,
                      "urn": "extension:io.substrait:functions_boolean"
                    }
                  ],
                  "extensions": [
                    {
                      "extensionFunction": {
                        "extensionUrnReference": 1,
                        "functionAnchor": 1,
                        "name": "not:bool"
                      }
                    }
                  ],
                  "relations": [
                    {
                      "root": {
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
                            "condition": {
                              "scalarFunction": {
                                "functionReference": 1,
                                "arguments": [
                                  {
                                    "value": {
                                      "singularOrList": {
                                        "value": {
                                          "selection": {
                                            "directReference": {
                                              "structField": {}
                                            },
                                            "rootReference": {}
                                          }
                                        },
                                        "options": [
                                          {
                                            "literal": {
                                              "i32": 1
                                            }
                                          },
                                          {
                                            "literal": {
                                              "i32": 2
                                            }
                                          }
                                        ]
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
                        "names": [
                          "a",
                          "b"
                        ]
                      }
                    }
                  ]
                }"#]],
        );
    }
}
