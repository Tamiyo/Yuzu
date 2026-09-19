//! Every query the end-to-end suites compile must lower cleanly: no
//! unsupported constructs, and a module that verifies.

use melior::ir::operation::OperationLike;
use yuzu_ast::AstNode;
use yuzu_ast::ast;
use yuzu_diagnostics::diagnostics::engine::DiagnosticsEngine;
use yuzu_diagnostics::source_map::SourceMap;
use yuzu_lexer::lexer::{Lexer, Token};
use yuzu_passes::{File, lower_ast_to_yzl};

const CORPUS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../python/tests");

/// The chunks of a Python test file that hold Yuzu source: the contents of
/// its triple-quoted strings, when they pipe from a relation or declare.
fn yuzu_chunks(source: &str) -> impl Iterator<Item = (&str, Option<&str>)> {
    let parts: Vec<&str> = source.split(r#"""""#).collect();
    parts
        .clone()
        .into_iter()
        .enumerate()
        .filter(|(index, chunk)| {
            index % 2 == 1
                && ((chunk.contains("|>") && chunk.contains("from")) || chunk.contains("struct "))
        })
        .map(move |(index, chunk)| (chunk, parts.get(index + 1).copied()))
}

#[test]
fn the_correctness_corpus_lowers() {
    let mut sources = vec![std::fs::read_to_string(format!("{CORPUS}/support.py")).unwrap()];
    for entry in std::fs::read_dir(format!("{CORPUS}/correctness")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|extension| extension == "py") {
            sources.push(std::fs::read_to_string(path).unwrap());
        }
    }

    // The harness compiles `schema + query`; the schema is the one
    // triple-quoted chunk of support.py that declares the tables.
    let schema = sources[0]
        .split(r#"""""#)
        .find(|chunk| chunk.contains("struct Employee"))
        .expect("support.py declares the schema")
        .to_string();

    let context = yuzu_mlir::context();
    let mut queries = 0;
    let mut failures = Vec::new();
    for source in &sources {
        for (chunk, after) in yuzu_chunks(source) {
            queries += 1;
            // The harness wraps a query it expects to fail in `error_of(...)`
            // in the code after the string, and such a query lowering
            // cleanly is the failure here — except when the rejection is the
            // target's, which is decided at emission.
            let expects_error = after.is_some_and(|after| {
                after.contains("error_of(") && !after.contains("not supported by the")
            });
            let program = if chunk.contains("struct ") {
                chunk.to_string()
            } else {
                format!("{schema}{chunk}")
            };

            let mut sources = SourceMap::new();
            let source_id = sources.add("corpus.yz".to_string(), program.clone());
            let mut diagnostics = DiagnosticsEngine::new();
            let tokens: Vec<Token> = Lexer::new(&program).collect();
            let root = ast::Root::cast(yuzu_parser::parse(&tokens, &mut diagnostics, source_id))
                .expect("a parse always yields a root");
            let module = lower_ast_to_yzl(
                &context,
                &sources,
                &[File::entry(source_id, root)],
                &mut diagnostics,
                &yuzu_types::Builtins,
            )
            .expect("an entry file lowers");
            let messages: Vec<&str> = diagnostics
                .diagnostics()
                .iter()
                .map(|diagnostic| diagnostic.message.as_str())
                .collect();

            match (expects_error, messages.is_empty()) {
                (false, true) if module.as_operation().verify() => {}
                (false, true) => {
                    failures.push(format!("{program}\n  -> the module does not verify"))
                }
                (false, false) => failures.push(format!("{program}\n  -> {messages:?}")),
                (true, false) => {}
                (true, true) => failures.push(format!(
                    "{program}\n  -> lowered, but the harness expects an error"
                )),
            }
        }
    }

    assert!(
        queries > 30,
        "the corpus extraction found only {queries} queries"
    );
    assert!(
        failures.is_empty(),
        "{} of {queries} corpus queries failed to lower:\n{}",
        failures.len(),
        failures.join("\n---\n")
    );
}
