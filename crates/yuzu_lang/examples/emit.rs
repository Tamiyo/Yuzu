//! Emit a Yuzu source file as yzl MLIR and print it.
//!
//!   cargo run -p yuzu_lang --example emit -- query.yz

use std::io::Read;

use melior::ir::operation::OperationLike;

fn main() {
    let source = match std::env::args().nth(1) {
        Some(path) => std::fs::read_to_string(&path).expect("the input file reads"),
        None => {
            let mut buffer = String::new();
            std::io::stdin()
                .read_to_string(&mut buffer)
                .expect("stdin reads");
            buffer
        }
    };

    let context = yuzu_mlir::context();
    let Some(module) = yuzu_lang::emit_source(&context, "input.yz", &source) else {
        std::process::exit(1);
    };
    if !module.as_operation().verify() {
        eprintln!("yuzu_lang: the emitted module does not verify");
    }
    print!("{}", module.as_operation());
}
