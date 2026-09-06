//! Convert a Yuzu source file to yzl MLIR and print it.
//!
//!   cargo run -p yuzu_lang --example convert -- query.yz

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
    let Some(conversion) = yuzu_lang::convert_source(&context, "input.yz", &source) else {
        std::process::exit(1);
    };
    for what in &conversion.unsupported {
        eprintln!("yuzu_lang: {what}");
    }
    if !conversion.module.as_operation().verify() {
        eprintln!("yuzu_lang: the converted module does not verify");
    }
    print!("{}", conversion.module.as_operation());
}
