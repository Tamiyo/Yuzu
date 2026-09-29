use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use yuzu_diagnostics::DiagnosticPrinter;
use yuzu_driver::modules::{FsResolver, base_of};
use yuzu_driver::stdlib::Engine;
use yuzu_driver::{CompileOptions, compile};

#[derive(Parser)]
#[command(name = "yuzu", version, about = "Compile a Yuzu source file")]
#[expect(
    clippy::struct_excessive_bools,
    reason = "each bool is a command-line flag"
)]
struct Cli {
    #[arg(help = "The source file to compile")]
    file: PathBuf,

    #[arg(long, help = "Dump the module as the frontend lowered it (yzl)")]
    debug_yzl: bool,

    #[arg(long, help = "Dump the module lowered to relations (yzr)")]
    debug_yzr: bool,

    #[arg(long, help = "Dump the Substrait plan")]
    debug_substrait: bool,

    #[arg(long, help = "Dump all of the above")]
    debug: bool,

    #[arg(long, default_value_t, help = "The engine to compile for")]
    target: Engine,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    let options = CompileOptions {
        engine: cli.target,
        dump_yzl: cli.debug_yzl || cli.debug,
        dump_yzr: cli.debug_yzr || cli.debug,
    };

    let source = match std::fs::read_to_string(&cli.file) {
        Ok(source) => source,
        Err(err) => {
            eprintln!("yuzu: cannot read '{}': {err}", cli.file.display());
            return ExitCode::FAILURE;
        }
    };

    // Every module path is resolved from the entry file's directory.
    let resolver = FsResolver {
        base: base_of(&cli.file),
    };
    let compilation = compile(
        &cli.file.display().to_string(),
        &source,
        &options,
        &resolver,
    );

    if let Some(yzl) = &compilation.yzl {
        println!("=== yzl ===");
        print!("{yzl}");
    }
    if let Some(yzr) = &compilation.yzr {
        println!("=== yzr ===");
        print!("{yzr}");
    }
    let printer = DiagnosticPrinter::new(&compilation.sources);
    for diagnostic in &compilation.diagnostics {
        eprintln!("{}", printer.render(diagnostic));
    }

    match &compilation.plan {
        Some(plan) => {
            if cli.debug_substrait || cli.debug {
                println!("=== substrait ===");
                println!("{}", plan.to_json());
            }
            ExitCode::SUCCESS
        }
        None => ExitCode::FAILURE,
    }
}
