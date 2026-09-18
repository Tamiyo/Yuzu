use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use yuzu_driver::{CompileOptions, compile};

#[derive(Parser)]
#[command(name = "yuzu", version, about = "Compile a Yuzu source file")]
struct Cli {
    #[arg(help = "The source file to compile")]
    file: PathBuf,

    #[arg(long, help = "Dump the token stream")]
    debug_tokens: bool,

    #[arg(long, help = "Dump the syntax tree")]
    debug_ast: bool,

    #[arg(long, help = "Dump the HIR")]
    debug_hir: bool,

    #[arg(long, help = "Dump the ANF")]
    debug_anf: bool,

    #[arg(long, help = "Time each compile phase")]
    time: bool,

    #[arg(long, help = "Target to validate against, e.g. postgres@16")]
    target: Option<String>,

    #[arg(long, help = "Dump the plan graph")]
    debug_plan: bool,

    #[arg(long, help = "Dump the reduced ANF")]
    debug_reduce: bool,

    #[arg(long, help = "Dump the Substrait plan")]
    debug_substrait: bool,

    #[arg(long, help = "Dump all of the above")]
    debug: bool,

    #[arg(long, help = "Compile through the MLIR pipeline instead")]
    pipeline: Option<String>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    let options = CompileOptions {
        debug_tokens: cli.debug_tokens || cli.debug,
        debug_ast: cli.debug_ast || cli.debug,
        debug_hir: cli.debug_hir || cli.debug,
        debug_anf: cli.debug_anf || cli.debug,
        debug_reduce: cli.debug_reduce || cli.debug,
        debug_plan: cli.debug_plan || cli.debug,
        debug_substrait: cli.debug_substrait || cli.debug,
        time_phases: cli.time,
        target: cli.target,
    };

    let source = match std::fs::read_to_string(&cli.file) {
        Ok(source) => source,
        Err(err) => {
            eprintln!("yuzu: cannot read '{}': {err}", cli.file.display());
            return ExitCode::FAILURE;
        }
    };

    let name = cli.file.display().to_string();
    if cli.pipeline.as_deref() == Some("mlir") {
        // A module is resolved beside the file that imported it.
        let resolver = yuzu_driver::modules::FsResolver {
            base: yuzu_driver::modules::base_of(&cli.file),
        };
        return yuzu_driver::compile_mlir(&name, &source, &options, &resolver);
    }

    compile(&name, &source, &options);
    ExitCode::SUCCESS
}
