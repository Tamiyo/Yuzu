use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use yuzu_driver::modules::{FsResolver, base_of};
use yuzu_driver::{CompileOptions, compile};

#[derive(Parser)]
#[command(name = "yuzu", version, about = "Compile a Yuzu source file")]
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

    #[arg(long, help = "The engine to compile for, e.g. datafusion")]
    target: Option<String>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    let options = CompileOptions {
        debug_yzl: cli.debug_yzl || cli.debug,
        debug_yzr: cli.debug_yzr || cli.debug,
        debug_substrait: cli.debug_substrait || cli.debug,
        target: cli.target,
    };

    let source = match std::fs::read_to_string(&cli.file) {
        Ok(source) => source,
        Err(err) => {
            eprintln!("yuzu: cannot read '{}': {err}", cli.file.display());
            return ExitCode::FAILURE;
        }
    };

    // A module is resolved beside the file that imported it.
    let resolver = FsResolver {
        base: base_of(&cli.file),
    };
    compile(
        &cli.file.display().to_string(),
        &source,
        &options,
        &resolver,
    )
}
