//! `yuzu-lsp`: the language server over stdio.

use std::backtrace::BacktraceStatus;
use std::error::Error;
use std::process::ExitCode;

use lsp_server::Connection;

fn main() -> ExitCode {
    let (connection, io_threads) = Connection::stdio();
    let result = yuzu_lsp::run(&connection);
    drop(connection);

    let joined = io_threads.join();
    match (result, joined) {
        (Ok(()), Ok(())) => ExitCode::SUCCESS,
        (Err(error), _) => {
            report(&error);
            if error.backtrace().status() == BacktraceStatus::Captured {
                eprintln!("{}", error.backtrace());
            }
            ExitCode::FAILURE
        }
        (Ok(()), Err(error)) => {
            report(&error);
            ExitCode::FAILURE
        }
    }
}

/// Prints an error and each error that caused it.
fn report(error: &dyn Error) {
    eprintln!("yuzu-lsp: {error}");
    let mut source = error.source();
    while let Some(cause) = source {
        eprintln!("  caused by: {cause}");
        source = cause.source();
    }
}
