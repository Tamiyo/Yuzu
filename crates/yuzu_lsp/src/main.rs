//! `yuzu-lsp`: the language server over stdio.

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
            eprintln!("yuzu-lsp: {error}");
            ExitCode::FAILURE
        }
        (Ok(()), Err(error)) => {
            eprintln!("yuzu-lsp: {error}");
            ExitCode::FAILURE
        }
    }
}
