//! Times the work the language server does on each path: a keystroke, a
//! check, and a request on a finished check. Run it in release mode:
//! `cargo run --release -p yuzu_ide --example ide_bench`.

use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use text_size::{TextRange, TextSize};
use yuzu_ide::{AnalysisHost, Change, Checked, FileId, FilePosition};

const FILE: FileId = FileId(0);

/// A program of `functions` functions, each with a local, a call and a
/// query: about seven lines each.
fn program(functions: usize) -> String {
    let mut text = String::from("table t = { a: int64, b: str }\nlet cap = 10\n");
    for i in 0..functions {
        write!(
            text,
            "def f{i}(x: int64) -> int64 {{\n    let y = x * {i} + cap\n    return y\n}}\n\
             from t |> where a > {i} |> select f{i}(a) as v{i}\n\n"
        )
        .expect("writing to a String cannot fail");
    }
    text
}

fn time(runs: usize, mut work: impl FnMut()) -> (Duration, Duration) {
    let mut samples: Vec<Duration> = (0..runs)
        .map(|_| {
            let start = Instant::now();
            work();
            start.elapsed()
        })
        .collect();
    samples.sort();
    (samples[runs / 2], samples[runs * 95 / 100])
}

fn report(what: &str, (median, p95): (Duration, Duration)) {
    println!("  {what:<34} median {median:>10.1?}   p95 {p95:>10.1?}");
}

fn new_host(path: &Path, text: &str) -> AnalysisHost {
    let mut host = AnalysisHost::default();
    let mut change = Change::default();
    change.set_file(FILE, Some(text.into()));
    change.set_path(FILE, Some(path.to_path_buf()));
    host.apply_change(change);
    host
}

fn offset_of(text: &str, needle: &str) -> TextSize {
    let at = text.find(needle).expect("the program holds the needle");
    TextSize::try_from(at).expect("a program is shorter than 4 GiB")
}

fn main() {
    let dir: PathBuf = std::env::temp_dir().join(format!("yuzu-bench-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("the temp dir is writable");
    let path = dir.join("main.yz");
    let library = yuzu_ide::install_library(&dir.join("cache")).expect("the library installs");

    for functions in [10, 100, 500] {
        let text = program(functions);
        let lines = text.lines().count();
        println!("{functions} functions, {lines} lines, {} bytes", text.len());
        let mut host = new_host(&path, &text);

        let analysis = host.analysis();
        let start = Instant::now();
        let checked = analysis.check(FILE).expect("the file has a path");
        let first = start.elapsed();
        assert!(
            checked.diagnostics().is_empty(),
            "{:?}",
            checked.diagnostics()
        );
        println!("  {:<34} {first:>17.1?}", "first check (cold)");

        println!(" keystroke (main thread)");
        let edited: Arc<str> = text.replacen("let cap = 10", "let cap = 11", 1).into();
        report(
            "edit: reparse the file",
            time(50, || {
                let mut change = Change::default();
                change.set_file(FILE, Some(Arc::clone(&edited)));
                host.apply_change(change);
            }),
        );
        let analysis = host.analysis();
        report(
            "semantic tokens (syntax)",
            time(50, || drop(analysis.highlight(FILE))),
        );
        report(
            "folding ranges",
            time(50, || drop(analysis.folding_ranges(FILE))),
        );
        report("outline", time(50, || drop(analysis.file_structure(FILE))));
        report(
            "selection ranges",
            time(50, || {
                drop(analysis.selection_ranges(FilePosition {
                    file_id: FILE,
                    offset: offset_of(&text, "return y"),
                }));
            }),
        );

        println!(" check (checker thread)");
        report("check, warm", time(20, || drop(analysis.check(FILE))));
        let mut installed = new_host(&path, &edited);
        installed.set_library_root(Some(&library));
        let with_library = installed.analysis();
        report(
            "check, warm (library installed)",
            time(20, || drop(with_library.check(FILE))),
        );

        println!(" requests (on a finished check)");
        let checked: Checked = analysis.check(FILE).expect("the file has a path");
        let at = |offset| FilePosition {
            file_id: FILE,
            offset,
        };
        let use_of_y = at(offset_of(&text, "return y") + TextSize::from(7));
        let call = at(offset_of(&text, "select f0") + TextSize::from(7));
        let whole = TextRange::up_to(TextSize::of(text.as_str()));
        report("hover", time(200, || drop(checked.hover(use_of_y))));
        report(
            "go to definition",
            time(200, || drop(checked.goto_definition(call))),
        );
        report("references", time(200, || drop(checked.references(call))));
        report(
            "highlight uses (whole file)",
            time(50, || drop(checked.highlight_uses(FILE))),
        );
        report(
            "inlay hints (whole file)",
            time(50, || drop(checked.inlay_hints(FILE, whole))),
        );
        println!();
    }

    let _ = std::fs::remove_dir_all(&dir);
}
