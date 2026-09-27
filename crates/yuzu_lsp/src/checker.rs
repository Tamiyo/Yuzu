//! The compiler's checks, on a thread of their own. The MLIR context a check
//! runs in lives on that thread, and a check takes longer than a keystroke.

use std::thread;
use std::time::Duration;

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, unbounded};
use yuzu_ide::{Analysis, Checked, FileId};

/// How long the checker waits for a newer request before it checks. While
/// someone types, each request replaces the one before it, so only the text
/// they stop at is checked.
const DEBOUNCE: Duration = Duration::from_millis(200);

/// Open documents to check, as a snapshot saw them. `generation` counts the
/// changes the server has seen, so an answer to an older one can be told
/// apart and dropped.
pub(crate) struct CheckRequest {
    pub(crate) generation: u64,
    pub(crate) analysis: Analysis,
    pub(crate) files: Vec<(FileId, i32)>,
}

/// Each document's check, with the version of the text it read.
pub(crate) struct CheckResult {
    pub(crate) generation: u64,
    pub(crate) checks: Vec<(FileId, i32, Checked)>,
}

pub(crate) struct Checker {
    requests: Sender<CheckRequest>,
    results: Receiver<CheckResult>,
}

impl Checker {
    pub(crate) fn spawn() -> Self {
        let (requests, incoming) = unbounded();
        let (outgoing, results) = unbounded();
        thread::Builder::new()
            .name("yuzu-checker".to_owned())
            .spawn(move || serve(&incoming, &outgoing))
            .expect("the checker thread starts");
        Checker { requests, results }
    }

    /// `false` once the checker thread has stopped.
    pub(crate) fn request(&self, request: CheckRequest) -> bool {
        self.requests.send(request).is_ok()
    }

    pub(crate) fn results(&self) -> &Receiver<CheckResult> {
        &self.results
    }
}

fn serve(incoming: &Receiver<CheckRequest>, outgoing: &Sender<CheckResult>) {
    while let Ok(mut request) = incoming.recv() {
        loop {
            match incoming.recv_timeout(DEBOUNCE) {
                Ok(newer) => request = newer,
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => return,
            }
        }

        let checks = request
            .files
            .iter()
            .filter_map(|&(file_id, version)| {
                Some((file_id, version, request.analysis.check(file_id)?))
            })
            .collect();
        let result = CheckResult {
            generation: request.generation,
            checks,
        };
        if outgoing.send(result).is_err() {
            return;
        }
    }
}
