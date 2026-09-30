//! The compiler's checks, on a thread of their own. The MLIR context a check
//! runs in lives on that thread, and a check takes longer than a keystroke.

use std::panic::{self, AssertUnwindSafe};
use std::thread;
use std::time::Duration;

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, unbounded};
use yuzu_ide::{Analysis, Checked, FileId};

use crate::RunError;

/// How long the checker waits for a newer request before it checks. While
/// someone types, each request replaces the one before it, so only the text
/// they stop at is checked.
const DEBOUNCE: Duration = Duration::from_millis(50);

/// Open documents to check, as a snapshot saw them, each with the version
/// of its text. The first is checked first. A newer request replaces an
/// older one, so each request holds every document not yet checked since
/// what it read last changed.
pub(crate) struct CheckRequest {
    pub(crate) analysis: Analysis,
    pub(crate) files: Vec<(FileId, i32)>,
    /// The request's number; a check it makes covers each change made
    /// before it.
    pub(crate) generation: u64,
}

/// One document's check, with the version of the text it read.
pub(crate) enum CheckResult {
    Checked {
        file_id: FileId,
        version: i32,
        generation: u64,
        checked: Box<Checked>,
    },
    /// The check panicked. The thread stops after it sends this, since the
    /// compiler's state on the thread cannot be trusted.
    Panicked {
        file_id: FileId,
        version: i32,
        message: String,
    },
}

/// The checker thread stopped.
#[derive(Debug)]
pub(crate) struct Stopped;

pub(crate) struct Checker {
    requests: Sender<CheckRequest>,
    results: Receiver<CheckResult>,
}

impl Checker {
    pub(crate) fn spawn() -> Result<Self, RunError> {
        let (requests, incoming) = unbounded();
        let (outgoing, results) = unbounded();
        thread::Builder::new()
            .name("yuzu-checker".to_owned())
            .spawn(move || serve(&incoming, &outgoing))
            .map_err(RunError::checker)?;
        Ok(Checker { requests, results })
    }

    pub(crate) fn request(&self, request: CheckRequest) -> Result<(), Stopped> {
        self.requests.send(request).map_err(|_unsent| Stopped)
    }

    pub(crate) fn results(&self) -> &Receiver<CheckResult> {
        &self.results
    }
}

fn serve(incoming: &Receiver<CheckRequest>, outgoing: &Sender<CheckResult>) {
    let mut next = None;
    loop {
        let mut request = match next.take() {
            Some(request) => request,
            None => match incoming.recv() {
                Ok(request) => request,
                Err(_) => return,
            },
        };
        loop {
            match incoming.recv_timeout(DEBOUNCE) {
                Ok(newer) => request = newer,
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => return,
            }
        }

        for &(file_id, version) in &request.files {
            if let Ok(newer) = incoming.try_recv() {
                next = Some(newer);
                break;
            }

            let outcome = panic::catch_unwind(AssertUnwindSafe(|| request.analysis.check(file_id)));
            let result = match outcome {
                Ok(Some(checked)) => CheckResult::Checked {
                    file_id,
                    version,
                    generation: request.generation,
                    checked: Box::new(checked),
                },
                Ok(None) => continue,
                Err(payload) => {
                    let _ = outgoing.send(CheckResult::Panicked {
                        file_id,
                        version,
                        message: panic_message(payload.as_ref()),
                    });
                    return;
                }
            };
            if outgoing.send(result).is_err() {
                return;
            }
        }
    }
}

/// The message a panic was raised with.
pub(crate) fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "a panic with no message".to_owned()
    }
}
