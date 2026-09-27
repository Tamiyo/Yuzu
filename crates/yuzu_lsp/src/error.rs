//! Why [`run`](crate::run) returns before the client asks the server to exit.

use std::backtrace::{Backtrace, BacktraceStatus};
use std::error::Error;
use std::fmt;

use lsp_server::ProtocolError;

/// Why the server stopped before the client asked it to exit.
#[derive(Debug)]
pub struct RunError {
    kind: Kind,
    backtrace: Backtrace,
}

#[derive(Debug)]
enum Kind {
    Protocol(ProtocolError),
    InitializeParams(serde_json::Error),
    Disconnected,
}

impl RunError {
    /// The connection broke the protocol, as in a handshake out of order.
    pub fn is_protocol(&self) -> bool {
        matches!(self.kind, Kind::Protocol(_))
    }

    /// The client's `initialize` params did not parse.
    pub fn is_initialize_params(&self) -> bool {
        matches!(self.kind, Kind::InitializeParams(_))
    }

    /// The client went away without asking the server to shut down.
    pub fn is_disconnected(&self) -> bool {
        matches!(self.kind, Kind::Disconnected)
    }

    pub(crate) fn protocol(error: ProtocolError) -> Self {
        Self::new(Kind::Protocol(error))
    }

    pub(crate) fn initialize_params(error: serde_json::Error) -> Self {
        Self::new(Kind::InitializeParams(error))
    }

    pub(crate) fn disconnected() -> Self {
        Self::new(Kind::Disconnected)
    }

    fn new(kind: Kind) -> Self {
        Self {
            kind,
            backtrace: Backtrace::capture(),
        }
    }
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            Kind::Protocol(error) => write!(f, "the connection broke the protocol: {error}")?,
            Kind::InitializeParams(error) => {
                write!(f, "the initialize params did not parse: {error}")?;
            }
            Kind::Disconnected => f.write_str("the client disconnected")?,
        }
        if self.backtrace.status() == BacktraceStatus::Captured {
            write!(f, "\n{}", self.backtrace)?;
        }
        Ok(())
    }
}

impl Error for RunError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.kind {
            Kind::Protocol(error) => Some(error),
            Kind::InitializeParams(error) => Some(error),
            Kind::Disconnected => None,
        }
    }
}
