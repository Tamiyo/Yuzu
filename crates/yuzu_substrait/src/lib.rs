use prost::Message;

mod extensions;
mod proto;
mod translate;

pub use translate::translate;

/// A Substrait plan: what the compiler hands an engine.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan(substrait::proto::Plan);

impl Plan {
    /// The plan in Substrait's protobuf encoding.
    #[must_use]
    pub fn to_protobuf(&self) -> Vec<u8> {
        self.0.encode_to_vec()
    }

    /// The plan as pretty-printed JSON.
    ///
    /// # Panics
    ///
    /// Panics if the plan does not serialize, which a plan always does.
    #[must_use]
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self.0).expect("a plan always serializes")
    }
}
