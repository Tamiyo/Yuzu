use prost::Message;

mod extensions;
mod proto;
mod translate;

pub use substrait::proto::Plan;
pub use translate::translate;

pub fn to_protobuf(plan: &Plan) -> Vec<u8> {
    plan.encode_to_vec()
}

/// A plan as pretty-printed JSON.
///
/// # Panics
///
/// Panics if the plan does not serialize, which a plan always does.
pub fn to_json(plan: &Plan) -> String {
    serde_json::to_string_pretty(plan).expect("a plan always serializes")
}
