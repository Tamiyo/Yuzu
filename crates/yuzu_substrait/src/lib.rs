use prost::Message;

mod emitter;
mod extensions;
mod proto;
mod translate;

pub use emitter::emit;
pub use substrait::proto::Plan;
pub use translate::translate;

pub fn to_protobuf(plan: &Plan) -> Vec<u8> {
    plan.encode_to_vec()
}

pub fn to_json(plan: &Plan) -> String {
    serde_json::to_string_pretty(plan).expect("a plan always serializes")
}
