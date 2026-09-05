use std::ffi::CString;

use melior::ir::{BlockLike, OperationRef, RegionLike, r#type::TypeId};
use melior::pass::{ExternalPass, Pass, external::create_external};

#[repr(align(8))]
struct PassId;

static TARGET_LEGALITY: PassId = PassId;

/// A pass that rejects every op the compilation target cannot run, reporting
/// each offender at its own location.
pub fn create(illegal: Vec<String>) -> Pass {
    create_external(
        move |operation: OperationRef, pass: ExternalPass| {
            let mut legal = true;
            walk(operation, &mut |op| {
                use melior::ir::operation::OperationLike;
                let name = op
                    .name()
                    .as_string_ref()
                    .as_str()
                    .unwrap_or_default()
                    .to_string();
                if illegal.contains(&name) {
                    let message =
                        CString::new(format!("`{name}` is not supported by the target")).unwrap();
                    unsafe { mlir_sys::mlirEmitError(op.location().to_raw(), message.as_ptr()) };
                    legal = false;
                }
            });
            if !legal {
                pass.signal_failure();
            }
        },
        TypeId::create(&TARGET_LEGALITY),
        "yuzu-target-legality",
        "yuzu-target-legality",
        "reject ops the compilation target does not support",
        "",
        &[],
    )
}

fn walk<'c, 'a>(operation: OperationRef<'c, 'a>, visit: &mut impl FnMut(OperationRef<'c, 'a>)) {
    use melior::ir::operation::OperationLike;
    visit(operation);
    for region in operation.regions() {
        let mut block = region.first_block();
        while let Some(current) = block {
            let mut op = current.first_operation();
            while let Some(inner) = op {
                walk(inner, visit);
                op = inner.next_in_block();
            }
            block = current.next_in_region();
        }
    }
}
