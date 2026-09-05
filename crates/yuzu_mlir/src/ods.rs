//! Typed op constructors, generated straight from the dialect definitions in
//! yuzu_ir_sys — the same records the C++ is generated from.

melior_macro::dialect! {
    name: "yz",
    files: ["YzDialect.td"],
    include_directories: ["./crates/yuzu_ir_sys/cpp"],
}

melior_macro::dialect! {
    name: "yzl",
    files: ["YzlDialect.td"],
    include_directories: ["./crates/yuzu_ir_sys/cpp"],
}

melior_macro::dialect! {
    name: "yzr",
    files: ["YzrDialect.td"],
    include_directories: ["./crates/yuzu_ir_sys/cpp"],
}
