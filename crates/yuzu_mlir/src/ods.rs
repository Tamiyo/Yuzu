//! Typed op constructors, generated straight from the dialect definitions in
//! `yuzu_mlir_sys`. The C++ build generates its code from the same records.

melior_macro::dialect! {
    name: "yz",
    files: ["YzDialect.td"],
    include_directories: ["./crates/yuzu_mlir_sys/cpp"],
}

melior_macro::dialect! {
    name: "yzl",
    files: ["YzlDialect.td"],
    include_directories: ["./crates/yuzu_mlir_sys/cpp"],
}

melior_macro::dialect! {
    name: "yzr",
    files: ["YzrDialect.td"],
    include_directories: ["./crates/yuzu_mlir_sys/cpp"],
}
