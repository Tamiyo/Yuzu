use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let llvm = PathBuf::from(env::var("MLIR_SYS_220_PREFIX").expect("MLIR_SYS_220_PREFIX is set"));
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let tblgen = llvm.join("bin/mlir-tblgen");
    let include = llvm.join("include");

    let generators = [
        ("--gen-dialect-decls", "Dialect.h.inc"),
        ("--gen-dialect-defs", "Dialect.cpp.inc"),
        ("--gen-typedef-decls", "Types.h.inc"),
        ("--gen-typedef-defs", "Types.cpp.inc"),
        ("--gen-op-decls", "Ops.h.inc"),
        ("--gen-op-defs", "Ops.cpp.inc"),
    ];
    for dialect in ["Yz", "Yzl", "Yzr"] {
        for (flag, suffix) in generators {
            // yzr declares no types of its own; its ops are typed by yz's.
            if dialect == "Yzr" && flag.starts_with("--gen-typedef") {
                continue;
            }

            let status = Command::new(&tblgen)
                .arg(flag)
                .arg(format!("cpp/{dialect}Dialect.td"))
                .arg("-I")
                .arg(&include)
                .arg("-o")
                .arg(out.join(format!("{dialect}{suffix}")))
                .status()
                .expect("mlir-tblgen runs");
            assert!(status.success(), "mlir-tblgen {flag} failed for {dialect}");
        }
    }

    cc::Build::new()
        .cpp(true)
        .std("c++20")
        .compiler(llvm.join("bin/clang++"))
        .files([
            "cpp/YzDialect.cpp",
            "cpp/YzlDialect.cpp",
            "cpp/YzrDialect.cpp",
            "cpp/Register.cpp",
        ])
        .include(&out)
        .include("cpp")
        .include(&include)
        .flag("-w")
        .compile("yuzu_dialects");

    println!("cargo:rerun-if-changed=cpp");
}
