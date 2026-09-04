use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let llvm = PathBuf::from(env::var("MLIR_SYS_220_PREFIX").expect("MLIR_SYS_220_PREFIX is set"));
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let tblgen = llvm.join("bin/mlir-tblgen");
    let include = llvm.join("include");
    let td = "cpp/YzirDialect.td";

    let generators = [
        ("--gen-dialect-decls", "YzirDialect.h.inc"),
        ("--gen-dialect-defs", "YzirDialect.cpp.inc"),
        ("--gen-typedef-decls", "YzirTypes.h.inc"),
        ("--gen-typedef-defs", "YzirTypes.cpp.inc"),
        ("--gen-op-decls", "YzirOps.h.inc"),
        ("--gen-op-defs", "YzirOps.cpp.inc"),
    ];
    for (flag, file) in generators {
        let status = Command::new(&tblgen)
            .arg(flag)
            .arg(td)
            .arg("-I")
            .arg(&include)
            .arg("-o")
            .arg(out.join(file))
            .status()
            .expect("mlir-tblgen runs");
        assert!(status.success(), "mlir-tblgen {flag} failed");
    }

    cc::Build::new()
        .cpp(true)
        .std("c++20")
        .compiler(llvm.join("bin/clang++"))
        .files(["cpp/YzirDialect.cpp", "cpp/Register.cpp"])
        .include(&out)
        .include("cpp")
        .include(&include)
        .flag("-w")
        .compile("yuzu_ir");

    println!("cargo:rerun-if-changed=cpp");
}
