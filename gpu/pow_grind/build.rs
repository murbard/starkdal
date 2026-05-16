use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernel_src = "kernel/pow_grind.cu";
    let cubin_path = out_dir.join("pow_grind.cubin");

    let nvcc = env::var("CUDA_HOME")
        .map(|h| format!("{h}/bin/nvcc"))
        .unwrap_or_else(|_| "nvcc".to_string());

    // Need to include the parent gpu/ directory so the shared header is found.
    let gpu_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .parent()
        .unwrap()
        .to_path_buf();

    let status = Command::new(&nvcc)
        .args([
            "--cubin",
            "-o",
            cubin_path.to_str().unwrap(),
            kernel_src,
            &format!(
                "-arch=sm_{}",
                env::var("CUDA_ARCH").unwrap_or_else(|_| "86".to_string())
            ),
            "-O3",
            "--use_fast_math",
            &format!("-I{}", gpu_dir.display()),
        ])
        .status()
        .unwrap_or_else(|e| panic!("Failed to run nvcc ({nvcc}): {e}"));

    assert!(status.success(), "nvcc compilation failed");

    println!("cargo:rerun-if-changed={kernel_src}");
    println!("cargo:rerun-if-changed=../../field/koalabear_field.cuh");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
}
