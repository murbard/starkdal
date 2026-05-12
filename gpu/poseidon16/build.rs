use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernel_src = "kernel/poseidon16.cu";
    let ptx_path = out_dir.join("poseidon16.ptx");

    // Find nvcc: prefer CUDA_HOME, then PATH.
    let nvcc = env::var("CUDA_HOME")
        .map(|h| format!("{h}/bin/nvcc"))
        .unwrap_or_else(|_| "nvcc".to_string());

    let status = Command::new(&nvcc)
        .args([
            "--ptx",
            "-o",
            ptx_path.to_str().unwrap(),
            kernel_src,
            // Target compute capability 8.6 (Ampere — RTX 3060).
            // Override with CUDA_ARCH env var for other GPUs.
            &format!(
                "-arch=sm_{}",
                env::var("CUDA_ARCH").unwrap_or_else(|_| "86".to_string())
            ),
            "-O3",
            "--use_fast_math",
        ])
        .status()
        .unwrap_or_else(|e| panic!("Failed to run nvcc ({nvcc}): {e}"));

    assert!(status.success(), "nvcc compilation failed");

    println!("cargo:rerun-if-changed={kernel_src}");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
}
