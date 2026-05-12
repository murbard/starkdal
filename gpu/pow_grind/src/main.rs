//! Benchmark GPU vs CPU proof-of-work grinding.

use std::time::Instant;

use clap::Parser;
use cudarc::driver::safe::CudaContext;
use gpu_pow_grind::{GpuPowGrinder, cpu_pow_grind};

#[derive(Parser)]
#[command(about = "GPU PoW grinding benchmark")]
struct Cli {
    /// Target number of trailing zero bits.
    #[arg(long, default_value_t = 16)]
    bits: u32,

    /// Also run CPU for comparison.
    #[arg(long)]
    bench_cpu: bool,
}

fn main() {
    let cli = Cli::parse();
    let state = [0u32; 16];
    let nonce_slot = 8u32;

    println!("=== PoW Grinding (target: {} bits) ===", cli.bits);

    // GPU
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let stream = ctx.default_stream();
    let gpu = GpuPowGrinder::new(stream.clone());

    let t0 = Instant::now();
    let gpu_nonce = gpu.grind(&state, nonce_slot, cli.bits, 1 << 28)
        .expect("GPU failed to find nonce");
    let gpu_time = t0.elapsed();
    println!("GPU: found nonce {gpu_nonce:#010x} in {:.3} ms", gpu_time.as_secs_f64() * 1e3);

    // CPU
    if cli.bench_cpu {
        let t0 = Instant::now();
        let cpu_nonce = cpu_pow_grind(&state, nonce_slot as usize, cli.bits);
        let cpu_time = t0.elapsed();
        println!("CPU: found nonce {cpu_nonce:#010x} in {:.3} ms", cpu_time.as_secs_f64() * 1e3);
        println!("Speedup: {:.1}x", cpu_time.as_secs_f64() / gpu_time.as_secs_f64());
    }
}
