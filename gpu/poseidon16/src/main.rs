//! Benchmark and correctness test for GPU Poseidon16.
//!
//! Usage:
//!   cargo run --release -- --n 1000000                      # bench 1M compress calls
//!   cargo run --release -- --n 1000 --verify                # verify GPU matches CPU
//!   cargo run --release -- --bench-cpu --n 1000000          # GPU vs CPU comparison

use std::time::Instant;

use koala_bear::{KoalaBear, default_koalabear_poseidon1_16};
use koala_bear::symmetric::Permutation;
use clap::Parser;
use cudarc::driver::safe::CudaContext;
use gpu_poseidon16::{GpuPoseidon16, cpu_compress};
use rand::{RngExt, SeedableRng, rngs::StdRng};

#[derive(Parser)]
#[command(about = "GPU Poseidon16 benchmark and verification")]
struct Cli {
    /// Number of Poseidon16 compress calls.
    #[arg(long, default_value_t = 1_000_000)]
    n: u32,

    /// Verify GPU output matches CPU reference for every input.
    #[arg(long)]
    verify: bool,

    /// Also benchmark CPU for direct comparison.
    #[arg(long)]
    bench_cpu: bool,

    /// GPU device ordinal.
    #[arg(long, default_value_t = 0)]
    device: usize,

    /// Number of warmup iterations before timing.
    #[arg(long, default_value_t = 3)]
    warmup: usize,

    /// Number of timed iterations (reports best).
    #[arg(long, default_value_t = 5)]
    iters: usize,

    /// RNG seed for random inputs.
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn main() {
    let cli = Cli::parse();
    let n = cli.n as usize;

    // Generate random inputs (Montgomery-form u32 values in [0, P)).
    let mut rng = StdRng::seed_from_u64(cli.seed);
    let p = 0x7F000001u32;
    let input_flat: Vec<u32> = (0..n * 16).map(|_| rng.random_range(0..p)).collect();

    println!("=== GPU Poseidon16 ===");
    println!("States: {n}");
    println!(
        "Data:   {:.2} MiB input, {:.2} MiB output",
        (n * 16 * 4) as f64 / (1 << 20) as f64,
        (n * 8 * 4) as f64 / (1 << 20) as f64,
    );

    // ── GPU setup ────────────────────────────────────────────────────────
    let ctx = CudaContext::new(cli.device).expect("failed to open CUDA device");
    let stream = ctx.default_stream();
    let gpu = GpuPoseidon16::new(stream.clone());

    let d_input = stream.memcpy_stod(&input_flat).unwrap();

    // Warmup.
    for _ in 0..cli.warmup {
        let _out = gpu.compress_batch(&d_input, cli.n);
        stream.synchronize().unwrap();
    }

    // Timed runs.
    let mut best_gpu = f64::MAX;
    let mut gpu_output_host: Vec<u32> = vec![];
    for _ in 0..cli.iters {
        let t0 = Instant::now();
        let d_output = gpu.compress_batch(&d_input, cli.n);
        stream.synchronize().unwrap();
        let elapsed = t0.elapsed().as_secs_f64();
        best_gpu = best_gpu.min(elapsed);

        // Copy last iteration's output for verification.
        gpu_output_host = stream.memcpy_dtov(&d_output).unwrap();
    }

    let gpu_throughput = n as f64 / best_gpu;
    let gpu_data_throughput = (n * 16 * 4) as f64 / best_gpu / (1 << 20) as f64;
    println!("\nGPU (best of {}):", cli.iters);
    println!("  Time:       {:.3} ms", best_gpu * 1e3);
    println!("  Throughput: {:.2} M compress/s", gpu_throughput / 1e6);
    println!("  Bandwidth:  {:.2} MiB/s input", gpu_data_throughput);

    // ── CPU benchmark ────────────────────────────────────────────────────
    if cli.bench_cpu {
        let poseidon = default_koalabear_poseidon1_16();

        // Warmup.
        {
            let mut tmp: [u32; 16] = input_flat[..16].try_into().unwrap();
            let kb: &mut [KoalaBear; 16] =
                unsafe { &mut *(&mut tmp as *mut [u32; 16] as *mut [KoalaBear; 16]) };
            poseidon.compress_in_place(kb);
        }

        let t0 = Instant::now();
        let mut cpu_output = vec![0u32; n * 8];
        for i in 0..n {
            let mut state: [u32; 16] = input_flat[i * 16..(i + 1) * 16].try_into().unwrap();
            let kb: &mut [KoalaBear; 16] =
                unsafe { &mut *(&mut state as *mut [u32; 16] as *mut [KoalaBear; 16]) };
            let initial = *kb;
            poseidon.permute_mut(kb);
            for j in 0..16 {
                kb[j] += initial[j];
            }
            let out: &[u32; 16] =
                unsafe { &*(&*kb as *const [KoalaBear; 16] as *const [u32; 16]) };
            cpu_output[i * 8..(i + 1) * 8].copy_from_slice(&out[..8]);
        }
        let cpu_elapsed = t0.elapsed().as_secs_f64();
        let cpu_throughput = n as f64 / cpu_elapsed;

        println!("\nCPU (single-threaded, {n} states):");
        println!("  Time:       {:.3} ms", cpu_elapsed * 1e3);
        println!("  Throughput: {:.2} M compress/s", cpu_throughput / 1e6);
        println!(
            "  Speedup:    {:.1}x GPU over CPU",
            gpu_throughput / cpu_throughput
        );

        if cli.verify {
            let mismatches = compare_outputs(&gpu_output_host, &cpu_output, n);
            if mismatches == 0 {
                println!("\nVerification: PASS ({n} states match)");
            } else {
                println!("\nVerification: FAIL ({mismatches}/{n} mismatches)");
                std::process::exit(1);
            }
            return;
        }
    }

    // ── Verification (GPU only, compute CPU on-the-fly) ──────────────────
    if cli.verify {
        println!("\nVerifying GPU output against CPU reference...");
        let mut mismatches = 0u64;
        for i in 0..n {
            let input: [u32; 16] = input_flat[i * 16..(i + 1) * 16].try_into().unwrap();
            let expected = cpu_compress(&input);
            let got: [u32; 8] = gpu_output_host[i * 8..(i + 1) * 8].try_into().unwrap();
            if got != expected {
                if mismatches < 5 {
                    eprintln!("  MISMATCH at state {i}:");
                    eprintln!("    input:    {input:08x?}");
                    eprintln!("    expected: {expected:08x?}");
                    eprintln!("    got:      {got:08x?}");
                }
                mismatches += 1;
            }
        }
        if mismatches == 0 {
            println!("Verification: PASS ({n} states match)");
        } else {
            println!("Verification: FAIL ({mismatches}/{n} mismatches)");
            std::process::exit(1);
        }
    }
}

fn compare_outputs(gpu: &[u32], cpu: &[u32], n: usize) -> u64 {
    let mut mismatches = 0u64;
    for i in 0..n {
        let g = &gpu[i * 8..(i + 1) * 8];
        let c = &cpu[i * 8..(i + 1) * 8];
        if g != c {
            if mismatches < 5 {
                eprintln!("  MISMATCH at state {i}: gpu={g:08x?} cpu={c:08x?}");
            }
            mismatches += 1;
        }
    }
    mismatches
}
