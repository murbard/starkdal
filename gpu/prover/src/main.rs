//! Status binary for the legacy `gpu-prover` prototype.

fn main() {
    println!("=== GPU Prover Status ===\n");
    println!("This crate is a legacy prototype / kernel harness.");
    println!("The target generic GPU prover is in leanvm/crates/lean_prover with --features gpu.");
    println!("\nCurrent generic validation command:");
    println!("  cd leanvm");
    println!("  cargo test -p lean_prover --lib --features gpu test_zkvm::test_zk_vm_all_precompiles -- --nocapture");
    println!("\nDo not treat gpu/prover's CPU-orchestrated helpers as the target architecture.");
}
