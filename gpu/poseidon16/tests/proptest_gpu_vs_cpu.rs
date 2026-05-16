//! Property-based tests: GPU Poseidon16 matches CPU reference on random inputs.
//!
//! Run with: cargo test --release -- --nocapture

use std::sync::Arc;

use cudarc::driver::safe::CudaContext;
use gpu_poseidon16::{GpuPoseidon16, cpu_compress, cpu_permute};
use koala_bear::KoalaBear;
use proptest::prelude::*;

const P: u32 = 0x7F000001;

fn arb_state() -> impl Strategy<Value = [u32; 16]> {
    prop::array::uniform16(0..P)
}

fn arb_batch(max_n: usize) -> impl Strategy<Value = Vec<[u32; 16]>> {
    prop::collection::vec(arb_state(), 1..=max_n)
}

fn gpu_context() -> (Arc<cudarc::driver::safe::CudaStream>, GpuPoseidon16) {
    let ctx = CudaContext::new(0).expect("CUDA device required for GPU tests");
    let stream = ctx.default_stream();
    let gpu = GpuPoseidon16::new(stream.clone());
    (stream, gpu)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_compress_single(state in arb_state()) {
        let (stream, gpu) = gpu_context();
        let d_input = stream.memcpy_stod(&state).unwrap();
        let d_output = gpu.compress_batch(&d_input, 1);
        stream.synchronize().unwrap();
        let gpu_out = stream.memcpy_dtov(&d_output).unwrap();

        let expected = cpu_compress(&state);
        prop_assert_eq!(gpu_out.as_slice(), expected.as_slice());
    }

    #[test]
    fn prop_compress_batch(states in arb_batch(512)) {
        let n = states.len();
        let (stream, gpu) = gpu_context();
        let input_flat: Vec<u32> = states.iter().flat_map(|s| s.iter().copied()).collect();
        let d_input = stream.memcpy_stod(&input_flat).unwrap();
        let d_output = gpu.compress_batch(&d_input, n as u32);
        stream.synchronize().unwrap();
        let gpu_out = stream.memcpy_dtov(&d_output).unwrap();

        for (i, state) in states.iter().enumerate() {
            let expected = cpu_compress(state);
            let got: &[u32] = &gpu_out[i * 8..(i + 1) * 8];
            prop_assert_eq!(got, expected.as_slice());
        }
    }

    #[test]
    fn prop_permute_single(state in arb_state()) {
        let (stream, gpu) = gpu_context();
        let d_input = stream.memcpy_stod(&state).unwrap();
        let d_output = gpu.permute_batch(&d_input, 1);
        stream.synchronize().unwrap();
        let gpu_out = stream.memcpy_dtov(&d_output).unwrap();

        let mut expected = state;
        cpu_permute(&mut expected);
        prop_assert_eq!(gpu_out.as_slice(), expected.as_slice());
    }
}

#[test]
fn test_known_vector_gpu() {
    let (stream, gpu) = gpu_context();

    let input_kb = KoalaBear::new_array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    let input_u32: [u32; 16] = unsafe { std::mem::transmute(input_kb) };

    let d_input = stream.memcpy_stod(&input_u32).unwrap();
    let d_output = gpu.permute_batch(&d_input, 1);
    stream.synchronize().unwrap();
    let gpu_out = stream.memcpy_dtov(&d_output).unwrap();

    let canonical: Vec<u32> = gpu_out
        .iter()
        .map(|&v| {
            let kb: KoalaBear = unsafe { std::mem::transmute(v) };
            field::PrimeField32::as_canonical_u32(&kb)
        })
        .collect();

    assert_eq!(
        canonical,
        vec![
            610090613, 935319874, 1893335292, 796792199, 356405232, 552237741, 55134556,
            1215104204, 1823723405, 1133298033, 1780633798, 1453946561, 710069176, 1128629550,
            1917333254, 1175481618,
        ],
    );
}

#[test]
fn test_edge_cases_gpu() {
    let (stream, gpu) = gpu_context();

    let cases: Vec<[u32; 16]> = vec![[0u32; 16], [1u32; 16], [P - 1; 16]];

    for (idx, state) in cases.iter().enumerate() {
        let d_input = stream.memcpy_stod(state.as_slice()).unwrap();
        let d_output = gpu.compress_batch(&d_input, 1);
        stream.synchronize().unwrap();
        let gpu_out = stream.memcpy_dtov(&d_output).unwrap();

        let expected = cpu_compress(state);
        assert_eq!(
            gpu_out.as_slice(),
            expected.as_slice(),
            "Edge case {idx} mismatch"
        );
    }
}
