//! Property-based tests: GPU Merkle tree matches CPU reference.

use cudarc::driver::safe::CudaContext;
use gpu_merkle::*;
use proptest::prelude::*;

const P: u32 = 0x7F000001;

fn gpu() -> GpuMerkle {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let stream = ctx.default_stream();
    GpuMerkle::new(stream)
}

// ── Leaf hashing ─────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Random 16-element rows (single sponge chunk).
    #[test]
    fn prop_leaf_hash_16(row in prop::collection::vec(0..P, 16)) {
        let g = gpu();
        let (gpu_root, gpu_layers) = g.build_tree(&row, 1, 16, 16);
        let cpu_digest = cpu_leaf_hash(&row);
        prop_assert_eq!(&gpu_layers[0][..8], &cpu_digest[..]);
    }

    /// Random 32-element rows (two sponge chunks).
    #[test]
    fn prop_leaf_hash_32(row in prop::collection::vec(0..P, 32)) {
        let g = gpu();
        let (_, gpu_layers) = g.build_tree(&row, 1, 32, 32);
        let cpu_digest = cpu_leaf_hash(&row);
        prop_assert_eq!(&gpu_layers[0][..8], &cpu_digest[..]);
    }

    /// Random 64-element rows (multiple sponge chunks).
    #[test]
    fn prop_leaf_hash_64(row in prop::collection::vec(0..P, 64)) {
        let g = gpu();
        let (_, gpu_layers) = g.build_tree(&row, 1, 64, 64);
        let cpu_digest = cpu_leaf_hash(&row);
        prop_assert_eq!(&gpu_layers[0][..8], &cpu_digest[..]);
    }
}

// ── Full tree construction ───────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    /// 4 rows × 16 elements: small tree with 2 reduction layers.
    #[test]
    fn prop_full_tree_4x16(data in prop::collection::vec(0..P, 4 * 16)) {
        let g = gpu();
        let (gpu_root, gpu_layers) = g.build_tree(&data, 4, 16, 16);
        let (cpu_root, cpu_layers) = cpu_build_tree(&data, 4, 16, 16);

        // Check leaf layer.
        prop_assert_eq!(&gpu_layers[0], &cpu_layers[0]);
        // Check all internal layers.
        for (i, (gl, cl)) in gpu_layers.iter().zip(cpu_layers.iter()).enumerate() {
            prop_assert_eq!(gl, cl);
        }
        // Check root.
        prop_assert_eq!(&gpu_root, &cpu_root);
    }

    /// 8 rows × 32 elements: bigger tree.
    #[test]
    fn prop_full_tree_8x32(data in prop::collection::vec(0..P, 8 * 32)) {
        let g = gpu();
        let (gpu_root, _) = g.build_tree(&data, 8, 32, 32);
        let (cpu_root, _) = cpu_build_tree(&data, 8, 32, 32);
        prop_assert_eq!(&gpu_root, &cpu_root);
    }

    /// 16 rows × 64 elements.
    #[test]
    fn prop_full_tree_16x64(data in prop::collection::vec(0..P, 16 * 64)) {
        let g = gpu();
        let (gpu_root, _) = g.build_tree(&data, 16, 64, 64);
        let (cpu_root, _) = cpu_build_tree(&data, 16, 64, 64);
        prop_assert_eq!(&gpu_root, &cpu_root);
    }
}

// ── Deterministic tests ──────────────────────────────────────────────────

#[test]
fn test_tree_2x16_all_zeros() {
    let g = gpu();
    let data = vec![0u32; 2 * 16];
    let (gpu_root, gpu_layers) = g.build_tree(&data, 2, 16, 16);
    let (cpu_root, cpu_layers) = cpu_build_tree(&data, 2, 16, 16);
    assert_eq!(gpu_layers, cpu_layers);
    assert_eq!(gpu_root, cpu_root);
}

#[test]
fn test_tree_4x16_sequential() {
    let g = gpu();
    let data: Vec<u32> = (0..4 * 16).map(|i| (i as u32) % P).collect();
    let (gpu_root, gpu_layers) = g.build_tree(&data, 4, 16, 16);
    let (cpu_root, cpu_layers) = cpu_build_tree(&data, 4, 16, 16);
    assert_eq!(gpu_layers, cpu_layers);
    assert_eq!(gpu_root, cpu_root);
}

#[test]
fn test_tree_large_256x32() {
    let g = gpu();
    let data: Vec<u32> = (0..256 * 32)
        .map(|i| ((i as u64 * 997 + 7) % P as u64) as u32)
        .collect();
    let (gpu_root, _) = g.build_tree(&data, 256, 32, 32);
    let (cpu_root, _) = cpu_build_tree(&data, 256, 32, 32);
    assert_eq!(gpu_root, cpu_root);
}

#[test]
fn test_tree_1024x16() {
    let g = gpu();
    let data: Vec<u32> = (0..1024 * 16)
        .map(|i| ((i as u64 * 1337 + 42) % P as u64) as u32)
        .collect();
    let (gpu_root, _) = g.build_tree(&data, 1024, 16, 16);
    let (cpu_root, _) = cpu_build_tree(&data, 1024, 16, 16);
    assert_eq!(gpu_root, cpu_root);
}

#[test]
fn test_row_stride_larger_than_width() {
    // Row stride > row width (padding between rows).
    let g = gpu();
    let stride = 24;
    let width = 16;
    let height = 4;
    let mut data = vec![0u32; height * stride];
    for r in 0..height {
        for c in 0..width {
            data[r * stride + c] = ((r * width + c) as u32) % P;
        }
    }
    let (gpu_root, _) = g.build_tree(&data, height as u32, width as u32, stride as u32);
    let (cpu_root, _) = cpu_build_tree(&data, height, width, stride);
    assert_eq!(gpu_root, cpu_root);
}
