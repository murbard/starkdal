//! Binary Blake2s Merkle tree over chunked FFT evaluations.
//!
//! Evaluations are partitioned into `n_total / 2^log_felts_per_leaf` consecutive chunks; each
//! chunk is hashed via `hash_chunk` to form a leaf. Internal nodes hash the concatenation of
//! their two children digests via `hash_pair`. The number of leaves must be a power of two.

use crate::blake_hash::{hash_chunk, hash_pair};

pub fn merkle_root(
    evals: @Array<felt252>, log_felts_per_leaf: u32,
) -> Box<[u32; 8]> {
    let f: u32 = pow2(log_felts_per_leaf);
    let n_total: u32 = evals.len();
    assert(n_total >= f, 'too few evals');
    assert(n_total % f == 0, 'evals not multiple of f');
    let n_leaves: u32 = n_total / f;
    assert(n_leaves > 0, 'no leaves');
    assert(is_pow2(n_leaves), 'n_leaves not pow2');

    // Layer 0: hash each chunk into a leaf digest.
    let mut layer: Array<Box<[u32; 8]>> = ArrayTrait::new();
    let mut chunk_idx: u32 = 0;
    while chunk_idx < n_leaves {
        let mut chunk: Array<felt252> = ArrayTrait::new();
        let base = chunk_idx * f;
        let mut j: u32 = 0;
        while j < f {
            chunk.append(*evals.at(base + j));
            j += 1;
        };
        layer.append(hash_chunk(chunk.span()));
        chunk_idx += 1;
    };

    // Reduce two-by-two until a single root remains.
    while layer.len() > 1 {
        let mut next: Array<Box<[u32; 8]>> = ArrayTrait::new();
        let span = layer.span();
        let mut k: u32 = 0;
        while k < span.len() {
            let l = *span.at(k);
            let r = *span.at(k + 1);
            next.append(hash_pair(l, r));
            k += 2;
        };
        layer = next;
    };

    *layer.at(0)
}

fn pow2(e: u32) -> u32 {
    let mut r: u32 = 1;
    let mut i: u32 = 0;
    while i < e {
        r = r * 2;
        i += 1;
    };
    r
}

fn is_pow2(x: u32) -> bool {
    if x == 0 {
        return false;
    }
    let mut v = x;
    while v > 1 {
        if v % 2 == 1 {
            return false;
        }
        v = v / 2;
    };
    true
}
