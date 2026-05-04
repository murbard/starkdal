//! Blake2s wrappers for Merkle leaf and internal-node hashing.
//!
//! The Cairo `blake2s_compress` / `blake2s_finalize` extern fns expect message blocks of
//! 16 little-endian u32 words (64 bytes). Each felt serializes to 8 u32 words (32 bytes), so
//! a single block holds two felts. For chunks of N felts we issue (N+1)/2 blocks: all but the
//! last via `blake2s_compress`, the last via `blake2s_finalize` (which sets the final flag and
//! emits the digest). For odd N the trailing block holds one felt + 32 zero bytes; this matches
//! the standard "zero-pad to a multiple of 64 bytes" behavior of Blake2s.

use core::blake::{blake2s_compress, blake2s_finalize};
use core::box::BoxTrait;
use crate::serialize::felt_to_u32x8;

/// Initial Blake2s state for a keyless hash with 32-byte output.
/// state[0] = IV[0] XOR (parameter block as u32 LE word) = 0x6A09E667 XOR 0x01010020.
const BLAKE2S_INIT: [u32; 8] = [
    0x6B08E647, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

fn init_state() -> Box<[u32; 8]> {
    BoxTrait::new(BLAKE2S_INIT)
}

fn pack_two_felts(a: felt252, b: felt252) -> Box<[u32; 16]> {
    let aw = felt_to_u32x8(a).span();
    let bw = felt_to_u32x8(b).span();
    BoxTrait::new(
        [
            *aw.at(0), *aw.at(1), *aw.at(2), *aw.at(3),
            *aw.at(4), *aw.at(5), *aw.at(6), *aw.at(7),
            *bw.at(0), *bw.at(1), *bw.at(2), *bw.at(3),
            *bw.at(4), *bw.at(5), *bw.at(6), *bw.at(7),
        ],
    )
}

fn pack_one_felt(a: felt252) -> Box<[u32; 16]> {
    let aw = felt_to_u32x8(a).span();
    BoxTrait::new(
        [
            *aw.at(0), *aw.at(1), *aw.at(2), *aw.at(3),
            *aw.at(4), *aw.at(5), *aw.at(6), *aw.at(7),
            0, 0, 0, 0, 0, 0, 0, 0,
        ],
    )
}

fn pack_two_digests(left: Box<[u32; 8]>, right: Box<[u32; 8]>) -> Box<[u32; 16]> {
    let l = left.unbox();
    let r = right.unbox();
    let l = l.span();
    let r = r.span();
    BoxTrait::new(
        [
            *l.at(0), *l.at(1), *l.at(2), *l.at(3),
            *l.at(4), *l.at(5), *l.at(6), *l.at(7),
            *r.at(0), *r.at(1), *r.at(2), *r.at(3),
            *r.at(4), *r.at(5), *r.at(6), *r.at(7),
        ],
    )
}

/// Hash `felts` as the concatenation of their 32-byte little-endian encodings.
/// Asserts `felts.len() > 0`.
pub fn hash_chunk(felts: Span<felt252>) -> Box<[u32; 8]> {
    let n: u32 = felts.len();
    assert(n > 0, 'hash_chunk: empty');
    let total_bytes: u32 = 32 * n;
    let n_pairs: u32 = n / 2;
    let has_singleton: u32 = n % 2;
    let n_blocks: u32 = n_pairs + has_singleton;

    let mut state = init_state();
    let mut bytes_done: u32 = 0;
    let mut i: u32 = 0;
    while i + 1 < n_blocks {
        let f0 = *felts.at(2 * i);
        let f1 = *felts.at(2 * i + 1);
        let block = pack_two_felts(f0, f1);
        bytes_done = bytes_done + 64;
        state = blake2s_compress(state, bytes_done, block);
        i = i + 1;
    };

    let final_block = if has_singleton == 1 {
        pack_one_felt(*felts.at(n - 1))
    } else {
        pack_two_felts(*felts.at(n - 2), *felts.at(n - 1))
    };
    blake2s_finalize(state, total_bytes, final_block)
}

/// Hash a pair of 32-byte digests (Blake2s of the 64-byte concatenation).
pub fn hash_pair(left: Box<[u32; 8]>, right: Box<[u32; 8]>) -> Box<[u32; 8]> {
    let block = pack_two_digests(left, right);
    blake2s_finalize(init_state(), 64, block)
}
