//! Coset NTT over Stark252.
//!
//! `coset_fft(coeffs, log_n, log_blowup)` evaluates the polynomial whose coefficients are
//! `coeffs[0..2^log_n)` at the `2^(log_n+log_blowup)` points `COSET_OFFSET * omega^k`, where
//! `omega` is a primitive `2^(log_n+log_blowup)`-th root of unity in Stark252.
//!
//! Algorithm: standard iterative Cooley-Tukey radix-2 NTT (decimation-in-time). Input is
//! pre-twisted by `g^i` (so an ordinary NTT then evaluates on the coset), zero-padded to the
//! evaluation-domain size, and stored in bit-reversed positions in a `Felt252Dict` for in-place
//! butterflies. Twiddles (`omega^i` for i in 0..N/2) are precomputed once.
//!
//! Output is returned as `Array<felt252>` in natural index order (output[k] = P(g * omega^k)).

use core::dict::Felt252Dict;
use crate::constants::{COSET_OFFSET, MAX_LOG, OMEGA_MAX};

pub fn coset_fft(coeffs: @Array<felt252>, log_n: u32, log_blowup: u32) -> Array<felt252> {
    let log_total: u32 = log_n + log_blowup;
    assert(log_total <= MAX_LOG, 'log_total exceeds MAX_LOG');
    let n: u32 = pow2(log_n);
    let n_eval: u32 = pow2(log_total);
    assert(coeffs.len() == n, 'wrong coeffs length');

    // omega = OMEGA_MAX^(2^(MAX_LOG - log_total)), i.e. squared down (MAX_LOG - log_total) times
    let mut omega: felt252 = OMEGA_MAX;
    let mut sq: u32 = MAX_LOG - log_total;
    while sq > 0 {
        omega = omega * omega;
        sq -= 1;
    };

    // Precompute twiddles[0..n_eval/2] = [1, omega, omega^2, ..., omega^(n_eval/2 - 1)]
    let twiddles = precompute_twiddles(omega, n_eval);
    let twiddles = twiddles.span();

    // Initialize: place pre-twisted coefficients at bit-reversed positions.
    // Twist: c'[i] = c[i] * COSET_OFFSET^i so that the NTT then evaluates on the coset.
    // Slots [n, n_eval) are implicitly zero (Felt252Dict default).
    let mut data: Felt252Dict<felt252> = Default::default();
    let mut g_pow: felt252 = 1;
    let mut i: u32 = 0;
    while i < n {
        let twisted = *coeffs.at(i) * g_pow;
        data.insert(bit_reverse(i, log_total).into(), twisted);
        g_pow = g_pow * COSET_OFFSET;
        i += 1;
    };

    // Iterative butterflies. Layer s has butterfly size m = 2^s, half = m/2,
    // twiddle stride = n_eval/m within the precomputed table.
    let mut layer: u32 = 1;
    while layer <= log_total {
        let m: u32 = pow2(layer);
        let half: u32 = m / 2;
        let stride: u32 = n_eval / m;
        let mut group_start: u32 = 0;
        while group_start < n_eval {
            let mut j: u32 = 0;
            while j < half {
                let w = *twiddles.at(j * stride);
                let u = data.get((group_start + j).into());
                let t = w * data.get((group_start + j + half).into());
                data.insert((group_start + j).into(), u + t);
                data.insert((group_start + j + half).into(), u - t);
                j += 1;
            };
            group_start += m;
        };
        layer += 1;
    };

    // Drain dict into array in natural index order.
    let mut out: Array<felt252> = ArrayTrait::new();
    let mut k: u32 = 0;
    while k < n_eval {
        out.append(data.get(k.into()));
        k += 1;
    };
    out
}

fn precompute_twiddles(omega: felt252, n: u32) -> Array<felt252> {
    let half = n / 2;
    let mut t: Array<felt252> = ArrayTrait::new();
    if half == 0 {
        return t;
    }
    t.append(1);
    let mut acc: felt252 = 1;
    let mut i: u32 = 1;
    while i < half {
        acc = acc * omega;
        t.append(acc);
        i += 1;
    };
    t
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

fn bit_reverse(x: u32, bits: u32) -> u32 {
    let mut r: u32 = 0;
    let mut v = x;
    let mut i: u32 = 0;
    while i < bits {
        r = r * 2 + (v % 2);
        v = v / 2;
        i += 1;
    };
    r
}
