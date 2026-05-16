// GPU Fiat-Shamir challenger — device code.
// Mirrors the CPU's Challenger<KoalaBear, Poseidon16Compress>.
// Uses the device-callable Poseidon16 from poseidon16_device.cuh.

#pragma once
#include "poseidon16_device.cuh"

#define FS_RATE 8
#define FS_WIDTH 16

// Challenger state: 8 KoalaBear elements (Montgomery form) + pointers to constants.
struct GpuChallenger {
    uint32_t state[FS_RATE];
    const uint32_t* rc;      // 28 * 16 round constants
    const uint32_t* mds;     // 16-element MDS circulant
    const uint32_t* sparse;  // sparse matrix data (912 elements)
};

// Initialize challenger with zero state and constant pointers.
__device__ void challenger_init(GpuChallenger* ch,
    const uint32_t* rc, const uint32_t* mds, const uint32_t* sparse)
{
    for (int i = 0; i < FS_RATE; i++) ch->state[i] = 0;
    ch->rc = rc;
    ch->mds = mds;
    ch->sparse = sparse;
}

// Observe 8 scalars: new_state = compress(state || input)[0..8]
__device__ void challenger_observe(GpuChallenger* ch, const uint32_t input[FS_RATE]) {
    uint32_t buf[FS_WIDTH];
    for (int i = 0; i < FS_RATE; i++) buf[i] = ch->state[i];
    for (int i = 0; i < FS_RATE; i++) buf[FS_RATE + i] = input[i];
    p16_compress(buf, ch->rc, ch->mds, ch->sparse);
    for (int i = 0; i < FS_RATE; i++) ch->state[i] = buf[i];
}

// Observe arbitrary number of scalars (padded to RATE chunks).
__device__ void challenger_observe_scalars(GpuChallenger* ch, const uint32_t* scalars, int n) {
    for (int off = 0; off < n; off += FS_RATE) {
        uint32_t buf[FS_RATE];
        for (int i = 0; i < FS_RATE; i++) {
            buf[i] = (off + i < n) ? scalars[off + i] : 0;
        }
        challenger_observe(ch, buf);
    }
}

// Sample n+1 blocks of 8 elements, return first n, use last as new state.
// Caller provides output buffer of size n * FS_RATE.
__device__ void challenger_sample_many(GpuChallenger* ch, int n, uint32_t* output) {
    for (int i = 0; i <= n; i++) {
        uint32_t buf[FS_WIDTH];
        buf[0] = kb_to_monty(i);
        for (int j = 1; j < FS_RATE; j++) buf[j] = 0;
        for (int j = 0; j < FS_RATE; j++) buf[FS_RATE + j] = ch->state[j];
        p16_compress(buf, ch->rc, ch->mds, ch->sparse);
        if (i < n) {
            for (int j = 0; j < FS_RATE; j++) output[i * FS_RATE + j] = buf[j];
        } else {
            for (int j = 0; j < FS_RATE; j++) ch->state[j] = buf[j];
        }
    }
}

// Sample one quintic extension field element (5 KoalaBear elements).
__device__ void challenger_sample_ext(GpuChallenger* ch, uint32_t out[5]) {
    uint32_t sampled[FS_RATE];
    challenger_sample_many(ch, 1, sampled);
    for (int i = 0; i < 5; i++) out[i] = sampled[i];
}
