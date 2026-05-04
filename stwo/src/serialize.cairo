//! Serialize a Stark252 felt to 8 little-endian u32 limbs for Blake2s consumption.
//!
//! Convention: limb[i] holds bits [32*i, 32*(i+1)) of the felt's non-negative integer
//! representation. Concatenating limbs as little-endian u32 bytes reproduces the felt's
//! 32-byte little-endian byte serialization. Stark252 felts fit in 252 bits, so the top
//! 4 bits of limb[7] are always zero.

pub fn felt_to_u32x8(x: felt252) -> [u32; 8] {
    let u: u256 = x.into();
    let mut lo: u128 = u.low;
    let mut hi: u128 = u.high;
    let shift: u128 = 0x100000000;

    let w0_u: u128 = lo % shift;
    lo = lo / shift;
    let w1_u: u128 = lo % shift;
    lo = lo / shift;
    let w2_u: u128 = lo % shift;
    lo = lo / shift;
    let w3_u: u128 = lo;

    let w4_u: u128 = hi % shift;
    hi = hi / shift;
    let w5_u: u128 = hi % shift;
    hi = hi / shift;
    let w6_u: u128 = hi % shift;
    hi = hi / shift;
    let w7_u: u128 = hi;

    [
        w0_u.try_into().unwrap(),
        w1_u.try_into().unwrap(),
        w2_u.try_into().unwrap(),
        w3_u.try_into().unwrap(),
        w4_u.try_into().unwrap(),
        w5_u.try_into().unwrap(),
        w6_u.try_into().unwrap(),
        w7_u.try_into().unwrap(),
    ]
}
