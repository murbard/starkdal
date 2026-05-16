//! Property test: GPU gkr_quotient_sumcheck_kernel vs CPU reference.

use field::{BasedVectorSpace, ExtensionField, PrimeCharacteristicRing};
use gpu_sumcheck::GpuSumcheck;
use koala_bear::{KoalaBear, extension::QuinticExtensionField};
use rand::{RngExt, SeedableRng, rngs::StdRng};

type F = KoalaBear;
type EF = QuinticExtensionField<F>;

fn ef_to_u32(v: &EF) -> [u32; 5] {
    unsafe { std::mem::transmute_copy(v) }
}
fn ef_from_u32(v: &[u32; 5]) -> EF {
    unsafe { std::mem::transmute_copy(v) }
}

/// CPU reference: compute (c0_num, c2_num, c0_den, c2_den) for the quotient sumcheck.
///
/// Arrays: num_l, num_r, den_l, den_r each with `2*half` elements (pairs at 2j, 2j+1).
/// eq: `half` elements.
fn cpu_quotient_sumcheck(
    num_l: &[EF],
    num_r: &[EF],
    den_l: &[EF],
    den_r: &[EF],
    eq: &[EF],
    half: usize,
) -> (EF, EF, EF, EF) {
    let mut c0_num = EF::ZERO;
    let mut c2_num = EF::ZERO;
    let mut c0_den = EF::ZERO;
    let mut c2_den = EF::ZERO;

    for j in 0..half {
        let nl_lo = num_l[2 * j];
        let nl_hi = num_l[2 * j + 1];
        let nr_lo = num_r[2 * j];
        let nr_hi = num_r[2 * j + 1];
        let dl_lo = den_l[2 * j];
        let dl_hi = den_l[2 * j + 1];
        let dr_lo = den_r[2 * j];
        let dr_hi = den_r[2 * j + 1];
        let eq_j = eq[j];

        c0_num += (nl_lo * dr_lo + nr_lo * dl_lo) * eq_j;
        c2_num += ((nl_hi - nl_lo) * (dr_hi - dr_lo) + (nr_hi - nr_lo) * (dl_hi - dl_lo)) * eq_j;
        c0_den += (dl_lo * dr_lo) * eq_j;
        c2_den += ((dl_hi - dl_lo) * (dr_hi - dr_lo)) * eq_j;
    }

    (c0_num, c2_num, c0_den, c2_den)
}

#[test]
fn test_gkr_quotient_sumcheck_small() {
    let ctx = cudarc::driver::safe::CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let gpu = GpuSumcheck::new(stream.clone());

    let mut rng = StdRng::seed_from_u64(42);

    for log_half in 1..=12 {
        let half = 1usize << log_half;
        let n = 2 * half;

        let num_l: Vec<EF> = (0..n).map(|_| rng.random()).collect();
        let num_r: Vec<EF> = (0..n).map(|_| rng.random()).collect();
        let den_l: Vec<EF> = (0..n).map(|_| rng.random()).collect();
        let den_r: Vec<EF> = (0..n).map(|_| rng.random()).collect();
        let eq: Vec<EF> = (0..half).map(|_| rng.random()).collect();

        // CPU reference
        let (cpu_c0n, cpu_c2n, cpu_c0d, cpu_c2d) =
            cpu_quotient_sumcheck(&num_l, &num_r, &den_l, &den_r, &eq, half);

        // GPU
        let flatten = |v: &[EF]| -> Vec<u32> {
            unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u32>(), v.len() * 5) }.to_vec()
        };
        let d_nl = stream.memcpy_stod(&flatten(&num_l)).unwrap();
        let d_nr = stream.memcpy_stod(&flatten(&num_r)).unwrap();
        let d_dl = stream.memcpy_stod(&flatten(&den_l)).unwrap();
        let d_dr = stream.memcpy_stod(&flatten(&den_r)).unwrap();
        let d_eq = stream.memcpy_stod(&flatten(&eq)).unwrap();

        let (gpu_c0n, gpu_c2n, gpu_c0d, gpu_c2d) =
            gpu.gkr_quotient_sumcheck_device(&d_nl, &d_nr, &d_dl, &d_dr, &d_eq, half as u32, 0);

        assert_eq!(
            ef_from_u32(&gpu_c0n),
            cpu_c0n,
            "c0_num mismatch at log_half={log_half}"
        );
        assert_eq!(
            ef_from_u32(&gpu_c2n),
            cpu_c2n,
            "c2_num mismatch at log_half={log_half}"
        );
        assert_eq!(
            ef_from_u32(&gpu_c0d),
            cpu_c0d,
            "c0_den mismatch at log_half={log_half}"
        );
        assert_eq!(
            ef_from_u32(&gpu_c2d),
            cpu_c2d,
            "c2_den mismatch at log_half={log_half}"
        );

        println!("log_half={log_half}: OK");
    }
}
