//! IIR filter sections and fast Gaussian noise for the per-line VHS
//! stages.  The sections model the analog circuits of a VCR (emphasis
//! networks, lowpass filters), which are causal; zero-phase use runs a
//! section forwards and then backwards.

use crate::constants::SAMPLE_RATE;
use crate::simd::multiversion;

/// Lines filtered side by side by the `*_lanes` functions.
pub const LANES: usize = 8;

/// A second-order IIR section, run in transposed direct form II.  First
/// order sections have `b2 = a2 = 0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Biquad {
    pub b0: f32,
    pub b1: f32,
    pub b2: f32,
    pub a1: f32,
    pub a2: f32,
}

impl Biquad {
    /// Lowpass with corner `fc` and quality `q` (bilinear transform with
    /// prewarping).  `q = 0.7071` is a 2nd-order Butterworth section.
    pub fn lowpass(fc: f64, q: f64) -> Self {
        let w = 2.0 * std::f64::consts::PI * fc / SAMPLE_RATE;
        let (sn, cs) = w.sin_cos();
        let alpha = sn / (2.0 * q);
        let a0 = 1.0 + alpha;
        Self::from_f64((1.0 - cs) / 2.0 / a0, (1.0 - cs) / a0, (1.0 - cs) / 2.0 / a0, -2.0 * cs / a0, (1.0 - alpha) / a0)
    }

    /// `lowpass` at sample rate `fs`.
    pub fn lowpass_at(fc: f64, q: f64, fs: f64) -> Self {
        let w = 2.0 * std::f64::consts::PI * fc / fs;
        let (sn, cs) = w.sin_cos();
        let alpha = sn / (2.0 * q);
        let a0 = 1.0 + alpha;
        Self::from_f64((1.0 - cs) / 2.0 / a0, (1.0 - cs) / a0, (1.0 - cs) / 2.0 / a0, -2.0 * cs / a0, (1.0 - alpha) / a0)
    }

    /// One-pole highpass with its −3 dB point at `fc`, at sample rate `fs`.
    pub fn highpass1_at(fc: f64, fs: f64) -> Self {
        let k = (std::f64::consts::PI * fc / fs).tan();
        Self::from_f64(1.0 / (1.0 + k), -1.0 / (1.0 + k), 0.0, (k - 1.0) / (k + 1.0), 0.0)
    }

    /// One-pole lowpass with its −3 dB point at `fc` (bilinear, prewarped).
    pub fn lowpass1(fc: f64) -> Self {
        let k = (std::f64::consts::PI * fc / SAMPLE_RATE).tan();
        Self::from_f64(k / (1.0 + k), k / (1.0 + k), 0.0, (k - 1.0) / (k + 1.0), 0.0)
    }

    /// One-pole highpass with its −3 dB point at `fc`.
    pub fn highpass1(fc: f64) -> Self {
        let k = (std::f64::consts::PI * fc / SAMPLE_RATE).tan();
        Self::from_f64(1.0 / (1.0 + k), -1.0 / (1.0 + k), 0.0, (k - 1.0) / (k + 1.0), 0.0)
    }

    /// The first-order analog section H(s) = (1 + s·tz) / (1 + s·tp), time
    /// constants in seconds, by the bilinear transform.  Its corners sit far
    /// below the sample rate, so no prewarping is needed.
    pub fn shelf1(tz: f64, tp: f64) -> Self {
        let c = 2.0 * SAMPLE_RATE;
        let a0 = 1.0 + c * tp;
        Self::from_f64((1.0 + c * tz) / a0, (1.0 - c * tz) / a0, 0.0, (1.0 - c * tp) / a0, 0.0)
    }

    fn from_f64(b0: f64, b1: f64, b2: f64, a1: f64, a2: f64) -> Self {
        Self { b0: b0 as f32, b1: b1 as f32, b2: b2 as f32, a1: a1 as f32, a2: a2 as f32 }
    }

    /// Gain at DC.
    pub fn dc_gain(&self) -> f32 {
        (self.b0 + self.b1 + self.b2) / (1.0 + self.a1 + self.a2)
    }

    /// Filter states for a signal that has been at `x` forever.
    fn steady(&self, x: f32) -> (f32, f32) {
        let y = self.dc_gain() * x;
        let s2 = self.b2 * x - self.a2 * y;
        (y - self.b0 * x, s2)
    }

    /// Filter `buf` in place, forwards, as if it had been at `buf[0]`
    /// before the start.
    pub fn run(&self, buf: &mut [f32]) {
        let Some(&first) = buf.first() else { return };
        let (mut s1, mut s2) = self.steady(first);
        for v in buf.iter_mut() {
            let x = *v;
            let y = self.b0 * x + s1;
            s1 = self.b1 * x - self.a1 * y + s2;
            s2 = self.b2 * x - self.a2 * y;
            *v = y;
        }
    }

    /// Filter `buf` in place, backwards, as if it stayed at its last value
    /// after the end.
    pub fn run_back(&self, buf: &mut [f32]) {
        let Some(&last) = buf.last() else { return };
        let (mut s1, mut s2) = self.steady(last);
        for v in buf.iter_mut().rev() {
            let x = *v;
            let y = self.b0 * x + s1;
            s1 = self.b1 * x - self.a1 * y + s2;
            s2 = self.b2 * x - self.a2 * y;
            *v = y;
        }
    }

    /// `run` on `LANES` interleaved signals: `buf[j * LANES + l]` is sample
    /// `j` of signal `l`.
    pub fn run_lanes(&self, buf: &mut [f32]) {
        biquad_lanes(self, buf, false);
    }

    /// `run_back` on `LANES` interleaved signals.
    pub fn run_back_lanes(&self, buf: &mut [f32]) {
        biquad_lanes(self, buf, true);
    }

    /// Impulse response, `n` samples.
    pub fn impulse(sections: &[Biquad], n: usize) -> Vec<f32> {
        let mut h = vec![0.0f32; n];
        h[0] = 1.0;
        // Run from rest (the steady state of a zero signal).
        for s in sections {
            let (mut s1, mut s2) = (0.0f32, 0.0f32);
            for v in h.iter_mut() {
                let x = *v;
                let y = s.b0 * x + s1;
                s1 = s.b1 * x - s.a1 * y + s2;
                s2 = s.b2 * x - s.a2 * y;
                *v = y;
            }
        }
        h
    }
}

multiversion! {
    fn biquad_lanes(f: &Biquad, buf: &mut [f32], back: bool) {
        const L: usize = LANES;
        let n = buf.len() / L;
        if n == 0 {
            return;
        }
        let edge = if back { n - 1 } else { 0 };
        let mut s1 = [0.0f32; L];
        let mut s2 = [0.0f32; L];
        for l in 0..L {
            let (a, b) = f.steady(buf[edge * L + l]);
            s1[l] = a;
            s2[l] = b;
        }
        let (b0, b1, b2, a1, a2) = (f.b0, f.b1, f.b2, f.a1, f.a2);
        for k in 0..n {
            let j = if back { n - 1 - k } else { k };
            let chunk: &mut [f32; L] = (&mut buf[j * L..j * L + L]).try_into().unwrap();
            for l in 0..L {
                let x = chunk[l];
                let y = b0 * x + s1[l];
                s1[l] = b1 * x - a1 * y + s2[l];
                s2[l] = b2 * x - a2 * y;
                chunk[l] = y;
            }
        }
    }
}

/// Run `sections` forwards over `LANES` interleaved signals.
pub fn causal_lanes(sections: &[Biquad], buf: &mut [f32]) {
    for s in sections {
        s.run_lanes(buf);
    }
}

/// Zero-phase filtering of `LANES` interleaved signals: `sections` run
/// forwards, then backwards (the magnitude response is squared).
pub fn zero_phase_lanes(sections: &[Biquad], buf: &mut [f32]) {
    for s in sections {
        s.run_lanes(buf);
    }
    for s in sections {
        s.run_back_lanes(buf);
    }
}

/// Copy rows `rows[l]` (each `len` samples from `src`) into the lane
/// layout `dst[j * LANES + l]`.  Missing rows (past `rows.len()`) repeat
/// the last one.
pub fn to_lanes(rows: &[&[f32]], len: usize, dst: &mut Vec<f32>) {
    dst.resize(len * LANES, 0.0);
    for l in 0..LANES {
        let r = rows[l.min(rows.len() - 1)];
        for j in 0..len {
            dst[j * LANES + l] = r[j];
        }
    }
}

/// Group delay at DC, in samples, of `sections` run forwards: the
/// centroid of the impulse response.
pub fn dc_group_delay(sections: &[Biquad]) -> f32 {
    let h = Biquad::impulse(sections, 4096);
    let s: f64 = h.iter().map(|&v| v as f64).sum();
    let m: f64 = h.iter().enumerate().map(|(i, &v)| i as f64 * v as f64).sum();
    (m / s) as f32
}

// ────────────────────────────────────────────────────────────────────────────
// Fast Gaussian noise
// ────────────────────────────────────────────────────────────────────────────

/// Fast Gaussian noise: a xorshift generator indexing a table of normal
/// deviates.  The sum of two table entries, scaled by 1/√2, has unit
/// variance and a smooth enough distribution for picture noise.
pub struct FastNoise {
    state: u64,
}

const TABLE_BITS: u32 = 12;

fn normal_table() -> &'static [f32] {
    static TABLE: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        // Inverse CDF at the centres of equal-probability bins.
        let n = 1usize << TABLE_BITS;
        let mut t: Vec<f32> = (0..n).map(|i| inverse_normal_cdf((i as f64 + 0.5) / n as f64) as f32).collect();
        let var: f64 = t.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / n as f64;
        let k = (1.0 / var.sqrt()) as f32;
        for v in t.iter_mut() {
            *v *= k;
        }
        t
    })
}

/// Acklam's rational approximation of the inverse normal CDF.
fn inverse_normal_cdf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.383577518672690e2, -3.066479806614716e1,
        2.506628277459239,
    ];
    const B: [f64; 5] =
        [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
    const C: [f64; 6] = [
        -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968,
        2.938163982698783,
    ];
    const D: [f64; 4] = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
    let pl = 0.02425;
    if p < pl {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p > 1.0 - pl {
        -inverse_normal_cdf(1.0 - p)
    } else {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    }
}

impl FastNoise {
    pub fn new(seed: u64) -> Self {
        Self { state: crate::effects::splitmix64(seed) | 1 }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// A uniform value in [0, 1).
    #[inline]
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// A unit-variance Gaussian value.
    #[inline]
    pub fn normal(&mut self) -> f32 {
        let t = normal_table();
        let r = self.next_u64();
        let a = t[(r >> (64 - TABLE_BITS)) as usize];
        let b = t[((r >> 20) & ((1 << TABLE_BITS) - 1)) as usize];
        (a + b) * std::f32::consts::FRAC_1_SQRT_2
    }

    /// Fill `out` with unit-variance Gaussian noise.
    pub fn fill(&mut self, out: &mut [f32]) {
        let t = normal_table();
        let mut x = self.state;
        for v in out.iter_mut() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let a = t[(x >> (64 - TABLE_BITS)) as usize];
            let b = t[((x >> 20) & ((1 << TABLE_BITS) - 1)) as usize];
            *v = (a + b) * std::f32::consts::FRAC_1_SQRT_2;
        }
        self.state = x;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lanes_match_single_runs() {
        let f = Biquad::lowpass(2.0e6, 0.7071);
        let rows: Vec<Vec<f32>> = (0..LANES).map(|l| (0..300).map(|j| ((j * (l + 3)) % 17) as f32).collect()).collect();
        let refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
        let mut lanes = Vec::new();
        to_lanes(&refs, 300, &mut lanes);
        f.run_lanes(&mut lanes);
        f.run_back_lanes(&mut lanes);
        for (l, r) in rows.iter().enumerate() {
            let mut one = r.clone();
            f.run(&mut one);
            f.run_back(&mut one);
            for j in 0..300 {
                assert_eq!(one[j], lanes[j * LANES + l]);
            }
        }
    }

    #[test]
    fn sections_have_expected_dc_gain() {
        assert!((Biquad::lowpass(3e6, 0.7).dc_gain() - 1.0).abs() < 1e-5);
        assert!(Biquad::highpass1(1e6).dc_gain().abs() < 1e-6);
        assert!((Biquad::shelf1(1.3e-6, 0.26e-6).dc_gain() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn fast_noise_is_unit_normal() {
        let mut n = FastNoise::new(3);
        let mut v = vec![0.0f32; 200_000];
        n.fill(&mut v);
        let m = v.iter().sum::<f32>() / v.len() as f32;
        let var = v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / v.len() as f32;
        assert!(m.abs() < 0.01 && (var - 1.0).abs() < 0.02, "{m} {var}");
    }
}
