//! Playback-deck stages that need consecutive frames.

use crate::simd::multiversion;

/// Recursive (temporal) noise reduction as in many VCRs and capture
/// devices: each output frame is blended with the previous output where
/// the picture is still.
///
/// The frame difference against the previous output is split into luma and
/// colour difference (B − Y, R − Y).  Each difference is smoothed by a 3×3
/// binomial lowpass, so noise largely cancels while the difference left by
/// a moving edge or thin line keeps its height, then rectified and spread
/// to the strongest value within ±2 pixels and ±1 row, as motion detectors
/// widen their motion signal so the whole of a moving edge counts.  Below
/// the lower threshold the previous output is mixed in with weight `k_max`;
/// the weight falls to zero at the upper threshold.  Luma is blended where
/// luma is still, colour where both luma and colour are still.  Stronger
/// settings raise the weight and the thresholds, which removes more noise
/// but lets faint moving detail leave short soft trails.
pub struct TemporalDnr {
    k_max: f32,
    /// Luma motion thresholds in 8-bit levels: fully still below `t0`,
    /// fully moving at `t1`.
    t0: f32,
    t1: f32,
    /// The previous output as Y, B − Y, R − Y planes.
    prev: Option<[Vec<f32>; 3]>,
    w: usize,
    h: usize,
}

/// Colour-difference thresholds relative to the luma ones: colour-under
/// noise is stronger and coarser than luma noise.
const CHROMA_THRESHOLD_SCALE: f32 = 2.0;

impl TemporalDnr {
    pub fn new(strength: f32) -> Self {
        let s = strength.clamp(0.0, 1.0);
        let t0 = 3.0 + 6.0 * s;
        Self { k_max: 0.45 + 0.45 * s, t0, t1: 2.0 * t0, prev: None, w: 0, h: 0 }
    }

    /// Filter one RGB24 frame in place.  Frames must arrive in order.
    pub fn process(&mut self, rgb: &mut [u8], w: usize, h: usize) {
        let n = w * h;
        let mut cur = [vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]];
        {
            let [y, u, v] = &mut cur;
            rgb_to_ycc(rgb, y, u, v);
        }
        let Some(prev) = self.prev.as_ref().filter(|_| self.w == w && self.h == h) else {
            self.prev = Some(cur);
            self.w = w;
            self.h = h;
            return;
        };
        let mut diff = vec![0.0f32; n];
        let mut tmp = vec![0.0f32; n];
        let mut luma_motion = vec![0.0f32; n];
        let mut chroma_motion = vec![0.0f32; n];
        let mut chroma_v = vec![0.0f32; n];

        subtract(&cur[0], &prev[0], &mut diff);
        lowpass_abs(&diff, w, h, &mut tmp, &mut luma_motion);
        subtract(&cur[1], &prev[1], &mut diff);
        lowpass_abs(&diff, w, h, &mut tmp, &mut chroma_motion);
        subtract(&cur[2], &prev[2], &mut diff);
        lowpass_abs(&diff, w, h, &mut tmp, &mut chroma_v);
        max_in_place(&mut chroma_motion, &chroma_v);
        spread_max(&mut luma_motion, w, h, &mut tmp);
        spread_max(&mut chroma_motion, w, h, &mut tmp);

        let (t0, t1, k_max) = (self.t0, self.t1, self.k_max);
        let mut out = [vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]];
        blend_planes(&cur, prev, &luma_motion, &chroma_motion, t0, t1, k_max, &mut out);
        {
            let [y, u, v] = &out;
            ycc_to_rgb(y, u, v, rgb);
        }
        self.prev = Some(out);
    }
}

multiversion! {
    fn rgb_to_ycc(rgb: &[u8], y: &mut [f32], u: &mut [f32], v: &mut [f32]) {
        for i in 0..y.len() {
            let (r, g, b) = (rgb[i * 3] as f32, rgb[i * 3 + 1] as f32, rgb[i * 3 + 2] as f32);
            let l = 0.299 * r + 0.587 * g + 0.114 * b;
            y[i] = l;
            u[i] = b - l;
            v[i] = r - l;
        }
    }
}

multiversion! {
    fn ycc_to_rgb(y: &[f32], u: &[f32], v: &[f32], rgb: &mut [u8]) {
        for i in 0..y.len() {
            let r = y[i] + v[i];
            let b = y[i] + u[i];
            let g = (y[i] - 0.299 * r - 0.114 * b) / 0.587;
            rgb[i * 3] = r.round().clamp(0.0, 255.0) as u8;
            rgb[i * 3 + 1] = g.round().clamp(0.0, 255.0) as u8;
            rgb[i * 3 + 2] = b.round().clamp(0.0, 255.0) as u8;
        }
    }
}

multiversion! {
    fn subtract(a: &[f32], b: &[f32], out: &mut [f32]) {
        for i in 0..out.len() {
            out[i] = a[i] - b[i];
        }
    }
}

multiversion! {
    fn max_in_place(a: &mut [f32], b: &[f32]) {
        for i in 0..a.len() {
            a[i] = a[i].max(b[i]);
        }
    }
}

multiversion! {
    /// `out` = |3×3 binomial lowpass of `d`| (edges repeat the border
    /// pixel); `tmp` is scratch of the same size.
    fn lowpass_abs(d: &[f32], w: usize, h: usize, tmp: &mut [f32], out: &mut [f32]) {
        for y in 0..h {
            let row = &d[y * w..(y + 1) * w];
            let t = &mut tmp[y * w..(y + 1) * w];
            for x in 0..w {
                let l = row[x.saturating_sub(1)];
                let r = row[(x + 1).min(w - 1)];
                t[x] = 0.25 * l + 0.5 * row[x] + 0.25 * r;
            }
        }
        for y in 0..h {
            let up = &tmp[y.saturating_sub(1) * w..][..w];
            let mid = &tmp[y * w..][..w];
            let down = &tmp[(y + 1).min(h - 1) * w..][..w];
            let o = &mut out[y * w..(y + 1) * w];
            for x in 0..w {
                o[x] = (0.25 * up[x] + 0.5 * mid[x] + 0.25 * down[x]).abs();
            }
        }
    }
}

multiversion! {
    /// Replace each value of `m` by the largest within ±2 pixels and ±1
    /// row; `tmp` is scratch of the same size.
    fn spread_max(m: &mut [f32], w: usize, h: usize, tmp: &mut [f32]) {
        for y in 0..h {
            let row = &m[y * w..(y + 1) * w];
            let t = &mut tmp[y * w..(y + 1) * w];
            for x in 0..w {
                let a = x.saturating_sub(2);
                let b = (x + 3).min(w);
                let mut v = row[a];
                for &s in &row[a + 1..b] {
                    v = v.max(s);
                }
                t[x] = v;
            }
        }
        for y in 0..h {
            let up = &tmp[y.saturating_sub(1) * w..][..w];
            let mid = &tmp[y * w..][..w];
            let down = &tmp[(y + 1).min(h - 1) * w..][..w];
            let o = &mut m[y * w..(y + 1) * w];
            for x in 0..w {
                o[x] = up[x].max(mid[x]).max(down[x]);
            }
        }
    }
}

multiversion! {
    /// Blend the previous output into the current frame by stillness: luma
    /// by the luma motion, colour by the larger of the luma motion and the
    /// colour motion (whose thresholds are `CHROMA_THRESHOLD_SCALE` higher).
    #[allow(clippy::too_many_arguments)]
    fn blend_planes(
        cur: &[Vec<f32>; 3],
        prev: &[Vec<f32>; 3],
        luma_motion: &[f32],
        chroma_motion: &[f32],
        t0: f32,
        t1: f32,
        k_max: f32,
        out: &mut [Vec<f32>; 3],
    ) {
        let span = t1 - t0;
        let (c0, cspan) = (t0 * CHROMA_THRESHOLD_SCALE, span * CHROMA_THRESHOLD_SCALE);
        let [oy, ou, ov] = out;
        for i in 0..oy.len() {
            let still_y = ((t1 - luma_motion[i]) / span).clamp(0.0, 1.0);
            let still_c = ((c0 + cspan - chroma_motion[i]) / cspan).clamp(0.0, 1.0);
            let ky = k_max * still_y;
            let kc = k_max * still_y.min(still_c);
            oy[i] = cur[0][i] + ky * (prev[0][i] - cur[0][i]);
            ou[i] = cur[1][i] + kc * (prev[1][i] - cur[1][i]);
            ov[i] = cur[2][i] + kc * (prev[2][i] - cur[2][i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduces_noise_on_still_frames_and_lets_motion_through() {
        let (w, h) = (32, 32);
        let mut dnr = TemporalDnr::new(0.8);
        let mut seed = 1u32;
        let mut noise = || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            ((seed >> 16) % 21) as i32 - 10
        };
        let mut last = Vec::new();
        for _ in 0..20 {
            let mut f: Vec<u8> = (0..w * h * 3).map(|_| (128 + noise()) as u8).collect();
            dnr.process(&mut f, w, h);
            last = f;
        }
        let var = |f: &[u8]| f.iter().map(|&v| (v as f32 - 128.0).powi(2)).sum::<f32>() / f.len() as f32;
        assert!(var(&last) < 0.4 * 36.0, "noise variance {}", var(&last));
        // A big change passes straight through.
        let mut white = vec![250u8; w * h * 3];
        dnr.process(&mut white, w, h);
        assert!(white.iter().all(|&v| v > 240));
    }

    /// A thin line (3 pixels, as wide as the tape's luma bandwidth lets
    /// it be) moving across a flat field leaves no copy where it was.
    #[test]
    fn moving_thin_line_leaves_no_echo() {
        let (w, h) = (64, 16);
        let mut dnr = TemporalDnr::new(1.0);
        let frame = |x0: usize| -> Vec<u8> {
            let mut f = vec![100u8; w * h * 3];
            for y in 0..h {
                let i = (y * w + x0 - 1) * 3;
                f[i..i + 9].fill(70);
            }
            f
        };
        for (k, x0) in [10usize, 20, 30, 40].into_iter().enumerate() {
            let mut f = frame(x0);
            dnr.process(&mut f, w, h);
            if k > 0 {
                // Where the line was one and two frames ago.
                for old in [x0 - 10, x0.saturating_sub(20)] {
                    for y in 0..h {
                        let v = f[(y * w + old) * 3];
                        assert!(v >= 99, "echo at x {old}: {v}");
                    }
                }
                // The line itself is not washed out.
                assert!(f[(8 * w + x0) * 3] <= 71);
            }
        }
    }
}
