//! Playback-deck stages that need consecutive frames.

/// Recursive (temporal) noise reduction as in many VCRs and capture
/// devices: each output frame is blended with the previous output where
/// the picture is still.
///
/// Per pixel, the luma difference between the new frame and the previous
/// output (averaged over a 3×3 neighbourhood so noise alone does not count
/// as motion) decides the blend: below the threshold the previous output
/// is mixed in with weight up to `k_max`, above it the new frame passes
/// through.  Stronger settings use a higher threshold and weight, which
/// removes more noise but lets moving objects drag smeared ghost trails,
/// the well-known side effect of these circuits.
pub struct TemporalDnr {
    k_max: f32,
    threshold: f32,
    prev: Option<Vec<f32>>,
    w: usize,
    h: usize,
}

impl TemporalDnr {
    pub fn new(strength: f32) -> Self {
        let s = strength.clamp(0.0, 1.0);
        Self { k_max: 0.45 + 0.45 * s, threshold: 6.0 + 30.0 * s, prev: None, w: 0, h: 0 }
    }

    /// Filter one RGB24 frame in place.  Frames must arrive in order.
    pub fn process(&mut self, rgb: &mut [u8], w: usize, h: usize) {
        let cur: Vec<f32> = rgb.iter().map(|&v| v as f32).collect();
        let Some(prev) = self.prev.as_ref().filter(|_| self.w == w && self.h == h) else {
            self.prev = Some(cur);
            self.w = w;
            self.h = h;
            return;
        };
        // Luma difference per pixel.
        let mut d = vec![0.0f32; w * h];
        for i in 0..w * h {
            let l = |p: &[f32]| 0.299 * p[i * 3] + 0.587 * p[i * 3 + 1] + 0.114 * p[i * 3 + 2];
            d[i] = (l(&cur) - l(prev)).abs();
        }
        let mut out = vec![0.0f32; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let mut m = 0.0;
                let mut n = 0.0;
                for yy in y.saturating_sub(1)..(y + 2).min(h) {
                    for xx in x.saturating_sub(1)..(x + 2).min(w) {
                        m += d[yy * w + xx];
                        n += 1.0;
                    }
                }
                let still = (1.0 - (m / n) / self.threshold).clamp(0.0, 1.0);
                let k = self.k_max * still;
                let i = (y * w + x) * 3;
                for c in 0..3 {
                    out[i + c] = cur[i + c] + k * (prev[i + c] - cur[i + c]);
                }
            }
        }
        for (dst, v) in rgb.iter_mut().zip(&out) {
            *dst = v.round().clamp(0.0, 255.0) as u8;
        }
        self.prev = Some(out);
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
}
