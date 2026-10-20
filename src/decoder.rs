//! NTSC composite video decoder: composite signal -> RGB frames.

use realfft::RealFftPlanner;

use crate::constants::*;
use crate::filters::{self, FilterKernel, FilterScratch};
use crate::profile;
use crate::simd::multiversion;

const RIGHT_PAD: usize = NUM_TAPS;

/// Precomputed data for the decoder.
/// Each thread should have its own Decoder to avoid contention on scratch buffers.
pub struct Decoder {
    /// Filter for full-line luma (910 samples)
    luma_filter: FilterKernel,
    /// Filters for I/Q (910 + RIGHT_PAD samples)
    i_filter: FilterKernel,
    q_filter: FilterKernel,
    /// Absolute line numbers for visible lines
    abs_lines: [usize; VISIBLE_LINES],
    /// Reference lines for 1H comb
    ref_lines: [usize; VISIBLE_LINES],
    /// FFT scratch for luma (sized for SAMPLES_PER_LINE FFT)
    luma_scratch: FilterScratch,
    /// FFT scratch for I/Q (sized for SAMPLES_PER_LINE + RIGHT_PAD FFT)
    iq_scratch: FilterScratch,
    // --- Pre-allocated working buffers ---
    full_lines: Vec<f32>,
    y_full: Vec<f32>,
    chroma_full: Vec<f32>,
    burst_phases: Vec<f32>,
    i_raw: Vec<f32>,
    q_raw: Vec<f32>,
    output: Vec<u8>,
    /// Cached output dimensions for buffer invalidation
    cached_out_dims: (usize, usize),
    /// Capture-chip luma peaking: gain at the centre frequency (0 = off,
    /// 1 = +6 dB) and centre frequency in Hz.
    pub peaking_gain: f32,
    pub peaking_hz: f64,
}

impl Decoder {
    pub fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();

        let fir_y = filters::design_lowpass(LUMA_BW, NUM_TAPS);
        let fir_i = filters::design_lowpass(I_BW, NUM_TAPS);
        let fir_q = filters::design_lowpass(Q_BW, NUM_TAPS);

        let iq_padded_len = SAMPLES_PER_LINE + RIGHT_PAD;

        let luma_filter = FilterKernel::new(&fir_y, SAMPLES_PER_LINE + RIGHT_PAD);
        let i_filter = FilterKernel::new(&fir_i, iq_padded_len);

        let luma_fft_n = luma_filter.fft_n();
        let iq_fft_n = i_filter.fft_n();

        Self {
            luma_filter,
            i_filter,
            q_filter: FilterKernel::new(&fir_q, iq_padded_len),
            abs_lines: build_abs_lines(),
            ref_lines: build_ref_lines(),
            luma_scratch: FilterScratch::with_planner(&mut planner, luma_fft_n),
            iq_scratch: FilterScratch::with_planner(&mut planner, iq_fft_n),
            full_lines: vec![0.0f32; VISIBLE_LINES * SAMPLES_PER_LINE],
            y_full: vec![0.0f32; VISIBLE_LINES * (SAMPLES_PER_LINE + RIGHT_PAD)],
            chroma_full: vec![0.0f32; VISIBLE_LINES * SAMPLES_PER_LINE],
            burst_phases: vec![0.0f32; VISIBLE_LINES],
            i_raw: vec![0.0f32; VISIBLE_LINES * iq_padded_len],
            q_raw: vec![0.0f32; VISIBLE_LINES * iq_padded_len],
            output: Vec::new(),
            cached_out_dims: (0, 0),
            peaking_gain: 0.0,
            peaking_hz: 2.6e6,
        }
    }

    /// Decode a composite NTSC signal back to an RGB frame.
    /// Returns a reference to the internal output buffer (valid until the next call).
    ///
    /// `legacy_burst_phase` uses the inverted burst phase estimate of older
    /// builds, which doubles any burst phase error instead of cancelling it.
    pub fn decode_frame(
        &mut self,
        signal: &[f32],
        output_width: usize,
        output_height: usize,
        comb_1h: bool,
        legacy_burst_phase: bool,
    ) -> &[u8] {
        let _p = profile::scope("decode comb");
        let spl = SAMPLES_PER_LINE;
        let ypl = spl + RIGHT_PAD;

        self.prepare_output(output_width, output_height);

        // Extract full lines for all visible lines (480 x 910)
        for vis in 0..VISIBLE_LINES {
            let abs = self.abs_lines[vis];
            let sig_start = abs * spl;
            let dst_start = vis * spl;
            self.full_lines[dst_start..dst_start + spl]
                .copy_from_slice(&signal[sig_start..sig_start + spl]);
        }

        // Comb filter for luma/chroma separation
        if comb_1h {
            for vis in 0..VISIBLE_LINES {
                let ref_line = self.ref_lines[vis];
                let ref_start = ref_line * spl;
                let cur_start = vis * spl;
                let y_start = vis * ypl;
                for n in 0..spl {
                    let cur = self.full_lines[cur_start + n];
                    let ref_val = signal[ref_start + n];
                    self.y_full[y_start + n] = (cur + ref_val) * 0.5;
                    self.chroma_full[cur_start + n] = (cur - ref_val) * 0.5;
                }
            }
        } else {
            for vis in 0..VISIBLE_LINES {
                let base = vis * spl;
                let ybase = vis * ypl;
                self.y_full[ybase] = self.full_lines[base] * 0.5;
                self.y_full[ybase + 1] = self.full_lines[base + 1] * 0.5;
                self.chroma_full[base] = self.full_lines[base] * 0.5;
                self.chroma_full[base + 1] = self.full_lines[base + 1] * 0.5;

                for n in 2..spl {
                    let cur = self.full_lines[base + n];
                    let delayed = self.full_lines[base + n - 2];
                    self.y_full[ybase + n] = (cur + delayed) * 0.5;
                    self.chroma_full[base + n] = (cur - delayed) * 0.5;
                }
            }
        }

        drop(_p);
        self.finish(output_width, output_height, legacy_burst_phase)
    }

    fn prepare_output(&mut self, output_width: usize, output_height: usize) {
        let out_size = output_width * output_height * 3;
        if self.cached_out_dims != (output_width, output_height) {
            self.output.resize(out_size, 0);
            self.cached_out_dims = (output_width, output_height);
        }
    }

    /// Decode with a motion-adaptive 3D (frame) comb filter.
    ///
    /// The NTSC subcarrier inverts from one frame to the next, so where the
    /// picture is still, half the sum of the current and previous frame
    /// (`prev`) is pure luma and half the difference pure chroma: no
    /// cross-colour and no dot crawl.  Motion is detected, as in real 3D
    /// combs, against the frame two back (`prev2`), which has the same
    /// subcarrier phase: in still areas the two are identical whatever the
    /// picture contains, while motion, noise and VHS chroma that changes
    /// from frame to frame all show up.  The difference is smoothed over
    /// ±4 samples; the blend is fully 3D below about 1 IRE and fully
    /// 2-sample comb above 3 IRE (conservative, so frame-to-frame chroma
    /// noise does not leak into luma as dots).
    ///
    /// With `luma_motion`, motion is also detected against the previous
    /// frame itself, from the luma part of the difference: a 4-sample
    /// average (one subcarrier cycle) cancels the chroma, whose phase
    /// inverts between the two frames.  The frame two back alone misses
    /// whatever is only in the previous frame, such as a thin line that
    /// has moved on, which the comb would otherwise copy into this frame.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_frame_3d(
        &mut self,
        signal: &[f32],
        prev: &[f32],
        prev2: &[f32],
        output_width: usize,
        output_height: usize,
        legacy_burst_phase: bool,
        luma_motion: bool,
    ) -> &[u8] {
        let p = profile::scope("decode comb");
        let spl = SAMPLES_PER_LINE;
        let ypl = spl + RIGHT_PAD;
        self.prepare_output(output_width, output_height);
        let ire = BLANKING_V / 40.0;
        let (lo, hi) = (1.0 * ire, 3.0 * ire);
        let mut diff = vec![0.0f32; spl];
        let mut diff1 = vec![0.0f32; spl];
        for vis in 0..VISIBLE_LINES {
            let abs = self.abs_lines[vis];
            let s = abs * spl;
            let cur = &signal[s..s + spl];
            let old = &prev[s..s + spl];
            let old2 = &prev2[s..s + spl];
            let base = vis * spl;
            let ybase = vis * ypl;
            self.full_lines[base..base + spl].copy_from_slice(cur);
            for n in 0..spl {
                diff[n] = (cur[n] - old2[n]).abs();
            }
            if luma_motion {
                for n in 0..spl {
                    let a = n.saturating_sub(2).min(spl - 4);
                    let d: f32 = (a..a + 4).map(|k| cur[k] - old[k]).sum();
                    diff1[n] = (0.25 * d).abs();
                }
            }
            for n in 0..spl {
                let a = n.saturating_sub(4);
                let b = (n + 5).min(spl);
                let mut m = diff[a..b].iter().sum::<f32>() / (b - a) as f32;
                if luma_motion {
                    m = m.max(diff1[a..b].iter().sum::<f32>() / (b - a) as f32);
                }
                let still = ((hi - m) / (hi - lo)).clamp(0.0, 1.0);
                let (y2, c2) = if n >= 2 {
                    ((cur[n] + cur[n - 2]) * 0.5, (cur[n] - cur[n - 2]) * 0.5)
                } else {
                    (cur[n] * 0.5, cur[n] * 0.5)
                };
                let y3 = (cur[n] + old[n]) * 0.5;
                let c3 = (cur[n] - old[n]) * 0.5;
                self.y_full[ybase + n] = y2 + still * (y3 - y2);
                self.chroma_full[base + n] = c2 + still * (c3 - c2);
            }
        }
        drop(p);
        self.finish(output_width, output_height, legacy_burst_phase)
    }

    /// Decode with a chroma trap, as capture chips do for VCR input (the
    /// TVP5150 switches to it automatically when the timing is unstable).
    /// Chroma is the band fsc ± 0.6 MHz (−6 dB), luma is everything else,
    /// so luma keeps its detail up to about 2.9 MHz; fine luma inside the
    /// band shows as cross-colour.
    pub fn decode_frame_trap(
        &mut self,
        signal: &[f32],
        output_width: usize,
        output_height: usize,
        legacy_burst_phase: bool,
    ) -> &[u8] {
        let p = profile::scope("decode comb");
        let spl = SAMPLES_PER_LINE;
        let ypl = spl + RIGHT_PAD;
        self.prepare_output(output_width, output_height);
        let r = crate::vhs::REGION_START;
        let mut group = crate::vhs::LaneGroup::new();
        for (k, chunk) in self.abs_lines.chunks(crate::iir::LANES).enumerate() {
            group.load(signal, chunk, false);
            for (l, &abs) in chunk.iter().enumerate() {
                let vis = k * crate::iir::LANES + l;
                let (base, ybase) = (vis * spl, vis * ypl);
                self.full_lines[base..base + spl].copy_from_slice(&signal[abs * spl..(abs + 1) * spl]);
                self.y_full[ybase..ybase + r].copy_from_slice(&signal[abs * spl..abs * spl + r]);
                self.chroma_full[base..base + r].fill(0.0);
                for n in r..spl {
                    let at = (n - r) * crate::iir::LANES + l;
                    self.y_full[ybase + n] = group.luma[at];
                    self.chroma_full[base + n] = group.chroma[at];
                }
            }
        }
        drop(p);
        self.finish(output_width, output_height, legacy_burst_phase)
    }

    /// Decode an S-Video signal pair: luma from the Y wire with no comb
    /// filter, chroma from the C wire, so there is no cross-colour or
    /// cross-luma.
    pub fn decode_svideo(
        &mut self,
        y_signal: &[f32],
        c_signal: &[f32],
        output_width: usize,
        output_height: usize,
        legacy_burst_phase: bool,
    ) -> &[u8] {
        let p = profile::scope("decode comb");
        let spl = SAMPLES_PER_LINE;
        let ypl = spl + RIGHT_PAD;
        self.prepare_output(output_width, output_height);
        for vis in 0..VISIBLE_LINES {
            let abs = self.abs_lines[vis];
            let sig = abs * spl;
            let base = vis * spl;
            // full_lines holds the C wire (burst phase is measured on it).
            self.full_lines[base..base + spl].copy_from_slice(&c_signal[sig..sig + spl]);
            for n in 0..spl {
                self.chroma_full[base + n] = c_signal[sig + n] - BLANKING_V;
            }
            self.y_full[vis * ypl..vis * ypl + spl].copy_from_slice(&y_signal[sig..sig + spl]);
        }
        drop(p);
        self.finish(output_width, output_height, legacy_burst_phase)
    }

    /// Everything after luma/chroma separation: luma lowpass, burst phase,
    /// product detection, I/Q lowpass, YIQ to RGB.
    fn finish(&mut self, output_width: usize, output_height: usize, legacy_burst_phase: bool) -> &[u8] {
        let _p = profile::scope("decode filter/rgb");
        let spl = SAMPLES_PER_LINE;
        let ypl = spl + RIGHT_PAD;

        // Edge-pad luma rows on the right for filter settling
        for vis in 0..VISIBLE_LINES {
            let ybase = vis * ypl;
            let right_val = self.y_full[ybase + spl - 1];
            self.y_full[ybase + spl..ybase + ypl].fill(right_val);
        }

        // Lowpass luma at 4.2 MHz
        filters::filter_rows_sequential(
            &self.luma_filter,
            &mut self.y_full,
            ypl,
            &mut self.luma_scratch,
        );

        // Capture-chip peaking: the sample minus the mean of its two
        // neighbours half a period of the centre frequency away.
        if self.peaking_gain > 0.0 {
            let d = (SAMPLE_RATE / (2.0 * self.peaking_hz)).round().max(1.0) as usize;
            let g = 0.5 * self.peaking_gain;
            let mut row = vec![0.0f32; ypl];
            for vis in 0..VISIBLE_LINES {
                let y = &mut self.y_full[vis * ypl..(vis + 1) * ypl];
                row.copy_from_slice(y);
                for n in d..ypl - d {
                    y[n] = row[n] + g * (row[n] - 0.5 * (row[n - d] + row[n + d]));
                }
            }
        }

        // Undo composite voltage scaling on luma
        for v in &mut self.y_full {
            *v = (*v - COMPOSITE_OFFSET) / COMPOSITE_SCALE;
        }

        // Burst phase detection using carrier LUT
        for vis in 0..VISIBLE_LINES {
            let abs_line = self.abs_lines[vis];
            let line_phase = std::f32::consts::PI * abs_line as f32;
            let base = vis * spl;

            let phi = std::f32::consts::FRAC_PI_2 * BURST_START as f32 + line_phase;
            let cos_phi = phi.cos();
            let sin_phi = phi.sin();

            let cos_lut = [cos_phi, -sin_phi, -cos_phi, sin_phi];
            let sin_lut = [sin_phi, cos_phi, -sin_phi, -cos_phi];

            let mut cos_corr = 0.0f32;
            let mut sin_corr = 0.0f32;

            for n in 0..BURST_SAMPLES {
                let burst_val = self.full_lines[base + BURST_START + n];
                cos_corr += burst_val * cos_lut[n & 3];
                sin_corr += burst_val * sin_lut[n & 3];
            }

            // atan2 gives π − δ for a burst that leads by δ.
            let measured = sin_corr.atan2(cos_corr);
            self.burst_phases[vis] = if legacy_burst_phase {
                measured - std::f32::consts::PI
            } else {
                std::f32::consts::PI - measured
            };
        }

        // Zero out chroma in blanking region
        for vis in 0..VISIBLE_LINES {
            let base = vis * spl;
            self.chroma_full[base..base + ACTIVE_START].fill(0.0);
        }

        // Product detection using carrier LUT, then pad right with reflect
        let iq_padded_len = spl + RIGHT_PAD;

        for vis in 0..VISIBLE_LINES {
            let abs_line = self.abs_lines[vis];
            let line_phase = std::f32::consts::PI * abs_line as f32;
            let bp = self.burst_phases[vis];
            let chroma_base = vis * spl;
            let iq_base = vis * iq_padded_len;

            let i_phi = line_phase + bp + I_PHASE_RAD;
            let q_phi = line_phase + bp + Q_PHASE_RAD;

            let i_cos = i_phi.cos();
            let i_sin = i_phi.sin();
            let q_cos = q_phi.cos();
            let q_sin = q_phi.sin();

            let i_lut = [i_cos, -i_sin, -i_cos, i_sin];
            let q_lut = [q_cos, -q_sin, -q_cos, q_sin];

            for n in 0..spl {
                let chroma_val = self.chroma_full[chroma_base + n];
                self.i_raw[iq_base + n] = 2.0 * chroma_val * i_lut[n & 3];
                self.q_raw[iq_base + n] = 2.0 * chroma_val * q_lut[n & 3];
            }

            // Reflect padding on right
            for j in 0..RIGHT_PAD {
                self.i_raw[iq_base + spl + j] = self.i_raw[iq_base + spl - 2 - j];
                self.q_raw[iq_base + spl + j] = self.q_raw[iq_base + spl - 2 - j];
            }
        }

        // Low-pass filter I and Q
        filters::filter_rows_sequential(
            &self.i_filter,
            &mut self.i_raw,
            iq_padded_len,
            &mut self.iq_scratch,
        );
        filters::filter_rows_sequential(
            &self.q_filter,
            &mut self.q_raw,
            iq_padded_len,
            &mut self.iq_scratch,
        );

        // Crop to active region and convert YIQ -> RGB
        let need_resize = output_width != ACTIVE_SAMPLES || output_height != VISIBLE_LINES;

        if !need_resize {
            for vis in 0..VISIBLE_LINES {
                let y_base = vis * ypl + ACTIVE_START;
                let iq_base = vis * iq_padded_len + ACTIVE_START;
                let out_base = vis * ACTIVE_SAMPLES * 3;

                for n in 0..ACTIVE_SAMPLES {
                    let y = self.y_full[y_base + n];
                    let i = self.i_raw[iq_base + n] / COMPOSITE_SCALE;
                    let q = self.q_raw[iq_base + n] / COMPOSITE_SCALE;

                    let r = (YIQ_TO_RGB[0][0] * y + YIQ_TO_RGB[0][1] * i + YIQ_TO_RGB[0][2] * q).clamp(0.0, 1.0);
                    let g = (YIQ_TO_RGB[1][0] * y + YIQ_TO_RGB[1][1] * i + YIQ_TO_RGB[1][2] * q).clamp(0.0, 1.0);
                    let b = (YIQ_TO_RGB[2][0] * y + YIQ_TO_RGB[2][1] * i + YIQ_TO_RGB[2][2] * q).clamp(0.0, 1.0);

                    self.output[out_base + n * 3] = (r * 255.0 + 0.5) as u8;
                    self.output[out_base + n * 3 + 1] = (g * 255.0 + 0.5) as u8;
                    self.output[out_base + n * 3 + 2] = (b * 255.0 + 0.5) as u8;
                }
            }
        } else {
            // Bilinear resize with fused YIQ->RGB
            let x_scale = if output_width > 1 {
                (ACTIVE_SAMPLES - 1) as f32 / (output_width - 1) as f32
            } else {
                0.0
            };
            let y_scale = if output_height > 1 {
                (VISIBLE_LINES - 1) as f32 / (output_height - 1) as f32
            } else {
                0.0
            };

            let inv_comp_scale = 1.0 / COMPOSITE_SCALE;
            let x_tab: Vec<(usize, usize, f32)> = (0..output_width)
                .map(|dx| {
                    let sx = dx as f32 * x_scale;
                    let sx0 = (sx as usize).min(ACTIVE_SAMPLES - 1);
                    let sx1 = (sx0 + 1).min(ACTIVE_SAMPLES - 1);
                    (sx0, sx1, sx - sx0 as f32)
                })
                .collect();
            for dy in 0..output_height {
                let sy = dy as f32 * y_scale;
                let sy0 = (sy as usize).min(VISIBLE_LINES - 1);
                let sy1 = (sy0 + 1).min(VISIBLE_LINES - 1);
                let yt = sy - sy0 as f32;
                let out = &mut self.output[dy * output_width * 3..(dy + 1) * output_width * 3];
                resize_row_to_rgb(
                    row_pair(&self.y_full, ypl, sy0, sy1),
                    row_pair(&self.i_raw, iq_padded_len, sy0, sy1),
                    row_pair(&self.q_raw, iq_padded_len, sy0, sy1),
                    &x_tab,
                    yt,
                    inv_comp_scale,
                    out,
                );
            }
        }

        &self.output
    }
}

/// Output pixels gathered and converted together by `resize_row_to_rgb`.
const RESIZE_CHUNK: usize = 64;

type RowPair<'a> = (&'a [f32], &'a [f32]);

/// Rows `a` and `b` of a plane with rows of `len`, from the active region.
fn row_pair(p: &[f32], len: usize, a: usize, b: usize) -> RowPair<'_> {
    (&p[a * len + ACTIVE_START..(a + 1) * len], &p[b * len + ACTIVE_START..(b + 1) * len])
}

multiversion! {
    /// One output row of the bilinear resize with fused YIQ to RGB.  Each
    /// `(row above, row below)` pair starts at the active region; `x_tab`
    /// holds each output pixel's two source columns and weight.  The
    /// source samples are gathered into arrays first so the arithmetic
    /// vectorises.
    fn resize_row_to_rgb(y: RowPair, i: RowPair, q: RowPair, x_tab: &[(usize, usize, f32)], yt: f32, inv_comp_scale: f32, out: &mut [u8]) {
        const C: usize = RESIZE_CHUNK;
        let yt_inv = 1.0 - yt;
        let gather = |(r0, r1): RowPair, tab: &[(usize, usize, f32)], g: &mut [[f32; C]; 4]| {
            for (k, &(x0, x1, _)) in tab.iter().enumerate() {
                g[0][k] = r0[x0];
                g[1][k] = r0[x1];
                g[2][k] = r1[x0];
                g[3][k] = r1[x1];
            }
        };
        for (c, tab) in x_tab.chunks(C).enumerate() {
            let (mut gy, mut gi, mut gq) = ([[0.0f32; C]; 4], [[0.0f32; C]; 4], [[0.0f32; C]; 4]);
            gather(y, tab, &mut gy);
            gather(i, tab, &mut gi);
            gather(q, tab, &mut gq);
            let mut xt = [0.0f32; C];
            for (k, &(_, _, t)) in tab.iter().enumerate() {
                xt[k] = t;
            }
            let mut rgb = [[0u8; C]; 3];
            for k in 0..C {
                let xt_inv = 1.0 - xt[k];
                let w00 = xt_inv * yt_inv;
                let w01 = xt[k] * yt_inv;
                let w10 = xt_inv * yt;
                let w11 = xt[k] * yt;
                let y = gy[0][k] * w00 + gy[1][k] * w01 + gy[2][k] * w10 + gy[3][k] * w11;
                let i = (gi[0][k] * w00 + gi[1][k] * w01 + gi[2][k] * w10 + gi[3][k] * w11) * inv_comp_scale;
                let q = (gq[0][k] * w00 + gq[1][k] * w01 + gq[2][k] * w10 + gq[3][k] * w11) * inv_comp_scale;

                let r = (YIQ_TO_RGB[0][0] * y + YIQ_TO_RGB[0][1] * i + YIQ_TO_RGB[0][2] * q).clamp(0.0, 1.0);
                let g = (YIQ_TO_RGB[1][0] * y + YIQ_TO_RGB[1][1] * i + YIQ_TO_RGB[1][2] * q).clamp(0.0, 1.0);
                let b = (YIQ_TO_RGB[2][0] * y + YIQ_TO_RGB[2][1] * i + YIQ_TO_RGB[2][2] * q).clamp(0.0, 1.0);
                rgb[0][k] = (r * 255.0 + 0.5) as u8;
                rgb[1][k] = (g * 255.0 + 0.5) as u8;
                rgb[2][k] = (b * 255.0 + 0.5) as u8;
            }
            let dst = &mut out[c * C * 3..(c * C + tab.len()) * 3];
            for (k, px) in dst.chunks_exact_mut(3).enumerate() {
                px[0] = rgb[0][k];
                px[1] = rgb[1][k];
                px[2] = rgb[2][k];
            }
        }
    }
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}
