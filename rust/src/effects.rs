//! Signal degradation effects for simulating weak/noisy NTSC reception.
//!
//! The ghosting model is physically motivated by real multipath propagation:
//!
//!   - **Multiple reflections** — each path (building, hill, ionospheric skip)
//!     arrives at a different delay and amplitude.
//!   - **Phase shift** — reflections often invert polarity (π shift) or arrive
//!     at an arbitrary phase depending on the path geometry.
//!   - **High-frequency rolloff** — rough surfaces scatter and absorb HF energy,
//!     so reflected signals are softer than the direct path. Modelled with a
//!     single-pole IIR lowpass per ghost.
//!   - **Sub-sample delay** — linear interpolation between adjacent samples for
//!     delays that don't land on an exact sample boundary.
//!   - **Dynamic amplitude** — environmental movement (foliage, traffic, aircraft)
//!     causes the ghost strength to vary slowly over time. Modelled with a sum of
//!     low-frequency sinusoids.

use rand::Rng;
use rand_distr::{Distribution, Normal};

use crate::constants::{ACTIVE_SAMPLES, ACTIVE_START, BLANKING_V, SAMPLES_PER_LINE, TOTAL_LINES};
use crate::filters;

// ────────────────────────────────────────────────────────────────────────────
// Ghost configuration
// ────────────────────────────────────────────────────────────────────────────

/// Description of a single multipath reflection.
#[derive(Clone, Debug)]
pub struct GhostConfig {
    /// Ghost strength 0–1 (relative to the direct signal).
    pub amplitude: f32,
    /// Propagation delay in microseconds.  Typical range 0.5–10 µs for
    /// terrestrial reception; 50–100+ µs for ionospheric skip.
    pub delay_us: f32,
    /// Phase rotation in radians.  0 = in-phase, π = polarity inversion
    /// ("negative ghost").  Any value is accepted.
    pub phase_shift: f32,
    /// −3 dB cutoff (Hz) of the 1-pole HF rolloff applied to this ghost.
    /// `None` uses the default (3 MHz).  Lower values make the ghost
    /// visibly softer/blurrier.
    pub rolloff_hz: Option<f64>,
    /// If `true`, the amplitude is slowly modulated over time to simulate
    /// environmental variation (swaying trees, passing vehicles, etc.).
    pub dynamic: bool,
    /// Rate (Hz) of the slowest dynamic modulation component.
    /// Only used when `dynamic` is `true`.  Default ≈ 0.5 Hz.
    pub dynamic_rate: f32,
}

impl Default for GhostConfig {
    fn default() -> Self {
        Self {
            amplitude: 0.3,
            delay_us: 2.0,
            phase_shift: 0.0,
            rolloff_hz: None,
            dynamic: false,
            dynamic_rate: 0.5,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Top-level effect bundle
// ────────────────────────────────────────────────────────────────────────────

/// Configurable signal degradation effects.
pub struct SignalEffects {
    /// Gaussian noise amplitude (e.g. 0.05 = subtle, 0.2 = heavy snow)
    pub noise: Option<f32>,
    /// One or more multipath ghosts.  Empty vec = no ghosting.
    pub ghosts: Vec<GhostConfig>,
    /// Attenuation strength 0–1 (0 = no change, 1 = flat at blanking)
    pub attenuation: Option<f32>,
    /// Jitter: std dev in subcarrier cycles
    pub jitter: Option<f32>,
    // ── VHS tape-path effects ──
    /// VHS luma bandwidth limit in Hz (e.g. 3.0e6 for SP, 1.6e6 for EP).
    /// `None` = no extra luma bandwidth reduction.
    pub vhs_luma_bw: Option<f32>,
    /// VHS color-under chroma bandwidth in Hz (e.g. 300e3–500e3).
    /// Simulates the heterodyne round-trip that defines the VHS look.
    /// `None` = no color-under processing.
    pub color_under_bw: Option<f32>,
    /// Average tape dropouts per frame (e.g. 2–20).  `None` = no dropouts.
    pub tape_dropout_rate: Option<f32>,
    /// Average dropout length in microseconds (default 15.0).
    pub tape_dropout_len: f32,
    /// Edge peaking / ringing gain (e.g. 0.5–3.0).  Simulates the
    /// sharpness-enhancement circuit in VHS playback decks.
    /// `None` = no ringing.
    pub edge_ringing: Option<f32>,
    /// Luminance-dependent noise amplitude.  Noise is stronger in dark
    /// areas, mimicking real VHS tape noise characteristics.
    /// `None` = no luma-dependent noise.
    pub luma_noise: Option<f32>,
    /// Head-wear smearing strength (e.g. 0.3–1.0).  Simulates the
    /// horizontal smudging caused by worn heads / degraded tape oxide
    /// from repeated playback.  Applies a per-line variable-width
    /// box blur whose width drifts slowly across groups of lines.
    /// `None` = no head-wear smearing.
    pub head_switching_smear: Option<f32>,
    /// Tape trailing / causal smear strength (e.g. 0.3–1.0).  Simulates
    /// the rightward-only horizontal smudge caused by the causal low-pass
    /// response of worn VHS playback electronics.  Uses a one-pole IIR
    /// (exponential moving average) so edges and noise leave a "comet
    /// tail" trailing to the right — the classic worn-tape look.
    /// `None` = no trailing smear.
    pub tape_trail_smear: Option<f32>,
}

impl SignalEffects {
    pub fn is_active(&self) -> bool {
        self.noise.is_some()
            || !self.ghosts.is_empty()
            || self.attenuation.is_some()
            || self.jitter.is_some()
            || self.vhs_luma_bw.is_some()
            || self.color_under_bw.is_some()
            || self.tape_dropout_rate.is_some()
            || self.edge_ringing.is_some()
            || self.luma_noise.is_some()
            || self.head_switching_smear.is_some()
            || self.tape_trail_smear.is_some()
    }

    /// Apply all active effects to the signal in-place.
    ///
    /// Processing order:
    ///   1. VHS tape-path (luma BW + color-under) — fundamental recording limits
    ///   2. Edge ringing — playback circuit detail enhancement
    ///   3. Head-wear smearing — worn head / degraded oxide blur
    ///   4. Luminance-dependent noise — tape media noise
    ///   5. Tape dropout — physical oxide damage
    ///   6. Gaussian noise — reception snow
    ///   7. Tape trailing — causal rightward luma smudge (after noise so
    ///      noise gets smeared too)
    ///   8. Ghosting / attenuation / jitter — reception effects
    pub fn apply(&self, signal: &mut [f32], sample_rate: f64, rng: &mut impl Rng) {
        // VHS tape-path effects (must come first)
        if self.vhs_luma_bw.is_some() || self.color_under_bw.is_some() {
            apply_vhs_tape_path(signal, self.vhs_luma_bw, self.color_under_bw, sample_rate);
        }
        if let Some(gain) = self.edge_ringing {
            apply_edge_ringing(signal, gain, sample_rate);
        }
        if let Some(strength) = self.head_switching_smear {
            apply_head_wear_smear(signal, strength, rng);
        }
        if let Some(amp) = self.luma_noise {
            apply_luma_noise(signal, amp, sample_rate, rng);
        }
        if let Some(rate) = self.tape_dropout_rate {
            apply_tape_dropout(signal, rate, self.tape_dropout_len, sample_rate, rng);
        }
        if let Some(amp) = self.noise {
            apply_noise(signal, amp, rng);
        }
        // Tape trail after noise so the rightward smear catches noise too
        if let Some(strength) = self.tape_trail_smear {
            apply_tape_trail_smear(signal, strength);
        }
        if !self.ghosts.is_empty() {
            apply_ghosting(signal, &self.ghosts, sample_rate, rng);
        }
        if let Some(strength) = self.attenuation {
            apply_attenuation(signal, strength);
        }
        if let Some(amp) = self.jitter {
            apply_jitter(signal, amp, rng);
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Noise
// ────────────────────────────────────────────────────────────────────────────

/// Additive white Gaussian noise (snow).
fn apply_noise(signal: &mut [f32], amplitude: f32, rng: &mut impl Rng) {
    let normal = Normal::new(0.0f32, amplitude).unwrap();
    for s in signal.iter_mut() {
        *s += normal.sample(rng);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Ghosting — physically motivated multipath model
// ────────────────────────────────────────────────────────────────────────────

const DEFAULT_ROLLOFF_HZ: f64 = 3.0e6; // 3 MHz – typical reflective surface cutoff

/// Apply one or more multipath ghosts to `signal`.
///
/// For every ghost, we:
///   1. Build a delayed copy using sub-sample linear interpolation.
///   2. Apply a 1-pole IIR lowpass to simulate HF absorption.
///   3. Apply a phase shift (cos rotation for real signals; exact at 0 and π).
///   4. Optionally modulate amplitude with a slow LFO envelope.
///   5. Mix into the output.
fn apply_ghosting(
    signal: &mut [f32],
    ghosts: &[GhostConfig],
    sample_rate: f64,
    rng: &mut impl Rng,
) {
    let n = signal.len();
    if n == 0 {
        return;
    }

    // We need the original (direct-path) signal as a read-only reference so
    // that successive ghosts are all derived from the *primary* signal, not
    // from already-ghosted data.
    let original: Vec<f32> = signal.to_vec();

    for (gi, g) in ghosts.iter().enumerate() {
        if g.amplitude.abs() < 1e-9 {
            continue;
        }

        // ── 1. Fractional-sample delay via linear interpolation ──────────
        let delay_samples_f = g.delay_us as f64 * sample_rate / 1e6;
        if delay_samples_f <= 0.0 || delay_samples_f >= n as f64 {
            continue;
        }
        let delay_int = delay_samples_f as usize; // integer part
        let frac = delay_samples_f - delay_int as f64; // fractional part [0, 1)
        let frac_f32 = frac as f32;
        let one_minus_frac = 1.0 - frac_f32;

        // Build interpolated delayed copy into a scratch buffer.
        let mut ghost_buf: Vec<f32> = vec![0.0; n];

        if frac_f32.abs() < 1e-6 {
            // Exact-sample delay — no interpolation needed.
            ghost_buf[delay_int..].copy_from_slice(&original[..n - delay_int]);
        } else {
            // Linear interpolation between sample[i - d] and sample[i - d - 1]
            // ghost[i] = (1-f) * orig[i - d] + f * orig[i - d - 1]
            //
            // First valid output index is delay_int + 1 (we need i-d-1 >= 0).
            let start = delay_int + 1;
            for i in start..n {
                let a = original[i - delay_int];     // nearer sample
                let b = original[i - delay_int - 1]; // farther sample
                ghost_buf[i] = one_minus_frac * a + frac_f32 * b;
            }
        }

        // ── 2. HF rolloff (1-pole IIR lowpass) ──────────────────────────
        let cutoff = g.rolloff_hz.unwrap_or(DEFAULT_ROLLOFF_HZ);
        let alpha = 1.0 - (-2.0 * std::f64::consts::PI * cutoff / sample_rate).exp() as f32;
        {
            let mut prev = 0.0f32;
            for s in ghost_buf.iter_mut() {
                prev += alpha * (*s - prev);
                *s = prev;
            }
        }

        // ── 3. Phase shift ──────────────────────────────────────────────
        // For a purely real signal the best we can do without a full Hilbert
        // transform is multiply by cos(phase).  This is exact for 0 (in-phase)
        // and π (polarity inversion) — the two most common real-world cases —
        // and a reasonable approximation in between.
        let phase_gain = g.phase_shift.cos();

        // ── 4. Optional dynamic amplitude modulation ────────────────────
        // Sum-of-sinusoids envelope: fundamental + 2 harmonics with random
        // phases.  Normalised to keep the envelope positive and bounded.
        let dynamic_env: Option<Vec<f32>> = if g.dynamic {
            let mut env = vec![1.0f32; n];
            let inv_sr = 1.0 / sample_rate as f32;
            for harmonic in 1..=3u32 {
                let phi: f32 = rng.random::<f32>() * std::f32::consts::TAU;
                let freq = g.dynamic_rate * harmonic as f32;
                for (j, e) in env.iter_mut().enumerate() {
                    *e += 0.15
                        * (std::f32::consts::TAU * freq * (j as f32) * inv_sr + phi).sin();
                }
            }
            // Normalise to [0.1, 1.0]
            let max_val = env.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            if max_val > 0.0 {
                let inv_max = 1.0 / max_val;
                for e in env.iter_mut() {
                    *e = (*e * inv_max).clamp(0.1, 1.0);
                }
            }
            Some(env)
        } else {
            None
        };

        // ── 5. Mix ghost into output ────────────────────────────────────
        let amp = g.amplitude * phase_gain;
        match dynamic_env {
            Some(ref env) => {
                for i in 0..n {
                    signal[i] += amp * env[i] * ghost_buf[i];
                }
            }
            None => {
                for i in 0..n {
                    signal[i] += amp * ghost_buf[i];
                }
            }
        }

        // Suppress unused-variable warning when ghosts are indexed but gi
        // isn't used outside the dynamic seed.
        let _ = gi;
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Attenuation
// ────────────────────────────────────────────────────────────────────────────

/// Compress signal toward blanking level, reducing contrast and saturation.
fn apply_attenuation(signal: &mut [f32], strength: f32) {
    let factor = 1.0 - strength;
    for s in signal.iter_mut() {
        *s = BLANKING_V + (*s - BLANKING_V) * factor;
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Jitter
// ────────────────────────────────────────────────────────────────────────────

/// Per-line horizontal timing instability.
/// Shifts lines by whole subcarrier cycles (4 samples) so the picture
/// wobbles horizontally without altering decoded color.
fn apply_jitter(signal: &mut [f32], amplitude: f32, rng: &mut impl Rng) {
    let expected = TOTAL_LINES * SAMPLES_PER_LINE;
    if signal.len() < expected {
        return;
    }
    let normal = Normal::new(0.0f32, amplitude).unwrap();
    for line in 0..TOTAL_LINES {
        let cycles = normal.sample(rng).round() as i32;
        if cycles == 0 {
            continue;
        }
        let shift = (cycles * 4).unsigned_abs() as usize % SAMPLES_PER_LINE;
        if shift == 0 {
            continue;
        }
        let base = line * SAMPLES_PER_LINE;
        let row = &mut signal[base..base + SAMPLES_PER_LINE];
        if cycles > 0 {
            row.rotate_right(shift);
        } else {
            row.rotate_left(shift);
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// VHS tape-path simulation
// ────────────────────────────────────────────────────────────────────────────

/// Number of FIR taps for VHS bandwidth-limiting filters.
const VHS_FILTER_TAPS: usize = 61;

/// Number of FIR taps for edge-ringing peaking filter (more taps = more ringing).
const RINGING_FILTER_TAPS: usize = 91;

/// Apply FIR filter in-place via direct convolution with edge extension.
///
/// `scratch` is reused across calls to avoid repeated allocation.
fn apply_fir_inplace(buf: &mut [f32], kernel: &[f32], scratch: &mut Vec<f32>) {
    let n = buf.len();
    let k = kernel.len();
    let half = k / 2;
    let padded = n + k - 1;

    scratch.resize(padded, 0.0);
    let left = buf[0];
    let right = buf[n - 1];
    scratch[..half].fill(left);
    scratch[half..half + n].copy_from_slice(buf);
    scratch[half + n..padded].fill(right);

    for i in 0..n {
        let mut sum = 0.0f32;
        let s = &scratch[i..i + k];
        for j in 0..k {
            sum += s[j] * kernel[j];
        }
        buf[i] = sum;
    }
}

/// Combined VHS luma bandwidth reduction and color-under chroma processing.
///
/// This models the two separate recording paths inside a VCR:
///   - **Luma path**: FM-modulated and bandwidth-limited by the tape.
///   - **Chroma path**: heterodyned down to 629 kHz ("color-under"),
///     recorded, and heterodyned back up on playback.  The round-trip
///     through the low-frequency carrier severely limits chroma bandwidth.
///
/// Both are done in a single per-line pass to avoid redundant luma/chroma
/// separation.
fn apply_vhs_tape_path(
    signal: &mut [f32],
    luma_bw_hz: Option<f32>,
    chroma_bw_hz: Option<f32>,
    _sample_rate: f64,
) {
    if luma_bw_hz.is_none() && chroma_bw_hz.is_none() {
        return;
    }

    let spl = SAMPLES_PER_LINE;

    // Pre-compute FIR kernels
    let luma_kernel = luma_bw_hz.map(|bw| filters::design_lowpass(bw as f64, VHS_FILTER_TAPS));
    let chroma_kernel =
        chroma_bw_hz.map(|bw| filters::design_lowpass(bw as f64, VHS_FILTER_TAPS));

    // Carrier LUTs for demod/remod at 4×fsc
    //   cos(π/2 · n) = [1, 0, −1, 0, …]
    //  −sin(π/2 · n) = [0, −1, 0, 1, …]
    let cos_lut: [f32; 4] = [1.0, 0.0, -1.0, 0.0];
    let nsin_lut: [f32; 4] = [0.0, -1.0, 0.0, 1.0];

    let mut luma = vec![0.0f32; spl];
    let mut chroma = vec![0.0f32; spl];
    let mut i_bb = vec![0.0f32; spl];
    let mut q_bb = vec![0.0f32; spl];
    let mut scratch = Vec::new();

    for line in 0..TOTAL_LINES {
        let base = line * spl;
        if base + spl > signal.len() {
            break;
        }

        // ── Separate luma and chroma via 2-sample delay ──
        luma[0] = signal[base];
        luma[1] = signal[base + 1];
        chroma[0] = 0.0;
        chroma[1] = 0.0;
        for n in 2..spl {
            let cur = signal[base + n];
            let delayed = signal[base + n - 2];
            luma[n] = (cur + delayed) * 0.5;
            chroma[n] = (cur - delayed) * 0.5;
        }

        // ── Luma bandwidth reduction ──
        if let Some(ref kernel) = luma_kernel {
            apply_fir_inplace(&mut luma, kernel, &mut scratch);
        }

        // ── Color-under chroma bandwidth limitation ──
        if let Some(ref kernel) = chroma_kernel {
            // Demodulate chroma to baseband I/Q
            for n in 0..spl {
                i_bb[n] = 2.0 * chroma[n] * cos_lut[n & 3];
                q_bb[n] = 2.0 * chroma[n] * nsin_lut[n & 3];
            }

            // Lowpass I and Q at the color-under bandwidth
            apply_fir_inplace(&mut i_bb, kernel, &mut scratch);
            apply_fir_inplace(&mut q_bb, kernel, &mut scratch);

            // Remodulate back to chroma
            for n in 0..spl {
                chroma[n] = i_bb[n] * cos_lut[n & 3] + q_bb[n] * nsin_lut[n & 3];
            }
        }

        // ── Recombine ──
        for n in 0..spl {
            signal[base + n] = luma[n] + chroma[n];
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tape dropout
// ────────────────────────────────────────────────────────────────────────────

/// Simulate VHS tape dropouts — random horizontal streaks caused by oxide
/// shedding or dust on the head.
///
/// - Small dropouts (< 20 samples): brief white flash / spike.
/// - Larger dropouts: the dropout compensator (DOC) replaces the damaged
///   segment with data from the previous scan line.
fn apply_tape_dropout(
    signal: &mut [f32],
    dropout_rate: f32,
    avg_len_us: f32,
    sample_rate: f64,
    rng: &mut impl Rng,
) {
    let spl = SAMPLES_PER_LINE;
    let avg_len_samples = (avg_len_us as f64 * sample_rate / 1e6).max(4.0);
    let prob_per_line = (dropout_rate / TOTAL_LINES as f32).clamp(0.0, 1.0);

    for line in 0..TOTAL_LINES {
        if rng.random::<f32>() >= prob_per_line {
            continue;
        }

        let base = line * spl;
        if base + spl > signal.len() {
            break;
        }

        // Random position within active picture area
        let offset = (rng.random::<f32>() * ACTIVE_SAMPLES as f32) as usize;
        let start_idx = base + ACTIVE_START + offset;

        // Exponentially distributed length (manual: −ln(u) × mean)
        let u: f32 = rng.random::<f32>().max(1e-10);
        let len = (-u.ln() * avg_len_samples as f32)
            .max(4.0)
            .min((ACTIVE_SAMPLES - offset) as f32) as usize;
        let end_idx = (start_idx + len).min(base + spl);

        if len < 20 {
            // Brief white spike (most visible VHS dropout type)
            let spike_level = BLANKING_V + 0.4 + rng.random::<f32>() * 0.25;
            for s in &mut signal[start_idx..end_idx] {
                *s = spike_level;
            }
        } else if line > 0 {
            // Dropout compensator: hold from previous scan line
            let prev_start = start_idx - spl;
            for i in 0..end_idx - start_idx {
                signal[start_idx + i] = signal[prev_start + i];
            }
        } else {
            // First line — fill with blanking
            signal[start_idx..end_idx].fill(BLANKING_V);
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Edge ringing / sharpness peaking
// ────────────────────────────────────────────────────────────────────────────

/// Simulate the detail-enhancement (peaking) circuit found in VHS playback
/// decks.  A lowpass extracts a "blurred" version of each line; the
/// difference (high-frequency detail) is added back with adjustable gain.
///
/// Because the windowed-sinc FIR has Gibbs-phenomenon overshoot, this
/// naturally creates the bright/dark halos around edges that are
/// characteristic of VHS sharpness enhancement.
fn apply_edge_ringing(signal: &mut [f32], strength: f32, _sample_rate: f64) {
    let spl = SAMPLES_PER_LINE;

    // The peaking filter extracts detail above ~1.5 MHz.  Using many taps
    // makes the ringing more pronounced (intentionally).
    let detail_lp = filters::design_lowpass(1.5e6, RINGING_FILTER_TAPS);

    let mut blurred = vec![0.0f32; spl];
    let mut scratch = Vec::new();

    for line in 0..TOTAL_LINES {
        let base = line * spl;
        if base + spl > signal.len() {
            break;
        }

        blurred.copy_from_slice(&signal[base..base + spl]);
        apply_fir_inplace(&mut blurred, &detail_lp, &mut scratch);

        // Unsharp-mask: output = signal + gain × (signal − blurred)
        for n in 0..spl {
            let detail = signal[base + n] - blurred[n];
            signal[base + n] += strength * detail;
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Luminance-dependent noise
// ────────────────────────────────────────────────────────────────────────────

/// Band-limited noise whose amplitude scales with signal darkness.
///
/// VHS tape noise is stronger in dark scenes because the FM carrier
/// deviation is lower (less signal energy relative to the noise floor).
/// The noise is band-limited to ≈ 3 MHz to mimic the granular character
/// of real tape noise (as opposed to pure white noise which looks like
/// sharp "snow").
fn apply_luma_noise(
    signal: &mut [f32],
    amplitude: f32,
    _sample_rate: f64,
    rng: &mut impl Rng,
) {
    let spl = SAMPLES_PER_LINE;
    let normal = Normal::new(0.0f32, 1.0).unwrap();

    // Band-limit noise to ~3 MHz for tape-like grain
    let noise_lp = filters::design_lowpass(3.0e6, 31);

    let mut noise_line = vec![0.0f32; spl];
    let mut scratch = Vec::new();

    for line in 0..TOTAL_LINES {
        let base = line * spl;
        if base + spl > signal.len() {
            break;
        }

        // Generate white noise
        for s in noise_line.iter_mut() {
            *s = normal.sample(rng);
        }

        // Band-limit to give it a granular / tape-like texture
        apply_fir_inplace(&mut noise_line, &noise_lp, &mut scratch);

        // Scale by darkness: full amplitude at black, 30 % at peak white
        for n in 0..spl {
            let luma_level = ((signal[base + n] - BLANKING_V) / 0.66).clamp(0.0, 1.0);
            let noise_scale = 1.0 - 0.7 * luma_level;
            signal[base + n] += amplitude * noise_scale * noise_line[n];
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Head-wear smearing (symmetric box blur)
// ────────────────────────────────────────────────────────────────────────────

/// Simulate the horizontal smudging artifact caused by worn video heads and
/// degraded tape oxide from repeated playback.
///
/// Real worn-tape smearing has a distinctive look:
///   - It varies **slowly** from line to line (smooth vertical coherence),
///     because the head-to-tape contact quality changes gradually as the
///     head sweeps across the tape.
///   - Some regions of the frame are clear while others are heavily smeared,
///     producing "patches" or "bands" of blur.
///   - The smeared regions shift slightly from frame to frame (random seed).
///
/// The box blur is applied only to the **luma** component (separated via
/// the 2-sample comb) so the color subcarrier is preserved.
///
/// Implementation: a per-line running-average (box filter) whose kernel
/// width is modulated by a sum of low-frequency sinusoids across line
/// number, creating smooth bands of varying smear intensity.
fn apply_head_wear_smear(signal: &mut [f32], strength: f32, rng: &mut impl Rng) {
    let spl = SAMPLES_PER_LINE;
    if signal.len() < TOTAL_LINES * spl {
        return;
    }

    // Maximum blur radius in samples.  At strength=1.0 the worst lines get
    // a ~32-sample box blur (~2.2 µs at 14.3 MHz), which is quite visible.
    let max_radius = (strength * 32.0).max(1.0) as usize;

    // Build a per-line smear-width map using 3 incommensurate sinusoids
    // so the blur amount drifts smoothly with interesting, non-repeating
    // variation across the frame.
    let phase1: f32 = rng.random::<f32>() * std::f32::consts::TAU;
    let phase2: f32 = rng.random::<f32>() * std::f32::consts::TAU;
    let phase3: f32 = rng.random::<f32>() * std::f32::consts::TAU;

    // Frequencies chosen so the "beats" span 40–120 lines
    let freq1 = std::f32::consts::TAU / 73.0;
    let freq2 = std::f32::consts::TAU / 127.0;
    let freq3 = std::f32::consts::TAU / 41.0;

    let mut luma = vec![0.0f32; spl];
    let mut chroma = vec![0.0f32; spl];
    let mut luma_active = vec![0.0f32; ACTIVE_SAMPLES];

    for line in 0..TOTAL_LINES {
        let t = line as f32;

        // Envelope in [0, 1] — peaks are where the smear is worst
        let env = ((freq1 * t + phase1).sin() * 0.5
            + (freq2 * t + phase2).sin() * 0.3
            + (freq3 * t + phase3).sin() * 0.2)
            .clamp(0.0, 1.0);

        let radius = (env * max_radius as f32) as usize;
        if radius == 0 {
            continue;
        }

        let base = line * spl;
        if base + spl > signal.len() {
            break;
        }

        // ── Separate luma and chroma via 2-sample comb ──
        luma[0] = signal[base];
        luma[1] = signal[base + 1];
        chroma[0] = 0.0;
        chroma[1] = 0.0;
        for n in 2..spl {
            let cur = signal[base + n];
            let delayed = signal[base + n - 2];
            luma[n] = (cur + delayed) * 0.5;
            chroma[n] = (cur - delayed) * 0.5;
        }

        // ── Box-blur the luma active region only ──
        let act_len = ACTIVE_SAMPLES;
        luma_active[..act_len].copy_from_slice(&luma[ACTIVE_START..ACTIVE_START + act_len]);

        // Prefix sums for O(1) box filter per sample
        let mut prefix = vec![0.0f32; act_len + 1];
        for i in 0..act_len {
            prefix[i + 1] = prefix[i] + luma_active[i];
        }

        for i in 0..act_len {
            let lo = if i >= radius { i - radius } else { 0 };
            let hi = (i + radius + 1).min(act_len);
            let count = (hi - lo) as f32;
            luma[ACTIVE_START + i] = (prefix[hi] - prefix[lo]) / count;
        }

        // ── Recombine: blurred luma + original chroma ──
        for n in 0..spl {
            signal[base + n] = luma[n] + chroma[n];
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tape trailing (causal rightward smear)
// ────────────────────────────────────────────────────────────────────────────

/// Simulate the rightward-only horizontal smudge from worn VHS playback
/// electronics and degraded tape.
///
/// Unlike the symmetric box blur of `apply_head_wear_smear`, this effect
/// is **causal** — it only trails to the right.  This matches the real
/// physics: the VHS head reads the tape in one direction and the limited
/// playback bandwidth acts as a one-pole IIR low-pass filter that can only
/// "remember" past (left-of) samples.  The result is:
///
///   - Sharp edges leave a decaying "comet tail" to the right.
///   - Noise is smudged into soft horizontal streaks instead of dots.
///
/// The IIR is applied only to the **luma** component of the composite
/// signal (separated via the 2-sample comb) so that the color subcarrier
/// is preserved.  This prevents color washout and dot-crawl artifacts.
///
/// The effect is applied **uniformly** to every scanline because it models
/// the playback circuit's bandwidth limitation, not head-to-tape contact
/// variation (that's what `head_switching_smear` is for).
///
/// `strength` controls how heavy the trailing is (0.3 = subtle, 1.0 = heavy).
fn apply_tape_trail_smear(signal: &mut [f32], strength: f32) {
    let spl = SAMPLES_PER_LINE;
    if signal.len() < TOTAL_LINES * spl {
        return;
    }

    // α controls the IIR: y[n] = α·x[n] + (1−α)·y[n−1]
    // Smaller α → heavier trailing.  Map strength 0–1 to α 0.85–0.15.
    let alpha = 1.0 - strength.clamp(0.0, 1.0) * 0.70;
    let one_minus_alpha = 1.0 - alpha;

    let mut luma = vec![0.0f32; spl];
    let mut chroma = vec![0.0f32; spl];

    for line in 0..TOTAL_LINES {
        let base = line * spl;
        if base + spl > signal.len() {
            break;
        }

        // ── Separate luma and chroma via 2-sample comb ──
        luma[0] = signal[base];
        luma[1] = signal[base + 1];
        chroma[0] = 0.0;
        chroma[1] = 0.0;
        for n in 2..spl {
            let cur = signal[base + n];
            let delayed = signal[base + n - 2];
            luma[n] = (cur + delayed) * 0.5;
            chroma[n] = (cur - delayed) * 0.5;
        }

        // ── Causal IIR on luma only (active region) ──
        let act_start = ACTIVE_START;
        let act_end = ACTIVE_START + ACTIVE_SAMPLES;
        let mut prev = luma[act_start];
        for n in (act_start + 1)..act_end {
            prev = alpha * luma[n] + one_minus_alpha * prev;
            luma[n] = prev;
        }

        // ── Recombine: smeared luma + original chroma ──
        for n in 0..spl {
            signal[base + n] = luma[n] + chroma[n];
        }
    }
}
