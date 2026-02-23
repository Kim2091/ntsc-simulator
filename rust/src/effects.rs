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

use crate::constants::{BLANKING_V, SAMPLES_PER_LINE, TOTAL_LINES};

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
}

impl SignalEffects {
    pub fn is_active(&self) -> bool {
        self.noise.is_some()
            || !self.ghosts.is_empty()
            || self.attenuation.is_some()
            || self.jitter.is_some()
    }

    /// Apply all active effects to the signal in-place.
    pub fn apply(&self, signal: &mut [f32], sample_rate: f64, rng: &mut impl Rng) {
        if let Some(amp) = self.noise {
            apply_noise(signal, amp, rng);
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
