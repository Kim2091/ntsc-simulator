//! Signal degradation effects for simulating weak/noisy NTSC reception.

use rand::Rng;
use rand_distr::{Distribution, Normal};

use crate::constants::{BLANKING_V, SAMPLES_PER_LINE, TOTAL_LINES};

/// Configurable signal degradation effects.
pub struct SignalEffects {
    /// Gaussian noise amplitude (e.g. 0.05 = subtle, 0.2 = heavy snow)
    pub noise: Option<f32>,
    /// Ghost: (amplitude 0-1, delay in microseconds)
    pub ghost: Option<(f32, f32)>,
    /// Attenuation strength 0-1 (0 = no change, 1 = flat at blanking)
    pub attenuation: Option<f32>,
    /// Jitter: std dev in subcarrier cycles
    pub jitter: Option<f32>,
}

impl SignalEffects {
    pub fn is_active(&self) -> bool {
        self.noise.is_some()
            || self.ghost.is_some()
            || self.attenuation.is_some()
            || self.jitter.is_some()
    }

    /// Apply all active effects to the signal in-place.
    pub fn apply(&self, signal: &mut [f32], sample_rate: f64, rng: &mut impl Rng) {
        if let Some(amp) = self.noise {
            apply_noise(signal, amp, rng);
        }
        if let Some((amp, delay_us)) = self.ghost {
            apply_ghosting(signal, amp, delay_us, sample_rate);
        }
        if let Some(strength) = self.attenuation {
            apply_attenuation(signal, strength);
        }
        if let Some(amp) = self.jitter {
            apply_jitter(signal, amp, rng);
        }
    }
}

/// Additive white Gaussian noise (snow).
fn apply_noise(signal: &mut [f32], amplitude: f32, rng: &mut impl Rng) {
    let normal = Normal::new(0.0f32, amplitude).unwrap();
    for s in signal.iter_mut() {
        *s += normal.sample(rng);
    }
}

/// Multipath ghost — adds a delayed, attenuated copy of the signal.
/// Iterates backwards to avoid reading already-modified values.
fn apply_ghosting(signal: &mut [f32], amplitude: f32, delay_us: f32, sample_rate: f64) {
    let delay_samples = (delay_us as f64 * sample_rate / 1e6).round() as usize;
    if delay_samples == 0 || delay_samples >= signal.len() {
        return;
    }
    // Backward iteration: signal[i] += amp * signal[i - delay]
    // Since i > i-delay, iterating from end to start means we read
    // signal[i-delay] before it gets modified (it would be modified at
    // a later iteration index that we've already passed).
    for i in (delay_samples..signal.len()).rev() {
        signal[i] += amplitude * signal[i - delay_samples];
    }
}

/// Compress signal toward blanking level, reducing contrast and saturation.
fn apply_attenuation(signal: &mut [f32], strength: f32) {
    let factor = 1.0 - strength;
    for s in signal.iter_mut() {
        *s = BLANKING_V + (*s - BLANKING_V) * factor;
    }
}

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
