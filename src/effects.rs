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

use crate::constants::{
    ACTIVE_SAMPLES, ACTIVE_START, BLANKING_V, BURST_SAMPLES, BURST_START, COMPOSITE_SCALE, FSC,
    FRONT_PORCH_SAMPLES, HSYNC_SAMPLES, I_PHASE_RAD, Q_PHASE_RAD, SAMPLES_PER_LINE, TOTAL_LINES,
};
use crate::encoder::Wire;
use crate::filters;
use crate::iir;
use crate::profile;
use crate::simd::multiversion;
use crate::vhs::{self, DeckLuma};

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
#[derive(Clone, Debug, Default)]
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
    /// Head-switching noise strength (0–1).  Tears the last two or three
    /// active lines of each field: a displaced switch line with a switching
    /// transient, then lines that have lost the signal (near black with
    /// bright noise streaks).  `None` = off.
    pub head_switch: Option<f32>,
    /// Number of active lines per field affected by head switching.
    pub head_switch_lines: usize,
    /// Top-of-frame chroma band strength (0–1).  Tints the first few active
    /// lines of each field blue/purple, as the deck's chroma phase loop
    /// settles after the vertical interval.  `None` = off.
    pub top_band: Option<f32>,
    /// Closed-caption data strength (0–1) on lines 21 and 284.  The
    /// waveform is blended over the picture on those lines, so it shows
    /// as short light dashes near the top of the frame.  `None` = off.
    pub vbi_data: Option<f32>,
    /// Colour-under chroma noise: std dev of the I/Q noise in YIQ units
    /// (e.g. 0.02–0.08).  `None` = off.
    pub chroma_noise: Option<f32>,
    /// Per-line chroma phase jitter: std dev in degrees.  `None` = off.
    pub chroma_phase_noise: Option<f32>,
    /// Playback 1H chroma comb strength (0–1).  `None` = off.
    pub chroma_vblur: Option<f32>,
    /// Horizontal chroma delay in output pixels at 640 wide (positive =
    /// right).  `None` = off.
    pub chroma_delay: Option<f32>,
    /// Vertical chroma delay in field lines (rounded).  `None` = off.
    pub chroma_delay_v: Option<f32>,
    /// Time-base wobble: peak horizontal displacement in output pixels at
    /// 640 wide.  `None` = off.
    pub wobble: Option<f32>,
    /// Frequency scale of the wobble components (1.0 = default periods).
    pub wobble_rate: f32,
    /// How slowly the per-frame random state drifts (0 = re-rolled every
    /// frame, 1 = fully coherent tape state).  See `TapeClock`.
    pub temporal_coherence: f32,
    /// Seed of the drifting tape state for this file.
    pub tape_seed: u64,
    /// Line time-base corrector strength (0–1).  `None` = off.
    pub line_tbc: Option<f32>,
    /// Frame TBC / frame synchroniser strength (0–1).  `None` = off.
    pub frame_tbc: Option<f32>,
    /// Vertical bounce, std dev in output rows.  `None` = off.
    pub v_jitter: Option<f32>,
    /// Deck sharpness control (−1 soft … 0 edit … 1 sharp).  `None` = off.
    pub deck_sharpness: Option<f32>,
    /// Deck recursive (temporal) noise reduction, applied to decoded frames
    /// in frame order by the pipeline.  `None` = off.
    pub deck_dnr: Option<f32>,
    /// Seed of the per-stage random streams of the stages in `vhs`.
    pub stage_seed: u64,
    /// Deck noise canceller limiter threshold in IRE.  `None` = off.
    pub deck_nc: Option<f32>,
    /// What the dropout compensator does with long dropouts.
    pub doc: Doc,
}

/// Default number of active lines per field affected by head switching.
pub const DEFAULT_HEAD_SWITCH_LINES: usize = 3;

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
            || self.head_switch.is_some()
            || self.top_band.is_some()
            || self.vbi_data.is_some()
            || self.chroma_noise.is_some()
            || self.chroma_phase_noise.is_some()
            || self.chroma_vblur.is_some()
            || self.chroma_delay.is_some()
            || self.chroma_delay_v.is_some()
            || self.wobble.is_some()
            || self.v_jitter.is_some()
            || self.deck_luma().is_active()
    }

    /// Settings of the deck's playback luma circuits.
    fn deck_luma(&self) -> DeckLuma {
        DeckLuma { nc_ire: self.deck_nc.unwrap_or(0.0).max(0.0), sharpness: self.deck_sharpness.unwrap_or(0.0) }
    }

    /// Settings for the playback chroma path; zero turns a stage off.
    fn chroma_path_config(&self) -> ChromaPathConfig {
        ChromaPathConfig {
            noise: self.chroma_noise.unwrap_or(0.0),
            phase_noise_deg: self.chroma_phase_noise.unwrap_or(0.0),
            vblur: self.chroma_vblur.unwrap_or(0.0),
            // Pixels at 640 wide to samples.
            delay_samples: self.chroma_delay.unwrap_or(0.0) * ACTIVE_SAMPLES as f32 / 640.0,
            delay_lines: self.chroma_delay_v.map_or(0, |v| v.round().max(0.0) as usize),
        }
    }

    /// Apply all active effects to the signal in-place.
    ///
    /// Processing order:
    ///   1. VBI closed-caption data — part of the recorded broadcast signal
    ///   2. VHS tape-path (luma BW + color-under) — fundamental recording limits,
    ///      then the playback chroma path (colour-under noise, phase jitter,
    ///      1H comb, chroma delay)
    ///   3. Edge ringing — playback circuit detail enhancement
    ///   4. Head-wear smearing — worn head / degraded oxide blur
    ///   5. Luminance-dependent noise — tape media noise
    ///   6. Tape dropout — physical oxide damage
    ///   7. Wobble / top chroma band / head switching — playback timing events
    ///   8. Gaussian noise — reception snow
    ///   9. Ghosting / attenuation / jitter — reception effects
    ///  10. Tape trailing — **last**: causal rightward luma smudge from
    ///      playback electronics.  Everything on the tape (noise, dropouts,
    ///      ghosts) passes through this bottleneck on readout.
    ///
    /// `frame` is the frame number within the file; it drives the drifting
    /// tape state when `temporal_coherence` is above zero.
    pub fn apply(&self, signal: &mut [f32], sample_rate: f64, rng: &mut (impl Rng + Clone), frame: u64) {
        self.apply_wire(signal, sample_rate, rng, frame, Wire::Composite);
    }

    /// Apply the effects to one wire of the signal.
    ///
    /// For S-Video the Y and C wires are processed with clones of the same
    /// RNG.  Luma-domain effects (luma bandwidth, ringing, smear, luma
    /// noise, caption data, trail) only change the Y wire and
    /// chroma-domain effects (colour-under, chroma path, top band) only
    /// the C wire; effects that draw random numbers still run (on a
    /// scratch copy) on the other wire, so both wires see the same
    /// dropouts, jitter, wobble and head switching.  Composite runs
    /// everything, exactly as `apply`.  [`Self::apply_svideo`] gives the
    /// same result for both wires with less work.
    pub fn apply_wire(&self, signal: &mut [f32], sample_rate: f64, rng: &mut (impl Rng + Clone), frame: u64, wire: Wire) {
        self.apply_wires(&mut [(signal, wire)], sample_rate, rng, frame);
    }

    /// Apply the effects to both S-Video wires.  The result is identical
    /// to `apply_wire` on each wire with a clone of `rng`: no effect's
    /// random draws depend on the signal, so one RNG can serve both wires,
    /// and an effect that belongs to one wire only runs on that wire.
    pub fn apply_svideo(&self, y: &mut [f32], c: &mut [f32], sample_rate: f64, rng: &mut (impl Rng + Clone), frame: u64) {
        self.apply_wires(&mut [(y, Wire::Luma), (c, Wire::Chroma)], sample_rate, rng, frame);
    }

    /// Run every effect on each of `wires` it belongs to.  An effect that
    /// draws random numbers runs on each target wire from the same RNG
    /// state; if no wire takes it, it runs on a scratch copy so the RNG
    /// still advances as it would on a wire.
    fn apply_wires<R: Rng + Clone>(&self, wires: &mut [(&mut [f32], Wire)], sample_rate: f64, rng: &mut R, frame: u64) {
        // A line TBC re-times every line from its sync and burst: it removes
        // most line-to-line timing error but not the noise at the head
        // switch.  A frame TBC holds the picture vertically.
        let tbc = self.line_tbc.unwrap_or(0.0).clamp(0.0, 1.0);
        let ftbc = self.frame_tbc.unwrap_or(0.0).clamp(0.0, 1.0);
        let clock = TapeClock {
            seed: self.tape_seed,
            frame,
            coherence: self.temporal_coherence.clamp(0.0, 1.0),
        };
        let luma_on = |w: Wire| w != Wire::Chroma;
        let chroma_on = |w: Wire| w != Wire::Luma;
        let all = |_: Wire| true;
        let mut scratch: Vec<f32> = Vec::new();
        // Run `f(signal, wire, rng)` on the wires `active` selects, each
        // from the current RNG state, leaving `rng` advanced past it.
        let mut run = |wires: &mut [(&mut [f32], Wire)],
                       rng: &mut R,
                       active: &dyn Fn(Wire) -> bool,
                       f: &mut dyn FnMut(&mut [f32], Wire, &mut R)| {
            let targets: Vec<usize> = (0..wires.len()).filter(|&i| active(wires[i].1)).collect();
            match targets.split_last() {
                None => {
                    scratch.clear();
                    scratch.extend_from_slice(wires[0].0);
                    f(&mut scratch, wires[0].1, rng);
                }
                Some((&last, rest)) => {
                    for &i in rest {
                        f(wires[i].0, wires[i].1, &mut rng.clone());
                    }
                    f(wires[last].0, wires[last].1, rng);
                }
            }
        };
        // Effects without random draws.
        let each = |wires: &mut [(&mut [f32], Wire)], f: &mut dyn FnMut(&mut [f32], Wire)| {
            for (signal, wire) in wires.iter_mut() {
                f(signal, *wire);
            }
        };

        // Closed-caption data is on the tape before any tape-path limits.
        if let Some(strength) = self.vbi_data {
            let _p = profile::scope("fx vbi-data");
            run(wires, rng, &luma_on, &mut |s, _, rng| apply_vbi_data(s, strength, sample_rate, rng));
        }
        // VHS tape-path effects
        if self.vhs_luma_bw.is_some() || self.color_under_bw.is_some() {
            let _p = profile::scope("fx tape path (luma/colour-under bw)");
            each(wires, &mut |s, w| {
                let luma_bw = if luma_on(w) { self.vhs_luma_bw } else { None };
                let chroma_bw = if chroma_on(w) { self.color_under_bw } else { None };
                apply_vhs_tape_path(s, luma_bw, chroma_bw, sample_rate, w == Wire::Luma);
            });
        }
        let chroma_cfg = self.chroma_path_config();
        if chroma_cfg.is_active() {
            let _p = profile::scope("fx chroma path");
            run(wires, rng, &chroma_on, &mut |s, _, rng| apply_chroma_path(s, &chroma_cfg, sample_rate, rng));
        }
        if let Some(gain) = self.edge_ringing {
            let _p = profile::scope("fx edge ringing");
            each(wires, &mut |s, w| {
                if luma_on(w) {
                    apply_edge_ringing(s, gain, sample_rate, w == Wire::Luma);
                }
            });
        }
        if let Some(strength) = self.head_switching_smear {
            let _p = profile::scope("fx head smear");
            run(wires, rng, &luma_on, &mut |s, w, rng| {
                apply_head_wear_smear(s, strength, rng, &clock, w == Wire::Luma)
            });
        }
        if let Some(amp) = self.luma_noise {
            let _p = profile::scope("fx luma noise");
            run(wires, rng, &luma_on, &mut |s, _, rng| apply_luma_noise(s, amp, sample_rate, rng));
        }
        if let Some(rate) = self.tape_dropout_rate {
            let _p = profile::scope("fx dropout");
            run(wires, rng, &all, &mut |s, w, rng| {
                apply_tape_dropout_doc(s, rate, self.tape_dropout_len, sample_rate, rng, self.doc, w)
            });
        }
        if let Some(amp) = self.wobble {
            let _p = profile::scope("fx wobble");
            run(wires, rng, &all, &mut |s, _, rng| {
                apply_wobble(s, amp * (1.0 - 0.95 * tbc), self.wobble_rate, rng, &clock)
            });
        }
        if let Some(amp) = self.v_jitter {
            let _p = profile::scope("fx v-jitter");
            run(wires, rng, &all, &mut |s, _, rng| apply_v_jitter(s, amp * (1.0 - ftbc), rng, &clock));
        }
        if let Some(strength) = self.top_band {
            let _p = profile::scope("fx top band");
            run(wires, rng, &chroma_on, &mut |s, _, rng| apply_top_band(s, strength, rng, &clock));
        }
        if let Some(strength) = self.head_switch {
            let _p = profile::scope("fx head switch");
            run(wires, rng, &all, &mut |s, _, rng| {
                apply_head_switch(s, strength, self.head_switch_lines, 1.0 - 0.75 * tbc, rng, &clock)
            });
        }
        // The deck's playback luma circuits act on everything the tape and
        // the heads produced.
        let deck_luma = self.deck_luma();
        if deck_luma.is_active() {
            let _p = profile::scope("fx deck luma (nc/ynr/peaking)");
            each(wires, &mut |s, w| {
                if luma_on(w) {
                    vhs::apply_deck_luma(s, &deck_luma, w == Wire::Luma);
                }
            });
        }
        if let Some(amp) = self.noise {
            let _p = profile::scope("fx noise");
            run(wires, rng, &all, &mut |s, _, rng| apply_noise(s, amp, rng));
        }
        if !self.ghosts.is_empty() {
            let _p = profile::scope("fx ghosting");
            run(wires, rng, &all, &mut |s, _, rng| apply_ghosting(s, &self.ghosts, sample_rate, rng));
        }
        if let Some(strength) = self.attenuation {
            let _p = profile::scope("fx attenuation");
            each(wires, &mut |s, _| apply_attenuation(s, strength));
        }
        if let Some(amp) = self.jitter {
            let _p = profile::scope("fx jitter");
            run(wires, rng, &all, &mut |s, _, rng| apply_jitter(s, (amp * (1.0 - 0.95 * tbc)).max(1e-6), rng));
        }
        // Tape trail is the very last step — the playback head's bandwidth
        // limit smears everything that's on the tape (noise, dropouts,
        // ghosts, etc.) rightward on readout.
        if let Some(strength) = self.tape_trail_smear {
            let _p = profile::scope("fx tape trail");
            each(wires, &mut |s, w| {
                if luma_on(w) {
                    apply_tape_trail_smear(s, strength, w == Wire::Luma);
                }
            });
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
pub(crate) fn apply_fir_inplace(buf: &mut [f32], kernel: &[f32], scratch: &mut Vec<f32>) {
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

    filters::convolve(scratch, kernel, buf);
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
    full_luma: bool,
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
        comb_split(&signal[base..base + spl], &mut luma, &mut chroma, full_luma);

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

/// Split one line into luma and chroma with the 2-sample comb.  On an
/// S-Video Y wire (`full_luma`) there is no chroma, so the whole signal is
/// luma: luma-domain effects then act on all of it.
pub(crate) fn comb_split(line: &[f32], luma: &mut [f32], chroma: &mut [f32], full_luma: bool) {
    if full_luma {
        luma.copy_from_slice(line);
        chroma.fill(0.0);
        return;
    }
    luma[0] = line[0];
    luma[1] = line[1];
    chroma[0] = 0.0;
    chroma[1] = 0.0;
    for n in 2..line.len() {
        let cur = line[n];
        let delayed = line[n - 2];
        luma[n] = (cur + delayed) * 0.5;
        chroma[n] = (cur - delayed) * 0.5;
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

/// What the deck's dropout compensator does with a long dropout.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Doc {
    /// Copies the whole signal of the line above, colour included.  The
    /// subcarrier of the line above is inverted, so the patch decodes in
    /// the complementary hue.  Kept for compatibility.
    #[default]
    Composite,
    /// Copies the luma of the line above and mutes the colour, as real
    /// decks do: a colourless patch.
    Luma,
    /// No compensator: every dropout is a white or black streak.
    Off,
}

/// Length of the raised-cosine ramps when colour is muted, in samples.
pub(crate) const MUTE_RAMP: usize = 4;

/// Mute the colour of `chroma[lo..hi]` with short ramps inside the
/// segment.
pub(crate) fn mute_segment(chroma: &mut [f32], lo: usize, hi: usize, rest: f32) {
    for j in lo..hi {
        let d = (j - lo).min(hi - 1 - j);
        let keep = if d < MUTE_RAMP { 0.5 + 0.5 * (std::f32::consts::PI * d as f32 / MUTE_RAMP as f32).cos() } else { 0.0 };
        chroma[j] = rest + keep * (chroma[j] - rest);
    }
}

/// `apply_tape_dropout` with a choice of compensator.  The dropouts are
/// placed exactly as there (same random draws); `Doc::Composite` is
/// `apply_tape_dropout` itself.
#[allow(clippy::too_many_arguments)]
fn apply_tape_dropout_doc(
    signal: &mut [f32],
    dropout_rate: f32,
    avg_len_us: f32,
    sample_rate: f64,
    rng: &mut impl Rng,
    doc: Doc,
    wire: Wire,
) {
    if doc == Doc::Composite {
        apply_tape_dropout(signal, dropout_rate, avg_len_us, sample_rate, rng);
        return;
    }
    let spl = SAMPLES_PER_LINE;
    let avg_len_samples = (avg_len_us as f64 * sample_rate / 1e6).max(4.0);
    let prob_per_line = (dropout_rate / TOTAL_LINES as f32).clamp(0.0, 1.0);
    let mut group = vhs::LaneGroup::new();

    for line in 0..TOTAL_LINES {
        if rng.random::<f32>() >= prob_per_line {
            continue;
        }
        let base = line * spl;
        if base + spl > signal.len() {
            break;
        }
        let offset = (rng.random::<f32>() * ACTIVE_SAMPLES as f32) as usize;
        let start_idx = base + ACTIVE_START + offset;
        let u: f32 = rng.random::<f32>().max(1e-10);
        let len = (-u.ln() * avg_len_samples as f32).max(4.0).min((ACTIVE_SAMPLES - offset) as f32) as usize;
        let end_idx = (start_idx + len).min(base + spl);
        let streak = len < 20 || doc == Doc::Off || line == 0;
        // A streak's level: mostly white, sometimes black (the carrier
        // lost after a bright area).
        let level = if streak {
            let r = rng.random::<f32>();
            if len >= 20 && r < 0.3 { BLANKING_V + 7.5 * IRE_V } else { BLANKING_V + 0.4 + r * 0.25 }
        } else {
            0.0
        };
        let (lo, hi) = (start_idx - base, end_idx - base);
        match wire {
            // Colour is muted over the whole dropout.
            Wire::Chroma => mute_segment(&mut signal[base..base + spl], lo, hi, BLANKING_V),
            Wire::Luma => {
                if streak {
                    signal[start_idx..end_idx].fill(level);
                } else {
                    signal.copy_within(start_idx - spl..end_idx - spl, start_idx);
                }
            }
            Wire::Composite => {
                if streak {
                    signal[start_idx..end_idx].fill(level);
                } else {
                    // Luma of the line above, colour muted.
                    group.load(signal, &[line - 1, line], false);
                    let r = vhs::REGION_START;
                    let mut chroma: Vec<f32> = (0..spl - r).map(|j| group.chroma[j * iir::LANES + 1]).collect();
                    mute_segment(&mut chroma, lo - r, hi - r, 0.0);
                    for j in lo..hi {
                        signal[base + j] = group.luma[(j - r) * iir::LANES] + chroma[j - r];
                    }
                }
            }
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
fn apply_edge_ringing(signal: &mut [f32], strength: f32, _sample_rate: f64, full_luma: bool) {
    let spl = SAMPLES_PER_LINE;
    if signal.len() < TOTAL_LINES * spl {
        return;
    }

    // The peaking filter extracts detail above ~1.5 MHz.  Using many taps
    // makes the ringing more pronounced (intentionally).
    let detail_lp = filters::design_lowpass(1.5e6, RINGING_FILTER_TAPS);

    let mut luma = vec![0.0f32; spl];
    let mut chroma = vec![0.0f32; spl];
    let mut luma_active = vec![0.0f32; ACTIVE_SAMPLES];
    let mut blurred_active = vec![0.0f32; ACTIVE_SAMPLES];
    let mut scratch = Vec::new();

    for line in 0..TOTAL_LINES {
        let base = line * spl;
        if base + spl > signal.len() {
            break;
        }

        // Separate luma/chroma so peaking does not boost chroma amplitude
        // (which appears as color oversaturation).
        comb_split(&signal[base..base + spl], &mut luma, &mut chroma, full_luma);

        // Apply peaking only on active picture luma.
        luma_active.copy_from_slice(&luma[ACTIVE_START..ACTIVE_START + ACTIVE_SAMPLES]);
        blurred_active.copy_from_slice(&luma_active);
        apply_fir_inplace(&mut blurred_active, &detail_lp, &mut scratch);

        // Unsharp-mask: output = luma + gain × (luma − lowpass(luma))
        for n in 0..ACTIVE_SAMPLES {
            let detail = luma_active[n] - blurred_active[n];
            luma[ACTIVE_START + n] = luma_active[n] + strength * detail;
        }

        // Recombine peaked luma with unchanged chroma.
        for n in 0..spl {
            signal[base + n] = luma[n] + chroma[n];
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
///   - The smeared regions shift from frame to frame: re-rolled each frame,
///     or drifting slowly with `--temporal-coherence` (see `TapeClock`).
///
/// The box blur is applied only to the **luma** component (separated via
/// the 2-sample comb) so the color subcarrier is preserved.
///
/// Implementation: a per-line running-average (box filter) whose kernel
/// width is modulated by a sum of low-frequency sinusoids across line
/// number, creating smooth bands of varying smear intensity.
fn apply_head_wear_smear(
    signal: &mut [f32],
    strength: f32,
    rng: &mut impl Rng,
    clock: &TapeClock,
    full_luma: bool,
) {
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
    let mut band_phase = |k: u64| {
        let r: f32 = rng.random::<f32>() * std::f32::consts::TAU;
        clock.blend(r, || clock.phase(KEY_SMEAR + k))
    };
    let phase1 = band_phase(0);
    let phase2 = band_phase(1);
    let phase3 = band_phase(2);

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
        comb_split(&signal[base..base + spl], &mut luma, &mut chroma, full_luma);

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
fn apply_tape_trail_smear(signal: &mut [f32], strength: f32, full_luma: bool) {
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
        comb_split(&signal[base..base + spl], &mut luma, &mut chroma, full_luma);

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

// ────────────────────────────────────────────────────────────────────────────
// Tape state: slowly drifting per-file randomness
// ────────────────────────────────────────────────────────────────────────────

/// SplitMix64 finaliser: spreads the bits of `x` over the whole word.
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Hash of (seed, key, knot) mapped to [−1, 1].
pub(crate) fn hash_signed(seed: u64, key: u64, knot: i64) -> f32 {
    let h = splitmix64(seed ^ splitmix64(key ^ splitmix64(knot as u64)));
    (h >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
}

/// Smooth 1-D value noise in [−1, 1]: random values at integer `t`,
/// joined with a smoothstep.  The same arguments always give the same
/// value, so any frame can be computed on its own.
pub(crate) fn drift(seed: u64, key: u64, t: f64) -> f32 {
    let k0 = t.floor();
    let f = (t - k0) as f32;
    let w = f * f * (3.0 - 2.0 * f);
    let a = hash_signed(seed, key, k0 as i64);
    let b = hash_signed(seed, key, k0 as i64 + 1);
    a + (b - a) * w
}

/// Frames per value-noise knot for slowly drifting tape state (~1.3 s).
const DRIFT_FRAMES: f64 = 40.0;

/// Keys that separate the drifting values of different effect parameters.
const KEY_SMEAR: u64 = 0x100;
const KEY_WOBBLE: u64 = 0x200;
const KEY_TOP_BAND: u64 = 0x300;
const KEY_HEAD_SWITCH: u64 = 0x400;

/// Per-frame view of the drifting "tape state".
///
/// Effects that keep a frame-wide random state (band phases, tear shape,
/// tint angle) draw it from the frame's RNG as before, then blend it toward
/// a smooth function of (tape seed, key, frame number) by `coherence`:
/// 0 keeps the old per-frame re-roll exactly, 1 makes the state drift
/// slowly from frame to frame like a real tape.  Because the smooth value
/// depends only on the frame number, frames can still be processed in
/// parallel without shared mutable state.
#[derive(Clone, Copy, Debug)]
pub struct TapeClock {
    pub seed: u64,
    pub frame: u64,
    pub coherence: f32,
}

impl TapeClock {
    /// A clock that leaves every per-frame draw unchanged.
    #[cfg(test)]
    pub fn off() -> Self {
        Self { seed: 0, frame: 0, coherence: 0.0 }
    }

    fn t(&self, scale: f64) -> f64 {
        self.frame as f64 / (DRIFT_FRAMES * scale)
    }

    /// Blend a per-frame draw `r` toward the drifting value `s`.  With zero
    /// coherence `r` is returned unchanged.
    fn blend(&self, r: f32, s: impl FnOnce() -> f32) -> f32 {
        if self.coherence <= 0.0 {
            r
        } else {
            r + self.coherence * (s() - r)
        }
    }

    /// Drifting value in [0, 1].
    fn unit(&self, key: u64) -> f32 {
        0.5 + 0.5 * drift(self.seed, key, self.t(1.0))
    }

    /// Drifting value with roughly unit standard deviation.
    fn normal(&self, key: u64) -> f32 {
        1.7 * drift(self.seed, key, self.t(1.0))
    }

    /// Drifting phase in radians: a fixed random start, a slow steady
    /// advance and a smooth wander.
    fn phase(&self, key: u64) -> f32 {
        let start = (hash_signed(self.seed, key, -1) + 1.0) * std::f32::consts::PI;
        start + 0.03 * self.frame as f32 + 2.0 * drift(self.seed, key ^ 0xF00D, self.t(1.5))
    }

    /// Drifting value in [−1, 1] along a free coordinate.
    fn along(&self, key: u64, x: f64) -> f32 {
        drift(self.seed, key, x)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Shared helpers for the field-timing effects
// ────────────────────────────────────────────────────────────────────────────

/// Voltage of one IRE unit (blanking = 0 IRE, sync tip = −40 IRE).
pub(crate) const IRE_V: f32 = BLANKING_V / 40.0;

/// First and one-past-last absolute line of the active picture in each field.
pub(crate) const FIELD_ACTIVE_LINES: [(usize, usize); 2] = [(20, 260), (283, 523)];

/// Draw a standard-normal sample.
pub(crate) fn randn(rng: &mut impl Rng) -> f32 {
    Normal::new(0.0f32, 1.0).unwrap().sample(rng)
}

/// Measure the colorburst phase of one line.
///
/// The burst is modelled as `A·cos(π/2·n + φ)` with `n` the sample index
/// within the line; the return value is `φ`.  Correlating against
/// `cos(π/2·n)` gives `A/2·cos φ` per sample, and against `sin(π/2·n)`
/// gives `−A/2·sin φ`.
pub(crate) fn measure_burst_phase(line: &[f32]) -> f32 {
    let cos_lut: [f32; 4] = [1.0, 0.0, -1.0, 0.0];
    let sin_lut: [f32; 4] = [0.0, 1.0, 0.0, -1.0];
    let mut c = 0.0f32;
    let mut s = 0.0f32;
    for n in BURST_START..BURST_START + BURST_SAMPLES {
        let v = line[n] - BLANKING_V;
        c += v * cos_lut[n & 3];
        s += v * sin_lut[n & 3];
    }
    (-s).atan2(c)
}

// ────────────────────────────────────────────────────────────────────────────
// Closed-caption (line 21) data
// ────────────────────────────────────────────────────────────────────────────

/// Closed-caption lines: line 21 of field 1 and line 284 of field 2.
const VBI_DATA_LINES: [usize; 2] = [21, 284];

/// Write a line-21 closed-caption waveform on lines 21 and 284.
///
/// EIA-608 data is sent at 32 × the line rate (≈ 503 kHz).  Each line
/// carries, starting 10.5 µs after the leading edge of horizontal sync:
///   - 7 cycles of clock run-in, a sine between 0 and 50 IRE,
///   - two 0 bits and a 1 start bit,
///   - two 7-bit characters with odd parity, sent LSB first as NRZ at
///     0 / 50 IRE.
///
/// These lines lie inside the picture area here, so the waveform shows as
/// a row of short light dashes near the top of the frame.  `strength`
/// blends it over the picture on those lines (1.0 = full replacement).
/// The characters are random printable ASCII, new for every frame.
fn apply_vbi_data(signal: &mut [f32], strength: f32, sample_rate: f64, rng: &mut impl Rng) {
    let spl = SAMPLES_PER_LINE;
    let strength = strength.clamp(0.0, 1.0);
    if strength <= 0.0 || signal.len() < TOTAL_LINES * spl {
        return;
    }

    let line_rate = 2.0 * FSC / 455.0;
    let bit_samples = (sample_rate / (32.0 * line_rate)) as f32; // ≈ 28.4
    let start = FRONT_PORCH_SAMPLES + (10.5e-6 * sample_rate).round() as usize;
    let high_v = 50.0 * IRE_V;

    // The 1.5 MHz lowpass gives the data edges a finite rise time.
    let edge_lp = filters::design_lowpass(1.5e6, 31);
    let mut wave = vec![0.0f32; ACTIVE_SAMPLES];
    let mut scratch = Vec::new();

    for &line in &VBI_DATA_LINES {
        // Bit sequence after the run-in: 0, 0, start 1, then two bytes.
        let mut bits: Vec<bool> = vec![false, false, true];
        for _ in 0..2 {
            let ch: u8 = rng.random_range(0x20u8..0x7F);
            let parity = ch.count_ones() % 2 == 0; // makes the total odd
            for b in 0..7 {
                bits.push((ch >> b) & 1 == 1);
            }
            bits.push(parity);
        }

        for (j, w) in wave.iter_mut().enumerate() {
            let n = ACTIVE_START + j;
            let level = if n < start {
                0.0
            } else {
                let t = (n - start) as f32 / bit_samples;
                if t < 7.0 {
                    0.5 * high_v * (1.0 - (std::f32::consts::TAU * t).cos())
                } else {
                    match bits.get((t - 7.0) as usize) {
                        Some(true) => high_v,
                        _ => 0.0,
                    }
                }
            };
            *w = BLANKING_V + level;
        }
        apply_fir_inplace(&mut wave, &edge_lp, &mut scratch);

        let base = line * spl + ACTIVE_START;
        for (j, &w) in wave.iter().enumerate() {
            let s = &mut signal[base + j];
            *s += strength * (w - *s);
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Top-of-frame chroma band
// ────────────────────────────────────────────────────────────────────────────

/// Simulate the blue/purple chroma bands at the top of a VHS picture.
///
/// After each vertical interval the deck's chroma phase loop needs a few
/// lines to settle, so the first active lines of each field carry a hue
/// error and a slight blue/purple cast.  For the first `M` active lines of
/// each field (M = 3–6, growing with `strength`):
///   1. Chroma is separated with the 2-sample comb and demodulated to
///      baseband with the 4×fsc LUTs.
///   2. The baseband vector is rotated by a random angle, and a bias in the
///      −I/+Q direction (blue-violet) is added, in the line's burst frame.
///   3. Only the difference is remodulated and added back, so luma and the
///      untouched part of the chroma are preserved exactly.
///   4. A slight luma lift with low-frequency streak noise is added.
///
/// All of it decays with the line index.  Angles, bias and decay are random
/// for every frame and field.
fn apply_top_band(signal: &mut [f32], strength: f32, rng: &mut impl Rng, clock: &TapeClock) {
    let spl = SAMPLES_PER_LINE;
    let strength = strength.clamp(0.0, 1.0);
    if strength <= 0.0 || signal.len() < TOTAL_LINES * spl {
        return;
    }

    let band_lines = (3.0 + 3.0 * strength).round() as usize;

    let cos_lut: [f32; 4] = [1.0, 0.0, -1.0, 0.0];
    let nsin_lut: [f32; 4] = [0.0, -1.0, 0.0, 1.0];

    // Baseband lowpass: keeps the hue rotation from acting on luma detail
    // that leaks into the comb's chroma output.
    let bb_lp = filters::design_lowpass(1.3e6, 31);
    let streak_lp = filters::design_lowpass(0.4e6, 61);

    let mut i_bb = vec![0.0f32; ACTIVE_SAMPLES];
    let mut q_bb = vec![0.0f32; ACTIVE_SAMPLES];
    let mut streak = vec![0.0f32; ACTIVE_SAMPLES];
    let mut scratch = Vec::new();

    for (field, &(first, _)) in FIELD_ACTIVE_LINES.iter().enumerate() {
        // Per-field random state (drifts with the tape clock).
        let key = KEY_TOP_BAND + 8 * field as u64;
        let r = randn(rng);
        let rot0 = clock.blend(r * 0.35 * strength + 0.25 * strength, || {
            clock.normal(key) * 0.35 * strength + 0.25 * strength
        });
        let r = randn(rng);
        let hue = clock
            .blend(105.0f32 + 12.0 * r, || 105.0 + 12.0 * clock.normal(key + 1))
            .to_radians();
        let r = rng.random::<f32>();
        let bias = clock.blend(strength * 0.05 * (0.7 + 0.6 * r), || {
            strength * 0.05 * (0.7 + 0.6 * clock.unit(key + 2))
        });
        let r = rng.random::<f32>();
        let lift = clock.blend(strength * 0.02 * (0.5 + r), || {
            strength * 0.02 * (0.5 + clock.unit(key + 3))
        });

        for k in 0..band_lines {
            let line = first + k;
            let base = line * spl;
            let decay = (1.0 - k as f32 / band_lines as f32).powf(1.5);
            if decay <= 0.0 {
                continue;
            }

            // ── Demodulate the comb chroma of the active region ──
            for j in 0..ACTIVE_SAMPLES {
                let n = ACTIVE_START + j;
                let chroma = (signal[base + n] - signal[base + n - 2]) * 0.5;
                i_bb[j] = 2.0 * chroma * cos_lut[n & 3];
                q_bb[j] = 2.0 * chroma * nsin_lut[n & 3];
            }
            apply_fir_inplace(&mut i_bb, &bb_lp, &mut scratch);
            apply_fir_inplace(&mut q_bb, &bb_lp, &mut scratch);

            // ── Bias vector in this line's burst frame ──
            // The I carrier sits at burst phase − π + 123°, Q at
            // burst phase − π + 33°.  The bias is (cos hue)·I + (sin hue)·Q.
            let phi_b = measure_burst_phase(&signal[base..base + spl]);
            let th_i = phi_b - std::f32::consts::PI + I_PHASE_RAD;
            let th_q = phi_b - std::f32::consts::PI + Q_PHASE_RAD;
            let b = bias * decay * COMPOSITE_SCALE;
            let (bi, bq) = (b * hue.cos(), b * hue.sin());
            let bias_i = bi * th_i.cos() + bq * th_q.cos();
            let bias_q = bi * th_i.sin() + bq * th_q.sin();

            let rot = rot0 * decay;
            let (rc, rs) = (rot.cos(), rot.sin());

            // ── Luma streaks: slow horizontal noise riding on the lift ──
            for s in streak.iter_mut() {
                *s = randn(rng);
            }
            apply_fir_inplace(&mut streak, &streak_lp, &mut scratch);

            // ── Add the remodulated difference and the luma lift ──
            for j in 0..ACTIVE_SAMPLES {
                let n = ACTIVE_START + j;
                let (i0, q0) = (i_bb[j], q_bb[j]);
                let di = i0 * rc - q0 * rs - i0 + bias_i;
                let dq = i0 * rs + q0 * rc - q0 + bias_q;
                let d_chroma = di * cos_lut[n & 3] + dq * nsin_lut[n & 3];
                let d_luma = decay * (lift + strength * 0.03 * streak[j]);
                signal[base + n] += d_chroma + d_luma;
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Head-switching noise
// ────────────────────────────────────────────────────────────────────────────

/// Simulate VHS head-switching noise at the bottom of each field.
///
/// The head drum swaps heads a few lines before vertical sync, so the last
/// two or three active lines of each field are read after the switch:
///   - **Switch line.**  At a random point along the line the signal
///     changes over to the other head.  Left of it the picture is normal;
///     right of it the picture is displaced by a time-base jump of tens of
///     samples (the heads are never perfectly aligned) and its subcarrier
///     no longer matches the line's burst, so colours there are off.  The
///     changeover itself is a short switching transient (a spike and
///     wideband noise lasting a few µs).  Parts of the displaced segment
///     drop out to black, and wide, soft, washed-out light smears lie
///     across it.
///   - **Lines after the switch.**  The head has no proper track here, so
///     the signal is mostly a low, noisy level near black, with sparse
///     bright streaks of tape noise (some with a random subcarrier phase,
///     so they come out green or purple) and now and then a dimmed,
///     displaced piece of the picture.
///
/// Over time the switch behaves as on a real tape: the switch point, the
/// jump and the number of torn lines (`lines` at full strength, a little
/// fewer at low strength) are a fixed property of the tape (its seed),
/// with a few samples of flicker every field, freshly drawn noise and
/// streaks, and now and then a sudden change that holds for a few frames
/// or seconds (`head_switch_hiccup`).  With zero `coherence` the state is
/// re-rolled every frame instead.  `timing_scale` (< 1 with a line TBC)
/// scales the jump, not the noise.  Sync and burst are left alone.
fn apply_head_switch<R: Rng>(
    signal: &mut [f32],
    strength: f32,
    lines: usize,
    timing_scale: f32,
    rng: &mut R,
    clock: &TapeClock,
) {
    let spl = SAMPLES_PER_LINE;
    let strength = strength.clamp(0.0, 1.0);
    if strength <= 0.0 || lines == 0 || signal.len() < TOTAL_LINES * spl {
        return;
    }
    let act = ACTIVE_SAMPLES;
    let black = BLANKING_V + 7.5 * IRE_V;
    let noise_lp = filters::design_lowpass(3.0e6, 15);
    let mut noise = vec![0.0f32; act];
    let mut moved = vec![0.0f32; spl];
    let mut scratch = Vec::new();
    // Band-limited noise of standard deviation about `sigma` (in volts).
    let mut fill_noise = |noise: &mut [f32], sigma: f32, rng: &mut R| {
        for s in noise.iter_mut() {
            *s = sigma * 1.6 * randn(rng);
        }
        apply_fir_inplace(noise, &noise_lp, &mut scratch);
    };

    for (field, &(first, end)) in FIELD_ACTIVE_LINES.iter().enumerate() {
        let key = KEY_HEAD_SWITCH + 8 * field as u64;
        // The tape's own switch point, jump size and number of torn
        // lines, unless a hiccup is under way.
        let tape = |k: u64| 0.5 + 0.5 * hash_signed(clock.seed, key + k, -7);
        let (mut at_t, mut jump_t, mut extra) = (tape(0), tape(1), 0.0f32);
        if let Some(h) = head_switch_hiccup(clock, key + 3, strength) {
            at_t = (at_t + 0.3 * h[0]).clamp(0.0, 1.0);
            jump_t = (jump_t + 0.6 * h[1]).clamp(0.0, 1.0);
            extra = (1.4 * h[2]).round();
        }
        let r = rng.random::<f32>();
        let at_frac = 0.15 + 0.6 * clock.blend(r, || at_t);
        let r = rng.random::<f32>();
        let jump_size = 15.0 + 35.0 * clock.blend(r, || jump_t);
        let r = rng.random::<f32>();
        let up = if clock.coherence > 0.0 { tape(2) } else { r };
        let sign = if up < 0.75 { 1.0 } else { -1.0 };
        let jump = sign * strength * (jump_size + 1.5 * randn(rng)) * timing_scale;
        let lines_here = lines as f32 * (0.5 + 0.5 * strength) + clock.blend(0.0, || extra);
        let n_lines = (lines_here.round().max(1.0) as usize).min(end - first);
        let switch_line = end - n_lines;

        // ── Switch line ──
        let base = switch_line * spl;
        let x_s = ((at_frac * act as f32 + 3.0 * randn(rng)).max(0.0) as usize).min(act - 1);
        moved.copy_from_slice(&signal[base..base + spl]);
        shift_segment(&mut moved, ACTIVE_START, spl, jump);
        let row = &mut signal[base + ACTIVE_START..base + spl];
        row[x_s..].copy_from_slice(&moved[ACTIVE_START + x_s..]);
        // Dropouts to black in the displaced part.
        let n_dark = (2.0 * strength * rng.random::<f32>()).round() as usize;
        for _ in 0..n_dark {
            let len = 20 + (rng.random::<f32>() * 120.0) as usize;
            let start = x_s + (rng.random::<f32>() * (act - x_s) as f32) as usize;
            let mix = 0.6 + 0.35 * rng.random::<f32>();
            blend_segment(row, start, len, 6.0, |_| black, mix);
        }
        // Washed-out light smears: wide, soft-edged and nearly colourless,
        // mostly in the displaced part.
        let n_light = 3 + (5.0 * strength * rng.random::<f32>()) as usize;
        for _ in 0..n_light {
            let len = 60 + (rng.random::<f32>() * 200.0) as usize;
            let from = if rng.random::<f32>() < 0.8 { x_s } else { 0 };
            let start = from + (rng.random::<f32>() * (act - from) as f32) as usize;
            let level = black + (70.0 + 25.0 * rng.random::<f32>()) * IRE_V;
            let mix = 0.8 + 0.2 * rng.random::<f32>();
            blend_segment(row, start, len, 24.0, |_| level, mix);
        }
        // Switching transient: a spike that decays over a few µs, with noise.
        fill_noise(&mut noise, 8.0 * IRE_V * strength, rng);
        let spike = (40.0 + 30.0 * rng.random::<f32>()) * IRE_V * strength;
        let t_len = 25 + (rng.random::<f32>() * 35.0) as usize;
        for j in x_s..(x_s + t_len).min(act) {
            let t = (j - x_s) as f32 / t_len as f32;
            row[j] += (spike * (-4.0 * t).exp() + noise[j]) * (1.0 - t);
        }

        // ── Lines after the switch: signal lost ──
        for k in 1..n_lines {
            let base = (switch_line + k) * spl;
            moved.copy_from_slice(&signal[base..base + spl]);
            let offset = jump * (1.0 + 0.5 * randn(rng)) + 20.0 * randn(rng);
            shift_segment(&mut moved, ACTIVE_START, spl, offset);
            fill_noise(&mut noise, 2.5 * IRE_V, rng);
            let lift = 2.0 * IRE_V * rng.random::<f32>();
            let row = &mut signal[base + ACTIVE_START..base + spl];
            for (s, &nz) in row.iter_mut().zip(&noise) {
                *s = black + lift + nz;
            }
            // Now and then a dimmed, displaced piece of the picture.
            if rng.random::<f32>() < 0.5 * strength {
                let len = 60 + (rng.random::<f32>() * 200.0) as usize;
                let start = (rng.random::<f32>() * act as f32) as usize;
                let dim = 0.5 + 0.4 * rng.random::<f32>();
                let src = &moved[ACTIVE_START..];
                blend_segment(row, start, len, 8.0, |j| black + dim * (src[j] - black), 1.0);
            }
            // Sparse bright streaks of tape noise, some carrying a
            // subcarrier at a random phase.
            let n_streaks = (strength * (2.0 + 4.0 * rng.random::<f32>())).round() as usize;
            for _ in 0..n_streaks {
                let len = 8 + (rng.random::<f32>() * 45.0) as usize;
                let start = (rng.random::<f32>() * act as f32) as usize;
                let luma = black + (30.0 + 50.0 * rng.random::<f32>()) * IRE_V;
                let chroma = if rng.random::<f32>() < 0.6 { (8.0 + 17.0 * rng.random::<f32>()) * IRE_V } else { 0.0 };
                let phase = std::f32::consts::TAU * rng.random::<f32>();
                let carrier = |j: usize| {
                    let n = (ACTIVE_START + j) as f32;
                    luma + chroma * (std::f32::consts::FRAC_PI_2 * n + phase).cos()
                };
                let mix = 0.7 + 0.3 * rng.random::<f32>();
                blend_segment(row, start, len, 4.0, carrier, mix);
            }
        }
    }
}

/// Blend `row[start..start + len]` (clipped to the row) toward `level(j)`
/// by `mix`, with raised-cosine edges `ramp` samples long.
pub(crate) fn blend_segment(row: &mut [f32], start: usize, len: usize, ramp: f32, level: impl Fn(usize) -> f32, mix: f32) {
    let end = (start + len).min(row.len());
    for j in start.min(end)..end {
        let d = (j - start).min(end - 1 - j) as f32;
        let env = if d < ramp { 0.5 - 0.5 * (std::f32::consts::PI * d / ramp).cos() } else { 1.0 };
        row[j] += env * mix * (level(j) - row[j]);
    }
}

/// Frames per slot in which a head-switch hiccup may start.
const HICCUP_SLOT: u64 = 16;

/// A sudden change of the head-switch state ("tracking hiccup"): `None`
/// most of the time, or three offsets in [−1, 1] while one is under way.
///
/// Hiccups start on average every 6 s / `strength` (at 30 fps).  Most last
/// 2–8 frames and snap back; some hold for 1–3 s.  It is a pure function
/// of the tape seed, `key` and the frame number.
fn head_switch_hiccup(clock: &TapeClock, key: u64, strength: f32) -> Option<[f32; 3]> {
    let unit = |k: u64, slot: i64| 0.5 + 0.5 * hash_signed(clock.seed, key + k, slot);
    let p_slot = (strength as f64 * HICCUP_SLOT as f64 / 180.0).min(0.9) as f32;
    let slot = (clock.frame / HICCUP_SLOT) as i64;
    // The longest hiccup spans 6 slots; the latest one under way wins.
    for s in (slot - 6..=slot).rev().filter(|&s| s >= 0) {
        if unit(0, s) >= p_slot {
            continue;
        }
        let start = s as u64 * HICCUP_SLOT + (unit(1, s) * HICCUP_SLOT as f32) as u64;
        let len = if unit(2, s) < 0.7 { 2 + (unit(3, s) * 7.0) as u64 } else { 30 + (unit(3, s) * 60.0) as u64 };
        if (start..start + len).contains(&clock.frame) {
            return Some([hash_signed(clock.seed, key + 4, s), hash_signed(clock.seed, key + 5, s), hash_signed(clock.seed, key + 6, s)]);
        }
    }
    None
}

// ────────────────────────────────────────────────────────────────────────────
// Playback chroma path (colour-under noise, phase jitter)
// ────────────────────────────────────────────────────────────────────────────

/// Settings for `apply_chroma_path`.  A zero value turns a stage off.
#[derive(Clone, Copy, Debug, Default)]
pub struct ChromaPathConfig {
    /// Colour-under noise: std dev of the I/Q noise in YIQ units.
    pub noise: f32,
    /// Per-line chroma phase jitter: std dev in degrees.
    pub phase_noise_deg: f32,
    /// Playback 1H chroma comb strength (0–1).  At 1 each line's chroma is
    /// the average of itself and the previous line of the field.
    pub vblur: f32,
    /// Horizontal chroma delay relative to luma, in samples (positive =
    /// right).
    pub delay_samples: f32,
    /// Vertical chroma delay in field lines (positive = down).
    pub delay_lines: usize,
}

impl ChromaPathConfig {
    fn is_active(&self) -> bool {
        self.noise > 0.0
            || self.phase_noise_deg > 0.0
            || self.vblur > 0.0
            || self.delay_samples != 0.0
            || self.delay_lines > 0
    }
}

/// −3 dB cutoff of the colour-under noise.  Well under the colour-under
/// bandwidth, so the noise forms blotches a few to tens of pixels long.
const CHROMA_NOISE_CUTOFF_HZ: f64 = 150e3;

/// Correlation of the chroma noise with the previous line of the field.
const CHROMA_NOISE_LINE_CORR: f32 = 0.5;

/// Zero-phase lowpass: a one-pole IIR run forward, then backward.
fn lowpass_zero_phase(buf: &mut [f32], alpha: f32) {
    if buf.is_empty() {
        return;
    }
    let mut prev = buf[0];
    for s in buf.iter_mut() {
        prev += alpha * (*s - prev);
        *s = prev;
    }
    let mut prev = buf[buf.len() - 1];
    for s in buf.iter_mut().rev() {
        prev += alpha * (*s - prev);
        *s = prev;
    }
}

/// Lines of colour-under noise generated and filtered together.
const NOISE_LANES: usize = 8;

multiversion! {
    /// `lowpass_zero_phase` on `NOISE_LANES` interleaved signals at once
    /// (`buf[j * NOISE_LANES + l]` is sample `j` of signal `l`), for both
    /// `a` and `b`.  Each signal gets exactly the operations of
    /// `lowpass_zero_phase`.
    fn lowpass_zero_phase_lanes(a: &mut [f32], b: &mut [f32], alpha: f32) {
        const L: usize = NOISE_LANES;
        let len = a.len() / L;
        if len == 0 {
            return;
        }
        let first = |v: &[f32], j: usize| -> [f32; L] { v[j * L..(j + 1) * L].try_into().unwrap() };
        let (mut pa, mut pb) = (first(a, 0), first(b, 0));
        for (sa, sb) in a.chunks_exact_mut(L).zip(b.chunks_exact_mut(L)) {
            for l in 0..L {
                pa[l] += alpha * (sa[l] - pa[l]);
                sa[l] = pa[l];
                pb[l] += alpha * (sb[l] - pb[l]);
                sb[l] = pb[l];
            }
        }
        let (mut pa, mut pb) = (first(a, len - 1), first(b, len - 1));
        for (sa, sb) in a.chunks_exact_mut(L).zip(b.chunks_exact_mut(L)).rev() {
            for l in 0..L {
                pa[l] += alpha * (sa[l] - pa[l]);
                sa[l] = pa[l];
                pb[l] += alpha * (sb[l] - pb[l]);
                sb[l] = pb[l];
            }
        }
    }
}

/// Simulate the VHS playback chroma path in demodulated I/Q baseband.
///
/// For every active line the comb-separated chroma is demodulated with the
/// 4×fsc LUTs, lowpassed, and rotated into the frame of the line's own
/// colorburst.  In that frame a given colour has the same I/Q vector on
/// every line, even though the subcarrier inverts from one line to the
/// next, so lines can be compared and mixed directly.  The stages are:
///   1. **Colour-under noise** — low-frequency noise added to I and Q.  It
///      is lowpassed far below the colour-under bandwidth and partly
///      correlated with the previous line of the field, so it shows as
///      horizontal colour blotches rather than single-line stripes.
///   2. **Phase jitter** — each line's I/Q vector is rotated by a random
///      angle, the hue flicker caused by time-base errors in the
///      colour-under heterodyne.
///   3. **1H comb** — many decks pass the playback chroma through a
///      one-line delay comb to cancel crosstalk from the neighbouring
///      track.  Each line's chroma becomes `(1−k)·c[n] + k·c[n−1]` with
///      `c[n−1]` the previous line of the same field and `k = 0.5` at full
///      strength, which halves vertical chroma resolution.
///   4. **Chroma delay** — the chroma path is slower than the luma path, so
///      colour lands a little to the right of (and in some decks a line
///      below) the luma.  The I/Q baseband is shifted by a fractional
///      number of samples (linear interpolation, edge value held) and by
///      whole lines within the field (first line held).
///
/// Only the change of the baseband chroma is remodulated and added back,
/// so luma and the high-frequency part of the comb's chroma output are
/// preserved exactly.
fn apply_chroma_path(
    signal: &mut [f32],
    cfg: &ChromaPathConfig,
    sample_rate: f64,
    rng: &mut impl Rng,
) {
    let spl = SAMPLES_PER_LINE;
    if !cfg.is_active() || signal.len() < TOTAL_LINES * spl {
        return;
    }

    let cos_lut: [f32; 4] = [1.0, 0.0, -1.0, 0.0];
    let nsin_lut: [f32; 4] = [0.0, -1.0, 0.0, 1.0];
    let bb_lp = filters::design_lowpass(1.3e6, 31);
    let mut scratch = Vec::new();

    // Noise filter and the gain that restores unit variance after it.
    let noise_alpha =
        1.0 - (-2.0 * std::f64::consts::PI * CHROMA_NOISE_CUTOFF_HZ / sample_rate).exp() as f32;
    let noise_gain = {
        let mut h = vec![0.0f32; 4096];
        h[2048] = 1.0;
        lowpass_zero_phase(&mut h, noise_alpha);
        1.0 / h.iter().map(|v| v * v).sum::<f32>().sqrt()
    };
    let line_corr = CHROMA_NOISE_LINE_CORR;
    let fresh = (1.0 - line_corr * line_corr).sqrt();

    let act = ACTIVE_SAMPLES;
    // The noise is generated with a margin on both sides so the filter has
    // settled by the time it reaches the picture.
    let pad = 256;
    // Noise for `NOISE_LANES` lines at a time, interleaved by line.
    let lanes = NOISE_LANES;
    let mut noise_i = vec![0.0f32; (act + 2 * pad) * lanes];
    let mut noise_q = vec![0.0f32; (act + 2 * pad) * lanes];
    let mut prev_i = vec![0.0f32; act];
    let mut prev_q = vec![0.0f32; act];
    let mut tmp_i = vec![0.0f32; act];
    let mut tmp_q = vec![0.0f32; act];

    for &(first, end) in &FIELD_ACTIVE_LINES {
        let n_lines = end - first;

        // ── Demodulate every line of the field into its burst frame ──
        let mut orig_i = vec![0.0f32; n_lines * act];
        let mut orig_q = vec![0.0f32; n_lines * act];
        let mut burst = vec![(1.0f32, 0.0f32); n_lines];
        for k in 0..n_lines {
            let base = (first + k) * spl;
            for j in 0..act {
                let n = ACTIVE_START + j;
                let chroma = (signal[base + n] - signal[base + n - 2]) * 0.5;
                tmp_i[j] = 2.0 * chroma * cos_lut[n & 3];
                tmp_q[j] = 2.0 * chroma * nsin_lut[n & 3];
            }
            apply_fir_inplace(&mut tmp_i, &bb_lp, &mut scratch);
            apply_fir_inplace(&mut tmp_q, &bb_lp, &mut scratch);
            let phi = measure_burst_phase(&signal[base..base + spl]);
            let (c, s) = (phi.cos(), phi.sin());
            burst[k] = (c, s);
            // Rotate by −φ into the burst frame.
            let row = k * act;
            for j in 0..act {
                orig_i[row + j] = c * tmp_i[j] + s * tmp_q[j];
                orig_q[row + j] = -s * tmp_i[j] + c * tmp_q[j];
            }
        }
        let mut new_i = orig_i.clone();
        let mut new_q = orig_q.clone();

        // ── 1. Colour-under noise ──
        if cfg.noise > 0.0 {
            let amp = cfg.noise * COMPOSITE_SCALE * noise_gain;
            for k0 in (0..n_lines).step_by(lanes) {
                let block = lanes.min(n_lines - k0);
                for l in 0..block {
                    for j in 0..act + 2 * pad {
                        noise_i[j * lanes + l] = randn(rng);
                        noise_q[j * lanes + l] = randn(rng);
                    }
                }
                lowpass_zero_phase_lanes(&mut noise_i, &mut noise_q, noise_alpha);
                for l in 0..block {
                    let k = k0 + l;
                    for j in 0..act {
                        let at = (pad + j) * lanes + l;
                        let (mut ni, mut nq) = (noise_i[at], noise_q[at]);
                        if k > 0 {
                            ni = line_corr * prev_i[j] + fresh * ni;
                            nq = line_corr * prev_q[j] + fresh * nq;
                        }
                        prev_i[j] = ni;
                        prev_q[j] = nq;
                        new_i[k * act + j] += amp * ni;
                        new_q[k * act + j] += amp * nq;
                    }
                }
            }
        }

        // ── 2. Phase jitter ──
        if cfg.phase_noise_deg > 0.0 {
            let sigma = cfg.phase_noise_deg.to_radians();
            for k in 0..n_lines {
                let theta = sigma * randn(rng);
                let (c, s) = (theta.cos(), theta.sin());
                for j in k * act..(k + 1) * act {
                    let (i, q) = (new_i[j], new_q[j]);
                    new_i[j] = c * i - s * q;
                    new_q[j] = s * i + c * q;
                }
            }
        }

        // ── 3. Playback 1H comb ──
        // Lines are mixed bottom-up so each line reads its predecessor
        // before that line is itself mixed.
        if cfg.vblur > 0.0 {
            let k = 0.5 * cfg.vblur.min(1.0);
            for line in (1..n_lines).rev() {
                for j in 0..act {
                    let (cur, prev) = (line * act + j, (line - 1) * act + j);
                    new_i[cur] = (1.0 - k) * new_i[cur] + k * new_i[prev];
                    new_q[cur] = (1.0 - k) * new_q[cur] + k * new_q[prev];
                }
            }
        }

        // ── 4. Chroma delay ──
        if cfg.delay_lines > 0 {
            let v = cfg.delay_lines;
            for line in (0..n_lines).rev() {
                let src = line.saturating_sub(v);
                for j in 0..act {
                    new_i[line * act + j] = new_i[src * act + j];
                    new_q[line * act + j] = new_q[src * act + j];
                }
            }
        }
        if cfg.delay_samples != 0.0 {
            let d = cfg.delay_samples;
            let whole = d.floor() as isize;
            let frac = d - whole as f32;
            let at = |row: &[f32], i: isize| row[i.clamp(0, act as isize - 1) as usize];
            for line in 0..n_lines {
                let r = line * act..(line + 1) * act;
                tmp_i.copy_from_slice(&new_i[r.clone()]);
                tmp_q.copy_from_slice(&new_q[r.clone()]);
                for j in 0..act {
                    let i = j as isize - whole;
                    new_i[r.start + j] = (1.0 - frac) * at(&tmp_i, i) + frac * at(&tmp_i, i - 1);
                    new_q[r.start + j] = (1.0 - frac) * at(&tmp_q, i) + frac * at(&tmp_q, i - 1);
                }
            }
        }

        // ── Remodulate the change and add it back ──
        for k in 0..n_lines {
            let base = (first + k) * spl;
            let (c, s) = burst[k];
            for j in 0..act {
                let idx = k * act + j;
                let di_b = new_i[idx] - orig_i[idx];
                let dq_b = new_q[idx] - orig_q[idx];
                // Rotate by +φ back into the LUT frame.
                let di = c * di_b - s * dq_b;
                let dq = s * di_b + c * dq_b;
                let n = ACTIVE_START + j;
                signal[base + n] += di * cos_lut[n & 3] + dq * nsin_lut[n & 3];
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Time-base wobble
// ────────────────────────────────────────────────────────────────────────────

/// Periods (in lines) and weights of the sinusoidal wobble components.
const WOBBLE_COMPONENTS: [((f32, f32), f32); 3] =
    [((20.0, 45.0), 0.2), ((60.0, 140.0), 0.3), ((150.0, 300.0), 0.5)];

/// Weight and line-to-line correlation of the random-walk component.
const WOBBLE_WALK_WEIGHT: f32 = 0.3;
const WOBBLE_WALK_CORR: f32 = 0.97;

/// Simulate correlated time-base error (wobble) from tape stretch, capstan
/// flutter and head-drum speed variation.
///
/// Unlike `apply_jitter`, which moves lines independently by whole
/// subcarrier cycles, the displacement here is smooth from line to line:
///   - three sinusoids across the lines of the frame, with periods of about
///     20–45, 60–140 and 150–300 lines (scaled by `rate`), random phases,
///     and weights favouring the slow components;
///   - a mean-reverting random walk (an AR(1) process with a correlation of
///     0.97 per line) for irregular drift.
///
/// Lines are taken in time order (absolute line number), so the wobble
/// continues from field 1 into field 2.  Each visible line is shifted by a
/// fractional number of samples together with its burst, so the decoder
/// stays locked to the chroma and hue is preserved, as on a real TV that
/// follows the time-base error of each line.
///
/// `amp_px` is the peak displacement in output pixels at 640 wide.  With
/// `--temporal-coherence` the periods, phases and walk drift slowly from
/// frame to frame instead of being re-rolled.
fn apply_wobble(
    signal: &mut [f32],
    amp_px: f32,
    rate: f32,
    rng: &mut impl Rng,
    clock: &TapeClock,
) {
    let spl = SAMPLES_PER_LINE;
    if amp_px == 0.0 || signal.len() < TOTAL_LINES * spl {
        return;
    }
    let amp = amp_px * ACTIVE_SAMPLES as f32 / 640.0;

    let mut comps = [(0.0f32, 0.0f32, 0.0f32); 3];
    for (k, (c, &((lo, hi), weight))) in comps.iter_mut().zip(&WOBBLE_COMPONENTS).enumerate() {
        let key = KEY_WOBBLE + 4 * k as u64;
        let r = rng.random::<f32>();
        let period = clock.blend(lo + (hi - lo) * r, || lo + (hi - lo) * clock.unit(key));
        let r = rng.random::<f32>() * std::f32::consts::TAU;
        let phase = clock.blend(r, || clock.phase(key + 1));
        *c = (std::f32::consts::TAU * rate / period, phase, weight);
    }

    let fresh = (1.0 - WOBBLE_WALK_CORR * WOBBLE_WALK_CORR).sqrt();
    let mut walk = randn(rng);
    let burst_lo = FRONT_PORCH_SAMPLES + HSYNC_SAMPLES;
    let norm = 1.0 / (1.0 + WOBBLE_WALK_WEIGHT);

    for line in 0..TOTAL_LINES {
        walk = WOBBLE_WALK_CORR * walk + fresh * randn(rng);
        let visible = FIELD_ACTIVE_LINES.iter().any(|&(a, b)| (a..b).contains(&line));
        if !visible {
            continue;
        }
        let t = line as f32;
        // With coherence the walk becomes smooth noise along the lines
        // that scrolls slowly from frame to frame.
        let w = clock.blend(walk, || {
            1.7 * clock.along(KEY_WOBBLE + 16, line as f64 / 30.0 + 0.02 * clock.frame as f64)
        });
        let mut d = WOBBLE_WALK_WEIGHT * w;
        for &(w, phase, weight) in &comps {
            d += weight * (w * t + phase).sin();
        }
        let base = line * spl;
        shift_segment(&mut signal[base..base + spl], burst_lo, spl, amp * norm * d);
    }
}

/// Half-width of the fractional-delay kernel used by `shift_segment`.
const SHIFT_HALF_TAPS: isize = 8;

// ────────────────────────────────────────────────────────────────────────────
// Vertical bounce
// ────────────────────────────────────────────────────────────────────────────

const KEY_V_JITTER: u64 = 0x500;

/// Simulate vertical bounce: the whole picture jumps up or down from frame
/// to frame (tape tension and servo errors), with a little extra
/// difference between the two fields.
///
/// Lines are moved by whole field lines, burst included, because blending
/// neighbouring composite lines would cancel their (inverted) chroma.
/// Lines moved in from outside the picture are blank (black).
/// `amp_rows` is the standard deviation in output rows.
fn apply_v_jitter(signal: &mut [f32], amp_rows: f32, rng: &mut impl Rng, clock: &TapeClock) {
    let spl = SAMPLES_PER_LINE;
    if amp_rows <= 0.0 || signal.len() < TOTAL_LINES * spl {
        return;
    }
    let r = randn(rng);
    let frame_bounce = clock.blend(r, || clock.normal(KEY_V_JITTER));
    for (field, &(first, end)) in FIELD_ACTIVE_LINES.iter().enumerate() {
        let r = randn(rng);
        let field_delta = clock.blend(r, || clock.normal(KEY_V_JITTER + 1 + field as u64));
        // One field line is two output rows.
        let shift = ((frame_bounce + 0.2 * field_delta) * amp_rows / 2.0).round() as isize;
        if shift == 0 {
            continue;
        }
        let src: Vec<f32> = signal[first * spl..end * spl].to_vec();
        let n = (end - first) as isize;
        for k in 0..n {
            let from = k - shift;
            let dst = &mut signal[(first + k as usize) * spl..(first + k as usize + 1) * spl];
            if (0..n).contains(&from) {
                dst.copy_from_slice(&src[from as usize * spl..(from as usize + 1) * spl]);
            } else {
                dst.fill(BLANKING_V);
            }
        }
    }
}

/// Taps of the fractional-delay kernel used by `shift_segment`.
const SHIFT_TAPS: usize = 2 * SHIFT_HALF_TAPS as usize;

/// Shift `line[lo..hi]` right by `shift` samples (negative = left).
/// Samples that would come from outside the segment read as blanking.
///
/// The fractional part uses a 16-tap Lanczos (windowed-sinc) kernel.
/// Linear interpolation would not do: at 4×fsc it scales the subcarrier by
/// `sqrt((1−f)² + f²)`, down to −3 dB at half a sample, so lines with
/// different fractional shifts would decode with different saturation.
pub(crate) fn shift_segment(line: &mut [f32], lo: usize, hi: usize, shift: f32) {
    if shift.abs() < 1e-6 {
        return;
    }
    let len = hi - lo;
    let whole = shift.floor() as isize;
    let frac = shift - whole as f32;

    // out[j] = Σ_m src[j − whole − m] · L(m − frac), m = 1 − H … H
    let sinc = |x: f32| {
        if x.abs() < 1e-6 {
            1.0
        } else {
            let px = std::f32::consts::PI * x;
            px.sin() / px
        }
    };
    let a = SHIFT_HALF_TAPS as f32;
    let mut taps = [0.0f32; SHIFT_TAPS];
    for (t, w) in taps.iter_mut().enumerate() {
        let x = (t as isize + 1 - SHIFT_HALF_TAPS) as f32 - frac;
        *w = if x.abs() < a { sinc(x) * sinc(x / a) } else { 0.0 };
    }
    let norm = 1.0 / taps.iter().sum::<f32>();

    // The segment with blanking around it: ext[k] = src[k − whole − H].
    let mut ext = [BLANKING_V; SAMPLES_PER_LINE + SHIFT_TAPS - 1];
    let ext = &mut ext[..len + SHIFT_TAPS - 1];
    let first = -whole - SHIFT_HALF_TAPS;
    for (k, e) in ext.iter_mut().enumerate() {
        let i = first + k as isize;
        if i >= 0 && (i as usize) < len {
            *e = line[lo + i as usize];
        }
    }
    shift_taps(ext, &taps, norm, &mut line[lo..hi]);
}

/// Output samples computed together by `shift_taps`.
const SHIFT_LANES: usize = 32;

multiversion! {
    /// `out[j] = norm · Σ_t taps[t]·ext[j + SHIFT_TAPS − 1 − t]`, the sum
    /// taken in order of `t` from −0.0 as `Iterator::sum` does.  Blocks of
    /// outputs are computed side by side so the loop vectorises.
    fn shift_taps(ext: &[f32], taps: &[f32; SHIFT_TAPS], norm: f32, out: &mut [f32]) {
        let n = out.len();
        assert!(ext.len() + 1 >= n + SHIFT_TAPS);
        let mut j = 0;
        while j + SHIFT_LANES <= n {
            let mut acc = [-0.0f32; SHIFT_LANES];
            for (t, &w) in taps.iter().enumerate() {
                let at = j + SHIFT_TAPS - 1 - t;
                let s: &[f32; SHIFT_LANES] = ext[at..at + SHIFT_LANES].try_into().unwrap();
                for l in 0..SHIFT_LANES {
                    acc[l] += w * s[l];
                }
            }
            for l in 0..SHIFT_LANES {
                out[j + l] = norm * acc[l];
            }
            j += SHIFT_LANES;
        }
        for (j, o) in out.iter_mut().enumerate().skip(j) {
            let mut acc = -0.0f32;
            for (t, &w) in taps.iter().enumerate() {
                acc += w * ext[j + SHIFT_TAPS - 1 - t];
            }
            *o = norm * acc;
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// A full frame of picture-like random levels with a colorburst on every
    /// line, so the effects have real content to change.
    fn test_signal() -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(7);
        let mut sig = vec![BLANKING_V; TOTAL_LINES * SAMPLES_PER_LINE];
        for line in 0..TOTAL_LINES {
            let base = line * SAMPLES_PER_LINE;
            for n in BURST_START..BURST_START + BURST_SAMPLES {
                let c = [1.0f32, 0.0, -1.0, 0.0][n & 3];
                sig[base + n] = BLANKING_V - 0.14 * c;
            }
            for n in ACTIVE_START..SAMPLES_PER_LINE {
                sig[base + n] = 0.4 + 0.4 * rng.random::<f32>();
            }
        }
        sig
    }

    fn changed_lines(a: &[f32], b: &[f32]) -> Vec<usize> {
        (0..TOTAL_LINES)
            .filter(|&l| {
                let r = l * SAMPLES_PER_LINE..(l + 1) * SAMPLES_PER_LINE;
                a[r.clone()] != b[r]
            })
            .collect()
    }

    /// Encode a flat 640×480 colour through the real encoder.
    fn encode_flat(rgb: [u8; 3]) -> Vec<f32> {
        let pixels: Vec<u8> = (0..640 * 480).flat_map(|_| rgb).collect();
        crate::encoder::Encoder::new().encode_frame(&pixels, 640, 480, 0).to_vec()
    }

    /// Decode and return the mean RGB of a block around (x, y).
    fn decode_mean(sig: &[f32], x: usize, y: usize, r: usize) -> [f32; 3] {
        let mut dec = crate::decoder::Decoder::new();
        let out = dec.decode_frame(sig, 640, 480, false, false);
        let mut m = [0.0f32; 3];
        let mut n = 0.0;
        for yy in y - r..=y + r {
            for xx in x - r..=x + r {
                for c in 0..3 {
                    m[c] += out[(yy * 640 + xx) * 3 + c] as f32;
                }
                n += 1.0;
            }
        }
        m.map(|v| v / n)
    }

    /// I/Q magnitude of an RGB colour (0–255 scale).
    fn chroma_mag(rgb: [f32; 3]) -> f32 {
        let i: f32 = (0..3).map(|c| crate::constants::RGB_TO_YIQ[1][c] * rgb[c]).sum();
        let q: f32 = (0..3).map(|c| crate::constants::RGB_TO_YIQ[2][c] * rgb[c]).sum();
        (i * i + q * q).sqrt()
    }

    #[test]
    fn chroma_noise_leaves_luma_and_blanking_alone() {
        let orig = encode_flat([180, 60, 50]);
        let mut sig = orig.clone();
        let mut rng = StdRng::seed_from_u64(5);
        let cfg = ChromaPathConfig { noise: 0.08, ..Default::default() };
        apply_chroma_path(&mut sig, &cfg, crate::constants::SAMPLE_RATE, &mut rng);
        let mut max_chroma = 0.0f32;
        let mut max_luma = 0.0f32;
        for line in 0..TOTAL_LINES {
            let base = line * SAMPLES_PER_LINE;
            assert_eq!(sig[base..base + ACTIVE_START], orig[base..base + ACTIVE_START]);
            let d = |k: usize| sig[k] - orig[k];
            for n in base + ACTIVE_START..base + SAMPLES_PER_LINE {
                max_chroma = max_chroma.max(d(n).abs());
            }
            // A mean over four subcarrier cycles cancels the carrier and
            // leaves the low-frequency (luma) part of the change.
            for n in base + ACTIVE_START..base + SAMPLES_PER_LINE - 16 {
                let mean: f32 = (n..n + 16).map(d).sum::<f32>() / 16.0;
                max_luma = max_luma.max(mean.abs());
            }
        }
        assert!(max_chroma > 0.01, "noise too weak: {max_chroma}");
        // The window does not fully cancel a carrier whose amplitude varies,
        // so a few percent remain (about 1 IRE here).
        assert!(max_luma < 0.05 * max_chroma, "luma changed: {max_luma} vs {max_chroma}");
    }

    #[test]
    fn luma_doc_patches_are_colourless_not_inverted() {
        let orig = encode_flat([200, 40, 40]);
        let count_inverted = |doc: Doc| {
            let mut sig = orig.clone();
            let mut rng = StdRng::seed_from_u64(12);
            apply_tape_dropout_doc(&mut sig, 300.0, 15.0, crate::constants::SAMPLE_RATE, &mut rng, doc, Wire::Composite);
            let mut dec = crate::decoder::Decoder::new();
            let out = dec.decode_frame(&sig, 640, 480, false, false).to_vec();
            out.chunks_exact(3).filter(|p| p[1] as i32 > p[0] as i32 + 20).count()
        };
        // Short dropouts are flat streaks whose sharp edges give a little
        // cross-colour in the decoder; long ones must not invert the hue.
        let (composite, luma) = (count_inverted(Doc::Composite), count_inverted(Doc::Luma));
        assert!(composite > 1000 && luma * 10 < composite, "{composite} {luma}");
    }

    #[test]
    fn chroma_phase_noise_keeps_saturation() {
        let orig = encode_flat([180, 60, 50]);
        let mut sig = orig.clone();
        let mut rng = StdRng::seed_from_u64(6);
        let cfg = ChromaPathConfig { phase_noise_deg: 20.0, ..Default::default() };
        apply_chroma_path(&mut sig, &cfg, crate::constants::SAMPLE_RATE, &mut rng);
        assert_ne!(sig, orig);
        let a = chroma_mag(decode_mean(&orig, 320, 240, 8));
        let b = chroma_mag(decode_mean(&sig, 320, 240, 8));
        // Averaging rotated vectors shortens them by about cos(σ); allow 15 %.
        assert!((a - b).abs() < 0.15 * a, "{a} vs {b}");
    }

    #[test]
    fn chroma_vblur_keeps_flat_colour_and_mixes_edges() {
        // Top half red, bottom half blue.
        let pixels: Vec<u8> = (0..640 * 480)
            .flat_map(|i| if i / 640 < 240 { [200u8, 40, 40] } else { [40, 60, 200] })
            .collect();
        let orig = crate::encoder::Encoder::new().encode_frame(&pixels, 640, 480, 0).to_vec();
        let mut sig = orig.clone();
        let mut rng = StdRng::seed_from_u64(8);
        let cfg = ChromaPathConfig { vblur: 1.0, ..Default::default() };
        apply_chroma_path(&mut sig, &cfg, crate::constants::SAMPLE_RATE, &mut rng);

        // Flat areas keep their saturation: the subcarrier inversion
        // between adjacent lines must not cancel the chroma.
        for y in [100, 380] {
            let a = chroma_mag(decode_mean(&orig, 320, y, 8));
            let b = chroma_mag(decode_mean(&sig, 320, y, 8));
            assert!((a - b).abs() < 0.03 * a, "row {y}: {a} vs {b}");
        }
        // Only the rows at the colour edge change visibly: visible rows 240
        // and 241 are the first blue lines of the two fields and take half
        // their chroma from the red line above.
        let mut dec = crate::decoder::Decoder::new();
        let a = dec.decode_frame(&orig, 640, 480, false, false).to_vec();
        let b = dec.decode_frame(&sig, 640, 480, false, false).to_vec();
        let changed: Vec<usize> = (0..480)
            .filter(|&y| {
                (40..600).any(|x| {
                    let i = (y * 640 + x) * 3;
                    (0..3).any(|c| (a[i + c] as i32 - b[i + c] as i32).abs() > 3)
                })
            })
            .collect();
        assert!(changed.contains(&240) && changed.contains(&241), "{changed:?}");
        assert!(changed.iter().all(|&y| (238..=243).contains(&y)), "{changed:?}");
    }

    /// Decoded I/Q magnitude along row `y`.
    fn decoded_chroma_row(sig: &[f32], y: usize) -> Vec<f32> {
        let mut dec = crate::decoder::Decoder::new();
        let out = dec.decode_frame(sig, 640, 480, false, false);
        (0..640)
            .map(|x| {
                let i = (y * 640 + x) * 3;
                chroma_mag([out[i] as f32, out[i + 1] as f32, out[i + 2] as f32])
            })
            .collect()
    }

    #[test]
    fn chroma_delay_moves_colour_right_and_down() {
        // Grey left half, red right half.
        let pixels: Vec<u8> = (0..640 * 480)
            .flat_map(|i| if i % 640 < 320 { [120u8, 120, 120] } else { [200, 40, 40] })
            .collect();
        let orig = crate::encoder::Encoder::new().encode_frame(&pixels, 640, 480, 0).to_vec();
        let mut sig = orig.clone();
        let mut rng = StdRng::seed_from_u64(9);
        let cfg = ChromaPathConfig {
            delay_samples: 4.0 * ACTIVE_SAMPLES as f32 / 640.0,
            ..Default::default()
        };
        apply_chroma_path(&mut sig, &cfg, crate::constants::SAMPLE_RATE, &mut rng);
        // Where the chroma magnitude crosses half of the red level.
        let crossing = |row: &[f32]| {
            let half = 0.5 * row[500];
            (200..450).find(|&x| row[x] >= half).unwrap() as i32
        };
        let shift = crossing(&decoded_chroma_row(&sig, 240)) - crossing(&decoded_chroma_row(&orig, 240));
        assert!((3..=5).contains(&shift), "shift {shift}");

        // Vertical: top half red, bottom half grey; one field line down
        // moves the colour edge down by two output rows.
        let pixels: Vec<u8> = (0..640 * 480)
            .flat_map(|i| if i / 640 < 240 { [200u8, 40, 40] } else { [120, 120, 120] })
            .collect();
        let orig = crate::encoder::Encoder::new().encode_frame(&pixels, 640, 480, 0).to_vec();
        let mut sig = orig.clone();
        let cfg = ChromaPathConfig { delay_lines: 1, ..Default::default() };
        apply_chroma_path(&mut sig, &cfg, crate::constants::SAMPLE_RATE, &mut rng);
        let red = decoded_chroma_row(&orig, 100)[320];
        for y in [240, 241] {
            assert!(decoded_chroma_row(&orig, y)[320] < 0.2 * red);
            assert!(decoded_chroma_row(&sig, y)[320] > 0.8 * red, "row {y}");
        }
        assert!(decoded_chroma_row(&sig, 244)[320] < 0.2 * red);
    }

    #[test]
    fn wobble_moves_lines_but_keeps_hue() {
        // Grey left half, red right half: the luma edge shows the wobble.
        let pixels: Vec<u8> = (0..640 * 480)
            .flat_map(|i| if i % 640 < 320 { [60u8, 60, 60] } else { [200, 40, 40] })
            .collect();
        let orig = crate::encoder::Encoder::new().encode_frame(&pixels, 640, 480, 0).to_vec();
        let mut sig = orig.clone();
        let mut rng = StdRng::seed_from_u64(10);
        apply_wobble(&mut sig, 6.0, 1.0, &mut rng, &TapeClock::off());

        let mut dec = crate::decoder::Decoder::new();
        let out = dec.decode_frame(&sig, 640, 480, false, false).to_vec();
        // Edge position per row: first pixel whose red channel passes 130.
        let edges: Vec<f32> = (20..460)
            .map(|y| (200..450).find(|&x| out[(y * 640 + x) * 3] > 130).unwrap() as f32)
            .collect();
        let mean = edges.iter().sum::<f32>() / edges.len() as f32;
        let std = (edges.iter().map(|e| (e - mean).powi(2)).sum::<f32>() / edges.len() as f32).sqrt();
        assert!(std > 1.0, "edge did not move: std {std}");
        // Neighbouring rows of one field move together (smooth, not jitter).
        let step: f32 = edges.windows(3).map(|w| (w[2] - w[0]).abs()).sum::<f32>() / edges.len() as f32;
        assert!(step < 1.0, "wobble not smooth: mean step {step}");

        // The burst moves with the line, so the red keeps its hue.
        let hue = |rgb: [f32; 3]| {
            let i: f32 = (0..3).map(|c| crate::constants::RGB_TO_YIQ[1][c] * rgb[c]).sum();
            let q: f32 = (0..3).map(|c| crate::constants::RGB_TO_YIQ[2][c] * rgb[c]).sum();
            q.atan2(i).to_degrees()
        };
        let a = hue(decode_mean(&orig, 500, 240, 8));
        let b = hue(decode_mean(&sig, 500, 240, 8));
        assert!((a - b).abs() < 3.0, "hue {a} vs {b}");
    }

    /// Mean absolute difference between two signals.
    fn mean_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32
    }

    #[test]
    fn zero_coherence_keeps_per_frame_draws() {
        let clock = TapeClock { seed: 99, frame: 17, coherence: 0.0 };
        for r in [0.0f32, 0.3, 5.9] {
            assert_eq!(clock.blend(r, || 123.0), r);
        }
    }

    #[test]
    fn coherent_state_drifts_between_frames() {
        // Consecutive frames, each with its own RNG as in the video path.
        // A smooth picture, so the difference tracks displacement rather
        // than the decorrelation of sample-level noise.
        let mut orig = test_signal();
        for line in 0..TOTAL_LINES {
            for n in ACTIVE_START..SAMPLES_PER_LINE {
                let x = (n + 3 * line) as f32;
                orig[line * SAMPLES_PER_LINE + n] = 0.6 + 0.2 * (x / 25.0).sin();
            }
        }
        let run = |frame: u64, coherence: f32, f: &dyn Fn(&mut [f32], &mut StdRng, &TapeClock)| {
            let mut sig = orig.clone();
            let mut rng = StdRng::seed_from_u64(1000 + frame);
            f(&mut sig, &mut rng, &TapeClock { seed: 5, frame, coherence });
            sig
        };
        let smear = |s: &mut [f32], r: &mut StdRng, c: &TapeClock| apply_head_wear_smear(s, 0.8, r, c, false);
        let wobble = |s: &mut [f32], r: &mut StdRng, c: &TapeClock| apply_wobble(s, 3.0, 1.0, r, c);
        for (name, f) in [("smear", &smear as &dyn Fn(&mut [f32], &mut StdRng, &TapeClock)), ("wobble", &wobble)] {
            let jump = mean_abs_diff(&run(10, 0.0, f), &run(11, 0.0, f));
            let drift = mean_abs_diff(&run(10, 1.0, f), &run(11, 1.0, f));
            assert!(drift < 0.3 * jump, "{name}: drift {drift} vs jump {jump}");
        }
    }

    #[test]
    fn v_jitter_moves_whole_lines_and_frame_tbc_stops_it() {
        let orig = test_signal();
        let clock = TapeClock { seed: 3, frame: 0, coherence: 1.0 };
        let mut moved = orig.clone();
        apply_v_jitter(&mut moved, 4.0, &mut StdRng::seed_from_u64(1), &clock);
        assert_ne!(moved, orig);
        // Every changed line is a copy of another line of the same field, or blank.
        for (first, end) in FIELD_ACTIVE_LINES {
            for l in first..end {
                let r = l * SAMPLES_PER_LINE..(l + 1) * SAMPLES_PER_LINE;
                let line = &moved[r.clone()];
                let blank = line.iter().all(|&v| v == BLANKING_V);
                let copy = (first..end).any(|m| &orig[m * SAMPLES_PER_LINE..(m + 1) * SAMPLES_PER_LINE] == line);
                assert!(blank || copy, "line {l}");
            }
        }
        let mut fx = SignalEffects {
            v_jitter: Some(4.0),
            frame_tbc: Some(1.0),
            ..SignalEffects::default()
        };
        let mut sig = orig.clone();
        fx.apply(&mut sig, crate::constants::SAMPLE_RATE, &mut StdRng::seed_from_u64(1), 0);
        assert_eq!(sig, orig);
        fx.frame_tbc = None;
        fx.apply(&mut sig, crate::constants::SAMPLE_RATE, &mut StdRng::seed_from_u64(1), 0);
        assert_ne!(sig, orig);
    }

    #[test]
    fn line_tbc_reduces_wobble() {
        let orig = test_signal();
        let run = |tbc: Option<f32>| {
            let fx = SignalEffects { wobble: Some(3.0), wobble_rate: 1.0, line_tbc: tbc, ..SignalEffects::default() };
            let mut sig = orig.clone();
            fx.apply(&mut sig, crate::constants::SAMPLE_RATE, &mut StdRng::seed_from_u64(9), 0);
            mean_abs_diff(&sig, &orig)
        };
        assert!(run(Some(1.0)) < 0.2 * run(None));
    }

    #[test]
    fn zero_strength_is_noop() {
        let orig = test_signal();
        let mut rng = StdRng::seed_from_u64(1);
        let mut sig = orig.clone();
        apply_head_switch(&mut sig, 0.0, DEFAULT_HEAD_SWITCH_LINES, 1.0, &mut rng, &TapeClock::off());
        apply_top_band(&mut sig, 0.0, &mut rng, &TapeClock::off());
        apply_vbi_data(&mut sig, 0.0, crate::constants::SAMPLE_RATE, &mut rng);
        let off = ChromaPathConfig::default();
        apply_chroma_path(&mut sig, &off, crate::constants::SAMPLE_RATE, &mut rng);
        apply_wobble(&mut sig, 0.0, 1.0, &mut rng, &TapeClock::off());
        assert_eq!(sig, orig);
    }

    #[test]
    fn svideo_matches_separate_wires() {
        let fx = SignalEffects {
            noise: Some(0.01),
            ghosts: vec![GhostConfig { dynamic: true, ..GhostConfig::default() }],
            jitter: Some(0.5),
            vhs_luma_bw: Some(2.0e6),
            color_under_bw: Some(400e3),
            tape_dropout_rate: Some(20.0),
            tape_dropout_len: 15.0,
            edge_ringing: Some(0.3),
            luma_noise: Some(0.03),
            head_switching_smear: Some(0.5),
            tape_trail_smear: Some(0.7),
            head_switch: Some(0.9),
            head_switch_lines: DEFAULT_HEAD_SWITCH_LINES,
            top_band: Some(0.7),
            vbi_data: Some(0.7),
            chroma_noise: Some(0.03),
            chroma_phase_noise: Some(3.0),
            chroma_vblur: Some(0.6),
            chroma_delay: Some(2.0),
            chroma_delay_v: Some(1.0),
            wobble: Some(1.5),
            wobble_rate: 1.0,
            temporal_coherence: 0.5,
            tape_seed: 3,
            v_jitter: Some(2.0),
            deck_sharpness: Some(0.5),
            ..SignalEffects::default()
        };
        let sr = crate::constants::SAMPLE_RATE;
        let (y0, c0) = (test_signal(), encode_flat([200, 60, 40]));
        let (mut y1, mut c1) = (y0.clone(), c0.clone());
        fx.apply_wire(&mut y1, sr, &mut StdRng::seed_from_u64(11), 4, Wire::Luma);
        fx.apply_wire(&mut c1, sr, &mut StdRng::seed_from_u64(11), 4, Wire::Chroma);
        let (mut y2, mut c2) = (y0, c0);
        fx.apply_svideo(&mut y2, &mut c2, sr, &mut StdRng::seed_from_u64(11), 4);
        assert!(y1 == y2 && c1 == c2);
    }

    #[test]
    fn head_switch_touches_only_last_lines_of_each_field() {
        let orig = test_signal();
        let n = DEFAULT_HEAD_SWITCH_LINES;
        let mut rng = StdRng::seed_from_u64(2);
        let mut sig = orig.clone();
        apply_head_switch(&mut sig, 1.0, n, 1.0, &mut rng, &TapeClock::off());
        let expected: Vec<usize> = (260 - n..260).chain(523 - n..523).collect();
        assert_eq!(changed_lines(&orig, &sig), expected);
        // Front porch and sync stay untouched on the affected lines.
        let sync_end = FRONT_PORCH_SAMPLES + HSYNC_SAMPLES;
        for &l in &expected {
            let base = l * SAMPLES_PER_LINE;
            assert_eq!(sig[base..base + sync_end], orig[base..base + sync_end]);
        }
    }

    #[test]
    fn head_switch_hiccups_are_rare_and_hold() {
        let state = |f: u64| head_switch_hiccup(&TapeClock { seed: 5, frame: f, coherence: 1.0 }, KEY_HEAD_SWITCH, 1.0);
        let frames: Vec<Option<[f32; 3]>> = (0..9000).map(state).collect();
        let busy = frames.iter().filter(|h| h.is_some()).count() as f32 / frames.len() as f32;
        assert!((0.03..0.4).contains(&busy), "in a hiccup {:.0}% of the time", busy * 100.0);
        // A hiccup holds one value; changes are few compared with frames.
        let changes = frames.windows(2).filter(|w| w[0] != w[1]).count();
        assert!(changes > 4 && changes < 200, "{} changes", changes);
        assert_eq!(state(1234), state(1234));
    }

    #[test]
    fn vbi_writes_only_lines_21_and_284() {
        let orig = test_signal();
        let mut rng = StdRng::seed_from_u64(3);
        let mut sig = orig.clone();
        apply_vbi_data(&mut sig, 1.0, crate::constants::SAMPLE_RATE, &mut rng);
        assert_eq!(changed_lines(&orig, &sig), vec![21, 284]);
    }

    #[test]
    fn top_band_touches_only_first_lines_of_each_field() {
        let orig = test_signal();
        let mut rng = StdRng::seed_from_u64(4);
        let mut sig = orig.clone();
        apply_top_band(&mut sig, 1.0, &mut rng, &TapeClock::off());
        let expected: Vec<usize> = (20..26).chain(283..289).collect();
        assert_eq!(changed_lines(&orig, &sig), expected);
    }

    #[test]
    fn burst_phase_matches_encoder_convention() {
        // The encoder writes the burst as −cos(π/2·n + line_phase).
        let line_phase = 0.7f32;
        let mut line = vec![BLANKING_V; SAMPLES_PER_LINE];
        for n in BURST_START..BURST_START + BURST_SAMPLES {
            let phi = std::f32::consts::FRAC_PI_2 * n as f32 + line_phase;
            line[n] = BLANKING_V - 0.14 * phi.cos();
        }
        let measured = measure_burst_phase(&line);
        let expected = line_phase + std::f32::consts::PI;
        let diff = (measured - expected).rem_euclid(std::f32::consts::TAU);
        assert!(diff < 1e-3 || diff > std::f32::consts::TAU - 1e-3, "{measured} vs {expected}");
    }
}
