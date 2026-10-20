//! VHS stages built from the recording and playback circuits of real
//! decks: the deck's luma noise canceller, line noise reducer and picture
//! control, and the helpers they share (luma/chroma split, per-stage random
//! streams).
//!
//! Every stage here is opt-in.  Stages draw their random numbers from
//! their own stream, derived from the file's stage seed, the frame number
//! and a stage key, so turning one on never changes the draws of another.

use crate::constants::{ACTIVE_START, SAMPLES_PER_LINE, TOTAL_LINES};
use crate::effects::{FIELD_ACTIVE_LINES, IRE_V};
use crate::iir::{self, Biquad, LANES};

// ────────────────────────────────────────────────────────────────────────────
// Luma/chroma split
// ────────────────────────────────────────────────────────────────────────────

/// Samples before the active picture included in the per-line stages, so
/// the filters have settled when the picture starts.
pub(crate) const PRE_ROLL: usize = 32;

/// First sample of the region the per-line stages work on.
pub(crate) const REGION_START: usize = ACTIVE_START - PRE_ROLL;

/// Samples per line in that region.
pub(crate) const REGION_LEN: usize = SAMPLES_PER_LINE - REGION_START;

/// Half-width (−6 dB) of the chroma band taken out of composite video.
pub(crate) const CHROMA_BAND_HZ: f64 = 0.6e6;

/// Zero-phase lowpass that, applied to demodulated I/Q, selects the
/// chroma band of composite video (fsc ± `CHROMA_BAND_HZ`).
fn chroma_band_filter() -> [Biquad; 1] {
    [Biquad::lowpass(CHROMA_BAND_HZ, std::f64::consts::FRAC_1_SQRT_2)]
}

const COS: [f32; 4] = [1.0, 0.0, -1.0, 0.0];
const NSIN: [f32; 4] = [0.0, -1.0, 0.0, 1.0];

/// Up to `LANES` lines of the region, split into luma and chroma in the
/// lane layout (`buf[j * LANES + l]` is sample `REGION_START + j` of line
/// `l`).
pub(crate) struct LaneGroup {
    pub lines: Vec<usize>,
    pub luma: Vec<f32>,
    pub chroma: Vec<f32>,
    i: Vec<f32>,
    q: Vec<f32>,
}

impl LaneGroup {
    pub fn new() -> Self {
        Self { lines: Vec::new(), luma: Vec::new(), chroma: Vec::new(), i: Vec::new(), q: Vec::new() }
    }

    /// Load `lines` (at most `LANES`) of `signal` and split them.  On an
    /// S-Video Y wire (`full_luma`) everything is luma.  Composite video is
    /// split with a band filter around the subcarrier: chroma is the band
    /// fsc ± 0.6 MHz, luma is the rest.  This is the kind of separation a
    /// VCR does before recording and a capture card does in VCR mode.
    pub fn load(&mut self, signal: &[f32], lines: &[usize], full_luma: bool) {
        self.lines.clear();
        self.lines.extend_from_slice(lines);
        let n = REGION_LEN;
        self.luma.resize(n * LANES, 0.0);
        self.chroma.resize(n * LANES, 0.0);
        for l in 0..LANES {
            let line = lines[l.min(lines.len() - 1)];
            let base = line * SAMPLES_PER_LINE + REGION_START;
            for j in 0..n {
                self.luma[j * LANES + l] = signal[base + j];
            }
        }
        if full_luma {
            self.chroma.fill(0.0);
            return;
        }
        self.i.resize(n * LANES, 0.0);
        self.q.resize(n * LANES, 0.0);
        for j in 0..n {
            let k = (REGION_START + j) & 3;
            for l in 0..LANES {
                let x = self.luma[j * LANES + l];
                self.i[j * LANES + l] = 2.0 * x * COS[k];
                self.q[j * LANES + l] = 2.0 * x * NSIN[k];
            }
        }
        let band = chroma_band_filter();
        iir::zero_phase_lanes(&band, &mut self.i);
        iir::zero_phase_lanes(&band, &mut self.q);
        for j in 0..n {
            let k = (REGION_START + j) & 3;
            for l in 0..LANES {
                let at = j * LANES + l;
                let c = self.i[at] * COS[k] + self.q[at] * NSIN[k];
                self.chroma[at] = c;
                self.luma[at] -= c;
            }
        }
    }

    /// Write luma + chroma back for the active picture of each line.
    pub fn store(&self, signal: &mut [f32]) {
        for (l, &line) in self.lines.iter().enumerate() {
            let base = line * SAMPLES_PER_LINE + REGION_START;
            for j in PRE_ROLL..REGION_LEN {
                let at = j * LANES + l;
                signal[base + j] = self.luma[at] + self.chroma[at];
            }
        }
    }
}

/// The active lines of both fields, in groups of up to `LANES`.
pub(crate) fn active_line_groups() -> Vec<Vec<usize>> {
    let mut groups = Vec::new();
    for &(first, end) in &FIELD_ACTIVE_LINES {
        let lines: Vec<usize> = (first..end).collect();
        for g in lines.chunks(LANES) {
            groups.push(g.to_vec());
        }
    }
    groups
}

// ────────────────────────────────────────────────────────────────────────────
// Deck luma processing: noise canceller, line noise reducer, picture control
// ────────────────────────────────────────────────────────────────────────────

/// Settings of the deck's playback luma circuits.  Zero turns a stage off.
#[derive(Clone, Copy, Debug, Default)]
pub struct DeckLuma {
    /// Noise canceller limiter threshold in IRE.
    pub nc_ire: f32,
}

impl DeckLuma {
    pub fn is_active(&self) -> bool {
        self.nc_ire > 0.0
    }
}

/// Highpass corner of the noise canceller.
const NC_HPF_HZ: f64 = 1.0e6;

/// The deck's playback luma circuits, on the luma of every active line.
///
/// **Noise canceller.**  The luma is highpassed at 1 MHz; a limiter keeps
/// only the part of that within ±threshold (noise and fine low-contrast
/// texture), which is subtracted from the signal.  Large high-frequency
/// components (real edges) exceed the limiter and lose only the threshold
/// amount.  Flat areas come out clean and slightly "waxy": texture below
/// the threshold goes with the noise.  The highpass runs forwards and
/// backwards, like decks that delay-match the cancelled signal.
pub fn apply_deck_luma(signal: &mut [f32], cfg: &DeckLuma, full_luma: bool) {
    if !cfg.is_active() || signal.len() < TOTAL_LINES * SAMPLES_PER_LINE {
        return;
    }
    let mut g = LaneGroup::new();
    let mut hp = Vec::new();
    let lp = [Biquad::lowpass1(NC_HPF_HZ)];
    let t = cfg.nc_ire * IRE_V;
    for lines in active_line_groups() {
        g.load(signal, &lines, full_luma);
        if cfg.nc_ire > 0.0 {
            hp.clear();
            hp.extend_from_slice(&g.luma);
            iir::zero_phase_lanes(&lp, &mut hp);
            for (y, &low) in g.luma.iter_mut().zip(&hp) {
                let h = *y - low;
                *y -= h.clamp(-t, t);
            }
        }
        g.store(signal);
    }
}
