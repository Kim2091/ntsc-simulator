//! The FM tape model (`--tape-model fm`): the VHS record and playback
//! chain built from the format's circuits instead of a single lowpass.
//!
//! Per active line:
//!   1. The recording deck separates luma and chroma with a band filter
//!      around the subcarrier (fsc ± 0.6 MHz); luma detail inside the band
//!      stays in the chroma and is recorded as cross-colour.
//!   2. Chroma is limited to the colour-under bandwidth.
//!   3. Luma passes the record lowpass and, on playback, the lowpass after
//!      the FM demodulator.  Both are causal 2nd-order Butterworth
//!      sections; together they are −3 dB at 0.66 × and −10 dB at 1.0 ×
//!      the luma bandwidth (240 TVL ≈ 3.0 MHz for SP).  The luma is then
//!      advanced by the filters' group delay, as decks delay-match luma
//!      and chroma.

use crate::constants::{BLANKING_V, SAMPLES_PER_LINE, TOTAL_LINES};
use crate::effects::IRE_V;
use crate::encoder::Wire;
use crate::iir::{self, Biquad, LANES};
use crate::vhs::{active_line_groups, LaneGroup};

/// Settings of the FM tape model.
#[derive(Clone, Copy, Debug)]
pub struct TapeFm {
    /// Luma bandwidth (the −10 dB point of record + playback), Hz.
    pub luma_bw: f32,
    /// Colour-under bandwidth (−6 dB), Hz.
    pub chroma_bw: f32,
}

/// Corner of each of the two luma lowpass sections relative to the luma
/// bandwidth: two 2nd-order Butterworth sections at `fc` are −10 dB at
/// 1.21·fc.
const LUMA_CORNER: f64 = 1.0 / 1.21;

/// Run the FM tape model on one wire.
pub fn apply_tape_fm(signal: &mut [f32], cfg: &TapeFm, wire: Wire) {
    if signal.len() < TOTAL_LINES * SAMPLES_PER_LINE {
        return;
    }
    let fc = cfg.luma_bw as f64 * LUMA_CORNER;
    let record = [Biquad::lowpass(fc, std::f64::consts::FRAC_1_SQRT_2)];
    let playback = [Biquad::lowpass(fc, std::f64::consts::FRAC_1_SQRT_2)];
    let both: Vec<Biquad> = record.iter().chain(&playback).copied().collect();
    let advance = iir::dc_group_delay(&both).round() as usize;
    let colour_under = [Biquad::lowpass(cfg.chroma_bw as f64, std::f64::consts::FRAC_1_SQRT_2)];

    let mut g = LaneGroup::new();
    for lines in active_line_groups() {
        g.load(signal, &lines, wire == Wire::Luma);
        if wire != Wire::Luma {
            iir::zero_phase_lanes(&colour_under, &mut g.i);
            iir::zero_phase_lanes(&colour_under, &mut g.q);
            g.remodulate();
        }
        if wire != Wire::Chroma {
            to_ire(&mut g.luma);
            iir::causal_lanes(&record, &mut g.luma);
            iir::causal_lanes(&playback, &mut g.luma);
            advance_lanes(&mut g.luma, advance);
            from_ire(&mut g.luma);
        }
        g.store(signal);
    }
}

/// Volts to IRE (blanking = 0).
fn to_ire(buf: &mut [f32]) {
    for v in buf.iter_mut() {
        *v = (*v - BLANKING_V) / IRE_V;
    }
}

/// IRE to volts.
fn from_ire(buf: &mut [f32]) {
    for v in buf.iter_mut() {
        *v = *v * IRE_V + BLANKING_V;
    }
}

/// Move every lane `d` samples earlier, holding the last value.
fn advance_lanes(buf: &mut [f32], d: usize) {
    let n = buf.len() / LANES;
    if d == 0 || d >= n {
        return;
    }
    buf.copy_within(d * LANES.., 0);
    for j in n - d..n {
        for l in 0..LANES {
            buf[j * LANES + l] = buf[(n - d - 1) * LANES + l];
        }
    }
}
