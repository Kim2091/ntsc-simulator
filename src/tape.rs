//! The FM tape model (`--tape-model fm`): the VHS record and playback
//! chain built from the format's circuits instead of a single lowpass.
//!
//! **Record** (per active line):
//!   1. The recording deck separates luma and chroma with a band filter
//!      around the subcarrier (fsc ± 0.6 MHz); luma detail inside the band
//!      stays in the chroma and is recorded as cross-colour.
//!   2. Chroma is limited to the colour-under bandwidth.
//!   3. Luma passes the record lowpass, the nonlinear emphasis (if on), the
//!      main pre-emphasis H(s) = (1 + sT)/(1 + sT/(1+X)) with T = 1.3 µs and
//!      X = 4 (a +14 dB shelf, corners 122 and 612 kHz), and the white and
//!      dark clips (160 % and 40 % of the sync-to-white span from the sync
//!      tip by default: +184 and −96 IRE).  Every large sharp edge
//!      overshoots the clips by far, so part of it is cut off.
//!
//! **Tape**, in the emphasised domain: FM noise and dropouts (see
//! `TapeFm`).
//!
//! **Playback**: the lowpass after the FM demodulator, the main
//! de-emphasis with the playback deck's own T and X (each deck is within
//! the format's tolerance, T ± 0.05 µs and X ± 0.3, so record and playback
//! rarely match exactly), and the nonlinear de-emphasis.  The luma is then
//! advanced by the lowpass filters' group delay, as decks delay-match luma
//! and chroma.
//!
//! What this looks like: small edges and fine detail are restored as they
//! were (up to the luma bandwidth), but the clipped part of a large edge is
//! missing after de-emphasis, so high-contrast edges are softened with a
//! causal smear: a white object on black rises slowly at its left edge and
//! leaves a light trail into the black after its right edge, decaying over
//! about 1-3 µs.  Noise added on the tape comes out rising with frequency
//! up to the lowpass (fine, horizontally streaky grain), and the demodulated
//! spike of a dropout gets a tail to the right.
//!
//! The record and playback luma lowpass are causal 2nd-order Butterworth
//! sections, together −3 dB at 0.66 × and −10 dB at 1.0 × the luma
//! bandwidth (240 TVL ≈ 3.0 MHz for SP).

use rand::Rng;
use rand_distr::{Distribution, LogNormal};

use crate::constants::{BLANKING_V, SAMPLES_PER_LINE, TOTAL_LINES};
use crate::effects::{drift, hash_signed, mute_segment, FIELD_ACTIVE_LINES, IRE_V};
use crate::encoder::Wire;
use crate::iir::{self, Biquad, FastNoise, LANES};
use crate::vhs::{active_line_groups, stage_rng, stage_seed, LaneGroup, PRE_ROLL, REGION_LEN};

/// Settings of the FM tape model.
#[derive(Clone, Copy, Debug)]
pub struct TapeFm {
    /// Luma bandwidth (the −10 dB point of record + playback), Hz.
    pub luma_bw: f32,
    /// Colour-under bandwidth (−6 dB), Hz.
    pub chroma_bw: f32,
    /// White clip, % of the sync-to-white span above the sync tip.
    pub white_clip: f32,
    /// Dark clip, % of the sync-to-white span below the sync tip.
    pub dark_clip: f32,
    /// Nonlinear emphasis strength (0 = off, 1 = the JVC curve: up to about
    /// +7 dB for small high-frequency signals, +2 dB at full level).
    pub nl_emphasis: f32,
    /// Luma S/N of the tape noise in dB (unweighted, after de-emphasis, on
    /// a flat 50 IRE field, without nonlinear de-emphasis).  `None` = no
    /// tape noise.
    pub snr_db: Option<f32>,
    /// Dropouts per minute counted as the industry does (≥ 15 µs and
    /// ≥ 16 dB deep); shorter and shallower ones are added in proportion.
    /// `None` = none.
    pub dropouts_per_min: Option<f32>,
    /// Most consecutive fields one tape defect can hit (1 for SP; up to 3-4
    /// for EP, whose tracks are a third as wide).
    pub dropout_span: usize,
    /// Whether the playback deck has a dropout compensator.
    pub doc: bool,
    /// Head-to-tape RF level variation (0-1): bands of lower RF level.
    pub rf_loss: f32,
    /// Tracking noise bar strength (0-1).
    pub tracking_bars: f32,
    /// EP carrier interleave left over after playback, IRE: a DC offset
    /// of ±half this between the two heads' fields.
    pub ep_interleave: f32,
    /// Adjacent-track luma crosstalk amplitude (fraction of the
    /// neighbouring track's carrier).
    pub luma_crosstalk: f32,
    /// Black/white inversion strength (0-1).
    pub bw_inversion: f32,
    /// Level difference between the two heads, as a fraction.
    pub head_level_step: f32,
    /// Simulate the FM RF channel itself (modulate, head and tape
    /// response, RF noise, limiter and demodulator) instead of adding
    /// shaped noise in the emphasised domain.
    pub rf: bool,
    /// Seed of the per-stage random streams, and the frame number.
    pub seed: u64,
    pub frame: u64,
}

/// Corner of each of the two luma lowpass sections relative to the luma
/// bandwidth: two 2nd-order Butterworth sections at `fc` are −10 dB at
/// 1.21·fc.
const LUMA_CORNER: f64 = 1.0 / 1.21;

/// Main emphasis time constant (s) and divider ratio, nominal.
const EMPH_T: f64 = 1.3e-6;
const EMPH_X: f64 = 4.0;

/// Nonlinear emphasis: band (highpass and lowpass corners, Hz), the gain
/// for the smallest signals and the envelope (IRE) at which it halves.
const NL_HP_HZ: f64 = 0.82e6;
const NL_LP_HZ: f64 = 2.0e6;
const NL_GAIN: f32 = 2.0;
const NL_KNEE_IRE: f32 = 7.0;
/// Release of the nonlinear emphasis envelope follower, seconds.
const NL_RELEASE_S: f64 = 0.4e-6;

/// Weight of the previous sample in the FM noise shaper n = w[k] − c·w[k−1]:
/// its amplitude rises with frequency (the FM noise triangle) over a small
/// flat floor.
const FM_NOISE_SHAPE: f32 = 0.85;

/// Stage keys of the random streams.
const KEY_NOISE: u64 = 0x7A9E_0001;
const KEY_DROPOUTS: u64 = 0x7A9E_0002;
const KEY_TOLERANCE: u64 = 0x7A9E_0003;
const KEY_RF: u64 = 0x7A9E_0004;
const KEY_TRACKING: u64 = 0x7A9E_0005;
const KEY_CLICKS: u64 = 0x7A9E_0006;
const KEY_INVERSION: u64 = 0x7A9E_0007;

/// FM deviation per IRE (1 MHz over the 140 IRE sync-to-white span) and the
/// EP carrier interleave offset (½ fH).
const HZ_PER_IRE: f32 = 1.0e6 / 140.0;
const INTERLEAVE_HZ: f32 = 7867.0;

/// RF loss (dB) of the worst bands at full `rf_loss`, and of the middle of
/// a tracking noise bar at full strength.
const RF_BAND_DB: f32 = 14.0;
const TRACKING_BAR_DB: f32 = 26.0;

/// RF loss of one line of this frame, in dB: smooth bands of poor head
/// contact (tension, oxide wear, a partly clogged head) that drift slowly
/// down the picture, plus a tracking noise bar.  The same in both fields
/// at the same line of the field; it depends only on the tape seed and
/// the frame, not the picture.
fn rf_loss_db(cfg: &TapeFm, line: usize) -> f32 {
    if cfg.rf_loss <= 0.0 && cfg.tracking_bars <= 0.0 {
        return 0.0;
    }
    let (first, end) = *FIELD_ACTIVE_LINES.iter().find(|(a, b)| (*a..*b).contains(&line)).unwrap_or(&FIELD_ACTIVE_LINES[0]);
    let pos = line.saturating_sub(first) as f64;
    let t = cfg.frame as f64 / 40.0;
    let mut loss = 0.0;
    if cfg.rf_loss > 0.0 {
        let a = drift(cfg.seed, KEY_RF, pos / 37.0 + t);
        let b = drift(cfg.seed, KEY_RF + 1, pos / 13.0 - 0.7 * t);
        let env = ((a + 0.6 * b) / 1.6).max(0.0);
        loss += RF_BAND_DB * cfg.rf_loss * env;
    }
    if cfg.tracking_bars > 0.0 {
        // One bar per tape, rolling slowly.
        let n = (end - first) as f64;
        let centre = n * (0.15 + 0.7 * (0.5 + 0.5 * drift(cfg.seed, KEY_TRACKING, cfg.frame as f64 / 600.0) as f64));
        let half = 5.0 + 7.0 * (0.5 + 0.5 * hash_signed(cfg.seed, KEY_TRACKING, -1) as f64);
        let d = (pos - centre).abs() / half;
        if d < 1.0 {
            let w = (0.5 + 0.5 * (std::f64::consts::PI * d).cos()) as f32;
            loss += TRACKING_BAR_DB * cfg.tracking_bars * w;
        }
    }
    loss
}

/// Chroma level left by an RF loss (the deck's ACC makes up most of it) and
/// the loss from which the colour killer mutes the colour.
fn chroma_gain(loss_db: f32) -> f32 {
    if loss_db >= 20.0 {
        0.0
    } else {
        10f32.powf(-0.3 * loss_db / 20.0)
    }
}

/// Samples a dropout compensator takes to switch over (≈ 0.3 µs): the start
/// of every compensated dropout still shows as a short streak.
const DOC_DELAY: usize = 4;

/// The record and playback emphasis networks of one tape, within the
/// format's tolerance (fixed per tape seed).
fn emphasis_pair(seed: u64) -> (Biquad, Biquad) {
    let u = |k: i64| hash_signed(seed, KEY_TOLERANCE, k);
    let (t_rec, x_rec) = (EMPH_T + 0.05e-6 * u(0) as f64, EMPH_X + 0.3 * u(1) as f64);
    let (t_pb, x_pb) = (EMPH_T + 0.05e-6 * u(2) as f64, EMPH_X + 0.3 * u(3) as f64);
    (Biquad::shelf1(t_rec, t_rec / (1.0 + x_rec)), Biquad::shelf1(t_pb / (1.0 + x_pb), t_pb))
}

/// One dropout on the tape.
#[derive(Clone, Copy, Debug)]
struct Dropout {
    /// Absolute line and region sample range.
    line: usize,
    x0: usize,
    x1: usize,
    /// RF loss in dB.
    depth_db: f32,
}

/// Fraction of all dropouts that the industry counts (≥ 15 µs, ≥ 16 dB):
/// lengths are log-normal with a median of 4 µs (mode ≈ 1.8 µs) and
/// σ = 0.9, depths uniform in 10-40 dB.
const COUNTED_FRACTION: f32 = 0.071 * 0.8;
const DROPOUT_MEDIAN_US: f32 = 4.0;
const DROPOUT_SIGMA: f32 = 0.9;
const DROPOUT_MIN_US: f32 = 0.5;

/// Fields per minute (NTSC).
const FIELDS_PER_MIN: f32 = 59.94 * 60.0;

/// The dropouts that start in global field `field` (2 × frame + field of
/// the frame) and the number of fields each one lasts.
fn field_dropouts(cfg: &TapeFm, field: u64) -> Vec<(Dropout, usize)> {
    let Some(rate) = cfg.dropouts_per_min.filter(|r| *r > 0.0) else { return Vec::new() };
    let mut rng = stage_rng(cfg.seed, field, KEY_DROPOUTS);
    let mean = rate / COUNTED_FRACTION / FIELDS_PER_MIN;
    // Poisson count by inversion.
    let (mut n, mut p, l) = (0usize, rng.random::<f32>(), (-mean).exp());
    while p > l && n < 200 {
        n += 1;
        p *= rng.random::<f32>();
    }
    let lengths = LogNormal::new(DROPOUT_MEDIAN_US.ln(), DROPOUT_SIGMA).unwrap();
    let (first, end) = FIELD_ACTIVE_LINES[(field % 2) as usize];
    (0..n)
        .map(|_| {
            let line = first + (rng.random::<f32>() * (end - first) as f32) as usize;
            let us: f32 = lengths.sample(&mut rng).max(DROPOUT_MIN_US);
            let len = (us * 1e-6 * crate::constants::SAMPLE_RATE as f32).round() as usize;
            let x0 = PRE_ROLL + (rng.random::<f32>() * (REGION_LEN - PRE_ROLL) as f32) as usize;
            let depth_db = 10.0 + 30.0 * rng.random::<f32>();
            let span = 1 + (rng.random::<f32>() * cfg.dropout_span.max(1) as f32) as usize;
            (Dropout { line, x0, x1: (x0 + len).min(REGION_LEN), depth_db }, span.min(cfg.dropout_span.max(1)))
        })
        .collect()
}

/// Every dropout visible in the two fields of the frame: those starting in
/// a field, and those of up to three earlier fields that are wide enough to
/// reach it (on the neighbouring track, at the same place, a field later).
/// Dropouts reaching a field of the other parity land on the same line
/// index of that field.
fn frame_dropouts(cfg: &TapeFm) -> Vec<Dropout> {
    let mut out = Vec::new();
    if cfg.dropouts_per_min.is_none_or(|r| r <= 0.0) {
        return out;
    }
    for f in 0..2u64 {
        let field = 2 * cfg.frame + f;
        for back in 0..4u64 {
            let Some(src) = field.checked_sub(back) else { break };
            for (d, span) in field_dropouts(cfg, src) {
                if back as usize >= span {
                    continue;
                }
                let (from_first, _) = FIELD_ACTIVE_LINES[(src % 2) as usize];
                let (to_first, to_end) = FIELD_ACTIVE_LINES[f as usize];
                let line = (to_first + (d.line - from_first)).min(to_end - 1);
                out.push(Dropout { line, ..d });
            }
        }
    }
    out
}

/// Gain from white noise through the FM noise shaper, the playback lowpass
/// and the de-emphasis to the output, as an rms factor.
fn noise_gain(playback: &[Biquad], deemph: Biquad) -> f32 {
    let mut sections: Vec<Biquad> = playback.to_vec();
    sections.push(deemph);
    // Convolve the shaper with the sections' impulse response.
    let g = Biquad::impulse(&sections, 4096);
    let mut sum = 0.0f64;
    for k in 0..4095 {
        let v = g[k + 1] as f64 - FM_NOISE_SHAPE as f64 * g[k] as f64;
        sum += v * v;
    }
    sum += (g[0] as f64).powi(2);
    sum.sqrt() as f32
}

/// Run the FM tape model on one wire.
pub fn apply_tape_fm(signal: &mut [f32], cfg: &TapeFm, wire: Wire) {
    if signal.len() < TOTAL_LINES * SAMPLES_PER_LINE {
        return;
    }
    let q = std::f64::consts::FRAC_1_SQRT_2;
    let fc = cfg.luma_bw as f64 * LUMA_CORNER;
    let record = [Biquad::lowpass(fc, q)];
    let playback = [Biquad::lowpass(fc, q)];
    let both: Vec<Biquad> = record.iter().chain(&playback).copied().collect();
    let advance = iir::dc_group_delay(&both).round() as usize;
    let colour_under = [Biquad::lowpass(cfg.chroma_bw as f64, q)];
    let (pre, de) = emphasis_pair(cfg.seed);
    let white = -40.0 + 1.4 * cfg.white_clip;
    let dark = -40.0 - 1.4 * cfg.dark_clip;
    let nl = NlEmphasis::new(cfg.nl_emphasis);
    let dropouts = frame_dropouts(cfg);
    let do_luma = wire != Wire::Chroma;

    // ── Record ──
    let groups_lines = active_line_groups();
    let mut groups: Vec<LaneGroup> = Vec::with_capacity(groups_lines.len());
    for lines in &groups_lines {
        let mut g = LaneGroup::new();
        g.load(signal, lines, wire == Wire::Luma);
        if wire != Wire::Luma {
            iir::zero_phase_lanes(&colour_under, &mut g.i);
            iir::zero_phase_lanes(&colour_under, &mut g.q);
            g.remodulate();
        }
        if do_luma {
            to_ire(&mut g.luma);
            iir::causal_lanes(&record, &mut g.luma);
            nl.record(&mut g.luma);
            pre.run_lanes(&mut g.luma);
            for v in g.luma.iter_mut() {
                *v = v.clamp(dark, white);
            }
        }
        groups.push(g);
    }

    // ── Tape ──
    // The neighbouring track of each line is the other field: its
    // recorded luma, for the crosstalk beats.
    let recorded: Vec<Vec<f32>> = if do_luma && cfg.luma_crosstalk > 0.0 {
        groups.iter().map(|g| g.luma.clone()).collect()
    } else {
        Vec::new()
    };
    let mut inversions = FastNoise::new(stage_seed(cfg.seed, cfg.frame, KEY_INVERSION));
    if do_luma {
        let gain = noise_gain(&playback, de);
        let sigma_of = |snr: f32| 100.0 * 10f32.powf(-snr / 20.0) / gain;
        let sigma = cfg.snr_db.map(sigma_of);
        // Noise reference for RF loss without --tape-snr: a good tape.
        let base = sigma.unwrap_or(sigma_of(45.0));
        let mut noise = FastNoise::new(stage_seed(cfg.seed, cfg.frame, KEY_NOISE));
        let mut clicks = FastNoise::new(stage_seed(cfg.seed, cfg.frame, KEY_CLICKS));
        let mut white_n = vec![0.0f32; REGION_LEN + 1];
        let mut line = vec![0.0f32; REGION_LEN];
        let mut rf_channel = cfg.rf.then(|| {
            let target = 100.0 * 10f32.powf(-cfg.snr_db.unwrap_or(45.0) / 20.0);
            RfChannel::new(stage_seed(cfg.seed, cfg.frame, KEY_NOISE), target, &playback, de)
        });
        let mut prev = vec![0.0f32; REGION_LEN];
        for gi in 0..groups.len() {
            let n_lanes = groups[gi].lines.len();
            for l in 0..n_lanes {
                let g = &mut groups[gi];
                let abs = g.lines[l];
                let first_of_field = FIELD_ACTIVE_LINES.iter().any(|&(a, _)| a == abs);
                for j in 0..REGION_LEN {
                    line[j] = g.luma[j * LANES + l];
                }
                let loss = rf_loss_db(cfg, abs);
                let m = 10f32.powf(loss / 20.0);
                let s_line = match sigma {
                    Some(s) => s * m,
                    None if loss > 0.0 => base * (m - 1.0),
                    None => 0.0,
                };
                let field = usize::from(abs >= FIELD_ACTIVE_LINES[1].0);
                if let Some(rf) = rf_channel.as_mut() {
                    let half = groups_lines.len() / 2;
                    let other = if gi >= half { gi - half } else { gi + half };
                    let o = recorded.get(other).map(|o| (o.as_slice(), l));
                    let line_drops: Vec<Dropout> = dropouts.iter().filter(|d| d.line == abs).copied().collect();
                    rf.line(&mut line, o, loss, &line_drops, field, cfg);
                } else if s_line > 0.0 {
                    noise.fill(&mut white_n);
                    for j in 0..REGION_LEN {
                        // Higher carrier (brighter, or an emphasised
                        // overshoot): more head loss, lower CNR.
                        let level = 0.85 + 0.3 * (line[j] / 100.0).clamp(-0.4, 1.84);
                        line[j] += s_line * level * (white_n[j + 1] - FM_NOISE_SHAPE * white_n[j]);
                    }
                }
                if loss > 1.0 && rf_channel.is_none() {
                    rf_line_loss(&mut line, loss, &mut clicks, white, dark);
                }
                if !recorded.is_empty() && rf_channel.is_none() {
                    // Group `gi` of one field and of the other differ by
                    // half the groups.
                    let half = groups_lines.len() / 2;
                    let other = if gi >= half { gi - half } else { gi + half };
                    let o = &recorded[other.min(recorded.len() - 1)];
                    crosstalk_beats(&mut line, o, l, cfg, field, abs);
                }
                if cfg.ep_interleave != 0.0 {
                    let dc = 0.5 * cfg.ep_interleave * if field == 0 { 1.0 } else { -1.0 };
                    for v in line.iter_mut() {
                        *v += dc;
                    }
                }
                if cfg.bw_inversion > 0.0 {
                    inversion(&mut line, cfg.bw_inversion, loss, white, dark, &mut inversions);
                }
                for d in dropouts.iter().filter(|d| d.line == abs) {
                    if rf_channel.is_some() {
                        // The RF channel made the streak; the compensator
                        // still replaces it.
                        if cfg.doc && !first_of_field && d.depth_db >= FULL_DROPOUT_DB {
                            let from = (d.x0 + DOC_DELAY).min(d.x1);
                            line[from..d.x1].copy_from_slice(&prev[from..d.x1]);
                        }
                    } else {
                        render_dropout(&mut line, &prev, d, cfg.doc && !first_of_field, sigma.unwrap_or(1.0), &mut noise, white, dark);
                    }
                }
                for j in 0..REGION_LEN {
                    g.luma[j * LANES + l] = line[j];
                }
                prev.copy_from_slice(&line);
            }
        }
    }

    // ── Playback ──
    for g in groups.iter_mut() {
        if do_luma {
            iir::causal_lanes(&playback, &mut g.luma);
            de.run_lanes(&mut g.luma);
            nl.playback(&mut g.luma);
            advance_lanes(&mut g.luma, advance);
            if cfg.head_level_step != 0.0 {
                for (l, &abs) in g.lines.iter().enumerate() {
                    let k = 1.0 + 0.5 * cfg.head_level_step * if abs < FIELD_ACTIVE_LINES[1].0 { 1.0 } else { -1.0 };
                    for j in 0..REGION_LEN {
                        g.luma[j * LANES + l] *= k;
                    }
                }
            }
            from_ire(&mut g.luma);
        }
        if wire != Wire::Luma && (cfg.rf_loss > 0.0 || cfg.tracking_bars > 0.0) {
            for (l, &abs) in g.lines.iter().enumerate() {
                let k = chroma_gain(rf_loss_db(cfg, abs));
                if k < 1.0 {
                    for j in 0..REGION_LEN {
                        g.chroma[j * LANES + l] *= k;
                    }
                }
            }
        }
        // Decks mute the colour over a dropout.
        if wire != Wire::Luma {
            for d in dropouts.iter().filter(|d| d.depth_db >= FULL_DROPOUT_DB) {
                if let Some(l) = g.lines.iter().position(|&x| x == d.line) {
                    mute_lane(&mut g.chroma, l, d.x0, d.x1);
                }
            }
        }
        g.store(signal);
    }
}

/// RF loss (dB) below which the limiters absorb a dropout, and from which
/// it is a full dropout (carrier lost).
const ABSORBED_DB: f32 = 12.0;
const FULL_DROPOUT_DB: f32 = 20.0;

/// Render one dropout on a line in the emphasised domain.
#[allow(clippy::too_many_arguments)]
fn render_dropout(line: &mut [f32], prev: &[f32], d: &Dropout, doc: bool, sigma: f32, noise: &mut FastNoise, white: f32, dark: f32) {
    if d.depth_db < ABSORBED_DB || d.x1 <= d.x0 {
        return;
    }
    if d.depth_db < FULL_DROPOUT_DB {
        // A partial loss: the carrier is still there, the noise rises.
        let extra = 10f32.powf((d.depth_db - ABSORBED_DB) / 20.0) * sigma.max(0.5) * 2.0;
        for v in &mut line[d.x0..d.x1] {
            *v += extra * noise.normal();
        }
        return;
    }
    // The carrier is gone: the demodulator puts out noise around a level
    // far from the picture.  After a bright area the limiter loses the
    // weak high carrier first and the streak goes black, otherwise it is
    // mostly white.
    let before = line[d.x0.saturating_sub(3)];
    let black = before > 70.0 || noise.uniform() < 0.25;
    let level = if black { -30.0 } else { 115.0 };
    let raw_end = if doc { (d.x0 + DOC_DELAY).min(d.x1) } else { d.x1 };
    for v in &mut line[d.x0..raw_end] {
        *v = (level + 20.0 * noise.normal()).clamp(dark, white);
    }
    // The compensator repeats the line above (it may itself be a repeat).
    if doc {
        line[raw_end..d.x1].copy_from_slice(&prev[raw_end..d.x1]);
    }
}

/// What a low RF level does to a line in the emphasised domain, besides
/// more noise: the spacing loss takes the upper FM sideband first, so fine
/// detail softens a little (a one-pole lowpass falling from 6 MHz to about
/// 2.4 MHz at 12 dB, no lower than 2 MHz), and below about 6 dB the FM
/// demodulator starts to
/// click: short white (and some black) sparkles whose de-emphasis tail
/// trails to the right.
fn rf_line_loss(line: &mut [f32], loss_db: f32, clicks: &mut FastNoise, white: f32, dark: f32) {
    let soft = Biquad::lowpass1((6.0e6 * 10f64.powf(-loss_db as f64 / 30.0)).max(2.0e6));
    soft.run(line);
    if loss_db > 6.0 {
        let rate = (1.0e-4 * 10f32.powf((loss_db - 6.0) / 8.0)).min(0.02);
        let mut j = 0;
        while j + 2 < line.len() {
            if clicks.uniform() < rate {
                let up = clicks.uniform() < 0.7;
                let a = 80.0 + 100.0 * clicks.uniform();
                for v in &mut line[j..j + 2] {
                    *v = (*v + if up { a } else { -a }).clamp(dark, white);
                }
                j += 2;
            }
            j += 1;
        }
    }
}

/// Luma crosstalk from the neighbouring track (lane `l` of `other`, the
/// other field's recorded luma).  Azimuth recording rejects it only
/// partly, and the two FM carriers beat: the demodulator puts out a tone
/// at their frequency difference (7.14 kHz per IRE of level difference,
/// plus ½ fH with the EP interleave) with an amplitude of the crosstalk
/// ratio times that difference.  Where the fields differ the picture gets
/// low-frequency, patterned noise ("orange peel").
fn crosstalk_beats(line: &mut [f32], other: &[f32], l: usize, cfg: &TapeFm, field: usize, abs: usize) {
    let r = cfg.luma_crosstalk;
    let offset = if cfg.ep_interleave != 0.0 { INTERLEAVE_HZ } else { 0.0 };
    let dt = 1.0 / crate::constants::SAMPLE_RATE as f32;
    // The beat's phase at the start of the line drifts from line to line.
    let mut phase = std::f32::consts::PI * (0.37 * abs as f32 + field as f32);
    for (j, v) in line.iter_mut().enumerate() {
        let diff = *v - other[j * LANES + l];
        phase += std::f32::consts::TAU * (HZ_PER_IRE * diff + offset) * dt;
        *v += r * diff * phase.cos();
    }
}

/// RF sample rate: twice the video rate (8 × fsc, as a CX capture card).
const RF_RATE: f64 = 2.0 * crate::constants::SAMPLE_RATE;
/// FM carrier at the sync tip (−40 IRE).
const SYNC_TIP_HZ: f32 = 3.4e6;
/// Head and tape response: the upper FM sideband falls off above about
/// 5.5 MHz; the record highpass clears the colour-under band below 1.2 MHz.
const RF_LOWPASS_HZ: f64 = 5.5e6;
const RF_HIGHPASS_HZ: f64 = 1.2e6;

/// The FM channel itself (`--tape-model rf`).  Per line, at 8 × fsc:
///   1. the emphasised luma sets the carrier frequency (3.4 MHz at the
///      sync tip, 4.4 MHz at white, 7.14 kHz per IRE; the clips limit it
///      to about 3.0-5.0 MHz) and a phasor is rotated by it;
///   2. the head and tape response (highpass 1.2 MHz, lowpass 5.5 MHz,
///      so the upper sideband is weaker than the lower one), the RF level
///      (head contact loss, dropouts) and the neighbouring track's carrier
///      (luma crosstalk, at the given ratio) shape it;
///   3. white RF noise is added (calibrated so that a flat 50 IRE field
///      gives the tape S/N after de-emphasis);
///   4. a limiter and an FM demodulator (the phase step between samples)
///      recover the frequency, and pairs of samples are averaged back to
///      4 × fsc.
/// FM noise, its triangular spectrum, clicks when the carrier is weak,
/// streaks in dropouts and the beats of crosstalk all come out of this
/// instead of being added separately.
struct RfChannel {
    noise: FastNoise,
    sigma: f32,
    shape: [Biquad; 2],
    re: Vec<f32>,
    im: Vec<f32>,
    other_re: Vec<f32>,
    other_im: Vec<f32>,
    amp: Vec<f32>,
    other_line: Vec<f32>,
}

impl RfChannel {
    fn new(seed: u64, target_ire: f32, playback: &[Biquad], de: Biquad) -> Self {
        let mut ch = Self {
            noise: FastNoise::new(seed ^ 0x52_46),
            sigma: 0.0,
            shape: [
                Biquad::highpass1_at(RF_HIGHPASS_HZ, RF_RATE),
                Biquad::lowpass_at(RF_LOWPASS_HZ, std::f64::consts::FRAC_1_SQRT_2, RF_RATE),
            ],
            re: Vec::new(),
            im: Vec::new(),
            other_re: Vec::new(),
            other_im: Vec::new(),
            amp: Vec::new(),
            other_line: Vec::new(),
        };
        // Calibrate: the output noise of a flat 50 IRE line at a known RF
        // noise level (the channel is linear in the noise at good CNR).
        let probe = 0.02;
        ch.sigma = probe;
        let mut sum2 = 0.0f64;
        let mut n = 0.0f64;
        let cfg_free = RfLine { loss_db: 0.0, crosstalk: 0.0, interleave: 0.0 };
        for _ in 0..6 {
            let mut line = vec![50.0f32; REGION_LEN];
            ch.process(&mut line, None, &cfg_free, &[]);
            for s in playback {
                s.run(&mut line);
            }
            de.run(&mut line);
            for &v in &line[200..REGION_LEN - 50] {
                sum2 += ((v - 50.0) as f64).powi(2);
                n += 1.0;
            }
        }
        let measured = (sum2 / n).sqrt() as f32;
        ch.sigma = probe * target_ire / measured.max(1e-6);
        ch
    }

    fn line(&mut self, line: &mut [f32], other: Option<(&[f32], usize)>, loss_db: f32, drops: &[Dropout], field: usize, cfg: &TapeFm) {
        let mut other_line = std::mem::take(&mut self.other_line);
        other_line.clear();
        if let Some((o, l)) = other {
            other_line.extend((0..REGION_LEN).map(|j| o[j * LANES + l]));
        }
        let rl = RfLine {
            loss_db,
            crosstalk: cfg.luma_crosstalk,
            interleave: if cfg.ep_interleave == 0.0 {
                0.0
            } else if field == 0 {
                0.5 * INTERLEAVE_HZ
            } else {
                -0.5 * INTERLEAVE_HZ
            },
        };
        self.process(line, (!other_line.is_empty()).then_some(other_line.as_slice()), &rl, drops);
        self.other_line = other_line;
    }

    /// Modulate, pass the channel, demodulate one line (IRE, emphasised).
    fn process(&mut self, line: &mut [f32], other: Option<&[f32]>, rl: &RfLine, drops: &[Dropout]) {
        let n = line.len();
        let n2 = 2 * n;
        let k = std::f32::consts::TAU / RF_RATE as f32;
        let hz_ire = 1.0e6 / 140.0;
        let freq = |e: f32| SYNC_TIP_HZ + (e + 40.0) * hz_ire;
        // Carrier amplitude: RF loss, dropouts (with 0.2 µs edges).
        self.amp.clear();
        self.amp.resize(n2, 10f32.powf(-rl.loss_db / 20.0));
        for d in drops {
            let g = 10f32.powf(-d.depth_db / 20.0);
            let ramp = 6.0;
            for j in (2 * d.x0).saturating_sub(6)..(2 * d.x1 + 6).min(n2) {
                let into = (j as f32 - (2 * d.x0) as f32 + ramp).min((2 * d.x1) as f32 + ramp - j as f32) / ramp;
                let w = into.clamp(0.0, 1.0);
                self.amp[j] *= 1.0 - w * (1.0 - g);
            }
        }
        modulate(line, rl.interleave, freq, k, Some(&self.amp), 1.0, &mut self.re, &mut self.im);
        if let (Some(o), true) = (other, rl.crosstalk > 0.0) {
            modulate(o, -rl.interleave, freq, k, None, rl.crosstalk, &mut self.other_re, &mut self.other_im);
            for j in 0..n2 {
                self.re[j] += self.other_re[j];
                self.im[j] += self.other_im[j];
            }
        }
        for s in &self.shape {
            s.run(&mut self.re);
            s.run(&mut self.im);
        }
        for j in 0..n2 {
            self.re[j] += self.sigma * self.noise.normal();
            self.im[j] += self.sigma * self.noise.normal();
        }
        // Limiter and demodulator: the phase step between samples.
        let mut prev = (self.re[0], self.im[0]);
        let mut out_prev = line[0];
        for j in 0..n2 {
            let (a, b) = (self.re[j], self.im[j]);
            let (x, y) = (a * prev.0 + b * prev.1, b * prev.0 - a * prev.1);
            prev = (a, b);
            let f = fast_atan2(y, x) / k - rl.interleave;
            let ire = ((f - SYNC_TIP_HZ) / hz_ire - 40.0).clamp(-140.0, 280.0);
            if j % 2 == 1 {
                line[j / 2] = 0.5 * (out_prev + ire);
            }
            out_prev = ire;
        }
    }
}

/// Settings of one line of the RF channel.
struct RfLine {
    loss_db: f32,
    crosstalk: f32,
    /// Carrier offset of this head (EP interleave), Hz.
    interleave: f32,
}

/// FM-modulate `e` (IRE, 4 × fsc) at 8 × fsc into an analytic carrier of
/// amplitude `gain` × `amp[j]` (`amp` absent: 1), in `re` / `im`.
#[allow(clippy::too_many_arguments)]
fn modulate(
    e: &[f32],
    offset: f32,
    freq: impl Fn(f32) -> f32,
    k: f32,
    amp: Option<&[f32]>,
    gain: f32,
    re: &mut Vec<f32>,
    im: &mut Vec<f32>,
) {
    let n2 = 2 * e.len();
    re.resize(n2, 0.0);
    im.resize(n2, 0.0);
    let (mut pr, mut pi) = (1.0f32, 0.0f32);
    for j in 0..n2 {
        // Linear interpolation to the RF rate.
        let a = e[j / 2];
        let b = e[(j / 2 + 1).min(e.len() - 1)];
        let v = if j % 2 == 0 { a } else { 0.5 * (a + b) };
        let w = k * (freq(v) + offset);
        let (s, c) = sin_cos_small(w);
        let nr = pr * c - pi * s;
        pi = pr * s + pi * c;
        pr = nr;
        if j % 64 == 63 {
            let m = 1.0 / (pr * pr + pi * pi).sqrt();
            pr *= m;
            pi *= m;
        }
        let a = gain * amp.map_or(1.0, |a| a[j]);
        re[j] = a * pr;
        im[j] = a * pi;
    }
}

/// sin and cos of a small angle (|w| < 1.6) by Taylor series, accurate to
/// about 1e-7.
#[inline]
fn sin_cos_small(w: f32) -> (f32, f32) {
    let w2 = w * w;
    let s = w * (1.0 - w2 / 6.0 * (1.0 - w2 / 20.0 * (1.0 - w2 / 42.0 * (1.0 - w2 / 72.0))));
    let c = 1.0 - w2 / 2.0 * (1.0 - w2 / 12.0 * (1.0 - w2 / 30.0 * (1.0 - w2 / 56.0 * (1.0 - w2 / 90.0))));
    (s, c)
}

/// atan2 with a minimax polynomial, error about 1e-5 rad.
#[inline]
fn fast_atan2(y: f32, x: f32) -> f32 {
    let (ax, ay) = (x.abs(), y.abs());
    let (num, den, swap) = if ay > ax { (ax, ay, true) } else { (ay, ax, false) };
    if den == 0.0 {
        return 0.0;
    }
    let t = num / den;
    let t2 = t * t;
    let mut r = t * (0.999_977_26 + t2 * (-0.332_623_47 + t2 * (0.193_543_46 + t2 * (-0.116_432_87 + t2 * (0.052_653_32 + t2 * -0.011_721_2)))));
    if swap {
        r = std::f32::consts::FRAC_PI_2 - r;
    }
    if x < 0.0 {
        r = std::f32::consts::PI - r;
    }
    if y < 0.0 {
        -r
    } else {
        r
    }
}

/// Black/white inversion: where the emphasised overshoot of a bright edge
/// sits on the white clip, the carrier is at its highest frequency and
/// weakest after the head and tape losses.  If the limiter loses it, the
/// demodulator drops to black for a moment right after the edge: black
/// specks with a tail just after strong dark-to-bright edges, more often
/// with weak RF (worn heads).
fn inversion(line: &mut [f32], strength: f32, loss_db: f32, white: f32, dark: f32, rng: &mut FastNoise) {
    let p_base = strength * (0.05 + loss_db / 25.0);
    let mut j = 0;
    while j < line.len() {
        if line[j] < white - 0.5 {
            j += 1;
            continue;
        }
        let start = j;
        while j < line.len() && line[j] >= white - 0.5 {
            j += 1;
        }
        let run = (j - start) as f32;
        if run >= 3.0 && rng.uniform() < (p_base * run / 10.0).min(0.9) {
            let k = 2 + (rng.uniform() * 7.0) as usize;
            let end = (start + k).min(line.len());
            for v in &mut line[start..end] {
                *v = (-20.0 + 15.0 * rng.normal()).clamp(dark, white);
            }
        }
    }
}

/// Mute lane `l` of a lane-layout chroma buffer over region samples
/// `x0..x1`.
fn mute_lane(chroma: &mut [f32], l: usize, x0: usize, x1: usize) {
    let mut tmp: Vec<f32> = (x0..x1).map(|j| chroma[j * LANES + l]).collect();
    let n = tmp.len();
    mute_segment(&mut tmp, 0, n, 0.0);
    for (k, j) in (x0..x1).enumerate() {
        chroma[j * LANES + l] = tmp[k];
    }
}

/// The nonlinear emphasis pair: on record, extra gain for small
/// high-frequency signals (a band around 1.3 MHz, up to about +7 dB for
/// the smallest, +2 dB at full level); on playback, the matching cut,
/// which also lowers tape noise on flat areas by up to about 7 dB but not
/// next to large edges.  The record stage sits before the main emphasis
/// and the playback stage after the main de-emphasis, mirror images.
struct NlEmphasis {
    gain: f32,
    band: [Biquad; 2],
    release: f32,
}

impl NlEmphasis {
    fn new(strength: f32) -> Self {
        Self {
            gain: NL_GAIN * strength.max(0.0),
            band: [Biquad::highpass1(NL_HP_HZ), Biquad::lowpass1(NL_LP_HZ)],
            release: (-1.0 / (NL_RELEASE_S * crate::constants::SAMPLE_RATE)).exp() as f32,
        }
    }

    fn g(&self, env: f32) -> f32 {
        self.gain / (1.0 + env / NL_KNEE_IRE)
    }

    /// The band signal and its envelope, per sample (lane layout).
    fn band_env(&self, buf: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut h = buf.to_vec();
        iir::causal_lanes(&self.band, &mut h);
        let mut env = vec![0.0f32; h.len()];
        let mut e = [0.0f32; LANES];
        for (j, chunk) in h.chunks_exact(LANES).enumerate() {
            for l in 0..LANES {
                e[l] = chunk[l].abs().max(e[l] * self.release);
                env[j * LANES + l] = e[l];
            }
        }
        (h, env)
    }

    fn record(&self, buf: &mut [f32]) {
        if self.gain <= 0.0 {
            return;
        }
        let (h, env) = self.band_env(buf);
        for k in 0..buf.len() {
            buf[k] += self.g(env[k]) * h[k];
        }
    }

    fn playback(&self, buf: &mut [f32]) {
        if self.gain <= 0.0 {
            return;
        }
        let (h, env) = self.band_env(buf);
        for k in 0..buf.len() {
            // The recorded band was (1 + g) times the original; estimate
            // the original envelope, then undo that gain.
            let g1 = self.g(env[k] / (1.0 + self.g(env[k])));
            buf[k] -= g1 / (1.0 + g1) * h[k];
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> TapeFm {
        TapeFm {
            luma_bw: 3.0e6,
            chroma_bw: 400e3,
            white_clip: 160.0,
            dark_clip: 40.0,
            nl_emphasis: 0.0,
            snr_db: None,
            dropouts_per_min: None,
            dropout_span: 1,
            doc: true,
            rf_loss: 0.0,
            tracking_bars: 0.0,
            ep_interleave: 0.0,
            luma_crosstalk: 0.0,
            bw_inversion: 0.0,
            head_level_step: 0.0,
            rf: false,
            seed: 1,
            frame: 0,
        }
    }

    /// A flat line at `a` IRE with a step to `b` IRE in the middle, on
    /// every active line; returns (signal, step sample).
    fn step_signal(a: f32, b: f32) -> (Vec<f32>, usize) {
        let mut sig = vec![BLANKING_V; TOTAL_LINES * SAMPLES_PER_LINE];
        let mid = 500;
        for &(f, e) in &FIELD_ACTIVE_LINES {
            for line in f..e {
                for n in crate::constants::ACTIVE_START..SAMPLES_PER_LINE {
                    let ire = if n < mid { a } else { b };
                    sig[line * SAMPLES_PER_LINE + n] = BLANKING_V + ire * IRE_V;
                }
            }
        }
        (sig, mid)
    }

    fn ire_at(sig: &[f32], n: usize) -> f32 {
        (sig[100 * SAMPLES_PER_LINE + n] - BLANKING_V) / IRE_V
    }

    #[test]
    fn small_edges_pass_large_edges_smear() {
        // A 10 IRE step is far below the clips: it settles within ~1 µs.
        let (mut sig, mid) = step_signal(40.0, 50.0);
        apply_tape_fm(&mut sig, &cfg(), Wire::Luma);
        assert!((ire_at(&sig, mid + 20) - 50.0).abs() < 0.5, "{}", ire_at(&sig, mid + 20));
        // A black-to-white step is clipped: 1.4 µs (20 samples) after the
        // edge the white is still well short of 100 IRE.
        let (mut sig, mid) = step_signal(7.5, 100.0);
        apply_tape_fm(&mut sig, &cfg(), Wire::Luma);
        let v = ire_at(&sig, mid + 20);
        assert!(v < 95.0 && v > 60.0, "{v}");
        assert!((ire_at(&sig, mid + 150) - 100.0).abs() < 1.0);
        // White to black leaves a light trail.
        let (mut sig, mid) = step_signal(100.0, 7.5);
        apply_tape_fm(&mut sig, &cfg(), Wire::Luma);
        let v = ire_at(&sig, mid + 20);
        assert!(v > 12.0, "{v}");
    }

    #[test]
    fn noise_is_calibrated() {
        let (mut sig, _) = step_signal(50.0, 50.0);
        let c = TapeFm { snr_db: Some(43.0), ..cfg() };
        apply_tape_fm(&mut sig, &c, Wire::Luma);
        let mut sum2 = 0.0f64;
        let mut n = 0.0;
        for line in 40..250 {
            for k in crate::constants::ACTIVE_START + 40..SAMPLES_PER_LINE - 40 {
                let v = (sig[line * SAMPLES_PER_LINE + k] - BLANKING_V) / IRE_V - 50.0;
                sum2 += (v * v) as f64;
                n += 1.0;
            }
        }
        let snr = 20.0 * (100.0 / (sum2 / n).sqrt()).log10();
        assert!((snr - 43.0).abs() < 0.5, "{snr}");
    }

    #[test]
    fn rf_channel_reproduces_levels_and_noise() {
        // A flat field comes back at its level with the set S/N.
        let (mut sig, _) = step_signal(50.0, 50.0);
        let c = TapeFm { snr_db: Some(43.0), rf: true, ..cfg() };
        apply_tape_fm(&mut sig, &c, Wire::Luma);
        let v: Vec<f32> = (100..250)
            .flat_map(|line| (crate::constants::ACTIVE_START + 60..SAMPLES_PER_LINE - 40).map(move |k| (line, k)))
            .map(|(line, k)| (sig[line * SAMPLES_PER_LINE + k] - BLANKING_V) / IRE_V)
            .collect();
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let rms = (v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / v.len() as f32).sqrt();
        assert!((mean - 50.0).abs() < 0.5, "{mean}");
        let snr = 20.0 * (100.0 / rms).log10();
        assert!((snr - 43.0).abs() < 1.0, "{snr}");
    }

    #[test]
    fn nl_emphasis_lowers_noise_on_flat_areas() {
        let rms = |nl: f32| {
            let (mut sig, _) = step_signal(50.0, 50.0);
            let c = TapeFm { snr_db: Some(40.0), nl_emphasis: nl, ..cfg() };
            apply_tape_fm(&mut sig, &c, Wire::Luma);
            let v: Vec<f32> = (crate::constants::ACTIVE_START + 40..SAMPLES_PER_LINE - 40)
                .map(|k| (sig[100 * SAMPLES_PER_LINE + k] - BLANKING_V) / IRE_V - 50.0)
                .collect();
            (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
        };
        assert!(rms(1.0) < 0.85 * rms(0.0), "{} {}", rms(1.0), rms(0.0));
    }

    #[test]
    fn dropout_counts_follow_the_rate() {
        let c = TapeFm { dropouts_per_min: Some(20.0), ..cfg() };
        let total: usize = (0..3600u64).map(|f| field_dropouts(&c, f).len()).sum();
        let counted: usize = (0..3600u64)
            .flat_map(|f| field_dropouts(&c, f))
            .filter(|(d, _)| (d.x1 - d.x0) as f32 >= 15e-6 * crate::constants::SAMPLE_RATE as f32 && d.depth_db >= 16.0)
            .count();
        // One minute of fields: about 20 counted, 350 in all.
        assert!((10..35).contains(&counted), "{counted}");
        assert!(total > 200, "{total}");
    }
}
