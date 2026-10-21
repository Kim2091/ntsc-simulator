//! Measure how much the two fields of a frame differ on a still picture.
//!
//! cargo run --release --example field_combing -- PRESET [key=value ...]
//!
//! PRESET is a built-in preset name or `none`.  Options:
//!   deck=NAME        add a deck profile
//!   seed=N           seed (default 42)
//!   frames=N         frames rendered in order (default 8)
//!   off=a,b,...      turn stages off after the ranges are resolved (see
//!                    `turn_off` for the names)
//!   input=PATH       an image instead of the synthetic scene
//!   crop=X,Y,W,H     with out=PATH: save a 2x crop of the last frame
//!   out=PATH
//!   perframe=1       print every frame
//!   scene=PATH       save the synthetic scene and exit
//!   measure=PATH     measure every 640x480 frame of a raw RGB24 file
//!                    (a roundtrip's --raw-copy) instead of rendering
//!   any other key=value is a parameter, as on the command line.
//!
//! The synthetic scene has three panels (grey, brown-red, blue), each with
//! a dark and a light vertical bar running the full height.  For every
//! output row it finds the sub-pixel position of the twelve vertical edges
//! and the mean luma and colour of a flat patch in each panel.  A field
//! difference shows as a second difference across rows of the two fields:
//! for row r, d = v[r+1] − (v[r] + v[r+2]) / 2 compares a row with the two
//! rows of the other field around it ("inter").  The same second
//! difference within one field, v[r+2] − (v[r] + v[r+4]) / 2, is the
//! reference ("intra").  Rows are split into bands: top 40, middle,
//! bottom (above the head switch) and the head-switch rows.

use ntsc_composite_simulator::deck::TemporalDnr;
use ntsc_composite_simulator::effects::{Doc, SignalEffects};
use ntsc_composite_simulator::params::ParamValues;
use ntsc_composite_simulator::pipeline::{render_frame, resolve_effects};
use ntsc_composite_simulator::presets;

const W: usize = 640;
const H: usize = 480;
const PANEL: usize = 214;

/// Edge positions (pixel boundary) and flat patches of the synthetic scene.
fn edges() -> Vec<usize> {
    (0..3).flat_map(|p| [p * PANEL + 40, p * PANEL + 60, p * PANEL + 150, p * PANEL + 170]).collect()
}

fn patches() -> Vec<(usize, usize)> {
    (0..3).map(|p| (p * PANEL + 75, p * PANEL + 140)).collect()
}

fn scene() -> Vec<u8> {
    let base = [[128u8, 128, 128], [176, 92, 64], [64, 88, 168]];
    let mut px = vec![0u8; W * H * 3];
    for y in 0..H {
        for x in 0..W {
            let p = (x / PANEL).min(2);
            let o = x - p * PANEL;
            let rgb = if (40..60).contains(&o) {
                [24, 24, 24]
            } else if (150..170).contains(&o) {
                [230, 230, 230]
            } else {
                base[p]
            };
            px[(y * W + x) * 3..(y * W + x) * 3 + 3].copy_from_slice(&rgb);
        }
    }
    px
}

fn yuv(rgb: &[u8], i: usize) -> [f64; 3] {
    let (r, g, b) = (rgb[i * 3] as f64, rgb[i * 3 + 1] as f64, rgb[i * 3 + 2] as f64);
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    [y, b - y, r - y]
}

/// Sub-pixel position of the edge near boundary `e` on row `y`.
fn edge_pos(rgb: &[u8], y: usize, e: usize) -> Option<f64> {
    let l = |x: usize| yuv(rgb, y * W + x)[0];
    let a = (e - 12..e - 6).map(l).sum::<f64>() / 6.0;
    let b = (e + 6..e + 12).map(l).sum::<f64>() / 6.0;
    if (b - a).abs() < 20.0 {
        return None;
    }
    let mid = 0.5 * (a + b);
    let mut best: Option<f64> = None;
    for j in e - 8..e + 8 {
        let (p, q) = (l(j - 1) - mid, l(j) - mid);
        if p * q <= 0.0 && p != q {
            let x = (j - 1) as f64 + p / (p - q) + 0.5;
            if best.is_none_or(|bx| (x - e as f64).abs() < (bx - e as f64).abs()) {
                best = Some(x);
            }
        }
    }
    best
}

/// Per-row measurements: edge positions and patch means (Y, U, V).
struct Rows {
    edge: Vec<Vec<Option<f64>>>,
    level: Vec<Vec<[f64; 3]>>,
}

fn measure_rows(rgb: &[u8], synthetic: bool) -> Rows {
    let es = if synthetic { edges() } else { Vec::new() };
    let ps = if synthetic { patches() } else { Vec::new() };
    let mut edge = vec![Vec::new(); H];
    let mut level = vec![Vec::new(); H];
    for y in 0..H {
        edge[y] = es.iter().map(|&e| edge_pos(rgb, y, e)).collect();
        level[y] = ps
            .iter()
            .map(|&(a, b)| {
                let mut m = [0.0; 3];
                for x in a..b {
                    let v = yuv(rgb, y * W + x);
                    for c in 0..3 {
                        m[c] += v[c];
                    }
                }
                m.map(|v| v / (b - a) as f64)
            })
            .collect();
    }
    Rows { edge, level }
}

/// Accumulated statistics of one band of rows.
#[derive(Default, Clone)]
struct Stat {
    sum: f64,
    max: f64,
    n: f64,
    signed: f64,
}

impl Stat {
    fn add(&mut self, v: f64, sign: f64) {
        self.sum += v.abs();
        self.max = self.max.max(v.abs());
        self.signed += sign * v;
        self.n += 1.0;
    }
    fn mean(&self) -> f64 {
        if self.n > 0.0 { self.sum / self.n } else { 0.0 }
    }
    fn smean(&self) -> f64 {
        if self.n > 0.0 { self.signed / self.n } else { 0.0 }
    }
}

const BANDS: [(&str, usize, usize); 5] =
    [("t14", 0, 14), ("top", 14, 40), ("mid", 40, 436), ("bot", 436, 466), ("hs", 466, 480)];

/// Statistics per band: [inter offset, intra offset, inter luma, intra
/// luma, inter chroma, intra chroma, inter offset averaged over the
/// edges of a row].
fn stats(rows: &Rows) -> Vec<[Stat; 7]> {
    let mut out = vec![<[Stat; 7]>::default(); BANDS.len()];
    for (b, &(_, lo, hi)) in BANDS.iter().enumerate() {
        let s = &mut out[b];
        for r in lo..hi.min(H - 4) {
            // Field 2 minus field 1, so the sign of a constant offset shows.
            let sign = if r % 2 == 0 { 1.0 } else { -1.0 };
            let (mut row_sum, mut row_n) = (0.0, 0.0);
            for k in 0..rows.edge[r].len() {
                let g = |y: usize| rows.edge[y][k];
                if let (Some(a), Some(m), Some(c)) = (g(r), g(r + 1), g(r + 2)) {
                    s[0].add(m - 0.5 * (a + c), sign);
                    row_sum += sign * (m - 0.5 * (a + c));
                    row_n += 1.0;
                }
                if let (Some(a), Some(m), Some(c)) = (g(r), g(r + 2), g(r + 4)) {
                    s[1].add(m - 0.5 * (a + c), 1.0);
                }
            }
            if row_n > 0.0 {
                s[6].add(row_sum / row_n, 1.0);
            }
            for k in 0..rows.level[r].len() {
                let g = |y: usize| rows.level[y][k];
                let d2 = |a: [f64; 3], m: [f64; 3], c: [f64; 3]| -> [f64; 3] {
                    [0, 1, 2].map(|i| m[i] - 0.5 * (a[i] + c[i]))
                };
                let inter = d2(g(r), g(r + 1), g(r + 2));
                let intra = d2(g(r), g(r + 2), g(r + 4));
                s[2].add(inter[0], sign);
                s[3].add(intra[0], 1.0);
                s[4].add((inter[1] * inter[1] + inter[2] * inter[2]).sqrt(), 1.0);
                s[5].add((intra[1] * intra[1] + intra[2] * intra[2]).sqrt(), 1.0);
            }
        }
    }
    out
}

fn accumulate(total: &mut [[Stat; 7]], s: &[[Stat; 7]]) {
    for (t, s) in total.iter_mut().zip(s) {
        for (a, b) in t.iter_mut().zip(s) {
            a.sum += b.sum;
            a.max = a.max.max(b.max);
            a.n += b.n;
            a.signed += b.signed;
        }
    }
}

fn print_table(label: &str, total: &[[Stat; 7]]) {
    println!(
        "{label} band: offset px inter mean/max (signed f2-f1) | intra mean/max | row-avg inter mean/max || luma inter (signed) | intra || chroma inter | intra"
    );
    for (b, s) in BANDS.iter().zip(total) {
        println!(
            "  {:>3}: {:5.2}/{:5.2} ({:+5.2}) | {:5.2}/{:5.2} | {:5.2}/{:5.2} || {:5.2} ({:+5.2}) | {:5.2} || {:5.2} | {:5.2}",
            b.0,
            s[0].mean(),
            s[0].max,
            s[0].smean(),
            s[1].mean(),
            s[1].max,
            s[6].mean(),
            s[6].max,
            s[2].mean(),
            s[2].smean(),
            s[3].mean(),
            s[4].mean(),
            s[5].mean()
        );
    }
}

/// Turn a stage off by name.  Returns false for an unknown name.
fn turn_off(fx: &mut SignalEffects, v: &mut ParamValues, name: &str) -> bool {
    match name {
        "flagging" => fx.flagging = None,
        "head-level-step" => fx.head_level_step = None,
        "ep-interleave" => fx.ep_interleave = None,
        "wobble" => fx.wobble = None,
        "jitter" => fx.jitter = None,
        "head-switch" => fx.head_switch = None,
        "top-band" => fx.top_band = None,
        "vbi-data" => fx.vbi_data = None,
        "rf-loss" => fx.rf_loss = None,
        "tracking-bars" => fx.tracking_bars = None,
        "chroma-crosstalk" => fx.chroma_crosstalk = None,
        "chroma-vblur" => fx.chroma_vblur = None,
        "chroma-delay-v" => fx.chroma_delay_v = None,
        "chroma-delay" => fx.chroma_delay = None,
        "chroma-noise" => fx.chroma_noise = None,
        "chroma-phase-noise" => fx.chroma_phase_noise = None,
        "luma-crosstalk" => fx.luma_crosstalk = None,
        "bw-inversion" => fx.bw_inversion = None,
        "tape-snr" => fx.tape_snr = None,
        "dropouts" => fx.dropouts = None,
        "deck-nc" => fx.deck_nc = None,
        "deck-ynr" => fx.deck_ynr = None,
        "deck-cnr" => fx.deck_cnr = None,
        "deck-sharpness" => fx.deck_sharpness = None,
        "deck-dnr" => fx.deck_dnr = None,
        "nl-emphasis" => fx.nl_emphasis = None,
        "edge-ringing" => fx.edge_ringing = None,
        "generations" => fx.generations = 1,
        "v-jitter" => fx.v_jitter = None,
        "coherence" => fx.temporal_coherence = 0.0,
        "doc" => fx.doc = Doc::Off,
        "comb" => v.set_from_str("comb", "2sample").unwrap(),
        _ => return false,
    }
    true
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let preset = args.first().map(String::as_str).unwrap_or("none");
    let mut v = ParamValues::default();
    let (mut deck, mut seed, mut frames) = (None, 42u64, 8u64);
    let (mut off, mut input, mut crop, mut out, mut perframe) = (Vec::new(), None, None, None, false);
    for a in &args[1.min(args.len())..] {
        let (k, val) = a.split_once('=').expect("key=value");
        match k {
            "seed" => seed = val.parse().unwrap(),
            "deck" => deck = Some(val.to_string()),
            "frames" => frames = val.parse().unwrap(),
            "off" => off = val.split(',').filter(|s| !s.is_empty()).map(String::from).collect(),
            "input" => input = Some(val.to_string()),
            "crop" => crop = Some(val.split(',').map(|s| s.parse::<usize>().unwrap()).collect::<Vec<_>>()),
            "out" => out = Some(val.to_string()),
            "perframe" => perframe = val == "1",
            "scene" => {
                image::RgbImage::from_raw(W as u32, H as u32, scene()).unwrap().save(val).unwrap();
                return;
            }
            "measure" => {
                let raw = std::fs::read(val).unwrap();
                let mut total = vec![<[Stat; 7]>::default(); BANDS.len()];
                for frame in raw.chunks_exact(W * H * 3) {
                    accumulate(&mut total, &stats(&measure_rows(frame, true)));
                }
                print_table(&format!("raw {val}"), &total);
                return;
            }
            _ => v.set_from_str(k, val).unwrap(),
        }
    }
    if let Some(d) = &deck {
        v.fill_from(&presets::deck(d).expect("deck name").to_values());
    }
    if preset != "none" {
        v.fill_from(&presets::builtin(preset).expect("preset name").to_values());
    }
    v.seed = Some(seed);
    let mut fx = resolve_effects(&v);
    for name in &off {
        assert!(turn_off(&mut fx, &mut v, name), "unknown stage {name}");
    }
    let (px, iw, ih, synthetic) = match &input {
        Some(p) => {
            let img = image::open(p).unwrap().to_rgb8();
            let (w, h) = (img.width() as usize, img.height() as usize);
            (img.into_raw(), w, h, false)
        }
        None => (scene(), W, H, true),
    };
    let mut dnr = fx.deck_dnr.filter(|s| *s > 0.0).map(TemporalDnr::new);
    let mut total = vec![<[Stat; 7]>::default(); BANDS.len()];
    let mut last = Vec::new();
    for f in 0..frames {
        let mut rgb = render_frame(&px, iw, ih, W, H, false, &v, &fx, f);
        if let Some(d) = dnr.as_mut() {
            d.process(&mut rgb, W, H);
        }
        if synthetic {
            let s = stats(&measure_rows(&rgb, true));
            if perframe {
                let parts: Vec<String> = BANDS
                    .iter()
                    .zip(&s)
                    .map(|(b, s)| format!("{} off {:.2}/{:.2} Y {:.2}", b.0, s[0].mean(), s[0].max, s[2].mean()))
                    .collect();
                println!("  frame {f}: {}", parts.join(" | "));
            }
            accumulate(&mut total, &s);
        }
        last = rgb;
    }
    if synthetic {
        let label = format!(
            "{preset}{}{}",
            deck.as_ref().map_or(String::new(), |d| format!("+{d}")),
            if off.is_empty() { String::new() } else { format!(" off={}", off.join(",")) },
        );
        print_table(&label, &total);
        println!(
            "  (head-level-step {:?} %, ep-interleave {:?} IRE, wobble {:?} px, flagging {:?}, line-tbc {:?})",
            fx.head_level_step, fx.ep_interleave, fx.wobble, fx.flagging, fx.line_tbc
        );
    }
    if let (Some(c), Some(path)) = (crop, out) {
        let (x0, y0, cw, ch) = (c[0], c[1], c[2], c[3]);
        let mut img = image::RgbImage::new(2 * cw as u32, 2 * ch as u32);
        for y in 0..2 * ch {
            for x in 0..2 * cw {
                let i = ((y0 + y / 2) * W + x0 + x / 2) * 3;
                img.put_pixel(x as u32, y as u32, image::Rgb([last[i], last[i + 1], last[i + 2]]));
            }
        }
        img.save(&path).unwrap();
    }
}
