//! Measure the luma frequency response and the luma S/N of a parameter set.
//!
//! cargo run --release --example vhs_measure -- PRESET [key=value ...]
//!
//! PRESET is a built-in preset name or `none`; `deck=NAME` adds a deck
//! profile.  Prints the response at each test frequency, the -3 dB and
//! -10 dB points, and the S/N of a flat 50 IRE field (unweighted, picture
//! luma, row means removed), for a composite and an S-Video connection.

use ntsc_composite_simulator::constants::{ACTIVE_SAMPLES, SAMPLE_RATE};
use ntsc_composite_simulator::params::ParamValues;
use ntsc_composite_simulator::pipeline::{render_frame, resolve_effects};
use ntsc_composite_simulator::presets;

const W: usize = ACTIVE_SAMPLES;
const H: usize = 480;
/// Rows per test frequency (frame rows).
const BAND: usize = 24;

fn freqs() -> Vec<f64> {
    (1..=H / BAND).map(|k| k as f64 * 0.25e6).collect()
}

fn luma(rgb: &[u8], i: usize) -> f64 {
    (0.299 * rgb[i * 3] as f64 + 0.587 * rgb[i * 3 + 1] as f64 + 0.114 * rgb[i * 3 + 2] as f64) / 255.0 * 100.0
}

/// Horizontal sine bands around 50 IRE, `amp` IRE peak, one frequency per band.
fn sweep(amp: f64) -> Vec<u8> {
    let mut px = vec![0u8; W * H * 3];
    for (b, f) in freqs().iter().enumerate() {
        for y in b * BAND..(b + 1) * BAND {
            for x in 0..W {
                let v = 50.0 + amp * (std::f64::consts::TAU * f * x as f64 / SAMPLE_RATE).cos();
                let c = (v / 100.0 * 255.0).round().clamp(0.0, 255.0) as u8;
                px[(y * W + x) * 3..(y * W + x) * 3 + 3].fill(c);
            }
        }
    }
    px
}

/// Amplitude of the component at `f` along a row: a whole number of
/// periods from the central 600 samples, mean removed.
fn tone(rgb: &[u8], y: usize, f: f64) -> f64 {
    let period = SAMPLE_RATE / f;
    let n = ((600.0 / period).floor() * period).round().max(1.0) as usize;
    let x0 = (W - n) / 2;
    let m = (x0..x0 + n).map(|x| luma(rgb, y * W + x)).sum::<f64>() / n as f64;
    let (mut c, mut s) = (0.0, 0.0);
    for x in x0..x0 + n {
        let v = luma(rgb, y * W + x) - m;
        let ph = std::f64::consts::TAU * f * x as f64 / SAMPLE_RATE;
        c += v * ph.cos();
        s += v * ph.sin();
    }
    2.0 * (c * c + s * s).sqrt() / n as f64
}

fn render(v: &ParamValues, px: &[u8], frame: u64) -> Vec<u8> {
    let fx = resolve_effects(v);
    render_frame(px, W, H, W, H, false, v, &fx, frame)
}

fn crossing(fs: &[f64], db: &[f64], level: f64) -> Option<f64> {
    for k in 1..fs.len() {
        if db[k] <= level && db[k - 1] > level {
            let t = (db[k - 1] - level) / (db[k - 1] - db[k]);
            return Some(fs[k - 1] + t * (fs[k] - fs[k - 1]));
        }
    }
    None
}

fn measure(v: &ParamValues, label: &str) {
    let amp = 20.0;
    let px = sweep(amp);
    let fs = freqs();
    let mut resp = vec![0.0; fs.len()];
    let frames = 4;
    for fr in 0..frames {
        let out = render(v, &px, fr);
        for (b, f) in fs.iter().enumerate() {
            // Middle rows of the band, away from its edges.
            for y in b * BAND + 6..(b + 1) * BAND - 6 {
                resp[b] += tone(&out, y, *f);
            }
        }
    }
    let n = (frames as usize * (BAND - 12)) as f64;
    let db: Vec<f64> = resp.iter().map(|r| 20.0 * (r / n / amp).log10()).collect();
    let row: Vec<String> = fs.iter().zip(&db).map(|(f, d)| format!("{:.2}:{:+.1}", f / 1e6, d)).collect();
    let fmt = |o: Option<f64>| o.map_or("-".to_string(), |f| format!("{:.2} MHz", f / 1e6));
    println!("{label}: -3 dB {}, -10 dB {}", fmt(crossing(&fs, &db, -3.0)), fmt(crossing(&fs, &db, -10.0)));
    println!("  response (MHz:dB) {}", row.join(" "));

    // S/N on a flat 50 IRE field, rows 40..440 (clear of the head switch
    // and the top band), row means removed.
    let grey = vec![(0.5f64 * 255.0).round() as u8; W * H * 3];
    let (mut sum2, mut count) = (0.0, 0.0);
    for fr in 0..frames {
        let out = render(v, &grey, 100 + fr);
        for y in 40..440 {
            let row: Vec<f64> = (40..W - 40).map(|x| luma(&out, y * W + x)).collect();
            let m = row.iter().sum::<f64>() / row.len() as f64;
            sum2 += row.iter().map(|v| (v - m) * (v - m)).sum::<f64>();
            count += row.len() as f64;
        }
    }
    let rms = (sum2 / count).sqrt();
    println!("  noise {:.2} IRE rms, S/N {:.1} dB", rms, 20.0 * (100.0 / rms.max(1e-6)).log10());
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let preset = args.first().map(String::as_str).unwrap_or("none");
    let mut v = ParamValues::default();
    let mut deck = None;
    for a in &args[1.min(args.len())..] {
        let (k, val) = a.split_once('=').expect("key=value");
        if k == "deck" {
            deck = Some(val.to_string());
        } else {
            v.set_from_str(k, val).unwrap();
        }
    }
    if let Some(d) = &deck {
        v.fill_from(&presets::deck(d).expect("deck name").to_values());
    }
    if preset != "none" {
        v.fill_from(&presets::builtin(preset).expect("preset name").to_values());
    }
    v.seed = Some(42);
    let explicit_out = v.is_set("video-out");
    measure(&v, &format!("{preset} {}", if explicit_out { v.choice("video-out") } else { "composite" }));
    if !explicit_out {
        v.set_from_str("video-out", "svideo").unwrap();
        measure(&v, &format!("{preset} svideo"));
    }
}
