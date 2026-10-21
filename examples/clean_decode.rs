//! Measure how cleanly the composite path reproduces colour bars.
//!
//! cargo run --release --example clean_decode -- [key=value ...]
//!
//! Options are parameters as on the command line (for example `comb=1h`,
//! `noise=0.02`, `preset=legacy-mq`), plus `frames=N` (default 4) and
//! `out=PATH` to save a 2x crop of the top bars' transitions.
//!
//! Reports, from 640x480 colour bars decoded at 640x480:
//! - per bar: hue error (degrees) and saturation ratio of the bar centre;
//! - chroma transition width (10-90 % of the change of the I/Q vector,
//!   pixels at 640) at each bar boundary, and the largest overshoot;
//! - cross-colour: the largest chroma magnitude (8-bit I/Q units) within
//!   12 px of the black-to-white edge of the castellation row, where the
//!   source has no colour at all;
//! - dot crawl: the largest and mean luma change between consecutive
//!   frames of the still picture (8 px clear of the picture's sides).

use ntsc_composite_simulator::colorbars::generate_colorbars;
use ntsc_composite_simulator::params::ParamValues;
use ntsc_composite_simulator::pipeline::{render_frame, resolve_effects};
use ntsc_composite_simulator::presets;

const W: usize = 640;
const H: usize = 480;

fn yiq(rgb: &[u8], i: usize) -> [f64; 3] {
    let (r, g, b) = (rgb[i * 3] as f64, rgb[i * 3 + 1] as f64, rgb[i * 3 + 2] as f64);
    [
        0.299 * r + 0.587 * g + 0.114 * b,
        0.595901 * r - 0.274557 * g - 0.321344 * b,
        0.211537 * r - 0.522736 * g + 0.311200 * b,
    ]
}

fn mean_yiq(rgb: &[u8], x0: usize, x1: usize, y0: usize, y1: usize) -> [f64; 3] {
    let mut m = [0.0; 3];
    for y in y0..y1 {
        for x in x0..x1 {
            let v = yiq(rgb, y * W + x);
            for c in 0..3 {
                m[c] += v[c];
            }
        }
    }
    m.map(|v| v / ((x1 - x0) * (y1 - y0)) as f64)
}

fn main() {
    let mut v = ParamValues::default();
    let (mut frames, mut out, mut preset) = (4u64, None, None);
    for a in std::env::args().skip(1) {
        let (k, val) = a.split_once('=').expect("key=value");
        match k {
            "frames" => frames = val.parse().unwrap(),
            "out" => out = Some(val.to_string()),
            "preset" => preset = Some(val.to_string()),
            _ => v.set_from_str(k, val).unwrap(),
        }
    }
    if let Some(p) = &preset {
        v.fill_from(&presets::builtin(p).expect("preset").to_values());
    }
    v.seed = Some(1);
    let fx = resolve_effects(&v);
    let src = generate_colorbars(W, H);
    let rendered: Vec<Vec<u8>> = (0..frames).map(|f| render_frame(&src, W, H, W, H, false, &v, &fx, f)).collect();
    let rgb = &rendered[0];
    let bw = W / 7;
    let names = ["white", "yellow", "cyan", "green", "magenta", "red", "blue"];

    // Bar centres.
    let mut parts = Vec::new();
    let (mut worst_hue, mut worst_sat) = (0.0f64, 0.0f64);
    for (i, name) in names.iter().enumerate() {
        let (x0, x1) = (i * bw + bw / 4, i * bw + 3 * bw / 4);
        let a = mean_yiq(&src, x0, x1, 60, 240);
        let b = mean_yiq(rgb, x0, x1, 60, 240);
        let (ca, cb) = (a[1].hypot(a[2]), b[1].hypot(b[2]));
        if ca > 5.0 {
            let mut dh = (b[2].atan2(b[1]) - a[2].atan2(a[1])).to_degrees();
            if dh > 180.0 {
                dh -= 360.0;
            }
            if dh < -180.0 {
                dh += 360.0;
            }
            worst_hue = worst_hue.max(dh.abs());
            worst_sat = worst_sat.max((cb / ca - 1.0).abs());
            parts.push(format!("{name} {dh:+.1}° {:.3}× Y{:+.1}", cb / ca, b[0] - a[0]));
        } else {
            parts.push(format!("{name} chroma {cb:.1} Y{:+.1}", b[0] - a[0]));
        }
    }
    println!("bars: {}", parts.join(", "));
    println!("  worst hue error {worst_hue:.1}°, worst saturation error {:.1} %", 100.0 * worst_sat);

    // Transitions along the mean of rows 100-140.
    let row: Vec<[f64; 3]> = (0..W).map(|x| mean_yiq(rgb, x, x + 1, 100, 140)).collect();
    let (mut widths, mut overs) = (Vec::new(), Vec::new());
    for i in 1..7 {
        let e = i * bw;
        let a = mean_yiq(rgb, e - 30, e - 20, 100, 140);
        let b = mean_yiq(rgb, e + 20, e + 30, 100, 140);
        let (dx, dy) = (b[1] - a[1], b[2] - a[2]);
        let n2 = dx * dx + dy * dy;
        if n2 < 100.0 {
            continue;
        }
        let t: Vec<f64> = (e - 20..e + 20).map(|x| ((row[x][1] - a[1]) * dx + (row[x][2] - a[2]) * dy) / n2).collect();
        let cross = |lvl: f64| {
            (1..t.len()).find(|&k| t[k - 1] < lvl && t[k] >= lvl).map(|k| k as f64 - 1.0 + (lvl - t[k - 1]) / (t[k] - t[k - 1]))
        };
        if let (Some(p10), Some(p90)) = (cross(0.1), cross(0.9)) {
            widths.push(p90 - p10);
        }
        let over = t.iter().fold(0.0f64, |m, &v| m.max(v - 1.0).max(-v));
        overs.push(over);
    }
    let mean_w = widths.iter().sum::<f64>() / widths.len().max(1) as f64;
    let max_w = widths.iter().fold(0.0f64, |m, &v| m.max(v));
    let max_o = overs.iter().fold(0.0f64, |m, &v| m.max(v));
    println!(
        "  chroma transition 10-90 %: mean {mean_w:.1} px, max {max_w:.1} px; largest overshoot {:.0} %",
        100.0 * max_o
    );

    // Cross-colour at the castellation row's black-to-white edge (x = 6 bw).
    let e = 6 * bw;
    let mut xc = 0.0f64;
    for x in e - 12..e + 12 {
        let c = mean_yiq(rgb, x, x + 1, 330, 350);
        xc = xc.max(c[1].hypot(c[2]));
    }
    println!("  cross-colour at a black-to-white edge: {xc:.1} (I/Q, 8-bit)");

    // Dot crawl: luma change between consecutive frames.
    let (mut dmax, mut dsum, mut n, mut at) = (0.0f64, 0.0f64, 0.0f64, 0usize);
    for p in rendered.windows(2) {
        // Away from the left and right picture edges.
        for i in (0..W * H).filter(|i| (8..W - 8).contains(&(i % W))) {
            let d = (yiq(&p[0], i)[0] - yiq(&p[1], i)[0]).abs();
            if d > dmax {
                at = i;
            }
            dmax = dmax.max(d);
            dsum += d;
            n += 1.0;
        }
    }
    if n > 0.0 {
        println!("  frame-to-frame luma change: mean {:.2}, max {:.1} at ({}, {})", dsum / n, dmax, at % W, at / W);
    }

    if let Some(path) = out {
        let (x0, y0, cw, ch) = (60, 100, 280, 40);
        let mut img = image::RgbImage::new(2 * cw as u32, 2 * ch as u32);
        for y in 0..2 * ch {
            for x in 0..2 * cw {
                let i = ((y0 + y / 2) * W + x0 + x / 2) * 3;
                img.put_pixel(x as u32, y as u32, image::Rgb([rgb[i], rgb[i + 1], rgb[i + 2]]));
            }
        }
        img.save(path).unwrap();
    }
}
