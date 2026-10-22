//! Measure the source enhancement's ringing beside a vertical edge.
//!
//! cargo run --release --example ringing_measure -- [key=value ...]
//!
//! Options are parameters as on the command line (`edge-ringing`,
//! `ringing-freq`, `ringing-direction`, ...; default `edge-ringing=0.8`),
//! plus `preset=NAME` and `pole=PATH` to save a 2x crop of a thin dark pole
//! on a flat background.  Without a preset the output is S-Video unless
//! `video-out` is given, so no comb filter is involved.
//!
//! The scene is a dark-to-light step at x = 320 of a 640x480 frame.  The
//! ringing is the decoded luma minus the same render without enhancement,
//! averaged over rows 100-380 and decoded at 2560 wide for sub-pixel
//! detail.  Reported: its energy on each side of the edge (2 to 40 px
//! away), the share on the right, and the profile around the edge.  The
//! unit test `causal_ringing_trails_the_edge_at_the_set_spacing` measures
//! the effect alone, with the ripple spacing.

use ntsc_composite_simulator::params::ParamValues;
use ntsc_composite_simulator::pipeline::{render_frame, resolve_effects};
use ntsc_composite_simulator::presets;

const W: usize = 640;
const H: usize = 480;
const EDGE: usize = 320;

/// Decoded width for the measurement (4 samples per output pixel).
const MW: usize = 4 * W;

fn luma_row(rgb: &[u8]) -> Vec<f64> {
    (0..MW)
        .map(|x| {
            (100..380)
                .map(|y| {
                    let i = (y * MW + x) * 3;
                    0.299 * rgb[i] as f64 + 0.587 * rgb[i + 1] as f64 + 0.114 * rgb[i + 2] as f64
                })
                .sum::<f64>()
                / 280.0
        })
        .collect()
}

fn render(v: &ParamValues, px: &[u8], out_w: usize) -> Vec<u8> {
    let fx = resolve_effects(v);
    render_frame(px, W, H, out_w, H, false, v, &fx, 0)
}

fn main() {
    let mut v = ParamValues::default();
    let (mut preset, mut pole) = (None, None);
    for a in std::env::args().skip(1) {
        let (k, val) = a.split_once('=').expect("key=value");
        match k {
            "preset" => preset = Some(val.to_string()),
            "pole" => pole = Some(val.to_string()),
            _ => v.set_from_str(k, val).unwrap(),
        }
    }
    if let Some(p) = &preset {
        v.fill_from(&presets::builtin(p).expect("preset").to_values());
    } else if !v.is_set("edge-ringing") {
        v.set_from_str("edge-ringing", "0.8").unwrap();
    }
    if preset.is_none() && !v.is_set("video-out") {
        v.set_from_str("video-out", "svideo").unwrap();
    }
    v.seed = Some(5);

    let step: Vec<u8> = (0..W * H).flat_map(|i| if i % W < EDGE { [50u8; 3] } else { [180u8; 3] }).collect();
    let mut plain = v.clone();
    plain.set_from_str("edge-ringing", "0").unwrap();
    let (a, b) = (luma_row(&render(&v, &step, MW)), luma_row(&render(&plain, &step, MW)));
    let r: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x - y).collect();
    let e = 4 * EDGE;
    let energy = |xs: std::ops::Range<usize>| xs.map(|x| r[x] * r[x]).sum::<f64>() / 4.0;
    let (left, right) = (energy(e - 160..e - 8), energy(e + 8..e + 160));
    let fx = resolve_effects(&v);
    println!(
        "edge-ringing {:?} freq {} MHz direction {}: left {:.2}, right {:.2} ({:.1} % right)",
        fx.edge_ringing,
        fx.ringing_freq,
        fx.ringing_direction,
        left,
        right,
        100.0 * right / (left + right).max(1e-12),
    );
    let profile: Vec<String> = (EDGE - 8..EDGE + 16).map(|x| format!("{:+.1}", r[4 * x + 2])).collect();
    println!("  ringing at x = {}..{}: {}", EDGE - 8, EDGE + 15, profile.join(" "));

    if let Some(path) = pole {
        // A dark pole 3 px wide on a light blue-grey background.
        let px: Vec<u8> = (0..W * H)
            .flat_map(|i| if (300..303).contains(&(i % W)) { [40u8, 40, 45] } else { [170, 190, 205] })
            .collect();
        let out = render(&v, &px, W);
        let (x0, y0, cw, ch) = (270, 200, 80, 60);
        let mut img = image::RgbImage::new(2 * cw as u32, 2 * ch as u32);
        for y in 0..2 * ch {
            for x in 0..2 * cw {
                let i = ((y0 + y / 2) * W + x0 + x / 2) * 3;
                img.put_pixel(x as u32, y as u32, image::Rgb([out[i], out[i + 1], out[i + 2]]));
            }
        }
        img.save(path).unwrap();
    }
}
