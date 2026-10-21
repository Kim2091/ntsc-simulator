//! Render the sample images of the README into `docs/images/`.
//!
//! cargo run --release --example readme_samples
//!
//! The source picture is a synthetic test scene generated here, and every
//! render uses a fixed seed, so the
//! images can be regenerated exactly.  The DVD presets need ffmpeg on the
//! PATH, as the DVD path itself does.

use std::path::{Path, PathBuf};

use ntsc_composite_simulator::constants::{ACTIVE_SAMPLES, SAMPLE_RATE};
use ntsc_composite_simulator::params::ParamValues;
use ntsc_composite_simulator::pipeline::{render_preview, FrameProcessor};
use ntsc_composite_simulator::presets;

const W: usize = 640;
const H: usize = 480;
const SEED: u64 = 7;

type Rgb = [f32; 3];

// ────────────────────────────────────────────────────────────────────────────
// Bitmap font: 5×7 glyphs drawn for this generator.
// ────────────────────────────────────────────────────────────────────────────

const GLYPHS: &[(char, [&str; 7])] = &[
    ('A', [" ### ", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"]),
    ('B', ["#### ", "#   #", "#   #", "#### ", "#   #", "#   #", "#### "]),
    ('C', [" ### ", "#   #", "#    ", "#    ", "#    ", "#   #", " ### "]),
    ('D', ["#### ", "#   #", "#   #", "#   #", "#   #", "#   #", "#### "]),
    ('E', ["#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#####"]),
    ('F', ["#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#    "]),
    ('G', [" ### ", "#   #", "#    ", "# ###", "#   #", "#   #", " ####"]),
    ('H', ["#   #", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"]),
    ('I', [" ### ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "]),
    ('J', ["  ###", "   # ", "   # ", "   # ", "   # ", "#  # ", " ##  "]),
    ('K', ["#   #", "#  # ", "# #  ", "##   ", "# #  ", "#  # ", "#   #"]),
    ('L', ["#    ", "#    ", "#    ", "#    ", "#    ", "#    ", "#####"]),
    ('M', ["#   #", "## ##", "# # #", "# # #", "#   #", "#   #", "#   #"]),
    ('N', ["#   #", "#   #", "##  #", "# # #", "#  ##", "#   #", "#   #"]),
    ('O', [" ### ", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "]),
    ('P', ["#### ", "#   #", "#   #", "#### ", "#    ", "#    ", "#    "]),
    ('Q', [" ### ", "#   #", "#   #", "#   #", "# # #", "#  # ", " ## #"]),
    ('R', ["#### ", "#   #", "#   #", "#### ", "# #  ", "#  # ", "#   #"]),
    ('S', [" ####", "#    ", "#    ", " ### ", "    #", "    #", "#### "]),
    ('T', ["#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  "]),
    ('U', ["#   #", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "]),
    ('V', ["#   #", "#   #", "#   #", "#   #", "#   #", " # # ", "  #  "]),
    ('W', ["#   #", "#   #", "#   #", "# # #", "# # #", "# # #", " # # "]),
    ('X', ["#   #", "#   #", " # # ", "  #  ", " # # ", "#   #", "#   #"]),
    ('Y', ["#   #", "#   #", " # # ", "  #  ", "  #  ", "  #  ", "  #  "]),
    ('Z', ["#####", "    #", "   # ", "  #  ", " #   ", "#    ", "#####"]),
    ('0', [" ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### "]),
    ('1', ["  #  ", " ##  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "]),
    ('2', [" ### ", "#   #", "    #", "   # ", "  #  ", " #   ", "#####"]),
    ('3', ["#####", "   # ", "  #  ", "   # ", "    #", "#   #", " ### "]),
    ('4', ["   # ", "  ## ", " # # ", "#  # ", "#####", "   # ", "   # "]),
    ('5', ["#####", "#    ", "#### ", "    #", "    #", "#   #", " ### "]),
    ('6', ["  ## ", " #   ", "#    ", "#### ", "#   #", "#   #", " ### "]),
    ('7', ["#####", "    #", "   # ", "  #  ", " #   ", " #   ", " #   "]),
    ('8', [" ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### "]),
    ('9', [" ### ", "#   #", "#   #", " ####", "    #", "   # ", " ##  "]),
    ('-', ["     ", "     ", "     ", "#####", "     ", "     ", "     "]),
    ('.', ["     ", "     ", "     ", "     ", "     ", " ##  ", " ##  "]),
    ('/', ["    #", "    #", "   # ", "  #  ", " #   ", "#    ", "#    "]),
    (':', ["     ", " ##  ", " ##  ", "     ", " ##  ", " ##  ", "     "]),
];

fn glyph(c: char) -> Option<&'static [&'static str; 7]> {
    GLYPHS.iter().find(|(g, _)| *g == c).map(|(_, rows)| rows)
}

/// Draw `text` with its top-left corner at (x, y), each font pixel
/// `scale`×`scale` image pixels, into an 8-bit RGB buffer `w` wide.
fn draw_text(buf: &mut [u8], w: usize, x: usize, y: usize, scale: usize, text: &str, col: [u8; 3]) {
    let h = buf.len() / (w * 3);
    for (i, c) in text.chars().enumerate() {
        let Some(rows) = glyph(c) else { continue };
        let gx = x + i * 6 * scale;
        for (ry, row) in rows.iter().enumerate() {
            for (rx, b) in row.bytes().enumerate() {
                if b != b'#' {
                    continue;
                }
                for dy in 0..scale {
                    for dx in 0..scale {
                        let (px, py) = (gx + rx * scale + dx, y + ry * scale + dy);
                        if px < w && py < h {
                            buf[(py * w + px) * 3..(py * w + px) * 3 + 3].copy_from_slice(&col);
                        }
                    }
                }
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Synthetic test scene
// ────────────────────────────────────────────────────────────────────────────

fn rgb(r: u8, g: u8, b: u8) -> Rgb {
    [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0]
}

fn lerp(a: Rgb, b: Rgb, t: f32) -> Rgb {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn hash(a: u32, b: u32) -> u32 {
    let mut h = a.wrapping_mul(0x9E37_79B1) ^ b.wrapping_mul(0x85EB_CA77);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^ (h >> 12)
}

/// Colour patches along the middle row: two skin tones, saturated colours
/// and 75 % white.
const PATCHES: [(u8, u8, u8); 10] = [
    (224, 172, 140),
    (141, 85, 54),
    (220, 30, 30),
    (240, 130, 20),
    (240, 220, 30),
    (40, 180, 60),
    (30, 190, 210),
    (30, 50, 200),
    (200, 40, 180),
    (191, 191, 191),
];

/// Top edge of the hill at column `x`.
fn hill_top(x: f32) -> f32 {
    150.0 + 12.0 * (x / 45.0).sin() + 6.0 * (x / 17.0).sin()
}

/// The smooth part of the scene at a point (pixel units), sampled several
/// times per pixel for anti-aliased edges.
fn scene_at(x: f32, y: f32) -> Rgb {
    // Sky gradient and a soft-edged sun.
    let mut c = lerp(rgb(30, 60, 140), rgb(160, 205, 240), (y / 170.0).clamp(0.0, 1.0));
    let d = ((x - 545.0).powi(2) + (y - 62.0).powi(2)).sqrt();
    c = lerp(c, rgb(255, 244, 210), 1.0 - smoothstep(22.0, 36.0, d));

    // Hill with a sharp skyline, darker towards the bottom.
    if y >= hill_top(x) && y < 300.0 {
        c = lerp(rgb(60, 130, 60), rgb(30, 80, 40), ((y - 140.0) / 160.0).clamp(0.0, 1.0));
    }

    // Tall building from the top of the frame: sharp vertical edges.
    if (20.0..110.0).contains(&x) && y < 300.0 {
        c = rgb(52, 52, 62);
        let (wx, wy) = (x - 28.0, y - 12.0);
        if wx >= 0.0 && x < 104.0 && wy >= 0.0 && y < 288.0 && wx % 19.0 < 10.0 && wy % 24.0 < 12.0 {
            let lit = hash((wx / 19.0) as u32, (wy / 24.0) as u32) % 3 != 0;
            c = if lit { rgb(235, 205, 95) } else { rgb(28, 30, 40) };
        }
    }

    // Shapes with sharp edges: red square, green circle, blue triangle.
    if (130.0..200.0).contains(&x) && (180.0..250.0).contains(&y) {
        c = rgb(215, 35, 35);
    }
    if (x - 265.0).powi(2) + (y - 215.0).powi(2) < 34.0 * 34.0 {
        c = rgb(40, 185, 70);
    }
    let (tx, ty) = (x - 350.0, y - 180.0);
    if ty >= 0.0 && y < 252.0 && tx.abs() < ty * 40.0 / 72.0 {
        c = rgb(35, 60, 210);
    }
    // Soft-edged magenta blob.
    let g = (-((x - 440.0).powi(2) + (y - 215.0).powi(2)) / (2.0 * 17.0 * 17.0)).exp();
    c = lerp(c, rgb(210, 50, 190), g);

    // Colour patches.
    if (300.0..350.0).contains(&y) {
        let (r, gg, b) = PATCHES[((x / 64.0) as usize).min(9)];
        c = rgb(r, gg, b);
    }
    c
}

/// Picture elements drawn per pixel (fine lines, test signals, text) over
/// the smooth scene.  Returns `None` where the smooth scene shows.
fn detail_at(px: usize, py: usize) -> Option<Rgb> {
    let (x, y) = (px as f64, py as f64);
    // Cycles per pixel of a frequency in Hz (640 pixels span the active line).
    let cpp = |f: f64| f / SAMPLE_RATE * ACTIVE_SAMPLES as f64 / W as f64;

    // Fine-line panel: horizontal, vertical and diagonal 1-pixel lines.
    if (490..630).contains(&px) && (168..292).contains(&py) {
        let inner = (494..626).contains(&px);
        let black = rgb(10, 10, 10);
        let white = rgb(245, 245, 245);
        let v = if inner && (172..200).contains(&py) {
            (py - 172) % 3 == 0
        } else if inner && (204..232).contains(&py) {
            (px - 494) % 3 == 0
        } else if inner && (236..264).contains(&py) {
            (px + py) % 4 == 0
        } else {
            false
        };
        return Some(if v { black } else { white });
    }

    // Multiburst: 0.5, 1, 2, 3, 3.58 and 4.2 MHz packets at 50 % grey.
    if (350..410).contains(&py) && px < 400 {
        let freqs = [0.5e6, 1.0e6, 2.0e6, 3.0e6, 3.58e6, 4.2e6];
        let k = px / 66;
        let local = px % 66;
        let v = if k < freqs.len() && (3..63).contains(&local) {
            0.5 + 0.35 * (std::f64::consts::TAU * cpp(freqs[k]) * (local as f64 - 3.0)).cos()
        } else {
            0.5
        };
        return Some([v as f32; 3]);
    }
    // Flat dark and bright areas.
    if (350..410).contains(&py) {
        return Some(if px < 520 { rgb(22, 22, 24) } else { rgb(238, 238, 236) });
    }

    // Bottom strip: dark blue background with text, and a frequency wedge
    // rising from 0.3 to 5 MHz.
    if py >= 410 {
        if (330..636).contains(&px) && (424..476).contains(&py) {
            let len = 306.0;
            let u = x - 330.0;
            let (f0, f1) = (cpp(0.3e6), cpp(5.0e6));
            let phase = f0 * u + (f1 - f0) * u * u / (2.0 * len);
            let v = 0.5 + 0.45 * (std::f64::consts::TAU * phase).cos();
            return Some([v as f32; 3]);
        }
        let _ = y;
        return Some(rgb(20, 30, 80));
    }
    None
}

/// Column of the wedge where it reaches `f` Hz.
fn wedge_x(f: f64) -> usize {
    let (f0, f1, len) = (0.3e6, 5.0e6, 306.0);
    (330.0 + (f - f0) / (f1 - f0) * len).round() as usize
}

/// The 640×480 synthetic test scene.
fn test_scene() -> Vec<u8> {
    const SS: usize = 4;
    let mut buf = vec![0u8; W * H * 3];
    for py in 0..H {
        for px in 0..W {
            let c = detail_at(px, py).unwrap_or_else(|| {
                let mut acc = [0.0f32; 3];
                for sy in 0..SS {
                    for sx in 0..SS {
                        let s = scene_at(
                            px as f32 + (sx as f32 + 0.5) / SS as f32,
                            py as f32 + (sy as f32 + 0.5) / SS as f32,
                        );
                        for k in 0..3 {
                            acc[k] += s[k];
                        }
                    }
                }
                acc.map(|v| v / (SS * SS) as f32)
            });
            for k in 0..3 {
                buf[(py * W + px) * 3 + k] = (c[k] * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    // Text.
    draw_text(&mut buf, W, 497, 272, 1, "FINE TEXT 0123", [10, 10, 10]);
    draw_text(&mut buf, W, 497, 282, 1, "ABCDEFGHIJKLMNOPQRS", [10, 10, 10]);
    draw_text(&mut buf, W, 132, 262, 2, "NTSC", [250, 250, 250]);
    draw_text(&mut buf, W, 300, 262, 2, "VHS DVD", [255, 230, 60]);
    draw_text(&mut buf, W, 12, 418, 2, "NTSC COMPOSITE", [245, 245, 245]);
    draw_text(&mut buf, W, 12, 440, 1, "THE QUICK BROWN FOX JUMPS 0123456789", [200, 200, 210]);
    draw_text(&mut buf, W, 12, 452, 2, "RED", [230, 40, 40]);
    draw_text(&mut buf, W, 60, 452, 2, "GREEN", [50, 200, 70]);
    draw_text(&mut buf, W, 132, 452, 2, "BLUE", [70, 100, 255]);
    draw_text(&mut buf, W, 192, 452, 2, "WHITE", [245, 245, 245]);
    // Multiburst and wedge labels.
    for (i, label) in ["0.5", "1", "2", "3", "3.58", "4.2"].iter().enumerate() {
        draw_text(&mut buf, W, i * 66 + 4, 341, 1, label, [20, 20, 20]);
    }
    for mhz in 1..=4 {
        let x = wedge_x(mhz as f64 * 1e6);
        draw_text(&mut buf, W, x - 2, 414, 1, &mhz.to_string(), [230, 230, 230]);
    }
    draw_text(&mut buf, W, 300, 414, 1, "MHZ", [230, 230, 230]);
    buf
}

// ────────────────────────────────────────────────────────────────────────────
// Rendering
// ────────────────────────────────────────────────────────────────────────────

/// Values of a built-in preset and/or deck profile with the fixed seed, as
/// the command line resolves them (deck first, then the preset).  `extra`
/// holds explicit `flag=value` settings.
fn values(preset: Option<&str>, deck: Option<&str>, extra: &[(&str, &str)]) -> ParamValues {
    let mut v = ParamValues::default();
    for (id, val) in extra {
        v.set_from_str(id, val).expect("valid setting");
    }
    if let Some(d) = deck {
        v.fill_from(&presets::deck(d).expect("deck name").to_values());
    }
    if let Some(p) = preset {
        v.fill_from(&presets::builtin(p).expect("preset name").to_values());
    }
    v.seed = Some(SEED);
    v
}

/// A still picture through the settings (frame 0), as `image` renders it.
fn still(src: &[u8], v: &ParamValues) -> (Vec<u8>, usize, usize) {
    let (rgb, w, h, _) = render_preview(src, W, H, W as f64 / H as f64, W, H, false, v, 0).expect("render");
    (rgb, w, h)
}

/// The same picture held for ten frames of video, as `roundtrip` renders
/// it: the last frame, after the deck's noise reduction and the 3D comb
/// have seen the frames before it.
fn held(src: &[u8], v: &ParamValues) -> Vec<u8> {
    let mut p = FrameProcessor::new(v, false, (W, H), (W, H), 0);
    let frames = vec![src.to_vec(); 10];
    p.process_batch(&frames).pop().expect("frame")
}

fn out_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("docs").join("images")
}

fn save(name: &str, rgb: &[u8], w: usize, h: usize) {
    let path = out_dir().join(name);
    image::RgbImage::from_raw(w as u32, h as u32, rgb.to_vec())
        .expect("image size")
        .save(&path)
        .unwrap_or_else(|e| panic!("cannot save {}: {}", path.display(), e));
    println!("{} ({}x{})", path.display(), w, h);
}

/// Tiles in a grid, each with a label bar above it.
fn grid(tiles: &[(&str, &[u8])], cols: usize) -> (Vec<u8>, usize, usize) {
    const BAR: usize = 22;
    const GAP: usize = 8;
    let rows = tiles.len().div_ceil(cols);
    let (gw, gh) = (cols * W + (cols - 1) * GAP, rows * (H + BAR) + (rows - 1) * GAP);
    let mut out = vec![255u8; gw * gh * 3];
    for (i, (label, px)) in tiles.iter().enumerate() {
        let (x0, y0) = ((i % cols) * (W + GAP), (i / cols) * (H + BAR + GAP));
        for y in 0..BAR {
            for x in 0..W {
                out[((y0 + y) * gw + x0 + x) * 3..((y0 + y) * gw + x0 + x) * 3 + 3].copy_from_slice(&[24, 24, 28]);
            }
        }
        draw_text(&mut out, gw, x0 + 6, y0 + 4, 2, label, [240, 240, 240]);
        for y in 0..H {
            let dst = ((y0 + BAR + y) * gw + x0) * 3;
            out[dst..dst + W * 3].copy_from_slice(&px[y * W * 3..(y + 1) * W * 3]);
        }
    }
    (out, gw, gh)
}

fn main() {
    std::fs::create_dir_all(out_dir()).expect("create docs/images");
    let scene = test_scene();
    save("source_scene.png", &scene, W, H);

    // VHS presets.
    let mut vhs = Vec::new();
    for name in ["hq", "mq", "lq"] {
        let (rgb, w, h) = still(&scene, &values(Some(name), None, &[]));
        save(&format!("preset_{name}.png"), &rgb, w, h);
        vhs.push((name, rgb));
    }
    let find = |n: &str| vhs.iter().find(|(m, _)| *m == n).map(|(_, p)| p.as_slice()).unwrap();
    let (hero, hw, hh) = grid(
        &[("SOURCE", &scene), ("HQ", find("hq")), ("MQ", find("mq")), ("LQ", find("lq"))],
        2,
    );
    save("hero.png", &hero, hw, hh);

    // DVD presets, at square output pixels so they compare with the rest.
    for name in ["dvd-hq", "dvd-mq", "dvd-mq-ee", "dvd-lq", "dvd-interlaced"] {
        let (rgb, w, h) = still(&scene, &values(Some(name), None, &[("dvd-output-size", "square")]));
        save(&format!("preset_{name}.png"), &rgb, w, h);
    }

    // Playback-deck profiles with lq, as a still shot held for ten frames.
    save("deck_none.png", &held(&scene, &values(Some("lq"), None, &[])), W, H);
    for d in presets::DECKS {
        save(&format!("{}.png", d.name), &held(&scene, &values(Some("lq"), Some(d.name), &[])), W, H);
    }
}
