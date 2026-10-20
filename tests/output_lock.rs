//! Output locks.  Fixed inputs are rendered with fixed seeds and the hash
//! of every output is compared with a stored value.
//!
//! - `tests/locks/legacy.txt` holds the legacy presets, effects off and the
//!   DVD path.  These must never change: models were trained on them.
//! - `tests/locks/presets.txt` holds the current `lq`/`mq`/`hq` presets and
//!   deck profiles.  It may be updated on purpose, so that every change to
//!   them shows up in a commit.
//!
//! To rewrite a lock file, run the tests with `NTSC_UPDATE_LOCKS=presets`
//! (or `legacy`, which should never be needed).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use ntsc_composite_simulator::colorbars::generate_colorbars;
use ntsc_composite_simulator::params::ParamValues;
use ntsc_composite_simulator::pipeline::{render_frame, render_preview, resolve_effects};
use ntsc_composite_simulator::presets;

const W: usize = 320;
const H: usize = 240;
const SEED: u64 = 1234;

/// 64-bit FNV-1a.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// A synthetic picture: gradients, fine stripes, sharp coloured shapes and
/// a white box on black, moved by `t` pixels.
fn synthetic(t: usize) -> Vec<u8> {
    let mut px = vec![0u8; W * H * 3];
    for y in 0..H {
        for x in 0..W {
            let i = (y * W + x) * 3;
            let mut rgb = [(x * 255 / W) as u8, (y * 255 / H) as u8, 128u8];
            if y > H / 2 && (x / 2) % 2 == 0 {
                rgb = [220, 220, 220];
            }
            let (cx, cy) = (80 + 3 * t, 60 + t);
            if x.abs_diff(cx) < 30 && y.abs_diff(cy) < 25 {
                rgb = [200, 30, 40];
            }
            if (200..260).contains(&x) && (40..90).contains(&y) {
                rgb = if (210..250).contains(&x) && (50..80).contains(&y) { [255, 255, 255] } else { [0, 0, 0] };
            }
            px[i..i + 3].copy_from_slice(&rgb);
        }
    }
    px
}

/// Values for a preset name ("none" = no effects), optional deck and
/// extra settings.
fn values(preset: &str, deck: Option<&str>, extra: &[(&str, &str)]) -> ParamValues {
    let mut v = ParamValues::default();
    for (k, val) in extra {
        v.set_from_str(k, val).unwrap();
    }
    if let Some(d) = deck {
        v.fill_from(&presets::deck(d).unwrap().to_values());
    }
    if preset != "none" {
        v.fill_from(&presets::builtin(preset).unwrap().to_values());
    }
    v.seed = Some(SEED);
    v
}

/// Hashes of every still render for one set of values.
fn render_set(name: &str, v: &ParamValues, out: &mut BTreeMap<String, String>) {
    let fx = resolve_effects(v);
    let bars = generate_colorbars(W, H);
    let mut put = |key: String, rgb: Vec<u8>| {
        out.insert(key, format!("{:016x}", fnv1a(&rgb)));
    };
    put(format!("{name}/bars"), render_frame(&bars, W, H, W, H, false, v, &fx, 0));
    put(format!("{name}/synthetic"), render_frame(&synthetic(0), W, H, W, H, false, v, &fx, 0));
    for f in 1..4u64 {
        put(format!("{name}/frame{f}"), render_frame(&synthetic(f as usize), W, H, W, H, false, v, &fx, f));
    }
    put(format!("{name}/synthetic-640"), render_frame(&synthetic(2), W, H, 640, 480, false, v, &fx, 5));
}

fn has_ffmpeg() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok_and(|o| o.status.success())
}

/// Roundtrip a short generated clip through the CLI; the hash of the raw
/// RGB frames.
fn roundtrip(dir: &Path, input: &Path, tag: &str, args: &[&str]) -> String {
    let raw = dir.join(format!("{tag}.rgb"));
    let status = Command::new(env!("CARGO_BIN_EXE_ntsc-composite-simulator"))
        .arg("roundtrip")
        .arg(input)
        .arg("-o")
        .arg(dir.join(format!("{tag}.mkv")))
        .args(["--width", "320", "--height", "240", "--seed", "7", "--threads", "2", "--raw-copy"])
        .arg(&raw)
        .args(args)
        .status()
        .unwrap();
    assert!(status.success(), "{tag}");
    format!("{:016x}", fnv1a(&std::fs::read(&raw).unwrap()))
}

fn clip(dir: &Path) -> PathBuf {
    let input = dir.join("in.mkv");
    let status = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=24", "-frames:v", "6"])
        .args(["-c:v", "ffv1"])
        .arg(&input)
        .status()
        .unwrap();
    assert!(status.success());
    input
}

fn work_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ntsc-lock-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Compare `got` with the lock file, or rewrite it when asked to.
fn check(file: &str, update_key: &str, got: BTreeMap<String, String>) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/locks").join(file);
    let text: String = got.iter().map(|(k, v)| format!("{k} {v}\n")).collect();
    if std::env::var("NTSC_UPDATE_LOCKS").is_ok_and(|v| v.split(',').any(|s| s == update_key)) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, text).unwrap();
        return;
    }
    let want = std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("missing {}", path.display()));
    let want: BTreeMap<&str, &str> = want.lines().filter_map(|l| l.split_once(' ')).collect();
    let mut bad = Vec::new();
    for (k, v) in &got {
        match want.get(k.as_str()) {
            Some(w) if *w == v => {}
            Some(w) => bad.push(format!("{k}: {v} (locked {w})")),
            None => bad.push(format!("{k}: not in the lock file")),
        }
    }
    for k in want.keys() {
        if !got.contains_key(*k) {
            bad.push(format!("{k}: locked but not rendered"));
        }
    }
    assert!(bad.is_empty(), "{} differs:\n{}", file, bad.join("\n"));
}

#[test]
fn legacy_outputs_are_locked() {
    let mut got = BTreeMap::new();
    for preset in ["none", "legacy-lq", "legacy-mq", "legacy-hq"] {
        render_set(preset, &values(preset, None, &[]), &mut got);
        render_set(&format!("{preset}+svideo"), &values(preset, None, &[("video-out", "svideo")]), &mut got);
        render_set(&format!("{preset}+1h"), &values(preset, None, &[("comb", "1h")]), &mut got);
    }
    if has_ffmpeg() {
        let v = values("dvd-mq", None, &[]);
        let (rgb, _, _, _) = render_preview(&synthetic(0), W, H, 4.0 / 3.0, W, H, false, &v, 0).unwrap();
        got.insert("dvd-mq/synthetic".into(), format!("{:016x}", fnv1a(&rgb)));
        let v = values("dvd-mq", None, &[("dvd-output", "composite")]);
        let (rgb, _, _, _) = render_preview(&synthetic(0), W, H, 4.0 / 3.0, W, H, false, &v, 0).unwrap();
        got.insert("dvd-mq+composite/synthetic".into(), format!("{:016x}", fnv1a(&rgb)));
        let dir = work_dir("legacy");
        let input = clip(&dir);
        for preset in ["legacy-lq", "legacy-mq", "legacy-hq"] {
            got.insert(format!("{preset}/video"), roundtrip(&dir, &input, preset, &["--preset", preset]));
        }
        got.insert("legacy-mq+3d/video".into(), roundtrip(&dir, &input, "mq3d", &["--preset", "legacy-mq", "--comb", "3d"]));
        let _ = std::fs::remove_dir_all(&dir);
    } else {
        eprintln!("ffmpeg not found; DVD and video locks skipped");
        return;
    }
    check("legacy.txt", "legacy", got);
}

#[test]
fn preset_outputs_are_locked() {
    let mut got = BTreeMap::new();
    for preset in ["lq", "mq", "hq"] {
        render_set(preset, &values(preset, None, &[]), &mut got);
        render_set(&format!("{preset}+svideo"), &values(preset, None, &[("video-out", "svideo")]), &mut got);
    }
    for deck in presets::DECKS {
        render_set(&format!("mq+{}", deck.name), &values("mq", Some(deck.name), &[]), &mut got);
    }
    if has_ffmpeg() {
        let dir = work_dir("presets");
        let input = clip(&dir);
        for preset in ["lq", "mq", "hq"] {
            got.insert(format!("{preset}/video"), roundtrip(&dir, &input, preset, &["--preset", preset]));
        }
        got.insert(
            "lq+deck-svhs-tbc/video".into(),
            roundtrip(&dir, &input, "lqtbc", &["--preset", "lq", "--deck", "deck-svhs-tbc"]),
        );
        got.insert(
            "lq+deck-svhs-tbc+composite/video".into(),
            roundtrip(&dir, &input, "lqtbc3d", &["--preset", "lq", "--deck", "deck-svhs-tbc", "--video-out", "composite"]),
        );
        let _ = std::fs::remove_dir_all(&dir);
    } else {
        eprintln!("ffmpeg not found; video locks skipped");
        return;
    }
    check("presets.txt", "presets", got);
}
