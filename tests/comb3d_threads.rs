//! The 3D comb reads earlier frames, which may sit in an earlier batch of
//! parallel frames.  Its output must not depend on the batch size, which
//! follows `--threads`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn has_ffmpeg() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok_and(|o| o.status.success())
}

fn work_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ntsc-comb3d-threads-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Roundtrip `input` with the 3D comb on `threads` threads; the RGB frames.
fn roundtrip(dir: &Path, input: &Path, threads: usize, extra: &[&str]) -> Vec<u8> {
    let raw = dir.join(format!("out{}.rgb", threads));
    let status = Command::new(env!("CARGO_BIN_EXE_ntsc-composite-simulator"))
        .arg("roundtrip")
        .arg(input)
        .arg("-o")
        .arg(dir.join(format!("out{}.mkv", threads)))
        .args(["--width", "320", "--height", "240", "--seed", "7", "--preset", "lq", "--comb", "3d"])
        .args(["--threads", &threads.to_string(), "--raw-copy"])
        .arg(&raw)
        .args(extra)
        .status()
        .unwrap();
    assert!(status.success());
    std::fs::read(&raw).unwrap()
}

#[test]
fn comb_3d_output_does_not_depend_on_threads() {
    if !has_ffmpeg() {
        eprintln!("ffmpeg not found; skipping");
        return;
    }
    let dir = work_dir();
    let input = dir.join("in.mkv");
    let status = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=24", "-frames:v", "8"])
        .args(["-c:v", "ffv1"])
        .arg(&input)
        .status()
        .unwrap();
    assert!(status.success());
    for extra in [&[][..], &["--telecine"][..]] {
        let one = roundtrip(&dir, &input, 1, extra);
        assert!(!one.is_empty());
        for threads in [2, 3] {
            assert!(roundtrip(&dir, &input, threads, extra) == one, "{:?} at {} threads", extra, threads);
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
