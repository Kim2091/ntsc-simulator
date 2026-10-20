//! Processing pipeline shared by the command line and the GUI: images,
//! colour bars and video roundtrips through the composite signal path.
//!
//! Progress, log messages and cancellation go through the [`Progress`]
//! trait so the CLI can show a progress bar and the GUI a progress dialog.

use std::cell::RefCell;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use rayon::prelude::*;

use crate::colorbars;
use crate::constants::SAMPLE_RATE;
use crate::deck::TemporalDnr;
use crate::decoder::Decoder;
use crate::dvd::{self, DvdConfig, DvdOutput};
use crate::effects::{SignalEffects, TapeModel};
use crate::encoder::{Encoder, Wire};
use crate::params::{frame_rng, ParamValues};
use crate::profile;

pub const VIDEO_EXTENSIONS: &[&str] =
    &["mp4", "mkv", "mov", "avi", "wmv", "flv", "webm", "m4v", "ts", "mts", "m2ts", "mpg", "mpeg"];
pub const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp"];

/// Error message used when a job is cancelled through [`Progress`].
pub const CANCELLED: &str = "cancelled";

// ────────────────────────────────────────────────────────────────────────────
// Progress reporting
// ────────────────────────────────────────────────────────────────────────────

/// Receives log lines and progress, and can cancel a running job.
pub trait Progress: Sync {
    /// A human-readable status line.
    fn log(&self, _msg: &str) {}
    /// A file starts; `total_frames` is 0 when unknown.
    fn start(&self, _total_frames: usize, _label: &str) {}
    /// `frames` more frames are done; `rate` is e.g. "42.0 fps".
    fn advance(&self, _frames: u64, _rate: &str) {}
    /// The current file is finished.
    fn finish(&self) {}
    /// Polled between batches; returning `true` stops the job.
    fn is_cancelled(&self) -> bool {
        false
    }
}

/// A [`Progress`] that ignores everything.
pub struct Silent;
impl Progress for Silent {}

fn check_cancel(progress: &dyn Progress) -> Result<()> {
    if progress.is_cancelled() {
        Err(anyhow!(CANCELLED))
    } else {
        Ok(())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// File helpers
// ────────────────────────────────────────────────────────────────────────────

fn has_extension(path: &Path, extensions: &[&str]) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| extensions.iter().any(|&ex| ex.eq_ignore_ascii_case(ext)))
}

pub fn is_video_path(path: &Path) -> bool {
    has_extension(path, VIDEO_EXTENSIONS)
}

pub fn is_image_path(path: &Path) -> bool {
    has_extension(path, IMAGE_EXTENSIONS)
}

/// List files in a directory filtered by extension (case-insensitive), sorted by name.
pub fn iter_files(dir: &Path, extensions: &[&str]) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|ft| ft.is_file()).unwrap_or(false))
        .filter(|e| has_extension(&e.path(), extensions))
        .map(|e| e.path())
        .collect();
    files.sort_by(|a, b| {
        a.file_name().unwrap().to_ascii_lowercase().cmp(&b.file_name().unwrap().to_ascii_lowercase())
    });
    files
}

/// Build an output path by joining the output directory with the input filename.
fn batch_output_path(output_dir: &Path, input_path: &Path) -> PathBuf {
    output_dir.join(input_path.file_name().unwrap())
}

/// A command for an external tool that never opens a console window.
fn tool_command(program: &str) -> Command {
    #[allow(unused_mut)]
    let mut cmd = Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

// ────────────────────────────────────────────────────────────────────────────
// Single frames
// ────────────────────────────────────────────────────────────────────────────

/// Resolve the parameter ranges for one file.
pub fn resolve_effects(values: &ParamValues) -> SignalEffects {
    let mut rng = values.file_rng();
    values.to_signal_effects(&mut rng)
}

/// Encode an RGB frame, apply the effects for frame number `frame`, and
/// decode it to `out_w`×`out_h` RGB.
#[allow(clippy::too_many_arguments)]
pub fn render_frame(
    pixels: &[u8],
    w: usize,
    h: usize,
    out_w: usize,
    out_h: usize,
    comb_1h: bool,
    values: &ParamValues,
    fx: &SignalEffects,
    frame: u64,
) -> Vec<u8> {
    let mut encoder = Encoder::new();
    let mut decoder = Decoder::new();
    let mut buf = Vec::new();
    let mut buf_c = Vec::new();
    analog_frame(
        &mut encoder, &mut decoder, &mut buf, &mut buf_c, pixels, None, Some([(pixels, None), (pixels, None)]), w, h, out_w, out_h,
        frame as u32, comb_1h, values, fx, fx.is_active(),
    )
}

/// Encode (one or two fields' sources) to the analog signal, apply the
/// effects for frame `fnum`, and decode, over a composite or S-Video
/// connection.  `buf`/`buf_c` are reusable scratch buffers.
/// Sources (field 1, field 2) of the previous frame and the one before
/// it, for the 3D comb.
pub type PrevFrame<'a> = Option<[(&'a [u8], Option<&'a [u8]>); 2]>;

/// Decoder comb filter for a composite connection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Comb {
    TwoSample,
    OneLine,
    Frame3d,
    Trap,
}

/// The comb in use: `--comb`, or `--comb-1h` for compatibility.
pub fn comb_mode(values: &ParamValues, comb_1h: bool) -> Comb {
    match values.choice("comb") {
        _ if comb_1h => Comb::OneLine,
        "1h" => Comb::OneLine,
        "3d" => Comb::Frame3d,
        "trap" => Comb::Trap,
        _ => Comb::TwoSample,
    }
}

#[allow(clippy::too_many_arguments)]
fn analog_frame(
    enc: &mut Encoder,
    dec: &mut Decoder,
    buf: &mut Vec<f32>,
    buf_c: &mut Vec<f32>,
    f1: &[u8],
    f2: Option<&[u8]>,
    prev: PrevFrame,
    in_w: usize,
    in_h: usize,
    out_w: usize,
    out_h: usize,
    fnum: u32,
    comb_1h: bool,
    values: &ParamValues,
    fx: &SignalEffects,
    effects_active: bool,
) -> Vec<u8> {
    let legacy = values.flag("legacy-burst-phase");
    set_capture_peaking(dec, fx);
    let encode = |enc: &mut Encoder, wire: Wire, out: &mut Vec<f32>, f1: &[u8], f2: Option<&[u8]>, fnum: u32| {
        enc.wire = wire;
        let s = match f2 {
            None => enc.encode_frame(f1, in_w, in_h, fnum),
            Some(f2) => enc.encode_frame_interlaced(f1, in_w, in_h, f2, fnum),
        };
        out.clear();
        out.extend_from_slice(s);
        enc.wire = Wire::Composite;
    };
    if values.choice("video-out") == "svideo" {
        enc.encode_svideo(f1, f2, in_w, in_h, fnum, buf, buf_c);
        if effects_active {
            fx.apply_svideo(buf, buf_c, SAMPLE_RATE, &mut frame_rng(values.seed, fnum as u64), fnum as u64);
        }
        dec.decode_svideo(buf, buf_c, out_w, out_h, legacy).to_vec()
    } else {
        encode(enc, Wire::Composite, buf, f1, f2, fnum);
        if effects_active {
            let mut rng = frame_rng(values.seed, fnum as u64);
            fx.apply(buf, SAMPLE_RATE, &mut rng, fnum as u64);
        }
        match (comb_mode(values, comb_1h), prev) {
            (Comb::Frame3d, Some([(a1, a2), (b1, b2)])) => {
                // The two previous frames as received: same encoder, their
                // own frame numbers and effects.  At the start of a video
                // the following frame numbers stand in.
                let (n1, n2) = if fnum >= 2 { (fnum - 1, fnum - 2) } else { (fnum + 1, fnum + 2) };
                let mut prev2 = Vec::new();
                encode(enc, Wire::Composite, buf_c, a1, a2, n1);
                encode(enc, Wire::Composite, &mut prev2, b1, b2, n2);
                if effects_active {
                    fx.apply(buf_c, SAMPLE_RATE, &mut frame_rng(values.seed, n1 as u64), n1 as u64);
                    fx.apply(&mut prev2, SAMPLE_RATE, &mut frame_rng(values.seed, n2 as u64), n2 as u64);
                }
                dec.decode_frame_3d(buf, buf_c, &prev2, out_w, out_h, legacy, comb_luma_motion(fx)).to_vec()
            }
            (Comb::OneLine, _) => dec.decode_frame(buf, out_w, out_h, true, legacy).to_vec(),
            (Comb::Trap, _) => dec.decode_frame_trap(buf, out_w, out_h, legacy).to_vec(),
            _ => dec.decode_frame(buf, out_w, out_h, false, legacy).to_vec(),
        }
    }
}

/// Log the `--profile` table, if timing is on.
fn log_profile(frames: u64, progress: &dyn Progress) {
    if profile::enabled() {
        progress.log(&format!("Profile (CPU time summed over threads):
{}", profile::report(frames)));
    }
}

fn save_rgb(path: &Path, w: usize, h: usize, rgb: Vec<u8>) -> Result<()> {
    let img = image::RgbImage::from_raw(w as u32, h as u32, rgb).context("Failed to create output image")?;
    img.save(path).with_context(|| format!("Cannot save image '{}'", path.display()))
}

fn log_sampled(values: &ParamValues, fx: &SignalEffects, progress: &dyn Progress) {
    if values.has_any_range() {
        let parts = values.sampled_summary(fx);
        if !parts.is_empty() {
            progress.log(&format!("  Randomized: {}", parts.join(", ")));
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Images
// ────────────────────────────────────────────────────────────────────────────

/// Process one image file.  `width`/`height` default to the input size.
pub fn process_image_file(
    input: &Path,
    output: &Path,
    width: Option<u32>,
    height: Option<u32>,
    comb_1h: bool,
    values: &ParamValues,
    progress: &dyn Progress,
    prefix: &str,
) -> Result<()> {
    let img = image::open(input).with_context(|| format!("Cannot open image '{}'", input.display()))?;
    let img_rgb = img.to_rgb8();
    let (w, h) = (img_rgb.width() as usize, img_rgb.height() as usize);
    let out_w = width.map(|v| v as usize).unwrap_or(w);
    let out_h = height.map(|v| v as usize).unwrap_or(h);

    let (fx, dvd_cfg) = resolve_all(values);
    log_sampled(values, &fx, progress);
    if !prefix.is_empty() {
        progress.log(prefix);
    }
    progress.log(&format!("Input: {} ({}x{})", input.display(), w, h));

    let t0 = Instant::now();
    let (rgb, out_w, out_h) = match &dvd_cfg {
        Some(cfg) => {
            log_dvd(cfg, progress);
            dvd_still(img_rgb.as_raw(), w, h, w as f64 / h as f64, cfg, values, &fx, comb_1h, 0)?
        }
        None => (render_frame(img_rgb.as_raw(), w, h, out_w, out_h, comb_1h, values, &fx, 0), out_w, out_h),
    };
    progress.log(&format!("  Processing: {:.1} ms", t0.elapsed().as_secs_f64() * 1000.0));

    save_rgb(output, out_w, out_h, rgb)?;
    progress.log(&format!("Output: {} ({}x{})", output.display(), out_w, out_h));
    Ok(())
}

/// Process an image file, or every image in a directory (`output` is then
/// a directory; each file gets its own seed derived from the base seed).
pub fn process_images(
    input: &Path,
    output: &Path,
    width: Option<u32>,
    height: Option<u32>,
    comb_1h: bool,
    values: &ParamValues,
    progress: &dyn Progress,
) -> Result<()> {
    if !input.is_dir() {
        return process_image_file(input, output, width, height, comb_1h, values, progress, "");
    }
    let files = iter_files(input, IMAGE_EXTENSIONS);
    if files.is_empty() {
        bail!("No image files found in '{}'", input.display());
    }
    std::fs::create_dir_all(output)?;
    progress.log(&format!("Batch image: {} file(s) -> '{}'", files.len(), output.display()));
    progress.start(files.len(), "Images");
    for (i, path) in files.iter().enumerate() {
        check_cancel(progress)?;
        let prefix = format!("[{}/{}] {}", i + 1, files.len(), path.file_name().unwrap().to_string_lossy());
        let out_path = batch_output_path(output, path);
        process_image_file(path, &out_path, width, height, comb_1h, &values.for_file(i), progress, &prefix)?;
        progress.advance(1, "");
    }
    progress.finish();
    Ok(())
}

/// Generate SMPTE colour bars and run them through the pipeline.
pub fn process_colorbars(
    output: &Path,
    save_source: Option<&Path>,
    width: u32,
    height: u32,
    comb_1h: bool,
    values: &ParamValues,
    progress: &dyn Progress,
) -> Result<()> {
    let (w, h) = (width as usize, height as usize);
    progress.log(&format!("Generating SMPTE color bars ({}x{})...", w, h));
    let bars_rgb = colorbars::generate_colorbars(w, h);

    if let Some(src_path) = save_source {
        save_rgb(src_path, w, h, bars_rgb.clone())?;
        progress.log(&format!("Source pattern saved: {}", src_path.display()));
    }

    let (fx, dvd_cfg) = resolve_all(values);
    log_sampled(values, &fx, progress);
    let (rgb, ow, oh) = match &dvd_cfg {
        Some(cfg) => {
            log_dvd(cfg, progress);
            dvd_still(&bars_rgb, w, h, w as f64 / h as f64, cfg, values, &fx, comb_1h, 0)?
        }
        None => (render_frame(&bars_rgb, w, h, w, h, comb_1h, values, &fx, 0), w, h),
    };
    save_rgb(output, ow, oh, rgb)?;
    progress.log(&format!("Output: {} ({}x{})", output.display(), ow, oh));
    Ok(())
}

// ────────────────────────────────────────────────────────────────────────────
// Video
// ────────────────────────────────────────────────────────────────────────────

/// Settings for a video roundtrip.
#[derive(Clone, Debug)]
pub struct VideoOptions {
    pub width: u32,
    pub height: u32,
    pub comb_1h: bool,
    /// x264 CRF quality (0 = lossless, 51 = worst); only with `lossy`.
    pub crf: u32,
    /// x264 preset, e.g. "fast".
    pub preset: String,
    /// Lossy x264 4:2:0 output (`crf`, `preset`).  The default is lossless
    /// RGB: FFV1 in .mkv/.avi, x264 RGB QP 0 in .mp4/.mov.
    pub lossy: bool,
    /// Also write the exact RGB24 frames sent to the encoder to this file
    /// (for verifying that the output is bit-exact).
    pub raw_copy: Option<PathBuf>,
    /// Parallel worker threads (default: all logical cores).
    pub threads: Option<usize>,
    /// 3:2 pulldown (24p film to 29.97 fps interlaced).
    pub telecine: bool,
}

impl Default for VideoOptions {
    fn default() -> Self {
        Self {
            width: 640,
            height: 480,
            comb_1h: false,
            crf: 17,
            preset: "fast".into(),
            lossy: false,
            raw_copy: None,
            threads: None,
            telecine: false,
        }
    }
}

pub fn resolve_threads(threads: Option<usize>) -> usize {
    threads.unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4))
}

fn init_thread_pool(threads: Option<usize>) -> usize {
    let n = resolve_threads(threads);
    rayon::ThreadPoolBuilder::new().num_threads(n).build_global().ok();
    n
}

/// Roundtrip a video file, or every video in a directory (`output` is then
/// a directory).
pub fn process_videos(
    input: &Path,
    output: &Path,
    opts: &VideoOptions,
    values: &ParamValues,
    progress: &dyn Progress,
) -> Result<()> {
    if !input.is_dir() {
        return process_video_file(input, output, opts, values, progress, "");
    }
    let files = iter_files(input, VIDEO_EXTENSIONS);
    if files.is_empty() {
        bail!("No video files found in '{}'", input.display());
    }
    std::fs::create_dir_all(output)?;
    progress.log(&format!("Batch roundtrip: {} file(s) -> '{}'", files.len(), output.display()));
    init_thread_pool(opts.threads);
    for (i, path) in files.iter().enumerate() {
        check_cancel(progress)?;
        let prefix = format!("[{}/{}] {}", i + 1, files.len(), path.file_name().unwrap().to_string_lossy());
        let out_path = batch_output_path(output, path);
        process_video_file(path, &out_path, opts, &values.for_file(i), progress, &prefix)?;
    }
    Ok(())
}

/// Roundtrip one video file.
pub fn process_video_file(
    input: &Path,
    output: &Path,
    opts: &VideoOptions,
    values: &ParamValues,
    progress: &dyn Progress,
    prefix: &str,
) -> Result<()> {
    // Give unseeded runs a random seed, so per-frame effects can be
    // re-created exactly (the 3D comb re-encodes the previous frame).
    let seeded;
    let values = if values.seed.is_none() {
        let mut v = values.clone();
        v.seed = Some(rand::random());
        seeded = v;
        &seeded
    } else {
        values
    };
    if values.choice("format") == "dvd" {
        dvd_roundtrip(input, output, opts, values, progress, prefix)
    } else if opts.telecine {
        roundtrip_telecine(input, output, opts, values, progress, prefix)
    } else {
        roundtrip(input, output, opts, values, progress, prefix)
    }
}

thread_local! {
    static TL_ENCODER: RefCell<Encoder> = RefCell::new(Encoder::new());
    static TL_DECODER: RefCell<Decoder> = RefCell::new(Decoder::new());
    static TL_SIGNAL_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static TL_SIGNAL_BUF_C: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Encode (one or two fields' sources), apply effects, decode — on the
/// calling worker thread's own encoder/decoder.
#[allow(clippy::too_many_arguments)]
fn process_frame_tl(
    f1: &[u8],
    f2: Option<&[u8]>,
    prev: PrevFrame,
    in_w: usize,
    in_h: usize,
    out_w: usize,
    out_h: usize,
    fnum: u32,
    comb_1h: bool,
    values: &ParamValues,
    fx: &SignalEffects,
    effects_active: bool,
) -> Vec<u8> {
    TL_ENCODER.with(|enc| {
        TL_DECODER.with(|dec| {
            TL_SIGNAL_BUF.with(|sb| {
                TL_SIGNAL_BUF_C.with(|sc| {
                    analog_frame(
                        &mut enc.borrow_mut(), &mut dec.borrow_mut(), &mut sb.borrow_mut(), &mut sc.borrow_mut(),
                        f1, f2, prev, in_w, in_h, out_w, out_h, fnum, comb_1h, values, fx, effects_active,
                    )
                })
            })
        })
    })
}

/// The composite signal of frame `fnum` as received (encoded, with the
/// frame's effects), as `analog_frame` makes it, on the calling worker
/// thread's encoder.
#[allow(clippy::too_many_arguments)]
fn received_signal_tl(
    src: &[u8],
    in_w: usize,
    in_h: usize,
    fnum: u32,
    values: &ParamValues,
    fx: &SignalEffects,
    effects_active: bool,
) -> Vec<f32> {
    let mut signal = TL_ENCODER.with(|enc| {
        let mut enc = enc.borrow_mut();
        enc.wire = Wire::Composite;
        enc.encode_frame(src, in_w, in_h, fnum).to_vec()
    });
    if effects_active {
        fx.apply(&mut signal, SAMPLE_RATE, &mut frame_rng(values.seed, fnum as u64), fnum as u64);
    }
    signal
}

/// Decode a received signal with the 3D comb, given the two frames
/// before it as received, on the calling worker thread's decoder.
#[allow(clippy::too_many_arguments)]
fn decode_3d_tl(
    signal: &[f32],
    prev: &[f32],
    prev2: &[f32],
    out_w: usize,
    out_h: usize,
    values: &ParamValues,
    fx: &SignalEffects,
) -> Vec<u8> {
    let legacy = values.flag("legacy-burst-phase");
    TL_DECODER.with(|dec| {
        let mut dec = dec.borrow_mut();
        set_capture_peaking(&mut dec, fx);
        dec.decode_frame_3d(signal, prev, prev2, out_w, out_h, legacy, comb_luma_motion(fx)).to_vec()
    })
}

/// Whether the 3D comb also detects motion against the previous frame.  The
/// classic tape model keeps the original detector, whose output is locked.
fn comb_luma_motion(fx: &SignalEffects) -> bool {
    fx.tape_model != TapeModel::Classic
}

/// The capture chip's luma peaking, from the effects.
fn set_capture_peaking(dec: &mut Decoder, fx: &SignalEffects) {
    dec.peaking_gain = fx.capture_peaking.unwrap_or(0.0).max(0.0);
    dec.peaking_hz = fx.capture_peaking_mhz as f64 * 1e6;
}

/// Stop both ffmpeg processes after a cancel.
fn abort_children(reader: &mut std::process::Child, writer: &mut std::process::Child) {
    let _ = reader.kill();
    let _ = writer.kill();
    let _ = reader.wait();
    let _ = writer.wait();
}

fn roundtrip(
    input: &Path,
    output: &Path,
    opts: &VideoOptions,
    values: &ParamValues,
    progress: &dyn Progress,
    prefix: &str,
) -> Result<()> {
    let (out_w, out_h) = (opts.width as usize, opts.height as usize);
    let fx = resolve_effects(values);
    log_sampled(values, &fx, progress);
    let effects_active = fx.is_active();

    let (in_w, in_h, fps, fps_raw, total_frames) = ffprobe_video(input)?;
    progress.log(&format!(
        "Input: {} ({}x{} @ {:.3} fps, {} frames)",
        input.display(), in_w, in_h, fps, total_frames
    ));
    progress.log(&format!("Output: {} ({}x{} @ {:.3} fps)", output.display(), out_w, out_h, fps));

    let frame_bytes = in_w * in_h * 3;
    let mut reader = spawn_ffmpeg_reader(input, values, progress)?;
    let mut writer = spawn_ffmpeg_writer(output, out_w, out_h, &fps_raw, opts, false)?;
    let mut raw_copy = open_raw_copy(opts)?;
    let mut reader_buf = std::io::BufReader::new(reader.stdout.take().unwrap());
    let mut writer_buf = std::io::BufWriter::new(writer.stdin.take().unwrap());
    // Sequential stage: runs on the decoded frames in order.
    let mut dnr = fx.deck_dnr.filter(|s| *s > 0.0).map(TemporalDnr::new);

    let batch_size = init_thread_pool(opts.threads);
    progress.log(&format!("  Batch size: {} (parallel frames)", batch_size));

    let mut frame_num = 0u32;
    let total_start = Instant::now();
    progress.start(total_frames, prefix);

    let mut frame_pool: Vec<Vec<u8>> = (0..batch_size).map(|_| vec![0u8; frame_bytes]).collect();
    // Last two source frames of the previous batch (for the 3D comb).
    let mut last_frames = [vec![0u8; frame_bytes], vec![0u8; frame_bytes]];
    let frame_3d = comb_mode(values, opts.comb_1h) == Comb::Frame3d && values.choice("video-out") != "svideo";
    // Received signals (frame number, signal) of the last two frames of
    // the previous batch, for the 3D comb.
    let mut last_signals: Vec<(u32, Vec<f32>)> = Vec::new();

    loop {
        if progress.is_cancelled() {
            drop(writer_buf);
            abort_children(&mut reader, &mut writer);
            progress.finish();
            bail!(CANCELLED);
        }
        let read_scope = profile::scope("read input (ffmpeg)");
        let mut batch_count = 0usize;
        for buf in frame_pool.iter_mut() {
            match reader_buf.read_exact(buf) {
                Ok(()) => batch_count += 1,
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => {
                    progress.log(&format!("Read error: {}", e));
                    break;
                }
            }
        }
        drop(read_scope);
        if batch_count == 0 {
            break;
        }

        let full_frame = |i: usize| -> Vec<u8> {
            let fnum = frame_num + i as u32;
            // Source frame `k` frames back, from this batch or the last
            // two frames of the previous one (the first frames of the
            // video stand in for missing ones).
            let back = |k: usize| -> &[u8] {
                if i >= k {
                    &frame_pool[i - k]
                } else if frame_num as usize >= k - i {
                    &last_frames[k - i - 1]
                } else {
                    &frame_pool[0]
                }
            };
            let prev: PrevFrame = Some([(back(1), None), (back(2), None)]);
            process_frame_tl(
                &frame_pool[i], None, prev, in_w, in_h, out_w, out_h, fnum,
                opts.comb_1h, values, &fx, effects_active,
            )
        };
        let results: Vec<Vec<u8>> = if frame_3d {
            // From frame 1 on, the 3D comb's previous frames are the frames
            // before as received, so each frame's signal is made once and
            // shared.  This keeps the output independent of the batch size.
            let signals: Vec<Vec<f32>> = (0..batch_count)
                .into_par_iter()
                .map(|i| received_signal_tl(&frame_pool[i], in_w, in_h, frame_num + i as u32, values, &fx, effects_active))
                .collect();
            let signal = |n: u32| -> &[f32] {
                match n.checked_sub(frame_num) {
                    Some(k) => &signals[k as usize],
                    None => &last_signals.iter().find(|(m, _)| *m == n).expect("signal of the previous batch").1,
                }
            };
            let results = (0..batch_count)
                .into_par_iter()
                .map(|i| {
                    let fnum = frame_num + i as u32;
                    if fnum >= 1 {
                        // Frame 1 has only frame 0 before it, which stands in
                        // for the frame two back as well.
                        let prev2 = signal(fnum.saturating_sub(2));
                        decode_3d_tl(signal(fnum), signal(fnum - 1), prev2, out_w, out_h, values, &fx)
                    } else {
                        full_frame(i)
                    }
                })
                .collect();
            let end = frame_num + batch_count as u32;
            last_signals = (end.saturating_sub(2)..end).map(|n| (n, signal(n).to_vec())).collect();
            results
        } else {
            (0..batch_count).into_par_iter().map(full_frame).collect()
        };
        // last_frames[0] = most recent, [1] = the one before.
        if batch_count >= 2 {
            last_frames[1].copy_from_slice(&frame_pool[batch_count - 2]);
        } else {
            let (a, b) = last_frames.split_at_mut(1);
            b[0].copy_from_slice(&a[0]);
        }
        last_frames[0].copy_from_slice(&frame_pool[batch_count - 1]);

        let mut results = results;
        if let Some(dnr) = dnr.as_mut() {
            let _p = profile::scope("deck dnr");
            for r in results.iter_mut() {
                dnr.process(r, out_w, out_h);
            }
        }
        let write_scope = profile::scope("write output (ffmpeg)");
        for result in &results {
            writer_buf.write_all(result).context("write failed")?;
            if let Some(raw) = raw_copy.as_mut() {
                raw.write_all(result)?;
            }
        }
        drop(write_scope);

        frame_num += batch_count as u32;
        let elapsed = total_start.elapsed().as_secs_f64();
        progress.advance(batch_count as u64, &format!("{:.1} fps", frame_num as f64 / elapsed));
    }

    progress.finish();
    drop(writer_buf);
    let _ = reader.wait();
    let _ = writer.wait();

    let total_elapsed = total_start.elapsed().as_secs_f64();
    progress.log(&format!(
        "Done: {} frames in {:.1}s ({:.1} fps)",
        frame_num, total_elapsed, frame_num as f64 / total_elapsed,
    ));
    log_profile(frame_num as u64, progress);
    mux_audio(input, output);
    Ok(())
}

fn roundtrip_telecine(
    input: &Path,
    output: &Path,
    opts: &VideoOptions,
    values: &ParamValues,
    progress: &dyn Progress,
    prefix: &str,
) -> Result<()> {
    let (out_w, out_h) = (opts.width as usize, opts.height as usize);
    let fx = resolve_effects(values);
    log_sampled(values, &fx, progress);
    let effects_active = fx.is_active();

    let (in_w, in_h, _input_fps, _fps_raw, total_frames) = ffprobe_video(input)?;
    progress.log(&format!("Input: {} ({}x{}, {} frames)", input.display(), in_w, in_h, total_frames));
    progress.log(&format!("Output: {} ({}x{} @ 29.97 fps interlaced TFF)", output.display(), out_w, out_h));

    let frame_bytes = in_w * in_h * 3;
    let mut reader = spawn_ffmpeg_reader(input, values, progress)?;
    let mut writer = spawn_ffmpeg_writer(output, out_w, out_h, "30000/1001", opts, true)?;
    let mut raw_copy = open_raw_copy(opts)?;
    let mut reader_buf = std::io::BufReader::new(reader.stdout.take().unwrap());
    let mut writer_buf = std::io::BufWriter::new(writer.stdin.take().unwrap());
    let mut dnr = fx.deck_dnr.filter(|s| *s > 0.0).map(TemporalDnr::new);

    let num_cpus = init_thread_pool(opts.threads);
    progress.log(&format!("  Workers: {}", num_cpus));

    let mut ntsc_num = 0u32;
    // Sources (field 1, field 2) of the last two NTSC frames of the
    // previous batch, for the 3D comb.
    let mut last_srcs: Vec<(Vec<u8>, Option<Vec<u8>>)> = Vec::new();
    let mut film_idx = 0u32;
    let total_start = Instant::now();
    progress.start(total_frames, prefix);
    let groups_per_batch = num_cpus.max(1);

    loop {
        if progress.is_cancelled() {
            drop(writer_buf);
            abort_children(&mut reader, &mut writer);
            progress.finish();
            bail!(CANCELLED);
        }
        // Read groups_per_batch * 4 film frames
        let read_scope = profile::scope("read input (ffmpeg)");
        let mut film_buf: Vec<Vec<u8>> = Vec::with_capacity(groups_per_batch * 4);
        for _ in 0..groups_per_batch * 4 {
            let mut buf = vec![0u8; frame_bytes];
            match reader_buf.read_exact(&mut buf) {
                Ok(()) => film_buf.push(buf),
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => {
                    progress.log(&format!("Read error: {}", e));
                    break;
                }
            }
        }
        drop(read_scope);
        if film_buf.is_empty() {
            break;
        }

        // Expand film frames into NTSC jobs: (field 1 source, field 2 source,
        // NTSC frame number), indices into film_buf.
        let mut jobs: Vec<(usize, usize, u32)> = Vec::new();
        let mut fi = 0;
        while fi + 3 < film_buf.len() {
            // 3:2 pulldown: A,B,C,D -> (A,A), (B,B), (B,C), (C,D), (D,D)
            let base_ntsc = ntsc_num + jobs.len() as u32;
            jobs.push((fi, fi, base_ntsc));
            jobs.push((fi + 1, fi + 1, base_ntsc + 1));
            jobs.push((fi + 1, fi + 2, base_ntsc + 2));
            jobs.push((fi + 2, fi + 3, base_ntsc + 3));
            jobs.push((fi + 3, fi + 3, base_ntsc + 4));
            fi += 4;
        }
        // Remaining < 4 frames as progressive
        while fi < film_buf.len() {
            let base_ntsc = ntsc_num + jobs.len() as u32;
            jobs.push((fi, fi, base_ntsc));
            fi += 1;
        }

        let results: Vec<Vec<u8>> = jobs
            .par_iter()
            .enumerate()
            .map(|(j, &(f1_idx, f2_idx, fnum))| {
                let f2 = (f1_idx != f2_idx).then(|| film_buf[f2_idx].as_slice());
                // The two previous NTSC frames, from this batch or the end
                // of the previous one (at the start of the video the frame
                // itself stands in).
                let src = |(a, b): (usize, usize)| (film_buf[a].as_slice(), (a != b).then(|| film_buf[b].as_slice()));
                let job = |k: usize| -> (&[u8], Option<&[u8]>) {
                    if j >= k {
                        src((jobs[j - k].0, jobs[j - k].1))
                    } else if last_srcs.len() >= k - j {
                        let (a, b) = &last_srcs[last_srcs.len() - (k - j)];
                        (a.as_slice(), b.as_deref())
                    } else {
                        src((f1_idx, f2_idx))
                    }
                };
                let prev: PrevFrame = Some([job(1), job(2)]);
                process_frame_tl(
                    &film_buf[f1_idx], f2, prev, in_w, in_h, out_w, out_h, fnum,
                    opts.comb_1h, values, &fx, effects_active,
                )
            })
            .collect();

        let mut results = results;
        if let Some(dnr) = dnr.as_mut() {
            let _p = profile::scope("deck dnr");
            for r in results.iter_mut() {
                dnr.process(r, out_w, out_h);
            }
        }
        let write_scope = profile::scope("write output (ffmpeg)");
        for result in &results {
            writer_buf.write_all(result).context("write failed")?;
            if let Some(raw) = raw_copy.as_mut() {
                raw.write_all(result)?;
            }
        }
        drop(write_scope);

        // Sources of the last two NTSC frames, oldest first.
        let tail: Vec<(Vec<u8>, Option<Vec<u8>>)> = jobs[jobs.len().saturating_sub(2)..]
            .iter()
            .map(|&(a, b, _)| (film_buf[a].clone(), (a != b).then(|| film_buf[b].clone())))
            .collect();
        last_srcs.extend(tail);
        let excess = last_srcs.len().saturating_sub(2);
        last_srcs.drain(..excess);

        film_idx += film_buf.len() as u32;
        ntsc_num += results.len() as u32;
        let elapsed = total_start.elapsed().as_secs_f64();
        progress.advance(film_buf.len() as u64, &format!("{:.1} film fps", film_idx as f64 / elapsed));
    }

    progress.finish();
    drop(writer_buf);
    let _ = reader.wait();
    let _ = writer.wait();

    let total_elapsed = total_start.elapsed().as_secs_f64();
    progress.log(&format!(
        "Done: {} film frames -> {} NTSC frames in {:.1}s",
        film_idx, ntsc_num, total_elapsed
    ));
    log_profile(ntsc_num as u64, progress);
    mux_audio(input, output);
    Ok(())
}

// ────────────────────────────────────────────────────────────────────────────
// DVD path
// ────────────────────────────────────────────────────────────────────────────

/// Resolve the effects and, when `--format dvd`, the DVD settings.  The
/// DVD ranges are sampled after the analog effects, so analog-only runs
/// keep their exact random stream.
pub fn resolve_all(values: &ParamValues) -> (SignalEffects, Option<DvdConfig>) {
    let mut rng = values.file_rng();
    let fx = values.to_signal_effects(&mut rng);
    let dvd = (values.choice("format") == "dvd").then(|| DvdConfig::from_values(values, &mut rng));
    (fx, dvd)
}

/// Display aspect ratio of a video (width/height × sample aspect).
pub fn probe_dar(path: &Path) -> Option<f64> {
    let out = tool_command("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,sample_aspect_ratio", "-of", "csv=p=0"])
        .arg(path)
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let parts: Vec<&str> = text.trim().split(',').collect();
    let w: f64 = parts.first()?.parse().ok()?;
    let h: f64 = parts.get(1)?.parse().ok()?;
    let sar = parts
        .get(2)
        .and_then(|s| s.split_once(':'))
        .and_then(|(a, b)| Some((a.parse::<f64>().ok()?, b.parse::<f64>().ok()?)))
        .filter(|(a, b)| *a > 0.0 && *b > 0.0)
        .map(|(a, b)| a / b)
        .unwrap_or(1.0);
    Some(w / h * sar)
}

/// The player's output for one decoded picture: the frame itself
/// (digital) or its S-Video / composite output captured through the
/// NTSC decoder, with any explicitly set analog effects.  `f2` gives a
/// second picture for the other field (3:2 pulldown on analog outputs).
#[allow(clippy::too_many_arguments)]
fn player_output(
    rgb: &[u8],
    f2: Option<&[u8]>,
    cfg: &DvdConfig,
    values: &ParamValues,
    fx: &SignalEffects,
    comb_1h: bool,
    fnum: u32,
) -> (Vec<u8>, usize, usize) {
    if cfg.output == DvdOutput::Digital {
        let _p = profile::scope("dvd digital output");
        return dvd::digital_output(rgb, cfg);
    }
    let analog_scope = profile::scope("dvd analog source");
    let (src, sw, sh) = dvd::analog_source(rgb, cfg);
    let src2 = f2.map(|f| dvd::analog_source(f, cfg).0);
    drop(analog_scope);
    let mut vals = values.clone();
    vals.choices.insert("video-out", if cfg.output == DvdOutput::Svideo { "svideo" } else { "composite" });
    let (ow, oh) = cfg.output_size();
    let decode_w = if cfg.square { ow } else { 704 };
    let out = TL_ENCODER.with(|enc| {
        TL_DECODER.with(|dec| {
            TL_SIGNAL_BUF.with(|sb| {
                TL_SIGNAL_BUF_C.with(|sc| {
                    analog_frame(
                        &mut enc.borrow_mut(), &mut dec.borrow_mut(), &mut sb.borrow_mut(), &mut sc.borrow_mut(),
                        &src, src2.as_deref(), Some([(&src, src2.as_deref()), (&src, src2.as_deref())]), sw, sh, decode_w, 480, fnum, comb_1h, &vals, fx, fx.is_active(),
                    )
                })
            })
        })
    });
    if cfg.square { (out, ow, oh) } else { (dvd::pad_to_720(&out, 480), 720, 480) }
}

/// Run a still picture through the DVD chain.  The picture is encoded as
/// a short sequence (one GOP plus a few pictures) so that it gets real I,
/// P and B pictures; the first picture of the requested type is returned.
/// For a still the P and B pictures differ from the I picture only by the
/// requantised residual.
#[allow(clippy::too_many_arguments)]
pub fn dvd_still(
    pixels: &[u8],
    w: usize,
    h: usize,
    src_dar: f64,
    cfg: &DvdConfig,
    values: &ParamValues,
    fx: &SignalEffects,
    comb_1h: bool,
    frame: u64,
) -> Result<(Vec<u8>, usize, usize)> {
    let mastered = dvd::to_420(&dvd::master(pixels, w, h, src_dar, cfg), cfg.interlaced());
    let mut enc = dvd::Encoder::start(cfg)?;
    for _ in 0..cfg.gop + 4 {
        enc.push(&mastered)?;
    }
    let encoded = enc.finish()?;
    let types = encoded.stream.display_types();
    let want = cfg.frame_type.to_ascii_uppercase();
    let idx = types.iter().position(|&t| t == want).unwrap_or(0);
    let pic = encoded.decoder()?.nth(idx).context("the MPEG-2 decoder returned too few frames")?;
    let rgb = dvd::yuv420_to_rgb(&pic.frame, pic.interlaced, cfg.cue);
    Ok(player_output(&rgb, None, cfg, values, fx, comb_1h, frame as u32))
}

/// Like `render_preview`, but for a video frame with the frames before it
/// (`history`, oldest first): when the deck noise reduction is on, the
/// earlier frames are rendered first so its recursive state builds up.
#[allow(clippy::too_many_arguments)]
pub fn render_preview_sequence(
    history: &[&[u8]],
    pixels: &[u8],
    w: usize,
    h: usize,
    src_dar: f64,
    out_w: usize,
    out_h: usize,
    comb_1h: bool,
    values: &ParamValues,
    frame: u64,
) -> Result<(Vec<u8>, usize, usize, Vec<String>)> {
    let (fx, dvd) = resolve_all(values);
    let Some(strength) = fx.deck_dnr.filter(|s| *s > 0.0 && dvd.is_none() && !history.is_empty()) else {
        return render_preview(pixels, w, h, src_dar, out_w, out_h, comb_1h, values, frame);
    };
    let mut dnr = TemporalDnr::new(strength);
    let n = history.len() as u64;
    let mut prev: &[u8] = history[0];
    let mut prev2: &[u8] = history[0];
    for (k, src) in history.iter().chain(std::iter::once(&pixels)).enumerate() {
        let fnum = (frame + k as u64).saturating_sub(n);
        let mut encoder = Encoder::new();
        let mut decoder = Decoder::new();
        let (mut b1, mut b2) = (Vec::new(), Vec::new());
        let mut rgb = analog_frame(
            &mut encoder, &mut decoder, &mut b1, &mut b2, src, None, Some([(prev, None), (prev2, None)]), w, h, out_w, out_h,
            fnum as u32, comb_1h, values, &fx, fx.is_active(),
        );
        dnr.process(&mut rgb, out_w, out_h);
        prev2 = prev;
        prev = src;
        if k as u64 == n {
            return Ok((rgb, out_w, out_h, values.sampled_summary(&fx)));
        }
    }
    unreachable!()
}

/// Render a preview or still: the analog path at `out_w`×`out_h`, or the
/// DVD path at its own output size.  Returns the RGB frame, its size and
/// the values the ranges resolved to.
#[allow(clippy::too_many_arguments)]
pub fn render_preview(
    pixels: &[u8],
    w: usize,
    h: usize,
    src_dar: f64,
    out_w: usize,
    out_h: usize,
    comb_1h: bool,
    values: &ParamValues,
    frame: u64,
) -> Result<(Vec<u8>, usize, usize, Vec<String>)> {
    let (fx, dvd) = resolve_all(values);
    let mut sampled = values.sampled_summary(&fx);
    match dvd {
        Some(cfg) => {
            sampled.extend(cfg.summary(values));
            let (rgb, ow, oh) = dvd_still(pixels, w, h, src_dar, &cfg, values, &fx, comb_1h, frame)?;
            Ok((rgb, ow, oh, sampled))
        }
        None => Ok((render_frame(pixels, w, h, out_w, out_h, comb_1h, values, &fx, frame), out_w, out_h, sampled)),
    }
}

fn log_dvd(cfg: &DvdConfig, progress: &dyn Progress) {
    progress.log(&format!(
        "  DVD: {}x{} {} {}, {:.2} Mbit/s avg, {:.2} peak, GOP {}{}{}, output {:?}{}",
        cfg.width,
        cfg.height,
        if cfg.widescreen { "16:9" } else { "4:3" },
        if cfg.film { "23.976p soft pulldown" } else if cfg.interlaced() { "29.97i TFF" } else { "29.97p" },
        cfg.bitrate,
        cfg.maxrate,
        cfg.gop,
        if cfg.two_pass { ", 2-pass" } else { "" },
        if cfg.low_rate_matrix { ", low-rate matrices" } else { "" },
        cfg.output,
        if cfg.square { " (square pixels)" } else { "" },
    ));
}

/// DVD video roundtrip: master and encode the whole clip, then decode it,
/// run the player output stage and write the result losslessly.
fn dvd_roundtrip(
    input: &Path,
    output: &Path,
    opts: &VideoOptions,
    values: &ParamValues,
    progress: &dyn Progress,
    prefix: &str,
) -> Result<()> {
    let (fx, cfg) = resolve_all(values);
    let cfg = cfg.expect("dvd format");
    log_sampled(values, &fx, progress);
    let summary = cfg.summary(values);
    if !summary.is_empty() {
        progress.log(&format!("  Randomized: {}", summary.join(", ")));
    }
    log_dvd(&cfg, progress);

    let (in_w, in_h, _fps, _fps_raw, total_frames) = ffprobe_video(input)?;
    let src_dar = probe_dar(input).unwrap_or(in_w as f64 / in_h as f64);
    let (rn, rd) = cfg.frame_rate();
    progress.log(&format!("Input: {} ({}x{}, display aspect {:.3}, {} frames)", input.display(), in_w, in_h, src_dar, total_frames));

    // ── 1. Master and encode ──
    let mut reader = spawn_ffmpeg_reader_with(input, values, progress, Some(&format!("fps={}/{}", rn, rd)))?;
    let mut reader_buf = std::io::BufReader::new(reader.stdout.take().unwrap());
    let frame_bytes = in_w * in_h * 3;
    let batch = init_thread_pool(opts.threads);
    let mut enc = dvd::Encoder::start(&cfg)?;
    let t0 = Instant::now();
    let mut done = 0u64;
    progress.start(total_frames, if prefix.is_empty() { "DVD encode" } else { prefix });
    loop {
        if progress.is_cancelled() {
            let _ = reader.kill();
            progress.finish();
            bail!(CANCELLED);
        }
        let read_scope = profile::scope("read input (ffmpeg)");
        let mut frames = Vec::with_capacity(batch);
        for _ in 0..batch {
            let mut buf = vec![0u8; frame_bytes];
            match reader_buf.read_exact(&mut buf) {
                Ok(()) => frames.push(buf),
                Err(_) => break,
            }
        }
        drop(read_scope);
        if frames.is_empty() {
            break;
        }
        let master_scope = profile::scope("dvd master");
        let yuv: Vec<dvd::Yuv420> = frames
            .par_iter()
            .map(|f| dvd::to_420(&dvd::master(f, in_w, in_h, src_dar, &cfg), cfg.interlaced()))
            .collect();
        drop(master_scope);
        let _p = profile::scope("dvd mpeg-2 encode (ffmpeg)");
        for y in &yuv {
            enc.push(y)?;
        }
        done += frames.len() as u64;
        progress.advance(frames.len() as u64, &format!("{:.1} fps", done as f64 / t0.elapsed().as_secs_f64()));
    }
    let _ = reader.wait();
    progress.finish();
    let encoded = {
        let _p = profile::scope("dvd mpeg-2 encode (ffmpeg)");
        enc.finish()?
    };
    let n_pics = encoded.stream.pictures.len();
    progress.log(&format!(
        "  Encoded {} pictures ({}), {:.2} Mbit/s mean",
        n_pics,
        encoded.stream.display_types().iter().take(18).collect::<String>(),
        encoded.es.len() as f64 * 8.0 / (n_pics as f64 * rd as f64 / rn as f64) / 1e6,
    ));
    if let Some(k) = &cfg.keep_mpeg2 {
        progress.log(&format!("  MPEG-2 stream kept: {}", k.display()));
    }

    // ── 2. Decode, player output, write ──
    let (ow, oh) = cfg.output_size();
    let analog = cfg.output != DvdOutput::Digital;
    // Film on an analog output: the player applies 3:2 pulldown and sends
    // 29.97 fps interlaced video.
    let pulldown = analog && cfg.film;
    let interlaced_out = cfg.interlaced() || analog;
    let fps_out = if cfg.film && !analog { "24000/1001" } else { "30000/1001" };
    let mut writer = spawn_ffmpeg_writer(output, ow, oh, fps_out, opts, interlaced_out)?;
    let mut writer_buf = std::io::BufWriter::new(writer.stdin.take().unwrap());
    let mut raw_copy = open_raw_copy(opts)?;
    let mut decoder = encoded.decoder()?;
    let mut dnr = if analog { fx.deck_dnr.filter(|s| *s > 0.0).map(TemporalDnr::new) } else { None };
    progress.start(n_pics, "Player output");
    let mut out_num = 0u32;
    loop {
        if progress.is_cancelled() {
            drop(writer_buf);
            let _ = writer.kill();
            progress.finish();
            bail!(CANCELLED);
        }
        // Batches are a multiple of 4 pictures so pulldown groups stay whole.
        let decode_scope = profile::scope("dvd mpeg-2 decode");
        let pics: Vec<dvd::DecodedFrame> = decoder.by_ref().take(batch.max(1) * 4).collect();
        drop(decode_scope);
        if pics.is_empty() {
            break;
        }
        let rgbs: Vec<Vec<u8>> = pics
            .par_iter()
            .map(|p| {
                let _p = profile::scope("dvd yuv to rgb");
                dvd::yuv420_to_rgb(&p.frame, p.interlaced, cfg.cue)
            })
            .collect();
        // (field 1 picture, field 2 picture) per output frame.
        let mut jobs: Vec<(usize, usize, u32)> = Vec::new();
        if pulldown {
            let mut i = 0;
            while i < rgbs.len() {
                let n = (rgbs.len() - i).min(4);
                if n == 4 {
                    for (a, b) in [(0, 0), (1, 1), (1, 2), (2, 3), (3, 3)] {
                        jobs.push((i + a, i + b, out_num + jobs.len() as u32));
                    }
                } else {
                    for k in 0..n {
                        jobs.push((i + k, i + k, out_num + jobs.len() as u32));
                    }
                }
                i += n;
            }
        } else {
            for k in 0..rgbs.len() {
                jobs.push((k, k, out_num + k as u32));
            }
        }
        let results: Vec<Vec<u8>> = jobs
            .par_iter()
            .map(|&(a, b, fnum)| {
                let f2 = (a != b).then(|| rgbs[b].as_slice());
                player_output(&rgbs[a], f2, &cfg, values, &fx, opts.comb_1h, fnum).0
            })
            .collect();
        let mut results = results;
        if let Some(dnr) = dnr.as_mut() {
            let _p = profile::scope("deck dnr");
            for r in results.iter_mut() {
                dnr.process(r, ow, oh);
            }
        }
        let write_scope = profile::scope("write output (ffmpeg)");
        for r in &results {
            writer_buf.write_all(r).context("write failed")?;
            if let Some(raw) = raw_copy.as_mut() {
                raw.write_all(r)?;
            }
        }
        drop(write_scope);
        out_num += results.len() as u32;
        progress.advance(pics.len() as u64, "");
    }
    progress.finish();
    drop(writer_buf);
    let _ = writer.wait();
    progress.log(&format!("Done: {} pictures -> {} frames in {:.1}s", n_pics, out_num, t0.elapsed().as_secs_f64()));
    log_profile(out_num as u64, progress);
    mux_audio(input, output);
    Ok(())
}

// ────────────────────────────────────────────────────────────────────────────
// ffmpeg helpers
// ────────────────────────────────────────────────────────────────────────────

/// How the source video is converted to RGB.
#[derive(Clone, Debug, PartialEq)]
pub struct InputColour {
    /// ffmpeg colour matrix name (`bt601`, `bt709`, `bt2020`), or `None`
    /// for sources that are already RGB or grey.
    pub matrix: Option<&'static str>,
    /// `tv` (limited) or `pc` (full).
    pub range: &'static str,
    /// Human-readable summary, e.g. "BT.709 limited (tagged)".
    pub description: String,
}

impl InputColour {
    /// The ffmpeg filter chain that converts to RGB24.
    pub fn filter(&self) -> String {
        match self.matrix {
            Some(m) => format!(
                "scale=in_color_matrix={}:in_range={}:out_range=pc:flags=bicubic+accurate_rnd+full_chroma_int,format=rgb24",
                m, self.range
            ),
            None => "format=rgb24".to_string(),
        }
    }
}

/// Probe a video's pixel format and colour tags and decide how to convert
/// it to RGB, honouring the `input-matrix` / `input-range` overrides.
///
/// Without tags, SD (< 720 lines) is taken as BT.601 and HD as BT.709;
/// the range defaults to limited except for full-range (`yuvj*`) formats.
pub fn input_colour(path: &Path, values: &ParamValues) -> InputColour {
    let mut info = std::collections::HashMap::new();
    if let Ok(out) = tool_command("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=pix_fmt,color_space,color_range,height", "-of", "default=nw=1"])
        .arg(path)
        .output()
    {
        for line in String::from_utf8_lossy(&out.stdout).lines() {
            if let Some((k, v)) = line.split_once('=') {
                info.insert(k.trim().to_string(), v.trim().to_string());
            }
        }
    }
    let get = |k: &str| info.get(k).map(String::as_str).unwrap_or("unknown");
    let pix_fmt = get("pix_fmt");
    let height: u32 = get("height").parse().unwrap_or(480);
    let is_rgb = ["rgb", "bgr", "gbr", "argb", "abgr", "gray", "pal8", "ya8"]
        .iter()
        .any(|p| pix_fmt.starts_with(p));

    let (matrix, m_note) = match values.choice("input-matrix") {
        "601" => (Some("bt601"), "BT.601 (forced)"),
        "709" => (Some("bt709"), "BT.709 (forced)"),
        _ if is_rgb => (None, "RGB source"),
        _ => match get("color_space") {
            "bt709" => (Some("bt709"), "BT.709 (tagged)"),
            "smpte170m" | "bt470bg" | "fcc" => (Some("bt601"), "BT.601 (tagged)"),
            "bt2020nc" | "bt2020c" => (Some("bt2020"), "BT.2020 (tagged)"),
            _ if height >= 720 => (Some("bt709"), "BT.709 (untagged HD)"),
            _ => (Some("bt601"), "BT.601 (untagged SD)"),
        },
    };
    let (range, r_note) = match values.choice("input-range") {
        "limited" => ("tv", "limited (forced)"),
        "full" => ("pc", "full (forced)"),
        _ => match get("color_range") {
            "tv" => ("tv", "limited (tagged)"),
            "pc" => ("pc", "full (tagged)"),
            _ if pix_fmt.starts_with("yuvj") => ("pc", "full (JPEG format)"),
            _ => ("tv", "limited (untagged)"),
        },
    };
    let description = if matrix.is_some() { format!("{}, {}", m_note, r_note) } else { m_note.to_string() };
    InputColour { matrix, range, description }
}

fn spawn_ffmpeg_reader(input: &Path, values: &ParamValues, progress: &dyn Progress) -> Result<std::process::Child> {
    spawn_ffmpeg_reader_with(input, values, progress, None)
}

/// The reader with an extra filter before the RGB conversion (e.g. a
/// frame-rate conversion).
/// Decoder options to put before `-i` for this input.
///
/// AV1 is the one codec where ffmpeg's default can fail on valid files:
/// builds without libdav1d fall back to libaom, which rejects streams that
/// signal a level it does not know.  For AV1 this prefers libdav1d, then
/// hardware decoding (bit-exact, as AV1 decoding is normative), then libaom.
pub fn input_decoder_args(path: &Path) -> Vec<String> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    static HAS_DAV1D: OnceLock<bool> = OnceLock::new();
    static CHOICE: OnceLock<Mutex<HashMap<PathBuf, Vec<String>>>> = OnceLock::new();

    let cache = CHOICE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(args) = cache.lock().unwrap().get(path) {
        return args.clone();
    }

    let codec = tool_command("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "csv=p=0"])
        .arg(path)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();

    let args: Vec<String> = if codec != "av1" {
        Vec::new()
    } else if *HAS_DAV1D.get_or_init(|| {
        tool_command("ffmpeg")
            .args(["-hide_banner", "-decoders"])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).contains("libdav1d"))
            .unwrap_or(false)
    }) {
        vec!["-c:v".into(), "libdav1d".into()]
    } else {
        let hw: Vec<String> = vec!["-hwaccel".into(), "auto".into(), "-c:v".into(), "av1".into()];
        let decodes = tool_command("ffmpeg")
            .args(["-v", "error"])
            .args(&hw)
            .arg("-i")
            .arg(path)
            .args(["-frames:v", "1", "-f", "null", "-"])
            .output()
            .map(|o| o.status.success() && o.stderr.is_empty())
            .unwrap_or(false);
        if decodes { hw } else { Vec::new() }
    };
    cache.lock().unwrap().insert(path.to_path_buf(), args.clone());
    args
}

fn spawn_ffmpeg_reader_with(
    input: &Path,
    values: &ParamValues,
    progress: &dyn Progress,
    pre_filter: Option<&str>,
) -> Result<std::process::Child> {
    let colour = input_colour(input, values);
    progress.log(&format!("  Input colour: {}", colour.description));
    let filter = match pre_filter {
        Some(f) => format!("{},{}", f, colour.filter()),
        None => colour.filter(),
    };
    tool_command("ffmpeg")
        .args(input_decoder_args(input))
        .arg("-i")
        .arg(input)
        .args(["-vf", &filter])
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-v", "error", "pipe:1"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("Failed to spawn ffmpeg reader. Is ffmpeg installed?")
}

fn open_raw_copy(opts: &VideoOptions) -> Result<Option<std::io::BufWriter<std::fs::File>>> {
    opts.raw_copy
        .as_ref()
        .map(|p| {
            std::fs::File::create(p)
                .map(std::io::BufWriter::new)
                .with_context(|| format!("cannot create {}", p.display()))
        })
        .transpose()
}

/// `true` if this ffmpeg build has the named encoder.
fn has_encoder(name: &str) -> bool {
    tool_command("ffmpeg")
        .args(["-hide_banner", "-h", &format!("encoder={}", name)])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).contains(&format!("Encoder {}", name)))
        .unwrap_or(false)
}

/// Codec arguments for the output file.
///
/// The default is lossless and RGB-exact, because 4:2:0 YUV at CRF 17
/// halves the chroma resolution and smooths away the noise and fine
/// artefacts this tool simulates: FFV1 (`bgr0`) for .mkv/.avi/.nut and
/// x264's RGB encoder at QP 0 for .mp4/.mov.  `--lossy` gives the old
/// x264 4:2:0 CRF output.
pub fn writer_codec_args(output: &Path, opts: &VideoOptions) -> Result<Vec<String>> {
    let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<String>>();
    if opts.lossy {
        let mut a = s(&["-c:v", "libx264", "-preset", &opts.preset, "-crf", &opts.crf.to_string(), "-pix_fmt", "yuv420p"]);
        a.extend(s(&["-colorspace", "smpte170m", "-color_primaries", "smpte170m", "-color_trc", "smpte170m", "-color_range", "tv"]));
        return Ok(a);
    }
    let ext = output.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
    let mut a = match ext.as_str() {
        "mkv" | "avi" | "nut" => s(&["-c:v", "ffv1", "-level", "3", "-g", "1", "-pix_fmt", "bgr0"]),
        "mp4" | "mov" => {
            if !has_encoder("libx264rgb") {
                bail!("lossless .{} output needs ffmpeg's libx264rgb encoder, which this ffmpeg lacks; \
                       use .mkv (FFV1) or --lossy", ext);
            }
            s(&["-c:v", "libx264rgb", "-preset", &opts.preset, "-qp", "0", "-pix_fmt", "rgb24"])
        }
        _ => bail!("lossless output needs .mkv, .avi or .nut (FFV1) or .mp4/.mov (x264 RGB); use --lossy for .{}", ext),
    };
    // Our RGB comes from an NTSC (SMPTE 170M) decode.
    a.extend(s(&["-colorspace", "rgb", "-color_primaries", "smpte170m", "-color_trc", "smpte170m", "-color_range", "pc"]));
    Ok(a)
}

fn spawn_ffmpeg_writer(
    output: &Path,
    out_w: usize,
    out_h: usize,
    fps: &str,
    opts: &VideoOptions,
    interlaced: bool,
) -> Result<std::process::Child> {
    let mut args: Vec<String> = vec![
        "-y".into(), "-f".into(), "rawvideo".into(), "-pix_fmt".into(), "rgb24".into(),
        "-s".into(), format!("{}x{}", out_w, out_h), "-r".into(), fps.into(),
        "-i".into(), "pipe:0".into(),
    ];
    if interlaced {
        args.extend(["-vf", "setfield=tff", "-flags", "+ilme+ildct", "-top", "1"].map(String::from));
    }
    args.extend(writer_codec_args(output, opts)?);
    args.extend(["-v".into(), "error".into()]);

    tool_command("ffmpeg")
        .args(&args)
        .arg(output)
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("Failed to spawn ffmpeg writer. Is ffmpeg installed?")
}

/// Probe a video: (width, height, fps, raw fps string, frame count or 0).
pub fn ffprobe_video(path: &Path) -> Result<(usize, usize, f64, String, usize)> {
    let output = tool_command("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=width,height,r_frame_rate,nb_frames", "-of", "csv=p=0"])
        .arg(path)
        .output()
        .context("Failed to run ffprobe. Is ffmpeg installed?")?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let parts: Vec<&str> = stdout.trim().split(',').collect();
    if parts.len() < 3 {
        bail!("Cannot probe video '{}'. ffprobe output: {}", path.display(), stdout);
    }
    let w: usize = parts[0].parse().unwrap_or(640);
    let h: usize = parts[1].parse().unwrap_or(480);
    let fps_raw = parts[2].to_string();
    let fps = if let Some((num, den)) = parts[2].split_once('/') {
        let n: f64 = num.parse().unwrap_or(30000.0);
        let d: f64 = den.parse().unwrap_or(1001.0);
        n / d
    } else {
        parts[2].parse().unwrap_or(29.97)
    };
    let total_frames: usize = parts.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    Ok((w, h, fps, fps_raw, total_frames))
}

/// Duration of a video in seconds, if ffprobe can tell.
pub fn video_duration(path: &Path) -> Option<f64> {
    let output = tool_command("ffprobe")
        .args(["-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0"])
        .arg(path)
        .output()
        .ok()?;
    String::from_utf8_lossy(&output.stdout).trim().parse().ok()
}

/// Grab `count` consecutive RGB frames ending at `time_s` (oldest first),
/// with the same colour conversion as the reader.  Fewer frames come back
/// near the start of the video.
pub fn grab_video_frames(
    path: &Path,
    time_s: f64,
    count: usize,
    values: &ParamValues,
) -> Result<Vec<Vec<u8>>> {
    let (w, h, fps, ..) = ffprobe_video(path)?;
    let start = (time_s - (count as f64 - 1.0) / fps.max(1.0)).max(0.0);
    let colour = input_colour(path, values);
    let output = tool_command("ffmpeg")
        .args(input_decoder_args(path))
        .args(["-v", "error", "-ss", &format!("{:.3}", start), "-i"])
        .arg(path)
        .args(["-vf", &colour.filter()])
        .args(["-frames:v", &count.to_string(), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
        .output()
        .context("Failed to run ffmpeg. Is ffmpeg installed?")?;
    let n = w * h * 3;
    let frames: Vec<Vec<u8>> = output.stdout.chunks_exact(n).map(|c| c.to_vec()).collect();
    if frames.is_empty() {
        bail!("ffmpeg returned no frames at {:.2} s: {}", time_s, ffmpeg_error_text(&output.stderr));
    }
    Ok(frames)
}

/// Grab one RGB frame at `time_s` seconds, converted with the same colour
/// handling as the video reader: (pixels, width, height).
pub fn grab_video_frame(path: &Path, time_s: f64, values: &ParamValues) -> Result<(Vec<u8>, usize, usize)> {
    let (w, h, ..) = ffprobe_video(path)?;
    let colour = input_colour(path, values);
    let output = tool_command("ffmpeg")
        .args(input_decoder_args(path))
        .args(["-v", "error", "-ss", &format!("{:.3}", time_s.max(0.0)), "-i"])
        .arg(path)
        .args(["-vf", &colour.filter()])
        .args(["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
        .output()
        .context("Failed to run ffmpeg. Is ffmpeg installed?")?;
    let need = w * h * 3;
    if output.stdout.len() < need {
        bail!("ffmpeg returned no frame at {:.2} s: {}", time_s, ffmpeg_error_text(&output.stderr));
    }
    Ok((output.stdout[..need].to_vec(), w, h))
}

/// The last few lines of ffmpeg's error output, for error messages.
fn ffmpeg_error_text(stderr: &[u8]) -> String {
    let text = String::from_utf8_lossy(stderr);
    let lines: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    if lines.is_empty() {
        return "no error output (decoded size may not match the probed size)".to_string();
    }
    lines[lines.len().saturating_sub(3)..].join(" | ")
}

fn mux_audio(source: &Path, video_path: &Path) {
    let probe = tool_command("ffprobe")
        .args(["-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type", "-of", "csv=p=0"])
        .arg(source)
        .output();
    let has_audio = match probe {
        Ok(out) => String::from_utf8_lossy(&out.stdout).contains("audio"),
        Err(_) => false,
    };
    if !has_audio {
        return;
    }

    let tmp = PathBuf::from(format!("{}.mux.mp4", video_path.display()));
    let result = tool_command("ffmpeg")
        .arg("-y")
        .arg("-i")
        .arg(video_path)
        .arg("-i")
        .arg(source)
        .args(["-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest"])
        .arg(&tmp)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    match result {
        Ok(status) if status.success() => {
            let _ = std::fs::rename(&tmp, video_path);
        }
        _ => {
            let _ = std::fs::remove_file(&tmp);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(rgb: [u8; 3]) -> Vec<u8> {
        (0..640 * 480).flat_map(|_| rgb).collect()
    }

    fn mean(img: &[u8], x0: usize, x1: usize) -> [f32; 3] {
        let mut m = [0.0f32; 3];
        let mut n = 0.0;
        for y in 100..380 {
            for x in x0..x1 {
                for c in 0..3 {
                    m[c] += img[(y * 640 + x) * 3 + c] as f32;
                }
                n += 1.0;
            }
        }
        m.map(|v| v / n)
    }

    #[test]
    fn svideo_reproduces_flat_colours() {
        let mut values = ParamValues::default();
        values.set_from_str("video-out", "svideo").unwrap();
        let fx = resolve_effects(&values);
        for rgb in [[200u8, 40, 40], [40, 160, 60], [50, 60, 200], [128, 128, 128]] {
            let out = render_frame(&flat(rgb), 640, 480, 640, 480, false, &values, &fx, 0);
            let m = mean(&out, 100, 540);
            for c in 0..3 {
                assert!((m[c] - rgb[c] as f32).abs() < 3.0, "{:?} -> {:?}", rgb, m);
            }
        }
    }

    #[test]
    fn svideo_has_no_cross_colour_on_fine_luma_detail() {
        // Vertical grey stripes two pixels wide: composite decoding turns
        // part of this luma detail into false colour, S-Video must not.
        let src: Vec<u8> = (0..640 * 480).flat_map(|i| if (i % 640) / 2 % 2 == 0 { [60u8; 3] } else { [190u8; 3] }).collect();
        let sat = |out: &[u8]| -> f32 {
            let mut s = 0.0;
            for p in out.chunks(3) {
                let (r, g, b) = (p[0] as f32, p[1] as f32, p[2] as f32);
                s += r.max(g).max(b) - r.min(g).min(b);
            }
            s / (out.len() / 3) as f32
        };
        let mut v = ParamValues::default();
        let fx = resolve_effects(&v);
        let composite = sat(&render_frame(&src, 640, 480, 640, 480, false, &v, &fx, 0));
        v.set_from_str("video-out", "svideo").unwrap();
        let svideo = sat(&render_frame(&src, 640, 480, 640, 480, false, &v, &fx, 0));
        assert!(svideo < 0.25 * composite, "svideo {} vs composite {}", svideo, composite);

        // The 3D comb separates a still picture cleanly too.
        let mut v = ParamValues::default();
        v.set_from_str("comb", "3d").unwrap();
        let comb3d = sat(&render_frame(&src, 640, 480, 640, 480, false, &v, &fx, 5));
        assert!(comb3d < 0.25 * composite, "3d comb {} vs 2-sample {}", comb3d, composite);
    }
}
