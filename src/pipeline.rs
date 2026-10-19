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
use crate::decoder::Decoder;
use crate::effects::SignalEffects;
use crate::encoder::Encoder;
use crate::params::{frame_rng, ParamValues};

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
    let legacy = values.flag("legacy-burst-phase");
    let mut encoder = Encoder::new();
    let signal = encoder.encode_frame(pixels, w, h, frame as u32);
    let mut decoder = Decoder::new();
    if fx.is_active() {
        let mut buf = signal.to_vec();
        let mut rng = frame_rng(values.seed, frame);
        fx.apply(&mut buf, SAMPLE_RATE, &mut rng, frame);
        decoder.decode_frame(&buf, out_w, out_h, comb_1h, legacy).to_vec()
    } else {
        decoder.decode_frame(signal, out_w, out_h, comb_1h, legacy).to_vec()
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

    let fx = resolve_effects(values);
    log_sampled(values, &fx, progress);
    if !prefix.is_empty() {
        progress.log(prefix);
    }
    progress.log(&format!("Input: {} ({}x{})", input.display(), w, h));

    let t0 = Instant::now();
    let rgb = render_frame(img_rgb.as_raw(), w, h, out_w, out_h, comb_1h, values, &fx, 0);
    progress.log(&format!("  Encode + effects + decode: {:.1} ms", t0.elapsed().as_secs_f64() * 1000.0));

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

    let fx = resolve_effects(values);
    log_sampled(values, &fx, progress);
    let rgb = render_frame(&bars_rgb, w, h, w, h, comb_1h, values, &fx, 0);
    save_rgb(output, w, h, rgb)?;
    progress.log(&format!("Output: {} ({}x{})", output.display(), w, h));
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
    if opts.telecine {
        roundtrip_telecine(input, output, opts, values, progress, prefix)
    } else {
        roundtrip(input, output, opts, values, progress, prefix)
    }
}

thread_local! {
    static TL_ENCODER: RefCell<Encoder> = RefCell::new(Encoder::new());
    static TL_DECODER: RefCell<Decoder> = RefCell::new(Decoder::new());
    static TL_SIGNAL_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Encode (one or two fields' sources), apply effects, decode — on the
/// calling worker thread's own encoder/decoder.
#[allow(clippy::too_many_arguments)]
fn process_frame_tl(
    f1: &[u8],
    f2: Option<&[u8]>,
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
    TL_ENCODER.with(|enc| {
        TL_DECODER.with(|dec| {
            let mut enc_ref = enc.borrow_mut();
            let signal = match f2 {
                None => enc_ref.encode_frame(f1, in_w, in_h, fnum),
                Some(f2) => enc_ref.encode_frame_interlaced(f1, in_w, in_h, f2, fnum),
            };
            if effects_active {
                TL_SIGNAL_BUF.with(|sb| {
                    let mut sb_ref = sb.borrow_mut();
                    sb_ref.resize(signal.len(), 0.0);
                    sb_ref.copy_from_slice(signal);
                    let mut rng = frame_rng(values.seed, fnum as u64);
                    fx.apply(&mut sb_ref, SAMPLE_RATE, &mut rng, fnum as u64);
                    let mut dec_ref = dec.borrow_mut();
                    dec_ref.decode_frame(&sb_ref, out_w, out_h, comb_1h, legacy).to_vec()
                })
            } else {
                let mut dec_ref = dec.borrow_mut();
                dec_ref.decode_frame(signal, out_w, out_h, comb_1h, legacy).to_vec()
            }
        })
    })
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

    let batch_size = init_thread_pool(opts.threads);
    progress.log(&format!("  Batch size: {} (parallel frames)", batch_size));

    let mut frame_num = 0u32;
    let total_start = Instant::now();
    progress.start(total_frames, prefix);

    let mut frame_pool: Vec<Vec<u8>> = (0..batch_size).map(|_| vec![0u8; frame_bytes]).collect();

    loop {
        if progress.is_cancelled() {
            drop(writer_buf);
            abort_children(&mut reader, &mut writer);
            progress.finish();
            bail!(CANCELLED);
        }
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
        if batch_count == 0 {
            break;
        }

        let results: Vec<Vec<u8>> = (0..batch_count)
            .into_par_iter()
            .map(|i| {
                let fnum = frame_num + i as u32;
                process_frame_tl(
                    &frame_pool[i], None, in_w, in_h, out_w, out_h, fnum,
                    opts.comb_1h, values, &fx, effects_active,
                )
            })
            .collect();

        for result in &results {
            writer_buf.write_all(result).context("write failed")?;
            if let Some(raw) = raw_copy.as_mut() {
                raw.write_all(result)?;
            }
        }

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

    let num_cpus = init_thread_pool(opts.threads);
    progress.log(&format!("  Workers: {}", num_cpus));

    let mut ntsc_num = 0u32;
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
            .map(|&(f1_idx, f2_idx, fnum)| {
                let f2 = (f1_idx != f2_idx).then(|| film_buf[f2_idx].as_slice());
                process_frame_tl(
                    &film_buf[f1_idx], f2, in_w, in_h, out_w, out_h, fnum,
                    opts.comb_1h, values, &fx, effects_active,
                )
            })
            .collect();

        for result in &results {
            writer_buf.write_all(result).context("write failed")?;
            if let Some(raw) = raw_copy.as_mut() {
                raw.write_all(result)?;
            }
        }

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
    let colour = input_colour(input, values);
    progress.log(&format!("  Input colour: {}", colour.description));
    tool_command("ffmpeg")
        .arg("-i")
        .arg(input)
        .args(["-vf", &colour.filter()])
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

/// Grab one RGB frame at `time_s` seconds, converted with the same colour
/// handling as the video reader: (pixels, width, height).
pub fn grab_video_frame(path: &Path, time_s: f64, values: &ParamValues) -> Result<(Vec<u8>, usize, usize)> {
    let (w, h, ..) = ffprobe_video(path)?;
    let colour = input_colour(path, values);
    let output = tool_command("ffmpeg")
        .args(["-v", "error", "-ss", &format!("{:.3}", time_s.max(0.0)), "-i"])
        .arg(path)
        .args(["-vf", &colour.filter()])
        .args(["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
        .output()
        .context("Failed to run ffmpeg. Is ffmpeg installed?")?;
    let need = w * h * 3;
    if output.stdout.len() < need {
        bail!("ffmpeg returned no frame at {:.2} s", time_s);
    }
    Ok((output.stdout[..need].to_vec(), w, h))
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
