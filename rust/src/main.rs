mod colorbars;
mod constants;
mod decoder;
mod effects;
mod encoder;
mod filters;

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;
use rayon::prelude::*;

use crate::constants::SAMPLE_RATE;
use crate::decoder::Decoder;
use crate::effects::{GhostConfig, SignalEffects};
use crate::encoder::Encoder;

const VIDEO_EXTENSIONS: &[&str] = &["mp4", "mkv", "mov", "avi", "wmv", "flv", "webm", "m4v", "ts", "mts", "m2ts", "mpg", "mpeg"];
const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp"];

// ────────────────────────────────────────────────────────────────────────────
// Range parameter support
// ────────────────────────────────────────────────────────────────────────────

/// A parameter value that can be either a single number (e.g. `0.05`) or a
/// `min..max` range (e.g. `0.02..0.10`).  When a range is given, a random
/// value is sampled uniformly for each file processed in a batch.
#[derive(Clone, Copy, Debug)]
struct RangeParam {
    min: f32,
    max: f32,
}

impl RangeParam {
    /// Sample a value — returns `min` if fixed, otherwise uniform in [min, max].
    fn sample<R: Rng>(self, rng: &mut R) -> f32 {
        if (self.max - self.min).abs() < f32::EPSILON {
            self.min
        } else {
            rng.random_range(self.min..=self.max)
        }
    }

    /// Sample from an `Option<RangeParam>`.
    fn sample_opt<R: Rng>(opt: Option<RangeParam>, rng: &mut R) -> Option<f32> {
        opt.map(|r| r.sample(rng))
    }

    /// `true` if this is a genuine range (min ≠ max).
    fn is_range(self) -> bool {
        (self.max - self.min).abs() > f32::EPSILON
    }
}

impl std::str::FromStr for RangeParam {
    type Err = Box<dyn std::error::Error + Send + Sync + 'static>;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        if let Some((a, b)) = s.split_once("..") {
            let min: f32 = a.parse()?;
            let max: f32 = b.parse()?;
            if max < min {
                return Err(format!("range max ({}) must be >= min ({})", max, min).into());
            }
            Ok(RangeParam { min, max })
        } else {
            let val: f32 = s.parse()?;
            Ok(RangeParam { min: val, max: val })
        }
    }
}

impl std::fmt::Display for RangeParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_range() {
            write!(f, "{}..{}", self.min, self.max)
        } else {
            write!(f, "{}", self.min)
        }
    }
}

/// List files in a directory filtered by extension (case-insensitive), sorted by name.
fn iter_files(dir: &Path, extensions: &[&str]) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|ft| ft.is_file()).unwrap_or(false))
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| extensions.iter().any(|&ex| ex.eq_ignore_ascii_case(ext)))
                .unwrap_or(false)
        })
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

#[derive(Parser)]
#[command(name = "ntsc-composite-simulator")]
#[command(about = "NTSC Composite Video Simulator (Rust)")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// Shared signal degradation effect arguments.
///
/// All numeric parameters accept either a single value (e.g. `0.05`) or a
/// `min..max` range (e.g. `0.02..0.10`).  When a range is given, a random
/// value is sampled uniformly for each file processed in a batch.
#[derive(clap::Args, Clone)]
struct EffectsArgs {
    /// Gaussian noise amplitude (e.g. 0.05, or 0.02..0.2 for random range)
    #[arg(long)]
    noise: Option<RangeParam>,
    /// Ghost amplitude 0-1 (e.g. 0.3, or 0.1..0.5 for random range)
    #[arg(long)]
    ghost: Option<RangeParam>,
    /// Ghost delay in microseconds (e.g. 2.0, or 1.0..5.0)
    #[arg(long, default_value = "2.0")]
    ghost_delay: RangeParam,
    /// Ghost phase shift in degrees (e.g. 0, or 0..180)
    #[arg(long, default_value = "0.0")]
    ghost_phase: RangeParam,
    /// Ghost HF rolloff cutoff in MHz (e.g. 3.0, or 2.0..4.0)
    #[arg(long)]
    ghost_rolloff_mhz: Option<RangeParam>,
    /// Enable dynamic (time-varying) ghost amplitude
    #[arg(long, default_value = "false")]
    ghost_dynamic: bool,
    /// Dynamic ghost modulation rate in Hz (e.g. 0.5, or 0.2..1.0)
    #[arg(long, default_value = "0.5")]
    ghost_dynamic_rate: RangeParam,
    /// Additional ghosts as "amp,delay_us,phase_deg" triples separated by
    /// semicolons.  Each value may be a range (min..max).
    /// Example: --ghost-multi "0.1..0.3,2.0..5.0,0..180;0.08,7.5,0"
    #[arg(long)]
    ghost_multi: Option<String>,
    /// Signal attenuation strength 0-1 (e.g. 0.3, or 0.1..0.5)
    #[arg(long)]
    attenuation: Option<RangeParam>,
    /// Horizontal jitter std dev in subcarrier cycles (e.g. 0.5, or 0.1..1.0)
    #[arg(long)]
    jitter: Option<RangeParam>,
    // ── VHS tape-path effects ──
    /// VHS luma bandwidth in MHz (SP ≈ 3.0, EP/SLP ≈ 1.6)
    #[arg(long)]
    vhs_luma_bw: Option<RangeParam>,
    /// Color-under chroma bandwidth in kHz (typical 300–500)
    #[arg(long)]
    color_under_bw: Option<RangeParam>,
    /// Average tape dropouts per frame (e.g. 2–20)
    #[arg(long)]
    tape_dropout_rate: Option<RangeParam>,
    /// Average dropout length in microseconds
    #[arg(long, default_value = "15.0")]
    tape_dropout_len: RangeParam,
    /// Edge peaking / ringing gain (e.g. 0.5–3.0)
    #[arg(long)]
    edge_ringing: Option<RangeParam>,
    /// Luminance-dependent noise amplitude (e.g. 0.02–0.10)
    #[arg(long)]
    luma_noise: Option<RangeParam>,
    /// Head-wear symmetric smear strength (0.3–1.0).  Worn-head box blur
    /// that varies in bands across the frame.
    #[arg(long)]
    head_smear: Option<RangeParam>,
    /// Tape trailing / causal rightward smear strength (0.3–1.0).  The
    /// classic worn-tape look where edges and noise leave a rightward
    /// “comet tail”.
    #[arg(long)]
    tape_trail: Option<RangeParam>,
}

impl EffectsArgs {
    fn to_signal_effects<R: Rng>(&self, rng: &mut R) -> SignalEffects {
        let mut ghosts: Vec<GhostConfig> = Vec::new();

        // Primary ghost from --ghost
        if let Some(amp_range) = self.ghost {
            ghosts.push(GhostConfig {
                amplitude: amp_range.sample(rng),
                delay_us: self.ghost_delay.sample(rng),
                phase_shift: self.ghost_phase.sample(rng).to_radians(),
                rolloff_hz: RangeParam::sample_opt(self.ghost_rolloff_mhz, rng)
                    .map(|m| m as f64 * 1e6),
                dynamic: self.ghost_dynamic,
                dynamic_rate: self.ghost_dynamic_rate.sample(rng),
            });
        }

        // Extra ghosts from --ghost-multi (each value supports min..max ranges)
        if let Some(ref multi) = self.ghost_multi {
            for entry in multi.split(';') {
                let parts: Vec<&str> = entry.trim().split(',').collect();
                if parts.len() >= 2 {
                    let amp_r: RangeParam = parts[0].trim().parse()
                        .unwrap_or(RangeParam { min: 0.0, max: 0.0 });
                    let delay_r: RangeParam = parts[1].trim().parse()
                        .unwrap_or(RangeParam { min: 2.0, max: 2.0 });
                    let phase_r: RangeParam = if parts.len() >= 3 {
                        parts[2].trim().parse()
                            .unwrap_or(RangeParam { min: 0.0, max: 0.0 })
                    } else {
                        RangeParam { min: 0.0, max: 0.0 }
                    };
                    ghosts.push(GhostConfig {
                        amplitude: amp_r.sample(rng),
                        delay_us: delay_r.sample(rng),
                        phase_shift: phase_r.sample(rng).to_radians(),
                        rolloff_hz: RangeParam::sample_opt(self.ghost_rolloff_mhz, rng)
                            .map(|m| m as f64 * 1e6),
                        dynamic: self.ghost_dynamic,
                        dynamic_rate: self.ghost_dynamic_rate.sample(rng),
                    });
                }
            }
        }

        SignalEffects {
            noise: RangeParam::sample_opt(self.noise, rng),
            ghosts,
            attenuation: RangeParam::sample_opt(self.attenuation, rng),
            jitter: RangeParam::sample_opt(self.jitter, rng),
            vhs_luma_bw: RangeParam::sample_opt(self.vhs_luma_bw, rng).map(|m| m * 1e6),
            color_under_bw: RangeParam::sample_opt(self.color_under_bw, rng).map(|k| k * 1e3),
            tape_dropout_rate: RangeParam::sample_opt(self.tape_dropout_rate, rng),
            tape_dropout_len: self.tape_dropout_len.sample(rng),
            edge_ringing: RangeParam::sample_opt(self.edge_ringing, rng),
            luma_noise: RangeParam::sample_opt(self.luma_noise, rng),
            head_switching_smear: RangeParam::sample_opt(self.head_smear, rng),
            tape_trail_smear: RangeParam::sample_opt(self.tape_trail, rng),
        }
    }

    fn has_any_range(&self) -> bool {
        let check = |o: Option<RangeParam>| o.map_or(false, |r| r.is_range());
        check(self.noise)
            || check(self.ghost)
            || self.ghost_delay.is_range()
            || self.ghost_phase.is_range()
            || check(self.ghost_rolloff_mhz)
            || self.ghost_dynamic_rate.is_range()
            || check(self.attenuation)
            || check(self.jitter)
            || check(self.vhs_luma_bw)
            || check(self.color_under_bw)
            || check(self.tape_dropout_rate)
            || self.tape_dropout_len.is_range()
            || check(self.edge_ringing)
            || check(self.luma_noise)
            || check(self.head_smear)
            || check(self.tape_trail)
    }

    fn log_sampled(&self, fx: &SignalEffects) {
        if !self.has_any_range() { return; }
        let mut parts: Vec<String> = Vec::new();
        macro_rules! check_opt {
            ($label:expr, $arg:expr, $val:expr) => {
                if $arg.map_or(false, |r| r.is_range()) {
                    if let Some(v) = $val {
                        parts.push(format!("{}={:.4}", $label, v));
                    }
                }
            };
        }
        macro_rules! check_req {
            ($label:expr, $arg:expr, $val:expr) => {
                if $arg.is_range() {
                    parts.push(format!("{}={:.4}", $label, $val));
                }
            };
        }
        check_opt!("noise", self.noise, fx.noise);
        if self.ghost.map_or(false, |r| r.is_range()) {
            if let Some(g) = fx.ghosts.first() {
                parts.push(format!("ghost={:.4}", g.amplitude));
            }
        }
        if self.ghost_delay.is_range() {
            if let Some(g) = fx.ghosts.first() {
                parts.push(format!("ghost_delay={:.2}", g.delay_us));
            }
        }
        if self.ghost_phase.is_range() {
            if let Some(g) = fx.ghosts.first() {
                parts.push(format!("ghost_phase={:.1}", g.phase_shift.to_degrees()));
            }
        }
        check_opt!("atten", self.attenuation, fx.attenuation);
        check_opt!("jitter", self.jitter, fx.jitter);
        check_opt!("vhs_bw_mhz", self.vhs_luma_bw, fx.vhs_luma_bw.map(|v| v / 1e6));
        check_opt!("color_under_khz", self.color_under_bw, fx.color_under_bw.map(|v| v / 1e3));
        check_opt!("dropout_rate", self.tape_dropout_rate, fx.tape_dropout_rate);
        check_req!("dropout_len", self.tape_dropout_len, fx.tape_dropout_len);
        check_opt!("ringing", self.edge_ringing, fx.edge_ringing);
        check_opt!("luma_noise", self.luma_noise, fx.luma_noise);
        check_opt!("head_smear", self.head_smear, fx.head_switching_smear);
        check_opt!("tape_trail", self.tape_trail, fx.tape_trail_smear);
        if !parts.is_empty() {
            eprintln!("  Randomized: {}", parts.join(", "));
        }
    }
}

#[derive(Subcommand)]
enum Commands {
    /// Roundtrip a single image through the NTSC composite pipeline
    Image {
        /// Input image file or directory of images
        input: String,
        /// Output image file
        #[arg(short, long, default_value = "output.png")]
        output: String,
        /// Output width (default: same as input)
        #[arg(long)]
        width: Option<u32>,
        /// Output height (default: same as input)
        #[arg(long)]
        height: Option<u32>,
        /// Use 1H line-delay comb filter
        #[arg(long)]
        comb_1h: bool,
        #[command(flatten)]
        effects: EffectsArgs,
    },
    /// Encode video to composite and decode back (roundtrip)
    Roundtrip {
        /// Input video file or directory of videos
        input: String,
        /// Output video file
        #[arg(short, long, default_value = "output.mp4")]
        output: String,
        /// Output width
        #[arg(long, default_value = "640")]
        width: u32,
        /// Output height
        #[arg(long, default_value = "480")]
        height: u32,
        /// Use 1H line-delay comb filter
        #[arg(long)]
        comb_1h: bool,
        /// x264 CRF quality (0=lossless, 51=worst)
        #[arg(long, default_value = "17")]
        crf: u32,
        /// x264 preset
        #[arg(long, default_value = "fast")]
        preset: String,
        /// Lossless output (FFV1 for .mkv, x264 QP 0 for .mp4)
        #[arg(long)]
        lossless: bool,
        /// Number of parallel worker threads (default: all logical cores)
        #[arg(long)]
        threads: Option<usize>,
        /// Enable 3:2 pulldown telecine (24fps film -> 29.97fps interlaced)
        #[arg(long)]
        telecine: bool,
        #[command(flatten)]
        effects: EffectsArgs,
    },
    /// Generate SMPTE color bars through the NTSC pipeline
    Colorbars {
        /// Output image file
        #[arg(short, long, default_value = "colorbars.png")]
        output: String,
        /// Also save the source color bar pattern (before NTSC processing)
        #[arg(long)]
        save_source: Option<String>,
        /// Image width
        #[arg(long, default_value = "640")]
        width: u32,
        /// Image height
        #[arg(long, default_value = "480")]
        height: u32,
        /// Use 1H line-delay comb filter
        #[arg(long)]
        comb_1h: bool,
        #[command(flatten)]
        effects: EffectsArgs,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Image {
            input,
            output,
            width,
            height,
            comb_1h,
            effects,
        } => {
            if Path::new(&input).is_dir() {
                let files = iter_files(Path::new(&input), IMAGE_EXTENSIONS);
                if files.is_empty() {
                    bail!("No image files found in '{}'", input);
                }
                let output_dir = Path::new(&output);
                std::fs::create_dir_all(output_dir)?;
                eprintln!("Batch image: {} file(s) -> '{}'", files.len(), output);
                for (i, path) in files.iter().enumerate() {
                    let out_path = batch_output_path(output_dir, path);
                    let prefix = format!("[{}/{}] {}", i + 1, files.len(),
                        path.file_name().unwrap().to_string_lossy());
                    cmd_image(
                        &path.to_string_lossy(), &out_path.to_string_lossy(),
                        width, height, comb_1h, &effects, &prefix,
                    )?;
                }
                Ok(())
            } else {
                cmd_image(&input, &output, width, height, comb_1h, &effects, "")
            }
        }
        Commands::Roundtrip {
            input,
            output,
            width,
            height,
            comb_1h,
            crf,
            preset,
            lossless,
            threads,
            telecine,
            effects,
        } => {
            if Path::new(&input).is_dir() {
                let files = iter_files(Path::new(&input), VIDEO_EXTENSIONS);
                if files.is_empty() {
                    bail!("No video files found in '{}'", input);
                }
                let output_dir = Path::new(&output);
                std::fs::create_dir_all(output_dir)?;
                eprintln!("Batch roundtrip: {} file(s) -> '{}'", files.len(), output);

                // Initialize thread pool once before batch loop
                let num_cpus = resolve_threads(threads);
                rayon::ThreadPoolBuilder::new()
                    .num_threads(num_cpus)
                    .build_global()
                    .ok();

                for (i, path) in files.iter().enumerate() {
                    let out_path = batch_output_path(output_dir, path);
                    let prefix = format!("[{}/{}] {}", i + 1, files.len(),
                        path.file_name().unwrap().to_string_lossy());
                    let inp = path.to_string_lossy();
                    let outp = out_path.to_string_lossy();
                    if telecine {
                        cmd_roundtrip_telecine(&inp, &outp, width, height, comb_1h, crf, &preset, lossless, threads, &effects, &prefix)?;
                    } else {
                        cmd_roundtrip(&inp, &outp, width, height, comb_1h, crf, &preset, lossless, threads, &effects, &prefix)?;
                    }
                }
                Ok(())
            } else {
                if telecine {
                    cmd_roundtrip_telecine(&input, &output, width, height, comb_1h, crf, &preset, lossless, threads, &effects, "")
                } else {
                    cmd_roundtrip(&input, &output, width, height, comb_1h, crf, &preset, lossless, threads, &effects, "")
                }
            }
        }
        Commands::Colorbars {
            output,
            save_source,
            width,
            height,
            comb_1h,
            effects,
        } => cmd_colorbars(&output, save_source.as_deref(), width, height, comb_1h, &effects),
    }
}

fn cmd_image(
    input: &str,
    output: &str,
    width: Option<u32>,
    height: Option<u32>,
    comb_1h: bool,
    effects_args: &EffectsArgs,
    prefix: &str,
) -> Result<()> {
    let img = image::open(input).with_context(|| format!("Cannot open image '{}'", input))?;

    let img_rgb = img.to_rgb8();
    let (w, h) = (img_rgb.width() as usize, img_rgb.height() as usize);
    let pixels = img_rgb.as_raw();

    let out_w = width.map(|v| v as usize).unwrap_or(w);
    let out_h = height.map(|v| v as usize).unwrap_or(h);

    let mut rng = rand::rng();
    let fx = effects_args.to_signal_effects(&mut rng);
    effects_args.log_sampled(&fx);

    if !prefix.is_empty() {
        eprintln!("{}", prefix);
    }
    eprintln!("Input: {} ({}x{})", input, w, h);
    eprintln!("Encoding to composite signal...");

    let mut encoder = Encoder::new();
    let t0 = Instant::now();
    let signal = encoder.encode_frame(pixels, w, h, 0);

    // Apply effects if any are active
    let signal_ref: &[f32] = if fx.is_active() {
        let mut buf = signal.to_vec();
        let mut rng = rand::rng();
        fx.apply(&mut buf, SAMPLE_RATE, &mut rng);
        // Leak into a temporary that lives long enough — use a boxed slice
        // We need owned data for decode; just decode from the vec.
        let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!("  Encode + effects: {:.1} ms", encode_ms);

        eprintln!("Decoding from composite signal...");
        let mut decoder = Decoder::new();
        let t1 = Instant::now();
        let result_rgb = decoder.decode_frame(&buf, out_w, out_h, comb_1h);
        let decode_ms = t1.elapsed().as_secs_f64() * 1000.0;
        eprintln!("  Decode: {:.1} ms", decode_ms);
        eprintln!(
            "  Total:  {:.1} ms ({:.1} fps)",
            encode_ms + decode_ms,
            1000.0 / (encode_ms + decode_ms)
        );

        let out_img =
            image::RgbImage::from_raw(out_w as u32, out_h as u32, result_rgb.to_vec())
                .context("Failed to create output image")?;
        out_img
            .save(output)
            .with_context(|| format!("Cannot save image '{}'", output))?;

        eprintln!("Output: {} ({}x{})", output, out_w, out_h);
        return Ok(());
    } else {
        signal
    };

    let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!("  Encode: {:.1} ms", encode_ms);

    eprintln!("Decoding from composite signal...");
    let mut decoder = Decoder::new();
    let t0 = Instant::now();
    let result_rgb = decoder.decode_frame(signal_ref, out_w, out_h, comb_1h);
    let decode_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!("  Decode: {:.1} ms", decode_ms);
    eprintln!(
        "  Total:  {:.1} ms ({:.1} fps)",
        encode_ms + decode_ms,
        1000.0 / (encode_ms + decode_ms)
    );

    let out_img =
        image::RgbImage::from_raw(out_w as u32, out_h as u32, result_rgb.to_vec())
            .context("Failed to create output image")?;
    out_img
        .save(output)
        .with_context(|| format!("Cannot save image '{}'", output))?;

    eprintln!("Output: {} ({}x{})", output, out_w, out_h);
    Ok(())
}

fn cmd_colorbars(
    output: &str,
    save_source: Option<&str>,
    width: u32,
    height: u32,
    comb_1h: bool,
    effects_args: &EffectsArgs,
) -> Result<()> {
    let w = width as usize;
    let h = height as usize;

    eprintln!("Generating SMPTE color bars ({}x{})...", w, h);
    let bars_rgb = colorbars::generate_colorbars(w, h);

    if let Some(src_path) = save_source {
        let src_img = image::RgbImage::from_raw(width, height, bars_rgb.clone())
            .context("Failed to create source image")?;
        src_img
            .save(src_path)
            .with_context(|| format!("Cannot save source image '{}'", src_path))?;
        eprintln!("Source pattern saved: {}", src_path);
    }

    let mut rng = rand::rng();
    let fx = effects_args.to_signal_effects(&mut rng);
    effects_args.log_sampled(&fx);

    let mut encoder = Encoder::new();
    let signal = encoder.encode_frame(&bars_rgb, w, h, 0);

    let decoded = if fx.is_active() {
        let mut buf = signal.to_vec();
        let mut rng = rand::rng();
        fx.apply(&mut buf, SAMPLE_RATE, &mut rng);
        let mut decoder = Decoder::new();
        decoder.decode_frame(&buf, w, h, comb_1h).to_vec()
    } else {
        let mut decoder = Decoder::new();
        decoder.decode_frame(signal, w, h, comb_1h).to_vec()
    };

    let out_img = image::RgbImage::from_raw(width, height, decoded)
        .context("Failed to create output image")?;
    out_img
        .save(output)
        .with_context(|| format!("Cannot save image '{}'", output))?;

    eprintln!("Output: {} ({}x{})", output, w, h);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_roundtrip(
    input: &str,
    output: &str,
    width: u32,
    height: u32,
    comb_1h: bool,
    crf: u32,
    preset: &str,
    lossless: bool,
    threads: Option<usize>,
    effects_args: &EffectsArgs,
    prefix: &str,
) -> Result<()> {
    let out_w = width as usize;
    let out_h = height as usize;
    let mut rng = rand::rng();
    let fx = effects_args.to_signal_effects(&mut rng);
    effects_args.log_sampled(&fx);
    let effects_active = fx.is_active();

    let (in_w, in_h, fps, fps_raw, total_frames) = ffprobe_video(input)?;
    eprintln!(
        "Input: {} ({}x{} @ {:.3} fps, {} frames)",
        input, in_w, in_h, fps, total_frames
    );
    eprintln!(
        "Output: {} ({}x{} @ {:.3} fps)",
        output, out_w, out_h, fps
    );

    let frame_bytes = in_w * in_h * 3;

    let mut reader = spawn_ffmpeg_reader(input)?;
    let mut writer = spawn_ffmpeg_writer(output, out_w, out_h, &fps_raw, preset, crf, false, lossless)?;

    let reader_stdout = reader.stdout.take().unwrap();
    let mut reader_buf = std::io::BufReader::new(reader_stdout);
    let writer_stdin = writer.stdin.take().unwrap();
    let mut writer_buf = std::io::BufWriter::new(writer_stdin);

    let num_cpus = resolve_threads(threads);
    let batch_size = num_cpus;
    eprintln!("  Batch size: {} (parallel frames)", batch_size);

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus)
        .build_global()
        .ok();

    let mut frame_num = 0u32;
    let total_start = Instant::now();

    let pb = make_progress_bar(total_frames, prefix);

    use std::cell::RefCell;
    thread_local! {
        static TL_ENCODER: RefCell<Encoder> = RefCell::new(Encoder::new());
        static TL_DECODER: RefCell<Decoder> = RefCell::new(Decoder::new());
        static TL_SIGNAL_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    }

    let mut frame_pool: Vec<Vec<u8>> = (0..batch_size)
        .map(|_| vec![0u8; frame_bytes])
        .collect();

    // Clone effects for move into closure
    let fx_noise = fx.noise;
    let fx_ghosts = fx.ghosts.clone();
    let fx_attenuation = fx.attenuation;
    let fx_jitter = fx.jitter;
    let fx_vhs_luma_bw = fx.vhs_luma_bw;
    let fx_color_under_bw = fx.color_under_bw;
    let fx_tape_dropout_rate = fx.tape_dropout_rate;
    let fx_tape_dropout_len = fx.tape_dropout_len;
    let fx_edge_ringing = fx.edge_ringing;
    let fx_luma_noise = fx.luma_noise;
    let fx_head_switching_smear = fx.head_switching_smear;
    let fx_tape_trail_smear = fx.tape_trail_smear;

    loop {
        let mut batch_count = 0usize;
        for buf in frame_pool.iter_mut() {
            match reader_buf.read_exact(buf) {
                Ok(()) => batch_count += 1,
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => {
                    eprintln!("Read error: {}", e);
                    break;
                }
            }
        }

        if batch_count == 0 {
            break;
        }

        let batch_items: Vec<(&[u8], u32)> = (0..batch_count)
            .map(|i| (frame_pool[i].as_slice(), frame_num + i as u32))
            .collect();

        let results: Vec<Vec<u8>> = batch_items
            .par_iter()
            .map(|(frame_buf, fnum)| {
                TL_ENCODER.with(|enc| {
                    TL_DECODER.with(|dec| {
                        let mut enc_ref = enc.borrow_mut();
                        let signal = enc_ref.encode_frame(frame_buf, in_w, in_h, *fnum);

                        if effects_active {
                            TL_SIGNAL_BUF.with(|sb| {
                                let mut sb_ref = sb.borrow_mut();
                                sb_ref.resize(signal.len(), 0.0);
                                sb_ref.copy_from_slice(signal);
                                let fx = SignalEffects {
                                    noise: fx_noise,
                                    ghosts: fx_ghosts.clone(),
                                    attenuation: fx_attenuation,
                                    jitter: fx_jitter,
                                    vhs_luma_bw: fx_vhs_luma_bw,
                                    color_under_bw: fx_color_under_bw,
                                    tape_dropout_rate: fx_tape_dropout_rate,
                                    tape_dropout_len: fx_tape_dropout_len,
                                    edge_ringing: fx_edge_ringing,
                                    luma_noise: fx_luma_noise,
                                    head_switching_smear: fx_head_switching_smear,
                                    tape_trail_smear: fx_tape_trail_smear,
                                };
                                let mut rng = rand::rng();
                                fx.apply(&mut sb_ref, SAMPLE_RATE, &mut rng);
                                let mut dec_ref = dec.borrow_mut();
                                dec_ref.decode_frame(&sb_ref, out_w, out_h, comb_1h).to_vec()
                            })
                        } else {
                            let mut dec_ref = dec.borrow_mut();
                            dec_ref.decode_frame(signal, out_w, out_h, comb_1h).to_vec()
                        }
                    })
                })
            })
            .collect();

        for result in &results {
            writer_buf.write_all(result).context("write failed")?;
        }

        frame_num += batch_count as u32;
        pb.inc(batch_count as u64);

        let elapsed = total_start.elapsed().as_secs_f64();
        let fps_actual = frame_num as f64 / elapsed;
        pb.set_message(format!("{:.1} fps", fps_actual));
    }

    pb.finish_and_clear();

    drop(writer_buf);
    let _ = reader.wait();
    let _ = writer.wait();

    let total_elapsed = total_start.elapsed().as_secs_f64();
    eprintln!(
        "Done: {} frames in {:.1}s ({:.1} fps)",
        frame_num,
        total_elapsed,
        frame_num as f64 / total_elapsed,
    );

    mux_audio(input, output);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_roundtrip_telecine(
    input: &str,
    output: &str,
    width: u32,
    height: u32,
    comb_1h: bool,
    crf: u32,
    preset: &str,
    lossless: bool,
    threads: Option<usize>,
    effects_args: &EffectsArgs,
    prefix: &str,
) -> Result<()> {
    let out_w = width as usize;
    let out_h = height as usize;
    let mut rng = rand::rng();
    let fx = effects_args.to_signal_effects(&mut rng);
    effects_args.log_sampled(&fx);
    let effects_active = fx.is_active();

    let (in_w, in_h, _input_fps, _fps_raw, total_frames) = ffprobe_video(input)?;
    eprintln!(
        "Input: {} ({}x{}, {} frames)",
        input, in_w, in_h, total_frames
    );
    eprintln!(
        "Output: {} ({}x{} @ 29.97 fps interlaced TFF)",
        output, out_w, out_h
    );

    let frame_bytes = in_w * in_h * 3;

    let mut reader = spawn_ffmpeg_reader(input)?;
    let mut writer = spawn_ffmpeg_writer(output, out_w, out_h, "30000/1001", preset, crf, true, lossless)?;

    let reader_stdout = reader.stdout.take().unwrap();
    let mut reader_buf = std::io::BufReader::new(reader_stdout);
    let writer_stdin = writer.stdin.take().unwrap();
    let mut writer_buf = std::io::BufWriter::new(writer_stdin);

    let num_cpus = resolve_threads(threads);
    eprintln!("  Workers: {}", num_cpus);

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus)
        .build_global()
        .ok();

    let mut ntsc_num = 0u32;
    let mut film_idx = 0u32;
    let total_start = Instant::now();

    let pb = make_progress_bar(total_frames, prefix);

    use std::cell::RefCell;
    thread_local! {
        static TL_ENCODER: RefCell<Encoder> = RefCell::new(Encoder::new());
        static TL_DECODER: RefCell<Decoder> = RefCell::new(Decoder::new());
        static TL_SIGNAL_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    }

    let groups_per_batch = num_cpus.max(1);

    // Clone effects for move into closure
    let fx_noise = fx.noise;
    let fx_ghosts = fx.ghosts.clone();
    let fx_attenuation = fx.attenuation;
    let fx_jitter = fx.jitter;
    let fx_vhs_luma_bw = fx.vhs_luma_bw;
    let fx_color_under_bw = fx.color_under_bw;
    let fx_tape_dropout_rate = fx.tape_dropout_rate;
    let fx_tape_dropout_len = fx.tape_dropout_len;
    let fx_edge_ringing = fx.edge_ringing;
    let fx_luma_noise = fx.luma_noise;
    let fx_head_switching_smear = fx.head_switching_smear;
    let fx_tape_trail_smear = fx.tape_trail_smear;

    loop {
        // Read groups_per_batch * 4 film frames
        let mut film_buf: Vec<Vec<u8>> = Vec::with_capacity(groups_per_batch * 4);
        for _ in 0..groups_per_batch * 4 {
            let mut buf = vec![0u8; frame_bytes];
            match reader_buf.read_exact(&mut buf) {
                Ok(()) => film_buf.push(buf),
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => {
                    eprintln!("Read error: {}", e);
                    break;
                }
            }
        }

        if film_buf.is_empty() {
            break;
        }

        // Expand film frames into NTSC jobs
        // Each job is: (field1_idx, field2_idx_or_same, ntsc_frame_num, is_interlaced)
        // where field indices refer to film_buf
        let mut jobs: Vec<(usize, usize, u32)> = Vec::new();
        let mut fi = 0;
        while fi + 3 < film_buf.len() {
            // 3:2 pulldown: A,B,C,D -> (A,A), (B,B), (B,C), (C,D), (D,D)
            let base_ntsc = ntsc_num + jobs.len() as u32;
            jobs.push((fi, fi, base_ntsc));         // A,A clean
            jobs.push((fi + 1, fi + 1, base_ntsc + 1)); // B,B clean
            jobs.push((fi + 1, fi + 2, base_ntsc + 2)); // B,C combed
            jobs.push((fi + 2, fi + 3, base_ntsc + 3)); // C,D combed
            jobs.push((fi + 3, fi + 3, base_ntsc + 4)); // D,D clean
            fi += 4;
        }
        // Remaining < 4 frames as progressive
        while fi < film_buf.len() {
            let base_ntsc = ntsc_num + jobs.len() as u32;
            jobs.push((fi, fi, base_ntsc));
            fi += 1;
        }

        if jobs.is_empty() {
            break;
        }

        let results: Vec<Vec<u8>> = jobs
            .par_iter()
            .map(|(f1_idx, f2_idx, fnum)| {
                let f1 = &film_buf[*f1_idx];
                let f2 = &film_buf[*f2_idx];

                TL_ENCODER.with(|enc| {
                    TL_DECODER.with(|dec| {
                        let mut enc_ref = enc.borrow_mut();

                        let signal = if f1_idx == f2_idx {
                            enc_ref.encode_frame(f1, in_w, in_h, *fnum)
                        } else {
                            enc_ref.encode_frame_interlaced(f1, in_w, in_h, f2, *fnum)
                        };

                        if effects_active {
                            TL_SIGNAL_BUF.with(|sb| {
                                let mut sb_ref = sb.borrow_mut();
                                sb_ref.resize(signal.len(), 0.0);
                                sb_ref.copy_from_slice(signal);
                                let fx = SignalEffects {
                                    noise: fx_noise,
                                    ghosts: fx_ghosts.clone(),
                                    attenuation: fx_attenuation,
                                    jitter: fx_jitter,
                                    vhs_luma_bw: fx_vhs_luma_bw,
                                    color_under_bw: fx_color_under_bw,
                                    tape_dropout_rate: fx_tape_dropout_rate,
                                    tape_dropout_len: fx_tape_dropout_len,
                                    edge_ringing: fx_edge_ringing,
                                    luma_noise: fx_luma_noise,
                                    head_switching_smear: fx_head_switching_smear,
                                    tape_trail_smear: fx_tape_trail_smear,
                                };
                                let mut rng = rand::rng();
                                fx.apply(&mut sb_ref, SAMPLE_RATE, &mut rng);
                                let mut dec_ref = dec.borrow_mut();
                                dec_ref.decode_frame(&sb_ref, out_w, out_h, comb_1h).to_vec()
                            })
                        } else {
                            let mut dec_ref = dec.borrow_mut();
                            dec_ref.decode_frame(signal, out_w, out_h, comb_1h).to_vec()
                        }
                    })
                })
            })
            .collect();

        for result in &results {
            writer_buf.write_all(result).context("write failed")?;
        }

        film_idx += film_buf.len() as u32;
        ntsc_num += results.len() as u32;
        pb.inc(film_buf.len() as u64);

        let elapsed = total_start.elapsed().as_secs_f64();
        let fps_actual = film_idx as f64 / elapsed;
        pb.set_message(format!("{:.1} film fps", fps_actual));
    }

    pb.finish_and_clear();

    drop(writer_buf);
    let _ = reader.wait();
    let _ = writer.wait();

    let total_elapsed = total_start.elapsed().as_secs_f64();
    eprintln!(
        "Done: {} film frames -> {} NTSC frames in {:.1}s",
        film_idx, ntsc_num, total_elapsed
    );

    mux_audio(input, output);
    Ok(())
}

// --- Helper functions ---

fn resolve_threads(threads: Option<usize>) -> usize {
    threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    })
}

fn make_progress_bar(total_frames: usize, prefix: &str) -> ProgressBar {
    let label = if prefix.is_empty() { "Processing" } else { prefix };
    if total_frames > 0 {
        let pb = ProgressBar::new(total_frames as u64);
        pb.set_style(
            ProgressStyle::with_template(
                &format!("{label} {{bar:40.cyan/blue}} {{pos}}/{{len}} [{{elapsed_precise}}<{{eta_precise}}, {{msg}}]"),
            )
            .unwrap()
            .progress_chars("##-"),
        );
        pb
    } else {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::with_template(
                &format!("{label} {{spinner}} {{pos}} frames [{{elapsed_precise}}, {{msg}}]"),
            )
            .unwrap(),
        );
        pb
    }
}

fn spawn_ffmpeg_reader(input: &str) -> Result<std::process::Child> {
    Command::new("ffmpeg")
        .args([
            "-i", input, "-f", "rawvideo", "-pix_fmt", "rgb24", "-v", "error", "pipe:1",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("Failed to spawn ffmpeg reader. Is ffmpeg installed?")
}

#[allow(clippy::too_many_arguments)]
fn spawn_ffmpeg_writer(
    output: &str,
    out_w: usize,
    out_h: usize,
    fps: &str,
    preset: &str,
    crf: u32,
    interlaced: bool,
    lossless: bool,
) -> Result<std::process::Child> {
    let mut args = vec![
        "-y".to_string(),
        "-f".to_string(),
        "rawvideo".to_string(),
        "-pix_fmt".to_string(),
        "rgb24".to_string(),
        "-s".to_string(),
        format!("{}x{}", out_w, out_h),
        "-r".to_string(),
        fps.to_string(),
        "-i".to_string(),
        "pipe:0".to_string(),
    ];

    if interlaced {
        args.extend([
            "-vf".to_string(),
            "setfield=tff".to_string(),
            "-flags".to_string(),
            "+ilme+ildct".to_string(),
            "-top".to_string(),
            "1".to_string(),
        ]);
    }

    if lossless {
        if output.ends_with(".mkv") {
            // FFV1 lossless in MKV
            args.extend([
                "-c:v".to_string(),
                "ffv1".to_string(),
                "-level".to_string(),
                "3".to_string(),
                "-pix_fmt".to_string(),
                "yuv444p".to_string(),
            ]);
        } else {
            // x264 lossless (QP 0) in MP4
            args.extend([
                "-c:v".to_string(),
                "libx264".to_string(),
                "-preset".to_string(),
                preset.to_string(),
                "-qp".to_string(),
                "0".to_string(),
                "-pix_fmt".to_string(),
                "yuv444p".to_string(),
            ]);
        }
    } else {
        // Lossy path
        args.extend([
            "-c:v".to_string(),
            "libx264".to_string(),
            "-preset".to_string(),
            preset.to_string(),
            "-crf".to_string(),
            format!("{}", crf),
            "-pix_fmt".to_string(),
            "yuv420p".to_string(),
        ]);
    }

    args.extend([
        "-v".to_string(),
        "error".to_string(),
        output.to_string(),
    ]);

    let args_ref: Vec<&str> = args.iter().map(|s| s.as_str()).collect();

    Command::new("ffmpeg")
        .args(&args_ref)
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("Failed to spawn ffmpeg writer. Is ffmpeg installed?")
}

fn ffprobe_video(path: &str) -> Result<(usize, usize, f64, String, usize)> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames",
            "-of",
            "csv=p=0",
            path,
        ])
        .output()
        .context("Failed to run ffprobe. Is ffmpeg installed?")?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let parts: Vec<&str> = stdout.trim().split(',').collect();

    if parts.len() < 3 {
        bail!("Cannot probe video '{}'. ffprobe output: {}", path, stdout);
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

    let total_frames: usize = parts
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    Ok((w, h, fps, fps_raw, total_frames))
}

fn mux_audio(source: &str, video_path: &str) {
    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            source,
        ])
        .output();

    let has_audio = match probe {
        Ok(out) => String::from_utf8_lossy(&out.stdout).contains("audio"),
        Err(_) => false,
    };

    if !has_audio {
        return;
    }

    let tmp = format!("{}.mux.mp4", video_path);
    let result = Command::new("ffmpeg")
        .args([
            "-y",
            "-i",
            video_path,
            "-i",
            source,
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            &tmp,
        ])
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
