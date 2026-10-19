//! Command-line front end for the NTSC composite simulator.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result};
use clap::builder::PossibleValuesParser;
use clap::{Arg, ArgAction, ArgMatches, Args, Command, FromArgMatches, Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};

use ntsc_composite_simulator::params::{Kind, ParamValues, RangeParam, SPECS};
use ntsc_composite_simulator::pipeline::{self, Progress, VideoOptions};
use ntsc_composite_simulator::presets;

#[derive(Parser)]
#[command(name = "ntsc-composite-simulator")]
#[command(about = "NTSC Composite Video Simulator (Rust)")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// Effect arguments, generated from the shared parameter table.
///
/// All numeric parameters accept either a single value (e.g. `0.05`) or a
/// `min..max` range (e.g. `0.02..0.10`).  When a range is given, a random
/// value is sampled uniformly for each file processed in a batch.
#[derive(Clone, Debug)]
struct EffectArgs {
    values: ParamValues,
    vhs_preset: Option<String>,
    preset_file: Option<PathBuf>,
}

fn parse_range(s: &str) -> Result<RangeParam, String> {
    s.parse()
}

impl Args for EffectArgs {
    fn augment_args(mut cmd: Command) -> Command {
        for spec in SPECS {
            let arg = Arg::new(spec.id).long(spec.id).help(spec.help).help_heading(spec.group.label());
            let arg = match spec.kind {
                Kind::Number { .. } => arg
                    .value_name("VALUE|MIN..MAX")
                    .value_parser(parse_range)
                    .action(ArgAction::Set),
                Kind::Flag => arg.action(ArgAction::SetTrue),
                Kind::Text => arg.value_name("TEXT").action(ArgAction::Set),
                Kind::Choice { options, .. } => arg
                    .value_parser(PossibleValuesParser::new(options.iter().copied()))
                    .action(ArgAction::Set),
            };
            cmd = cmd.arg(arg);
        }
        let names: Vec<&'static str> = presets::BUILTIN.iter().map(|p| p.name).collect();
        cmd.arg(
            Arg::new("seed")
                .long("seed")
                .value_parser(clap::value_parser!(u64))
                .help_heading("Presets and seed")
                .help(
                    "Seed for every random choice (sampled ranges and per-frame effects), so a run \
                     can be reproduced exactly. In batch mode each file gets its own seed derived from this one.",
                ),
        )
        .arg(
            Arg::new("vhs-preset")
                .long("vhs-preset")
                .value_parser(PossibleValuesParser::new(names))
                .help_heading("Presets and seed")
                .help(
                    "Built-in preset: lq, mq, hq (full VHS look) or legacy-lq, legacy-mq, legacy-hq (the original \
                     tape-path settings only). Fills in every flag that is not given explicitly; explicit flags always win.",
                ),
        )
        .arg(
            Arg::new("preset-file")
                .long("preset-file")
                .value_name("PATH")
                .value_parser(clap::value_parser!(PathBuf))
                .help_heading("Presets and seed")
                .help(
                    "JSON preset file (as saved by the GUI). Its values fill in flags not given explicitly, \
                     before --vhs-preset.",
                ),
        )
    }

    fn augment_args_for_update(cmd: Command) -> Command {
        Self::augment_args(cmd)
    }
}

impl FromArgMatches for EffectArgs {
    fn from_arg_matches(m: &ArgMatches) -> Result<Self, clap::Error> {
        let mut values = ParamValues::default();
        for spec in SPECS {
            match spec.kind {
                Kind::Number { .. } => {
                    if let Some(v) = m.get_one::<RangeParam>(spec.id) {
                        values.numbers.insert(spec.id, *v);
                    }
                }
                Kind::Flag => {
                    if m.get_flag(spec.id) {
                        values.flags.insert(spec.id, true);
                    }
                }
                Kind::Text => {
                    if let Some(t) = m.get_one::<String>(spec.id) {
                        values.texts.insert(spec.id, t.clone());
                    }
                }
                Kind::Choice { .. } => {
                    if let Some(c) = m.get_one::<String>(spec.id) {
                        values.set_from_str(spec.id, c).expect("validated by clap");
                    }
                }
            }
        }
        values.seed = m.get_one::<u64>("seed").copied();
        Ok(Self {
            values,
            vhs_preset: m.get_one::<String>("vhs-preset").cloned(),
            preset_file: m.get_one::<PathBuf>("preset-file").cloned(),
        })
    }

    fn update_from_arg_matches(&mut self, m: &ArgMatches) -> Result<(), clap::Error> {
        *self = Self::from_arg_matches(m)?;
        Ok(())
    }
}

impl EffectArgs {
    /// The final values: explicit flags first, then the preset file, then
    /// the built-in preset for whatever is still unset.
    fn resolve(&self) -> Result<ParamValues> {
        let mut values = self.values.clone();
        if let Some(path) = &self.preset_file {
            values.fill_from(&load_preset(path)?.values);
        }
        if let Some(name) = &self.vhs_preset {
            let preset = presets::builtin(name).expect("validated by clap");
            values.fill_from(&preset.to_values());
        }
        Ok(values)
    }
}

#[derive(Subcommand)]
enum Commands {
    /// Roundtrip a single image through the NTSC composite pipeline
    Image {
        /// Input image file or directory of images
        input: PathBuf,
        /// Output image file
        #[arg(short, long, default_value = "output.png")]
        output: PathBuf,
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
        effects: EffectArgs,
    },
    /// Encode video to composite and decode back (roundtrip)
    Roundtrip {
        /// Input video file or directory of videos
        input: PathBuf,
        /// Output video file. Lossless RGB by default: .mkv/.avi (FFV1) or .mp4/.mov (x264 RGB)
        #[arg(short, long, default_value = "output.mkv")]
        output: PathBuf,
        /// Output width
        #[arg(long, default_value = "640")]
        width: u32,
        /// Output height
        #[arg(long, default_value = "480")]
        height: u32,
        /// Use 1H line-delay comb filter
        #[arg(long)]
        comb_1h: bool,
        /// Lossy output: x264 4:2:0 at --crf. Halves chroma resolution and smooths
        /// noise, so it is no longer the default.
        #[arg(long)]
        lossy: bool,
        /// x264 CRF quality with --lossy (0=lossless, 51=worst)
        #[arg(long, default_value = "17")]
        crf: u32,
        /// x264 preset (speed) for --lossy and lossless .mp4
        #[arg(long, default_value = "fast")]
        preset: String,
        /// Lossless output (the default now; kept for compatibility)
        #[arg(long, hide = true)]
        lossless: bool,
        /// Also write the raw RGB24 frames given to the encoder (bit-exactness checks)
        #[arg(long, hide = true)]
        raw_copy: Option<PathBuf>,
        /// Number of parallel worker threads (default: all logical cores)
        #[arg(long)]
        threads: Option<usize>,
        /// Enable 3:2 pulldown telecine (24fps film -> 29.97fps interlaced)
        #[arg(long)]
        telecine: bool,
        #[command(flatten)]
        effects: EffectArgs,
    },
    /// Extract one frame of a video as an RGB PNG, with the same colour
    /// conversion as `roundtrip` (no effects)
    ExtractFrame {
        /// Input video file
        input: PathBuf,
        /// Output PNG
        #[arg(short, long, default_value = "frame.png")]
        output: PathBuf,
        /// Time in seconds
        #[arg(long, default_value = "0")]
        time: f64,
        /// Colour matrix: auto, 601 or 709
        #[arg(long, default_value = "auto")]
        input_matrix: String,
        /// Range: auto, limited or full
        #[arg(long, default_value = "auto")]
        input_range: String,
    },
    /// Check an MPEG-2 stream (.m2v elementary stream or .vob/.mpg program
    /// stream) against the DVD-Video NTSC video limits
    DvdCheck {
        /// MPEG-2 file
        input: PathBuf,
    },
    /// Generate SMPTE color bars through the NTSC pipeline
    Colorbars {
        /// Output image file
        #[arg(short, long, default_value = "colorbars.png")]
        output: PathBuf,
        /// Also save the source color bar pattern (before NTSC processing)
        #[arg(long)]
        save_source: Option<PathBuf>,
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
        effects: EffectArgs,
    },
}

/// Progress on stderr, with an indicatif bar while a file is processed.
struct CliProgress {
    bar: Mutex<Option<ProgressBar>>,
}

impl Progress for CliProgress {
    fn log(&self, msg: &str) {
        match self.bar.lock().unwrap().as_ref() {
            Some(bar) => bar.println(msg),
            None => eprintln!("{}", msg),
        }
    }

    fn start(&self, total_frames: usize, label: &str) {
        *self.bar.lock().unwrap() = Some(make_progress_bar(total_frames, label));
    }

    fn advance(&self, frames: u64, rate: &str) {
        if let Some(bar) = self.bar.lock().unwrap().as_ref() {
            bar.inc(frames);
            if !rate.is_empty() {
                bar.set_message(rate.to_string());
            }
        }
    }

    fn finish(&self) {
        if let Some(bar) = self.bar.lock().unwrap().take() {
            bar.finish_and_clear();
        }
    }
}

/// Print a DVD-Video compliance report for an MPEG-2 stream.
fn dvd_check(input: &Path) -> Result<()> {
    use ntsc_composite_simulator::{dvd, mpeg2};
    let ext = input.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
    let es = if ext == "m2v" || ext == "mpv" {
        std::fs::read(input).with_context(|| format!("cannot read {}", input.display()))?
    } else {
        // Extract the video elementary stream from a program stream.
        let work = dvd::WorkDir::new()?;
        let es_path = work.0.join("video.m2v");
        let st = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-i"])
            .arg(input)
            .args(["-map", "0:v:0", "-c", "copy", "-f", "mpeg2video"])
            .arg(&es_path)
            .status()
            .context("Failed to run ffmpeg")?;
        if !st.success() {
            anyhow::bail!("cannot extract the video stream from {}", input.display());
        }
        std::fs::read(&es_path)?
    };
    let a = mpeg2::analyze(&es)?;
    println!("DVD-Video compliance: {}", input.display());
    println!("{} pictures, {} GOPs; {} with repeat_first_field", a.pictures, a.stream.gops.len(), a.field_counts.0);
    let types: String = a.stream.display_types().into_iter().take(36).collect();
    println!("picture types (display order, first 36): {}", types);
    let mut all_ok = true;
    for (name, value, ok) in &a.checks {
        println!("  [{}] {:<22} {}", if *ok { "ok" } else { "FAIL" }, name, value);
        all_ok &= *ok;
    }
    if let Some(m) = a.stream.seq.intra_matrix {
        println!("  custom intra matrix (zigzag): {:?}", m);
    }
    if let Some(m) = a.stream.seq.non_intra_matrix {
        println!("  custom non-intra matrix (zigzag): {:?}", m);
    }
    let probe = std::process::Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=codec_name,profile,level,width,height,pix_fmt,field_order,sample_aspect_ratio,display_aspect_ratio,r_frame_rate,avg_frame_rate,color_space,color_range,color_primaries",
               "-of", "default=nw=1"])
        .arg(input)
        .output();
    if let Ok(o) = probe {
        println!("ffprobe:");
        for line in String::from_utf8_lossy(&o.stdout).lines() {
            println!("  {}", line);
        }
    }
    println!("{}", if all_ok { "RESULT: compliant with the checked DVD-Video limits" } else { "RESULT: NOT compliant" });
    Ok(())
}

fn load_preset(path: &Path) -> Result<presets::Preset> {
    presets::load_preset_file(path).context("--preset-file")
}

fn make_progress_bar(total_frames: usize, prefix: &str) -> ProgressBar {
    let label = if prefix.is_empty() { "Processing" } else { prefix };
    if total_frames > 0 {
        let pb = ProgressBar::new(total_frames as u64);
        pb.set_style(
            ProgressStyle::with_template(&format!(
                "{label} {{bar:40.cyan/blue}} {{pos}}/{{len}} [{{elapsed_precise}}<{{eta_precise}}, {{msg}}]"
            ))
            .unwrap()
            .progress_chars("##-"),
        );
        pb
    } else {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::with_template(&format!("{label} {{spinner}} {{pos}} frames [{{elapsed_precise}}, {{msg}}]"))
                .unwrap(),
        );
        pb
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let progress = CliProgress { bar: Mutex::new(None) };

    match cli.command {
        Commands::Image { input, output, width, height, comb_1h, effects } => {
            pipeline::process_images(&input, &output, width, height, comb_1h, &effects.resolve()?, &progress)
        }
        Commands::Roundtrip {
            input, output, width, height, comb_1h, lossy, crf, preset, lossless, raw_copy, threads, telecine, effects,
        } => {
            let _ = lossless;
            let opts = VideoOptions { width, height, comb_1h, crf, preset, lossy, raw_copy, threads, telecine };
            pipeline::process_videos(&input, &output, &opts, &effects.resolve()?, &progress)
        }
        Commands::ExtractFrame { input, output, time, input_matrix, input_range } => {
            let mut values = ParamValues::default();
            values.set_from_str("input-matrix", &input_matrix).map_err(anyhow::Error::msg)?;
            values.set_from_str("input-range", &input_range).map_err(anyhow::Error::msg)?;
            let colour = pipeline::input_colour(&input, &values);
            let (rgb, w, h) = pipeline::grab_video_frame(&input, time, &values)?;
            image::RgbImage::from_raw(w as u32, h as u32, rgb)
                .context("bad frame")?
                .save(&output)
                .with_context(|| format!("cannot save {}", output.display()))?;
            eprintln!("{} ({}x{}, {}) -> {}", input.display(), w, h, colour.description, output.display());
            Ok(())
        }
        Commands::DvdCheck { input } => dvd_check(&input),
        Commands::Colorbars { output, save_source, width, height, comb_1h, effects } => {
            pipeline::process_colorbars(
                &output, save_source.as_deref(), width, height, comb_1h, &effects.resolve()?, &progress,
            )
        }
    }
}
