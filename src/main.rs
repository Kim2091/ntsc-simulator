//! Command-line front end for the NTSC composite simulator.

use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::Result;
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
                .help("VHS preset. Fills in every flag that is not given explicitly; explicit flags always win."),
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
            }
        }
        values.seed = m.get_one::<u64>("seed").copied();
        Ok(Self { values, vhs_preset: m.get_one::<String>("vhs-preset").cloned() })
    }

    fn update_from_arg_matches(&mut self, m: &ArgMatches) -> Result<(), clap::Error> {
        *self = Self::from_arg_matches(m)?;
        Ok(())
    }
}

impl EffectArgs {
    /// The final values: explicit flags, then the preset for the rest.
    fn resolve(&self) -> ParamValues {
        let mut values = self.values.clone();
        if let Some(name) = &self.vhs_preset {
            let preset = presets::builtin(name).expect("validated by clap");
            values.fill_from(&preset.to_values());
        }
        values
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
        /// Output video file
        #[arg(short, long, default_value = "output.mp4")]
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
        effects: EffectArgs,
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
            pipeline::process_images(&input, &output, width, height, comb_1h, &effects.resolve(), &progress)
        }
        Commands::Roundtrip {
            input, output, width, height, comb_1h, crf, preset, lossless, threads, telecine, effects,
        } => {
            let opts = VideoOptions { width, height, comb_1h, crf, preset, lossless, threads, telecine };
            pipeline::process_videos(&input, &output, &opts, &effects.resolve(), &progress)
        }
        Commands::Colorbars { output, save_source, width, height, comb_1h, effects } => {
            pipeline::process_colorbars(
                &output, save_source.as_deref(), width, height, comb_1h, &effects.resolve(), &progress,
            )
        }
    }
}
