//! Graphical front end for the NTSC composite simulator: live preview of
//! the effect settings, presets, and batch processing.

#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod app;
mod crash;
mod params_ui;
mod playback;
mod process;
mod viewer;
mod worker;

use std::path::PathBuf;

use clap::Parser;
use eframe::egui;

/// Command-line options of the GUI.  All are optional; they set the
/// starting state, which is handy for scripted screenshots.
#[derive(Parser, Debug, Clone, Default)]
#[command(name = "ntsc-gui", about = "NTSC composite simulator GUI")]
pub struct GuiArgs {
    /// Image, video or preset (.json) to open at start
    pub open: Option<PathBuf>,
    /// Built-in preset to start with (default: mq; "none" for no effects)
    #[arg(long)]
    pub preset: Option<String>,
    /// Playback-deck profile to start with (deck-basic, deck-svhs-tbc, deck-edit)
    #[arg(long)]
    pub deck: Option<String>,
    /// Start a video at this time (seconds)
    #[arg(long)]
    pub time: Option<f64>,
    /// Starting zoom: fit, 1, 2 or 4
    #[arg(long)]
    pub zoom: Option<String>,
    /// Starting view: split, side or after
    #[arg(long)]
    pub view: Option<String>,
    /// Open the Process dialog at start
    #[arg(long)]
    pub process: bool,
    /// Start the Process job immediately (with --process and a file to open)
    #[arg(long, hide = true)]
    pub autostart: bool,
    /// With --screenshot: also write the preview (preview.png) and the
    /// equivalent command line (cli.txt) into this directory
    #[arg(long, hide = true)]
    pub dump: Option<PathBuf>,
    /// Open a dropdown at start: preset or deck (for screenshots)
    #[arg(long, hide = true)]
    pub open_menu: Option<String>,
    /// Save a screenshot of the window to this PNG once the preview has
    /// rendered, then exit
    #[arg(long)]
    pub screenshot: Option<PathBuf>,
    /// Start playing the video once its first frame is loaded
    #[arg(long, hide = true)]
    pub play: bool,
    /// Play for this many seconds, then take the screenshot and exit
    /// (with --dump: also write each shown frame and the playback figures)
    #[arg(long, hide = true)]
    pub play_seconds: Option<f64>,
    /// Play every frame instead of keeping to real time
    #[arg(long, hide = true)]
    pub play_every_frame: bool,
    /// With --play-seconds: pause at the end and capture the paused
    /// preview instead of the playing one
    #[arg(long, hide = true)]
    pub play_pause: bool,
    /// Worker threads for preview and playback processing (default: all
    /// logical cores)
    #[arg(long, hide = true)]
    pub threads: Option<usize>,
    /// Output size WxH instead of the automatic size
    #[arg(long, hide = true)]
    pub size: Option<String>,
    /// Test the crash reports: panic in the "render" worker or on the
    /// "ui" thread once the window is up, or fail an "alloc"ation
    #[arg(long, hide = true)]
    pub crash_test: Option<String>,
}

#[global_allocator]
static ALLOC: crash::LoggingAlloc = crash::LoggingAlloc;

fn main() -> eframe::Result {
    crash::install();
    let args = GuiArgs::parse();
    if args.crash_test.as_deref() == Some("alloc") {
        // A request no system can meet; the failure is logged, then the
        // process aborts as Rust does for any failed allocation.
        let v: Vec<u8> = Vec::with_capacity(isize::MAX as usize / 2);
        std::hint::black_box(&v);
    }
    if let Some(n) = args.threads {
        rayon::ThreadPoolBuilder::new().num_threads(n.max(1)).build_global().ok();
    }
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("NTSC Composite Simulator")
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([900.0, 600.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native(
        "ntsc-gui",
        options,
        Box::new(move |cc| Ok(Box::new(app::App::new(cc, args)))),
    )
}
