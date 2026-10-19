//! Graphical front end for the NTSC composite simulator: live preview of
//! the effect settings, presets, and batch processing.

#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod app;
mod params_ui;
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
}

fn main() -> eframe::Result {
    let args = GuiArgs::parse();
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
