//! Main window: settings panel, preview, toolbar and status bar.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::time::{Duration, Instant};

use eframe::egui::{self, Vec2};
use ntsc_composite_simulator::colorbars;
use ntsc_composite_simulator::params::{Group, ParamValues};
use ntsc_composite_simulator::pipeline;
use ntsc_composite_simulator::presets::{self, Preset, BUILTIN};

use crate::dialog::FileDialogTask;
use crate::params_ui::ParamEditor;
use crate::playback::{PlayRequest, Playback};
use crate::process::{JobSettings, ProcessDialog};
use crate::viewer::{ViewMode, Viewer, Zoom};
use crate::worker::{grab_frame_async, RenderRequest, Renderer, SourceImage};
use crate::GuiArgs;

/// How long settings must stay unchanged before a preview render starts.
const DEBOUNCE: Duration = Duration::from_millis(150);

#[derive(Clone, Debug, PartialEq)]
enum PresetChoice {
    None,
    Builtin(&'static str),
    Custom(usize),
}

enum Source {
    Bars,
    Image(PathBuf),
    Video { path: PathBuf, duration: f64, fps: f64, time: f64, w: usize, h: usize },
}

/// What the open file dialog's choice is for.
enum DialogFor {
    Open,
    LoadPreset,
    /// With the preset name to use if the file has none.
    SavePreset(String),
    /// With the frame to save.
    SavePng(Vec<u8>, usize, usize),
}

/// Settings that playback is running with; a change restarts it.
#[derive(Clone, PartialEq)]
struct PlayKey {
    values: ParamValues,
    out: (usize, usize),
    comb_1h: bool,
    every_frame: bool,
}

/// Everything that affects the preview; a change triggers a new render.
#[derive(Clone, PartialEq)]
struct RenderKey {
    values: ParamValues,
    out: (usize, usize),
    comb_1h: bool,
    frame: u64,
    source_gen: u64,
}

pub struct App {
    editor: ParamEditor,
    /// Editor state of the selected preset, to show "modified" and reset.
    base: ParamEditor,
    preset: PresetChoice,
    /// Playback-deck profile combined with the preset.
    deck: Option<&'static str>,
    custom_presets: Vec<Preset>,

    seed: u64,
    frame: u64,
    comb_1h: bool,
    auto_size: bool,
    out_w: u32,
    out_h: u32,

    source: Source,
    src: Arc<SourceImage>,
    source_gen: u64,
    grab: Option<Receiver<Result<SourceImage, String>>>,

    before_tex: Option<egui::TextureHandle>,
    after_tex: Option<egui::TextureHandle>,
    after_rgb: Option<(Vec<u8>, usize, usize)>,

    renderer: Renderer,
    next_id: u64,
    shown_id: u64,
    sent_key: Option<RenderKey>,
    dirty_since: Option<Instant>,
    last_ms: f64,
    sampled: Vec<String>,

    viewer: Viewer,
    process: ProcessDialog,
    status: Option<(bool, String)>,

    screenshot: Option<PathBuf>,
    dump: Option<PathBuf>,
    open_menu: Option<String>,
    screenshot_wait: u32,
    screenshot_done: Arc<AtomicBool>,
    screenshot_sent: bool,

    playback: Option<Playback>,
    play_loop: bool,
    play_every_frame: bool,
    /// The last frame shown by playback: (frame number, source pixels).
    play_shown: Option<(u64, Vec<u8>)>,
    /// When the playback settings started to differ from the running ones.
    play_dirty_since: Option<Instant>,
    /// Figures of the last playback, shown in the status bar after pausing.
    play_summary: Option<String>,
    /// The running grab only adds the frames before the paused frame;
    /// the shown preview stays.
    grab_quiet: bool,
    /// Scripted playback (--play / --play-seconds): (seconds, started at).
    auto_play: Option<(Option<f64>, Option<Instant>)>,
    /// Scripted playback pauses before the capture (--play-pause).
    play_pause: bool,
    auto_paused: Option<Instant>,
    /// Panic once the preview has rendered (--crash-test ui).
    crash_test_ui: bool,
    /// The open file dialog and what its choice is for.
    dialog: Option<(DialogFor, FileDialogTask)>,
}

fn bars_source() -> SourceImage {
    SourceImage { rgb: colorbars::generate_colorbars(640, 480), w: 640, h: 480, history: Vec::new() }
}

fn quote(arg: &str) -> String {
    if arg.is_empty() || arg.contains(|c: char| c.is_whitespace() || ";&|<>\"'".contains(c)) {
        format!("\"{}\"", arg.replace('"', "\\\""))
    } else {
        arg.to_string()
    }
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>, args: GuiArgs) -> Self {
        cc.egui_ctx.set_theme(egui::Theme::Dark);
        let mut app = Self {
            editor: ParamEditor::from_values(&ParamValues::default()),
            base: ParamEditor::from_values(&ParamValues::default()),
            preset: PresetChoice::None,
            deck: None,
            custom_presets: Vec::new(),
            seed: 1,
            frame: 0,
            comb_1h: false,
            auto_size: true,
            out_w: 640,
            out_h: 480,
            source: Source::Bars,
            src: Arc::new(bars_source()),
            source_gen: 0,
            grab: None,
            before_tex: None,
            after_tex: None,
            after_rgb: None,
            renderer: Renderer::new(cc.egui_ctx.clone(), args.crash_test.as_deref() == Some("render")),
            next_id: 1,
            shown_id: 0,
            sent_key: None,
            dirty_since: None,
            last_ms: 0.0,
            sampled: Vec::new(),
            viewer: Viewer::default(),
            process: ProcessDialog::default(),
            status: None,
            screenshot: args.screenshot.clone(),
            dump: args.dump.clone(),
            open_menu: args.open_menu.clone(),
            screenshot_wait: 0,
            screenshot_done: Arc::new(AtomicBool::new(false)),
            screenshot_sent: false,
            playback: None,
            play_loop: true,
            play_every_frame: args.play_every_frame,
            play_shown: None,
            play_dirty_since: None,
            play_summary: None,
            grab_quiet: false,
            auto_play: (args.play || args.play_seconds.is_some()).then_some((args.play_seconds, None)),
            play_pause: args.play_pause,
            auto_paused: None,
            crash_test_ui: args.crash_test.as_deref() == Some("ui"),
            dialog: None,
        };
        app.deck = args.deck.as_deref().and_then(presets::deck).map(|d| d.name);
        if let Some((w, h)) = args.size.as_deref().and_then(|s| s.split_once('x')) {
            if let (Ok(w), Ok(h)) = (w.parse(), h.parse()) {
                (app.auto_size, app.out_w, app.out_h) = (false, w, h);
            }
        }
        let preset = args.preset.as_deref().unwrap_or("mq");
        match presets::builtin(preset) {
            Some(p) => app.select_preset(PresetChoice::Builtin(p.name)),
            None => app.select_preset(PresetChoice::None),
        }
        app.viewer.zoom = match args.zoom.as_deref() {
            Some("1") => Zoom::X1,
            Some("2") => Zoom::X2,
            Some("4") => Zoom::X4,
            _ => Zoom::Fit,
        };
        app.viewer.mode = match args.view.as_deref() {
            Some("side") => ViewMode::SideBySide,
            Some("after") => ViewMode::After,
            _ => ViewMode::Split,
        };
        app.set_before_texture(&cc.egui_ctx);
        if let Some(path) = &args.open {
            app.open_path(&cc.egui_ctx, path);
        }
        if let Some(t) = args.time {
            if let Source::Video { path, fps, time, .. } = &mut app.source {
                *time = t;
                app.frame = (t * *fps).round() as u64;
                app.grab = Some(grab_frame_async(cc.egui_ctx.clone(), path.clone(), t, ParamValues::default()));
            }
        }
        if args.process {
            app.process.open = true;
            app.process.autostart = args.autostart;
            if let Some(path) = &args.open {
                app.process.set_input(path);
            }
        }
        app
    }

    // ── Presets ──────────────────────────────────────────────────────────

    fn preset_values(&self, choice: &PresetChoice) -> ParamValues {
        match choice {
            PresetChoice::None => ParamValues::default(),
            PresetChoice::Builtin(name) => presets::builtin(name).map(|p| p.to_values()).unwrap_or_default(),
            PresetChoice::Custom(i) => self.custom_presets[*i].values.clone(),
        }
    }

    fn preset_label(&self, choice: &PresetChoice) -> String {
        match choice {
            PresetChoice::None => "None".into(),
            PresetChoice::Builtin(name) => name.to_string(),
            PresetChoice::Custom(i) => self.custom_presets[*i].name.clone(),
        }
    }

    fn select_preset(&mut self, choice: PresetChoice) {
        // The deck profile's values win over the preset's.
        let mut values = self.deck.and_then(presets::deck).map(|d| d.to_values()).unwrap_or_default();
        values.fill_from(&self.preset_values(&choice));
        if let Some(seed) = values.seed {
            self.seed = seed;
        }
        self.editor = ParamEditor::from_values(&values);
        self.base = self.editor.clone();
        self.preset = choice;
    }

    fn is_modified(&self) -> bool {
        self.editor != self.base
    }

    fn current_values(&self) -> ParamValues {
        let mut v = self.editor.to_values();
        v.seed = Some(self.seed);
        v
    }

    fn settings_label(&self) -> String {
        format!(
            "preset {}{}{}, seed {}",
            self.preset_label(&self.preset),
            self.deck.map(|d| format!(" + {}", d)).unwrap_or_default(),
            if self.is_modified() { " (modified)" } else { "" },
            self.seed
        )
    }

    fn load_preset_file(&mut self, path: &Path) {
        match presets::load_preset_file(path) {
            Ok(p) => {
                let name = p.name.clone();
                self.custom_presets.push(p);
                self.select_preset(PresetChoice::Custom(self.custom_presets.len() - 1));
                self.status = Some((true, format!("Loaded preset \"{}\"", name)));
            }
            Err(e) => self.status = Some((false, format!("{:#}", e))),
        }
    }

    fn save_preset_dialog(&mut self, ctx: &egui::Context) {
        let default_name = match &self.preset {
            PresetChoice::None => "custom".to_string(),
            c => format!("{}-custom", self.preset_label(c)),
        };
        let file_name = format!("{}.json", default_name);
        let task = FileDialogTask::spawn(ctx, move || {
            rfd::FileDialog::new().add_filter("Preset (JSON)", &["json"]).set_file_name(file_name).save_file()
        });
        self.dialog = Some((DialogFor::SavePreset(default_name), task));
    }

    fn save_preset(&mut self, path: &Path, default_name: String) {
        let name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or(default_name);
        let preset = Preset { name: name.clone(), values: self.current_values() };
        match presets::save_preset_file(path, &preset) {
            Ok(()) => {
                self.custom_presets.push(preset);
                let i = self.custom_presets.len() - 1;
                self.select_preset(PresetChoice::Custom(i));
                self.status = Some((true, format!("Saved preset \"{}\" to {}", name, path.display())));
            }
            Err(e) => self.status = Some((false, format!("{:#}", e))),
        }
    }

    // ── Sources ──────────────────────────────────────────────────────────

    fn out_size(&self) -> (usize, usize) {
        if !self.auto_size {
            return (self.out_w.max(16) as usize, self.out_h.max(16) as usize);
        }
        match self.source {
            Source::Image(_) => (self.src.w, self.src.h),
            _ => (640, 480),
        }
    }

    fn set_source_image(&mut self, ctx: &egui::Context, img: SourceImage) {
        self.src = Arc::new(img);
        self.source_gen += 1;
        self.set_before_texture(ctx);
    }

    fn set_before_texture(&mut self, ctx: &egui::Context) {
        let img = egui::ColorImage::from_rgb([self.src.w, self.src.h], &self.src.rgb);
        self.before_tex = Some(ctx.load_texture("before", img, egui::TextureOptions::NEAREST));
    }

    fn open_dialog(&mut self, ctx: &egui::Context) {
        let task = FileDialogTask::spawn(ctx, || {
            let mut exts: Vec<&str> = pipeline::IMAGE_EXTENSIONS.to_vec();
            exts.extend_from_slice(pipeline::VIDEO_EXTENSIONS);
            rfd::FileDialog::new()
                .add_filter("Images and videos", &exts)
                .add_filter("Presets (JSON)", &["json"])
                .pick_file()
        });
        self.dialog = Some((DialogFor::Open, task));
    }

    fn load_preset_dialog(&mut self, ctx: &egui::Context) {
        let task = FileDialogTask::spawn(ctx, || rfd::FileDialog::new().add_filter("Preset (JSON)", &["json"]).pick_file());
        self.dialog = Some((DialogFor::LoadPreset, task));
    }

    /// Act on a file dialog that has closed.
    fn poll_dialog(&mut self, ctx: &egui::Context) {
        let Some(choice) = self.dialog.as_ref().and_then(|(_, task)| task.poll()) else { return };
        let Some((what, _)) = self.dialog.take() else { return };
        let Some(path) = choice else { return };
        match what {
            DialogFor::Open => self.open_path(ctx, &path),
            DialogFor::LoadPreset => self.load_preset_file(&path),
            DialogFor::SavePreset(default_name) => self.save_preset(&path, default_name),
            DialogFor::SavePng(rgb, w, h) => self.save_png(&path, rgb, w, h),
        }
    }

    fn open_path(&mut self, ctx: &egui::Context, path: &Path) {
        if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("json")) {
            self.load_preset_file(path);
        } else if pipeline::is_video_path(path) {
            self.stop_playback_for_new_source();
            let duration = pipeline::video_duration(path).unwrap_or(0.0);
            let (w, h, fps) = pipeline::ffprobe_video(path).map(|v| (v.0, v.1, v.2)).unwrap_or((0, 0, 29.97));
            let time = (duration * 0.1).min(10.0);
            self.source = Source::Video { path: path.to_path_buf(), duration, fps, time, w, h };
            self.grab = Some(grab_frame_async(ctx.clone(), path.to_path_buf(), time, self.current_values()));
            self.grab_quiet = false;
            self.frame = (time * fps).round() as u64;
            self.status = Some((true, format!("Loading frame from {}…", path.display())));
        } else {
            match image::open(path) {
                Ok(img) => {
                    self.stop_playback_for_new_source();
                    let rgb = img.to_rgb8();
                    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
                    self.source = Source::Image(path.to_path_buf());
                    self.set_source_image(ctx, SourceImage { rgb: rgb.into_raw(), w, h, history: Vec::new() });
                    self.status = Some((true, format!("Opened {} ({}×{})", path.display(), w, h)));
                }
                Err(e) => self.status = Some((false, format!("Cannot open {}: {}", path.display(), e))),
            }
        }
    }

    fn source_label(&self) -> String {
        let name = |p: &Path| p.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        match &self.source {
            Source::Bars => "SMPTE colour bars".into(),
            Source::Image(p) => format!("{} ({}×{})", name(p), self.src.w, self.src.h),
            Source::Video { path, .. } => format!("{} (video, {}×{})", name(path), self.src.w, self.src.h),
        }
    }

    // ── Rendering ────────────────────────────────────────────────────────

    fn render_key(&self) -> RenderKey {
        RenderKey {
            values: self.current_values(),
            out: self.out_size(),
            comb_1h: self.comb_1h,
            frame: self.frame,
            source_gen: self.source_gen,
        }
    }

    fn update_render(&mut self, ctx: &egui::Context) {
        if self.playback.is_some() {
            // Playback shows its own frames; still-preview renders that
            // were under way are not shown.
            if let Some(result) = self.renderer.poll() {
                self.shown_id = self.shown_id.max(result.id);
            }
            self.dirty_since = None;
            return;
        }
        if let Some(result) = self.renderer.poll() {
            if result.id > self.shown_id && result.error.is_some() {
                self.shown_id = result.id;
                self.status = Some((false, format!("Render failed: {}", result.error.unwrap_or_default())));
            } else if result.id > self.shown_id {
                self.shown_id = result.id;
                self.play_summary = None;
                self.last_ms = result.ms;
                self.sampled = result.sampled;
                let img = egui::ColorImage::from_rgb([result.w, result.h], &result.rgb);
                self.after_tex = Some(ctx.load_texture("after", img, egui::TextureOptions::NEAREST));
                self.after_rgb = Some((result.rgb, result.w, result.h));
            }
        }

        let key = self.render_key();
        if self.sent_key.as_ref() != Some(&key) {
            let since = *self.dirty_since.get_or_insert_with(Instant::now);
            let first = self.shown_id == 0 && self.next_id == 1;
            let waited = since.elapsed();
            if first || waited >= DEBOUNCE {
                let (out_w, out_h) = key.out;
                self.renderer.submit(RenderRequest {
                    id: self.next_id,
                    src: self.src.clone(),
                    out_w,
                    out_h,
                    comb_1h: key.comb_1h,
                    values: key.values.clone(),
                    frame: key.frame,
                });
                self.next_id += 1;
                self.sent_key = Some(key);
                self.dirty_since = None;
            } else {
                ctx.request_repaint_after(DEBOUNCE - waited);
            }
        } else {
            self.dirty_since = None;
        }
    }

    /// On-screen size for a processed frame: native DVD frames have
    /// non-square samples, so they are shown at their display aspect
    /// (the 704/352-sample active area spans the 4:3 or 16:9 width).
    fn display_size(&self, w: usize, h: usize) -> Vec2 {
        let v = self.current_values();
        let native_dvd = v.choice("format") == "dvd" && v.choice("dvd-output-size") == "native" && v.choice("dvd-output") == "digital";
        if native_dvd && matches!(w, 720 | 704 | 352) {
            let active = if w == 720 { 704.0 } else { w as f32 };
            let dar = if v.choice("dvd-aspect") == "16:9" { 16.0 / 9.0 } else { 4.0 / 3.0 };
            let lines = if h == 240 { 480.0 } else { h as f32 };
            Vec2::new(lines * dar * w as f32 / active, lines)
        } else {
            Vec2::new(w as f32, h as f32)
        }
    }

    fn rendering(&self) -> bool {
        self.shown_id + 1 < self.next_id || self.dirty_since.is_some()
    }

    // ── Playback ─────────────────────────────────────────────────────────

    fn play_key(&self) -> PlayKey {
        PlayKey { values: self.current_values(), out: self.out_size(), comb_1h: self.comb_1h, every_frame: self.play_every_frame }
    }

    /// Why the current settings cannot be played, if they cannot.
    fn play_blocker(&self) -> Option<&'static str> {
        match &self.source {
            Source::Video { duration, w, .. } if *duration > 0.0 && *w > 0 => {}
            _ => return Some("Playback needs a video source"),
        }
        if self.current_values().choice("format") == "dvd" {
            return Some("The DVD path encodes the whole clip first; use Process… to see it in motion");
        }
        None
    }

    /// Start playing at frame `first` (from the start when `first` is past
    /// the end).
    fn start_playback(&mut self, ctx: &egui::Context, first: u64) {
        let Source::Video { path, fps, duration, w, h, .. } = &self.source else { return };
        if self.play_blocker().is_some() {
            return;
        }
        let first = if first as f64 >= (*duration * *fps).floor() { 0 } else { first };
        let key = self.play_key();
        let request = PlayRequest {
            path: path.clone(),
            first_frame: first,
            fps: *fps,
            in_w: *w,
            in_h: *h,
            out_w: key.out.0,
            out_h: key.out.1,
            comb_1h: key.comb_1h,
            values: key.values,
            every_frame: key.every_frame,
        };
        self.stop_playback();
        self.playback = Some(Playback::start(ctx.clone(), request));
        self.play_dirty_since = None;
        self.play_summary = None;
    }

    /// Stop playback without touching the preview.
    fn stop_playback(&mut self) {
        if let Some(mut pb) = self.playback.take() {
            pb.stop(false);
        }
    }

    /// Stop playback before another source is opened.  The last played
    /// frame belongs to the old source, so pausing must not keep it.
    fn stop_playback_for_new_source(&mut self) {
        self.stop_playback();
        self.play_shown = None;
    }

    /// Stop playback and keep the last shown frame as the still preview.
    fn pause(&mut self, ctx: &egui::Context) {
        let Some(mut pb) = self.playback.take() else { return };
        pb.stop(false);
        self.play_summary = Some(format!(
            "Played {} frames at {:.1} fps, {} dropped",
            pb.shown,
            pb.achieved_fps(),
            pb.dropped
        ));
        let Some((frame, src)) = self.play_shown.take() else { return };
        let values = self.current_values();
        let Source::Video { path, fps, time, w, h, .. } = &mut self.source else { return };
        if src.len() != *w * *h * 3 {
            return;
        }
        *time = frame as f64 / *fps;
        let (path, t, w, h) = (path.clone(), *time, *w, *h);
        self.frame = frame;
        // The shown frame is the preview of this frame and source; the
        // frames before it (for the still preview's noise reduction) are
        // grabbed in the background.
        self.src = Arc::new(SourceImage { rgb: src, w, h, history: Vec::new() });
        self.source_gen += 1;
        self.sent_key = Some(self.render_key());
        self.dirty_since = None;
        self.grab = Some(grab_frame_async(ctx.clone(), path, t, values));
        self.grab_quiet = true;
    }

    fn toggle_play(&mut self, ctx: &egui::Context) {
        if self.playback.is_some() {
            self.pause(ctx);
        } else {
            self.start_playback(ctx, self.frame);
        }
    }

    fn update_playback(&mut self, ctx: &egui::Context) {
        if matches!(self.auto_play, Some((_, None))) && self.grab.is_none() && self.shown_id > 0 {
            if let Some((_, started)) = &mut self.auto_play {
                *started = Some(Instant::now());
            }
            self.start_playback(ctx, self.frame);
        }
        if self.playback.is_some() && self.play_blocker().is_some() {
            self.pause(ctx);
        }
        let Some(pb) = &mut self.playback else { return };
        let poll = pb.poll();
        let fps = pb.request.fps;
        let (in_w, in_h, out_w, out_h) = (pb.request.in_w, pb.request.in_h, pb.request.out_w, pb.request.out_h);
        if let Some(f) = poll.frame {
            let before = egui::ColorImage::from_rgb([in_w, in_h], &f.src);
            match &mut self.before_tex {
                Some(t) => t.set(before, egui::TextureOptions::NEAREST),
                None => self.before_tex = Some(ctx.load_texture("before", before, egui::TextureOptions::NEAREST)),
            }
            let after = egui::ColorImage::from_rgb([out_w, out_h], &f.rgb);
            match &mut self.after_tex {
                Some(t) => t.set(after, egui::TextureOptions::NEAREST),
                None => self.after_tex = Some(ctx.load_texture("after", after, egui::TextureOptions::NEAREST)),
            }
            if let (Some(dir), Some((Some(_), _))) = (&self.dump, &self.auto_play) {
                let _ = std::fs::create_dir_all(dir);
                let _ = std::fs::write(dir.join(format!("play_{:05}.rgb", f.frame)), &f.rgb);
            }
            self.frame = f.frame;
            if let Source::Video { time, .. } = &mut self.source {
                *time = f.frame as f64 / fps;
            }
            self.after_rgb = Some((f.rgb, out_w, out_h));
            self.play_shown = Some((f.frame, f.src));
        }
        if let Some(e) = poll.error {
            self.status = Some((false, format!("Playback failed: {}", e)));
            self.pause(ctx);
            return;
        }
        if poll.ended {
            if self.play_loop {
                self.start_playback(ctx, 0);
            } else {
                self.pause(ctx);
            }
            return;
        }
        if let Some(d) = poll.wake_in {
            ctx.request_repaint_after(d);
        }
        // Changed settings restart playback at the next frame, once they
        // have stayed unchanged for a moment.
        let key = self.play_key();
        let running = self.playback.as_ref().map(|pb| {
            let r = &pb.request;
            PlayKey { values: r.values.clone(), out: (r.out_w, r.out_h), comb_1h: r.comb_1h, every_frame: r.every_frame }
        });
        if running.as_ref() != Some(&key) {
            let waited = self.play_dirty_since.get_or_insert_with(Instant::now).elapsed();
            if waited >= DEBOUNCE {
                let next = self.play_shown.as_ref().map_or(self.frame, |(f, _)| f + 1);
                self.start_playback(ctx, next);
            } else {
                ctx.request_repaint_after(DEBOUNCE - waited);
            }
        } else {
            self.play_dirty_since = None;
        }
    }

    // ── Actions ──────────────────────────────────────────────────────────

    fn cli_command(&self) -> String {
        let mut args: Vec<String> = vec!["ntsc-composite-simulator".into()];
        let (w, h) = self.out_size();
        match &self.source {
            Source::Bars => {
                args.extend(["colorbars".into(), "-o".into(), "colorbars.png".into()]);
                args.extend(["--width".into(), w.to_string(), "--height".into(), h.to_string()]);
            }
            Source::Image(p) => {
                args.extend(["image".into(), p.display().to_string(), "-o".into(), "output.png".into()]);
                if !self.auto_size {
                    args.extend(["--width".into(), w.to_string(), "--height".into(), h.to_string()]);
                }
            }
            Source::Video { path, .. } => {
                args.extend(["roundtrip".into(), path.display().to_string(), "-o".into(), "output.mp4".into()]);
                args.extend(["--width".into(), w.to_string(), "--height".into(), h.to_string()]);
            }
        }
        if self.comb_1h {
            args.push("--comb-1h".into());
        }
        args.extend(self.current_values().cli_args());
        args.iter().map(|a| quote(a)).collect::<Vec<_>>().join(" ")
    }

    /// Ask where to save the processed preview; the frame shown now is saved.
    fn save_preview_png(&mut self, ctx: &egui::Context) {
        let Some((rgb, w, h)) = self.after_rgb.clone() else { return };
        let task = FileDialogTask::spawn(ctx, || {
            rfd::FileDialog::new().add_filter("PNG", &["png"]).set_file_name("preview.png").save_file()
        });
        self.dialog = Some((DialogFor::SavePng(rgb, w, h), task));
    }

    fn save_png(&mut self, path: &Path, rgb: Vec<u8>, w: usize, h: usize) {
        let r = image::RgbImage::from_raw(w as u32, h as u32, rgb)
            .ok_or_else(|| "bad image".to_string())
            .and_then(|img| img.save(path).map_err(|e| e.to_string()));
        self.status = Some(match r {
            Ok(()) => (true, format!("Saved {}", path.display())),
            Err(e) => (false, format!("Cannot save {}: {}", path.display(), e)),
        });
    }

    /// Put the processed preview frame on the clipboard, at its output size.
    fn copy_preview(&mut self, ctx: &egui::Context) {
        let Some((rgb, w, h)) = &self.after_rgb else { return };
        ctx.copy_image(egui::ColorImage::from_rgb([*w, *h], rgb));
        self.status = Some((true, format!("Copied the {}×{} frame to the clipboard", w, h)));
    }

    // ── UI ───────────────────────────────────────────────────────────────

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            let no_dialog = self.dialog.is_none();
            if ui.add_enabled(no_dialog, egui::Button::new("Open…"))
                .on_hover_text("Open an image, a video (scrub or play it with the controls at the bottom) or a preset .json. You can also drop files onto the window.")
                .clicked()
            {
                self.open_dialog(ui.ctx());
            }
            if !matches!(self.source, Source::Bars) && ui.button("Colour bars").on_hover_text("Go back to the built-in test pattern").clicked() {
                self.stop_playback_for_new_source();
                self.source = Source::Bars;
                let ctx = ui.ctx().clone();
                self.set_source_image(&ctx, bars_source());
            }
            ui.label(egui::RichText::new(self.source_label()).weak());
            ui.separator();
            ui.label("View");
            ui.selectable_value(&mut self.viewer.mode, ViewMode::Split, "Split").on_hover_text("Before/after with a draggable divider");
            ui.selectable_value(&mut self.viewer.mode, ViewMode::SideBySide, "Side by side");
            ui.selectable_value(&mut self.viewer.mode, ViewMode::After, "After");
            ui.separator();
            ui.label("Zoom");
            for z in [Zoom::Fit, Zoom::X1, Zoom::X2, Zoom::X4] {
                ui.selectable_value(&mut self.viewer.zoom, z, z.label())
                    .on_hover_text("Nearest-neighbour zoom; scroll over the image to zoom, drag to pan, double-click to centre");
            }
            if let Zoom::Level(_) = self.viewer.zoom {
                ui.label(egui::RichText::new(self.viewer.zoom.label()).strong());
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("Process…").on_hover_text("Run these settings over a file or folder").clicked() {
                    self.process.open = true;
                    match &self.source {
                        Source::Image(p) => self.process.set_input(&p.clone()),
                        Source::Video { path, .. } => self.process.set_input(&path.clone()),
                        Source::Bars => {}
                    }
                }
                if ui.add_enabled(self.after_rgb.is_some() && no_dialog, egui::Button::new("Save PNG…")).on_hover_text("Save the processed preview").clicked() {
                    self.save_preview_png(ui.ctx());
                }
                if ui.add_enabled(self.after_rgb.is_some(), egui::Button::new("Copy frame"))
                    .on_hover_text("Copy the processed preview frame to the clipboard (Ctrl+C)")
                    .clicked()
                {
                    self.copy_preview(ui.ctx());
                }
                if ui.button("Copy CLI command").on_hover_text("Copy the equivalent ntsc-composite-simulator command line").clicked() {
                    let cmd = self.cli_command();
                    ui.ctx().copy_text(cmd.clone());
                    self.status = Some((true, format!("Copied: {}", cmd)));
                }
            });
        });
    }

    fn preset_row(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Preset").strong());
            let mut label = self.preset_label(&self.preset);
            if self.is_modified() {
                label.push_str(" (modified)");
            }
            let mut pick: Option<PresetChoice> = None;
            let combo = egui::ComboBox::from_id_salt("preset")
                .selected_text(label)
                .width(170.0)
                .height(600.0)
                .show_ui(ui, |ui| {
                    if ui.selectable_label(self.preset == PresetChoice::None, "None").on_hover_text("All effects off").clicked() {
                        pick = Some(PresetChoice::None);
                    }
                    let mut last_group = "";
                    for p in BUILTIN {
                        if p.group != last_group {
                            ui.separator();
                            ui.label(egui::RichText::new(p.group).small().weak());
                            last_group = p.group;
                        }
                        let selected = self.preset == PresetChoice::Builtin(p.name);
                        if ui.selectable_label(selected, p.name).on_hover_text(p.description).clicked() {
                            pick = Some(PresetChoice::Builtin(p.name));
                        }
                    }
                    if !self.custom_presets.is_empty() {
                        ui.separator();
                        ui.label(egui::RichText::new("Custom").small().weak());
                        for (i, p) in self.custom_presets.iter().enumerate() {
                            if ui.selectable_label(self.preset == PresetChoice::Custom(i), &p.name).clicked() {
                                pick = Some(PresetChoice::Custom(i));
                            }
                        }
                    }
                });
            if self.open_menu.as_deref() == Some("preset") {
                egui::Popup::open_id(ui.ctx(), combo.response.id.with("popup"));
            }
            if let Some(p) = pick {
                self.select_preset(p);
            }
        });
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Playback deck").strong()).on_hover_text(
                "Deck / capture profile, combined with the preset above (the deck's values win where both set something).",
            );
            let mut pick: Option<Option<&'static str>> = None;
            let combo = egui::ComboBox::from_id_salt("deck")
                .selected_text(self.deck.unwrap_or("None"))
                .width(150.0)
                .show_ui(ui, |ui| {
                    if ui.selectable_label(self.deck.is_none(), "None").clicked() {
                        pick = Some(None);
                    }
                    for d in presets::DECKS {
                        if ui.selectable_label(self.deck == Some(d.name), d.name).on_hover_text(d.description).clicked() {
                            pick = Some(Some(d.name));
                        }
                    }
                });
            if self.open_menu.as_deref() == Some("deck") {
                egui::Popup::open_id(ui.ctx(), combo.response.id.with("popup"));
            }
            if let Some(d) = pick {
                self.deck = d;
                let p = self.preset.clone();
                self.select_preset(p);
            }
        });
        ui.horizontal(|ui| {
            if ui.add_enabled(self.is_modified(), egui::Button::new("Reset all"))
                .on_hover_text("Back to the selected preset's values")
                .clicked()
            {
                self.editor = self.base.clone();
            }
            let no_dialog = self.dialog.is_none();
            if ui.add_enabled(no_dialog, egui::Button::new("Load…")).on_hover_text("Load a preset .json").clicked() {
                self.load_preset_dialog(ui.ctx());
            }
            if ui.add_enabled(no_dialog, egui::Button::new("Save…"))
                .on_hover_text("Save the current settings as a preset .json (usable with --preset-file)")
                .clicked()
            {
                self.save_preset_dialog(ui.ctx());
            }
        });
        if let PresetChoice::Builtin(name) = self.preset {
            if let Some(p) = presets::builtin(name) {
                ui.label(egui::RichText::new(p.description).small().weak());
            }
        }
    }

    fn group_section(&mut self, ui: &mut egui::Ui, group: Group) {
        let dvd = self.current_values().choice("format") == "dvd";
        // On the DVD path show the DVD group; otherwise the analog ones.
        let default_open = if dvd {
            matches!(group, Group::Dvd | Group::Output)
        } else {
            matches!(group, Group::Tape | Group::Colour | Group::Timing | Group::Output)
        };
        let differs = self.editor.group_differs(group, &self.base);
        let title = if differs { format!("{} •", group.label()) } else { group.label().to_string() };
        egui::CollapsingHeader::new(egui::RichText::new(title).strong())
            .id_salt((group.label(), dvd))
            .default_open(default_open)
            .show(ui, |ui| {
                if group == Group::Output {
                    self.output_controls(ui);
                }
                self.editor.ui_group(ui, group);
                ui.horizontal(|ui| {
                    if ui.add_enabled(differs, egui::Button::new("Reset group").small())
                        .on_hover_text("Back to the preset's values for this group")
                        .clicked()
                    {
                        self.editor.reset_group(group, &self.base);
                    }
                });
            });
    }

    fn output_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.auto_size, "Auto size").on_hover_text(
                "Images keep their own size; video and colour bars are 640×480. Untick to set --width/--height.",
            );
            ui.add_enabled(!self.auto_size, egui::DragValue::new(&mut self.out_w).range(16..=4096).suffix(" w"));
            ui.add_enabled(!self.auto_size, egui::DragValue::new(&mut self.out_h).range(16..=4096).suffix(" h"));
        });
        ui.horizontal(|ui| {
            ui.label("Seed").on_hover_text("Seed for every random choice. The same seed gives the same output. CLI: --seed");
            ui.add(egui::DragValue::new(&mut self.seed).speed(1.0));
            if ui.button("Reroll").on_hover_text("New random seed").clicked() {
                self.seed = rand_seed();
            }
        });
        ui.horizontal(|ui| {
            ui.label("Frame").on_hover_text(
                "Frame number for the preview. Per-frame effects change with it; with temporal coherence on, the tape state drifts slowly from frame to frame.",
            );
            ui.add(egui::DragValue::new(&mut self.frame).range(0..=1_000_000));
            if ui.small_button("◀").clicked() {
                self.frame = self.frame.saturating_sub(1);
            }
            if ui.small_button("▶").clicked() {
                self.frame += 1;
            }
        });
        ui.add_space(4.0);
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        let values = self.current_values();
        let playing = self.playback.is_some();
        let blocker = self.play_blocker();
        let mut toggle = false;
        let mut seek: Option<u64> = None;
        let current = self.frame;
        if let Source::Video { duration, fps, time, path, .. } = &mut self.source {
            if *duration > 0.0 {
                let mut t = *time;
                let r = ui.horizontal(|ui| {
                    let label = if playing { "⏸ Pause" } else { "▶ Play" };
                    let hover = match blocker {
                        Some(why) => why.to_string(),
                        None => "Play the processed video in the preview (Space)".to_string(),
                    };
                    if ui.add_enabled(blocker.is_none() || playing, egui::Button::new(label).min_size(Vec2::new(70.0, 0.0)))
                        .on_hover_text(hover)
                        .on_disabled_hover_text(blocker.unwrap_or_default())
                        .clicked()
                    {
                        toggle = true;
                    }
                    ui.checkbox(&mut self.play_loop, "Loop").on_hover_text("Start again from the beginning at the end of the video");
                    ui.checkbox(&mut self.play_every_frame, "Every frame").on_hover_text(
                        "Show every frame, slower than real time if processing cannot keep up. \
                         Off: keep to the source frame rate and leave out frames that are late.",
                    );
                    ui.label("Time");
                    ui.spacing_mut().slider_width = (ui.available_width() - 120.0).max(100.0);
                    ui.add(egui::Slider::new(&mut t, 0.0..=*duration).suffix(" s").fixed_decimals(2)).on_hover_text(
                        "Scrub through the video; the frame is grabbed when you release the slider. \
                         While playing, playback continues from there.",
                    )
                })
                .inner;
                *time = t;
                // The slider rounds the time it shows, so only a move to
                // another frame counts as a seek.
                let frame = (t * *fps).round() as u64;
                if (r.drag_stopped() || (r.changed() && !r.dragged())) && frame != current {
                    if playing {
                        seek = Some(frame);
                    } else {
                        self.frame = frame;
                        self.grab = Some(grab_frame_async(ui.ctx().clone(), path.clone(), t, values.clone()));
                        self.grab_quiet = false;
                    }
                }
            }
        }
        if toggle {
            self.toggle_play(ui.ctx());
        }
        if let Some(frame) = seek {
            self.start_playback(ui.ctx(), frame);
        }
        ui.horizontal(|ui| {
            if let Some(pb) = &self.playback {
                ui.label(format!(
                    "Playing frame {} at {:.1} fps (source {:.3}), {} dropped",
                    self.frame,
                    pb.achieved_fps(),
                    pb.request.fps,
                    pb.dropped
                ))
                .on_hover_text("Frames shown per second over the last second, and frames left out to keep to real time");
                if pb.behind() {
                    let why = if pb.request.every_frame { "" } else { ": the deck noise reduction needs every frame" };
                    ui.label(egui::RichText::new(format!("slower than real time{}", why)).weak());
                }
            } else if self.rendering() || (self.grab.is_some() && !self.grab_quiet) {
                ui.spinner();
                ui.label("Rendering…");
            } else if let Some(summary) = &self.play_summary {
                ui.label(summary);
            } else if self.shown_id > 0 {
                ui.label(format!("Rendered in {:.0} ms", self.last_ms));
            }
            let (w, h) = self.after_rgb.as_ref().map(|(_, w, h)| (*w, *h)).unwrap_or(self.out_size());
            ui.separator();
            ui.label(format!("Output {}×{}", w, h)).on_hover_text("Size of the processed frame (DVD: the stored frame size unless square output is chosen)");
            if !self.sampled.is_empty() {
                ui.separator();
                ui.label("Sampled values:").on_hover_text("What the min..max ranges resolved to for this preview (changes with the seed)");
                let text = self.sampled.join(", ");
                ui.add(egui::Label::new(egui::RichText::new(&text).monospace().small()).truncate())
                    .on_hover_text(self.sampled.join("
"));
            }
        });
        if let Some((ok, msg)) = &self.status {
            let col = if *ok { ui.visuals().weak_text_color() } else { egui::Color32::from_rgb(230, 110, 100) };
            ui.add(egui::Label::new(egui::RichText::new(msg).color(col).small()).truncate());
        }
    }

    fn handle_drops(&mut self, ctx: &egui::Context) {
        let dropped: Vec<PathBuf> = ctx.input(|i| i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).collect());
        if let Some(p) = dropped.first() {
            self.open_path(ctx, p);
        }
    }

    fn poll_grab(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.grab else { return };
        if let Ok(result) = rx.try_recv() {
            self.grab = None;
            if std::mem::take(&mut self.grab_quiet) {
                // Adopt the frames before the paused frame if the grab
                // found the same frame, without rendering it again.
                if let Ok(img) = result {
                    if img.rgb == self.src.rgb && img.w == self.src.w && img.h == self.src.h {
                        let unchanged = self.sent_key.as_ref() == Some(&self.render_key());
                        self.src = Arc::new(img);
                        self.source_gen += 1;
                        if unchanged {
                            self.sent_key = Some(self.render_key());
                        }
                    }
                }
                return;
            }
            match result {
                Ok(img) => {
                    self.set_source_image(ctx, img);
                    self.status = Some((true, self.source_label()));
                }
                Err(e) => self.status = Some((false, e)),
            }
        }
    }

    fn handle_screenshot(&mut self, ctx: &egui::Context) {
        let Some(path) = self.screenshot.clone() else { return };
        ctx.request_repaint();
        if self.screenshot_done.load(Ordering::Relaxed) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }
        if let Some((Some(seconds), started)) = self.auto_play {
            // Scripted playback: capture while playing, after `seconds`.
            if self.screenshot_sent || !started.is_some_and(|t| t.elapsed().as_secs_f64() >= seconds) {
                return;
            }
            if let (Some(dir), Some(pb)) = (&self.dump, &self.playback) {
                let _ = std::fs::create_dir_all(dir);
                let stats = format!(
                    "shown {}\ndropped {}\nachieved_fps {:.2}\nmean_fps {:.2}\nsource_fps {:.3}\nlast_frame {}\nthreads {}\nevery_frame {}\n",
                    pb.shown,
                    pb.dropped,
                    pb.achieved_fps(),
                    pb.shown as f64 / seconds,
                    pb.request.fps,
                    self.frame,
                    rayon::current_num_threads(),
                    pb.request.every_frame,
                );
                let _ = std::fs::write(dir.join("play_stats.txt"), stats);
            }
            if self.play_pause {
                // Pause, then capture the paused preview two seconds later.
                if self.playback.is_some() {
                    self.pause(ctx);
                    self.auto_paused = Some(Instant::now());
                }
                if self.auto_paused.is_none_or(|t| t.elapsed() < Duration::from_secs(2)) || self.rendering() || self.grab.is_some() {
                    return;
                }
            }
        } else if self.shown_id == 0 || self.rendering() || self.grab.is_some() || self.screenshot_sent {
            return;
        }
        if self.process.autostart && !self.process.has_finished() {
            return;
        }
        // Let a few frames settle (fonts, layout) before capturing.
        self.screenshot_wait += 1;
        if self.screenshot_wait < if self.open_menu.is_some() { 40 } else { 10 } {
            return;
        }
        self.screenshot_sent = true;
        if let (Some(dir), Some((rgb, w, h))) = (&self.dump, &self.after_rgb) {
            let _ = std::fs::create_dir_all(dir);
            if let Some(img) = image::RgbImage::from_raw(*w as u32, *h as u32, rgb.clone()) {
                let _ = img.save(dir.join("preview.png"));
            }
            let _ = std::fs::write(dir.join("cli.txt"), self.cli_command());
        }
        let done = self.screenshot_done.clone();
        let ctx2 = ctx.clone();
        ctx.request_screenshot(move |img| {
            let [w, h] = img.size;
            let rgba: Vec<u8> = img.pixels.iter().flat_map(|c| c.to_array()).collect();
            if let Some(buf) = image::RgbaImage::from_raw(w as u32, h as u32, rgba) {
                let _ = buf.save(&path);
            }
            done.store(true, Ordering::Relaxed);
            ctx2.request_repaint();
        });
    }
}

fn rand_seed() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u128(Instant::now().elapsed().as_nanos());
    h.finish() % 1_000_000
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        if self.crash_test_ui && self.shown_id > 0 {
            panic!("crash test on the UI thread");
        }
        self.handle_drops(&ctx);
        self.poll_dialog(&ctx);
        self.poll_grab(&ctx);

        // Ctrl+C copies the frame unless a text field has focus.
        let copy = !ctx.egui_wants_keyboard_input() && ctx.input(|i| i.events.iter().any(|e| matches!(e, egui::Event::Copy)));
        if copy {
            self.copy_preview(&ctx);
        }
        // Space plays and pauses unless a text field has focus.  The key is
        // consumed so a focused button does not also take it as a click.
        if !ctx.egui_wants_keyboard_input()
            && ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Space))
            && (self.playback.is_some() || self.play_blocker().is_none())
        {
            self.toggle_play(&ctx);
        }
        self.update_playback(&ctx);

        egui::Panel::top("toolbar").show(ui, |ui| {
            ui.add_space(4.0);
            self.toolbar(ui);
            ui.add_space(2.0);
        });
        egui::Panel::bottom("status").show(ui, |ui| {
            ui.add_space(2.0);
            self.status_bar(ui);
            ui.add_space(2.0);
        });
        egui::Panel::left("settings").resizable(true).default_size(370.0).size_range(300.0..=560.0).show(ui, |ui| {
            ui.add_space(6.0);
            self.preset_row(ui);
            ui.separator();
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                for group in Group::ALL {
                    self.group_section(ui, group);
                }
            });
        });
        egui::CentralPanel::default().show(ui, |ui| {
            let (w, h) = self.after_rgb.as_ref().map(|(_, w, h)| (*w, *h)).unwrap_or(self.out_size());
            let size = self.display_size(w, h);
            self.viewer.show(ui, self.before_tex.as_ref(), self.after_tex.as_ref(), size);
        });

        let values = self.current_values();
        let comb_1h = self.comb_1h;
        let size = if self.auto_size { None } else { Some((self.out_w, self.out_h)) };
        let label = self.settings_label();
        self.process.show(&ctx, || JobSettings { values, comb_1h, size, settings_label: label });

        self.update_render(&ctx);
        self.handle_screenshot(&ctx);
    }
}
