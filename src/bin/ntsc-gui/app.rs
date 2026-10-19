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

use crate::params_ui::ParamEditor;
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
    Video { path: PathBuf, duration: f64, fps: f64, time: f64 },
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
    screenshot_wait: u32,
    screenshot_done: Arc<AtomicBool>,
    screenshot_sent: bool,
}

fn bars_source() -> SourceImage {
    SourceImage { rgb: colorbars::generate_colorbars(640, 480), w: 640, h: 480 }
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
            renderer: Renderer::new(cc.egui_ctx.clone()),
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
            screenshot_wait: 0,
            screenshot_done: Arc::new(AtomicBool::new(false)),
            screenshot_sent: false,
        };
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
        let values = self.preset_values(&choice);
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
            "preset {}{}, seed {}",
            self.preset_label(&self.preset),
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

    fn save_preset_dialog(&mut self) {
        let default_name = match &self.preset {
            PresetChoice::None => "custom".to_string(),
            c => format!("{}-custom", self.preset_label(c)),
        };
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Preset (JSON)", &["json"])
            .set_file_name(format!("{}.json", default_name))
            .save_file()
        else {
            return;
        };
        let name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or(default_name);
        let preset = Preset { name: name.clone(), values: self.current_values() };
        match presets::save_preset_file(&path, &preset) {
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
        let mut exts: Vec<&str> = pipeline::IMAGE_EXTENSIONS.to_vec();
        exts.extend_from_slice(pipeline::VIDEO_EXTENSIONS);
        if let Some(p) = rfd::FileDialog::new()
            .add_filter("Images and videos", &exts)
            .add_filter("Presets (JSON)", &["json"])
            .pick_file()
        {
            self.open_path(ctx, &p);
        }
    }

    fn open_path(&mut self, ctx: &egui::Context, path: &Path) {
        if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("json")) {
            self.load_preset_file(path);
        } else if pipeline::is_video_path(path) {
            let duration = pipeline::video_duration(path).unwrap_or(0.0);
            let fps = pipeline::ffprobe_video(path).map(|v| v.2).unwrap_or(29.97);
            let time = (duration * 0.1).min(10.0);
            self.source = Source::Video { path: path.to_path_buf(), duration, fps, time };
            self.grab = Some(grab_frame_async(ctx.clone(), path.to_path_buf(), time, self.current_values()));
            self.frame = (time * fps).round() as u64;
            self.status = Some((true, format!("Loading frame from {}…", path.display())));
        } else {
            match image::open(path) {
                Ok(img) => {
                    let rgb = img.to_rgb8();
                    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
                    self.source = Source::Image(path.to_path_buf());
                    self.set_source_image(ctx, SourceImage { rgb: rgb.into_raw(), w, h });
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
        if let Some(result) = self.renderer.poll() {
            if result.id > self.shown_id {
                self.shown_id = result.id;
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

    fn rendering(&self) -> bool {
        self.shown_id + 1 < self.next_id || self.dirty_since.is_some()
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

    fn save_preview_png(&mut self) {
        let Some((rgb, w, h)) = &self.after_rgb else { return };
        let Some(path) = rfd::FileDialog::new().add_filter("PNG", &["png"]).set_file_name("preview.png").save_file() else {
            return;
        };
        let r = image::RgbImage::from_raw(*w as u32, *h as u32, rgb.clone())
            .ok_or_else(|| "bad image".to_string())
            .and_then(|img| img.save(&path).map_err(|e| e.to_string()));
        self.status = Some(match r {
            Ok(()) => (true, format!("Saved {}", path.display())),
            Err(e) => (false, format!("Cannot save {}: {}", path.display(), e)),
        });
    }

    // ── UI ───────────────────────────────────────────────────────────────

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            if ui.button("Open…").on_hover_text("Open an image, a video (one frame is previewed) or a preset .json. You can also drop files onto the window.").clicked() {
                self.open_dialog(ui.ctx());
            }
            if !matches!(self.source, Source::Bars) && ui.button("Colour bars").on_hover_text("Go back to the built-in test pattern").clicked() {
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
                    .on_hover_text("Nearest-neighbour zoom; drag the image to pan, double-click to centre");
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
                if ui.add_enabled(self.after_rgb.is_some(), egui::Button::new("Save PNG…")).on_hover_text("Save the processed preview").clicked() {
                    self.save_preview_png();
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
            egui::ComboBox::from_id_salt("preset")
                .selected_text(label)
                .width(170.0)
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
            if let Some(p) = pick {
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
            if ui.button("Load…").on_hover_text("Load a preset .json").clicked() {
                if let Some(p) = rfd::FileDialog::new().add_filter("Preset (JSON)", &["json"]).pick_file() {
                    self.load_preset_file(&p);
                }
            }
            if ui.button("Save…").on_hover_text("Save the current settings as a preset .json (usable with --preset-file)").clicked() {
                self.save_preset_dialog();
            }
        });
        if let PresetChoice::Builtin(name) = self.preset {
            if let Some(p) = presets::builtin(name) {
                ui.label(egui::RichText::new(p.description).small().weak());
            }
        }
    }

    fn group_section(&mut self, ui: &mut egui::Ui, group: Group) {
        let default_open = matches!(group, Group::Tape | Group::Colour | Group::Timing | Group::Output);
        let differs = self.editor.group_differs(group, &self.base);
        let title = if differs { format!("{} •", group.label()) } else { group.label().to_string() };
        egui::CollapsingHeader::new(egui::RichText::new(title).strong())
            .id_salt(group.label())
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
        ui.checkbox(&mut self.comb_1h, "1H comb filter (decoder)")
            .on_hover_text("Line-delay comb instead of the 2-sample comb: less rainbow, but hanging dots on colour edges. CLI: --comb-1h");
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
        if let Source::Video { duration, fps, time, path } = &mut self.source {
            if *duration > 0.0 {
                let mut t = *time;
                let r = ui.horizontal(|ui| {
                    ui.label("Time");
                    ui.spacing_mut().slider_width = (ui.available_width() - 120.0).max(100.0);
                    ui.add(egui::Slider::new(&mut t, 0.0..=*duration).suffix(" s").fixed_decimals(2))
                        .on_hover_text("Scrub through the video; the frame is grabbed when you release the slider")
                })
                .inner;
                *time = t;
                if r.drag_stopped() || (r.changed() && !r.dragged()) {
                    self.frame = (t * *fps).round() as u64;
                    self.grab = Some(grab_frame_async(ui.ctx().clone(), path.clone(), t, values.clone()));
                }
            }
        }
        ui.horizontal(|ui| {
            if self.rendering() || self.grab.is_some() {
                ui.spinner();
                ui.label("Rendering…");
            } else if self.shown_id > 0 {
                ui.label(format!("Rendered in {:.0} ms", self.last_ms));
            }
            let (w, h) = self.out_size();
            ui.separator();
            ui.label(format!("Output {}×{}", w, h));
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
        if self.shown_id == 0 || self.rendering() || self.grab.is_some() || self.screenshot_sent {
            return;
        }
        if self.process.autostart && !self.process.has_finished() {
            return;
        }
        // Let a few frames settle (fonts, layout) before capturing.
        self.screenshot_wait += 1;
        if self.screenshot_wait < 10 {
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
        self.handle_drops(&ctx);
        self.poll_grab(&ctx);

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
            self.viewer.show(ui, self.before_tex.as_ref(), self.after_tex.as_ref(), Vec2::new(w as f32, h as f32));
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
