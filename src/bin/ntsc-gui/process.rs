//! "Process…" dialog: run the current settings over a file or folder on a
//! background thread, with progress and cancel.  Uses the same library
//! pipeline as the command-line tool.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use eframe::egui;
use ntsc_composite_simulator::params::ParamValues;
use ntsc_composite_simulator::pipeline::{self, Progress, VideoOptions, CANCELLED};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobKind {
    Image,
    Video,
}

/// Shared state between the dialog and the worker thread.
#[derive(Default)]
struct JobState {
    cancel: AtomicBool,
    total: AtomicUsize,
    done: AtomicU64,
    label: Mutex<String>,
    rate: Mutex<String>,
    log: Mutex<Vec<String>>,
    result: Mutex<Option<Result<(), String>>>,
}

impl Progress for JobState {
    fn log(&self, msg: &str) {
        let mut log = self.log.lock().unwrap();
        log.push(msg.to_string());
        if log.len() > 200 {
            log.remove(0);
        }
    }
    fn start(&self, total_frames: usize, label: &str) {
        self.total.store(total_frames, Ordering::Relaxed);
        self.done.store(0, Ordering::Relaxed);
        *self.label.lock().unwrap() = label.to_string();
    }
    fn advance(&self, frames: u64, rate: &str) {
        self.done.fetch_add(frames, Ordering::Relaxed);
        *self.rate.lock().unwrap() = rate.to_string();
    }
    fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

struct Job {
    state: Arc<JobState>,
    handle: Option<JoinHandle<()>>,
}

/// What the dialog needs from the main window to start a job.
pub struct JobSettings {
    pub values: ParamValues,
    pub comb_1h: bool,
    /// Explicit output size; `None` = images keep their size, video 640×480.
    pub size: Option<(u32, u32)>,
    pub settings_label: String,
}

pub struct ProcessDialog {
    pub open: bool,
    /// Start a job as soon as the dialog is shown (scripted runs).
    pub autostart: bool,
    input: String,
    output: String,
    kind: JobKind,
    crf: u32,
    lossy: bool,
    telecine: bool,
    x264_preset: String,
    job: Option<Job>,
    message: Option<(bool, String)>,
}

impl Default for ProcessDialog {
    fn default() -> Self {
        Self {
            open: false,
            autostart: false,
            input: String::new(),
            output: String::new(),
            kind: JobKind::Image,
            crf: 17,
            lossy: false,
            telecine: false,
            x264_preset: "fast".into(),
            job: None,
            message: None,
        }
    }
}

fn detect_kind(path: &Path) -> JobKind {
    if path.is_dir() {
        let videos = pipeline::iter_files(path, pipeline::VIDEO_EXTENSIONS).len();
        let images = pipeline::iter_files(path, pipeline::IMAGE_EXTENSIONS).len();
        if videos > images { JobKind::Video } else { JobKind::Image }
    } else if pipeline::is_video_path(path) {
        JobKind::Video
    } else {
        JobKind::Image
    }
}

/// Suggested output: `name_ntsc.ext` next to a file, `folder_ntsc` for a folder.
fn suggest_output(input: &Path, kind: JobKind) -> PathBuf {
    let stem = input.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "output".into());
    let parent = input.parent().unwrap_or(Path::new("."));
    if input.is_dir() {
        return parent.join(format!("{}_ntsc", stem));
    }
    let ext = match kind {
        JobKind::Image => "png".to_string(),
        JobKind::Video => "mkv".to_string(),
    };
    parent.join(format!("{}_ntsc.{}", stem, ext))
}

impl ProcessDialog {
    /// Pre-fill the input (e.g. with the file being previewed).
    pub fn set_input(&mut self, path: &Path) {
        self.input = path.display().to_string();
        self.kind = detect_kind(path);
        self.output = suggest_output(path, self.kind).display().to_string();
    }

    /// `true` once a job has finished (successfully or not).
    pub fn has_finished(&self) -> bool {
        self.message.is_some()
    }

    pub fn is_running(&self) -> bool {
        self.job.as_ref().is_some_and(|j| j.state.result.lock().unwrap().is_none())
    }

    pub fn show(&mut self, ctx: &egui::Context, settings: impl FnOnce() -> JobSettings) {
        if !self.open {
            return;
        }
        let mut open = self.open;
        let mut start: Option<JobSettings> = None;
        let mut settings = Some(settings);
        let screen = ctx.content_rect();
        egui::Window::new("Process")
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_width(620.0)
            .pivot(egui::Align2::CENTER_TOP)
            .default_pos(screen.center_top() + egui::vec2(0.0, 90.0))
            .show(ctx, |ui| {
                let running = self.is_running();
                ui.label("Runs the current settings from the left panel over a file, or over every file in a folder.");
                ui.add_space(4.0);
                ui.add_enabled_ui(!running, |ui| {
                    egui::Grid::new("process_grid").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                        ui.label("Input");
                        ui.horizontal(|ui| {
                            let r = ui.add(egui::TextEdit::singleline(&mut self.input).desired_width(400.0).hint_text("file or folder"));
                            if r.lost_focus() && !self.input.is_empty() {
                                let p = PathBuf::from(&self.input);
                                self.kind = detect_kind(&p);
                                if self.output.is_empty() {
                                    self.output = suggest_output(&p, self.kind).display().to_string();
                                }
                            }
                            if ui.button("File…").clicked() {
                                let mut exts: Vec<&str> = pipeline::IMAGE_EXTENSIONS.to_vec();
                                exts.extend_from_slice(pipeline::VIDEO_EXTENSIONS);
                                if let Some(p) = rfd::FileDialog::new().add_filter("Images and videos", &exts).pick_file() {
                                    self.set_input(&p);
                                }
                            }
                            if ui.button("Folder…").clicked() {
                                if let Some(p) = rfd::FileDialog::new().pick_folder() {
                                    self.set_input(&p);
                                }
                            }
                        });
                        ui.end_row();

                        ui.label("Output");
                        ui.horizontal(|ui| {
                            ui.add(egui::TextEdit::singleline(&mut self.output).desired_width(400.0).hint_text("file, or folder for batch input"));
                            if ui.button("Browse…").clicked() {
                                let input = PathBuf::from(&self.input);
                                let picked = if input.is_dir() {
                                    rfd::FileDialog::new().pick_folder()
                                } else {
                                    rfd::FileDialog::new().set_file_name(
                                        Path::new(&self.output).file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
                                    ).save_file()
                                };
                                if let Some(p) = picked {
                                    self.output = p.display().to_string();
                                }
                            }
                        });
                        ui.end_row();

                        ui.label("Type");
                        ui.horizontal(|ui| {
                            ui.radio_value(&mut self.kind, JobKind::Image, "Images");
                            ui.radio_value(&mut self.kind, JobKind::Video, "Video roundtrip");
                        });
                        ui.end_row();

                        if self.kind == JobKind::Video {
                            ui.label("Quality");
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut self.lossy, "Lossy (x264 4:2:0)").on_hover_text(
                                    "Off (default): lossless RGB, FFV1 in .mkv or x264 RGB QP 0 in .mp4, so nothing                                      the simulation produces is lost. On: smaller x264 4:2:0 files at the CRF below,                                      which halves colour resolution and smooths noise.",
                                );
                                ui.add_enabled(self.lossy, egui::Slider::new(&mut self.crf, 0..=51).text("CRF"))
                                    .on_hover_text("x264 CRF: 17 = visually lossless, 51 = worst");
                            });
                            ui.end_row();
                            ui.label("Encoder");
                            ui.horizontal(|ui| {
                                egui::ComboBox::from_id_salt("x264_preset")
                                    .selected_text(&self.x264_preset)
                                    .show_ui(ui, |ui| {
                                        for p in ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower"] {
                                            ui.selectable_value(&mut self.x264_preset, p.to_string(), p);
                                        }
                                    })
                                    .response
                                    .on_hover_text("x264 preset (speed vs size)");
                                ui.checkbox(&mut self.telecine, "Telecine (3:2 pulldown)")
                                    .on_hover_text("24p film to 29.97 fps interlaced");
                            });
                            ui.end_row();
                        }
                    });
                });

                ui.separator();
                let s = settings.take().map(|f| f());
                if let Some(s) = &s {
                    ui.weak(format!("Settings: {}", s.settings_label));
                }

                if let Some(job) = &self.job {
                    let st = &job.state;
                    let total = st.total.load(Ordering::Relaxed);
                    let done = st.done.load(Ordering::Relaxed);
                    let label = st.label.lock().unwrap().clone();
                    let rate = st.rate.lock().unwrap().clone();
                    if running {
                        let bar = if total > 0 {
                            egui::ProgressBar::new((done as f32 / total as f32).min(1.0))
                                .text(format!("{} {}/{} {}", label, done, total, rate))
                        } else {
                            egui::ProgressBar::new(0.0).animate(true).text(format!("{} {} {}", label, done, rate))
                        };
                        ui.add(bar);
                        ctx.request_repaint_after(std::time::Duration::from_millis(100));
                    }
                    let log = st.log.lock().unwrap();
                    egui::ScrollArea::vertical().max_height(120.0).stick_to_bottom(true).show(ui, |ui| {
                        for line in log.iter() {
                            ui.monospace(line);
                        }
                    });
                }

                if let Some((ok, msg)) = &self.message {
                    let col = if *ok { egui::Color32::from_rgb(120, 200, 120) } else { egui::Color32::from_rgb(230, 110, 100) };
                    ui.colored_label(col, msg);
                }

                ui.horizontal(|ui| {
                    if running {
                        if ui.button("Cancel").clicked() {
                            if let Some(job) = &self.job {
                                job.state.cancel.store(true, Ordering::Relaxed);
                            }
                        }
                    } else {
                        let ready = !self.input.trim().is_empty() && !self.output.trim().is_empty();
                        let auto = self.autostart && self.job.is_none() && ready;
                        if ui.add_enabled(ready, egui::Button::new("Start")).clicked() || auto {
                            start = s;
                        }
                    }
                });
            });
        self.open = open;

        if let Some(s) = start {
            self.start(ctx.clone(), s);
        }
        self.collect_result();
    }

    fn start(&mut self, ctx: egui::Context, s: JobSettings) {
        let state = Arc::new(JobState::default());
        let (input, output, kind) = (PathBuf::from(self.input.trim()), PathBuf::from(self.output.trim()), self.kind);
        let opts = VideoOptions {
            width: s.size.map_or(640, |sz| sz.0),
            height: s.size.map_or(480, |sz| sz.1),
            comb_1h: s.comb_1h,
            crf: self.crf,
            preset: self.x264_preset.clone(),
            lossy: self.lossy,
            raw_copy: None,
            threads: None,
            telecine: self.telecine,
        };
        let st = state.clone();
        let handle = std::thread::spawn(move || {
            let r = crate::crash::catch(|| match kind {
                JobKind::Image => pipeline::process_images(
                    &input, &output, s.size.map(|z| z.0), s.size.map(|z| z.1), s.comb_1h, &s.values, &*st,
                ),
                JobKind::Video => pipeline::process_videos(&input, &output, &opts, &s.values, &*st),
            })
            .and_then(|r| r.map_err(|e| format!("{:#}", e)));
            *st.result.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(r);
            ctx.request_repaint();
        });
        self.message = None;
        self.job = Some(Job { state, handle: Some(handle) });
    }

    fn collect_result(&mut self) {
        let Some(job) = &mut self.job else { return };
        let Some(result) = job.state.result.lock().unwrap().clone() else { return };
        if let Some(h) = job.handle.take() {
            let _ = h.join();
            self.message = Some(match result {
                Ok(()) => (true, format!("Done: {}", self.output)),
                Err(e) if e == CANCELLED => (false, "Cancelled.".to_string()),
                Err(e) => (false, format!("Failed: {}", e)),
            });
        }
    }
}
