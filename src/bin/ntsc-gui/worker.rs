//! Background preview rendering and video frame grabbing.
//!
//! The UI sends render requests; the worker always renders the newest one
//! and drops any that were superseded while it was busy.

use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::Instant;

use eframe::egui;
use ntsc_composite_simulator::params::ParamValues;
use ntsc_composite_simulator::pipeline;

/// An RGB image used as the preview source.
pub struct SourceImage {
    pub rgb: Vec<u8>,
    pub w: usize,
    pub h: usize,
}

pub struct RenderRequest {
    pub id: u64,
    pub src: Arc<SourceImage>,
    pub out_w: usize,
    pub out_h: usize,
    pub comb_1h: bool,
    pub values: ParamValues,
    pub frame: u64,
}

pub struct RenderResult {
    pub id: u64,
    pub rgb: Vec<u8>,
    pub w: usize,
    pub h: usize,
    pub ms: f64,
    /// What the ranges resolved to for this render.
    pub sampled: Vec<String>,
}

pub struct Renderer {
    tx: Sender<RenderRequest>,
    rx: Receiver<RenderResult>,
}

impl Renderer {
    pub fn new(ctx: egui::Context) -> Self {
        let (tx, req_rx) = channel::<RenderRequest>();
        let (res_tx, rx) = channel::<RenderResult>();
        std::thread::Builder::new()
            .name("preview-render".into())
            .spawn(move || {
                while let Ok(mut req) = req_rx.recv() {
                    // Latest wins: skip requests that are already outdated.
                    while let Ok(newer) = req_rx.try_recv() {
                        req = newer;
                    }
                    let t0 = Instant::now();
                    let fx = pipeline::resolve_effects(&req.values);
                    let rgb = pipeline::render_frame(
                        &req.src.rgb, req.src.w, req.src.h, req.out_w, req.out_h,
                        req.comb_1h, &req.values, &fx, req.frame,
                    );
                    let sampled = req.values.sampled_summary(&fx);
                    let result = RenderResult {
                        id: req.id,
                        rgb,
                        w: req.out_w,
                        h: req.out_h,
                        ms: t0.elapsed().as_secs_f64() * 1000.0,
                        sampled,
                    };
                    if res_tx.send(result).is_err() {
                        break;
                    }
                    ctx.request_repaint();
                }
            })
            .expect("spawn render thread");
        Self { tx, rx }
    }

    pub fn submit(&self, req: RenderRequest) {
        let _ = self.tx.send(req);
    }

    /// The newest finished render, if any.
    pub fn poll(&self) -> Option<RenderResult> {
        let mut last = None;
        while let Ok(r) = self.rx.try_recv() {
            last = Some(r);
        }
        last
    }
}

/// Grab a video frame on a background thread.
pub fn grab_frame_async(
    ctx: egui::Context,
    path: PathBuf,
    time_s: f64,
) -> Receiver<Result<SourceImage, String>> {
    let (tx, rx) = channel();
    std::thread::spawn(move || {
        let r = pipeline::grab_video_frame(&path, time_s)
            .map(|(rgb, w, h)| SourceImage { rgb, w, h })
            .map_err(|e| format!("{:#}", e));
        let _ = tx.send(r);
        ctx.request_repaint();
    });
    rx
}
