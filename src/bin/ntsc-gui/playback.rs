//! Video playback in the preview: an ffmpeg reader and a processing thread
//! run the source through the same `FrameProcessor` as the `roundtrip`
//! command, from the current frame on, and hand finished frames to the UI
//! through a small bounded queue.  The UI shows each frame when it is due
//! at the source frame rate.
//!
//! In real-time mode a frame that is already late is not shown when a
//! newer one is due, and when no stage needs every frame the processing
//! thread skips ahead to frames it can finish in time.  The deck noise
//! reduction needs every frame, so with it on every frame is processed
//! and playback falls behind instead.  In every-frame mode each frame is
//! shown, slower than real time if processing cannot keep up.

use std::io::Read;
use std::path::PathBuf;
use std::process::Child;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use eframe::egui;
use ntsc_composite_simulator::params::ParamValues;
use ntsc_composite_simulator::pipeline::{self, FrameProcessor};

use crate::crash;

/// Frames processed before the first shown one when the deck noise
/// reduction is on, so its recursive state builds up (as in the still
/// preview).
const DNR_WARMUP: u64 = 8;

/// How late the newest frame may be before the clock is moved instead of
/// dropping frames, when frames cannot be skipped.
const MAX_LAG: Duration = Duration::from_millis(250);

/// What to play.
#[derive(Clone)]
pub struct PlayRequest {
    pub path: PathBuf,
    pub first_frame: u64,
    pub fps: f64,
    pub in_w: usize,
    pub in_h: usize,
    pub out_w: usize,
    pub out_h: usize,
    pub comb_1h: bool,
    pub values: ParamValues,
    /// Show every frame (slower than real time if need be) instead of
    /// keeping to the source frame rate.
    pub every_frame: bool,
}

/// One processed frame with its source.
pub struct PlayFrame {
    pub frame: u64,
    pub src: Vec<u8>,
    pub rgb: Vec<u8>,
}

enum Msg {
    Frame(PlayFrame),
    /// The processing thread skipped this many frames to keep up.
    Skipped(u64),
    End,
    Error(String),
}

/// The presentation clock: frame `.1` is due at instant `.0`.
type Clock = Arc<Mutex<Option<(Instant, u64)>>>;

/// What `Playback::poll` found.
#[derive(Default)]
pub struct Poll {
    /// The frame to show now.
    pub frame: Option<PlayFrame>,
    pub ended: bool,
    pub error: Option<String>,
    /// When to poll again for the next due frame.
    pub wake_in: Option<Duration>,
}

pub struct Playback {
    pub request: PlayRequest,
    stop: Arc<AtomicBool>,
    child: Arc<Mutex<Option<Child>>>,
    rx: Option<Receiver<Msg>>,
    threads: Vec<JoinHandle<()>>,
    clock: Clock,
    /// A received frame that is not due yet.
    held: Option<PlayFrame>,
    /// Frames shown in the last second, for the achieved rate.
    shown_at: std::collections::VecDeque<Instant>,
    pub shown: u64,
    pub dropped: u64,
    /// The deck noise reduction is on, so no frame is skipped.
    pub needs_every_frame: bool,
    /// Until when to report playback as slower than real time (set when
    /// the clock had to wait for a late frame).
    behind_until: Option<Instant>,
}

impl Playback {
    pub fn start(ctx: egui::Context, request: PlayRequest) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let child: Arc<Mutex<Option<Child>>> = Arc::new(Mutex::new(None));
        let clock: Clock = Arc::new(Mutex::new(None));
        let batch = rayon::current_num_threads().max(1);
        let (tx, rx) = sync_channel::<Msg>(batch);
        let needs_every_frame = pipeline::resolve_effects(&request.values).deck_dnr.is_some_and(|s| s > 0.0);

        let mut threads = Vec::new();
        let (st, ch, cl, req) = (stop.clone(), child.clone(), clock.clone(), request.clone());
        let spawned = std::thread::Builder::new()
            .name("playback".into())
            .spawn(move || {
                // A panic (logged by the crash hook) ends playback with an
                // error instead of taking the program down.
                let result = crash::catch(|| run(&ctx, &req, &st, &ch, &cl, &tx, batch))
                    .and_then(|r| r.map_err(|e| format!("{:#}", e)));
                if let Err(e) = result {
                    if !st.load(Ordering::Relaxed) {
                        let _ = tx.send(Msg::Error(e));
                    }
                }
                ctx.request_repaint();
            });
        match spawned {
            Ok(h) => threads.push(h),
            Err(e) => {
                let (etx, erx) = sync_channel(1);
                let _ = etx.send(Msg::Error(format!("cannot start the playback thread: {}", e)));
                return Self::with(request, stop, child, erx, threads, clock, needs_every_frame);
            }
        }
        Self::with(request, stop, child, rx, threads, clock, needs_every_frame)
    }

    fn with(
        request: PlayRequest,
        stop: Arc<AtomicBool>,
        child: Arc<Mutex<Option<Child>>>,
        rx: Receiver<Msg>,
        threads: Vec<JoinHandle<()>>,
        clock: Clock,
        needs_every_frame: bool,
    ) -> Self {
        Self {
            request,
            stop,
            child,
            rx: Some(rx),
            threads,
            clock,
            held: None,
            shown_at: Default::default(),
            shown: 0,
            dropped: 0,
            needs_every_frame,
            behind_until: None,
        }
    }

    /// `true` while playback runs slower than real time.
    pub fn behind(&self) -> bool {
        self.behind_until.is_some_and(|t| Instant::now() < t)
    }

    /// Frames shown per second over the last second.
    pub fn achieved_fps(&self) -> f64 {
        let (Some(first), Some(last)) = (self.shown_at.front(), self.shown_at.back()) else { return 0.0 };
        let span = last.duration_since(*first).as_secs_f64();
        if self.shown_at.len() < 2 || span <= 0.0 { 0.0 } else { (self.shown_at.len() - 1) as f64 / span }
    }

    /// Take the frame to show now, if one is due.
    pub fn poll(&mut self) -> Poll {
        let mut out = Poll::default();
        let Some(rx) = &self.rx else { return out };
        let fps = self.request.fps.max(1.0);
        let now = Instant::now();
        loop {
            let msg = match self.held.take() {
                Some(f) => Msg::Frame(f),
                None => match rx.try_recv() {
                    Ok(m) => m,
                    Err(_) => break,
                },
            };
            let f = match msg {
                Msg::Frame(f) => f,
                Msg::Skipped(n) => {
                    self.dropped += n;
                    continue;
                }
                Msg::End => {
                    out.ended = true;
                    break;
                }
                Msg::Error(e) => {
                    out.error = Some(e);
                    break;
                }
            };
            let mut clock = self.clock.lock().unwrap_or_else(PoisonError::into_inner);
            let (t0, f0) = *clock.get_or_insert((now, f.frame));
            let due = t0 + Duration::from_secs_f64(f.frame.saturating_sub(f0) as f64 / fps);
            if due > now {
                out.wake_in = Some(due - now);
                self.held = Some(f);
                break;
            }
            let late = now - due;
            let can_drop = !self.request.every_frame && !self.needs_every_frame;
            let max_lag = if self.request.every_frame { Duration::from_secs_f64(1.5 / fps) } else { MAX_LAG };
            if !can_drop && late > max_lag {
                // Behind: show this frame now and pace the following ones
                // from here.
                *clock = Some((now, f.frame));
                self.behind_until = Some(now + Duration::from_secs(1));
            }
            drop(clock);
            if out.frame.replace(f).is_some() {
                self.dropped += 1;
            }
            if self.request.every_frame {
                // One frame per poll; the next one is due later.
                out.wake_in = Some(Duration::from_secs_f64(0.5 / fps));
                break;
            }
        }
        if out.frame.is_some() {
            self.shown += 1;
            self.shown_at.push_back(now);
            while self.shown_at.front().is_some_and(|t| now.duration_since(*t) > Duration::from_secs(1)) {
                self.shown_at.pop_front();
            }
        }
        out
    }

    /// Stop ffmpeg and let the threads end.  `join` waits for them.
    pub fn stop(&mut self, join: bool) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(mut c) = self.child.lock().unwrap_or_else(PoisonError::into_inner).take() {
            let _ = c.kill();
            let _ = c.wait();
        }
        // A thread blocked on the full queue gets an error and ends.
        self.rx = None;
        self.held = None;
        let threads = std::mem::take(&mut self.threads);
        if join {
            for t in threads {
                let _ = t.join();
            }
        }
    }
}

impl Drop for Playback {
    fn drop(&mut self) {
        self.stop(true);
    }
}

/// Read one frame of `n` bytes; `None` at the end of the stream.
fn read_frame(r: &mut impl Read, n: usize) -> Option<Vec<u8>> {
    let mut buf = vec![0u8; n];
    r.read_exact(&mut buf).ok().map(|_| buf)
}

/// The processing thread: reads frames from `first_frame` (minus any
/// warm-up frames) and sends the processed ones to the UI in order.
fn run(
    ctx: &egui::Context,
    req: &PlayRequest,
    stop: &AtomicBool,
    child: &Mutex<Option<Child>>,
    clock: &Clock,
    tx: &SyncSender<Msg>,
    batch: usize,
) -> anyhow::Result<()> {
    let mut processor =
        FrameProcessor::new(&req.values, req.comb_1h, (req.in_w, req.in_h), (req.out_w, req.out_h), req.first_frame as u32);
    let warmup = if processor.needs_every_frame() {
        DNR_WARMUP
    } else if processor.uses_previous_frames() {
        2
    } else {
        0
    }
    .min(req.first_frame);
    let read_from = req.first_frame - warmup;

    let mut c = pipeline::spawn_frame_reader(&req.path, &req.values, read_from, req.fps)?;
    let stdout = c.stdout.take().expect("piped stdout");
    {
        let mut slot = child.lock().unwrap_or_else(PoisonError::into_inner);
        if stop.load(Ordering::Relaxed) {
            let _ = c.kill();
            let _ = c.wait();
            return Ok(());
        }
        *slot = Some(c);
    }

    // The reader thread keeps up to one batch of source frames ready
    // while a batch is processed.
    let frame_bytes = req.in_w * req.in_h * 3;
    let (src_tx, src_rx) = sync_channel::<Vec<u8>>(batch);
    let reader = std::thread::Builder::new().name("playback-read".into()).spawn(move || {
        let mut r = std::io::BufReader::with_capacity(1 << 20, stdout);
        while let Some(f) = read_frame(&mut r, frame_bytes) {
            if src_tx.send(f).is_err() {
                break;
            }
        }
    })?;
    let result = process(ctx, req, stop, clock, tx, batch, &mut processor, warmup, &src_rx);
    drop(src_rx);
    if let Some(mut c) = child.lock().unwrap_or_else(PoisonError::into_inner).take() {
        let _ = c.kill();
        let _ = c.wait();
    }
    let _ = reader.join();
    result
}

#[allow(clippy::too_many_arguments)]
fn process(
    ctx: &egui::Context,
    req: &PlayRequest,
    stop: &AtomicBool,
    clock: &Clock,
    tx: &SyncSender<Msg>,
    batch: usize,
    processor: &mut FrameProcessor,
    warmup: u64,
    src_rx: &Receiver<Vec<u8>>,
) -> anyhow::Result<()> {
    // Warm-up frames: run through the noise reduction, or only kept as the
    // 3D comb's frames before the first one.
    let mut before: Vec<Vec<u8>> = Vec::new();
    for _ in 0..warmup {
        match src_rx.recv() {
            Ok(f) => before.push(f),
            Err(_) => break,
        }
    }
    if processor.needs_every_frame() && !before.is_empty() {
        let mut warm = FrameProcessor::new(
            &req.values, req.comb_1h, (req.in_w, req.in_h), (req.out_w, req.out_h),
            (req.first_frame - before.len() as u64) as u32,
        );
        warm.process_batch(&before);
        // The warmed-up processor carries on from the first shown frame.
        *processor = warm;
    } else {
        processor.skip_to(req.first_frame as u32, &before);
    }

    let fps = req.fps.max(1.0);
    let can_skip = !req.every_frame && !processor.needs_every_frame();
    // Time one batch takes, for skipping ahead.
    let mut batch_time = Duration::ZERO;
    let mut frames: Vec<Vec<u8>> = Vec::with_capacity(batch);
    loop {
        if stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        // Real time: when this batch would finish after its first frame is
        // due, continue at the first frame that can still be on time.
        let anchor = *clock.lock().unwrap_or_else(PoisonError::into_inner);
        if let (true, Some((t0, f0))) = (can_skip, anchor) {
            let ready_at = Instant::now() + batch_time;
            let target = f0 + (ready_at.saturating_duration_since(t0).as_secs_f64() * fps).ceil() as u64;
            let next = processor.frame_num() as u64;
            if target > next {
                let mut kept: Vec<Vec<u8>> = Vec::new();
                let mut skipped = 0;
                for _ in next..target {
                    match src_rx.recv() {
                        Ok(f) => {
                            kept.push(f);
                            if kept.len() > 2 {
                                kept.remove(0);
                            }
                            skipped += 1;
                        }
                        Err(_) => break,
                    }
                }
                processor.skip_to((next + skipped) as u32, &kept);
                if skipped > 0 && tx.send(Msg::Skipped(skipped)).is_err() {
                    return Ok(());
                }
            }
        }

        // A batch of consecutive frames: wait for the first, then take what
        // the reader has ready (up to the batch size).
        frames.clear();
        let mut end = false;
        loop {
            let got = if frames.is_empty() {
                src_rx.recv_timeout(Duration::from_millis(200))
            } else if frames.len() < batch {
                src_rx.recv_timeout(Duration::from_millis(50))
            } else {
                break;
            };
            match got {
                Ok(f) => frames.push(f),
                Err(RecvTimeoutError::Timeout) if frames.is_empty() => {
                    if stop.load(Ordering::Relaxed) {
                        return Ok(());
                    }
                }
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => {
                    end = true;
                    break;
                }
            }
        }
        if !frames.is_empty() {
            let t = Instant::now();
            let first = processor.frame_num() as u64;
            let results = processor.process_batch(&frames);
            batch_time = t.elapsed();
            for (k, (src, rgb)) in frames.drain(..).zip(results).enumerate() {
                if tx.send(Msg::Frame(PlayFrame { frame: first + k as u64, src, rgb })).is_err() {
                    return Ok(());
                }
                ctx.request_repaint();
            }
        }
        if end {
            let _ = tx.send(Msg::End);
            ctx.request_repaint();
            return Ok(());
        }
    }
}
