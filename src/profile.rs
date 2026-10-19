//! Optional per-stage timing (`roundtrip --profile`).
//!
//! When enabled, [`scope`] guards add the time between their creation and
//! drop to a named stage.  Times from all worker threads are summed, so a
//! stage's mean per frame is CPU time, not wall time.  When disabled a
//! guard costs one relaxed atomic load.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

static ENABLED: AtomicBool = AtomicBool::new(false);
/// (stage, total time, calls) in the order stages were first seen.
static STAGES: Mutex<Vec<(&'static str, Duration, u64)>> = Mutex::new(Vec::new());

/// Turn timing on or off and clear the collected times.
pub fn enable(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
    STAGES.lock().unwrap().clear();
}

pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Times the enclosing block as stage `name` until dropped.
pub struct Scope {
    name: &'static str,
    start: Option<Instant>,
}

impl Drop for Scope {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            let dt = start.elapsed();
            let mut stages = STAGES.lock().unwrap();
            match stages.iter_mut().find(|e| e.0 == self.name) {
                Some(e) => {
                    e.1 += dt;
                    e.2 += 1;
                }
                None => stages.push((self.name, dt, 1)),
            }
        }
    }
}

/// Start timing stage `name`.
pub fn scope(name: &'static str) -> Scope {
    Scope { name, start: enabled().then(Instant::now) }
}

/// The collected stages as (name, total time, calls), in first-seen order.
pub fn stages() -> Vec<(&'static str, Duration, u64)> {
    STAGES.lock().unwrap().clone()
}

/// A table of mean milliseconds per frame for each stage.
pub fn report(frames: u64) -> String {
    let frames = frames.max(1) as f64;
    let mut out = format!("{:<28} {:>10} {:>8}\n", "stage", "ms/frame", "calls");
    let mut total = 0.0;
    for (name, d, n) in stages() {
        let ms = d.as_secs_f64() * 1000.0 / frames;
        total += ms;
        out += &format!("{:<28} {:>10.3} {:>8}\n", name, ms, n);
    }
    out += &format!("{:<28} {:>10.3}\n", "sum", total);
    out
}
