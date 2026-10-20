//! Native file dialogs on their own thread.
//!
//! A modal dialog shown from inside the UI callback blocks winit's event
//! loop for as long as it is open, and winit queues every event that
//! arrives meanwhile (repaint requests from worker threads, accessibility
//! requests) in a buffer without a limit.  Shown from a thread, the dialog
//! leaves the event loop running; the UI polls for the chosen path.

use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, TryRecvError};

use eframe::egui;

pub struct FileDialogTask {
    rx: Receiver<Option<PathBuf>>,
}

impl FileDialogTask {
    /// Run `show` (a blocking `rfd::FileDialog` call) on a new thread.
    pub fn spawn(ctx: &egui::Context, show: impl FnOnce() -> Option<PathBuf> + Send + 'static) -> Self {
        let (tx, rx) = channel();
        let ctx = ctx.clone();
        // If the thread cannot start, the sender is dropped and `poll`
        // reports a cancelled dialog.
        let _ = std::thread::Builder::new().name("file-dialog".into()).spawn(move || {
            let _ = tx.send(crate::crash::catch(show).ok().flatten());
            ctx.request_repaint();
        });
        Self { rx }
    }

    /// `Some(choice)` once the dialog has closed (`choice` is `None` when
    /// it was cancelled), `None` while it is open.
    pub fn poll(&self) -> Option<Option<PathBuf>> {
        match self.rx.try_recv() {
            Ok(choice) => Some(choice),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => Some(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn result_arrives_from_the_thread() {
        let ctx = egui::Context::default();
        let task = FileDialogTask::spawn(&ctx, || Some(PathBuf::from("picked.png")));
        let t0 = std::time::Instant::now();
        let got = loop {
            if let Some(r) = task.poll() {
                break r;
            }
            assert!(t0.elapsed().as_secs() < 5, "no result");
            std::thread::yield_now();
        };
        assert_eq!(got, Some(PathBuf::from("picked.png")));
        let cancelled = FileDialogTask::spawn(&ctx, || None);
        while cancelled.poll().is_none() {
            std::thread::yield_now();
        }
    }
}
