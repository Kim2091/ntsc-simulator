//! NTSC composite video simulator: encode RGB frames to a simulated NTSC
//! composite signal, degrade it (reception and VHS tape effects), and
//! decode it back.
//!
//! The command-line tool (`ntsc-composite-simulator`) and the GUI
//! (`ntsc-gui`) are thin front ends over this library:
//!
//! - [`params`] — the shared parameter model (every effect flag, its range,
//!   default and help text) and the resolution of ranges into effects;
//! - [`presets`] — built-in presets;
//! - [`dvd`] / [`mpeg2`] — the DVD-Video path (mastering, MPEG-2 within
//!   the DVD limits via ffmpeg, player decode and output);
//! - [`pipeline`] — image, colour-bar and video processing with progress
//!   reporting and cancellation.

pub mod colorbars;
pub mod constants;
pub mod decoder;
pub mod dvd;
pub mod effects;
pub mod encoder;
pub mod filters;
pub mod mpeg2;
pub mod params;
pub mod pipeline;
pub mod presets;
