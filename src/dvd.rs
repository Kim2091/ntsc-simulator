//! DVD-Video (NTSC) path: mastering → MPEG-2 encode within the DVD-Video
//! limits → player decode → player output.
//!
//! The MPEG-2 encoding and decoding are done by ffmpeg (run as a separate
//! program, with `mpeg2video`); everything around it is done here:
//!   - mastering: fitting the source into the 720×480 (or 704/352)
//!     frame with the correct non-square sample aspect, optional
//!     pre-filter, spatial noise reduction and edge enhancement;
//!   - conversion to BT.601 studio-range Y'CbCr with 4:2:0 chroma
//!     subsampling that is field-based for interlaced pictures;
//!   - soft pulldown flags for 23.976 fps film titles (see `mpeg2`);
//!   - the player's chroma upsampling (optionally with the "chroma
//!     upsampling error" of players that treat interlaced 4:2:0 as
//!     progressive) and the conversion back to RGB.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{bail, Context, Result};
use rand::Rng;

use crate::mpeg2;
use crate::params::{ParamValues, RangeParam};

/// Where the decoded DVD picture goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DvdOutput {
    /// The decoded frames as they are (like a rip).
    Digital,
    /// The player's S-Video output, captured through our NTSC decoder.
    Svideo,
    /// The player's composite output, captured through our NTSC decoder.
    Composite,
}

/// Resolved DVD settings for one file.
#[derive(Clone, Debug)]
pub struct DvdConfig {
    /// Coded frame size.
    pub width: usize,
    pub height: usize,
    /// Width of the picture area that maps to the 4:3 or 16:9 display
    /// (704 for 720/704-wide frames, 352 for 352-wide ones).
    pub active_w: usize,
    pub widescreen: bool,
    /// 23.976 fps progressive with soft pulldown instead of 29.97 fps
    /// interlaced.
    pub film: bool,
    /// Average and maximum video bit rate in Mbit/s.
    pub bitrate: f32,
    pub maxrate: f32,
    /// Pictures per GOP (≤ 18).
    pub gop: usize,
    /// Use the low-bit-rate quantiser matrices instead of the MPEG defaults.
    pub low_rate_matrix: bool,
    pub two_pass: bool,
    /// Mastering stages, 0 = off.
    pub ee: f32,
    pub dnr: f32,
    pub prefilter: f32,
    /// Player chroma upsampling error, 0 = correct field-based upsampling.
    pub cue: f32,
    pub output: DvdOutput,
    /// Scale the output to square pixels (640×480 / 854×480).
    pub square: bool,
    /// Which decoded frame a still image returns: 'i', 'p' or 'b'.
    pub frame_type: char,
    /// Keep the MPEG-2 stream (.m2v elementary stream, or .vob/.mpg
    /// program stream).
    pub keep_mpeg2: Option<PathBuf>,
}

/// DVD-Video NTSC limits.
pub const MAX_VIDEO_RATE_MBPS: f32 = 9.8;
pub const VBV_BUFFER_BITS: u32 = 1_835_008;
pub const MAX_GOP: usize = 18;

impl DvdConfig {
    /// Resolve the DVD parameters.  Ranges are sampled from `rng`, which
    /// continues the per-file RNG after the analog effects.
    pub fn from_values<R: Rng>(v: &ParamValues, rng: &mut R) -> Self {
        let num = |id: &str, rng: &mut R| RangeParam::sample_opt(v.number(id), rng);
        let (width, height) = match v.choice("dvd-size") {
            "704x480" => (704, 480),
            "352x480" => (352, 480),
            "352x240" => (352, 240),
            _ => (720, 480),
        };
        let bitrate = num("dvd-bitrate", rng).unwrap_or(6.0);
        let maxrate = num("dvd-maxrate", rng).unwrap_or(9.5).min(MAX_VIDEO_RATE_MBPS);
        let gop = num("dvd-gop", rng).unwrap_or(15.0).round().clamp(1.0, MAX_GOP as f32) as usize;
        let ee = num("dvd-ee", rng).unwrap_or(0.0);
        let dnr = num("dvd-dnr", rng).unwrap_or(0.0);
        let prefilter = num("dvd-prefilter", rng).unwrap_or(0.0);
        let cue = num("dvd-cue", rng).unwrap_or(0.0);
        Self {
            width,
            height,
            active_w: if width == 720 { 704 } else { width },
            widescreen: v.choice("dvd-aspect") == "16:9",
            film: v.choice("dvd-rate") == "film",
            bitrate: bitrate.clamp(0.5, maxrate),
            maxrate,
            gop,
            low_rate_matrix: v.choice("dvd-matrix") == "lowrate",
            two_pass: v.flag("dvd-2pass"),
            ee,
            dnr,
            prefilter,
            cue,
            output: match v.choice("dvd-output") {
                "svideo" => DvdOutput::Svideo,
                "composite" => DvdOutput::Composite,
                _ => DvdOutput::Digital,
            },
            square: v.choice("dvd-output-size") == "square",
            frame_type: v.choice("dvd-frame-type").chars().next().unwrap_or('b'),
            keep_mpeg2: v.text("dvd-keep-mpeg2").map(PathBuf::from),
        }
    }

    /// Interlaced coding: 480-line video titles.  SIF (352×240) and film
    /// titles are progressive.
    pub fn interlaced(&self) -> bool {
        !self.film && self.height == 480
    }

    pub fn frame_rate(&self) -> (u32, u32) {
        if self.film { (24000, 1001) } else { (30000, 1001) }
    }

    /// Display aspect ratio of the active picture area.
    pub fn display_aspect(&self) -> f64 {
        if self.widescreen { 16.0 / 9.0 } else { 4.0 / 3.0 }
    }

    /// Square-pixel size of the active picture: 640×480 (4:3) or 854×480
    /// (16:9) for 480-line titles, 320×240 / 427×240 for SIF.
    pub fn square_size(&self) -> (usize, usize) {
        let h = self.height;
        let w = (h as f64 * self.display_aspect()).round() as usize;
        (w + (w & 1), h)
    }

    /// Size of the frames this path outputs.
    pub fn output_size(&self) -> (usize, usize) {
        match (self.output, self.square) {
            (_, true) => {
                let (w, h) = self.square_size();
                // Analog outputs are always 480 lines.
                if self.output != DvdOutput::Digital && h == 240 { (w * 2, 480) } else { (w, h) }
            }
            (DvdOutput::Digital, false) => (self.width, self.height),
            // Players output 720×480 on their analog outputs.
            (_, false) => (720, 480),
        }
    }

    /// `name=value` lines describing the resolved settings.
    pub fn summary(&self, v: &ParamValues) -> Vec<String> {
        let mut s = Vec::new();
        let ranged = |id: &str| v.number(id).is_some_and(|r| r.is_range());
        if ranged("dvd-bitrate") {
            s.push(format!("dvd_bitrate={:.2}", self.bitrate));
        }
        if ranged("dvd-maxrate") {
            s.push(format!("dvd_maxrate={:.2}", self.maxrate));
        }
        if ranged("dvd-gop") {
            s.push(format!("dvd_gop={}", self.gop));
        }
        for (id, val) in [("dvd-ee", self.ee), ("dvd-dnr", self.dnr), ("dvd-prefilter", self.prefilter), ("dvd-cue", self.cue)] {
            if ranged(id) {
                s.push(format!("{}={:.3}", id.replace('-', "_"), val));
            }
        }
        s
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Planes and resampling
// ────────────────────────────────────────────────────────────────────────────

/// Separable Lanczos-3 resampling of one plane.  When shrinking, the
/// kernel is widened by the scale factor so it also acts as the
/// anti-aliasing filter.
pub fn resize_plane(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    if sw == dw && sh == dh {
        return src.to_vec();
    }
    let hw = weights(sw, dw);
    let mut tmp = vec![0.0f32; dw * sh];
    for y in 0..sh {
        let row = &src[y * sw..(y + 1) * sw];
        for (x, (start, w)) in hw.iter().enumerate() {
            tmp[y * dw + x] = w.iter().enumerate().map(|(k, c)| c * row[start + k]).sum();
        }
    }
    if sh == dh {
        return tmp;
    }
    let vw = weights(sh, dh);
    let mut out = vec![0.0f32; dw * dh];
    for (y, (start, w)) in vw.iter().enumerate() {
        for (k, c) in w.iter().enumerate() {
            let src_row = &tmp[(start + k) * dw..(start + k + 1) * dw];
            let dst = &mut out[y * dw..(y + 1) * dw];
            for x in 0..dw {
                dst[x] += c * src_row[x];
            }
        }
    }
    out
}

fn lanczos3(x: f32) -> f32 {
    if x.abs() < 1e-6 {
        return 1.0;
    }
    if x.abs() >= 3.0 {
        return 0.0;
    }
    let px = std::f32::consts::PI * x;
    3.0 * px.sin() * (px / 3.0).sin() / (px * px)
}

/// For every output sample: first input index and normalised weights
/// (edge samples are repeated).
fn weights(n_in: usize, n_out: usize) -> Vec<(usize, Vec<f32>)> {
    let scale = n_out as f32 / n_in as f32;
    let stretch = (1.0 / scale).max(1.0);
    let support = 3.0 * stretch;
    (0..n_out)
        .map(|i| {
            let center = (i as f32 + 0.5) / scale - 0.5;
            let lo = (center - support).floor() as isize + 1;
            let hi = (center + support).floor() as isize;
            let mut idx = Vec::new();
            let mut w = Vec::new();
            for j in lo..=hi {
                idx.push(j.clamp(0, n_in as isize - 1) as usize);
                w.push(lanczos3((j as f32 - center) / stretch));
            }
            let sum: f32 = w.iter().sum();
            // Express as a contiguous run starting at min index.
            let start = *idx.iter().min().unwrap();
            let end = *idx.iter().max().unwrap();
            let mut run = vec![0.0f32; end - start + 1];
            for (j, c) in idx.iter().zip(&w) {
                run[j - start] += c / sum;
            }
            (start, run)
        })
        .collect()
}

/// Y'CbCr planes (BT.601 studio-range code values as f32), full resolution.
#[derive(Clone, Debug)]
pub struct Planes {
    pub w: usize,
    pub h: usize,
    pub y: Vec<f32>,
    pub cb: Vec<f32>,
    pub cr: Vec<f32>,
}

/// RGB24 to BT.601 studio-range Y'CbCr (Y 16–235, Cb/Cr 16–240).
pub fn rgb_to_planes(rgb: &[u8], w: usize, h: usize) -> Planes {
    let n = w * h;
    let (mut y, mut cb, mut cr) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let r = rgb[i * 3] as f32 / 255.0;
        let g = rgb[i * 3 + 1] as f32 / 255.0;
        let b = rgb[i * 3 + 2] as f32 / 255.0;
        let yl = 0.299 * r + 0.587 * g + 0.114 * b;
        y[i] = 16.0 + 219.0 * yl;
        cb[i] = 128.0 + 224.0 * (b - yl) / 1.772;
        cr[i] = 128.0 + 224.0 * (r - yl) / 1.402;
    }
    Planes { w, h, y, cb, cr }
}

fn to_rgb_pixel(y: f32, cb: f32, cr: f32) -> [u8; 3] {
    let yl = (y - 16.0) / 219.0;
    let pb = (cb - 128.0) / 224.0;
    let pr = (cr - 128.0) / 224.0;
    let r = yl + 1.402 * pr;
    let b = yl + 1.772 * pb;
    let g = (yl - 0.299 * r - 0.114 * b) / 0.587;
    [r, g, b].map(|v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
}

/// Resize RGB24 with Lanczos-3.
pub fn resize_rgb(rgb: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    if sw == dw && sh == dh {
        return rgb.to_vec();
    }
    let mut out = vec![0u8; dw * dh * 3];
    for c in 0..3 {
        let plane: Vec<f32> = rgb.iter().skip(c).step_by(3).map(|&v| v as f32).collect();
        let r = resize_plane(&plane, sw, sh, dw, dh);
        for (i, v) in r.iter().enumerate() {
            out[i * 3 + c] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

// ────────────────────────────────────────────────────────────────────────────
// Mastering
// ────────────────────────────────────────────────────────────────────────────

/// Fit a source picture into the DVD frame and apply the mastering stages.
///
/// `src_dar` is the source's display aspect ratio.  The active area
/// (704 or 352 samples wide) carries the full 4:3 or 16:9 picture, so the
/// samples are not square: 10:11 for 4:3 and 40:33 for 16:9 at 704
/// wide.  A source with a different aspect is letterboxed or
/// pillarboxed; 720-wide frames get 8 samples of black on each side.
pub fn master(rgb: &[u8], w: usize, h: usize, src_dar: f64, cfg: &DvdConfig) -> Planes {
    let src = rgb_to_planes(rgb, w, h);
    let (aw, ah) = (cfg.active_w, cfg.height);
    let dar = cfg.display_aspect();
    let (cw, ch) = if src_dar > dar + 1e-3 {
        (aw, ((ah as f64 * dar / src_dar).round() as usize).max(2) & !1)
    } else if src_dar < dar - 1e-3 {
        (((aw as f64 * src_dar / dar).round() as usize).max(2) & !1, ah)
    } else {
        (aw, ah)
    };
    let (ox, oy) = ((cfg.width - cw) / 2, (ah - ch) / 2);
    let mut out = Planes {
        w: cfg.width,
        h: cfg.height,
        y: vec![16.0; cfg.width * cfg.height],
        cb: vec![128.0; cfg.width * cfg.height],
        cr: vec![128.0; cfg.width * cfg.height],
    };
    for (dst, plane) in [(&mut out.y, &src.y), (&mut out.cb, &src.cb), (&mut out.cr, &src.cr)] {
        let r = resize_plane(plane, w, h, cw, ch);
        for row in 0..ch {
            let d = (oy + row) * cfg.width + ox;
            dst[d..d + cw].copy_from_slice(&r[row * cw..(row + 1) * cw]);
        }
    }
    if cfg.prefilter > 0.0 {
        prefilter(&mut out, cfg.prefilter);
    }
    if cfg.dnr > 0.0 {
        spatial_dnr(&mut out, cfg.dnr);
    }
    if cfg.ee > 0.0 {
        edge_enhance(&mut out, cfg.ee);
    }
    out
}

/// Normalised windowed-sinc lowpass; `cutoff` in cycles per sample.
fn lowpass_kernel(cutoff: f32, taps: usize) -> Vec<f32> {
    let m = (taps - 1) as f32 / 2.0;
    let mut k: Vec<f32> = (0..taps)
        .map(|i| {
            let x = i as f32 - m;
            let sinc = if x == 0.0 { 2.0 * cutoff } else { (2.0 * std::f32::consts::PI * cutoff * x).sin() / (std::f32::consts::PI * x) };
            let window = 0.42 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (taps - 1) as f32).cos()
                + 0.08 * (4.0 * std::f32::consts::PI * i as f32 / (taps - 1) as f32).cos();
            sinc * window
        })
        .collect();
    let s: f32 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= s);
    k
}

fn convolve_rows(plane: &mut [f32], w: usize, h: usize, k: &[f32]) {
    let half = k.len() / 2;
    let mut row = vec![0.0f32; w];
    for y in 0..h {
        let line = &mut plane[y * w..(y + 1) * w];
        row.copy_from_slice(line);
        for x in 0..w {
            let mut acc = 0.0;
            for (j, c) in k.iter().enumerate() {
                let xi = (x + j).saturating_sub(half).min(w - 1);
                acc += c * row[xi];
            }
            line[x] = acc;
        }
    }
}

fn convolve_cols(plane: &mut [f32], w: usize, h: usize, k: &[f32]) {
    let half = k.len() / 2;
    let src = plane.to_vec();
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (j, c) in k.iter().enumerate() {
                let yi = (y + j).saturating_sub(half).min(h - 1);
                acc += c * src[yi * w + x];
            }
            plane[y * w + x] = acc;
        }
    }
}

fn gaussian(sigma: f32) -> Vec<f32> {
    let r = (sigma * 3.0).ceil() as i32;
    let mut k: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let s: f32 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= s);
    k
}

/// Studio pre-compression lowpass on luma: horizontal cutoff from about
/// 6 MHz (strength 0) down to about 3.8 MHz (strength 1) at 13.5 MHz
/// sampling, which removes the finest detail before MPEG-2 encoding.
fn prefilter(p: &mut Planes, strength: f32) {
    let cutoff = 0.45 - 0.17 * strength.clamp(0.0, 1.0);
    let k = lowpass_kernel(cutoff, 21);
    convolve_rows(&mut p.y, p.w, p.h, &k);
}

/// Mastering noise reduction: an edge-preserving (bilateral) 5×5 filter
/// that flattens fine grain and noise while keeping strong edges, the
/// waxy look of heavily noise-reduced transfers.
fn spatial_dnr(p: &mut Planes, strength: f32) {
    let s = strength.clamp(0.0, 1.0);
    let (w, h) = (p.w, p.h);
    let spatial: Vec<f32> = (-2i32..=2)
        .flat_map(|dy| (-2i32..=2).map(move |dx| (-((dx * dx + dy * dy) as f32) / (2.0 * 1.5 * 1.5)).exp()))
        .collect();
    for (plane, range) in [(&mut p.y, 2.0 + 10.0 * s), (&mut p.cb, 3.0 + 12.0 * s), (&mut p.cr, 3.0 + 12.0 * s)] {
        let src = plane.clone();
        let inv = 1.0 / (2.0 * range * range);
        for y in 0..h {
            for x in 0..w {
                let c = src[y * w + x];
                let (mut acc, mut norm) = (0.0f32, 0.0f32);
                for (k, (dy, dx)) in (-2i32..=2).flat_map(|dy| (-2i32..=2).map(move |dx| (dy, dx))).enumerate() {
                    let yy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                    let xx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                    let v = src[yy * w + xx];
                    let wgt = spatial[k] * (-(v - c) * (v - c) * inv).exp();
                    acc += wgt * v;
                    norm += wgt;
                }
                plane[y * w + x] = c + s * (acc / norm - c);
            }
        }
    }
}

/// Telecine / mastering edge enhancement: an unsharp mask with a wide,
/// soft kernel (σ ≈ 2 samples horizontally, 1 line vertically) and a
/// gain up to 1.6, so edges get the bright and dark outlines, several
/// pixels wide, that many DVD transfers show.  Overshoots may exceed
/// nominal white/black, as on real discs.
fn edge_enhance(p: &mut Planes, strength: f32) {
    let gain = 1.6 * strength.clamp(0.0, 1.0);
    let mut blur = p.y.clone();
    convolve_rows(&mut blur, p.w, p.h, &gaussian(2.0));
    convolve_cols(&mut blur, p.w, p.h, &gaussian(1.0));
    for (y, b) in p.y.iter_mut().zip(&blur) {
        *y = (*y + gain * (*y - b)).clamp(1.0, 254.0);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// 4:2:0
// ────────────────────────────────────────────────────────────────────────────

/// An 8-bit 4:2:0 frame as MPEG-2 stores it.
#[derive(Clone, Debug, PartialEq)]
pub struct Yuv420 {
    pub w: usize,
    pub h: usize,
    pub y: Vec<u8>,
    pub cb: Vec<u8>,
    pub cr: Vec<u8>,
}

impl Yuv420 {
    pub fn bytes_len(w: usize, h: usize) -> usize {
        w * h + 2 * (w / 2) * (h / 2)
    }

    pub fn from_bytes(data: &[u8], w: usize, h: usize) -> Self {
        let (ys, cs) = (w * h, (w / 2) * (h / 2));
        Self { w, h, y: data[..ys].to_vec(), cb: data[ys..ys + cs].to_vec(), cr: data[ys + cs..ys + 2 * cs].to_vec() }
    }

    pub fn write_to(&self, out: &mut impl Write) -> std::io::Result<()> {
        out.write_all(&self.y)?;
        out.write_all(&self.cb)?;
        out.write_all(&self.cr)
    }
}

fn q8(v: f32) -> u8 {
    v.round().clamp(1.0, 254.0) as u8
}

/// Subsample chroma to 4:2:0 with MPEG-2 siting: horizontally co-sited
/// with even luma samples ([1 2 1]/4), vertically between lines.
///
/// For interlaced pictures each field is subsampled on its own: top-field
/// chroma line k sits a quarter of the way from field line 2k to 2k+1
/// (weights 3/4, 1/4), bottom-field chroma three quarters of the way
/// (1/4, 3/4), and frame chroma lines alternate top/bottom.  Progressive
/// pictures average frame lines 2k and 2k+1.
pub fn to_420(p: &Planes, interlaced: bool) -> Yuv420 {
    let (w, h) = (p.w, p.h);
    let (cw, chh) = (w / 2, h / 2);
    let sub = |plane: &[f32]| -> Vec<u8> {
        // Horizontal [1 2 1] at even positions.
        let mut hs = vec![0.0f32; cw * h];
        for y in 0..h {
            for x in 0..cw {
                let c = 2 * x;
                let l = plane[y * w + c.saturating_sub(1)];
                let m = plane[y * w + c];
                let r = plane[y * w + (c + 1).min(w - 1)];
                hs[y * cw + x] = 0.25 * l + 0.5 * m + 0.25 * r;
            }
        }
        let mut out = vec![0u8; cw * chh];
        for k in 0..chh {
            for x in 0..cw {
                let at = |line: usize| hs[line.min(h - 1) * cw + x];
                out[k * cw + x] = q8(if interlaced {
                    // Frame chroma line k: field = k % 2, field chroma line k / 2.
                    let field = k % 2;
                    let fk = k / 2;
                    let a = at(4 * fk + field); // field line 2fk
                    let b = at(4 * fk + 2 + field); // field line 2fk + 1
                    if field == 0 { 0.75 * a + 0.25 * b } else { 0.25 * a + 0.75 * b }
                } else {
                    0.5 * (at(2 * k) + at(2 * k + 1))
                });
            }
        }
        out
    };
    Yuv420 { w, h, y: p.y.iter().map(|&v| q8(v)).collect(), cb: sub(&p.cb), cr: sub(&p.cr) }
}

/// Upsample one 4:2:0 chroma plane to full resolution.
///
/// `field_based`: each field's chroma is interpolated only from that
/// field's chroma lines at their true positions (correct for interlaced
/// pictures).  Otherwise frame chroma lines are treated as one
/// progressive set at frame positions 2k + 0.5, which for interlaced
/// pictures blends the two fields' colour (the chroma upsampling error).
fn upsample_chroma(c: &[u8], w: usize, h: usize, field_based: bool) -> Vec<f32> {
    let (cw, chh) = (w / 2, h / 2);
    let cv = |line: isize, x: usize| c[(line.clamp(0, chh as isize - 1) as usize) * cw + x] as f32;
    // Vertical first, at chroma width.
    let mut v = vec![0.0f32; cw * h];
    for y in 0..h {
        for x in 0..cw {
            v[y * cw + x] = if field_based {
                let field = y % 2;
                let fy = (y / 2) as f32; // line within the field
                // Field chroma line k sits at field position 2k + 0.25 (top) or 2k + 0.75 (bottom).
                let pos = (fy - if field == 0 { 0.25 } else { 0.75 }) / 2.0;
                let k0 = pos.floor();
                let t = pos - k0;
                let line = |k: isize| cv(2 * k + field as isize, x);
                // Clamp to this field's own chroma lines.
                let max_k = (chh / 2) as isize - 1;
                let a = line((k0 as isize).clamp(0, max_k));
                let b = line((k0 as isize + 1).clamp(0, max_k));
                a + (b - a) * t
            } else {
                let pos = (y as f32 - 0.5) / 2.0;
                let k0 = pos.floor();
                let t = pos - k0;
                let a = cv(k0 as isize, x);
                let b = cv(k0 as isize + 1, x);
                a + (b - a) * t
            };
        }
    }
    // Horizontal: co-sited with even luma samples.
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let k = x / 2;
            out[y * w + x] = if x % 2 == 0 {
                v[y * cw + k]
            } else {
                0.5 * (v[y * cw + k] + v[y * cw + (k + 1).min(cw - 1)])
            };
        }
    }
    out
}

/// Player decode to RGB: upsample chroma and convert from BT.601.
/// `interlaced_picture` selects field-based upsampling; `cue` blends in
/// the progressive (wrong) upsampling for interlaced pictures.
pub fn yuv420_to_rgb(f: &Yuv420, interlaced_picture: bool, cue: f32) -> Vec<u8> {
    let (w, h) = (f.w, f.h);
    let up = |c: &[u8]| -> Vec<f32> {
        if interlaced_picture {
            let good = upsample_chroma(c, w, h, true);
            if cue > 0.0 {
                let bad = upsample_chroma(c, w, h, false);
                good.iter().zip(&bad).map(|(g, b)| g + cue.min(1.0) * (b - g)).collect()
            } else {
                good
            }
        } else {
            upsample_chroma(c, w, h, false)
        }
    };
    let cb = up(&f.cb);
    let cr = up(&f.cr);
    let mut rgb = vec![0u8; w * h * 3];
    for i in 0..w * h {
        rgb[i * 3..i * 3 + 3].copy_from_slice(&to_rgb_pixel(f.y[i] as f32, cb[i], cr[i]));
    }
    rgb
}

// ────────────────────────────────────────────────────────────────────────────
// Player output geometry
// ────────────────────────────────────────────────────────────────────────────

/// Digital output: the decoded frame as is, or scaled to square pixels
/// (the 704/352 active area to 640×480 or 854×480, 320×240 for SIF).
pub fn digital_output(rgb: &[u8], cfg: &DvdConfig) -> (Vec<u8>, usize, usize) {
    if !cfg.square {
        return (rgb.to_vec(), cfg.width, cfg.height);
    }
    let active = crop_active(rgb, cfg);
    let (sw, sh) = cfg.square_size();
    (resize_rgb(&active, cfg.active_w, cfg.height, sw, sh), sw, sh)
}

fn crop_active(rgb: &[u8], cfg: &DvdConfig) -> Vec<u8> {
    let off = (cfg.width - cfg.active_w) / 2;
    let mut out = Vec::with_capacity(cfg.active_w * cfg.height * 3);
    for y in 0..cfg.height {
        let s = (y * cfg.width + off) * 3;
        out.extend_from_slice(&rgb[s..s + cfg.active_w * 3]);
    }
    out
}

/// The picture a player sends to its analog output: the active area at
/// 704×480 (players scale 352-wide and 240-line titles up).  The NTSC
/// encoder maps this to the 52.6 µs active line.
pub fn analog_source(rgb: &[u8], cfg: &DvdConfig) -> (Vec<u8>, usize, usize) {
    let active = crop_active(rgb, cfg);
    (resize_rgb(&active, cfg.active_w, cfg.height, 704, 480), 704, 480)
}

/// Pad a 704-wide analog capture to 720 with 8 black samples each side.
pub fn pad_to_720(rgb: &[u8], h: usize) -> Vec<u8> {
    let mut out = vec![0u8; 720 * h * 3];
    for y in 0..h {
        out[(y * 720 + 8) * 3..(y * 720 + 712) * 3].copy_from_slice(&rgb[y * 704 * 3..(y + 1) * 704 * 3]);
    }
    out
}

// ────────────────────────────────────────────────────────────────────────────
// MPEG-2 encode / decode (ffmpeg)
// ────────────────────────────────────────────────────────────────────────────

/// The low-bit-rate quantiser matrices, in raster order.  They follow the
/// usual shape (weights rising with spatial frequency u + v) but rise
/// faster than the MPEG defaults, spending fewer bits on fine detail; the
/// intra DC weight must stay 8.
pub fn low_rate_matrices() -> ([u8; 64], [u8; 64]) {
    let mut intra = [0u8; 64];
    let mut inter = [0u8; 64];
    for v in 0..8 {
        for u in 0..8 {
            let f = (u + v) as f32;
            intra[v * 8 + u] = (16.0 + 2.6 * f.powf(1.35)).round().min(99.0) as u8;
            inter[v * 8 + u] = (18.0 + 2.0 * f.powf(1.3)).round().min(80.0) as u8;
        }
    }
    intra[0] = 8;
    (intra, inter)
}

/// ffmpeg arguments for encoding raw yuv420p frames to a DVD-Video
/// compliant MPEG-2 elementary stream (the video part of ffmpeg's
/// `-target ntsc-dvd`, with our own rate, GOP and matrix settings).
pub fn encoder_args(cfg: &DvdConfig) -> Vec<String> {
    let (rn, rd) = cfg.frame_rate();
    let mut a: Vec<String> = [
        "-c:v", "mpeg2video", "-pix_fmt", "yuv420p",
        "-profile:v", "4", "-level:v", "8", // Main Profile @ Main Level
        "-minrate", "0",
        "-bufsize", &VBV_BUFFER_BITS.to_string(),
        "-bf", "2", // M = 3
        // Closed GOPs of fixed length: no extra I pictures at scene cuts
        // (ffmpeg cannot combine closed GOPs with scene-cut detection).
        "-sc_threshold", "1000000000",
        "-intra_dc_precision", "1", // 9-bit DC, as most DVD encoders use
        "-intra_vlc", "1",
        "-mbd", "rd", "-trellis", "1",
        "-seq_disp_ext", "1", "-video_format", "2", // NTSC
        "-color_primaries", "smpte170m", "-color_trc", "smpte170m",
        "-colorspace", "smpte170m", "-color_range", "tv",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    a.extend([
        "-b:v".into(), format!("{:.0}", cfg.bitrate * 1e6),
        "-maxrate".into(), format!("{:.0}", cfg.maxrate * 1e6),
        "-g".into(), cfg.gop.to_string(),
        "-r".into(), format!("{}/{}", rn, rd),
        "-aspect".into(), if cfg.widescreen { "16:9".into() } else { "4:3".into() },
    ]);
    // Closed GOPs; interlaced coding with field DCT and alternate scan.
    if cfg.interlaced() {
        a.extend(["-flags", "+cgop+ilme+ildct", "-alternate_scan", "1", "-top", "1"].map(String::from));
    } else {
        a.extend(["-flags", "+cgop"].map(String::from));
    }
    if cfg.low_rate_matrix {
        let (intra, inter) = low_rate_matrices();
        let join = |m: &[u8; 64]| m.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",");
        a.extend(["-intra_matrix".into(), join(&intra), "-inter_matrix".into(), join(&inter)]);
    }
    a
}

fn tool(program: &str) -> std::process::Command {
    #[allow(unused_mut)]
    let mut cmd = std::process::Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    cmd
}

/// A temporary working directory, removed on drop.
pub struct WorkDir(pub PathBuf);

impl WorkDir {
    pub fn new() -> Result<Self> {
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "ntsc-dvd-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&p)?;
        Ok(Self(p))
    }
}

impl Drop for WorkDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Streaming MPEG-2 encoder.  One-pass encodes pipe straight into ffmpeg;
/// two-pass encodes spool the frames to a temporary file first.
pub struct Encoder {
    cfg: DvdConfig,
    work: WorkDir,
    child: Option<std::process::Child>,
    sink: Box<dyn Write + Send>,
    frames: usize,
}

impl Encoder {
    pub fn start(cfg: &DvdConfig) -> Result<Self> {
        let work = WorkDir::new()?;
        let (child, sink): (Option<std::process::Child>, Box<dyn Write + Send>) = if cfg.two_pass {
            let f = std::fs::File::create(work.0.join("frames.yuv"))?;
            (None, Box::new(std::io::BufWriter::new(f)))
        } else {
            let mut child = encode_command(cfg, "pipe:0", &work.0.join("video.m2v"), None)
                .stdin(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .context("Failed to start ffmpeg (mpeg2video). Is ffmpeg installed?")?;
            let stdin = child.stdin.take().unwrap();
            (Some(child), Box::new(std::io::BufWriter::new(stdin)))
        };
        Ok(Self { cfg: cfg.clone(), work, child, sink, frames: 0 })
    }

    pub fn push(&mut self, f: &Yuv420) -> Result<()> {
        if let Err(e) = f.write_to(&mut self.sink) {
            // ffmpeg stopped reading: report its own error message.
            let mut err = String::new();
            if let Some(child) = self.child.as_mut() {
                if let Some(mut s) = child.stderr.take() {
                    let _ = s.read_to_string(&mut err);
                }
                let _ = child.wait();
            }
            bail!("ffmpeg mpeg2video encode failed: {} ({})", err.trim(), e);
        }
        self.frames += 1;
        Ok(())
    }

    /// Finish encoding: returns the elementary stream (soft pulldown
    /// applied for film) and keeps a copy if requested.
    pub fn finish(mut self) -> Result<Encoded> {
        drop(std::mem::replace(&mut self.sink, Box::new(std::io::sink())));
        let es_path = self.work.0.join("video.m2v");
        if let Some(mut child) = self.child.take() {
            let mut err = String::new();
            if let Some(mut e) = child.stderr.take() {
                let _ = e.read_to_string(&mut err);
            }
            if !child.wait()?.success() {
                bail!("ffmpeg mpeg2video encode failed: {}", err.trim());
            }
        } else {
            let raw = self.work.0.join("frames.yuv");
            let log = self.work.0.join("pass");
            for pass in [1, 2] {
                let out = if pass == 1 { PathBuf::from(if cfg!(windows) { "NUL" } else { "/dev/null" }) } else { es_path.clone() };
                let o = encode_command(&self.cfg, &raw.to_string_lossy(), &out, Some((pass, &log)))
                    .stderr(Stdio::piped())
                    .output()
                    .context("Failed to start ffmpeg (mpeg2video)")?;
                if !o.status.success() {
                    bail!("ffmpeg mpeg2video pass {} failed: {}", pass, String::from_utf8_lossy(&o.stderr).trim());
                }
            }
        }
        let mut es = std::fs::read(&es_path).context("reading the MPEG-2 stream")?;
        let stream = if self.cfg.film { mpeg2::apply_soft_pulldown(&mut es)? } else { mpeg2::parse(&es)? };
        std::fs::write(&es_path, &es)?;
        if let Some(keep) = &self.cfg.keep_mpeg2 {
            keep_stream(&es_path, keep)?;
        }
        Ok(Encoded { es, stream, work: self.work, frames: self.frames })
    }
}

fn encode_command(cfg: &DvdConfig, input: &str, output: &Path, pass: Option<(u32, &Path)>) -> std::process::Command {
    let (rn, rd) = cfg.frame_rate();
    let mut cmd = tool("ffmpeg");
    cmd.args(["-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .args(["-s", &format!("{}x{}", cfg.width, cfg.height), "-r", &format!("{}/{}", rn, rd)])
        .args(["-i", input])
        .args(encoder_args(cfg));
    if let Some((n, log)) = pass {
        cmd.args(["-pass", &n.to_string(), "-passlogfile"]).arg(log);
    }
    if pass.is_some_and(|(n, _)| n == 1) {
        cmd.args(["-f", "null"]);
    } else {
        cmd.args(["-f", "mpeg2video"]);
    }
    cmd.arg(output);
    cmd
}

/// Save the stream: `.vob`/`.mpg` as a DVD program stream (ffmpeg's `dvd`
/// muxer), anything else as the raw elementary stream.
fn keep_stream(es: &Path, dest: &Path) -> Result<()> {
    let ext = dest.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
    if ext == "vob" || ext == "mpg" || ext == "mpeg" {
        let o = tool("ffmpeg")
            .args(["-v", "error", "-y", "-f", "mpegvideo", "-i"])
            .arg(es)
            .args(["-c", "copy", "-f", "dvd"])
            .arg(dest)
            .output()?;
        if !o.status.success() {
            bail!("muxing {} failed: {}", dest.display(), String::from_utf8_lossy(&o.stderr).trim());
        }
    } else {
        std::fs::copy(es, dest).with_context(|| format!("cannot write {}", dest.display()))?;
    }
    Ok(())
}

/// An encoded elementary stream.
pub struct Encoded {
    pub es: Vec<u8>,
    pub stream: mpeg2::Stream,
    work: WorkDir,
    pub frames: usize,
}

impl Encoded {
    pub fn path(&self) -> PathBuf {
        self.work.0.join("video.m2v")
    }

    /// Start decoding: frames come out in display order, one per coded
    /// picture (no frame-rate conversion).
    pub fn decoder(&self) -> Result<Decoder> {
        let mut child = tool("ffmpeg")
            .args(["-v", "error", "-f", "mpegvideo", "-i"])
            .arg(self.path())
            .args(["-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .context("Failed to start ffmpeg (MPEG-2 decode)")?;
        let out = child.stdout.take().unwrap();
        let order: Vec<(char, bool)> = self.stream.display_order().iter().map(|p| (p.kind, !p.progressive_frame)).collect();
        Ok(Decoder {
            child,
            reader: std::io::BufReader::new(out),
            w: self.stream.seq.width as usize,
            h: self.stream.seq.height as usize,
            order,
            next: 0,
        })
    }
}

/// Decoded frames with their picture type and interlace flag.
pub struct Decoder {
    child: std::process::Child,
    reader: std::io::BufReader<std::process::ChildStdout>,
    w: usize,
    h: usize,
    order: Vec<(char, bool)>,
    next: usize,
}

/// A decoded picture.
pub struct DecodedFrame {
    pub frame: Yuv420,
    /// 'I', 'P' or 'B'.
    pub kind: char,
    /// Coded as an interlaced picture (progressive_frame = 0).
    pub interlaced: bool,
}

impl Iterator for Decoder {
    type Item = DecodedFrame;

    fn next(&mut self) -> Option<DecodedFrame> {
        let mut buf = vec![0u8; Yuv420::bytes_len(self.w, self.h)];
        self.reader.read_exact(&mut buf).ok()?;
        let (kind, interlaced) = self.order.get(self.next).copied().unwrap_or(('?', true));
        self.next += 1;
        Some(DecodedFrame { frame: Yuv420::from_bytes(&buf, self.w, self.h), kind, interlaced })
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> DvdConfig {
        let v = ParamValues::default();
        DvdConfig::from_values(&v, &mut rand::rng())
    }

    #[test]
    fn colour_conversion_round_trips() {
        let rgb: Vec<u8> = (0..64 * 4).flat_map(|i| [(i * 7 % 256) as u8, (i * 13 % 256) as u8, (i * 29 % 256) as u8]).collect();
        let p = rgb_to_planes(&rgb, 64, 4);
        for i in 0..64 * 4 {
            let back = to_rgb_pixel(p.y[i], p.cb[i], p.cr[i]);
            for c in 0..3 {
                assert!((back[c] as i32 - rgb[i * 3 + c] as i32).abs() <= 1);
            }
        }
    }

    #[test]
    fn interlaced_chroma_stays_within_its_field() {
        // Top field red, bottom field blue: field-based 4:2:0 must keep
        // them apart, progressive upsampling (CUE) mixes them.
        let (w, h) = (16, 16);
        let rgb: Vec<u8> = (0..w * h).flat_map(|i| if (i / w) % 2 == 0 { [200u8, 30, 30] } else { [30, 30, 200] }).collect();
        let p = rgb_to_planes(&rgb, w, h);
        let f = to_420(&p, true);
        let good = yuv420_to_rgb(&f, true, 0.0);
        let bad = yuv420_to_rgb(&f, true, 1.0);
        let px = |img: &[u8], y: usize| [img[(y * w + 8) * 3], img[(y * w + 8) * 3 + 2]];
        let g = px(&good, 8);
        assert!(g[0] > 150 && g[1] < 80, "top-field line stays red: {:?}", g);
        let b = px(&bad, 8);
        assert!(b[1] as i32 > g[1] as i32 + 30, "CUE mixes in the other field's blue: {:?} vs {:?}", b, g);
    }

    #[test]
    fn geometry_and_aspect() {
        let c = cfg();
        assert_eq!((c.width, c.height, c.active_w), (720, 480, 704));
        assert_eq!(c.square_size(), (640, 480));
        // A square-pixel 4:3 source fills the active area; 720 has 8-sample
        // black borders.
        let rgb = vec![255u8; 640 * 480 * 3];
        let m = master(&rgb, 640, 480, 4.0 / 3.0, &c);
        assert!(m.y[240 * 720 + 4] < 20.0 && m.y[240 * 720 + 20] > 230.0);
        // 16:9 source in a 4:3 frame is letterboxed.
        let m = master(&vec![255u8; 854 * 480 * 3], 854, 480, 16.0 / 9.0, &c);
        assert!(m.y[10 * 720 + 360] < 20.0 && m.y[240 * 720 + 360] > 230.0);
    }

    #[test]
    fn low_rate_matrix_shape() {
        let (intra, inter) = low_rate_matrices();
        assert_eq!(intra[0], 8);
        assert!(intra[63] > intra[1] && inter[63] > inter[0]);
        assert!(intra.iter().all(|&v| v >= 8) && inter.iter().all(|&v| v >= 16));
    }
}
