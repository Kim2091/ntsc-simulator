//! MPEG-2 video elementary-stream inspection and patching (ISO/IEC 13818-2).
//!
//! Only the headers are read: sequence header and extension, GOP header,
//! picture header and picture coding extension.  This is enough to
//!   - check a stream against the DVD-Video limits ([`analyze`]),
//!   - apply soft 3:2 pulldown to a 23.976 fps progressive stream by
//!     setting the repeat_first_field / top_field_first flags
//!     ([`apply_soft_pulldown`]),
//!   - find the display order and type of every picture.

use anyhow::{bail, Result};

const PICTURE_START: u8 = 0x00;
const SEQUENCE_HEADER: u8 = 0xB3;
const EXTENSION_START: u8 = 0xB5;
const SEQUENCE_END: u8 = 0xB7;
const GOP_START: u8 = 0xB8;

const EXT_SEQUENCE: u8 = 1;
const EXT_PICTURE_CODING: u8 = 8;

/// Frame rates for frame_rate_code 1–8.
pub const FRAME_RATES: [(u32, u32); 8] =
    [(24000, 1001), (24, 1), (25, 1), (30000, 1001), (30, 1), (50, 1), (60000, 1001), (60, 1)];

/// MSB-first bit reader.
struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Bits<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read(&mut self, n: u32) -> u32 {
        let mut v = 0u32;
        for _ in 0..n {
            let byte = self.data.get(self.pos / 8).copied().unwrap_or(0);
            let bit = (byte >> (7 - (self.pos % 8))) & 1;
            v = (v << 1) | bit as u32;
            self.pos += 1;
        }
        v
    }

    fn flag(&mut self) -> bool {
        self.read(1) == 1
    }
}

/// Offsets of every start code `00 00 01 xx`, with the code byte.
fn start_codes(es: &[u8]) -> Vec<(usize, u8)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + 3 < es.len() {
        if es[i] == 0 && es[i + 1] == 0 && es[i + 2] == 1 {
            out.push((i, es[i + 3]));
            i += 4;
        } else {
            i += 1;
        }
    }
    out
}

/// Sequence-level parameters.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SequenceInfo {
    pub width: u32,
    pub height: u32,
    /// aspect_ratio_information: 1 = square samples, 2 = 4:3, 3 = 16:9 display.
    pub aspect_code: u8,
    pub frame_rate_code: u8,
    /// Bit rate in units of 400 bit/s (header value, 18 + 12 bits).
    pub bit_rate_400: u32,
    /// VBV buffer size in units of 16 384 bits (10 + 8 bits).
    pub vbv_size_16k: u32,
    pub profile_and_level: u8,
    pub progressive_sequence: bool,
    /// 1 = 4:2:0, 2 = 4:2:2, 3 = 4:4:4.
    pub chroma_format: u8,
    /// Custom quantiser matrices (zigzag order) if the header loads them.
    pub intra_matrix: Option<[u8; 64]>,
    pub non_intra_matrix: Option<[u8; 64]>,
}

impl SequenceInfo {
    pub fn frame_rate(&self) -> Option<(u32, u32)> {
        FRAME_RATES.get((self.frame_rate_code as usize).wrapping_sub(1)).copied()
    }

    pub fn display_aspect(&self) -> &'static str {
        match self.aspect_code {
            1 => "1:1 samples",
            2 => "4:3",
            3 => "16:9",
            4 => "2.21:1",
            _ => "invalid",
        }
    }

    /// e.g. "Main Profile @ Main Level".
    pub fn profile_level_name(&self) -> String {
        let profile = match (self.profile_and_level >> 4) & 7 {
            1 => "High",
            2 => "Spatially Scalable",
            3 => "SNR Scalable",
            4 => "Main",
            5 => "Simple",
            _ => "unknown",
        };
        let level = match self.profile_and_level & 15 {
            4 => "High",
            6 => "High 1440",
            8 => "Main",
            10 => "Low",
            _ => "unknown",
        };
        format!("{} Profile @ {} Level", profile, level)
    }
}

/// One coded picture.
#[derive(Clone, Debug, PartialEq)]
pub struct Picture {
    /// Offset of the picture start code.
    pub offset: usize,
    /// Bytes up to the next picture, GOP or sequence header.
    pub size: usize,
    /// 'I', 'P' or 'B'.
    pub kind: char,
    pub temporal_reference: u16,
    /// Index of the GOP the picture belongs to.
    pub gop: usize,
    /// Offset of the picture coding extension, if found.
    pub coding_ext: Option<usize>,
    pub top_field_first: bool,
    pub repeat_first_field: bool,
    pub progressive_frame: bool,
    /// 3 = frame picture, 1/2 = field pictures.
    pub picture_structure: u8,
    /// Position in display order.
    pub display_index: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gop {
    pub closed: bool,
    pub broken_link: bool,
    /// Number of pictures in the GOP.
    pub pictures: usize,
}

/// Parsed headers of an elementary stream.
#[derive(Clone, Debug, Default)]
pub struct Stream {
    pub seq: SequenceInfo,
    /// Offset of the first sequence header and of its extension.
    pub seq_offsets: Vec<usize>,
    pub seq_ext_offsets: Vec<usize>,
    pub gops: Vec<Gop>,
    /// Pictures in coded (stream) order.
    pub pictures: Vec<Picture>,
}

fn read_matrix(b: &mut Bits) -> [u8; 64] {
    let mut m = [0u8; 64];
    for v in m.iter_mut() {
        *v = b.read(8) as u8;
    }
    m
}

/// Parse the headers of an MPEG-2 video elementary stream.
pub fn parse(es: &[u8]) -> Result<Stream> {
    let codes = start_codes(es);
    let mut s = Stream::default();
    let mut gop_index: Option<usize> = None;

    for (ci, &(off, code)) in codes.iter().enumerate() {
        let body = &es[off + 4..];
        match code {
            SEQUENCE_HEADER => {
                let mut b = Bits::new(body);
                let width = b.read(12);
                let height = b.read(12);
                let aspect = b.read(4) as u8;
                let frc = b.read(4) as u8;
                let rate_lo = b.read(18);
                b.read(1); // marker
                let vbv_lo = b.read(10);
                b.read(1); // constrained_parameters_flag
                let intra = if b.flag() { Some(read_matrix(&mut b)) } else { None };
                let non_intra = if b.flag() { Some(read_matrix(&mut b)) } else { None };
                if s.seq_offsets.is_empty() {
                    s.seq.width = width;
                    s.seq.height = height;
                    s.seq.aspect_code = aspect;
                    s.seq.frame_rate_code = frc;
                    s.seq.bit_rate_400 = rate_lo;
                    s.seq.vbv_size_16k = vbv_lo;
                    s.seq.intra_matrix = intra;
                    s.seq.non_intra_matrix = non_intra;
                }
                s.seq_offsets.push(off);
            }
            EXTENSION_START => {
                let mut b = Bits::new(body);
                let id = b.read(4) as u8;
                if id == EXT_SEQUENCE {
                    let pl = b.read(8) as u8;
                    let prog = b.flag();
                    let chroma = b.read(2) as u8;
                    let h_ext = b.read(2);
                    let v_ext = b.read(2);
                    let rate_hi = b.read(12);
                    b.read(1); // marker
                    let vbv_hi = b.read(8);
                    if s.seq_ext_offsets.is_empty() {
                        s.seq.profile_and_level = pl;
                        s.seq.progressive_sequence = prog;
                        s.seq.chroma_format = chroma;
                        s.seq.width |= h_ext << 12;
                        s.seq.height |= v_ext << 12;
                        s.seq.bit_rate_400 |= rate_hi << 18;
                        s.seq.vbv_size_16k |= vbv_hi << 10;
                    }
                    s.seq_ext_offsets.push(off);
                } else if id == EXT_PICTURE_CODING {
                    if let Some(p) = s.pictures.last_mut() {
                        if p.coding_ext.is_none() {
                            b.read(16); // f_codes
                            b.read(2); // intra_dc_precision
                            p.picture_structure = b.read(2) as u8;
                            p.top_field_first = b.flag();
                            b.read(5); // frame_pred_frame_dct .. alternate_scan
                            p.repeat_first_field = b.flag();
                            b.read(1); // chroma_420_type
                            p.progressive_frame = b.flag();
                            p.coding_ext = Some(off);
                        }
                    }
                }
            }
            GOP_START => {
                let mut b = Bits::new(body);
                b.read(25); // time_code
                let closed = b.flag();
                let broken = b.flag();
                s.gops.push(Gop { closed, broken_link: broken, pictures: 0 });
                gop_index = Some(s.gops.len() - 1);
            }
            PICTURE_START => {
                let mut b = Bits::new(body);
                let tr = b.read(10) as u16;
                let kind = match b.read(3) {
                    1 => 'I',
                    2 => 'P',
                    3 => 'B',
                    _ => '?',
                };
                // The picture runs until the next picture, GOP, sequence
                // header or sequence end.
                let end = codes[ci + 1..]
                    .iter()
                    .find(|&&(_, c)| matches!(c, PICTURE_START | GOP_START | SEQUENCE_HEADER | SEQUENCE_END))
                    .map(|&(o, _)| o)
                    .unwrap_or(es.len());
                let gop = match gop_index {
                    Some(g) => g,
                    None => {
                        s.gops.push(Gop { closed: false, broken_link: false, pictures: 0 });
                        gop_index = Some(0);
                        0
                    }
                };
                s.gops[gop].pictures += 1;
                s.pictures.push(Picture {
                    offset: off,
                    size: end - off,
                    kind,
                    temporal_reference: tr,
                    gop,
                    coding_ext: None,
                    top_field_first: false,
                    repeat_first_field: false,
                    progressive_frame: false,
                    picture_structure: 3,
                    display_index: 0,
                });
            }
            _ => {}
        }
    }
    if s.seq_offsets.is_empty() {
        bail!("no MPEG-2 sequence header found");
    }

    // Display order: temporal_reference counts within each GOP.
    let mut base = 0usize;
    let mut gop_base = vec![0usize; s.gops.len()];
    for (g, gop) in s.gops.iter().enumerate() {
        gop_base[g] = base;
        base += gop.pictures;
    }
    for p in s.pictures.iter_mut() {
        p.display_index = gop_base[p.gop] + p.temporal_reference as usize;
    }
    Ok(s)
}

impl Stream {
    /// Picture types in display order.
    pub fn display_types(&self) -> Vec<char> {
        let mut t = vec!['?'; self.pictures.len()];
        for p in &self.pictures {
            if p.display_index < t.len() {
                t[p.display_index] = p.kind;
            }
        }
        t
    }

    /// Pictures sorted by display order.
    pub fn display_order(&self) -> Vec<&Picture> {
        let mut v: Vec<&Picture> = self.pictures.iter().collect();
        v.sort_by_key(|p| p.display_index);
        v
    }
}

/// Turn a 23.976 fps progressive stream into a soft-telecined 29.97 fps
/// stream, as DVD film titles are stored: the sequence header's frame
/// rate becomes 30000/1001, progressive_sequence becomes 0, and every
/// picture is marked progressive_frame = 1 with the 2:3 field pattern in
/// display order:
///
/// | display frame | top_field_first | repeat_first_field | fields |
/// |---|---|---|---|
/// | 4n     | 1 | 1 | T B T |
/// | 4n + 1 | 0 | 0 | B T   |
/// | 4n + 2 | 0 | 1 | B T B |
/// | 4n + 3 | 1 | 0 | T B   |
///
/// The flags are patched in place; nothing else in the stream changes.
pub fn apply_soft_pulldown(es: &mut [u8]) -> Result<Stream> {
    let s = parse(es)?;
    if s.seq.frame_rate() != Some((24000, 1001)) {
        bail!("soft pulldown needs a 23.976 fps stream (frame_rate_code {})", s.seq.frame_rate_code);
    }
    for &off in &s.seq_offsets {
        // Byte 3 after the start code: aspect (4) | frame_rate_code (4).
        let b = off + 4 + 3;
        es[b] = (es[b] & 0xF0) | 4;
    }
    for &off in &s.seq_ext_offsets {
        // Byte 1 after the start code: profile_and_level low nibble (4) |
        // progressive_sequence (1) | chroma_format (2) | h_ext high (1).
        es[off + 5] &= !0x08;
    }
    for p in &s.pictures {
        let Some(off) = p.coding_ext else { bail!("picture without coding extension") };
        let (tff, rff) = match p.display_index % 4 {
            0 => (true, true),
            1 => (false, false),
            2 => (false, true),
            _ => (true, false),
        };
        // Byte 3: top_field_first (bit 7) .. repeat_first_field (bit 1),
        // chroma_420_type (bit 0).  Byte 4: progressive_frame (bit 7).
        let b3 = off + 4 + 3;
        es[b3] = (es[b3] & !0x82) | if tff { 0x80 } else { 0 } | if rff { 0x02 } else { 0 };
        es[b3] |= 0x01; // chroma_420_type = progressive_frame
        es[off + 4 + 4] |= 0x80;
    }
    parse(es)
}

/// Results of checking a stream against the DVD-Video (NTSC) limits.
#[derive(Clone, Debug)]
pub struct Analysis {
    pub stream: Stream,
    pub pictures: usize,
    pub max_gop: usize,
    pub all_closed: bool,
    /// Peak bit rate over any one-second window (bit/s).
    pub peak_bitrate_1s: f64,
    pub mean_bitrate: f64,
    /// Fields per picture in display order, from the RFF flags.
    pub field_counts: (usize, usize),
    pub checks: Vec<(String, String, bool)>,
}

/// Check a stream against the DVD-Video NTSC video limits: MPEG-2
/// MP@ML 4:2:0, 720/704/352 × 480/240, 29.97 fps (23.976 only via soft
/// pulldown), GOP ≤ 18 pictures, VBV buffer ≤ 224 KiB (1 835 008 bits),
/// ≤ 9.8 Mbit/s peak.
///
/// The peak rate is measured over every window of consecutive pictures
/// (in coded order) that lasts one second of display time, using the
/// field counts given by the RFF flags.
pub fn analyze(es: &[u8]) -> Result<Analysis> {
    let stream = parse(es)?;
    let seq = &stream.seq;
    let pics = &stream.pictures;
    let max_gop = stream.gops.iter().map(|g| g.pictures).max().unwrap_or(0);
    let all_closed = stream.gops.iter().all(|g| g.closed);

    // Display duration of each picture in seconds.
    let (rn, rd) = seq.frame_rate().unwrap_or((30000, 1001));
    let frame_time = rd as f64 / rn as f64;
    let duration = |p: &Picture| -> f64 {
        if seq.progressive_sequence {
            // Progressive sequence: RFF repeats the whole frame (twice
            // more with top_field_first set).
            frame_time * match (p.repeat_first_field, p.top_field_first) {
                (false, _) => 1.0,
                (true, false) => 2.0,
                (true, true) => 3.0,
            }
        } else if p.picture_structure != 3 {
            frame_time / 2.0
        } else if p.repeat_first_field {
            frame_time * 1.5
        } else {
            frame_time
        }
    };
    let durations: Vec<f64> = pics.iter().map(duration).collect();
    let mut peak = 0.0f64;
    let mut j = 0usize;
    let mut bits = 0.0f64;
    let mut time = 0.0f64;
    for i in 0..pics.len() {
        while j < pics.len() && time < 1.0 - 1e-9 {
            bits += pics[j].size as f64 * 8.0;
            time += durations[j];
            j += 1;
        }
        if time >= 1.0 - 1e-9 {
            peak = peak.max(bits / time);
        }
        bits -= pics[i].size as f64 * 8.0;
        time -= durations[i];
    }
    let total_bits: f64 = pics.iter().map(|p| p.size as f64 * 8.0).sum();
    let total_time: f64 = durations.iter().sum();
    let mean = if total_time > 0.0 { total_bits / total_time } else { 0.0 };
    if peak == 0.0 {
        peak = mean;
    }
    let three = pics.iter().filter(|p| p.repeat_first_field).count();

    let mut checks = Vec::new();
    let mut check = |name: &str, value: String, ok: bool| checks.push((name.to_string(), value, ok));
    check("profile/level", seq.profile_level_name(), seq.profile_and_level == 0x48);
    check("chroma format", match seq.chroma_format { 1 => "4:2:0", 2 => "4:2:2", 3 => "4:4:4", _ => "?" }.into(), seq.chroma_format == 1);
    let size_ok = matches!((seq.width, seq.height), (720, 480) | (704, 480) | (352, 480) | (352, 240));
    check("frame size", format!("{}x{}", seq.width, seq.height), size_ok);
    let rate = seq.frame_rate().map(|(n, d)| format!("{:.3} fps", n as f64 / d as f64)).unwrap_or_else(|| "invalid".into());
    let film = !pics.is_empty() && pics.iter().all(|p| p.progressive_frame) && three > 0;
    check(
        "frame rate",
        format!("{}{}", rate, if film { " (soft pulldown, 23.976 fps coded)" } else { "" }),
        seq.frame_rate() == Some((30000, 1001)),
    );
    check("display aspect", seq.display_aspect().into(), matches!(seq.aspect_code, 2 | 3));
    check("max GOP length", format!("{} pictures", max_gop), max_gop <= 18 && max_gop > 0);
    check("closed GOPs", if all_closed { "all closed".into() } else { "some open".to_string() }, true);
    check(
        "VBV buffer",
        format!("{} bits ({} KiB)", seq.vbv_size_16k * 16384, seq.vbv_size_16k * 2),
        seq.vbv_size_16k * 16384 <= 1_835_008 && seq.vbv_size_16k > 0,
    );
    check(
        "header bit rate",
        format!("{:.3} Mbit/s", seq.bit_rate_400 as f64 * 400.0 / 1e6),
        seq.bit_rate_400 as f64 * 400.0 <= 9.8e6 + 1.0,
    );
    check("peak bit rate (1 s)", format!("{:.3} Mbit/s", peak / 1e6), peak <= 9.8e6);
    check("mean bit rate", format!("{:.3} Mbit/s", mean / 1e6), true);
    if let Some(p) = pics.first() {
        check(
            "field order",
            if film { "progressive frames, 2:3 pulldown flags".into() } else if seq.progressive_sequence { "progressive".to_string() } else if p.top_field_first { "top field first".into() } else { "bottom field first".into() },
            true,
        );
    }
    Ok(Analysis {
        pictures: pics.len(),
        max_gop,
        all_closed,
        peak_bitrate_1s: peak,
        mean_bitrate: mean,
        field_counts: (three, pics.len() - three),
        checks,
        stream,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal synthetic stream: sequence header + extension, one closed
    /// GOP of I B B P in coded order (display I0 B1 B2 P3 → coded I P B B).
    fn synthetic() -> Vec<u8> {
        let mut v = Vec::new();
        // Sequence header: 720x480, aspect 2, frame_rate_code 1 (23.976),
        // bit_rate 0x2EE (300 kbit/s units of 400), marker, vbv 112, flags 0.
        v.extend([0, 0, 1, SEQUENCE_HEADER]);
        let mut bits: Vec<(u32, u32)> = vec![(720, 12), (480, 12), (2, 4), (1, 4), (0x2EE, 18), (1, 1), (112, 10), (0, 1), (0, 1), (0, 1)];
        v.extend(pack(&mut bits));
        // Sequence extension: id 1, MP@ML 0x48, progressive 1, 4:2:0.
        v.extend([0, 0, 1, EXTENSION_START]);
        let mut bits = vec![(1, 4), (0x48, 8), (1, 1), (1, 2), (0, 2), (0, 2), (0, 12), (1, 1), (0, 8), (0, 1), (0, 2), (0, 5)];
        v.extend(pack(&mut bits));
        // GOP header: closed.
        v.extend([0, 0, 1, GOP_START]);
        let mut bits = vec![(0, 25), (1, 1), (0, 1), (0, 5)];
        v.extend(pack(&mut bits));
        for (tr, ty) in [(0u32, 1u32), (3, 2), (1, 3), (2, 3)] {
            v.extend([0, 0, 1, PICTURE_START]);
            let mut bits = vec![(tr, 10), (ty, 3), (0xFFFF, 16), (0, 3)];
            v.extend(pack(&mut bits));
            v.extend([0, 0, 1, EXTENSION_START]);
            // id 8, f_codes, dc prec 0, structure 3, tff 0, 5 bits, rff 0, c420 0, prog 1, composite 0.
            let mut bits = vec![(8, 4), (0xFFFF, 16), (0, 2), (3, 2), (0, 1), (0b10000, 5), (0, 1), (0, 1), (1, 1), (0, 1), (0, 6)];
            v.extend(pack(&mut bits));
            v.extend([0, 0, 1, 0x01, 0xAA, 0xBB]); // a slice
        }
        v.extend([0, 0, 1, SEQUENCE_END]);
        v
    }

    fn pack(fields: &mut Vec<(u32, u32)>) -> Vec<u8> {
        let mut out = Vec::new();
        let mut acc = 0u64;
        let mut n = 0u32;
        for &(val, len) in fields.iter() {
            acc = (acc << len) | (val as u64 & ((1u64 << len) - 1));
            n += len;
            while n >= 8 {
                out.push((acc >> (n - 8)) as u8);
                n -= 8;
            }
        }
        if n > 0 {
            out.push((acc << (8 - n)) as u8);
        }
        out
    }

    #[test]
    fn parses_headers_and_display_order() {
        let s = parse(&synthetic()).unwrap();
        assert_eq!((s.seq.width, s.seq.height), (720, 480));
        assert_eq!(s.seq.frame_rate(), Some((24000, 1001)));
        assert_eq!(s.seq.profile_and_level, 0x48);
        assert_eq!(s.seq.vbv_size_16k, 112);
        assert_eq!(s.gops.len(), 1);
        assert!(s.gops[0].closed);
        assert_eq!(s.display_types(), vec!['I', 'B', 'B', 'P']);
    }

    #[test]
    fn soft_pulldown_sets_2_3_pattern() {
        let mut es = synthetic();
        let s = apply_soft_pulldown(&mut es).unwrap();
        assert_eq!(s.seq.frame_rate(), Some((30000, 1001)));
        assert!(!s.seq.progressive_sequence);
        let flags: Vec<(bool, bool, bool)> = s
            .display_order()
            .iter()
            .map(|p| (p.top_field_first, p.repeat_first_field, p.progressive_frame))
            .collect();
        assert_eq!(flags, vec![(true, true, true), (false, false, true), (false, true, true), (true, false, true)]);
        // Nothing but the flags changed.
        let orig = synthetic();
        let diffs = orig.iter().zip(&es).filter(|(a, b)| a != b).count();
        assert!(diffs <= 2 + 4);
    }
}
