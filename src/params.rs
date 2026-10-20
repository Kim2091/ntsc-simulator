//! Shared parameter model for the command line and the GUI.
//!
//! Every effect parameter is described once in [`SPECS`]: its flag name,
//! label, help text, unit, slider range, default and group.  The CLI builds
//! its flags from this table and the GUI builds its controls from it, so a
//! new parameter only needs an entry here (plus its use in
//! [`ParamValues::to_signal_effects`]).

use std::collections::BTreeMap;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::effects::{
    splitmix64, Doc, GhostConfig, HeadSwitchAfter, SignalEffects, TapeModel, DEFAULT_HEAD_SWITCH_LINES,
};

// ────────────────────────────────────────────────────────────────────────────
// Range parameters
// ────────────────────────────────────────────────────────────────────────────

/// A parameter value that can be either a single number (e.g. `0.05`) or a
/// `min..max` range (e.g. `0.02..0.10`).  When a range is given, a random
/// value is sampled uniformly for each file processed in a batch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RangeParam {
    pub min: f32,
    pub max: f32,
}

impl RangeParam {
    /// A single fixed value.
    pub fn fixed(v: f32) -> Self {
        Self { min: v, max: v }
    }

    /// Sample a value — returns `min` if fixed, otherwise uniform in [min, max].
    pub fn sample<R: Rng>(self, rng: &mut R) -> f32 {
        if (self.max - self.min).abs() < f32::EPSILON {
            self.min
        } else {
            rng.random_range(self.min..=self.max)
        }
    }

    /// Sample from an `Option<RangeParam>`.
    pub fn sample_opt<R: Rng>(opt: Option<RangeParam>, rng: &mut R) -> Option<f32> {
        opt.map(|r| r.sample(rng))
    }

    /// `true` if this is a genuine range (min ≠ max).
    pub fn is_range(self) -> bool {
        (self.max - self.min).abs() > f32::EPSILON
    }
}

impl std::str::FromStr for RangeParam {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let num = |t: &str| t.trim().parse::<f32>().map_err(|e| format!("'{}': {}", t.trim(), e));
        if let Some((a, b)) = s.split_once("..") {
            let (min, max) = (num(a)?, num(b)?);
            if max < min {
                return Err(format!("range max ({}) must be >= min ({})", max, min));
            }
            Ok(RangeParam { min, max })
        } else {
            Ok(RangeParam::fixed(num(s)?))
        }
    }
}

impl std::fmt::Display for RangeParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_range() {
            write!(f, "{}..{}", self.min, self.max)
        } else {
            write!(f, "{}", self.min)
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Parameter table
// ────────────────────────────────────────────────────────────────────────────

/// Groups in signal-chain order.  The GUI shows one collapsible section per
/// group and the CLI help uses them as headings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Group {
    Dvd,
    Tape,
    Colour,
    NoiseDropouts,
    Timing,
    Reception,
    Deck,
    Output,
}

impl Group {
    pub const ALL: [Group; 8] = [
        Group::Dvd,
        Group::Tape,
        Group::Colour,
        Group::NoiseDropouts,
        Group::Timing,
        Group::Reception,
        Group::Deck,
        Group::Output,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Group::Dvd => "DVD",
            Group::Tape => "Tape",
            Group::Colour => "Colour",
            Group::NoiseDropouts => "Noise & dropouts",
            Group::Timing => "Timing",
            Group::Reception => "Reception",
            Group::Deck => "Playback deck / capture",
            Group::Output => "Input / decoder / output",
        }
    }
}

/// What kind of value a parameter takes.
#[derive(Clone, Copy, Debug)]
pub enum Kind {
    /// A number or `min..max` range.  `min`/`max` are the slider limits
    /// (the CLI accepts any value).  `default` applies when the parameter
    /// is not set; `None` means the effect is off.  `typical` is a sensible
    /// starting value when the user turns the effect on.  `integer` values
    /// are rounded where they are used.
    Number { min: f32, max: f32, default: Option<f32>, typical: f32, integer: bool },
    /// An on/off switch.
    Flag,
    /// Free text.
    Text,
    /// One of a fixed list of options; `default` applies when unset.
    Choice { options: &'static [&'static str], default: &'static str },
}

/// Description of one parameter, shared by the CLI and the GUI.
#[derive(Debug)]
pub struct ParamSpec {
    /// Flag name without dashes, e.g. `vhs-luma-bw`.
    pub id: &'static str,
    /// Short label for the GUI.
    pub label: &'static str,
    /// What the parameter does and what it looks like (CLI help, GUI tooltip).
    pub help: &'static str,
    /// Unit shown next to the value, or "".
    pub unit: &'static str,
    pub group: Group,
    pub kind: Kind,
}

const fn num(min: f32, max: f32, typical: f32) -> Kind {
    Kind::Number { min, max, default: None, typical, integer: false }
}

const fn num_def(min: f32, max: f32, default: f32) -> Kind {
    Kind::Number { min, max, default: Some(default), typical: default, integer: false }
}

/// All effect parameters, in signal-chain order.
pub static SPECS: &[ParamSpec] = &[
    // ── DVD ──
    ParamSpec {
        id: "format",
        label: "Format",
        help: "Signal path. analog: the NTSC composite / VHS simulation. dvd: DVD-Video, i.e. mastering, MPEG-2 \
               encoding within the DVD-Video limits (via ffmpeg), player decode and player output. Tape and \
               reception effects only apply on the DVD path when set explicitly and the player output is analog.",
        unit: "",
        group: Group::Dvd,
        kind: Kind::Choice { options: &["analog", "dvd"], default: "analog" },
    },
    ParamSpec {
        id: "dvd-size",
        label: "Frame size",
        help: "Coded frame size. 720x480 (full D1, most discs), 704x480, 352x480 (half D1, long-play discs: soft), \
               352x240 (SIF, progressive, VCD-like quality).",
        unit: "",
        group: Group::Dvd,
        kind: Kind::Choice { options: &["720x480", "704x480", "352x480", "352x240"], default: "720x480" },
    },
    ParamSpec {
        id: "dvd-aspect",
        label: "Aspect",
        help: "Display aspect flag: 4:3, or 16:9 anamorphic (the picture is squeezed horizontally on disc). Sources \
               with another aspect are letterboxed or pillarboxed.",
        unit: "",
        group: Group::Dvd,
        kind: Kind::Choice { options: &["4:3", "16:9"], default: "4:3" },
    },
    ParamSpec {
        id: "dvd-rate",
        label: "Frame structure",
        help: "interlaced: 29.97 fps interlaced, top field first (video-sourced discs). film: 23.976 fps progressive \
               with soft 3:2 pulldown flags (repeat_first_field), as film titles are stored.",
        unit: "",
        group: Group::Dvd,
        kind: Kind::Choice { options: &["interlaced", "film"], default: "interlaced" },
    },
    ParamSpec {
        id: "dvd-bitrate",
        label: "Average bit rate",
        help: "Average video bit rate in Mbit/s (VBR). Typical: 7–8 high quality film, 5–6 average, 3–4 budget / \
               long play. Lower rates show blocking, mosquito noise around edges and smeared texture.",
        unit: "Mbit/s",
        group: Group::Dvd,
        kind: num_def(1.0, 9.8, 6.0),
    },
    ParamSpec {
        id: "dvd-maxrate",
        label: "Peak bit rate",
        help: "Maximum video bit rate in Mbit/s (DVD-Video allows at most 9.8). Lower peaks starve complex scenes.",
        unit: "Mbit/s",
        group: Group::Dvd,
        kind: num_def(2.0, 9.8, 9.5),
    },
    ParamSpec {
        id: "dvd-gop",
        label: "GOP length",
        help: "Maximum pictures per GOP (DVD allows at most 18). GOPs are closed, with two B pictures between \
               reference pictures (M = 3). ffmpeg ends each closed GOP on a P picture, so the actual length is the \
               largest 3k+1 at or below this value: 15 gives 13, 16 to 18 give 16.",
        unit: "pictures",
        group: Group::Dvd,
        kind: Kind::Number { min: 1.0, max: 18.0, default: Some(15.0), typical: 15.0, integer: true },
    },
    ParamSpec {
        id: "dvd-matrix",
        label: "Quant matrices",
        help: "standard: the MPEG-2 default quantiser matrices. lowrate: steeper matrices that spend fewer bits on \
               fine detail (softer picture, fewer blocks at low rates).",
        unit: "",
        group: Group::Dvd,
        kind: Kind::Choice { options: &["standard", "lowrate"], default: "standard" },
    },
    ParamSpec {
        id: "dvd-2pass",
        label: "Two-pass encode",
        help: "Two-pass VBR: distributes bits better over the whole clip (slower; spools frames to a temporary file).",
        unit: "",
        group: Group::Dvd,
        kind: Kind::Flag,
    },
    ParamSpec {
        id: "dvd-ee",
        label: "Edge enhancement",
        help: "Mastering edge enhancement (0–1): bright and dark outlines several pixels wide around edges, the \
               classic DVD halo. Wider and softer than VHS peaking.",
        unit: "",
        group: Group::Dvd,
        kind: num(0.0, 1.0, 0.5),
    },
    ParamSpec {
        id: "dvd-dnr",
        label: "Mastering DNR",
        help: "Mastering noise reduction (0–1): edge-preserving smoothing that wipes out fine grain and texture \
               (waxy faces, flat backgrounds).",
        unit: "",
        group: Group::Dvd,
        kind: num(0.0, 1.0, 0.5),
    },
    ParamSpec {
        id: "dvd-prefilter",
        label: "Pre-filter",
        help: "Pre-compression lowpass on luma (0–1), cutting horizontal detail from ~6 MHz down to ~3.8 MHz, as \
               some studios did to save bits.",
        unit: "",
        group: Group::Dvd,
        kind: num(0.0, 1.0, 0.5),
    },
    ParamSpec {
        id: "dvd-cue",
        label: "Chroma upsampling error",
        help: "Player chroma upsampling error (0–1): many players upsampled interlaced 4:2:0 colour as if it were \
               progressive, giving jagged, streaky colour edges on interlaced material.",
        unit: "",
        group: Group::Dvd,
        kind: num(0.0, 1.0, 1.0),
    },
    ParamSpec {
        id: "dvd-output",
        label: "Player output",
        help: "digital: the decoded frames as they are (like a rip). svideo / composite: the player's analog output \
               captured through the NTSC decoder (composite adds cross-colour and dot crawl).",
        unit: "",
        group: Group::Dvd,
        kind: Kind::Choice { options: &["digital", "svideo", "composite"], default: "digital" },
    },
    ParamSpec {
        id: "dvd-output-size",
        label: "Output pixels",
        help: "native: the stored frame (e.g. 720x480 with non-square pixels). square: the active area scaled to \
               square pixels (640x480 for 4:3, 854x480 for 16:9).",
        unit: "",
        group: Group::Dvd,
        kind: Kind::Choice { options: &["native", "square"], default: "native" },
    },
    ParamSpec {
        id: "dvd-frame-type",
        label: "Still frame type",
        help: "For still images: which decoded picture type to return. A still is encoded as a short sequence; b \
               (default) is the most representative, since two thirds of the pictures on a DVD are B pictures with \
               the coarsest quantisation.",
        unit: "",
        group: Group::Dvd,
        kind: Kind::Choice { options: &["b", "p", "i"], default: "b" },
    },
    ParamSpec {
        id: "dvd-keep-mpeg2",
        label: "Keep MPEG-2 stream",
        help: "Also save the spec-compliant MPEG-2 stream: .m2v as an elementary stream, .vob/.mpg as a DVD program \
               stream.",
        unit: "",
        group: Group::Dvd,
        kind: Kind::Text,
    },
    // ── Tape ──
    ParamSpec {
        id: "tape-model",
        label: "Tape model",
        help: "How the VHS tape path is simulated. classic: the original model, a lowpass on luma and colour \
               (legacy output). fm: the record and playback circuits of the format: the recording deck's \
               band-filter luma/chroma split (fine luma near the subcarrier is recorded as cross-colour), the \
               record and playback luma lowpass filters (−10 dB at Luma bandwidth, causal like the analog \
               circuits) and the colour-under filter. Opt-in; the newer tape stages need it.",
        unit: "",
        group: Group::Tape,
        kind: Kind::Choice { options: &["classic", "fm"], default: "classic" },
    },
    ParamSpec {
        id: "vhs-luma-bw",
        label: "Luma bandwidth",
        help: "VHS luma bandwidth in MHz (SP ≈ 3.0, EP/SLP ≈ 1.6). Lower values blur fine horizontal detail.",
        unit: "MHz",
        group: Group::Tape,
        kind: num(0.5, 4.2, 3.0),
    },
    ParamSpec {
        id: "color-under-bw",
        label: "Colour-under bandwidth",
        help: "Colour-under chroma bandwidth in kHz (typical 300–500). Lower values smear colour sideways past edges.",
        unit: "kHz",
        group: Group::Tape,
        kind: num(100.0, 1000.0, 400.0),
    },
    ParamSpec {
        id: "edge-ringing",
        label: "Source enhancement",
        help: "Edge enhancement that is already on the tape (0.5–3.0): ringing and halos from the source chain \
               (broadcast filters, telecine aperture correction, duplicator enhancers). Adds bright and dark halos \
               beside sharp edges. The deck's own sharpening is Deck sharpness.",
        unit: "",
        group: Group::Tape,
        kind: num(0.0, 3.0, 0.3),
    },
    ParamSpec {
        id: "head-smear",
        label: "Head-wear smear",
        help: "Worn-head symmetric luma blur (0.3–1.0). Soft horizontal smudging that varies in bands down the frame.",
        unit: "",
        group: Group::Tape,
        kind: num(0.0, 1.0, 0.2),
    },
    ParamSpec {
        id: "tape-trail",
        label: "Tape trail",
        help: "Causal rightward luma smear (0.3–1.0). Edges and noise leave a comet tail to the right, the classic worn-tape look.",
        unit: "",
        group: Group::Tape,
        kind: num(0.0, 1.0, 0.6),
    },
    // ── Colour ──
    ParamSpec {
        id: "chroma-noise",
        label: "Chroma noise",
        help: "Colour-under chroma noise, std dev of I/Q noise (subtle 0.01–0.03, heavy 0.06). Horizontal colour blotches in flat areas.",
        unit: "",
        group: Group::Colour,
        kind: num(0.0, 0.1, 0.02),
    },
    ParamSpec {
        id: "chroma-phase-noise",
        label: "Chroma phase jitter",
        help: "Per-line hue jitter, std dev in degrees (subtle 1–3). Lines flicker slightly in hue.",
        unit: "°",
        group: Group::Colour,
        kind: num(0.0, 15.0, 2.0),
    },
    ParamSpec {
        id: "chroma-vblur",
        label: "Chroma vertical blur",
        help: "Playback 1H chroma comb (0–1). Colour is mixed with the line above, softening colour edges vertically.",
        unit: "",
        group: Group::Colour,
        kind: num(0.0, 1.0, 0.6),
    },
    ParamSpec {
        id: "chroma-delay",
        label: "Chroma delay",
        help: "Horizontal chroma delay relative to luma in output pixels at 640 wide (typical 1–4, positive = right). Colour sits beside its object.",
        unit: "px",
        group: Group::Colour,
        kind: num(-4.0, 8.0, 2.0),
    },
    ParamSpec {
        id: "chroma-delay-v",
        label: "Chroma delay (vertical)",
        help: "Vertical chroma delay in field lines (typical 0–1; 1 field line = 2 output rows). Colour sits below its object.",
        unit: "lines",
        group: Group::Colour,
        kind: Kind::Number { min: 0.0, max: 3.0, default: None, typical: 1.0, integer: true },
    },
    // ── Noise & dropouts ──
    ParamSpec {
        id: "luma-noise",
        label: "Luma noise",
        help: "Luminance-dependent tape noise amplitude (0.02–0.10). Grainy noise, stronger in dark areas.",
        unit: "",
        group: Group::NoiseDropouts,
        kind: num(0.0, 0.1, 0.025),
    },
    ParamSpec {
        id: "tape-dropout-rate",
        label: "Dropouts",
        help: "Average tape dropouts per frame (2–20). Short white streaks or repeated line segments.",
        unit: "/frame",
        group: Group::NoiseDropouts,
        kind: num(0.0, 30.0, 2.0),
    },
    ParamSpec {
        id: "tape-dropout-len",
        label: "Dropout length",
        help: "Average dropout length in microseconds (default 15).",
        unit: "µs",
        group: Group::NoiseDropouts,
        kind: num_def(1.0, 60.0, 15.0),
    },
    ParamSpec {
        id: "doc",
        label: "Dropout compensator",
        help: "What the deck does with long dropouts. composite: copies the whole line above, colour included \
               (the original behaviour; the copied colour comes out in the complementary hue). luma: copies the \
               luma of the line above and mutes the colour, as real decks do, so patches are colourless. off: no \
               compensator, every dropout is a white or black streak.",
        unit: "",
        group: Group::NoiseDropouts,
        kind: Kind::Choice { options: &["composite", "luma", "off"], default: "composite" },
    },
    // ── Timing ──
    ParamSpec {
        id: "jitter",
        label: "Line jitter",
        help: "Random horizontal line jitter, std dev in subcarrier cycles (0.1–1.0). Lines jump sideways independently.",
        unit: "cycles",
        group: Group::Timing,
        kind: num(0.0, 3.0, 0.3),
    },
    ParamSpec {
        id: "wobble",
        label: "Wobble",
        help: "Time-base wobble: peak horizontal displacement in output pixels (typical 0.5–3). Vertical edges bend in slow waves.",
        unit: "px",
        group: Group::Timing,
        kind: num(0.0, 6.0, 1.0),
    },
    ParamSpec {
        id: "wobble-rate",
        label: "Wobble rate",
        help: "Wobble frequency scale (default 1.0 = periods of ~20–300 lines; higher = faster wobble).",
        unit: "×",
        group: Group::Timing,
        kind: num_def(0.1, 4.0, 1.0),
    },
    ParamSpec {
        id: "v-jitter",
        label: "Vertical bounce",
        help: "Vertical bounce: the picture jumps up and down by whole lines from frame to frame (std dev in output \
               rows), from tape tension and servo errors. A frame TBC removes it.",
        unit: "rows",
        group: Group::Timing,
        kind: num(0.0, 4.0, 1.0),
    },
    ParamSpec {
        id: "head-switch",
        label: "Head switching",
        help: "Head-switching noise (0–1). The last lines of each field tear: a displaced switch line, then near-black lines with bright noise streaks. Stable over a tape, flickering, with occasional sudden jumps.",
        unit: "",
        group: Group::Timing,
        kind: num(0.0, 1.0, 0.6),
    },
    ParamSpec {
        id: "head-switch-lines",
        label: "Head-switch lines",
        help: "Active lines per field torn by head switching at full strength (default 3, i.e. 6 rows at 480).",
        unit: "lines",
        group: Group::Timing,
        kind: Kind::Number { min: 1.0, max: 20.0, default: None, typical: 3.0, integer: true },
    },
    ParamSpec {
        id: "head-switch-point",
        label: "Head-switch point",
        help: "Opt-in. Where the heads switch, in lines before vertical sync (the format allows 6.5 ± 1.5). Sets \
               how many lines at the bottom are torn, about 2·point − 7 rows of a 480-line capture (5-7 rows at \
               6.5, 2-10 over the format's range), independent of the head-switching strength. Replaces \
               Head-switch lines.",
        unit: "H",
        group: Group::Timing,
        kind: num(4.0, 9.0, 6.5),
    },
    ParamSpec {
        id: "head-switch-after",
        label: "After the switch",
        help: "What the lines after the head switch show. black: signal lost, near black with light streaks (as \
               on many worn tapes). picture: the picture, shifted sideways by the switch's time-base jump with a \
               small level step (the common case on healthy tapes). A line TBC straightens the shifted lines but \
               not the switching noise.",
        unit: "",
        group: Group::Timing,
        kind: Kind::Choice { options: &["black", "picture"], default: "black" },
    },
    ParamSpec {
        id: "flagging",
        label: "Top skew (flagging)",
        help: "Opt-in. Top-of-field skew (0–1): the head switch's time-base step carried into the next field by a \
               slow horizontal PLL in the TV or capture card, so the top lines bend sideways and straighten out \
               over about 30 lines, the two fields in opposite directions. 1 = about 1 µs (12 px at 640 wide) at the \
               top line. A line TBC removes it.",
        unit: "",
        group: Group::Timing,
        kind: num(0.0, 1.0, 0.5),
    },
    ParamSpec {
        id: "top-band",
        label: "Top colour band",
        help: "Faint blue/purple bands on the first lines of each field (0–1), as the deck's chroma loop settles.",
        unit: "",
        group: Group::Timing,
        kind: num(0.0, 1.0, 0.4),
    },
    ParamSpec {
        id: "vbi-data",
        label: "Caption data line",
        help: "Closed-caption data on lines 21/284 (0–1). A row of short light dashes near the top of the picture.",
        unit: "",
        group: Group::Timing,
        kind: num(0.0, 1.0, 0.5),
    },
    // ── Reception ──
    ParamSpec {
        id: "noise",
        label: "Snow",
        help: "Gaussian RF noise amplitude (0.05 = subtle, 0.2 = heavy snow).",
        unit: "",
        group: Group::Reception,
        kind: num(0.0, 0.3, 0.05),
    },
    ParamSpec {
        id: "ghost",
        label: "Ghost",
        help: "Primary multipath ghost amplitude (0–1). A faint delayed copy of the picture to the right.",
        unit: "",
        group: Group::Reception,
        kind: num(0.0, 1.0, 0.2),
    },
    ParamSpec {
        id: "ghost-delay",
        label: "Ghost delay",
        help: "Primary ghost delay in microseconds (default 2.0). Larger delays move the ghost further right.",
        unit: "µs",
        group: Group::Reception,
        kind: num_def(0.0, 20.0, 2.0),
    },
    ParamSpec {
        id: "ghost-phase",
        label: "Ghost phase",
        help: "Ghost phase shift in degrees; 180 = polarity inversion, a negative ghost (default 0).",
        unit: "°",
        group: Group::Reception,
        kind: num_def(0.0, 360.0, 0.0),
    },
    ParamSpec {
        id: "ghost-rolloff-mhz",
        label: "Ghost softness",
        help: "Ghost HF rolloff cutoff in MHz (default 3.0). Lower values make the ghost softer.",
        unit: "MHz",
        group: Group::Reception,
        kind: num(0.5, 4.2, 3.0),
    },
    ParamSpec {
        id: "ghost-dynamic",
        label: "Ghost drifts",
        help: "Slowly vary ghost amplitude over time (environmental drift).",
        unit: "",
        group: Group::Reception,
        kind: Kind::Flag,
    },
    ParamSpec {
        id: "ghost-dynamic-rate",
        label: "Ghost drift rate",
        help: "Dynamic ghost modulation rate in Hz (default 0.5).",
        unit: "Hz",
        group: Group::Reception,
        kind: num_def(0.05, 5.0, 0.5),
    },
    ParamSpec {
        id: "ghost-multi",
        label: "Extra ghosts",
        help: "Additional ghosts as \"amp,delay_us,phase_deg\" triples separated by semicolons; each value may be a min..max range. Example: \"0.1..0.3,2.0..5.0,0..180;0.08,7.5,0\"",
        unit: "",
        group: Group::Reception,
        kind: Kind::Text,
    },
    ParamSpec {
        id: "attenuation",
        label: "Attenuation",
        help: "Signal attenuation (0–1). Lowers contrast and saturation toward blanking.",
        unit: "",
        group: Group::Reception,
        kind: num(0.0, 1.0, 0.1),
    },
    // ── Playback deck / capture ──
    ParamSpec {
        id: "line-tbc",
        label: "Line TBC",
        help: "Line time-base corrector (0–1): re-times every line from its sync and burst, removing most jitter and \
               wobble and much of the head-switching displacement (the switching noise line stays). Like high-end \
               S-VHS decks with a built-in TBC.",
        unit: "",
        group: Group::Deck,
        kind: num(0.0, 1.0, 1.0),
    },
    ParamSpec {
        id: "frame-tbc",
        label: "Frame TBC",
        help: "Frame TBC / frame synchroniser (0–1): holds the picture vertically steady, removing vertical bounce.",
        unit: "",
        group: Group::Deck,
        kind: num(0.0, 1.0, 1.0),
    },
    ParamSpec {
        id: "deck-dnr",
        label: "Deck noise reduction",
        help: "Deck recursive noise reduction (0–1): averages each frame with the previous output where the picture \
               is still, which cleans noise but leaves smeared ghost trails behind moving objects. Needs \
               consecutive frames: no effect on single images; the GUI's video preview renders the previous 8 \
               frames so it shows.",
        unit: "",
        group: Group::Deck,
        kind: num(0.0, 1.0, 0.5),
    },
    ParamSpec {
        id: "deck-nc",
        label: "Noise canceller",
        help: "Opt-in. The deck's luma noise canceller (coring): high-frequency detail smaller than this many IRE \
               (typical 4–10) is removed together with the noise. Flat areas look clean and slightly waxy, fine \
               low-contrast texture (skin, fabric, grass) disappears, strong edges keep their noise. Nearly every \
               VCR from the mid-80s has one.",
        unit: "IRE",
        group: Group::Deck,
        kind: num(0.0, 15.0, 7.0),
    },
    ParamSpec {
        id: "deck-ynr",
        label: "Line noise reducer",
        help: "Opt-in. HQ decks' recursive line noise reducer (YNR), 0–1: each line is averaged with the line above \
               where they differ by less than 6 IRE, so noise and faint detail are smoothed vertically (a slight \
               vertical smear below small features) while strong edges pass. Not every HQ deck has it.",
        unit: "",
        group: Group::Deck,
        kind: num(0.0, 1.0, 1.0),
    },
    ParamSpec {
        id: "deck-cnr",
        label: "Chroma noise reducer",
        help: "Opt-in. HQ decks' recursive chroma noise reducer (CNR), 0–1: colour noise is averaged with the line \
               above where the colour changes little, which reduces colour blotches but smears colour vertically \
               a little more.",
        unit: "",
        group: Group::Deck,
        kind: num(0.0, 1.0, 1.0),
    },
    ParamSpec {
        id: "deck-sharpness",
        label: "Deck sharpness",
        help: "Deck picture / sharpness control (−1 to 1), applied after the noise canceller: positive adds a \
               short symmetric peaker around 2.4 MHz (up to +6 dB) with one thin halo on each side of an edge and \
               slightly lifted grain; 0 is the edit setting (no enhancement); negative softens. Detail under \
               1.5 IRE is not boosted.",
        unit: "",
        group: Group::Deck,
        kind: num(-1.0, 1.0, 0.0),
    },
    ParamSpec {
        id: "comb",
        label: "Decoder comb filter",
        help: "Luma/chroma separation in the TV or capture card (composite only). 2sample: simple, rainbow on fine \
               detail, and soft luma (−3 dB at 1.8 MHz). 1h: line comb, less rainbow but hanging dots on colour \
               edges. 3d: frame comb using the previous frame where the picture is still (motion-adaptive), the \
               cleanest on static areas. trap: a chroma trap (notch ± 0.6 MHz around the subcarrier), what \
               capture chips use for VCR input: luma keeps its detail to about 2.9 MHz, fine detail near the \
               subcarrier shows as rainbow.",
        unit: "",
        group: Group::Deck,
        kind: Kind::Choice { options: &["2sample", "1h", "3d", "trap"], default: "2sample" },
    },
    // ── Decoder / output ──
    ParamSpec {
        id: "temporal-coherence",
        label: "Temporal coherence",
        help: "How slowly the tape state drifts in video (0–1). 0 re-rolls head-smear bands, wobble, head-switch shape and top-band tint every frame; 1 makes them drift slowly like a real tape.",
        unit: "",
        group: Group::Output,
        kind: num(0.0, 1.0, 1.0),
    },
    ParamSpec {
        id: "video-out",
        label: "Video connection",
        help: "How the playback device connects to the TV or capture card. composite: one wire, the decoder's \
               comb filter separates luma and colour (rainbow cross-colour on fine detail, dot crawl on colour \
               edges). svideo: separate luma and colour wires, no comb filter and no cross-colour.",
        unit: "",
        group: Group::Output,
        kind: Kind::Choice { options: &["composite", "svideo"], default: "composite" },
    },
    ParamSpec {
        id: "input-matrix",
        label: "Input colour matrix",
        help: "Colour matrix used to convert source video to RGB. auto: the stream's colour tag if present, \
               otherwise BT.709 for HD (720 lines or more) and BT.601 for SD. Wrong matrices shift hues slightly \
               (e.g. greens and reds).",
        unit: "",
        group: Group::Output,
        kind: Kind::Choice { options: &["auto", "601", "709"], default: "auto" },
    },
    ParamSpec {
        id: "input-range",
        label: "Input range",
        help: "Value range of source video. auto: the stream's tag if present, otherwise limited (16–235) \
               except for full-range (JPEG) pixel formats. A wrong range makes the picture washed out or crushed.",
        unit: "",
        group: Group::Output,
        kind: Kind::Choice { options: &["auto", "limited", "full"], default: "auto" },
    },
    ParamSpec {
        id: "legacy-burst-phase",
        label: "Legacy burst phase",
        help: "Decode with the inverted burst phase estimate of older builds. With --seed, reproduces output from builds before fb47e5c exactly.",
        unit: "",
        group: Group::Output,
        kind: Kind::Flag,
    },
];

/// Look up a parameter by flag name.
pub fn spec(id: &str) -> Option<&'static ParamSpec> {
    SPECS.iter().find(|s| s.id == id)
}

// ────────────────────────────────────────────────────────────────────────────
// Parameter values
// ────────────────────────────────────────────────────────────────────────────

/// SplitMix-based combination of a seed with an index.
pub fn mix_seed(seed: u64, index: u64) -> u64 {
    splitmix64(seed ^ splitmix64(index))
}

/// RNG for the effects of one frame: derived from the seed and the frame
/// number when a seed is given, otherwise fresh entropy.
pub fn frame_rng(seed: Option<u64>, frame: u64) -> StdRng {
    match seed {
        Some(seed) => StdRng::seed_from_u64(mix_seed(mix_seed(seed, 0x5EED), frame)),
        None => StdRng::from_rng(&mut rand::rng()),
    }
}

/// The values a user has set.  Parameters that are not set fall back to
/// their spec default (or are off).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParamValues {
    pub numbers: BTreeMap<&'static str, RangeParam>,
    pub flags: BTreeMap<&'static str, bool>,
    pub texts: BTreeMap<&'static str, String>,
    pub choices: BTreeMap<&'static str, &'static str>,
    /// Seed for every random choice; `None` = fresh entropy each run.
    pub seed: Option<u64>,
}

impl ParamValues {
    /// Number or range for `id`: the set value, else the spec default.
    pub fn number(&self, id: &str) -> Option<RangeParam> {
        if let Some(v) = self.numbers.get(id) {
            return Some(*v);
        }
        match spec(id).map(|s| s.kind) {
            Some(Kind::Number { default: Some(d), .. }) => Some(RangeParam::fixed(d)),
            _ => None,
        }
    }

    pub fn flag(&self, id: &str) -> bool {
        self.flags.get(id).copied().unwrap_or(false)
    }

    pub fn text(&self, id: &str) -> Option<&str> {
        self.texts.get(id).map(String::as_str)
    }

    /// The chosen option for `id`: the set value, else the spec default.
    pub fn choice(&self, id: &str) -> &'static str {
        if let Some(v) = self.choices.get(id) {
            return v;
        }
        match spec(id).map(|s| s.kind) {
            Some(Kind::Choice { default, .. }) => default,
            _ => "",
        }
    }

    /// `true` if `id` was set explicitly (not just defaulted).
    pub fn is_set(&self, id: &str) -> bool {
        self.numbers.contains_key(id)
            || self.flags.contains_key(id)
            || self.texts.contains_key(id)
            || self.choices.contains_key(id)
    }

    /// Set a value from its text form (as on the command line).  Flags
    /// accept true/false, 1/0, yes/no, on/off.
    pub fn set_from_str(&mut self, id: &str, value: &str) -> Result<(), String> {
        let spec = spec(id).ok_or_else(|| format!("unknown parameter '{}'", id))?;
        match spec.kind {
            Kind::Number { .. } => {
                let v: RangeParam = value.parse().map_err(|e| format!("--{}: {}", id, e))?;
                self.numbers.insert(spec.id, v);
            }
            Kind::Flag => {
                let b = match value.trim().to_ascii_lowercase().as_str() {
                    "true" | "1" | "yes" | "on" => true,
                    "false" | "0" | "no" | "off" => false,
                    other => return Err(format!("--{}: '{}' is not a boolean", id, other)),
                };
                self.flags.insert(spec.id, b);
            }
            Kind::Text => {
                self.texts.insert(spec.id, value.to_string());
            }
            Kind::Choice { options, .. } => {
                let v = value.trim();
                let opt = options
                    .iter()
                    .find(|o| o.eq_ignore_ascii_case(v))
                    .ok_or_else(|| format!("--{}: '{}' is not one of {}", id, v, options.join(", ")))?;
                self.choices.insert(spec.id, opt);
            }
        }
        Ok(())
    }

    /// Fill every parameter that is not set here with the value from
    /// `other`.  Values set here always win.
    pub fn fill_from(&mut self, other: &ParamValues) {
        for (k, v) in &other.numbers {
            self.numbers.entry(k).or_insert(*v);
        }
        for (k, v) in &other.flags {
            self.flags.entry(k).or_insert(*v);
        }
        for (k, v) in &other.texts {
            self.texts.entry(k).or_insert_with(|| v.clone());
        }
        for (k, v) in &other.choices {
            self.choices.entry(k).or_insert(v);
        }
        if self.seed.is_none() {
            self.seed = other.seed;
        }
    }

    /// RNG for the per-file sampling of range parameters.
    pub fn file_rng(&self) -> StdRng {
        match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_rng(&mut rand::rng()),
        }
    }

    /// Copy of these values for file `index` of a batch, with its own seed.
    pub fn for_file(&self, index: usize) -> ParamValues {
        let mut v = self.clone();
        v.seed = self.seed.map(|s| mix_seed(s, index as u64 + 1));
        v
    }

    /// Resolve ranges and build the effect configuration for one file.
    ///
    /// The sampling order is fixed: changing it would change the output of
    /// seeded runs.
    pub fn to_signal_effects<R: Rng>(&self, rng: &mut R) -> SignalEffects {
        let n = |id: &str| self.number(id);
        let req = |id: &str| self.number(id).expect("parameter with a default");
        let ghost_dynamic = self.flag("ghost-dynamic");
        let mut ghosts: Vec<GhostConfig> = Vec::new();

        // Primary ghost from --ghost
        if let Some(amp_range) = n("ghost") {
            ghosts.push(GhostConfig {
                amplitude: amp_range.sample(rng),
                delay_us: req("ghost-delay").sample(rng),
                phase_shift: req("ghost-phase").sample(rng).to_radians(),
                rolloff_hz: RangeParam::sample_opt(n("ghost-rolloff-mhz"), rng)
                    .map(|m| m as f64 * 1e6),
                dynamic: ghost_dynamic,
                dynamic_rate: req("ghost-dynamic-rate").sample(rng),
            });
        }

        // Extra ghosts from --ghost-multi (each value supports min..max ranges)
        if let Some(multi) = self.text("ghost-multi") {
            for entry in multi.split(';') {
                let parts: Vec<&str> = entry.trim().split(',').collect();
                if parts.len() >= 2 {
                    let amp_r: RangeParam =
                        parts[0].trim().parse().unwrap_or(RangeParam::fixed(0.0));
                    let delay_r: RangeParam =
                        parts[1].trim().parse().unwrap_or(RangeParam::fixed(2.0));
                    let phase_r: RangeParam = if parts.len() >= 3 {
                        parts[2].trim().parse().unwrap_or(RangeParam::fixed(0.0))
                    } else {
                        RangeParam::fixed(0.0)
                    };
                    ghosts.push(GhostConfig {
                        amplitude: amp_r.sample(rng),
                        delay_us: delay_r.sample(rng),
                        phase_shift: phase_r.sample(rng).to_radians(),
                        rolloff_hz: RangeParam::sample_opt(n("ghost-rolloff-mhz"), rng)
                            .map(|m| m as f64 * 1e6),
                        dynamic: ghost_dynamic,
                        dynamic_rate: req("ghost-dynamic-rate").sample(rng),
                    });
                }
            }
        }

        let s = |id: &str, rng: &mut R| RangeParam::sample_opt(self.number(id), rng);
        let mut fx = SignalEffects {
            noise: s("noise", rng),
            ghosts,
            attenuation: s("attenuation", rng),
            jitter: s("jitter", rng),
            vhs_luma_bw: s("vhs-luma-bw", rng).map(|m| m * 1e6),
            color_under_bw: s("color-under-bw", rng).map(|k| k * 1e3),
            tape_dropout_rate: s("tape-dropout-rate", rng),
            tape_dropout_len: req("tape-dropout-len").sample(rng),
            edge_ringing: s("edge-ringing", rng),
            luma_noise: s("luma-noise", rng),
            head_switching_smear: s("head-smear", rng),
            tape_trail_smear: s("tape-trail", rng),
            head_switch: s("head-switch", rng),
            head_switch_lines: s("head-switch-lines", rng)
                .map_or(DEFAULT_HEAD_SWITCH_LINES, |n| n.round().max(0.0) as usize),
            top_band: s("top-band", rng),
            vbi_data: s("vbi-data", rng),
            chroma_noise: s("chroma-noise", rng),
            chroma_phase_noise: s("chroma-phase-noise", rng),
            chroma_vblur: s("chroma-vblur", rng),
            chroma_delay: s("chroma-delay", rng),
            chroma_delay_v: s("chroma-delay-v", rng),
            wobble: s("wobble", rng),
            wobble_rate: 1.0,
            temporal_coherence: 0.0,
            tape_seed: 0,
            line_tbc: None,
            frame_tbc: None,
            v_jitter: None,
            deck_sharpness: None,
            deck_dnr: None,
            stage_seed: 0,
            deck_nc: None,
            deck_ynr: None,
            deck_cnr: None,
            head_switch_point: None,
            head_switch_after: match self.choice("head-switch-after") {
                "picture" => HeadSwitchAfter::Picture,
                _ => HeadSwitchAfter::Black,
            },
            flagging: None,
            tape_model: match self.choice("tape-model") {
                "fm" => TapeModel::Fm,
                _ => TapeModel::Classic,
            },
            doc: match self.choice("doc") {
                "luma" => Doc::Luma,
                "off" => Doc::Off,
                _ => Doc::Composite,
            },
        };
        if fx.wobble.is_some() {
            fx.wobble_rate = req("wobble-rate").sample(rng);
        }
        // Drawn last so the values above match runs without coherence.
        fx.temporal_coherence = s("temporal-coherence", rng).unwrap_or(0.0);
        if fx.temporal_coherence > 0.0 {
            fx.tape_seed = match self.seed {
                Some(seed) => mix_seed(seed, 0x7A9E),
                None => rng.random(),
            };
        }
        // Deck stages (newer; drawn after everything above).
        fx.line_tbc = s("line-tbc", rng);
        fx.frame_tbc = s("frame-tbc", rng);
        fx.v_jitter = s("v-jitter", rng);
        fx.deck_sharpness = s("deck-sharpness", rng);
        fx.deck_dnr = s("deck-dnr", rng);
        // Stages with their own random streams (drawn after everything
        // above, so older parameter sets keep their exact values).
        fx.deck_nc = s("deck-nc", rng);
        fx.deck_ynr = s("deck-ynr", rng);
        fx.deck_cnr = s("deck-cnr", rng);
        fx.head_switch_point = s("head-switch-point", rng);
        fx.flagging = s("flagging", rng);
        fx.stage_seed = match self.seed {
            Some(seed) => mix_seed(seed, 0x57A6E),
            None => rng.random(),
        };
        fx
    }

    /// `true` if any parameter that is in use is a genuine range.
    pub fn has_any_range(&self) -> bool {
        SPECS.iter().any(|spec| {
            if spec.id == "wobble-rate" && self.number("wobble").is_none() {
                return false;
            }
            matches!(spec.kind, Kind::Number { .. })
                && self.number(spec.id).is_some_and(|r| r.is_range())
        })
    }

    /// The values that the ranges resolved to, as `name=value` strings.
    pub fn sampled_summary(&self, fx: &SignalEffects) -> Vec<String> {
        let mut parts: Vec<String> = Vec::new();
        let ranged = |id: &str| self.number(id).is_some_and(|r| r.is_range());
        let mut opt = |label: &str, id: &str, val: Option<f32>| {
            if ranged(id) {
                if let Some(v) = val {
                    parts.push(format!("{}={:.4}", label, v));
                }
            }
        };
        opt("noise", "noise", fx.noise);
        let first = fx.ghosts.first();
        opt("ghost", "ghost", first.map(|g| g.amplitude));
        if ranged("ghost-delay") {
            if let Some(g) = first {
                // Kept in this position to match earlier log output.
                parts.push(format!("ghost_delay={:.2}", g.delay_us));
            }
        }
        if ranged("ghost-phase") {
            if let Some(g) = first {
                parts.push(format!("ghost_phase={:.1}", g.phase_shift.to_degrees()));
            }
        }
        let mut opt = |label: &str, id: &str, val: Option<f32>| {
            if ranged(id) {
                if let Some(v) = val {
                    parts.push(format!("{}={:.4}", label, v));
                }
            }
        };
        opt("atten", "attenuation", fx.attenuation);
        opt("jitter", "jitter", fx.jitter);
        opt("vhs_bw_mhz", "vhs-luma-bw", fx.vhs_luma_bw.map(|v| v / 1e6));
        opt("color_under_khz", "color-under-bw", fx.color_under_bw.map(|v| v / 1e3));
        opt("dropout_rate", "tape-dropout-rate", fx.tape_dropout_rate);
        opt("dropout_len", "tape-dropout-len", Some(fx.tape_dropout_len));
        opt("ringing", "edge-ringing", fx.edge_ringing);
        opt("luma_noise", "luma-noise", fx.luma_noise);
        opt("head_smear", "head-smear", fx.head_switching_smear);
        opt("tape_trail", "tape-trail", fx.tape_trail_smear);
        opt("head_switch", "head-switch", fx.head_switch);
        opt("head_switch_lines", "head-switch-lines", Some(fx.head_switch_lines as f32));
        opt("top_band", "top-band", fx.top_band);
        opt("vbi_data", "vbi-data", fx.vbi_data);
        opt("chroma_noise", "chroma-noise", fx.chroma_noise);
        opt("chroma_phase_noise", "chroma-phase-noise", fx.chroma_phase_noise);
        opt("chroma_vblur", "chroma-vblur", fx.chroma_vblur);
        opt("chroma_delay", "chroma-delay", fx.chroma_delay);
        opt("chroma_delay_v", "chroma-delay-v", fx.chroma_delay_v);
        opt("wobble", "wobble", fx.wobble);
        if fx.wobble.is_some() {
            opt("wobble_rate", "wobble-rate", Some(fx.wobble_rate));
        }
        opt("temporal_coherence", "temporal-coherence", Some(fx.temporal_coherence));
        opt("line_tbc", "line-tbc", fx.line_tbc);
        opt("frame_tbc", "frame-tbc", fx.frame_tbc);
        opt("v_jitter", "v-jitter", fx.v_jitter);
        opt("deck_sharpness", "deck-sharpness", fx.deck_sharpness);
        opt("deck_dnr", "deck-dnr", fx.deck_dnr);
        opt("deck_nc", "deck-nc", fx.deck_nc);
        opt("deck_ynr", "deck-ynr", fx.deck_ynr);
        opt("deck_cnr", "deck-cnr", fx.deck_cnr);
        opt("head_switch_point", "head-switch-point", fx.head_switch_point);
        opt("flagging", "flagging", fx.flagging);
        parts
    }

    /// Command-line arguments that reproduce these values (explicitly set
    /// parameters only, in table order, plus the seed).
    pub fn cli_args(&self) -> Vec<String> {
        let mut args = Vec::new();
        for spec in SPECS {
            match spec.kind {
                Kind::Number { .. } => {
                    if let Some(v) = self.numbers.get(spec.id) {
                        args.push(format!("--{}", spec.id));
                        args.push(v.to_string());
                    }
                }
                Kind::Flag => {
                    if self.flag(spec.id) {
                        args.push(format!("--{}", spec.id));
                    }
                }
                Kind::Text => {
                    if let Some(t) = self.texts.get(spec.id) {
                        args.push(format!("--{}", spec.id));
                        args.push(t.clone());
                    }
                }
                Kind::Choice { .. } => {
                    if let Some(c) = self.choices.get(spec.id) {
                        args.push(format!("--{}", spec.id));
                        args.push(c.to_string());
                    }
                }
            }
        }
        if let Some(seed) = self.seed {
            args.push("--seed".into());
            args.push(seed.to_string());
        }
        args
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_ids_are_unique_and_defaults_resolve() {
        let mut ids: Vec<&str> = SPECS.iter().map(|s| s.id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), SPECS.len());
        let v = ParamValues::default();
        assert_eq!(v.number("ghost-delay"), Some(RangeParam::fixed(2.0)));
        assert_eq!(v.number("noise"), None);
    }

    #[test]
    fn explicit_values_win_over_filled_ones() {
        let mut a = ParamValues::default();
        a.set_from_str("noise", "0.1").unwrap();
        let mut b = ParamValues::default();
        b.set_from_str("noise", "0.3").unwrap();
        b.set_from_str("jitter", "0.2..0.4").unwrap();
        a.fill_from(&b);
        assert_eq!(a.number("noise"), Some(RangeParam::fixed(0.1)));
        assert_eq!(a.number("jitter"), Some(RangeParam { min: 0.2, max: 0.4 }));
    }

    #[test]
    fn cli_args_round_trip() {
        let mut v = ParamValues::default();
        v.set_from_str("color-under-bw", "400..450").unwrap();
        v.set_from_str("ghost-dynamic", "true").unwrap();
        v.set_from_str("ghost-multi", "0.1,4,0").unwrap();
        v.seed = Some(42);
        let args = v.cli_args();
        let mut w = ParamValues::default();
        let mut it = args.iter();
        while let Some(a) = it.next() {
            let id = a.trim_start_matches("--");
            match id {
                "seed" => w.seed = Some(it.next().unwrap().parse().unwrap()),
                "ghost-dynamic" => w.set_from_str(id, "true").unwrap(),
                _ => w.set_from_str(id, it.next().unwrap()).unwrap(),
            }
        }
        assert_eq!(v, w);
    }
}
