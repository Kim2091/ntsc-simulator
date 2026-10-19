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

use crate::effects::{splitmix64, GhostConfig, SignalEffects, DEFAULT_HEAD_SWITCH_LINES};

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
    Tape,
    Colour,
    NoiseDropouts,
    Timing,
    Reception,
    Output,
}

impl Group {
    pub const ALL: [Group; 6] = [
        Group::Tape,
        Group::Colour,
        Group::NoiseDropouts,
        Group::Timing,
        Group::Reception,
        Group::Output,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Group::Tape => "Tape",
            Group::Colour => "Colour",
            Group::NoiseDropouts => "Noise & dropouts",
            Group::Timing => "Timing",
            Group::Reception => "Reception",
            Group::Output => "Decoder / output",
        }
    }
}

/// What kind of value a parameter takes.
#[derive(Clone, Copy, Debug)]
pub enum Kind {
    /// A number or `min..max` range.  `min`/`max` are the slider limits
    /// (the CLI accepts any value).  `default` applies when the parameter
    /// is not set; `None` means the effect is off.  `integer` values are
    /// rounded where they are used.
    Number { min: f32, max: f32, default: Option<f32>, integer: bool },
    /// An on/off switch.
    Flag,
    /// Free text.
    Text,
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

const fn num(min: f32, max: f32) -> Kind {
    Kind::Number { min, max, default: None, integer: false }
}

const fn num_def(min: f32, max: f32, default: f32) -> Kind {
    Kind::Number { min, max, default: Some(default), integer: false }
}

/// All effect parameters, in signal-chain order.
pub static SPECS: &[ParamSpec] = &[
    // ── Tape ──
    ParamSpec {
        id: "vhs-luma-bw",
        label: "Luma bandwidth",
        help: "VHS luma bandwidth in MHz (SP ≈ 3.0, EP/SLP ≈ 1.6). Lower values blur fine horizontal detail.",
        unit: "MHz",
        group: Group::Tape,
        kind: num(0.5, 4.2),
    },
    ParamSpec {
        id: "color-under-bw",
        label: "Colour-under bandwidth",
        help: "Colour-under chroma bandwidth in kHz (typical 300–500). Lower values smear colour sideways past edges.",
        unit: "kHz",
        group: Group::Tape,
        kind: num(100.0, 1000.0),
    },
    ParamSpec {
        id: "edge-ringing",
        label: "Edge ringing",
        help: "Playback peaking / sharpening gain (0.5–3.0). Adds bright and dark halos beside sharp edges.",
        unit: "",
        group: Group::Tape,
        kind: num(0.0, 3.0),
    },
    ParamSpec {
        id: "head-smear",
        label: "Head-wear smear",
        help: "Worn-head symmetric luma blur (0.3–1.0). Soft horizontal smudging that varies in bands down the frame.",
        unit: "",
        group: Group::Tape,
        kind: num(0.0, 1.0),
    },
    ParamSpec {
        id: "tape-trail",
        label: "Tape trail",
        help: "Causal rightward luma smear (0.3–1.0). Edges and noise leave a comet tail to the right, the classic worn-tape look.",
        unit: "",
        group: Group::Tape,
        kind: num(0.0, 1.0),
    },
    // ── Colour ──
    ParamSpec {
        id: "chroma-noise",
        label: "Chroma noise",
        help: "Colour-under chroma noise, std dev of I/Q noise (subtle 0.01–0.03, heavy 0.06). Horizontal colour blotches in flat areas.",
        unit: "",
        group: Group::Colour,
        kind: num(0.0, 0.1),
    },
    ParamSpec {
        id: "chroma-phase-noise",
        label: "Chroma phase jitter",
        help: "Per-line hue jitter, std dev in degrees (subtle 1–3). Lines flicker slightly in hue.",
        unit: "°",
        group: Group::Colour,
        kind: num(0.0, 15.0),
    },
    ParamSpec {
        id: "chroma-vblur",
        label: "Chroma vertical blur",
        help: "Playback 1H chroma comb (0–1). Colour is mixed with the line above, softening colour edges vertically.",
        unit: "",
        group: Group::Colour,
        kind: num(0.0, 1.0),
    },
    ParamSpec {
        id: "chroma-delay",
        label: "Chroma delay",
        help: "Horizontal chroma delay relative to luma in output pixels at 640 wide (typical 1–4, positive = right). Colour sits beside its object.",
        unit: "px",
        group: Group::Colour,
        kind: num(-4.0, 8.0),
    },
    ParamSpec {
        id: "chroma-delay-v",
        label: "Chroma delay (vertical)",
        help: "Vertical chroma delay in field lines (typical 0–1; 1 field line = 2 output rows). Colour sits below its object.",
        unit: "lines",
        group: Group::Colour,
        kind: Kind::Number { min: 0.0, max: 3.0, default: None, integer: true },
    },
    // ── Noise & dropouts ──
    ParamSpec {
        id: "luma-noise",
        label: "Luma noise",
        help: "Luminance-dependent tape noise amplitude (0.02–0.10). Grainy noise, stronger in dark areas.",
        unit: "",
        group: Group::NoiseDropouts,
        kind: num(0.0, 0.1),
    },
    ParamSpec {
        id: "tape-dropout-rate",
        label: "Dropouts",
        help: "Average tape dropouts per frame (2–20). Short white streaks or repeated line segments.",
        unit: "/frame",
        group: Group::NoiseDropouts,
        kind: num(0.0, 30.0),
    },
    ParamSpec {
        id: "tape-dropout-len",
        label: "Dropout length",
        help: "Average dropout length in microseconds (default 15).",
        unit: "µs",
        group: Group::NoiseDropouts,
        kind: num_def(1.0, 60.0, 15.0),
    },
    // ── Timing ──
    ParamSpec {
        id: "jitter",
        label: "Line jitter",
        help: "Random horizontal line jitter, std dev in subcarrier cycles (0.1–1.0). Lines jump sideways independently.",
        unit: "cycles",
        group: Group::Timing,
        kind: num(0.0, 3.0),
    },
    ParamSpec {
        id: "wobble",
        label: "Wobble",
        help: "Time-base wobble: peak horizontal displacement in output pixels (typical 0.5–3). Vertical edges bend in slow waves.",
        unit: "px",
        group: Group::Timing,
        kind: num(0.0, 6.0),
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
        id: "head-switch",
        label: "Head switching",
        help: "Head-switching noise (0–1). The bottom lines tear to the right with a dark, speckled switching line.",
        unit: "",
        group: Group::Timing,
        kind: num(0.0, 1.0),
    },
    ParamSpec {
        id: "head-switch-lines",
        label: "Head-switch lines",
        help: "Active lines per field torn by head switching (default 6, i.e. 12 rows at 480).",
        unit: "lines",
        group: Group::Timing,
        kind: Kind::Number { min: 1.0, max: 20.0, default: None, integer: true },
    },
    ParamSpec {
        id: "top-band",
        label: "Top colour band",
        help: "Faint blue/purple bands on the first lines of each field (0–1), as the deck's chroma loop settles.",
        unit: "",
        group: Group::Timing,
        kind: num(0.0, 1.0),
    },
    ParamSpec {
        id: "vbi-data",
        label: "Caption data line",
        help: "Closed-caption data on lines 21/284 (0–1). A row of short light dashes near the top of the picture.",
        unit: "",
        group: Group::Timing,
        kind: num(0.0, 1.0),
    },
    // ── Reception ──
    ParamSpec {
        id: "noise",
        label: "Snow",
        help: "Gaussian RF noise amplitude (0.05 = subtle, 0.2 = heavy snow).",
        unit: "",
        group: Group::Reception,
        kind: num(0.0, 0.3),
    },
    ParamSpec {
        id: "ghost",
        label: "Ghost",
        help: "Primary multipath ghost amplitude (0–1). A faint delayed copy of the picture to the right.",
        unit: "",
        group: Group::Reception,
        kind: num(0.0, 1.0),
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
        kind: num(0.5, 4.2),
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
        kind: num(0.0, 1.0),
    },
    // ── Decoder / output ──
    ParamSpec {
        id: "temporal-coherence",
        label: "Temporal coherence",
        help: "How slowly the tape state drifts in video (0–1). 0 re-rolls head-smear bands, wobble, head-switch shape and top-band tint every frame; 1 makes them drift slowly like a real tape.",
        unit: "",
        group: Group::Output,
        kind: num(0.0, 1.0),
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

    /// `true` if `id` was set explicitly (not just defaulted).
    pub fn is_set(&self, id: &str) -> bool {
        self.numbers.contains_key(id) || self.flags.contains_key(id) || self.texts.contains_key(id)
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
