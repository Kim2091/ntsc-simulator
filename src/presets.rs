//! Built-in presets and preset files.
//!
//! A preset is a set of parameter values.  It only fills in parameters the
//! user has not set, so explicit flags (or GUI edits) always win.
//!
//! Preset files are JSON:
//!
//! ```json
//! {
//!   "name": "My tape",
//!   "params": { "vhs-luma-bw": "1.6", "color-under-bw": "400..450", "ghost-dynamic": true },
//!   "seed": 42
//! }
//! ```
//!
//! Parameter values are written as on the command line: a number, a
//! `"min..max"` string, `true`/`false` for switches, or text.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::params::{spec, Kind, ParamValues, SPECS};

/// A built-in preset.
pub struct BuiltinPreset {
    pub name: &'static str,
    /// Menu group, e.g. "VHS" or "VHS legacy".
    pub group: &'static str,
    pub description: &'static str,
    pub values: &'static [(&'static str, &'static str)],
}

/// Values of the original VHS presets, used by both preset families.
const LEGACY_LQ: [(&str, &str); 7] = [
    ("vhs-luma-bw", "1.6"),
    ("color-under-bw", "400..450"),
    ("edge-ringing", "0.3..0.4"),
    ("luma-noise", "0.025..0.035"),
    ("tape-dropout-rate", "1"),
    ("head-smear", "0.2..0.3"),
    ("tape-trail", "0.75..0.9"),
];
const LEGACY_MQ: [(&str, &str); 7] = [
    ("vhs-luma-bw", "3"),
    ("color-under-bw", "400..450"),
    ("edge-ringing", "0.25..0.35"),
    ("luma-noise", "0.02..0.03"),
    ("tape-dropout-rate", "0..1"),
    ("head-smear", "0.15..0.2"),
    ("tape-trail", "0.6..0.75"),
];
const LEGACY_HQ: [(&str, &str); 7] = [
    ("vhs-luma-bw", "3"),
    ("color-under-bw", "300..400"),
    ("edge-ringing", "0.2..0.3"),
    ("luma-noise", "0.015..0.025"),
    ("tape-dropout-rate", "0"),
    ("head-smear", "0.1..0.15"),
    ("tape-trail", "0.5..0.7"),
];

pub static BUILTIN: &[BuiltinPreset] = &[
    BuiltinPreset {
        name: "lq",
        group: "VHS",
        description: "Worn EP-mode tape: soft, noisy, torn bottom, colour blotches and wobble",
        values: &[
            ("tape-model", "fm"),
            ("vhs-luma-bw", "2.5"),
            ("color-under-bw", "400..450"),
            ("nl-emphasis", "1"),
            ("tape-snr", "30..33"),
            ("dropouts", "40..100"),
            ("dropout-span", "2..4"),
            ("edge-ringing", "0.15..0.25"),
            ("rf-loss", "0.6..0.9"),
            ("tracking-bars", "0..0.3"),
            ("luma-crosstalk", "0.4..0.7"),
            ("ep-interleave", "0.2..0.8"),
            ("bw-inversion", "0.3..0.6"),
            ("head-level-step", "1..3"),
            ("generations", "1..2"),
            ("head-switch", "0.8..1.0"),
            ("head-switch-point", "5.5..7.5"),
            ("flagging", "0.4..0.6"),
            ("top-band", "0.6..0.8"),
            ("vbi-data", "0.6..0.8"),
            ("chroma-noise", "0.025..0.035"),
            ("chroma-phase-noise", "0"),
            ("chroma-vblur", "1"),
            ("chroma-noise-bw", "450"),
            ("chroma-crosstalk", "0.6..0.9"),
            ("chroma-delay", "2..3"),
            ("chroma-delay-v", "1"),
            ("wobble", "1..1.5"),
            ("temporal-coherence", "1"),
            ("deck-nc", "3..5"),
            ("deck-sharpness", "0.3..0.5"),
            ("doc", "luma"),
            ("comb", "trap"),
        ],
    },
    BuiltinPreset {
        name: "mq",
        group: "VHS",
        description: "Typical rental tape (LP-like wear)",
        values: &[
            ("tape-model", "fm"),
            ("vhs-luma-bw", "2.9"),
            ("color-under-bw", "400..450"),
            ("nl-emphasis", "0.5..1"),
            ("tape-snr", "34..36"),
            ("dropouts", "15..30"),
            ("dropout-span", "1..2"),
            ("edge-ringing", "0.25..0.35"),
            ("rf-loss", "0.3..0.5"),
            ("luma-crosstalk", "0.1..0.3"),
            ("bw-inversion", "0.1..0.2"),
            ("head-level-step", "0.5..2"),
            ("head-switch", "0.55..0.7"),
            ("head-switch-point", "6..7"),
            ("flagging", "0.25..0.4"),
            ("top-band", "0.4..0.55"),
            ("vbi-data", "0.4..0.55"),
            ("chroma-noise", "0.015..0.025"),
            ("chroma-phase-noise", "0"),
            ("chroma-vblur", "1"),
            ("chroma-noise-bw", "450"),
            ("chroma-crosstalk", "0.3..0.5"),
            ("chroma-delay", "1.5..2.5"),
            ("chroma-delay-v", "0..1"),
            ("wobble", "0.5..1"),
            ("temporal-coherence", "1"),
            ("deck-nc", "3..5"),
            ("deck-sharpness", "0.3..0.5"),
            ("doc", "luma"),
            ("comb", "trap"),
        ],
    },
    BuiltinPreset {
        name: "hq",
        group: "VHS",
        description: "Clean SP recording",
        values: &[
            ("tape-model", "fm"),
            ("vhs-luma-bw", "3"),
            ("color-under-bw", "400"),
            ("nl-emphasis", "0..0.5"),
            ("tape-snr", "37..38"),
            ("dropouts", "5..15"),
            ("edge-ringing", "0.2..0.3"),
            ("rf-loss", "0.1..0.2"),
            ("head-level-step", "0..1"),
            ("head-switch", "0.35..0.45"),
            ("head-switch-point", "6..7"),
            ("flagging", "0.15..0.25"),
            ("top-band", "0.2..0.3"),
            ("vbi-data", "0.25..0.35"),
            ("chroma-noise", "0.008..0.015"),
            ("chroma-phase-noise", "0"),
            ("chroma-vblur", "1"),
            ("chroma-noise-bw", "450"),
            ("chroma-crosstalk", "0.15..0.3"),
            ("chroma-delay", "1..2"),
            ("chroma-delay-v", "0"),
            ("wobble", "0.3..0.6"),
            ("temporal-coherence", "1"),
            ("deck-nc", "3..5"),
            ("deck-sharpness", "0.3..0.5"),
            ("doc", "luma"),
            ("comb", "trap"),
        ],
    },
    BuiltinPreset {
        name: "dvd-hq",
        group: "DVD",
        description: "Well-made film title: 23.976p soft pulldown, 7-8 Mbit/s, two-pass",
        values: &[
            ("format", "dvd"),
            ("dvd-rate", "film"),
            ("dvd-bitrate", "7..8"),
            ("dvd-maxrate", "9.5"),
            ("dvd-2pass", "true"),
        ],
    },
    BuiltinPreset {
        name: "dvd-mq",
        group: "DVD",
        description: "Average film title: 23.976p soft pulldown, 5-6 Mbit/s",
        values: &[
            ("format", "dvd"),
            ("dvd-rate", "film"),
            ("dvd-bitrate", "5..6"),
            ("dvd-maxrate", "8..9"),
        ],
    },
    BuiltinPreset {
        name: "dvd-mq-ee",
        group: "DVD",
        description: "Average film title from a processed master: edge-enhancement halos and noise reduction",
        values: &[
            ("format", "dvd"),
            ("dvd-rate", "film"),
            ("dvd-bitrate", "5..6"),
            ("dvd-maxrate", "8..9"),
            ("dvd-ee", "0.5..0.7"),
            ("dvd-dnr", "0.3..0.5"),
        ],
    },
    BuiltinPreset {
        name: "dvd-lq",
        group: "DVD",
        description: "Budget / long-play disc: half-D1 352x480, 3-4 Mbit/s, low-rate matrices, pre-filtered",
        values: &[
            ("format", "dvd"),
            ("dvd-rate", "film"),
            ("dvd-size", "352x480"),
            ("dvd-bitrate", "3..4"),
            ("dvd-maxrate", "6..7"),
            ("dvd-matrix", "lowrate"),
            ("dvd-prefilter", "0.3..0.5"),
        ],
    },
    BuiltinPreset {
        name: "dvd-interlaced",
        group: "DVD",
        description: "Video-sourced title: 480i top field first, 5-6 Mbit/s",
        values: &[
            ("format", "dvd"),
            ("dvd-rate", "interlaced"),
            ("dvd-bitrate", "5..6"),
            ("dvd-maxrate", "8..9"),
        ],
    },
    BuiltinPreset {
        name: "legacy-lq",
        group: "VHS legacy",
        description: "Original LQ settings (tape path only)",
        values: &LEGACY_LQ,
    },
    BuiltinPreset {
        name: "legacy-mq",
        group: "VHS legacy",
        description: "Original MQ settings (tape path only)",
        values: &LEGACY_MQ,
    },
    BuiltinPreset {
        name: "legacy-hq",
        group: "VHS legacy",
        description: "Original HQ settings (tape path only)",
        values: &LEGACY_HQ,
    },
];

/// Playback-deck profiles.  They combine with the tape / DVD presets: a
/// deck's values win over the tape preset's (`--deck` is applied before
/// `--preset`).
pub static DECKS: &[BuiltinPreset] = &[
    BuiltinPreset {
        name: "deck-basic",
        group: "Deck",
        description: "Ordinary consumer VCR: composite output, noise canceller, colour-killing dropout compensator, \
                      mild sharpening, no TBC",
        values: &[("deck-nc", "4"), ("deck-sharpness", "0.4"), ("doc", "luma")],
    },
    BuiltinPreset {
        name: "deck-svhs-tbc",
        group: "Deck",
        description: "High-end S-VHS deck: line TBC, frame TBC, recursive noise reduction, S-Video out (3D comb for composite captures)",
        values: &[
            ("line-tbc", "0.9"),
            ("frame-tbc", "1"),
            ("deck-dnr", "0.5"),
            ("comb", "3d"),
            ("video-out", "svideo"),
            ("deck-nc", "5"),
            ("deck-ynr", "1"),
            ("deck-cnr", "1"),
            ("deck-sharpness", "0.2"),
            ("doc", "luma"),
        ],
    },
    BuiltinPreset {
        name: "deck-edit",
        group: "Deck",
        description: "Deck in edit mode: no sharpening or peaking",
        values: &[("deck-sharpness", "0")],
    },
];

/// Look up a deck profile by name.
pub fn deck(name: &str) -> Option<&'static BuiltinPreset> {
    DECKS.iter().find(|p| p.name == name)
}

/// Look up a built-in preset by name.
pub fn builtin(name: &str) -> Option<&'static BuiltinPreset> {
    BUILTIN.iter().find(|p| p.name == name)
}

impl BuiltinPreset {
    pub fn to_values(&self) -> ParamValues {
        let mut v = ParamValues::default();
        for (id, val) in self.values {
            v.set_from_str(id, val).expect("valid built-in preset value");
        }
        v
    }
}

/// A named set of values loaded from or saved to a preset file.
#[derive(Clone, Debug, PartialEq)]
pub struct Preset {
    pub name: String,
    pub values: ParamValues,
}

#[derive(Serialize, Deserialize)]
struct PresetFile {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    params: BTreeMap<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
}

/// Parse a preset from JSON text.  `fallback_name` is used when the file
/// has no name.
pub fn preset_from_json(text: &str, fallback_name: &str) -> Result<Preset> {
    let file: PresetFile = serde_json::from_str(text).context("invalid preset JSON")?;
    let mut values = ParamValues::default();
    for (key, val) in &file.params {
        let Some(spec) = spec(key) else {
            bail!("unknown parameter '{}' in preset", key)
        };
        let text = match (val, spec.kind) {
            (serde_json::Value::Bool(b), Kind::Flag) => b.to_string(),
            (serde_json::Value::Number(n), Kind::Number { .. }) => n.to_string(),
            (serde_json::Value::String(s), _) => s.clone(),
            _ => bail!("parameter '{}' has a value of the wrong type", key),
        };
        values.set_from_str(spec.id, &text).map_err(anyhow::Error::msg)?;
    }
    values.seed = file.seed;
    Ok(Preset { name: file.name.unwrap_or_else(|| fallback_name.to_string()), values })
}

/// Serialise a preset to pretty JSON (parameters in alphabetical order).
pub fn preset_to_json(preset: &Preset) -> String {
    let mut params = BTreeMap::new();
    for spec in SPECS {
        let v = &preset.values;
        match spec.kind {
            Kind::Number { .. } => {
                if let Some(r) = v.numbers.get(spec.id) {
                    params.insert(spec.id.to_string(), serde_json::Value::String(r.to_string()));
                }
            }
            Kind::Flag => {
                if let Some(b) = v.flags.get(spec.id) {
                    params.insert(spec.id.to_string(), serde_json::Value::Bool(*b));
                }
            }
            Kind::Text => {
                if let Some(t) = v.texts.get(spec.id) {
                    params.insert(spec.id.to_string(), serde_json::Value::String(t.clone()));
                }
            }
            Kind::Choice { .. } => {
                if let Some(c) = v.choices.get(spec.id) {
                    params.insert(spec.id.to_string(), serde_json::Value::String(c.to_string()));
                }
            }
        }
    }
    let file = PresetFile { name: Some(preset.name.clone()), params, seed: preset.values.seed };
    serde_json::to_string_pretty(&file).expect("preset serialises")
}

/// Load a preset file.
pub fn load_preset_file(path: &Path) -> Result<Preset> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("cannot read preset file '{}'", path.display()))?;
    let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    preset_from_json(&text, &stem).with_context(|| format!("in preset file '{}'", path.display()))
}

/// Save a preset file.
pub fn save_preset_file(path: &Path, preset: &Preset) -> Result<()> {
    std::fs::write(path, preset_to_json(preset))
        .with_context(|| format!("cannot write preset file '{}'", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_presets_parse() {
        for p in BUILTIN {
            let v = p.to_values();
            let set = v.numbers.len() + v.flags.len() + v.texts.len() + v.choices.len();
            assert_eq!(set, p.values.len(), "{}", p.name);
        }
        // Legacy presets hold the original seven values and nothing else.
        for name in ["legacy-lq", "legacy-mq", "legacy-hq"] {
            assert_eq!(builtin(name).unwrap().values.len(), 7);
        }
    }

    #[test]
    fn deck_profiles_parse() {
        for d in DECKS {
            let v = d.to_values();
            let set = v.numbers.len() + v.flags.len() + v.texts.len() + v.choices.len();
            assert_eq!(set, d.values.len(), "{}", d.name);
        }
    }

    #[test]
    fn preset_json_round_trip() {
        let mut values = builtin("lq").unwrap().to_values();
        values.set_from_str("ghost-dynamic", "true").unwrap();
        values.set_from_str("ghost-multi", "0.1,4,0;0.05,8,180").unwrap();
        values.seed = Some(7);
        let preset = Preset { name: "test".into(), values };
        let back = preset_from_json(&preset_to_json(&preset), "x").unwrap();
        assert_eq!(back, preset);
    }

    #[test]
    fn preset_json_accepts_numbers_and_rejects_unknown_keys() {
        let p = preset_from_json(r#"{"params": {"noise": 0.05, "chroma-delay": "1..2"}}"#, "f").unwrap();
        assert_eq!(p.name, "f");
        assert_eq!(p.values.number("noise").unwrap().min, 0.05);
        assert!(preset_from_json(r#"{"params": {"bogus": 1}}"#, "f").is_err());
    }
}
