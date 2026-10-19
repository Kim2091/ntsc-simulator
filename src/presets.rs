//! Built-in presets.
//!
//! A preset is a set of parameter values.  It only fills in parameters the
//! user has not set, so explicit flags always win.

use crate::params::ParamValues;

/// A built-in preset.
pub struct BuiltinPreset {
    pub name: &'static str,
    /// Menu group, e.g. "VHS".
    pub group: &'static str,
    pub description: &'static str,
    pub values: &'static [(&'static str, &'static str)],
}

pub static BUILTIN: &[BuiltinPreset] = &[
    BuiltinPreset {
        name: "lq",
        group: "VHS",
        description: "Low quality: worn EP-mode tape",
        values: &[
            ("vhs-luma-bw", "1.6"),
            ("color-under-bw", "400..450"),
            ("edge-ringing", "0.3..0.4"),
            ("luma-noise", "0.025..0.035"),
            ("tape-dropout-rate", "1"),
            ("head-smear", "0.2..0.3"),
            ("tape-trail", "0.75..0.9"),
            ("head-switch", "0.8..1.0"),
            ("head-switch-lines", "7"),
            ("top-band", "0.6..0.8"),
            ("vbi-data", "0.6..0.8"),
        ],
    },
    BuiltinPreset {
        name: "mq",
        group: "VHS",
        description: "Medium quality: typical rental tape",
        values: &[
            ("vhs-luma-bw", "3"),
            ("color-under-bw", "400..450"),
            ("edge-ringing", "0.25..0.35"),
            ("luma-noise", "0.02..0.03"),
            ("tape-dropout-rate", "0..1"),
            ("head-smear", "0.15..0.2"),
            ("tape-trail", "0.6..0.75"),
            ("head-switch", "0.55..0.7"),
            ("head-switch-lines", "6"),
            ("top-band", "0.4..0.55"),
            ("vbi-data", "0.4..0.55"),
        ],
    },
    BuiltinPreset {
        name: "hq",
        group: "VHS",
        description: "High quality: clean SP recording",
        values: &[
            ("vhs-luma-bw", "3"),
            ("color-under-bw", "300..400"),
            ("edge-ringing", "0.2..0.3"),
            ("luma-noise", "0.015..0.025"),
            ("tape-dropout-rate", "0"),
            ("head-smear", "0.1..0.15"),
            ("tape-trail", "0.5..0.7"),
            ("head-switch", "0.35..0.45"),
            ("head-switch-lines", "5"),
            ("top-band", "0.2..0.3"),
            ("vbi-data", "0.25..0.35"),
        ],
    },
];

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
