//! Editor state and controls for the effect parameters, generated from the
//! shared parameter table.

use eframe::egui;
use ntsc_composite_simulator::params::{Group, Kind, ParamSpec, ParamValues, RangeParam, SPECS};

/// UI state of one parameter.  Values are kept while a parameter is
/// switched off, so turning it back on restores them.
#[derive(Clone, Debug, PartialEq)]
pub struct ParamState {
    /// Number/text: the value is set.  Flag: the switch is on.
    pub enabled: bool,
    /// Use a min..max range instead of a single value.
    pub range: bool,
    pub min: f32,
    pub max: f32,
    pub text: String,
}

fn default_state(spec: &ParamSpec) -> ParamState {
    let typical = match spec.kind {
        Kind::Number { typical, .. } => typical,
        _ => 0.0,
    };
    let text = match spec.kind {
        Kind::Choice { default, .. } => default.to_string(),
        _ => String::new(),
    };
    ParamState { enabled: false, range: false, min: typical, max: typical, text }
}

/// All parameter states, in the order of [`SPECS`].
#[derive(Clone, Debug, PartialEq)]
pub struct ParamEditor {
    pub states: Vec<ParamState>,
}

impl ParamEditor {
    /// Editor showing `values`; everything not set is off.
    pub fn from_values(values: &ParamValues) -> Self {
        let states = SPECS
            .iter()
            .map(|spec| {
                let mut st = default_state(spec);
                match spec.kind {
                    Kind::Number { .. } => {
                        if let Some(r) = values.numbers.get(spec.id) {
                            st = ParamState { enabled: true, range: r.is_range(), min: r.min, max: r.max, ..st };
                        }
                    }
                    Kind::Flag => st.enabled = values.flag(spec.id),
                    Kind::Text => {
                        if let Some(t) = values.text(spec.id) {
                            st.enabled = true;
                            st.text = t.to_string();
                        }
                    }
                    Kind::Choice { .. } => {
                        st.text = values.choice(spec.id).to_string();
                        st.enabled = values.choices.contains_key(spec.id);
                    }
                }
                st
            })
            .collect();
        Self { states }
    }

    /// The values the editor describes (without a seed).
    pub fn to_values(&self) -> ParamValues {
        let mut v = ParamValues::default();
        for (spec, st) in SPECS.iter().zip(&self.states) {
            if let Kind::Choice { default, .. } = spec.kind {
                if st.enabled || st.text != default {
                    let _ = v.set_from_str(spec.id, &st.text);
                }
                continue;
            }
            if !st.enabled {
                continue;
            }
            match spec.kind {
                Kind::Number { integer, .. } => {
                    let r = |x: f32| if integer { x.round() } else { x };
                    let (min, max) = (r(st.min), r(st.max));
                    let max = if st.range { max.max(min) } else { min };
                    v.numbers.insert(spec.id, RangeParam { min, max });
                }
                Kind::Flag => {
                    v.flags.insert(spec.id, true);
                }
                Kind::Text => {
                    if !st.text.trim().is_empty() {
                        v.texts.insert(spec.id, st.text.trim().to_string());
                    }
                }
                Kind::Choice { .. } => unreachable!(),
            }
        }
        v
    }

    /// Copy the states of one group from `other`.
    pub fn reset_group(&mut self, group: Group, other: &ParamEditor) {
        for (i, spec) in SPECS.iter().enumerate() {
            if spec.group == group {
                self.states[i] = other.states[i].clone();
            }
        }
    }

    /// `true` if any parameter of `group` differs from `other`.
    pub fn group_differs(&self, group: Group, other: &ParamEditor) -> bool {
        SPECS
            .iter()
            .enumerate()
            .any(|(i, spec)| spec.group == group && self.states[i] != other.states[i])
    }

    /// Controls for every parameter of `group`.
    pub fn ui_group(&mut self, ui: &mut egui::Ui, group: Group) {
        for (i, spec) in SPECS.iter().enumerate() {
            if spec.group == group {
                param_ui(ui, spec, &mut self.states[i]);
            }
        }
    }
}

fn tooltip(spec: &ParamSpec) -> String {
    let mut t = spec.help.to_string();
    if let Kind::Number { min, max, default, .. } = spec.kind {
        t.push_str(&format!("\n\nSlider: {}–{} {}", min, max, spec.unit));
        if let Some(d) = default {
            t.push_str(&format!("\nWhen unticked the default ({} {}) is used.", d, spec.unit));
        }
        t.push_str(&format!("\nCLI: --{}", spec.id));
    } else {
        t.push_str(&format!("\n\nCLI: --{}", spec.id));
    }
    t
}

fn slider<'a>(value: &'a mut f32, spec: &ParamSpec, text: &str) -> egui::Slider<'a> {
    let Kind::Number { min, max, integer, .. } = spec.kind else { unreachable!() };
    let mut s = egui::Slider::new(value, min..=max).text(text.to_string());
    if !spec.unit.is_empty() {
        s = s.suffix(format!(" {}", spec.unit));
    }
    if integer {
        s = s.integer();
    }
    s
}

fn param_ui(ui: &mut egui::Ui, spec: &ParamSpec, st: &mut ParamState) {
    let tip = tooltip(spec);
    match spec.kind {
        Kind::Number { default, .. } => {
            ui.horizontal(|ui| {
                ui.checkbox(&mut st.enabled, "").on_hover_text(if default.is_some() {
                    "Tick to override the default"
                } else {
                    "Tick to turn this effect on"
                });
                ui.label(spec.label).on_hover_text(&tip);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let r = ui
                        .add_enabled(st.enabled, egui::Button::selectable(st.range, "range"))
                        .on_hover_text(
                            "Use a min..max range: a random value is picked per file (and per preview render) \
                             from the range. See \"Sampled values\" below the preview.",
                        );
                    if r.clicked() {
                        st.range = !st.range;
                        if st.range && st.max <= st.min {
                            let Kind::Number { max, .. } = spec.kind else { unreachable!() };
                            st.max = (st.min + (max - st.min) * 0.2).min(max);
                        }
                    }
                });
            });
            ui.add_enabled_ui(st.enabled, |ui| {
                ui.indent(spec.id, |ui| {
                    if st.range {
                        ui.spacing_mut().slider_width = (ui.available_width() - 175.0).max(60.0);
                        ui.add(slider(&mut st.min, spec, "min")).on_hover_text(&tip);
                        ui.add(slider(&mut st.max, spec, "max")).on_hover_text(&tip);
                        if st.max < st.min {
                            st.max = st.min;
                        }
                    } else {
                        ui.spacing_mut().slider_width = (ui.available_width() - 175.0).max(60.0);
                        ui.add(slider(&mut st.min, spec, "")).on_hover_text(&tip);
                    }
                });
            });
        }
        Kind::Flag => {
            ui.checkbox(&mut st.enabled, spec.label).on_hover_text(&tip);
        }
        Kind::Choice { options, .. } => {
            ui.horizontal(|ui| {
                ui.label(spec.label).on_hover_text(&tip);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    egui::ComboBox::from_id_salt(spec.id)
                        .selected_text(st.text.as_str())
                        .width(130.0)
                        .show_ui(ui, |ui| {
                            for o in options {
                                if ui.selectable_label(st.text == *o, *o).clicked() {
                                    st.text = o.to_string();
                                    st.enabled = true;
                                }
                            }
                        })
                        .response
                        .on_hover_text(&tip);
                });
            });
        }
        Kind::Text => {
            ui.horizontal(|ui| {
                ui.checkbox(&mut st.enabled, "");
                ui.label(spec.label).on_hover_text(&tip);
            });
            ui.add_enabled_ui(st.enabled, |ui| {
                ui.indent(spec.id, |ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut st.text)
                            .hint_text("amp,delay_us,phase;…")
                            .desired_width(f32::INFINITY),
                    )
                    .on_hover_text(&tip);
                });
            });
        }
    }
    ui.add_space(2.0);
}
