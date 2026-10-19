//! Preview area: before/after split with a draggable divider, side by
//! side, or after only; nearest-neighbour zoom with panning.

use eframe::egui::{self, Color32, Pos2, Rect, Sense, Stroke, TextureHandle, Vec2};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewMode {
    Split,
    SideBySide,
    After,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Zoom {
    Fit,
    X1,
    X2,
    X4,
    /// Any other wheel zoom step, as an index into `WHEEL_LEVELS`.
    Level(usize),
}

/// Zoom factors (image pixels to screen pixels) the mouse wheel steps through.
const WHEEL_LEVELS: [f32; 12] = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0];

/// Scroll distance in points that counts as one wheel step.
const WHEEL_STEP: f32 = 40.0;

impl Zoom {
    pub fn label(self) -> String {
        match self {
            Zoom::Fit => "Fit".into(),
            Zoom::X1 => "1×".into(),
            Zoom::X2 => "2×".into(),
            Zoom::X4 => "4×".into(),
            Zoom::Level(i) => format!("{}×", WHEEL_LEVELS[i]),
        }
    }

    fn factor(self) -> Option<f32> {
        match self {
            Zoom::Fit => None,
            Zoom::X1 => Some(1.0),
            Zoom::X2 => Some(2.0),
            Zoom::X4 => Some(4.0),
            Zoom::Level(i) => Some(WHEEL_LEVELS[i]),
        }
    }

    /// The wheel level `i`, using the named variants where they exist so
    /// the toolbar buttons stay highlighted.
    fn from_level(i: usize) -> Zoom {
        match WHEEL_LEVELS[i] {
            f if f == 1.0 => Zoom::X1,
            f if f == 2.0 => Zoom::X2,
            f if f == 4.0 => Zoom::X4,
            _ => Zoom::Level(i),
        }
    }
}

pub struct Viewer {
    pub mode: ViewMode,
    pub zoom: Zoom,
    /// Pan offset in screen points (zoomed views only).
    pub pan: Vec2,
    /// Divider position as a fraction of the image width.
    pub split: f32,
    /// Scroll not yet turned into a wheel zoom step.
    scroll_acc: f32,
}

impl Default for Viewer {
    fn default() -> Self {
        Self { mode: ViewMode::Split, zoom: Zoom::Fit, pan: Vec2::ZERO, split: 0.5, scroll_acc: 0.0 }
    }
}

/// Screen points per image pixel for `zoom` inside `area`.
fn scale_for(area: Rect, size: Vec2, zoom: Zoom, ppp: f32) -> f32 {
    match zoom.factor() {
        Some(f) => f / ppp,
        None => (area.width() / size.x).min(area.height() / size.y),
    }
}

/// Where an image of `size` pixels lands inside `area`.
fn place(area: Rect, size: Vec2, zoom: Zoom, pan: Vec2, ppp: f32) -> Rect {
    let scale = scale_for(area, size, zoom, ppp);
    let rect = Rect::from_center_size(area.center() + pan, size * scale);
    // Snap to the pixel grid so nearest-neighbour zoom stays crisp.
    let snap = |p: Pos2| Pos2::new((p.x * ppp).round() / ppp, (p.y * ppp).round() / ppp);
    Rect::from_min_size(snap(rect.min), rect.size())
}

fn caption(painter: &egui::Painter, pos: Pos2, align: egui::Align2, text: &str) {
    let galley = painter.layout_no_wrap(text.to_string(), egui::FontId::proportional(13.0), Color32::WHITE);
    let rect = align.anchor_size(pos, galley.size());
    painter.rect_filled(rect.expand(4.0), 3.0, Color32::from_black_alpha(160));
    painter.galley(rect.min, galley, Color32::WHITE);
}

impl Viewer {
    /// Draw the preview into the remaining space of `ui`.  `size` is the
    /// output image size in pixels.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        before: Option<&TextureHandle>,
        after: Option<&TextureHandle>,
        size: Vec2,
    ) {
        let area = ui.available_rect_before_wrap();
        let response = ui.allocate_rect(area, Sense::click_and_drag());
        let painter = ui.painter_at(area);
        painter.rect_filled(area, 0.0, Color32::from_gray(20));
        let ppp = ui.ctx().pixels_per_point();
        let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));

        if response.hovered() {
            // In side-by-side view each half zooms about its own centre.
            let view = match self.mode {
                ViewMode::SideBySide => {
                    let half = (area.width() - 8.0) / 2.0;
                    let left = Rect::from_min_size(area.min, Vec2::new(half, area.height()));
                    let right = Rect::from_min_size(area.min + Vec2::new(half + 8.0, 0.0), Vec2::new(half, area.height()));
                    match response.hover_pos() {
                        Some(p) if right.contains(p) => right,
                        _ => left,
                    }
                }
                _ => area,
            };
            self.wheel_zoom(ui, &response, view, size, ppp);
        }

        if self.zoom == Zoom::Fit {
            self.pan = Vec2::ZERO;
        }
        // Keep the frame centred along any axis where it is smaller than the
        // view, and stop it being dragged past its edges where it is larger.
        let view = match self.mode {
            ViewMode::SideBySide => Rect::from_min_size(area.min, Vec2::new((area.width() - 8.0) / 2.0, area.height())),
            _ => area,
        };
        let shown = size * scale_for(view, size, self.zoom, ppp);
        let limit = ((shown - view.size()) / 2.0).max(Vec2::ZERO);
        self.pan = self.pan.clamp(-limit, limit);

        match self.mode {
            ViewMode::SideBySide => {
                let gap = 8.0;
                let half = (area.width() - gap) / 2.0;
                let left = Rect::from_min_size(area.min, Vec2::new(half, area.height()));
                let right = Rect::from_min_size(area.min + Vec2::new(half + gap, 0.0), Vec2::new(half, area.height()));
                for (rect, tex, label) in [(left, before, "Before"), (right, after, "After")] {
                    let p = painter.with_clip_rect(rect);
                    if let Some(tex) = tex {
                        p.image(tex.id(), place(rect, size, self.zoom, self.pan, ppp), uv, Color32::WHITE);
                    }
                    caption(&p, rect.left_top() + Vec2::new(8.0, 8.0), egui::Align2::LEFT_TOP, label);
                }
                if response.dragged() && self.zoom != Zoom::Fit {
                    self.pan += response.drag_delta();
                }
            }
            ViewMode::After | ViewMode::Split => {
                let rect = place(area, size, self.zoom, self.pan, ppp);
                if let Some(tex) = after {
                    painter.image(tex.id(), rect, uv, Color32::WHITE);
                }
                let mut on_divider = false;
                if self.mode == ViewMode::Split {
                    let x = rect.min.x + rect.width() * self.split;
                    if let Some(tex) = before {
                        let clip = Rect::from_min_max(rect.min, Pos2::new(x, rect.max.y)).intersect(area);
                        painter.with_clip_rect(clip).image(tex.id(), rect, uv, Color32::WHITE);
                    }
                    // Divider with a grab handle.
                    let line_top = rect.min.y.max(area.min.y);
                    let line_bottom = rect.max.y.min(area.max.y);
                    let handle_rect = Rect::from_min_max(Pos2::new(x - 7.0, line_top), Pos2::new(x + 7.0, line_bottom));
                    let handle = ui.interact(handle_rect, response.id.with("divider"), Sense::drag());
                    on_divider = handle.hovered() || handle.dragged();
                    if on_divider {
                        ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
                    }
                    if handle.dragged() {
                        if let Some(p) = handle.interact_pointer_pos() {
                            self.split = ((p.x - rect.min.x) / rect.width()).clamp(0.0, 1.0);
                        }
                    }
                    let col = if on_divider { Color32::from_rgb(255, 210, 90) } else { Color32::WHITE };
                    painter.line_segment([Pos2::new(x, line_top), Pos2::new(x, line_bottom)], Stroke::new(2.0, col));
                    let mid = Pos2::new(x, (line_top + line_bottom) / 2.0);
                    painter.circle(mid, 9.0, Color32::from_black_alpha(180), Stroke::new(2.0, col));
                    painter.text(mid, egui::Align2::CENTER_CENTER, "↔", egui::FontId::proportional(12.0), col);
                    caption(&painter, area.left_top() + Vec2::new(8.0, 8.0), egui::Align2::LEFT_TOP, "Before");
                    caption(&painter, area.right_top() + Vec2::new(-8.0, 8.0), egui::Align2::RIGHT_TOP, "After");
                }
                if response.dragged() && !on_divider && self.zoom != Zoom::Fit {
                    self.pan += response.drag_delta();
                }
            }
        }
        if self.zoom != Zoom::Fit && response.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
        }
        if response.double_clicked() {
            self.pan = Vec2::ZERO;
        }
        // Apply the same limit to this frame's drag.
        self.pan = self.pan.clamp(-limit, limit);
    }

    /// Step the zoom with the mouse wheel, keeping the image point under the
    /// cursor in place.  `view` is the rectangle the image is centred in.
    fn wheel_zoom(&mut self, ui: &egui::Ui, response: &egui::Response, view: Rect, size: Vec2, ppp: f32) {
        // Raw wheel events: one mouse notch (a "line") is one zoom step,
        // trackpad points accumulate.
        let delta: f32 = ui.input(|i| {
            i.events
                .iter()
                .filter_map(|e| match e {
                    egui::Event::MouseWheel { unit, delta, .. } => Some(match unit {
                        egui::MouseWheelUnit::Point => delta.y,
                        egui::MouseWheelUnit::Line => delta.y * WHEEL_STEP,
                        egui::MouseWheelUnit::Page => delta.y * WHEEL_STEP * 3.0,
                    }),
                    _ => None,
                })
                .sum()
        });
        if delta == 0.0 {
            return;
        }
        self.scroll_acc += delta;
        let steps = (self.scroll_acc / WHEEL_STEP).trunc() as i32;
        if steps == 0 {
            return;
        }
        self.scroll_acc -= steps as f32 * WHEEL_STEP;

        let old_scale = scale_for(view, size, self.zoom, ppp);
        // Current factor in image pixels to screen pixels; "Fit" counts as
        // whatever factor it is showing at.
        let current = old_scale * ppp;
        let mut i = match steps > 0 {
            true => WHEEL_LEVELS.iter().position(|&f| f > current * 1.001).unwrap_or(WHEEL_LEVELS.len() - 1),
            false => WHEEL_LEVELS.iter().rposition(|&f| f < current * 0.999).unwrap_or(0),
        };
        let extra = steps.unsigned_abs() as usize - 1;
        i = if steps > 0 { (i + extra).min(WHEEL_LEVELS.len() - 1) } else { i.saturating_sub(extra) };

        self.zoom = Zoom::from_level(i);
        let new_scale = scale_for(view, size, self.zoom, ppp);
        let pan = if self.zoom == Zoom::Fit { Vec2::ZERO } else { self.pan };
        if let Some(cursor) = response.hover_pos() {
            let from_centre = cursor - view.center() - pan;
            self.pan = cursor - view.center() - from_centre * (new_scale / old_scale);
        }
    }
}
