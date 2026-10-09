//! Right-click menu. Drawn by rimg into a small pixel buffer that either renderer
//! composites over the image; text uses a font found on the host system.

use std::collections::HashMap;
use std::path::PathBuf;

use fontdue::{Font, FontSettings, Metrics};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Action {
    Fit,
    Fill,
    ActualSize,
    ZoomIn,
    ZoomOut,
    RotateLeft,
    RotateRight,
    FlipHorizontal,
    FlipVertical,
    TogglePause,
    Fullscreen,
    OpenFolder,
}

enum Entry {
    /// Action, label, shortcut hint.
    Item(Action, &'static str, &'static str),
    Separator,
}

fn entries(animated: bool) -> Vec<Entry> {
    use Action::*;
    use Entry::*;
    let mut entries = vec![
        Item(Fit, "Fit to window", "0"),
        Item(Fill, "Fill window", "W"),
        Item(ActualSize, "Actual size (100%)", "1"),
        Item(ZoomIn, "Zoom in", "+"),
        Item(ZoomOut, "Zoom out", "\u{2212}"),
        Separator,
        Item(RotateLeft, "Rotate left", "["),
        Item(RotateRight, "Rotate right", "]"),
        Item(FlipHorizontal, "Flip horizontal", "H"),
        Item(FlipVertical, "Flip vertical", "V"),
        Separator,
    ];
    if animated {
        entries.push(Item(TogglePause, "Pause / Play", "Space"));
    }
    entries.extend([
        Item(Fullscreen, "Fullscreen", "F"),
        Separator,
        Item(OpenFolder, "Open containing folder", ""),
    ]);
    entries
}

// Sizes in logical pixels, scaled by the window's scale factor.
const FONT_SIZE: f32 = 14.0;
const ITEM_HEIGHT: f32 = 26.0;
const SEPARATOR_HEIGHT: f32 = 9.0;
const PAD_X: f32 = 12.0;
const PAD_Y: f32 = 4.0;
const SHORTCUT_GAP: f32 = 32.0;

const BACKGROUND: u32 = 0x2b2b2b;
const BORDER: u32 = 0x4a4a4a;
const HOVER: u32 = 0x3a6fb5;
const TEXT: u32 = 0xe8e8e8;
const SHORTCUT_TEXT: u32 = 0x9c9c9c;
const SHORTCUT_TEXT_HOVER: u32 = 0xd0d8e8;

/// A font with a cache of rasterized glyphs.
pub struct Text {
    font: Font,
    glyphs: HashMap<(char, u32), (Metrics, Vec<u8>)>,
}

impl Text {
    /// Loads the system's default sans-serif font.
    pub fn load() -> Result<Text, String> {
        for path in font_candidates() {
            let Ok(bytes) = std::fs::read(&path) else { continue };
            if let Ok(font) = Font::from_bytes(bytes, FontSettings::default()) {
                return Ok(Text { font, glyphs: HashMap::new() });
            }
        }
        Err("no usable system font found".into())
    }

    fn glyph(&mut self, c: char, px: f32) -> &(Metrics, Vec<u8>) {
        let font = &self.font;
        self.glyphs.entry((c, px.to_bits())).or_insert_with(|| font.rasterize(c, px))
    }

    fn width(&mut self, s: &str, px: f32) -> f32 {
        s.chars().map(|c| self.glyph(c, px).0.advance_width).sum()
    }

    /// Draws `s` starting at `x` with its baseline at `baseline`, blending onto opaque pixels.
    fn draw(&mut self, buf: &mut [u32], buf_width: u32, (x, baseline): (f32, f32), s: &str, px: f32, color: u32) {
        let buf_height = buf.len() as i32 / buf_width as i32;
        let mut pen = x;
        for c in s.chars() {
            let (metrics, coverage) = self.glyph(c, px);
            let left = (pen + metrics.xmin as f32).round() as i32;
            let top = (baseline - metrics.height as f32 - metrics.ymin as f32).round() as i32;
            for (row, line) in coverage.chunks(metrics.width.max(1)).enumerate() {
                let y = top + row as i32;
                if y < 0 || y >= buf_height {
                    continue;
                }
                for (col, &alpha) in line.iter().enumerate() {
                    let x = left + col as i32;
                    if alpha == 0 || x < 0 || x >= buf_width as i32 {
                        continue;
                    }
                    let d = &mut buf[(y * buf_width as i32 + x) as usize];
                    *d = lerp(*d, color, alpha as u32);
                }
            }
            pen += metrics.advance_width;
        }
    }
}

fn lerp(from: u32, to: u32, t: u32) -> u32 {
    let mix = |shift: u32| {
        let (a, b) = ((from >> shift) & 0xff, (to >> shift) & 0xff);
        ((a * (255 - t) + b * t + 127) / 255) << shift
    };
    mix(16) | mix(8) | mix(0)
}

/// Where to look for a regular sans-serif font, best first.
fn font_candidates() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        // fontconfig knows the desktop's configured default.
        if let Ok(out) = std::process::Command::new("fc-match").args(["--format=%{file}", "sans-serif:style=Regular"]).output()
            && out.status.success()
        {
            paths.push(PathBuf::from(String::from_utf8_lossy(&out.stdout).into_owned()));
        }
        paths.extend(
            [
                "/usr/share/fonts/noto/NotoSans-Regular.ttf",
                "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
                "/usr/share/fonts/TTF/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            ]
            .map(PathBuf::from),
        );
    }
    #[cfg(target_os = "macos")]
    paths.extend(
        [
            "/System/Library/Fonts/SFNS.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
        ]
        .map(PathBuf::from),
    );
    #[cfg(windows)]
    {
        let fonts = std::env::var_os("WINDIR").map_or_else(|| PathBuf::from(r"C:\Windows"), PathBuf::from).join("Fonts");
        paths.extend(["segoeui.ttf", "arial.ttf"].map(|f| fonts.join(f)));
    }
    paths
}

pub struct Menu {
    entries: Vec<Entry>,
    /// Top-left corner in window pixels.
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
    /// Rendered menu, opaque 0x00RRGGBB.
    pub pixels: Vec<u32>,
    /// Increments whenever `pixels` changes, so the GPU knows to re-upload.
    pub version: u64,
    hovered: Option<usize>,
    /// Top edge and height of each entry, relative to the menu.
    rows: Vec<(f32, f32)>,
    scale: f32,
}

impl Menu {
    /// Opens a menu at `at` (window pixels), shifted to stay inside the window.
    pub fn open(text: &mut Text, animated: bool, at: (f64, f64), window: (u32, u32), scale: f64, version: u64) -> Menu {
        let scale = scale as f32;
        let entries = entries(animated);
        let px = FONT_SIZE * scale;
        let (mut label_w, mut shortcut_w) = (0.0f32, 0.0f32);
        let mut rows = Vec::new();
        let mut y = PAD_Y * scale;
        for entry in &entries {
            let h = match entry {
                Entry::Item(_, label, shortcut) => {
                    label_w = label_w.max(text.width(label, px));
                    shortcut_w = shortcut_w.max(text.width(shortcut, px));
                    ITEM_HEIGHT * scale
                }
                Entry::Separator => SEPARATOR_HEIGHT * scale,
            };
            rows.push((y, h));
            y += h;
        }
        let width = (2.0 * PAD_X * scale + label_w + SHORTCUT_GAP * scale + shortcut_w).ceil() as u32;
        let height = (y + PAD_Y * scale).ceil() as u32;

        // Open toward the bottom-right of the cursor, flipping where it would overflow.
        let place = |at: f64, size: u32, limit: u32| {
            let at = at.round() as i32;
            if at + size as i32 <= limit as i32 { at } else { (at - size as i32).max(0) }
        };
        let mut menu = Menu {
            entries,
            x: place(at.0, width, window.0),
            y: place(at.1, height, window.1),
            width,
            height,
            pixels: Vec::new(),
            version,
            hovered: None,
            rows,
            scale,
        };
        menu.render(text);
        menu
    }

    fn render(&mut self, text: &mut Text) {
        let (w, h) = (self.width as usize, self.height as usize);
        self.pixels = vec![BACKGROUND; w * h];
        for x in 0..w {
            self.pixels[x] = BORDER;
            self.pixels[(h - 1) * w + x] = BORDER;
        }
        for y in 0..h {
            self.pixels[y * w] = BORDER;
            self.pixels[y * w + w - 1] = BORDER;
        }

        let px = FONT_SIZE * self.scale;
        let line = text.font.horizontal_line_metrics(px);
        let (ascent, descent) = line.map_or((px * 0.8, -px * 0.2), |m| (m.ascent, m.descent));
        for (i, (entry, &(top, height))) in self.entries.iter().zip(&self.rows).enumerate() {
            let (top_px, bottom_px) = (top.round() as usize, (top + height).round() as usize);
            match entry {
                Entry::Separator => {
                    let y = (top + height / 2.0) as usize;
                    let inset = (PAD_X * self.scale / 2.0) as usize;
                    self.pixels[y * w + inset..y * w + w - inset].fill(BORDER);
                }
                Entry::Item(_, label, shortcut) => {
                    let hovered = self.hovered == Some(i);
                    if hovered {
                        for y in top_px..bottom_px {
                            self.pixels[y * w + 1..y * w + w - 1].fill(HOVER);
                        }
                    }
                    // Center the font's ascent-to-descent box in the row.
                    let baseline = top + (height + ascent + descent) / 2.0;
                    let pad = PAD_X * self.scale;
                    text.draw(&mut self.pixels, self.width, (pad, baseline), label, px, TEXT);
                    let shortcut_x = w as f32 - pad - text.width(shortcut, px);
                    let color = if hovered { SHORTCUT_TEXT_HOVER } else { SHORTCUT_TEXT };
                    text.draw(&mut self.pixels, self.width, (shortcut_x, baseline), shortcut, px, color);
                }
            }
        }
        self.version += 1;
    }

    fn contains(&self, (x, y): (f64, f64)) -> bool {
        let (x, y) = (x - self.x as f64, y - self.y as f64);
        x >= 0.0 && y >= 0.0 && x < self.width as f64 && y < self.height as f64
    }

    /// The item under the cursor (window pixels), if any.
    pub fn action_at(&self, pos: (f64, f64)) -> Option<Action> {
        self.item_at(pos).map(|i| self.action(i))
    }

    fn item_at(&self, pos: (f64, f64)) -> Option<usize> {
        if !self.contains(pos) {
            return None;
        }
        let y = (pos.1 - self.y as f64) as f32;
        let i = self.rows.iter().position(|&(top, h)| y >= top && y < top + h)?;
        matches!(self.entries[i], Entry::Item(..)).then_some(i)
    }

    fn action(&self, i: usize) -> Action {
        match self.entries[i] {
            Entry::Item(action, ..) => action,
            Entry::Separator => unreachable!("separators are never hovered"),
        }
    }

    /// Highlights the item under the cursor. Returns whether the menu changed.
    pub fn hover(&mut self, text: &mut Text, pos: (f64, f64)) -> bool {
        self.set_hovered(text, self.item_at(pos))
    }

    /// Moves the highlight by `step` items (keyboard), skipping separators.
    pub fn step(&mut self, text: &mut Text, step: isize) -> bool {
        let items: Vec<usize> =
            self.entries.iter().enumerate().filter(|(_, e)| matches!(e, Entry::Item(..))).map(|(i, _)| i).collect();
        let current = self.hovered.and_then(|h| items.iter().position(|&i| i == h));
        let next = match current {
            Some(c) => (c as isize + step).rem_euclid(items.len() as isize) as usize,
            None if step > 0 => 0,
            None => items.len() - 1,
        };
        self.set_hovered(text, Some(items[next]))
    }

    pub fn hovered_action(&self) -> Option<Action> {
        self.hovered.map(|i| self.action(i))
    }

    fn set_hovered(&mut self, text: &mut Text, hovered: Option<usize>) -> bool {
        if hovered == self.hovered {
            return false;
        }
        self.hovered = hovered;
        self.render(text);
        true
    }
}
