//! Zoom, pan and orientation state, independent of how the image gets drawn.
//! All coordinates are physical window pixels. Sizes passed in are the image as
//! displayed, i.e. after `Orientation::display_size`.

pub const MAX_ZOOM: f64 = 64.0;
/// Screen pixels of the image that panning always leaves visible.
const MIN_VISIBLE: f64 = 64.0;
/// Zooming out stops when the image's longest side reaches this many screen pixels.
const MIN_SIZE: f64 = 32.0;

/// Where the image lands on screen: `zoom` screen pixels per image pixel,
/// with the displayed image's top-left corner at (`x`, `y`).
#[derive(Clone, Copy)]
pub struct Placement {
    pub zoom: f64,
    pub x: f64,
    pub y: f64,
    pub orientation: Orientation,
}

/// How the image is turned on screen: mirrored horizontally if `flip`, then
/// rotated clockwise by `turns` quarter turns. Display only; files are untouched.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct Orientation {
    pub turns: u8,
    pub flip: bool,
}

impl Orientation {
    pub fn display_size(self, (w, h): Size) -> Size {
        if self.turns % 2 == 1 { (h, w) } else { (w, h) }
    }

    /// Maps a point on the displayed image to the source image, both in image pixels.
    /// `src` is the source image size.
    pub fn to_source(self, (u, v): (f64, f64), (w, h): Size) -> (f64, f64) {
        let (x, y) = match self.turns {
            0 => (u, v),
            1 => (v, h - u),
            2 => (w - u, h - v),
            _ => (w - v, u),
        };
        if self.flip { (w - x, y) } else { (x, y) }
    }

    /// Maps a point on the source image to the displayed image; inverse of `to_source`.
    pub fn to_display(self, (x, y): (f64, f64), (w, h): Size) -> (f64, f64) {
        let x = if self.flip { w - x } else { x };
        match self.turns {
            0 => (x, y),
            1 => (h - y, x),
            2 => (w - x, h - y),
            _ => (y, w - x),
        }
    }

    fn rotated(self, clockwise: bool) -> Self {
        Orientation { turns: (self.turns + if clockwise { 1 } else { 3 }) % 4, ..self }
    }

    /// Mirrors what's on screen left-to-right. On a quarter-turned image that's
    /// a vertical mirror of the source, i.e. a horizontal one plus a half turn.
    fn flipped_horizontally(self) -> Self {
        let turns = if self.turns % 2 == 1 { (self.turns + 2) % 4 } else { self.turns };
        Orientation { turns, flip: !self.flip }
    }

    fn flipped_vertically(self) -> Self {
        self.flipped_horizontally().rotated(true).rotated(true)
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Mode {
    /// Centered, shrunk to fit the window but never enlarged. Follows window resizes.
    Fit,
    /// Centered, scaled up or down until one side touches the window edge. Follows window resizes.
    Fill,
    /// Explicit zoom and offset, set by zooming or panning.
    Manual,
}

pub struct View {
    mode: Mode,
    zoom: f64,
    offset: (f64, f64),
    /// Clockwise quarter turns, kept apart from the mirrors so flips can be
    /// reset without undoing rotation.
    turns: u8,
    /// Mirrors applied to what's on screen after rotating: (horizontal, vertical).
    mirrored: (bool, bool),
}

type Size = (f64, f64);

/// Zoom that makes the image touch the window edges on one axis.
fn fill_zoom(win: Size, img: Size) -> f64 {
    (win.0 / img.0).min(win.1 / img.1)
}

/// Zoom that fits the image in the window, never upscaling.
fn fit_zoom(win: Size, img: Size) -> f64 {
    fill_zoom(win, img).min(1.0)
}

impl View {
    pub fn new() -> View {
        View { mode: Mode::Fit, zoom: 1.0, offset: (0.0, 0.0), turns: 0, mirrored: (false, false) }
    }

    pub fn orientation(&self) -> Orientation {
        let mut o = Orientation { turns: self.turns, flip: false };
        if self.mirrored.0 {
            o = o.flipped_horizontally();
        }
        if self.mirrored.1 {
            o = o.flipped_vertically();
        }
        o
    }

    pub fn placement(&self, win: Size, img: Size) -> Placement {
        let orientation = self.orientation();
        let centered = |zoom: f64| Placement {
            zoom,
            x: (win.0 - img.0 * zoom) / 2.0,
            y: (win.1 - img.1 * zoom) / 2.0,
            orientation,
        };
        match self.mode {
            Mode::Fit => centered(fit_zoom(win, img)),
            Mode::Fill => centered(fill_zoom(win, img).min(MAX_ZOOM)),
            Mode::Manual => Placement { zoom: self.zoom, x: self.offset.0, y: self.offset.1, orientation },
        }
    }

    /// Turns the image a quarter turn. `img` is the displayed size before turning.
    /// A zoomed or panned image keeps its zoom and is re-centered.
    pub fn rotate(&mut self, clockwise: bool, win: Size, img: Size) {
        self.turns = (self.turns + if clockwise { 1 } else { 3 }) % 4;
        // Turning a mirrored image a quarter turn is the same as turning it and
        // then mirroring along the other axis.
        self.mirrored = (self.mirrored.1, self.mirrored.0);
        let img = (img.1, img.0);
        self.offset = ((win.0 - img.0 * self.zoom) / 2.0, (win.1 - img.1 * self.zoom) / 2.0);
    }

    pub fn flip_horizontally(&mut self) {
        self.mirrored.0 = !self.mirrored.0;
    }

    pub fn flip_vertically(&mut self) {
        self.mirrored.1 = !self.mirrored.1;
    }

    /// Multiplies zoom by `factor`, keeping the image point under `anchor` fixed.
    pub fn zoom_at(&mut self, factor: f64, anchor: (f64, f64), win: Size, img: Size) {
        let p = self.placement(win, img);
        let fit = fit_zoom(win, img);
        let min = (MIN_SIZE / img.0.max(img.1)).min(fit);
        let mut zoom = (p.zoom * factor).clamp(min, MAX_ZOOM);
        // Stop at exactly 100% or the fit zoom when passing through it (the nearer one first).
        let crossed = [1.0, fit].into_iter().filter(|&stop| (p.zoom - stop) * (zoom - stop) < 0.0);
        if let Some(stop) = crossed.min_by(|a, b| (a - p.zoom).abs().total_cmp(&(b - p.zoom).abs())) {
            zoom = stop;
        }
        self.mode = Mode::Manual;
        self.zoom = zoom;
        self.offset = (
            anchor.0 - (anchor.0 - p.x) * zoom / p.zoom,
            anchor.1 - (anchor.1 - p.y) * zoom / p.zoom,
        );
        self.keep_visible(win, img);
    }

    pub fn pan(&mut self, dx: f64, dy: f64, win: Size, img: Size) {
        let p = self.placement(win, img);
        self.mode = Mode::Manual;
        self.zoom = p.zoom;
        self.offset = (p.x + dx, p.y + dy);
        self.keep_visible(win, img);
    }

    pub fn fit(&mut self) {
        self.mode = Mode::Fit;
    }

    /// Also undoes any flips; rotation stays.
    pub fn fill(&mut self) {
        self.mode = Mode::Fill;
        self.mirrored = (false, false);
    }

    /// Shows the image centered at 100%, undoing any flips; rotation stays.
    pub fn actual_size(&mut self, win: Size, img: Size) {
        self.mirrored = (false, false);
        self.mode = Mode::Manual;
        self.zoom = 1.0;
        self.offset = ((win.0 - img.0) / 2.0, (win.1 - img.1) / 2.0);
    }

    pub fn resized(&mut self, win: Size, img: Size) {
        if self.mode == Mode::Manual {
            self.keep_visible(win, img);
        }
    }

    /// Panning is free, but at least `MIN_VISIBLE` pixels of the image stay on screen.
    fn keep_visible(&mut self, win: Size, img: Size) {
        let axis = |offset: f64, win: f64, img: f64| {
            let size = img * self.zoom;
            let margin = size.min(win).min(MIN_VISIBLE);
            offset.clamp(margin - size, win - margin)
        };
        self.offset = (axis(self.offset.0, win.0, img.0), axis(self.offset.1, win.1, img.1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WIN: Size = (800.0, 600.0);
    const BIG: Size = (4000.0, 2000.0);
    const SMALL: Size = (200.0, 100.0);

    fn image_point_under(p: Placement, screen: (f64, f64)) -> (f64, f64) {
        ((screen.0 - p.x) / p.zoom, (screen.1 - p.y) / p.zoom)
    }

    #[test]
    fn fit_centers_and_never_upscales() {
        let v = View::new();
        let p = v.placement(WIN, BIG);
        assert_eq!(p.zoom, 0.2);
        assert_eq!((p.x, p.y), (0.0, 100.0));
        let p = v.placement(WIN, SMALL);
        assert_eq!(p.zoom, 1.0);
        assert_eq!((p.x, p.y), (300.0, 250.0));
    }

    #[test]
    fn zoom_keeps_anchor_fixed() {
        let mut v = View::new();
        v.actual_size(WIN, BIG); // both axes overflow the window, so neither gets centered
        let before = image_point_under(v.placement(WIN, BIG), (100.0, 400.0));
        v.zoom_at(1.25, (100.0, 400.0), WIN, BIG);
        let after = image_point_under(v.placement(WIN, BIG), (100.0, 400.0));
        assert!((before.0 - after.0).abs() < 1e-9 && (before.1 - after.1).abs() < 1e-9);
    }

    #[test]
    fn zoom_snaps_to_100_percent_and_fit() {
        let mut v = View::new();
        v.zoom_at(3.0, (0.0, 0.0), WIN, BIG); // 0.2 -> 0.6
        v.zoom_at(3.0, (0.0, 0.0), WIN, BIG); // would be 1.8, stops at 1.0
        assert_eq!(v.placement(WIN, BIG).zoom, 1.0);
        v.zoom_at(0.1, (0.0, 0.0), WIN, BIG); // would be 0.1, stops at fit (0.2)
        assert_eq!(v.placement(WIN, BIG).zoom, 0.2);
    }

    #[test]
    fn zoom_out_below_100_percent_on_small_images() {
        let mut v = View::new();
        v.zoom_at(0.5, (400.0, 300.0), WIN, SMALL);
        assert_eq!(v.placement(WIN, SMALL).zoom, 0.5);
        v.zoom_at(1e-6, (400.0, 300.0), WIN, SMALL); // longest side bottoms out at MIN_SIZE
        assert_eq!(v.placement(WIN, SMALL).zoom * SMALL.0, MIN_SIZE);
    }

    #[test]
    fn actual_size_centers() {
        let mut v = View::new();
        v.pan(123.0, -45.0, WIN, BIG);
        v.actual_size(WIN, BIG);
        let p = v.placement(WIN, BIG);
        assert_eq!((p.zoom, p.x, p.y), (1.0, -1600.0, -700.0));
    }

    #[test]
    fn pan_works_at_fit_and_on_small_images() {
        for img in [BIG, SMALL] {
            let mut v = View::new();
            let before = v.placement(WIN, img);
            v.pan(30.0, -20.0, WIN, img);
            let after = v.placement(WIN, img);
            assert_eq!((after.zoom, after.x, after.y), (before.zoom, before.x + 30.0, before.y - 20.0));
        }
    }

    #[test]
    fn pan_keeps_part_of_image_visible() {
        let mut v = View::new();
        v.actual_size(WIN, BIG);
        v.pan(1e6, 1e6, WIN, BIG);
        let p = v.placement(WIN, BIG);
        assert_eq!((p.x, p.y), (WIN.0 - MIN_VISIBLE, WIN.1 - MIN_VISIBLE));
        v.pan(-1e6, -1e6, WIN, BIG);
        let p = v.placement(WIN, BIG);
        assert_eq!((p.x, p.y), (MIN_VISIBLE - BIG.0, MIN_VISIBLE - BIG.1));
    }

    #[test]
    fn zooming_out_never_recenters() {
        let mut v = View::new();
        v.zoom_at(2.0, (100.0, 100.0), WIN, SMALL); // 200%, off-center
        let before = image_point_under(v.placement(WIN, SMALL), (100.0, 100.0));
        v.zoom_at(0.5, (100.0, 100.0), WIN, SMALL); // back to 100% (= fit)
        assert_eq!(v.mode, Mode::Manual);
        let after = image_point_under(v.placement(WIN, SMALL), (100.0, 100.0));
        assert!((before.0 - after.0).abs() < 1e-9 && (before.1 - after.1).abs() < 1e-9);
    }

    #[test]
    fn fill_scales_up_and_down_to_touch_edges() {
        let mut v = View::new();
        v.fill();
        let p = v.placement(WIN, SMALL); // 200x100 in 800x600: width limits, 4x
        assert_eq!((p.zoom, p.x, p.y), (4.0, 0.0, 100.0));
        let p = v.placement(WIN, BIG); // 4000x2000: width limits, 0.2x
        assert_eq!((p.zoom, p.x, p.y), (0.2, 0.0, 100.0));
        let p = v.placement((800.0, 100.0), SMALL); // height limits, 1x
        assert_eq!((p.zoom, p.x, p.y), (1.0, 300.0, 0.0));
    }

    /// Where a source pixel's center lands on the displayed image, by brute force.
    fn displayed_position(o: Orientation, src: Size, (x, y): (f64, f64)) -> (f64, f64) {
        let (dw, dh) = o.display_size(src);
        for v in 0..dh as usize {
            for u in 0..dw as usize {
                let (sx, sy) = o.to_source((u as f64 + 0.5, v as f64 + 0.5), src);
                if (sx.floor(), sy.floor()) == (x, y) {
                    return (u as f64, v as f64);
                }
            }
        }
        panic!("source pixel ({x}, {y}) not displayed");
    }

    #[test]
    fn orientation_moves_corners_where_expected() {
        let src = (4.0, 2.0);
        let top_left = (0.0, 0.0);
        let top_right = (3.0, 0.0);
        let cw = Orientation::default().rotated(true);
        assert_eq!(cw.display_size(src), (2.0, 4.0));
        assert_eq!(displayed_position(cw, src, top_left), (1.0, 0.0)); // goes to the top-right
        assert_eq!(displayed_position(cw, src, top_right), (1.0, 3.0)); // goes to the bottom-right
        let ccw = Orientation::default().rotated(false);
        assert_eq!(displayed_position(ccw, src, top_left), (0.0, 3.0)); // goes to the bottom-left
        let mirrored = Orientation::default().flipped_horizontally();
        assert_eq!(displayed_position(mirrored, src, top_left), (3.0, 0.0));
        let upside_down = Orientation::default().flipped_vertically();
        assert_eq!(displayed_position(upside_down, src, top_left), (0.0, 1.0));
    }

    #[test]
    fn flips_mirror_the_screen_even_when_rotated() {
        let src = (4.0, 2.0);
        let cw = Orientation::default().rotated(true);
        let (dw, dh) = cw.display_size(src);
        for pixel in [(0.0, 0.0), (3.0, 0.0), (2.0, 1.0)] {
            let (u, v) = displayed_position(cw, src, pixel);
            assert_eq!(displayed_position(cw.flipped_horizontally(), src, pixel), (dw - 1.0 - u, v));
            assert_eq!(displayed_position(cw.flipped_vertically(), src, pixel), (u, dh - 1.0 - v));
        }
        let o = cw.flipped_horizontally();
        assert_eq!(o.flipped_horizontally(), cw);
        assert_eq!(cw.rotated(true).rotated(true).rotated(true).rotated(true), cw);
    }

    #[test]
    fn to_display_inverts_to_source() {
        let src = (4.0, 2.0);
        for turns in 0..4 {
            for flip in [false, true] {
                let o = Orientation { turns, flip };
                let p = (1.25, 0.5);
                let (x, y) = o.to_source(o.to_display(p, src), src);
                assert!((x - p.0).abs() < 1e-12 && (y - p.1).abs() < 1e-12, "{o:?}");
            }
        }
    }

    #[test]
    fn rotating_a_flipped_image_matches_flipping_a_rotated_one() {
        // Flip, then rotate...
        let mut a = View::new();
        a.flip_horizontally();
        a.rotate(true, WIN, BIG);
        // ...looks the same as rotating, then flipping along the other axis.
        let mut b = View::new();
        b.rotate(true, WIN, BIG);
        b.flip_vertically();
        assert_eq!(a.orientation(), b.orientation());
    }

    #[test]
    fn fill_and_actual_size_reset_flips_but_keep_rotation() {
        let rotated = {
            let mut v = View::new();
            v.rotate(true, WIN, BIG);
            v.orientation()
        };
        for reset in [View::fill as fn(&mut View), |v: &mut View| v.actual_size(WIN, BIG)] {
            let mut v = View::new();
            v.rotate(true, WIN, BIG);
            v.flip_horizontally();
            v.flip_vertically();
            assert_ne!(v.orientation(), rotated);
            reset(&mut v);
            assert_eq!(v.orientation(), rotated);
        }
    }
}
