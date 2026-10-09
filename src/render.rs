use std::thread;

use crate::decode::Image;
use crate::view::Placement;

const BACKGROUND: u32 = 0x1e1e1e;
const CHECKER_LIGHT: u32 = 0x999999;
const CHECKER_DARK: u32 = 0x666666;
const CHECKER_SIZE_LOG2: u32 = 3;
/// Marks a screen row/column that falls outside the image.
const OUTSIDE: u32 = u32::MAX;

/// The image downscaled to its on-screen size, kept between redraws.
pub struct Scaled {
    width: u32,
    height: u32,
    pixels: Vec<u32>,
}

/// Draws `image` at `p` into a `bw`x`bh` 0RGB buffer.
pub fn draw(buf: &mut [u32], bw: u32, bh: u32, image: Option<&Image>, p: Placement, cache: &mut Option<Scaled>) {
    buf.fill(BACKGROUND);
    let Some(img) = image else { return };

    // Zooming out samples a filtered copy at the on-screen size; zooming in samples the original.
    let (src, sw, sh, scale) = if p.zoom < 1.0 {
        let dw = ((img.width as f64 * p.zoom).round() as u32).max(1);
        let dh = ((img.height as f64 * p.zoom).round() as u32).max(1);
        if !cache.as_ref().is_some_and(|c| c.width == dw && c.height == dh) {
            *cache = Some(downscale(img, dw, dh));
        }
        let c = cache.as_ref().unwrap();
        (&c.pixels[..], dw, dh, 1.0)
    } else {
        (&img.pixels[..], img.width, img.height, p.zoom)
    };

    // Nearest source index for each screen pixel center.
    let map = |n: u32, origin: f64, len: u32| -> Vec<u32> {
        (0..n)
            .map(|i| {
                let u = ((i as f64 + 0.5 - origin) / scale).floor();
                if u >= 0.0 && u < len as f64 { u as u32 } else { OUTSIDE }
            })
            .collect()
    };
    let xmap = map(bw, p.x, sw);
    let ymap = map(bh, p.y, sh);
    let Some(x0) = xmap.iter().position(|&x| x != OUTSIDE) else { return };
    let x1 = xmap.iter().rposition(|&x| x != OUTSIDE).unwrap() + 1;
    let xmap = &xmap[x0..x1];

    for (y, &sy) in ymap.iter().enumerate() {
        if sy == OUTSIDE {
            continue;
        }
        let srow = &src[sy as usize * sw as usize..][..sw as usize];
        let drow = &mut buf[y * bw as usize + x0..][..xmap.len()];
        if !img.has_alpha {
            for (d, &sx) in drow.iter_mut().zip(xmap) {
                *d = srow[sx as usize] & 0x00ff_ffff;
            }
            continue;
        }
        let cy = y as u32 >> CHECKER_SIZE_LOG2;
        for (x, (d, &sx)) in drow.iter_mut().zip(xmap).enumerate() {
            let cx = (x0 + x) as u32 >> CHECKER_SIZE_LOG2;
            let bg = if (cx + cy) & 1 == 0 { CHECKER_LIGHT } else { CHECKER_DARK };
            *d = blend(srow[sx as usize], bg);
        }
    }
}

/// Composites a premultiplied pixel over an opaque background.
#[inline]
fn blend(src: u32, bg: u32) -> u32 {
    let a = src >> 24;
    match a {
        255 => src & 0x00ff_ffff,
        0 => bg,
        _ => {
            let mix = |shift: u32| {
                let s = (src >> shift) & 0xff;
                let b = (bg >> shift) & 0xff;
                (s + (b * (255 - a) + 127) / 255).min(255) << shift
            };
            mix(16) | mix(8) | mix(0)
        }
    }
}

/// Box-filter downscale of premultiplied pixels. Rows are split across threads.
fn downscale(img: &Image, dw: u32, dh: u32) -> Scaled {
    let (sw, sh) = (img.width as usize, img.height as usize);
    let (dw_, dh_) = (dw as usize, dh as usize);
    let span = |i: usize, s: usize, d: usize| {
        let lo = i * s / d;
        (lo, ((i + 1) * s / d).max(lo + 1))
    };
    let xspans: Vec<(usize, usize)> = (0..dw_).map(|x| span(x, sw, dw_)).collect();
    let mut pixels = vec![0u32; dw_ * dh_];

    let threads = thread::available_parallelism().map_or(1, |n| n.get());
    let rows_per_chunk = dh_.div_ceil(threads);
    thread::scope(|s| {
        for (chunk_i, chunk) in pixels.chunks_mut(rows_per_chunk * dw_).enumerate() {
            let xspans = &xspans;
            s.spawn(move || {
                for (row_i, out_row) in chunk.chunks_mut(dw_).enumerate() {
                    let (y0, y1) = span(chunk_i * rows_per_chunk + row_i, sh, dh_);
                    for (out, &(x0, x1)) in out_row.iter_mut().zip(xspans) {
                        *out = box_average(img, sw, x0, x1, y0, y1);
                    }
                }
            });
        }
    });
    Scaled { width: dw, height: dh, pixels }
}

#[inline]
fn box_average(img: &Image, sw: usize, x0: usize, x1: usize, y0: usize, y1: usize) -> u32 {
    let mut sums = [0u64; 4];
    for row in y0..y1 {
        for &p in &img.pixels[row * sw + x0..row * sw + x1] {
            for (i, sum) in sums.iter_mut().enumerate() {
                *sum += ((p >> (24 - 8 * i)) & 0xff) as u64;
            }
        }
    }
    let n = ((x1 - x0) * (y1 - y0)) as u64;
    sums.iter().fold(0, |acc, &sum| acc << 8 | ((sum + n / 2) / n) as u32)
}
