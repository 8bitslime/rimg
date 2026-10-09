//! Format detection and decoding. `image` handles most formats; JPEG XL goes to
//! jxl-oxide and HEIF/HEIC to libheif.

use std::io::Cursor;
use std::path::Path;
use std::time::Duration;

use image::codecs::gif::GifDecoder;
use image::codecs::png::PngDecoder;
use image::codecs::webp::WebPDecoder;
use image::{AnimationDecoder, DynamicImage, ImageDecoder, ImageFormat, ImageReader};
use jxl_oxide::{EnumColourEncoding, InitializeResult, JxlImage, RenderingIntent};
use libheif_rs::{ColorSpace, FileTypeResult, HeifContext, LibHeif, RgbChroma};

/// A decoded image in premultiplied-alpha 0xAARRGGBB pixels, row-major.
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u32>,
    pub has_alpha: bool,
}

impl Image {
    /// Takes straight-alpha pixels and premultiplies them, so filtering never
    /// bleeds the color of fully transparent pixels.
    fn new(width: u32, height: u32, mut pixels: Vec<u32>, has_alpha: bool) -> Image {
        if has_alpha {
            for p in &mut pixels {
                let a = *p >> 24;
                if a == 255 {
                    continue;
                }
                let mul = |shift: u32| ((((*p >> shift) & 0xff) * a + 127) / 255) << shift;
                *p = a << 24 | mul(16) | mul(8) | mul(0);
            }
        }
        Image { width, height, pixels, has_alpha }
    }
}

pub type Result<T> = std::result::Result<T, String>;

enum Kind {
    Jxl,
    Heif,
    /// Anything else; `image` sniffs the exact format.
    Generic,
}

fn kind(data: &[u8]) -> Kind {
    if data.starts_with(&[0xff, 0x0a]) || data.starts_with(b"\0\0\0\x0cJXL \r\n\x87\n") {
        Kind::Jxl
    } else if image::guess_format(data).is_err() && libheif_rs::check_file_type(data) == FileTypeResult::Supported {
        // AVIF is also HEIF-based, but `image` claims it first.
        Kind::Heif
    } else {
        Kind::Generic
    }
}

/// `image` sniffs formats from content; fall back to the extension for ones with no magic bytes (TGA).
fn reader<'a>(data: &'a [u8], path: &Path) -> Result<ImageReader<Cursor<&'a [u8]>>> {
    let mut reader = ImageReader::new(Cursor::new(data)).with_guessed_format().map_err(|e| e.to_string())?;
    if reader.format().is_none() {
        match ImageFormat::from_path(path) {
            Ok(format) => reader.set_format(format),
            Err(_) => return Err("unrecognized image format".into()),
        }
    }
    Ok(reader)
}

/// Reads just the (oriented) dimensions, without decoding pixels.
pub fn probe(data: &[u8], path: &Path) -> Option<(u32, u32)> {
    if matches!(image::guess_format(data), Ok(ImageFormat::Avif)) {
        // The AVIF decoder decodes the whole image on construction, so read the header ourselves.
        return isobmff_dims(data);
    }
    match kind(data) {
        Kind::Jxl => {
            let mut uninit = JxlImage::builder().build_uninit();
            uninit.feed_bytes(data).ok()?;
            match uninit.try_init().ok()? {
                InitializeResult::Initialized(img) => Some((img.width(), img.height())),
                InitializeResult::NeedMoreData(_) => None,
            }
        }
        // Avoids libheif's plugin loading, which costs ~10ms on first use.
        Kind::Heif => isobmff_dims(data),
        Kind::Generic => {
            let mut decoder = reader(data, path).ok()?.into_decoder().ok()?;
            let (w, h) = decoder.dimensions();
            let swapped = decoder.orientation().is_ok_and(|o| {
                use image::metadata::Orientation::*;
                matches!(o, Rotate90 | Rotate270 | Rotate90FlipH | Rotate270FlipH)
            });
            Some(if swapped { (h, w) } else { (w, h) })
        }
    }
}

/// Decodes `data`. `path` is only consulted for its extension when the content isn't recognized.
pub fn decode(data: &[u8], path: &Path) -> Result<Image> {
    match kind(data) {
        Kind::Jxl => decode_jxl(data),
        Kind::Heif => decode_heif(data),
        Kind::Generic => decode_generic(data, path),
    }
}

/// An animation frame (composited onto the full canvas) and its display time.
pub type Frame = (Image, Duration);

/// Returns the frames of an animated GIF, APNG or WebP, decoded lazily as the
/// iterator advances. GIFs always count as animations, possibly of one frame.
pub fn animation(data: &[u8]) -> Option<Box<dyn Iterator<Item = Result<Frame>> + '_>> {
    let cursor = Cursor::new(data);
    let frames = match image::guess_format(data).ok()? {
        ImageFormat::Gif => GifDecoder::new(cursor).ok()?.into_frames(),
        ImageFormat::Png => {
            let decoder = PngDecoder::new(cursor).ok()?;
            if !decoder.is_apng().ok()? {
                return None;
            }
            decoder.apng().ok()?.into_frames()
        }
        ImageFormat::WebP => {
            let decoder = WebPDecoder::new(cursor).ok()?;
            if !decoder.has_animation() {
                return None;
            }
            decoder.into_frames()
        }
        _ => return None,
    };
    Some(Box::new(frames.map(|frame| {
        let frame = frame.map_err(|e| e.to_string())?;
        let (numer, denom) = frame.delay().numer_denom_ms();
        let delay = Duration::from_secs_f64(numer as f64 / denom.max(1) as f64 / 1000.0);
        let rgba = frame.into_buffer();
        let (width, height) = rgba.dimensions();
        let pixels: Vec<u32> = rgba.pixels().map(|p| pack(p.0)).collect();
        let has_alpha = pixels.iter().any(|p| p >> 24 != 255);
        Ok((Image::new(width, height, pixels, has_alpha), delay))
    })))
}

fn decode_generic(data: &[u8], path: &Path) -> Result<Image> {
    let mut decoder = reader(data, path)?.into_decoder().map_err(|e| e.to_string())?;
    let orientation = decoder.orientation().ok();
    let mut img = DynamicImage::from_decoder(decoder).map_err(|e| e.to_string())?;
    if let Some(o) = orientation {
        img.apply_orientation(o);
    }
    let has_alpha = img.color().has_alpha();
    let rgba = img.into_rgba8();
    let (width, height) = rgba.dimensions();
    let pixels = rgba.pixels().map(|p| pack(p.0)).collect();
    Ok(Image::new(width, height, pixels, has_alpha))
}

fn decode_jxl(data: &[u8]) -> Result<Image> {
    let mut img = JxlImage::builder().read(data).map_err(|e| e.to_string())?;
    img.request_color_encoding(EnumColourEncoding::srgb(RenderingIntent::Relative));
    let render = img.render_frame(0).map_err(|e| e.to_string())?;
    let mut stream = render.stream();
    let (width, height, channels) = (stream.width(), stream.height(), stream.channels() as usize);
    let mut buf = vec![0u8; width as usize * height as usize * channels];
    stream.write_to_buffer(&mut buf);

    let pixels = buf
        .chunks_exact(channels)
        .map(|c| match channels {
            1 => pack([c[0], c[0], c[0], 255]),
            2 => pack([c[0], c[0], c[0], c[1]]),
            3 => pack([c[0], c[1], c[2], 255]),
            _ => pack([c[0], c[1], c[2], c[3]]),
        })
        .collect();
    Ok(Image::new(width, height, pixels, channels == 2 || channels >= 4))
}

fn decode_heif(data: &[u8]) -> Result<Image> {
    let ctx = HeifContext::read_from_bytes(data).map_err(|e| e.to_string())?;
    let handle = ctx.primary_image_handle().map_err(|e| e.to_string())?;
    let has_alpha = handle.has_alpha_channel();
    let decoded = LibHeif::new()
        .decode(&handle, ColorSpace::Rgb(RgbChroma::Rgba), None)
        .map_err(|e| e.to_string())?;
    let Some(plane) = decoded.planes().interleaved else {
        return Err("libheif returned no interleaved plane".into());
    };

    let (width, height) = (plane.width, plane.height);
    let mut pixels = Vec::with_capacity(width as usize * height as usize);
    for row in plane.data.chunks(plane.stride).take(height as usize) {
        pixels.extend(row[..width as usize * 4].chunks_exact(4).map(|c| pack([c[0], c[1], c[2], c[3]])));
    }
    Ok(Image::new(width, height, pixels, has_alpha))
}

/// Reads the image size from the `ispe` properties of a HEIF/AVIF file. Thumbnails and grid
/// tiles carry their own smaller `ispe`, so the largest one is the primary image.
fn isobmff_dims(data: &[u8]) -> Option<(u32, u32)> {
    let meta = find_box(data, b"meta")?;
    let iprp = find_box(meta.get(4..)?, b"iprp")?;
    let ipco = find_box(iprp, b"ipco")?;

    let mut best: Option<(u32, u32)> = None;
    let mut rotated = false;
    for (kind, body) in boxes(ipco) {
        match kind {
            b"ispe" => {
                let w = u32::from_be_bytes(body.get(4..8)?.try_into().unwrap());
                let h = u32::from_be_bytes(body.get(8..12)?.try_into().unwrap());
                if best.is_none_or(|(bw, bh)| w as u64 * h as u64 > bw as u64 * bh as u64) {
                    best = Some((w, h));
                }
            }
            b"irot" => rotated |= body.first().is_some_and(|a| a & 1 == 1),
            _ => {}
        }
    }
    best.map(|(w, h)| if rotated { (h, w) } else { (w, h) })
}

fn find_box<'a>(data: &'a [u8], kind: &[u8; 4]) -> Option<&'a [u8]> {
    boxes(data).find(|(k, _)| *k == kind).map(|(_, body)| body)
}

/// Iterates over ISO-BMFF boxes as (type, body).
fn boxes(mut data: &[u8]) -> impl Iterator<Item = (&[u8; 4], &[u8])> {
    std::iter::from_fn(move || {
        let size = u32::from_be_bytes(data.get(0..4)?.try_into().unwrap()) as u64;
        let kind: &[u8; 4] = data.get(4..8)?.try_into().unwrap();
        let (header, size) = match size {
            0 => (8, data.len() as u64),
            1 => (16, u64::from_be_bytes(data.get(8..16)?.try_into().unwrap())),
            _ => (8, size),
        };
        let size = usize::try_from(size).ok().filter(|&s| s >= header && s <= data.len())?;
        let body = &data[header..size];
        data = &data[size..];
        Some((kind, body))
    })
}

#[inline]
fn pack([r, g, b, a]: [u8; 4]) -> u32 {
    u32::from_be_bytes([a, r, g, b])
}
