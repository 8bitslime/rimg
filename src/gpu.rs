//! wgpu renderer. Device setup runs on a background thread (`init`) so the
//! first frame doesn't wait on GPU driver loading; `Renderer` takes over once
//! it's ready.

use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use winit::window::Window;

use crate::decode::Image;
use crate::menu::Menu;
use crate::view::Placement;

/// Background gray; matches `render::BACKGROUND`.
const BACKGROUND: f64 = 0x1e as f64 / 255.0;
const CHECKER_SIZE: f32 = 8.0;
/// Texture format of uploaded images. Matches the byte order of `Image` pixels
/// (0xAARRGGBB in little-endian memory), so uploads need no conversion.
const IMAGE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;
/// Surface format the image pipeline is compiled for up front; the common case on all backends.
const LIKELY_SURFACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;
/// Bytes of the `Params` uniform in shader.wgsl.
const PARAMS_SIZE: u64 = 64;

/// Device and pipelines, created off the main thread.
pub struct GpuContext {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    image_layout: wgpu::BindGroupLayout,
    image_pipeline_layout: wgpu::PipelineLayout,
    image_pipeline: wgpu::RenderPipeline,
    image_sampler: wgpu::Sampler,
    shader: wgpu::ShaderModule,
    mip_layout: wgpu::BindGroupLayout,
    mip_pipeline: wgpu::RenderPipeline,
    mip_sampler: wgpu::Sampler,
}

/// wgpu's adapter/device futures complete immediately on native backends, so a
/// busy poll is enough to drive them.
fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(v) = future.as_mut().poll(&mut cx) {
            return v;
        }
        std::thread::yield_now();
    }
}

fn request_adapter(backends: wgpu::Backends, window: &Arc<Window>) -> Option<(wgpu::Instance, wgpu::Adapter)> {
    // OpenGL on Wayland/X11 (EGL) needs the display connection up front.
    let mut desc = wgpu::InstanceDescriptor::new_with_display_handle(Box::new(window.clone()));
    desc.backends = backends;
    let instance = wgpu::Instance::new(desc.with_env());
    // An image viewer doesn't need a discrete GPU; integrated ones use less power.
    let options = wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::LowPower, ..Default::default() };
    let adapter = block_on(instance.request_adapter(&options)).ok()?;
    Some((instance, adapter))
}

/// Creates the device and pipelines. Prefers Vulkan/Metal/DX12, falling back to OpenGL.
pub fn init(window: Arc<Window>) -> Result<GpuContext, String> {
    let (instance, adapter) = request_adapter(wgpu::Backends::PRIMARY, &window)
        .or_else(|| request_adapter(wgpu::Backends::GL, &window))
        .ok_or("no GPU adapter found")?;

    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("rimg"),
        // Ask for the adapter's real max texture size so fewer images need tiling.
        required_limits: wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits()),
        ..Default::default()
    }))
    .map_err(|e| e.to_string())?;

    let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
    let texture_entry = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let sampler_entry = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    };

    let image_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("image"),
        entries: &[
            texture_entry(0),
            sampler_entry(1),
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let image_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("image"),
        bind_group_layouts: &[Some(&image_layout)],
        immediate_size: 0,
    });
    let image_pipeline = create_image_pipeline(&device, &image_pipeline_layout, &shader, LIKELY_SURFACE_FORMAT);
    // Nearest when magnifying keeps pixels crisp; trilinear when minifying.
    let image_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("image"),
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    });

    let mip_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mip"),
        entries: &[texture_entry(3), sampler_entry(4)],
    });
    let mip_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("mip"),
        bind_group_layouts: &[Some(&mip_layout)],
        immediate_size: 0,
    });
    let mip_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("mip"),
        layout: Some(&mip_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_fullscreen"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_downsample"),
            compilation_options: Default::default(),
            targets: &[Some(IMAGE_FORMAT.into())],
        }),
        multiview_mask: None,
        cache: None,
    });
    let mip_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("mip"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    Ok(GpuContext {
        instance,
        adapter,
        device,
        queue,
        image_layout,
        image_pipeline_layout,
        image_pipeline,
        image_sampler,
        shader,
        mip_layout,
        mip_pipeline,
        mip_sampler,
    })
}

/// On hybrid-GPU Linux laptops, keeps the Vulkan loader from loading drivers for
/// GPUs that are runtime-suspended while another vendor's GPU is awake. Merely
/// enumerating a sleeping discrete GPU wakes it (~2.5s on NVIDIA) and keeps it
/// powered for as long as the instance lives.
///
/// Must run before any other threads start, since it sets an environment variable.
#[cfg(target_os = "linux")]
pub fn skip_sleeping_gpus() {
    const VK_DISABLE: &str = "VK_LOADER_DRIVERS_DISABLE";
    if std::env::var_os(VK_DISABLE).is_some() {
        return;
    }
    let Ok(cards) = std::fs::read_dir("/sys/class/drm") else { return };
    // (PCI vendor id, runtime-suspended) for each GPU.
    let gpus: Vec<(String, bool)> = cards
        .flatten()
        .filter(|e| e.file_name().to_str().is_some_and(|n| n.starts_with("card") && !n.contains('-')))
        .filter_map(|e| {
            let device = e.path().join("device");
            let vendor = std::fs::read_to_string(device.join("vendor")).ok()?.trim().to_string();
            let status = std::fs::read_to_string(device.join("power/runtime_status")).unwrap_or_default();
            Some((vendor, status.trim() == "suspended"))
        })
        .collect();

    let awake: Vec<&str> = gpus.iter().filter(|(_, asleep)| !asleep).map(|(v, _)| v.as_str()).collect();
    if awake.is_empty() {
        return;
    }
    let globs: Vec<&str> = gpus
        .iter()
        .filter(|(vendor, asleep)| *asleep && !awake.contains(&vendor.as_str()))
        .filter_map(|(vendor, _)| match vendor.as_str() {
            "0x10de" => Some("*nvidia*"),
            "0x1002" => Some("*radeon*"),
            "0x8086" => Some("*intel*"),
            _ => None,
        })
        .collect();
    if !globs.is_empty() {
        // SAFETY: called at the top of main, before any other thread exists.
        unsafe { std::env::set_var(VK_DISABLE, globs.join(",")) };
    }
}

#[cfg(not(target_os = "linux"))]
pub fn skip_sleeping_gpus() {}

fn create_image_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("image"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_image"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, ..Default::default() },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_image"),
            compilation_options: Default::default(),
            targets: &[Some(format.into())],
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// One texture covering part of the image, for images larger than the max texture size.
struct Tile {
    /// Position and size in image pixels.
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    texture: wgpu::Texture,
    params: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

struct Uploaded {
    width: u32,
    height: u32,
    tiles: Vec<Tile>,
    has_alpha: bool,
}

/// The menu's texture, drawn over the image.
struct Overlay {
    tile: Tile,
    version: u64,
}

pub struct Renderer {
    ctx: GpuContext,
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    image: Option<Uploaded>,
    overlay: Option<Overlay>,
}

impl Renderer {
    pub fn new(ctx: GpuContext, window: Arc<Window>) -> Result<Renderer, String> {
        let surface = ctx.instance.create_surface(window.clone()).map_err(|e| e.to_string())?;
        let caps = surface.get_capabilities(&ctx.adapter);
        // Prefer a non-sRGB format: blending and filtering then happen on the
        // encoded values, matching the CPU renderer.
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .or(caps.formats.first().copied())
            .ok_or("surface not supported by GPU adapter")?;
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes.first().copied().unwrap_or(wgpu::CompositeAlphaMode::Auto),
            view_formats: vec![],
        };
        surface.configure(&ctx.device, &config);

        let mut ctx = ctx;
        if format != LIKELY_SURFACE_FORMAT {
            ctx.image_pipeline = create_image_pipeline(&ctx.device, &ctx.image_pipeline_layout, &ctx.shader, format);
        }
        Ok(Renderer { ctx, window, surface, config, image: None, overlay: None })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.ctx.device, &self.config);
    }

    /// Uploads the image as textures with full mip chains, split into tiles if it
    /// exceeds the max texture size. Textures are reused when the size matches,
    /// as it does for every frame of an animation.
    pub fn set_image(&mut self, img: &Image) {
        let mut encoder = self.ctx.device.create_command_encoder(&Default::default());
        if let Some(up) = self.image.as_mut().filter(|up| (up.width, up.height) == (img.width, img.height)) {
            up.has_alpha = img.has_alpha;
            let up = self.image.as_ref().unwrap();
            for tile in &up.tiles {
                self.write_tile(&mut encoder, tile, img.width, &img.pixels);
            }
            self.ctx.queue.submit([encoder.finish()]);
            return;
        }

        let max = self.ctx.device.limits().max_texture_dimension_2d;
        let mut tiles = Vec::new();

        for y in (0..img.height).step_by(max as usize) {
            for x in (0..img.width).step_by(max as usize) {
                let (width, height) = ((img.width - x).min(max), (img.height - y).min(max));
                let mip_level_count = 32 - width.max(height).leading_zeros();
                let tile = self.create_tile(x, y, width, height, mip_level_count);
                self.write_tile(&mut encoder, &tile, img.width, &img.pixels);
                tiles.push(tile);
            }
        }
        self.ctx.queue.submit([encoder.finish()]);
        self.image = Some(Uploaded { width: img.width, height: img.height, tiles, has_alpha: img.has_alpha });
    }

    fn create_tile(&self, x: u32, y: u32, width: u32, height: u32, mip_level_count: u32) -> Tile {
        let texture = self.ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("image"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: IMAGE_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let params = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: PARAMS_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let view = texture.create_view(&Default::default());
        let bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("image"),
            layout: &self.ctx.image_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.ctx.image_sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
            ],
        });
        Tile { x, y, width, height, texture, params, bind_group }
    }

    /// Copies the tile's region of a `src_width`-wide pixel buffer into its texture and rebuilds the mips.
    fn write_tile(&self, encoder: &mut wgpu::CommandEncoder, tile: &Tile, src_width: u32, pixels: &[u32]) {
        self.ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tile.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            pixel_bytes(pixels),
            wgpu::TexelCopyBufferLayout {
                offset: (tile.y as u64 * src_width as u64 + tile.x as u64) * 4,
                bytes_per_row: Some(src_width * 4),
                rows_per_image: Some(tile.height),
            },
            wgpu::Extent3d { width: tile.width, height: tile.height, depth_or_array_layers: 1 },
        );
        self.generate_mips(encoder, &tile.texture, tile.texture.mip_level_count());
    }

    fn generate_mips(&self, encoder: &mut wgpu::CommandEncoder, texture: &wgpu::Texture, levels: u32) {
        let level_view = |level| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: level,
                mip_level_count: Some(1),
                ..Default::default()
            })
        };
        for level in 1..levels {
            let src = level_view(level - 1);
            let dst = level_view(level);
            let bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mip"),
                layout: &self.ctx.mip_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&src) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.ctx.mip_sampler) },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mip"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dst,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.ctx.mip_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    /// Uploads the menu if it changed since the last call; `None` hides it.
    pub fn set_overlay(&mut self, menu: Option<&Menu>) {
        let Some(menu) = menu else {
            self.overlay = None;
            return;
        };
        if self.overlay.as_ref().is_some_and(|o| o.version == menu.version) {
            return;
        }
        let reuse = self.overlay.take().filter(|o| (o.tile.width, o.tile.height) == (menu.width, menu.height));
        let tile = reuse.map_or_else(|| self.create_tile(0, 0, menu.width, menu.height, 1), |o| o.tile);
        let mut encoder = self.ctx.device.create_command_encoder(&Default::default());
        self.write_tile(&mut encoder, &tile, menu.width, &menu.pixels);
        self.ctx.queue.submit([encoder.finish()]);
        self.overlay = Some(Overlay { tile, version: menu.version });
    }

    pub fn render(&mut self, placement: Placement, menu: Option<&Menu>) {
        self.set_overlay(menu);
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                // Present this one; the next frame uses a fresh configuration.
                self.surface.configure(&self.ctx.device, &self.config);
                frame
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.ctx.device, &self.config);
                self.window.request_redraw();
                return;
            }
            _ => return,
        };
        let view = frame.texture.create_view(&Default::default());
        let srgb = self.config.format.is_srgb();
        let (ww, wh) = (self.config.width as f32, self.config.height as f32);

        if let Some(img) = &self.image {
            // Whole pixels at the image origin keep 1:1 zoom from sampling between texels.
            let (ox, oy, zoom) = (placement.x.round(), placement.y.round(), placement.zoom);
            let o = placement.orientation;
            let src = (img.width as f64, img.height as f64);
            for tile in &img.tiles {
                // The tile's rectangle on the displayed image.
                let (tx0, ty0) = (tile.x as f64, tile.y as f64);
                let (tx1, ty1) = (tx0 + tile.width as f64, ty0 + tile.height as f64);
                let (a, b) = (o.to_display((tx0, ty0), src), o.to_display((tx1, ty1), src));
                let (u0, v0, u1, v1) = (a.0.min(b.0), a.1.min(b.1), a.0.max(b.0), a.1.max(b.1));
                // Texture coordinates at three of its corners.
                let uv = |u: f64, v: f64| {
                    let (x, y) = o.to_source((u, v), src);
                    ((x - tx0) / tile.width as f64, (y - ty0) / tile.height as f64)
                };
                let (top_left, top_right, bottom_left) = (uv(u0, v0), uv(u1, v0), uv(u0, v1));
                let (x0, y0) = (ox + u0 * zoom, oy + v0 * zoom);
                let (x1, y1) = (ox + u1 * zoom, oy + v1 * zoom);
                let params = [
                    x0 as f32 / ww * 2.0 - 1.0,
                    1.0 - y0 as f32 / wh * 2.0,
                    x1 as f32 / ww * 2.0 - 1.0,
                    1.0 - y1 as f32 / wh * 2.0,
                    top_left.0 as f32,
                    top_left.1 as f32,
                    (top_right.0 - top_left.0) as f32,
                    (top_right.1 - top_left.1) as f32,
                    (bottom_left.0 - top_left.0) as f32,
                    (bottom_left.1 - top_left.1) as f32,
                    0.0,
                    0.0,
                    if img.has_alpha { 1.0 } else { 0.0 },
                    if srgb { 1.0 } else { 0.0 },
                    CHECKER_SIZE,
                    0.0,
                ];
                let bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
                self.ctx.queue.write_buffer(&tile.params, 0, &bytes);
            }
        }

        if let (Some(overlay), Some(menu)) = (&self.overlay, menu) {
            // Drawn opaque at whole pixels, so it samples texels 1:1.
            let (x0, y0) = (menu.x as f32, menu.y as f32);
            let (x1, y1) = (x0 + menu.width as f32, y0 + menu.height as f32);
            let params = [
                x0 / ww * 2.0 - 1.0,
                1.0 - y0 / wh * 2.0,
                x1 / ww * 2.0 - 1.0,
                1.0 - y1 / wh * 2.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                if srgb { 1.0 } else { 0.0 },
                CHECKER_SIZE,
                0.0,
            ];
            let bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
            self.ctx.queue.write_buffer(&overlay.tile.params, 0, &bytes);
        }

        let mut encoder = self.ctx.device.create_command_encoder(&Default::default());
        {
            let bg = if srgb { srgb_to_linear(BACKGROUND) } else { BACKGROUND };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("frame"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: bg, g: bg, b: bg, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.ctx.image_pipeline);
            if let Some(img) = &self.image {
                for tile in &img.tiles {
                    pass.set_bind_group(0, &tile.bind_group, &[]);
                    pass.draw(0..4, 0..1);
                }
            }
            if let (Some(overlay), Some(_)) = (&self.overlay, menu) {
                pass.set_bind_group(0, &overlay.tile.bind_group, &[]);
                pass.draw(0..4, 0..1);
            }
        }
        self.ctx.queue.submit([encoder.finish()]);
        self.window.pre_present_notify();
        self.ctx.queue.present(frame);
    }
}

fn srgb_to_linear(c: f64) -> f64 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

fn pixel_bytes(pixels: &[u32]) -> &[u8] {
    // SAFETY: u32 has no padding and any byte pattern is a valid u8; the length
    // covers exactly the same memory.
    unsafe { std::slice::from_raw_parts(pixels.as_ptr().cast(), pixels.len() * 4) }
}
