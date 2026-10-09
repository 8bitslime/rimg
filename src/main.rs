mod decode;
mod gpu;
mod render;
mod view;

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};
use std::{env, fs, process, thread};

use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::{Key, NamedKey};
use winit::window::{CursorIcon, Fullscreen, Window, WindowId};

use decode::Image;
use view::{Placement, View};

type Loaded = decode::Result<Image>;

/// Results from the worker threads, delivered through the event loop.
enum Event {
    /// A still image (or the error that stopped decoding).
    Loaded(Loaded),
    /// The next frame of an animation, streamed as it decodes.
    Frame(decode::Frame),
    /// Every animation frame has been sent, so playback can loop.
    AnimationComplete,
    Gpu(Result<gpu::GpuContext, String>),
}

/// How long window creation waits for the header probe before picking a default size.
const PROBE_TIMEOUT: Duration = Duration::from_millis(100);
/// Zoom factor per mouse wheel notch or +/- key press.
const ZOOM_STEP: f64 = 1.25;
/// Zoom exponent per pixel of touchpad scroll.
const ZOOM_PER_PIXEL: f64 = 0.005;
/// Arrow keys pan by this fraction of the window.
const PAN_STEP: f64 = 0.125;
/// Frame delays at or below this are shown for `DEFAULT_FRAME_DELAY` instead, as
/// browsers do: many GIFs declare 0ms and expect that.
const MIN_FRAME_DELAY: Duration = Duration::from_millis(10);
const DEFAULT_FRAME_DELAY: Duration = Duration::from_millis(100);

type Size = (f64, f64);

struct App {
    name: String,
    /// Image dimensions from the header probe, used to size the window.
    dims_rx: Receiver<Option<(u32, u32)>>,
    /// Decoded frames with their delays; a still image is a single frame.
    frames: Vec<decode::Frame>,
    /// Index of the frame on screen.
    frame: usize,
    /// Whether the decoder has delivered every frame, so playback can loop.
    all_frames: bool,
    paused: bool,
    /// When the current animation frame's time is up.
    next_frame_at: Option<Instant>,
    scaled: Option<render::Scaled>,
    view: View,
    cursor: (f64, f64),
    /// Last cursor position while dragging to pan.
    drag: Option<(f64, f64)>,
    window: Option<Arc<Window>>,
    /// CPU renderer: draws the first frames, and everything if the GPU is unavailable.
    cpu: Option<softbuffer::Surface<Arc<Window>, Arc<Window>>>,
    gpu: Option<gpu::Renderer>,
    /// Starts GPU setup after the first CPU frame. Starting it earlier slows that
    /// frame: loading the GPU driver competes with the windowing libraries.
    start_gpu: Option<EventLoopProxy<Event>>,
}

impl App {
    fn image(&self) -> Option<&Image> {
        self.frames.get(self.frame).map(|(img, _)| img)
    }

    /// Window and image sizes, once both exist.
    fn sizes(&self) -> Option<(Size, Size)> {
        let win = self.window.as_ref()?.inner_size();
        let img = self.image()?;
        Some(((win.width as f64, win.height as f64), (img.width as f64, img.height as f64)))
    }

    /// Applies a view change, then refreshes the title and redraws.
    fn update_view(&mut self, change: impl FnOnce(&mut View, Size, Size)) {
        let Some((win, img)) = self.sizes() else { return };
        change(&mut self.view, win, img);
        let zoom = self.view.placement(win, img).zoom;
        if let (Some(window), Some(image)) = (&self.window, self.image()) {
            let percent = (zoom * 100.0).round();
            window.set_title(&format!("{} ({}x{}) {percent}% - rimg", self.name, image.width, image.height));
            window.request_redraw();
        }
    }

    fn redraw(&mut self) {
        let placement = match self.sizes() {
            Some((win, img)) => self.view.placement(win, img),
            None => Placement { zoom: 1.0, x: 0.0, y: 0.0 },
        };
        if let Some(gpu) = &mut self.gpu {
            gpu.render(placement);
            return;
        }
        let (Some(window), Some(surface)) = (&self.window, &mut self.cpu) else { return };
        let size = window.inner_size();
        let (Some(w), Some(h)) = (NonZeroU32::new(size.width), NonZeroU32::new(size.height)) else {
            return;
        };
        surface.resize(w, h).expect("failed to resize surface");
        let mut buf = surface.buffer_mut().expect("failed to get surface buffer");
        let image = self.frames.get(self.frame).map(|(img, _)| img);
        render::draw(&mut buf, w.get(), h.get(), image, placement, &mut self.scaled);
        buf.present().expect("failed to present");
        if let Some(proxy) = self.start_gpu.take() {
            let window = window.clone();
            thread::spawn(move || {
                let _ = proxy.send_event(Event::Gpu(gpu::init(window)));
            });
        }
    }

    /// Switches drawing to the GPU once its context is ready. Stays on the CPU if
    /// the window's surface can't be presented with it.
    fn activate_gpu(&mut self, ctx: gpu::GpuContext) {
        let Some(window) = &self.window else { return };
        match gpu::Renderer::new(ctx, window.clone()) {
            Ok(mut renderer) => {
                if let Some(img) = self.image() {
                    renderer.set_image(img);
                }
                self.gpu = Some(renderer);
                self.cpu = None;
                self.scaled = None;
                window.request_redraw();
            }
            Err(e) => eprintln!("rimg: GPU unavailable, using CPU rendering: {e}"),
        }
    }

    /// Adds a decoded image or animation frame; the first one goes on screen.
    fn add_frame(&mut self, frame: decode::Frame) {
        self.frames.push(frame);
        if self.frames.len() == 1 {
            self.show_frame(0);
            self.update_view(|v, _, _| v.fit());
        }
    }

    fn show_frame(&mut self, index: usize) {
        self.frame = index;
        self.scaled = None;
        if let Some(gpu) = &mut self.gpu {
            gpu.set_image(&self.frames[index].0);
        }
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn is_animating(&self) -> bool {
        !self.paused && self.frames.len() > 1
    }

    /// Advances the animation if the current frame's time is up, and returns
    /// when to check next.
    fn tick_animation(&mut self) -> Option<Instant> {
        if !self.is_animating() {
            return None;
        }
        let now = Instant::now();
        let due = *self.next_frame_at.get_or_insert_with(|| now + frame_delay(self.frames[self.frame].1));
        if now < due {
            return Some(due);
        }
        let next = if self.frame + 1 < self.frames.len() {
            self.frame + 1
        } else if self.all_frames {
            0
        } else {
            // Caught up with the decoder; the next Frame event wakes us.
            return None;
        };
        self.show_frame(next);
        // Schedule from the previous deadline so timing doesn't drift, unless
        // we've fallen more than a frame behind.
        let delay = frame_delay(self.frames[next].1);
        let next_at = if due + delay > now { due + delay } else { now + delay };
        self.next_frame_at = Some(next_at);
        Some(next_at)
    }

    fn toggle_pause(&mut self) {
        self.paused = !self.paused;
        self.next_frame_at = None;
    }

    fn is_fullscreen(&self) -> bool {
        self.window.as_ref().is_some_and(|w| w.fullscreen().is_some())
    }

    /// The resulting resize event re-lays out the view.
    fn set_fullscreen(&self, on: bool) {
        if let Some(window) = &self.window {
            window.set_fullscreen(on.then_some(Fullscreen::Borderless(None)));
        }
    }

    fn window_center(&self) -> (f64, f64) {
        self.sizes().map_or((0.0, 0.0), |(win, _)| (win.0 / 2.0, win.1 / 2.0))
    }

    fn pan_by_window_fraction(&mut self, fx: f64, fy: f64) {
        self.update_view(|v, win, img| v.pan(-fx * win.0, -fy * win.1, win, img));
    }
}

fn frame_delay(delay: Duration) -> Duration {
    if delay <= MIN_FRAME_DELAY { DEFAULT_FRAME_DELAY } else { delay }
}

/// Fits image dimensions inside 90% of the monitor, keeping aspect ratio.
fn initial_size(event_loop: &ActiveEventLoop, (w, h): (u32, u32)) -> PhysicalSize<u32> {
    let monitor = event_loop.primary_monitor().or_else(|| event_loop.available_monitors().next());
    let (max_w, max_h) = monitor.map_or((1600.0, 1000.0), |m| {
        let s = m.size();
        (s.width as f64 * 0.9, s.height as f64 * 0.9)
    });
    let scale = (max_w / w as f64).min(max_h / h as f64).min(1.0);
    PhysicalSize::new(((w as f64 * scale) as u32).max(200), ((h as f64 * scale) as u32).max(150))
}

impl ApplicationHandler<Event> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let mut attrs = Window::default_attributes().with_title(format!("{} - rimg", self.name));
        if let Ok(Some(dims)) = self.dims_rx.recv_timeout(PROBE_TIMEOUT) {
            attrs = attrs.with_inner_size(initial_size(event_loop, dims));
        }
        let window = Arc::new(event_loop.create_window(attrs).expect("failed to create window"));
        let context = softbuffer::Context::new(window.clone()).expect("failed to create graphics context");
        self.cpu = Some(softbuffer::Surface::new(&context, window.clone()).expect("failed to create surface"));
        self.window = Some(window);
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: Event) {
        match event {
            Event::Loaded(Ok(img)) => {
                self.all_frames = true;
                self.add_frame((img, Duration::ZERO));
            }
            Event::Frame(frame) => self.add_frame(frame),
            Event::AnimationComplete if self.frames.is_empty() => {
                eprintln!("rimg: {}: animation has no frames", self.name);
                event_loop.exit();
            }
            Event::AnimationComplete => self.all_frames = true,
            Event::Loaded(Err(e)) => {
                eprintln!("rimg: {}: {e}", self.name);
                event_loop.exit();
            }
            Event::Gpu(Ok(ctx)) => self.activate_gpu(ctx),
            Event::Gpu(Err(e)) => eprintln!("rimg: GPU unavailable, using CPU rendering: {e}"),
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(match self.tick_animation() {
            Some(at) => ControlFlow::WaitUntil(at),
            None => ControlFlow::Wait,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event: KeyEvent { state: ElementState::Pressed, logical_key, .. },
                ..
            } => match logical_key.as_ref() {
                Key::Named(NamedKey::Escape) if self.is_fullscreen() => self.set_fullscreen(false),
                Key::Named(NamedKey::Escape) | Key::Character("q") => event_loop.exit(),
                Key::Named(NamedKey::F11) | Key::Character("f") => self.set_fullscreen(!self.is_fullscreen()),
                Key::Character("+" | "=") => {
                    let c = self.window_center();
                    self.update_view(|v, win, img| v.zoom_at(ZOOM_STEP, c, win, img));
                }
                Key::Character("-") => {
                    let c = self.window_center();
                    self.update_view(|v, win, img| v.zoom_at(1.0 / ZOOM_STEP, c, win, img));
                }
                Key::Character("0") => self.update_view(|v, _, _| v.fit()),
                Key::Character("w") => self.update_view(|v, _, _| v.fill()),
                Key::Character("1") => self.update_view(|v, win, img| v.actual_size(win, img)),
                Key::Named(NamedKey::Space) => self.toggle_pause(),
                Key::Named(NamedKey::ArrowLeft) => self.pan_by_window_fraction(-PAN_STEP, 0.0),
                Key::Named(NamedKey::ArrowRight) => self.pan_by_window_fraction(PAN_STEP, 0.0),
                Key::Named(NamedKey::ArrowUp) => self.pan_by_window_fraction(0.0, -PAN_STEP),
                Key::Named(NamedKey::ArrowDown) => self.pan_by_window_fraction(0.0, PAN_STEP),
                _ => {}
            },
            WindowEvent::MouseWheel { delta, .. } => {
                let factor = match delta {
                    MouseScrollDelta::LineDelta(_, y) => ZOOM_STEP.powf(y as f64),
                    MouseScrollDelta::PixelDelta(d) => (d.y * ZOOM_PER_PIXEL).exp(),
                };
                let c = self.cursor;
                self.update_view(|v, win, img| v.zoom_at(factor, c, win, img));
            }
            WindowEvent::PinchGesture { delta, .. } => {
                let c = self.cursor;
                self.update_view(|v, win, img| v.zoom_at(1.0 + delta, c, win, img));
            }
            WindowEvent::CursorMoved { position, .. } => {
                let pos = (position.x, position.y);
                if let Some(last) = self.drag {
                    self.drag = Some(pos);
                    self.update_view(|v, win, img| v.pan(pos.0 - last.0, pos.1 - last.1, win, img));
                }
                self.cursor = pos;
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                let grabbing = state == ElementState::Pressed;
                self.drag = grabbing.then_some(self.cursor);
                if let Some(window) = &self.window {
                    window.set_cursor(if grabbing { CursorIcon::Grabbing } else { CursorIcon::Default });
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.resize(size.width, size.height);
                }
                self.update_view(|v, win, img| v.resized(win, img));
            }
            WindowEvent::RedrawRequested => self.redraw(),
            _ => {}
        }
    }
}

fn main() {
    let Some(path) = env::args_os().nth(1).map(PathBuf::from) else {
        eprintln!("usage: rimg <image>");
        process::exit(2);
    };
    let use_gpu = env::var("RIMG_RENDERER").map_or(true, |r| r != "cpu");
    if use_gpu {
        gpu::skip_sleeping_gpus();
    }
    let name = path.file_name().unwrap_or(path.as_os_str()).to_string_lossy().into_owned();

    // Read and decode on a worker while the main thread connects to the display server.
    let (dims_tx, dims_rx) = mpsc::channel();
    let (proxy_tx, proxy_rx) = mpsc::channel::<EventLoopProxy<Event>>();
    let worker_name = name.clone();
    thread::spawn(move || {
        // The proxy arrives once the event loop exists; usually before the first frame decodes.
        let mut proxy = None;
        let mut send = move |event| {
            if proxy.is_none() {
                proxy = proxy_rx.recv().ok();
            }
            proxy.as_ref().is_some_and(|p: &EventLoopProxy<Event>| p.send_event(event).is_ok())
        };
        let data = match fs::read(&path) {
            Ok(data) => data,
            Err(e) => {
                send(Event::Loaded(Err(e.to_string())));
                return;
            }
        };
        let _ = dims_tx.send(decode::probe(&data, &path));

        let Some(frames) = decode::animation(&data) else {
            send(Event::Loaded(decode::decode(&data, &path)));
            return;
        };
        for (i, frame) in frames.enumerate() {
            match frame {
                Ok(frame) => {
                    if !send(Event::Frame(frame)) {
                        return;
                    }
                }
                Err(e) if i == 0 => {
                    send(Event::Loaded(Err(e)));
                    return;
                }
                Err(e) => {
                    // Keep playing the frames that did decode.
                    eprintln!("rimg: {worker_name}: frame {}: {e}", i + 1);
                    break;
                }
            }
        }
        send(Event::AnimationComplete);
    });

    let event_loop = EventLoop::<Event>::with_user_event().build().expect("failed to create event loop");
    proxy_tx.send(event_loop.create_proxy()).unwrap();

    let mut app = App {
        name,
        dims_rx,
        frames: Vec::new(),
        frame: 0,
        all_frames: false,
        paused: false,
        next_frame_at: None,
        scaled: None,
        view: View::new(),
        cursor: (0.0, 0.0),
        drag: None,
        window: None,
        cpu: None,
        gpu: None,
        start_gpu: use_gpu.then(|| event_loop.create_proxy()),
    };
    event_loop.run_app(&mut app).expect("event loop failed");
}
