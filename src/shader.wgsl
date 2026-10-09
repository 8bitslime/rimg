// Image drawing: one quad per texture tile.

struct Params {
    // Tile corners in clip space: x0, y0, x1, y1.
    rect: vec4<f32>,
    // Texture coordinates at the quad's top-left corner (xy), and their change
    // across its width (zw); rotation and mirroring live here.
    uv: vec4<f32>,
    // Change in texture coordinates down the quad's height (xy).
    uv_down: vec4<f32>,
    // x: 1 if the image has alpha (draw a checkerboard behind it),
    // y: 1 if the surface is sRGB (shader output must be linear),
    // z: checkerboard cell size in pixels.
    flags: vec4<f32>,
}

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> params: Params;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Triangle strip over vertex indices 0..4.
@vertex
fn vs_image(@builtin(vertex_index) i: u32) -> VsOut {
    let corner = vec2<f32>(f32(i & 1u), f32(i >> 1u));
    var out: VsOut;
    out.pos = vec4<f32>(mix(params.rect.xy, params.rect.zw, corner), 0.0, 1.0);
    out.uv = params.uv.xy + corner.x * params.uv.zw + corner.y * params.uv_down.xy;
    return out;
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045));
}

@fragment
fn fs_image(in: VsOut) -> @location(0) vec4<f32> {
    // Texels are premultiplied, so compositing is `src + bg * (1 - a)`.
    let c = textureSample(tex, samp, in.uv);
    var rgb = c.rgb;
    if params.flags.x > 0.5 {
        let cell = vec2<u32>(in.pos.xy / params.flags.z);
        let bg = select(0.4, 0.6, ((cell.x + cell.y) & 1u) == 0u);
        rgb += vec3<f32>(bg) * (1.0 - c.a);
    }
    if params.flags.y > 0.5 {
        rgb = srgb_to_linear(rgb);
    }
    return vec4<f32>(rgb, 1.0);
}

// Mipmap generation: draws the previous level into the next with linear filtering.

@group(0) @binding(3) var mip_src: texture_2d<f32>;
@group(0) @binding(4) var mip_samp: sampler;

// One triangle covering the whole target.
@vertex
fn vs_fullscreen(@builtin(vertex_index) i: u32) -> VsOut {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var out: VsOut;
    out.pos = vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
    out.uv = uv;
    return out;
}

@fragment
fn fs_downsample(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(mip_src, mip_samp, in.uv);
}
