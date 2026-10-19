//! The conversions the host runs on frames on the device, through one entry
//! point: [`convert`] makes a frame's planes, held in one [`Layout`], into
//! new planes in another layout at a [`Target`] size, with the matrix and
//! range a [`Yuv`] names. A frame keeps what it was converted to
//! ([`super::frames::GpuFrame::converted`]), so every reader asking for the
//! same target shares one conversion.
//!
//! What it makes:
//!
//! - The same layout at another size: each plane resized on its own with a
//!   bilinear filter, a triangle stretched over every texel a pixel covers
//!   where the plane shrinks and the edges repeated, which is what ffmpeg's
//!   scale filter does with `flags=bilinear`.
//! - YUV (NV12, yuv420p, yuv422p, yuv444p) to rgba, at any size: luma and
//!   chroma resampled at each pixel, the chroma sited at the centre of the
//!   luma samples it covers, then the matrix and range. Alpha is one, so
//!   the result is premultiplied as a module renders it.
//! - rgba to NV12 or yuv420p: luma at full size, and chroma at half size
//!   the way ffmpeg's scale filter makes it by default, each pair of pixels
//!   averaged across and a bicubic filter of eight rows down the column.
//!   At another size each plane is the rgba resampled to it, as above,
//!   then the matrix. The rgba is premultiplied and is read as it is, so a
//!   transparent pixel is black.
//! - One YUV layout to another: NV12's chroma split into two planes or two
//!   planes interleaved into one, each plane copied texel for texel where
//!   the sizes agree and resampled where they do not.
//!
//! Each conversion is a few render passes on the queue every module
//! submits to, so it runs after whatever wrote the source and before
//! whatever reads the result. Each distinct one says so on stderr the
//! first time it runs: `[gpu] convert nv12 1920x1080 to rgba 1920x1080`.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use anyhow::{anyhow, bail, Result};
use wasi_webgpu_wasmtime::reexports::{wgpu_core, wgpu_types};
use wasi_webgpu_wasmtime::SharedDevice;
use wgpu_core::global::Global;
use wgpu_core::id::{BindGroupId, BindGroupLayoutId, BufferId, TextureViewId};
use wgpu_core::id::{PipelineLayoutId, RenderPipelineId, ShaderModuleId};
use wgpu_types::TextureFormat;

use super::frames::{create_plane, release_plane, Plane};
use crate::profile::PROFILE;

const SHADER: &str = r#"
struct Params {
    // The size of the plane the pass writes.
    out: vec4<f32>,
    // The size of the picture in the first source, then in the others (a
    // decoder's texture may be larger than its picture).
    sizes: vec4<f32>,
    // The matrix's red and blue weights, then 1 for full range and 0 for
    // limited (luma 16 to 235, chroma 16 to 240).
    colour: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var first: texture_2d<f32>;
@group(0) @binding(2) var second: texture_2d<f32>;
@group(0) @binding(3) var third: texture_2d<f32>;

@vertex
fn whole(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(i & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

struct Two {
    @location(0) a: vec4<f32>,
    @location(1) b: vec4<f32>,
}

// `t`, whose picture is `size`, at pixel `at` of the plane being written:
// the texel itself where the two are one size, else a triangle filter over
// the texels the pixel covers, stretched by the scale where the plane
// shrinks, with the edges repeated.
fn resample(t: texture_2d<f32>, size: vec2<f32>, at: vec2<f32>) -> vec4<f32> {
    let out = params.out.xy;
    if all(size == out) {
        return textureLoad(t, vec2<i32>(at), 0);
    }
    let last = vec2<i32>(size) - vec2<i32>(1, 1);
    let scale = size / out;
    let centre = at * scale - vec2<f32>(0.5);
    let reach = max(scale, vec2<f32>(1.0));
    let lo = vec2<i32>(floor(centre - reach)) + vec2<i32>(1, 1);
    let hi = vec2<i32>(ceil(centre + reach)) - vec2<i32>(1, 1);
    var sum = vec4<f32>(0.0);
    var total = 0.0;
    for (var y = lo.y; y <= hi.y; y++) {
        let wy = max(0.0, 1.0 - abs(f32(y) - centre.y) / reach.y);
        for (var x = lo.x; x <= hi.x; x++) {
            let w = wy * max(0.0, 1.0 - abs(f32(x) - centre.x) / reach.x);
            let texel = clamp(vec2<i32>(x, y), vec2<i32>(0, 0), last);
            sum += w * textureLoad(t, texel, 0);
            total += w;
        }
    }
    return sum / total;
}

// One plane resampled to the size of the plane written: r8, rg8 or rgba8.
@fragment
fn plane(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
    return resample(first, params.sizes.xy, at.xy);
}

// NV12's chroma as two planes.
@fragment
fn split(@builtin(position) at: vec4<f32>) -> Two {
    let uv = resample(first, params.sizes.xy, at.xy);
    return Two(vec4<f32>(uv.r, 0.0, 0.0, 1.0), vec4<f32>(uv.g, 0.0, 0.0, 1.0));
}

// Two chroma planes as NV12's one.
@fragment
fn interleave(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
    let u = resample(first, params.sizes.xy, at.xy).r;
    let v = resample(second, params.sizes.zw, at.xy).r;
    return vec4<f32>(u, v, 0.0, 1.0);
}

fn rgb(y: f32, u: f32, v: f32) -> vec4<f32> {
    let kr = params.colour.x;
    let kb = params.colour.y;
    var luma = (y * 255.0 - 16.0) / 219.0;
    var cb = (u * 255.0 - 128.0) / 224.0;
    var cr = (v * 255.0 - 128.0) / 224.0;
    if params.colour.z > 0.5 {
        luma = y;
        cb = (u * 255.0 - 128.0) / 255.0;
        cr = (v * 255.0 - 128.0) / 255.0;
    }
    let r = luma + 2.0 * (1.0 - kr) * cr;
    let b = luma + 2.0 * (1.0 - kb) * cb;
    let g = (luma - kr * r - kb * b) / (1.0 - kr - kb);
    return vec4<f32>(clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}

@fragment
fn nv12_rgba(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
    let y = resample(first, params.sizes.xy, at.xy).r;
    let uv = resample(second, params.sizes.zw, at.xy).rg;
    return rgb(y, uv.x, uv.y);
}

@fragment
fn planar_rgba(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
    let y = resample(first, params.sizes.xy, at.xy).r;
    let u = resample(second, params.sizes.zw, at.xy).r;
    let v = resample(third, params.sizes.zw, at.xy).r;
    return rgb(y, u, v);
}

fn luma_of(c: vec3<f32>) -> f32 {
    let kr = params.colour.x;
    let kb = params.colour.y;
    return kr * c.r + (1.0 - kr - kb) * c.g + kb * c.b;
}

fn coded_luma(y: f32) -> f32 {
    return select((16.0 + 219.0 * y) / 255.0, y, params.colour.z > 0.5);
}

fn coded_chroma(c: f32) -> f32 {
    return select((128.0 + 224.0 * c) / 255.0, (128.0 + 255.0 * c) / 255.0, params.colour.z > 0.5);
}

@fragment
fn rgba_luma(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
    let c = textureLoad(first, vec2<i32>(at.xy), 0).rgb;
    return vec4<f32>(coded_luma(luma_of(c)), 0.0, 0.0, 1.0);
}

// The pair of pixels at `x` and `x + 1` on row `y`, averaged, the edges
// repeated past the picture's.
fn pair(x: i32, y: i32, last: vec2<i32>) -> vec3<f32> {
    let row = clamp(y, 0, last.y);
    let a = textureLoad(first, vec2<i32>(min(x, last.x), row), 0).rgb;
    let b = textureLoad(first, vec2<i32>(min(x + 1, last.x), row), 0).rgb;
    return (a + b) * 0.5;
}

// A bicubic filter (a = -0.6) halving a column, from the centre between
// two rows outwards: what ffmpeg's scale filter applies by default.
const TAPS = array<f32, 4>(0.4359375, 0.1203125, -0.0421875, -0.0140625);

// The coded chroma of a colour.
fn chroma_of(c: vec3<f32>) -> vec2<f32> {
    let kr = params.colour.x;
    let kb = params.colour.y;
    let y = luma_of(c);
    let u = (c.b - y) / (2.0 * (1.0 - kb));
    let v = (c.r - y) / (2.0 * (1.0 - kr));
    return vec2<f32>(coded_chroma(u), coded_chroma(v));
}

// The coded chroma of the rgba around chroma sample `at`, at half the
// rgba's size.
fn chroma_at(at: vec2<f32>) -> vec2<f32> {
    let last = vec2<i32>(params.sizes.xy) - vec2<i32>(1, 1);
    let p = vec2<i32>(at) * 2;
    var c = vec3<f32>(0.0);
    for (var i = 0; i < 4; i++) {
        c += TAPS[i] * (pair(p.x, p.y - i, last) + pair(p.x, p.y + 1 + i, last));
    }
    return chroma_of(c);
}

@fragment
fn rgba_chroma(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = chroma_at(at.xy);
    return vec4<f32>(uv.x, uv.y, 0.0, 1.0);
}

@fragment
fn rgba_chroma_planar(@builtin(position) at: vec4<f32>) -> Two {
    let uv = chroma_at(at.xy);
    return Two(vec4<f32>(uv.x, 0.0, 0.0, 1.0), vec4<f32>(uv.y, 0.0, 0.0, 1.0));
}

// The same at another size: the rgba resampled to the plane written, then
// the matrix.
@fragment
fn rgba_luma_resized(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
    let c = resample(first, params.sizes.xy, at.xy).rgb;
    return vec4<f32>(coded_luma(luma_of(c)), 0.0, 0.0, 1.0);
}

@fragment
fn rgba_chroma_resized(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = chroma_of(resample(first, params.sizes.xy, at.xy).rgb);
    return vec4<f32>(uv.x, uv.y, 0.0, 1.0);
}

@fragment
fn rgba_chroma_planar_resized(@builtin(position) at: vec4<f32>) -> Two {
    let uv = chroma_of(resample(first, params.sizes.xy, at.xy).rgb);
    return Two(vec4<f32>(uv.x, 0.0, 0.0, 1.0), vec4<f32>(uv.y, 0.0, 0.0, 1.0));
}
"#;

/// How a frame's planes are laid out on the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Layout {
    /// One `rgba8unorm` texture, premultiplied.
    Rgba,
    /// Three `r8unorm` planes, chroma at half size both ways.
    Yuv420p,
    /// Three `r8unorm` planes, chroma at half width.
    Yuv422p,
    /// Three `r8unorm` planes at full size.
    Yuv444p,
    /// An `r8unorm` luma plane and an `rg8unorm` chroma plane at half size
    /// both ways: what a decoder writes.
    Nv12,
}

impl Layout {
    /// The layout a pixel format names, where frames are held in it.
    pub fn of(pix_fmt: &str) -> Option<Layout> {
        Some(match pix_fmt {
            "rgba" => Layout::Rgba,
            "yuv420p" => Layout::Yuv420p,
            "yuv422p" => Layout::Yuv422p,
            "yuv444p" => Layout::Yuv444p,
            "nv12" => Layout::Nv12,
            _ => return None,
        })
    }

    /// The pixel format's name, as ffmpeg spells it.
    pub fn name(self) -> &'static str {
        match self {
            Layout::Rgba => "rgba",
            Layout::Yuv420p => "yuv420p",
            Layout::Yuv422p => "yuv422p",
            Layout::Yuv444p => "yuv444p",
            Layout::Nv12 => "nv12",
        }
    }

    pub fn is_rgb(self) -> bool {
        self == Layout::Rgba
    }

    /// The wire format a frame in this layout is read as by default:
    /// itself, and yuv420p for NV12, which the wire does not carry.
    pub fn wire(self) -> Layout {
        match self {
            Layout::Nv12 => Layout::Yuv420p,
            other => other,
        }
    }

    /// The size of the chroma planes of a `width` x `height` picture.
    fn chroma(self, width: u32, height: u32) -> (u32, u32) {
        match self {
            Layout::Yuv420p | Layout::Nv12 => (width.div_ceil(2), height.div_ceil(2)),
            Layout::Yuv422p => (width.div_ceil(2), height),
            Layout::Yuv444p | Layout::Rgba => (width, height),
        }
    }

    /// The planes of a `width` x `height` picture, each as width, height,
    /// texture format and bytes per pixel, in the order the format lays
    /// them out.
    pub fn planes(self, width: u32, height: u32) -> Vec<(u32, u32, TextureFormat, u32)> {
        let r8 = TextureFormat::R8Unorm;
        let (cw, ch) = self.chroma(width, height);
        match self {
            Layout::Rgba => vec![(width, height, TextureFormat::Rgba8Unorm, 4)],
            Layout::Nv12 => vec![(width, height, r8, 1), (cw, ch, TextureFormat::Rg8Unorm, 2)],
            _ => vec![(width, height, r8, 1), (cw, ch, r8, 1), (cw, ch, r8, 1)],
        }
    }

    /// How many bytes a `width` x `height` picture is, packed.
    pub fn len(self, width: u32, height: u32) -> usize {
        self.planes(width, height)
            .iter()
            .map(|(w, h, _, bpp)| *w as usize * *h as usize * *bpp as usize)
            .sum()
    }
}

/// The matrix YUV is written or read with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Matrix {
    /// BT.709: what HD and every newer stream uses.
    Bt709,
    /// BT.601 (`bt470bg`, `smpte170m`): standard definition.
    Bt601,
}

/// The YUV of a conversion between rgba and YUV: a matrix and a range.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Yuv {
    pub matrix: Matrix,
    /// Full range (`pc`) rather than limited (`tv`).
    pub full_range: bool,
}

impl Yuv {
    /// BT.709, limited range: what a stream that says nothing is written in.
    pub const DEFAULT: Yuv = Yuv {
        matrix: Matrix::Bt709,
        full_range: false,
    };

    /// The YUV a stream's colour names, in ffmpeg's names, read as libswscale
    /// reads it: an unknown matrix is BT.601 and an unknown range is limited.
    /// None for a matrix this does not convert (`bt2020nc`, `fcc` and the
    /// like).
    pub fn read_as(space: &str, range: &str) -> Option<Yuv> {
        let matrix = match space {
            "bt709" => Matrix::Bt709,
            "bt470bg" | "smpte170m" | "bt601" | "" | "unknown" | "unspecified" | "reserved" => {
                Matrix::Bt601
            }
            _ => return None,
        };
        Some(Yuv {
            matrix,
            full_range: matches!(range, "pc" | "jpeg" | "full"),
        })
    }

    /// The matrix's red and blue weights.
    fn weights(self) -> (f32, f32) {
        match self.matrix {
            Matrix::Bt709 => (0.2126, 0.0722),
            Matrix::Bt601 => (0.299, 0.114),
        }
    }
}

/// What a frame is converted to: a layout at a size, and the YUV a
/// conversion between rgba and YUV uses on its YUV side.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Target {
    pub layout: Layout,
    pub width: u32,
    pub height: u32,
    pub yuv: Yuv,
}

impl Target {
    pub fn new(layout: Layout, width: u32, height: u32, yuv: Yuv) -> Target {
        Target {
            layout,
            width,
            height,
            yuv,
        }
    }

    /// The target as a frame in `from` keeps it: the YUV counts only where
    /// one side is rgba and the other YUV.
    pub(super) fn key(self, from: Layout) -> Target {
        if from.is_rgb() == self.layout.is_rgb() {
            Target {
                yuv: Yuv::DEFAULT,
                ..self
            }
        } else {
            self
        }
    }
}

/// Whether a frame in `from` can be made `to` on the device: any layout at
/// another size, any YUV into rgba or another YUV, and rgba into NV12 or
/// yuv420p.
pub fn converts(from: Layout, to: Layout) -> bool {
    match (from, to) {
        (a, b) if a == b => true,
        (Layout::Rgba, Layout::Nv12 | Layout::Yuv420p) => true,
        (Layout::Rgba, _) => false,
        _ => true,
    }
}

/// What a pass reads besides its textures: the size of the plane it
/// writes, the sizes of its sources' pictures, and the YUV.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Params {
    out: (u32, u32),
    first: (u32, u32),
    rest: (u32, u32),
    yuv: Yuv,
}

impl Params {
    fn bytes(self) -> Vec<u8> {
        let (kr, kb) = self.yuv.weights();
        let full = if self.yuv.full_range { 1.0 } else { 0.0 };
        let values: [f32; 12] = [
            self.out.0 as f32,
            self.out.1 as f32,
            0.0,
            0.0,
            self.first.0 as f32,
            self.first.1 as f32,
            self.rest.0 as f32,
            self.rest.1 as f32,
            kr,
            kb,
            full,
            0.0,
        ];
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
}

/// The shader, its one bind group layout (the parameters and three
/// textures), and the pipelines and parameter buffers made from them as
/// they are first asked for.
struct Pipelines {
    module: ShaderModuleId,
    group: BindGroupLayoutId,
    layout: PipelineLayoutId,
    made: Mutex<HashMap<(&'static str, Vec<TextureFormat>), RenderPipelineId>>,
    params: Mutex<HashMap<Params, BufferId>>,
}

fn pipelines(global: &Global, device: &SharedDevice) -> Result<&'static Pipelines> {
    static MADE: OnceLock<std::result::Result<Pipelines, String>> = OnceLock::new();
    MADE.get_or_init(|| make_pipelines(global, device).map_err(|e| format!("{e:#}")))
        .as_ref()
        .map_err(|e| anyhow!("the frame conversion pipelines: {e}"))
}

fn make_pipelines(global: &Global, device: &SharedDevice) -> Result<Pipelines> {
    let (module, err) = global.device_create_shader_module(
        device.device,
        &wgpu_core::pipeline::ShaderModuleDescriptor {
            label: Some(Cow::Borrowed("ffrwd frame conversion")),
            runtime_checks: wgpu_types::ShaderRuntimeChecks::checked(),
        },
        wgpu_core::pipeline::ShaderModuleSource::Wgsl(Cow::Borrowed(SHADER)),
        None,
    );
    if let Some(e) = err {
        bail!("compiling the conversion shader: {e}");
    }
    let texture = |binding| wgpu_types::BindGroupLayoutEntry {
        binding,
        visibility: wgpu_types::ShaderStages::FRAGMENT,
        ty: wgpu_types::BindingType::Texture {
            sample_type: wgpu_types::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu_types::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let entries = vec![
        wgpu_types::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu_types::ShaderStages::FRAGMENT,
            ty: wgpu_types::BindingType::Buffer {
                ty: wgpu_types::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        texture(1),
        texture(2),
        texture(3),
    ];
    let (group, err) = global.device_create_bind_group_layout(
        device.device,
        &wgpu_core::binding_model::BindGroupLayoutDescriptor {
            label: None,
            entries: Cow::Owned(entries),
        },
        None,
    );
    if let Some(e) = err {
        bail!("the bind group layout: {e}");
    }
    let (layout, err) = global.device_create_pipeline_layout(
        device.device,
        &wgpu_core::binding_model::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: Cow::Owned(vec![Some(group)]),
            immediate_size: 0,
        },
        None,
    );
    if let Some(e) = err {
        bail!("the pipeline layout: {e}");
    }
    Ok(Pipelines {
        module,
        group,
        layout,
        made: Mutex::new(HashMap::new()),
        params: Mutex::new(HashMap::new()),
    })
}

impl Pipelines {
    /// The pipeline drawing one triangle over `targets` with `entry`, made
    /// the first time it is asked for.
    fn pipeline(
        &self,
        global: &Global,
        device: &SharedDevice,
        entry: &'static str,
        targets: &[TextureFormat],
    ) -> Result<RenderPipelineId> {
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        let key = (entry, targets.to_vec());
        if let Some(id) = made.get(&key) {
            return Ok(*id);
        }
        let stage = |name: &'static str| wgpu_core::pipeline::ProgrammableStageDescriptor {
            module: self.module,
            entry_point: Some(Cow::Borrowed(name)),
            constants: Default::default(),
            zero_initialize_workgroup_memory: true,
        };
        let colour_targets = targets
            .iter()
            .map(|format| {
                Some(wgpu_types::ColorTargetState {
                    format: *format,
                    blend: None,
                    write_mask: wgpu_types::ColorWrites::ALL,
                })
            })
            .collect();
        let (id, err) = global.device_create_render_pipeline(
            device.device,
            &wgpu_core::pipeline::RenderPipelineDescriptor {
                label: Some(Cow::Borrowed(entry)),
                layout: Some(self.layout),
                vertex: wgpu_core::pipeline::VertexState {
                    stage: stage("whole"),
                    buffers: Cow::Borrowed(&[]),
                },
                primitive: wgpu_types::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu_types::MultisampleState::default(),
                fragment: Some(wgpu_core::pipeline::FragmentState {
                    stage: stage(entry),
                    targets: Cow::Owned(colour_targets),
                }),
                multiview_mask: None,
                cache: None,
            },
            None,
        );
        if let Some(e) = err {
            bail!("the {entry} pipeline: {e}");
        }
        made.insert(key, id);
        Ok(id)
    }

    /// The buffer holding `params`, written the first time it is asked for
    /// and kept: a run converts a few shapes of frame, each many times.
    fn params(&self, global: &Global, device: &SharedDevice, params: Params) -> Result<BufferId> {
        let mut made = self.params.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(id) = made.get(&params) {
            return Ok(*id);
        }
        let bytes = params.bytes();
        let (buffer, err) = global.device_create_buffer(
            device.device,
            &wgpu_types::BufferDescriptor {
                label: Some(Cow::Borrowed("ffrwd conversion parameters")),
                size: bytes.len() as u64,
                usage: wgpu_types::BufferUsages::UNIFORM | wgpu_types::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
            None,
        );
        if let Some(e) = err {
            global.buffer_drop(buffer);
            bail!("the conversion's parameter buffer: {e}");
        }
        if let Err(e) = global.queue_write_buffer(*device.queue, buffer, 0, &bytes) {
            global.buffer_drop(buffer);
            bail!("writing the conversion's parameters: {e}");
        }
        made.insert(params, buffer);
        Ok(buffer)
    }
}

/// A view of one plane as its own format, the aspect it is in.
pub(super) fn view(global: &Global, plane: &Plane) -> Result<TextureViewId> {
    let (id, err) = global.texture_create_view(
        plane.texture,
        &wgpu_core::resource::TextureViewDescriptor {
            label: None,
            format: Some(plane.format),
            dimension: Some(wgpu_types::TextureViewDimension::D2),
            usage: None,
            range: wgpu_types::ImageSubresourceRange {
                aspect: plane.aspect,
                ..Default::default()
            },
        },
        None,
    );
    match err {
        Some(e) => {
            global.texture_view_drop(id);
            Err(anyhow!("a view of a {:?} plane: {e}", plane.format))
        }
        None => Ok(id),
    }
}

/// One render pass: `entry` reading up to three `sources` (the parameters
/// say their sizes), writing `targets` (indexes into the planes made).
struct Pass<'a> {
    entry: &'static str,
    sources: Vec<&'a Plane>,
    targets: Vec<usize>,
    params: Params,
}

fn size(plane: &Plane) -> (u32, u32) {
    (plane.width, plane.height)
}

/// A pass reading `sources` into `targets` of `made`, with `yuv`.
fn pass<'a>(
    entry: &'static str,
    sources: Vec<&'a Plane>,
    targets: Vec<usize>,
    made: &[(u32, u32, TextureFormat, u32)],
    yuv: Yuv,
) -> Pass<'a> {
    let out = (made[targets[0]].0, made[targets[0]].1);
    let first = size(sources[0]);
    let rest = sources.get(1).map_or(first, |p| size(p));
    Pass {
        entry,
        sources,
        targets,
        params: Params {
            out,
            first,
            rest,
            yuv,
        },
    }
}

/// Makes a texture per `(width, height, format, bytes per pixel)` in
/// `made`, runs `passes` into them in one submission, and hands them to
/// the caller, who owns them. `what` names the work for an error.
fn render(
    global: &Global,
    device: &SharedDevice,
    what: &str,
    made: &[(u32, u32, TextureFormat, u32)],
    passes: Vec<Pass<'_>>,
) -> Result<Vec<Plane>> {
    let pipes = pipelines(global, device)?;
    let mut planes: Vec<Plane> = Vec::with_capacity(made.len());
    let mut views: Vec<TextureViewId> = Vec::new();
    let mut groups: Vec<BindGroupId> = Vec::new();
    let result = (|| -> Result<()> {
        for (w, h, format, bpp) in made {
            planes.push(create_plane(global, device, *w, *h, *format, *bpp)?);
        }
        let mut targets = Vec::with_capacity(planes.len());
        for plane in &planes {
            let v = view(global, plane)?;
            views.push(v);
            targets.push(v);
        }
        let mut recorded = Vec::with_capacity(passes.len());
        for pass in &passes {
            let formats: Vec<TextureFormat> = pass.targets.iter().map(|&i| made[i].2).collect();
            let pipeline = pipes.pipeline(global, device, pass.entry, &formats)?;
            let buffer = pipes.params(global, device, pass.params)?;
            let mut entries = vec![wgpu_core::binding_model::BindGroupEntry {
                binding: 0,
                resource: wgpu_core::binding_model::BindingResource::Buffer(
                    wgpu_core::binding_model::BufferBinding {
                        buffer,
                        offset: 0,
                        size: None,
                    },
                ),
            }];
            for binding in 0..3 {
                let source = pass.sources.get(binding).unwrap_or(&pass.sources[0]);
                let v = view(global, source)?;
                views.push(v);
                entries.push(wgpu_core::binding_model::BindGroupEntry {
                    binding: binding as u32 + 1,
                    resource: wgpu_core::binding_model::BindingResource::TextureView(v),
                });
            }
            let (group, err) = global.device_create_bind_group(
                device.device,
                &wgpu_core::binding_model::BindGroupDescriptor {
                    label: None,
                    layout: pipes.group,
                    entries: Cow::Owned(entries),
                },
                None,
            );
            groups.push(group);
            if let Some(e) = err {
                bail!("a bind group over the planes: {e}");
            }
            recorded.push((pipeline, group));
        }

        let (encoder, err) = global.device_create_command_encoder(
            device.device,
            &wgpu_types::CommandEncoderDescriptor { label: None },
            None,
        );
        if let Some(e) = err {
            bail!("a command encoder for {what}: {e}");
        }
        for (pass, (pipeline, group)) in passes.iter().zip(recorded) {
            let attachments = pass
                .targets
                .iter()
                .map(|&i| {
                    Some(wgpu_core::command::RenderPassColorAttachment {
                        view: targets[i],
                        resolve_target: None,
                        load_op: wgpu_core::command::LoadOp::Clear(wgpu_types::Color::BLACK),
                        store_op: wgpu_core::command::StoreOp::Store,
                        depth_slice: None,
                    })
                })
                .collect();
            let (mut rpass, err) = global.command_encoder_begin_render_pass(
                encoder,
                &wgpu_core::command::RenderPassDescriptor {
                    label: None,
                    color_attachments: Cow::Owned(attachments),
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                },
            );
            if let Some(e) = err {
                bail!("beginning {what}: {e}");
            }
            global
                .render_pass_set_pipeline(&mut rpass, pipeline)
                .map_err(|e| anyhow!("{e}"))?;
            global
                .render_pass_set_bind_group(&mut rpass, 0, Some(group), &[])
                .map_err(|e| anyhow!("{e}"))?;
            global
                .render_pass_draw(&mut rpass, 3, 1, 0, 0)
                .map_err(|e| anyhow!("{e}"))?;
            global
                .render_pass_end(&mut rpass)
                .map_err(|e| anyhow!("ending {what}: {e}"))?;
        }
        let (commands, err) = global.command_encoder_finish(
            encoder,
            &wgpu_types::CommandBufferDescriptor { label: None },
            None,
        );
        if let Some((_, e)) = err {
            bail!("finishing {what}: {e}");
        }
        global
            .queue_submit(*device.queue, &[commands])
            .map_err(|(_, e)| anyhow!("submitting {what}: {e}"))?;
        Ok(())
    })();
    for group in groups {
        global.bind_group_drop(group);
    }
    for v in views {
        global.texture_view_drop(v);
    }
    match result {
        Ok(()) => Ok(planes),
        Err(e) => {
            for plane in &planes {
                release_plane(global, plane.texture);
            }
            Err(e)
        }
    }
}

/// The planes of a picture held as `from` (`planes`, in that layout's
/// order; the first is its luma or its rgba, at the picture's size) made
/// into `target`, as new textures the caller owns, written by commands
/// submitted here. Says on stderr the first time each distinct conversion
/// runs, and counts it in the profile's `convert` row.
pub(super) fn convert(
    global: &Global,
    device: &SharedDevice,
    from: Layout,
    planes: &[Plane],
    target: Target,
) -> Result<Vec<Plane>> {
    let started = Instant::now();
    let made = passes(global, device, from, planes, target)?;
    PROFILE
        .convert
        .add(target.layout.len(target.width, target.height), started);
    /// A conversion said on stderr: from what at what size, to what at
    /// what size.
    type Said = (Layout, (u32, u32), Layout, (u32, u32));
    static SAID: OnceLock<Mutex<HashSet<Said>>> = OnceLock::new();
    let source = size(&planes[0]);
    let kind = (from, source, target.layout, (target.width, target.height));
    let said = SAID.get_or_init(Default::default);
    if said.lock().unwrap_or_else(|e| e.into_inner()).insert(kind) {
        eprintln!(
            "[gpu] convert {} {}x{} to {} {}x{}",
            from.name(),
            source.0,
            source.1,
            target.layout.name(),
            target.width,
            target.height
        );
    }
    Ok(made)
}

fn passes(
    global: &Global,
    device: &SharedDevice,
    from: Layout,
    planes: &[Plane],
    target: Target,
) -> Result<Vec<Plane>> {
    let to = target.layout;
    if !converts(from, to) {
        bail!(
            "a {} frame is not made {} on the device",
            from.name(),
            to.name()
        );
    }
    let wanted = from.planes(1, 1).len();
    if planes.len() != wanted {
        bail!(
            "a {} frame has {wanted} plane(s), and this one has {}",
            from.name(),
            planes.len()
        );
    }
    let (width, height) = (target.width, target.height);
    if width == 0 || height == 0 {
        bail!("a frame is converted to a size of at least 1x1, and {width}x{height} was asked for");
    }
    let what = format!("the {} to {} conversion", from.name(), to.name());
    let made = to.planes(width, height);
    let yuv = target.yuv;
    let passes = match (from, to) {
        (Layout::Rgba, Layout::Nv12) if size(&planes[0]) != (width, height) => vec![
            pass("rgba_luma_resized", vec![&planes[0]], vec![0], &made, yuv),
            pass("rgba_chroma_resized", vec![&planes[0]], vec![1], &made, yuv),
        ],
        (Layout::Rgba, Layout::Yuv420p) if size(&planes[0]) != (width, height) => vec![
            pass("rgba_luma_resized", vec![&planes[0]], vec![0], &made, yuv),
            pass(
                "rgba_chroma_planar_resized",
                vec![&planes[0]],
                vec![1, 2],
                &made,
                yuv,
            ),
        ],
        (Layout::Rgba, Layout::Nv12) => vec![
            pass("rgba_luma", vec![&planes[0]], vec![0], &made, yuv),
            pass("rgba_chroma", vec![&planes[0]], vec![1], &made, yuv),
        ],
        (Layout::Rgba, Layout::Yuv420p) => vec![
            pass("rgba_luma", vec![&planes[0]], vec![0], &made, yuv),
            pass(
                "rgba_chroma_planar",
                vec![&planes[0]],
                vec![1, 2],
                &made,
                yuv,
            ),
        ],
        (a, b) if a == b => planes
            .iter()
            .enumerate()
            .map(|(i, plane)| pass("plane", vec![plane], vec![i], &made, yuv))
            .collect(),
        (Layout::Nv12, Layout::Rgba) => {
            vec![pass(
                "nv12_rgba",
                planes.iter().collect(),
                vec![0],
                &made,
                yuv,
            )]
        }
        (_, Layout::Rgba) => vec![pass(
            "planar_rgba",
            planes.iter().collect(),
            vec![0],
            &made,
            yuv,
        )],
        (Layout::Nv12, _) => vec![
            pass("plane", vec![&planes[0]], vec![0], &made, yuv),
            pass("split", vec![&planes[1]], vec![1, 2], &made, yuv),
        ],
        (_, Layout::Nv12) => vec![
            pass("plane", vec![&planes[0]], vec![0], &made, yuv),
            pass(
                "interleave",
                vec![&planes[1], &planes[2]],
                vec![1],
                &made,
                yuv,
            ),
        ],
        _ => planes
            .iter()
            .enumerate()
            .map(|(i, plane)| pass("plane", vec![plane], vec![i], &made, yuv))
            .collect(),
    };
    render(global, device, &what, &made, passes)
}
