//! A frame made into a model's input tensor: resized to the tensor's size,
//! its colour made RGB, scaled and normalised, written as fp32 in the
//! layout the model takes.
//!
//! On the device this is one compute pass over the frame's planes as they
//! are held (an NV12 picture's two, a planar YUV frame's three, an rgba
//! frame's one) into a buffer of its own. [`host_tensor`] does the same on
//! the CPU from a frame's bytes, in the same order of operations, so a
//! tensor made either way holds the same numbers to within the rounding of
//! the two processors' arithmetic.
//!
//! The resize is bilinear from pixel centres, with the edges repeated; the
//! chroma of a subsampled frame is sited at the centre of the pixels it
//! covers. YUV is made RGB with the stream's matrix and range and clamped
//! to 0 to 1; rgba is read as it is (premultiplied, alpha dropped).
//!
//! The buffers come from a pool and go back to it when nothing holds the
//! tensor any longer, so a buffer another API imported once (CUDA, through
//! a shared heap on D3D12) is imported once for as long as the process
//! runs, however many frames pass through it.

use std::borrow::Cow;
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::time::Instant;

use anyhow::{anyhow, bail, Result};
use wasi_webgpu_wasmtime::reexports::{wgpu_core, wgpu_types};
use wasi_webgpu_wasmtime::SharedDevice;
use wgpu_core::global::Global;
use wgpu_core::id::{BindGroupLayoutId, BufferId, ComputePipelineId, TextureViewId};
use wgpu_types::BufferUsages;

use super::convert::{Matrix, Yuv};
use super::frames::Plane;
use crate::profile::PROFILE;

/// How a tensor's elements are laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// [1, 3, height, width]: one plane per channel.
    Nchw,
    /// [1, height, width, 3]: each pixel's channels together.
    Nhwc,
}

/// The order of a tensor's three channels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    Rgb,
    Bgr,
}

/// How a frame is fitted to a tensor's size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fit {
    /// Each side scaled on its own.
    Stretch,
    /// Both sides by the largest factor that fits, centred, the rest black.
    Letterbox,
}

/// What a frame is made into.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TensorSpec {
    pub width: u32,
    pub height: u32,
    pub layout: Layout,
    pub order: Order,
    pub fit: Fit,
    /// Each channel, 0 to 1, becomes `(value * scale - mean) / std`.
    pub scale: f32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    /// The stream's matrix and range, for a YUV frame.
    pub yuv: Yuv,
}

impl TensorSpec {
    /// Refuses a spec no tensor can be made to.
    pub fn check(&self) -> Result<()> {
        if self.width == 0 || self.height == 0 {
            bail!("a {}x{} tensor has no pixels", self.width, self.height);
        }
        if self.std.contains(&0.0) || self.std.iter().any(|s| !s.is_finite()) {
            bail!("std {:?} divides by zero or is not a number", self.std);
        }
        if self.elements() > (u32::MAX / 4) as usize {
            bail!(
                "a {}x{} tensor is too large to make",
                self.width,
                self.height
            );
        }
        Ok(())
    }

    /// The tensor's dimensions, as `wasi:nn` gives them.
    pub fn dimensions(&self) -> Vec<u32> {
        match self.layout {
            Layout::Nchw => vec![1, 3, self.height, self.width],
            Layout::Nhwc => vec![1, self.height, self.width, 3],
        }
    }

    pub fn elements(&self) -> usize {
        3 * self.width as usize * self.height as usize
    }

    /// Bytes the tensor's fp32 elements take.
    pub fn byte_len(&self) -> u64 {
        4 * self.elements() as u64
    }

    /// Where a `source_width` x `source_height` picture lands in the
    /// tensor: its left, top, width and height.
    fn placement(&self, source_width: u32, source_height: u32) -> [u32; 4] {
        match self.fit {
            Fit::Stretch => [0, 0, self.width, self.height],
            Fit::Letterbox => {
                let scale = (self.width as f32 / source_width as f32)
                    .min(self.height as f32 / source_height as f32);
                let w = ((source_width as f32 * scale).round() as u32).clamp(1, self.width);
                let h = ((source_height as f32 * scale).round() as u32).clamp(1, self.height);
                [(self.width - w) / 2, (self.height - h) / 2, w, h]
            }
        }
    }
}

/// The matrix's red and blue weights.
fn weights(yuv: Yuv) -> (f32, f32) {
    match yuv.matrix {
        Matrix::Bt709 => (0.2126, 0.0722),
        Matrix::Bt601 => (0.299, 0.114),
    }
}

// The CPU's half. Every step is the device pass's, in its order.

/// One plane of a frame's bytes: `channels` bytes per pixel, rows packed.
struct HostPlane<'a> {
    bytes: &'a [u8],
    width: u32,
    height: u32,
    channels: usize,
}

impl HostPlane<'_> {
    fn texel(&self, x: i32, y: i32, channel: usize) -> f32 {
        let x = x.clamp(0, self.width as i32 - 1) as usize;
        let y = y.clamp(0, self.height as i32 - 1) as usize;
        let at = (y * self.width as usize + x) * self.channels + channel;
        f32::from(self.bytes[at]) / 255.0
    }

    fn bilinear(&self, x: f32, y: f32, channel: usize) -> f32 {
        let (bx, by) = (x.floor(), y.floor());
        let (fx, fy) = (x - bx, y - by);
        let (x0, y0) = (bx as i32, by as i32);
        let a = self.texel(x0, y0, channel);
        let b = self.texel(x0 + 1, y0, channel);
        let c = self.texel(x0, y0 + 1, channel);
        let d = self.texel(x0 + 1, y0 + 1, channel);
        let top = a + (b - a) * fx;
        let bottom = c + (d - c) * fx;
        top + (bottom - top) * fy
    }
}

/// One coded YUV sample made RGB, as the device pass does it.
fn yuv_to_rgb(y: f32, cb: f32, cr: f32, yuv: Yuv) -> [f32; 3] {
    let (kr, kb) = weights(yuv);
    let (l, u, v) = if yuv.full_range {
        (
            y,
            (cb * 255.0 - 128.0) / 255.0,
            (cr * 255.0 - 128.0) / 255.0,
        )
    } else {
        (
            (y * 255.0 - 16.0) / 219.0,
            (cb * 255.0 - 128.0) / 224.0,
            (cr * 255.0 - 128.0) / 224.0,
        )
    };
    let r = l + 2.0 * (1.0 - kr) * v;
    let b = l + 2.0 * (1.0 - kb) * u;
    let g = (l - kr * r - kb * b) / (1.0 - kr - kb);
    [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)]
}

/// The tensor's elements, made on the CPU from a frame's bytes packed as
/// the wire lays `pix_fmt` out (rgba, yuv420p, yuv422p or yuv444p) at
/// `width` x `height`.
pub fn host_tensor(
    bytes: &[u8],
    pix_fmt: &str,
    width: u32,
    height: u32,
    spec: &TensorSpec,
) -> Result<Vec<f32>> {
    spec.check()?;
    let specs = super::frames::layout(pix_fmt, width, height)
        .ok_or_else(|| anyhow!("a {pix_fmt} frame is not made into a tensor"))?;
    let expected: usize = specs
        .iter()
        .map(|(w, h, _, bpp)| *w as usize * *h as usize * *bpp as usize)
        .sum();
    if bytes.len() != expected {
        bail!(
            "a {width}x{height} {pix_fmt} frame is {expected} bytes, and {} were handed over",
            bytes.len()
        );
    }
    let mut planes = Vec::with_capacity(specs.len());
    let mut offset = 0;
    for (w, h, _, bpp) in specs {
        let len = w as usize * h as usize * bpp as usize;
        planes.push(HostPlane {
            bytes: &bytes[offset..offset + len],
            width: w,
            height: h,
            channels: bpp as usize,
        });
        offset += len;
    }
    let [left, top, box_w, box_h] = spec.placement(width, height);
    let out_w = spec.width as usize;
    let plane_len = out_w * spec.height as usize;
    let mut out = vec![0f32; spec.elements()];
    for oy in 0..spec.height {
        for ox in 0..spec.width {
            let inside = ox >= left && oy >= top && ox < left + box_w && oy < top + box_h;
            let mut rgb = [0f32; 3];
            if inside {
                let x = ((ox - left) as f32 + 0.5) * width as f32 / box_w as f32 - 0.5;
                let y = ((oy - top) as f32 + 0.5) * height as f32 / box_h as f32 - 0.5;
                rgb = if planes.len() == 1 {
                    let p = &planes[0];
                    [
                        p.bilinear(x, y, 0),
                        p.bilinear(x, y, 1),
                        p.bilinear(x, y, 2),
                    ]
                } else {
                    let luma = planes[0].bilinear(x, y, 0);
                    let (cw, ch) = (planes[1].width, planes[1].height);
                    let cx = (x + 0.5) * cw as f32 / width as f32 - 0.5;
                    let cy = (y + 0.5) * ch as f32 / height as f32 - 0.5;
                    let cb = planes[1].bilinear(cx, cy, 0);
                    let cr = planes[2].bilinear(cx, cy, 0);
                    yuv_to_rgb(luma, cb, cr, spec.yuv)
                };
            }
            if spec.order == Order::Bgr {
                rgb.swap(0, 2);
            }
            let pixel = oy as usize * out_w + ox as usize;
            for (c, value) in rgb.iter().enumerate() {
                let v = (value * spec.scale - spec.mean[c]) / spec.std[c];
                match spec.layout {
                    Layout::Nchw => out[c * plane_len + pixel] = v,
                    Layout::Nhwc => out[pixel * 3 + c] = v,
                }
            }
        }
    }
    Ok(out)
}

/// fp32 elements as the little-endian bytes a `wasi:nn` tensor carries.
pub fn to_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

// The device's half.

const PASS: &str = r#"
struct Params {
    out_w: u32, out_h: u32, src_w: u32, src_h: u32,
    chroma_w: u32, chroma_h: u32, box_x: u32, box_y: u32,
    box_w: u32, box_h: u32, kind: u32, nhwc: u32,
    bgr: u32, full: u32, kr: f32, kb: f32,
    scale: f32, pad0: f32, pad1: f32, pad2: f32,
    mean: vec4<f32>,
    deviation: vec4<f32>,
}

@group(0) @binding(0) var plane0: texture_2d<f32>;
@group(0) @binding(1) var plane1: texture_2d<f32>;
@group(0) @binding(2) var plane2: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> tensor: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;

fn texel(t: texture_2d<f32>, x: i32, y: i32, w: u32, h: u32) -> vec4<f32> {
    let at = vec2<i32>(clamp(x, 0, i32(w) - 1), clamp(y, 0, i32(h) - 1));
    return textureLoad(t, at, 0);
}

fn bilinear(t: texture_2d<f32>, x: f32, y: f32, w: u32, h: u32) -> vec4<f32> {
    let bx = floor(x);
    let by = floor(y);
    let fx = x - bx;
    let fy = y - by;
    let x0 = i32(bx);
    let y0 = i32(by);
    let a = texel(t, x0, y0, w, h);
    let b = texel(t, x0 + 1, y0, w, h);
    let c = texel(t, x0, y0 + 1, w, h);
    let d = texel(t, x0 + 1, y0 + 1, w, h);
    let top = a + (b - a) * fx;
    let bottom = c + (d - c) * fx;
    return top + (bottom - top) * fy;
}

fn yuv_to_rgb(y: f32, cb: f32, cr: f32) -> vec3<f32> {
    var l: f32;
    var u: f32;
    var v: f32;
    if (p.full != 0u) {
        l = y;
        u = (cb * 255.0 - 128.0) / 255.0;
        v = (cr * 255.0 - 128.0) / 255.0;
    } else {
        l = (y * 255.0 - 16.0) / 219.0;
        u = (cb * 255.0 - 128.0) / 224.0;
        v = (cr * 255.0 - 128.0) / 224.0;
    }
    let r = l + 2.0 * (1.0 - p.kr) * v;
    let b = l + 2.0 * (1.0 - p.kb) * u;
    let g = (l - p.kr * r - p.kb * b) / (1.0 - p.kr - p.kb);
    return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

// kind 0: one rgba plane. 1: three planes, Y, Cb, Cr. 2: Y, and Cb and Cr
// interleaved in one plane (NV12).
fn sample(x: f32, y: f32) -> vec3<f32> {
    if (p.kind == 0u) {
        return bilinear(plane0, x, y, p.src_w, p.src_h).rgb;
    }
    let luma = bilinear(plane0, x, y, p.src_w, p.src_h).r;
    let cx = (x + 0.5) * f32(p.chroma_w) / f32(p.src_w) - 0.5;
    let cy = (y + 0.5) * f32(p.chroma_h) / f32(p.src_h) - 0.5;
    var cb: f32;
    var cr: f32;
    if (p.kind == 2u) {
        let uv = bilinear(plane1, cx, cy, p.chroma_w, p.chroma_h);
        cb = uv.r;
        cr = uv.g;
    } else {
        cb = bilinear(plane1, cx, cy, p.chroma_w, p.chroma_h).r;
        cr = bilinear(plane2, cx, cy, p.chroma_w, p.chroma_h).r;
    }
    return yuv_to_rgb(luma, cb, cr);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.out_w || id.y >= p.out_h) {
        return;
    }
    var rgb = vec3<f32>(0.0);
    let inside = id.x >= p.box_x && id.y >= p.box_y
        && id.x < p.box_x + p.box_w && id.y < p.box_y + p.box_h;
    if (inside) {
        let x = (f32(id.x - p.box_x) + 0.5) * f32(p.src_w) / f32(p.box_w) - 0.5;
        let y = (f32(id.y - p.box_y) + 0.5) * f32(p.src_h) / f32(p.box_h) - 0.5;
        rgb = sample(x, y);
    }
    if (p.bgr != 0u) {
        rgb = rgb.bgr;
    }
    let plane = p.out_w * p.out_h;
    let pixel = id.y * p.out_w + id.x;
    for (var c = 0u; c < 3u; c++) {
        let v = (rgb[c] * p.scale - p.mean[c]) / p.deviation[c];
        if (p.nhwc != 0u) {
            tensor[pixel * 3u + c] = v;
        } else {
            tensor[c * plane + pixel] = v;
        }
    }
}
"#;

/// Bytes of the pass's `Params`.
const PARAMS_LEN: usize = 112;

/// How a frame's planes are held, as the pass reads them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// One rgba plane.
    Rgba,
    /// Y, Cb and Cr in three planes.
    Planar,
    /// Y, then Cb and Cr interleaved (NV12).
    Interleaved,
}

impl Source {
    /// How a frame held as `pix_fmt` is read.
    pub fn of(pix_fmt: &str) -> Option<Source> {
        match pix_fmt {
            "rgba" => Some(Source::Rgba),
            "yuv420p" | "yuv422p" | "yuv444p" => Some(Source::Planar),
            "nv12" => Some(Source::Interleaved),
            _ => None,
        }
    }

    fn kind(self) -> u32 {
        match self {
            Source::Rgba => 0,
            Source::Planar => 1,
            Source::Interleaved => 2,
        }
    }

    fn planes(self) -> usize {
        match self {
            Source::Rgba => 1,
            Source::Planar => 3,
            Source::Interleaved => 2,
        }
    }
}

fn params(spec: &TensorSpec, source: Source, planes: &[Plane]) -> [u8; PARAMS_LEN] {
    let (width, height) = (planes[0].width, planes[0].height);
    let (chroma_w, chroma_h) = planes
        .get(1)
        .map(|c| (c.width, c.height))
        .unwrap_or((width, height));
    let [left, top, box_w, box_h] = spec.placement(width, height);
    let (kr, kb) = weights(spec.yuv);
    let words: [u32; 20] = [
        spec.width,
        spec.height,
        width,
        height,
        chroma_w,
        chroma_h,
        left,
        top,
        box_w,
        box_h,
        source.kind(),
        u32::from(spec.layout == Layout::Nhwc),
        u32::from(spec.order == Order::Bgr),
        u32::from(spec.yuv.full_range),
        kr.to_bits(),
        kb.to_bits(),
        spec.scale.to_bits(),
        0,
        0,
        0,
    ];
    let mut out = [0u8; PARAMS_LEN];
    for (i, word) in words.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    for c in 0..3 {
        out[80 + c * 4..84 + c * 4].copy_from_slice(&spec.mean[c].to_le_bytes());
        out[96 + c * 4..100 + c * 4].copy_from_slice(&spec.std[c].to_le_bytes());
    }
    out[108..112].copy_from_slice(&1f32.to_le_bytes());
    out
}

/// The pass's pipeline and its bind group layout, made once a process.
struct Pipeline {
    layout: BindGroupLayoutId,
    pipeline: ComputePipelineId,
}

fn pipeline(global: &Global, device: &SharedDevice) -> Result<&'static Pipeline> {
    static MADE: OnceLock<std::result::Result<Pipeline, String>> = OnceLock::new();
    MADE.get_or_init(|| make_pipeline(global, device).map_err(|e| format!("{e:#}")))
        .as_ref()
        .map_err(|e| anyhow!("the tensor pipeline: {e}"))
}

fn make_pipeline(global: &Global, device: &SharedDevice) -> Result<Pipeline> {
    let (module, err) = global.device_create_shader_module(
        device.device,
        &wgpu_core::pipeline::ShaderModuleDescriptor {
            label: Some(Cow::Borrowed("ffrwd frame tensor")),
            runtime_checks: wgpu_types::ShaderRuntimeChecks::checked(),
        },
        wgpu_core::pipeline::ShaderModuleSource::Wgsl(Cow::Borrowed(PASS)),
        None,
    );
    if let Some(e) = err {
        bail!("compiling the shader: {e}");
    }
    let texture = |binding| wgpu_types::BindGroupLayoutEntry {
        binding,
        visibility: wgpu_types::ShaderStages::COMPUTE,
        ty: wgpu_types::BindingType::Texture {
            sample_type: wgpu_types::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu_types::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let entries = vec![
        texture(0),
        texture(1),
        texture(2),
        wgpu_types::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu_types::ShaderStages::COMPUTE,
            ty: wgpu_types::BindingType::Buffer {
                ty: wgpu_types::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu_types::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu_types::ShaderStages::COMPUTE,
            ty: wgpu_types::BindingType::Buffer {
                ty: wgpu_types::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
    ];
    let (layout, err) = global.device_create_bind_group_layout(
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
    let (pipeline_layout, err) = global.device_create_pipeline_layout(
        device.device,
        &wgpu_core::binding_model::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: Cow::Owned(vec![Some(layout)]),
            immediate_size: 0,
        },
        None,
    );
    if let Some(e) = err {
        bail!("the pipeline layout: {e}");
    }
    let (pipeline, err) = global.device_create_compute_pipeline(
        device.device,
        &wgpu_core::pipeline::ComputePipelineDescriptor {
            label: Some(Cow::Borrowed("ffrwd frame tensor")),
            layout: Some(pipeline_layout),
            stage: wgpu_core::pipeline::ProgrammableStageDescriptor {
                module,
                entry_point: Some(Cow::Borrowed("main")),
                constants: Default::default(),
                zero_initialize_workgroup_memory: true,
            },
            cache: None,
        },
        None,
    );
    if let Some(e) = err {
        bail!("the compute pipeline: {e}");
    }
    Ok(Pipeline { layout, pipeline })
}

/// A buffer the pass writes, and whatever another API made of it.
pub struct PooledBuffer {
    global: Arc<Global>,
    pub id: BufferId,
    pub size: u64,
    /// The NT handle another API imports the buffer through, for a buffer
    /// on a shared heap.
    #[cfg(windows)]
    pub shared: Option<super::d3d12::SharedHandle>,
    /// The buffer as CUDA maps it, imported the first time a CUDA session
    /// binds it, and kept for the buffer's life.
    #[cfg(windows)]
    pub cuda: OnceLock<std::result::Result<crate::nn::cuda::CudaBuffer, String>>,
    /// The D3D12 buffer, where it was made on a shared heap.
    #[cfg(windows)]
    pub resource: Option<windows::Win32::Graphics::Direct3D12::ID3D12Resource>,
    /// The buffer as a DirectML allocation, made the first time a DirectML
    /// session binds it.
    #[cfg(windows)]
    pub dml: OnceLock<std::result::Result<Arc<crate::nn::directml::Allocation>, String>>,
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        // CUDA's mapping and DirectML's allocation go before the memory
        // under them.
        #[cfg(windows)]
        drop(self.cuda.take());
        #[cfg(windows)]
        drop(self.dml.take());
        self.global.buffer_drop(self.id);
    }
}

/// Buffers of one kind, reused once nothing but the pool holds them.
#[derive(Default)]
struct Pool {
    buffers: Mutex<Vec<Arc<PooledBuffer>>>,
}

impl Pool {
    fn take(
        &self,
        global: &Arc<Global>,
        device: &SharedDevice,
        size: u64,
        shared: bool,
    ) -> Result<Arc<PooledBuffer>> {
        let mut buffers = self.buffers.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(free) = buffers
            .iter()
            .find(|b| b.size == size && Arc::strong_count(b) == 1)
        {
            return Ok(Arc::clone(free));
        }
        let made = if shared {
            shared_buffer(global, device, size)?
        } else {
            let (id, err) = global.device_create_buffer(
                device.device,
                &wgpu_types::BufferDescriptor {
                    label: Some(Cow::Borrowed("ffrwd frame tensor")),
                    size,
                    usage: USAGE,
                    mapped_at_creation: false,
                },
                None,
            );
            if let Some(e) = err {
                global.buffer_drop(id);
                bail!("a {size}-byte tensor buffer: {e}");
            }
            PooledBuffer {
                global: Arc::clone(global),
                id,
                size,
                #[cfg(windows)]
                shared: None,
                #[cfg(windows)]
                cuda: OnceLock::new(),
                #[cfg(windows)]
                resource: None,
                #[cfg(windows)]
                dml: OnceLock::new(),
            }
        };
        let made = Arc::new(made);
        buffers.push(Arc::clone(&made));
        Ok(made)
    }
}

/// What a tensor buffer is used as: written by the pass, copied out.
const USAGE: BufferUsages = BufferUsages::STORAGE.union(BufferUsages::COPY_SRC);

/// A pool buffer on a shared heap, which another API imports.
#[cfg(windows)]
fn shared_buffer(global: &Arc<Global>, device: &SharedDevice, size: u64) -> Result<PooledBuffer> {
    let (id, handle, resource) = super::d3d12::shared_buffer(global, device, size, USAGE)?;
    Ok(PooledBuffer {
        global: Arc::clone(global),
        id,
        size,
        shared: Some(handle),
        cuda: OnceLock::new(),
        resource: Some(resource),
        dml: OnceLock::new(),
    })
}

#[cfg(not(windows))]
fn shared_buffer(
    _global: &Arc<Global>,
    _device: &SharedDevice,
    _size: u64,
) -> Result<PooledBuffer> {
    bail!("a buffer another API imports is made on D3D12 only, so far")
}

/// The pools: buffers on a shared heap, and ordinary ones.
/// A `size`-byte storage buffer of the process's device from the pool, on a
/// shared heap where `shared` asks; it goes back to the pool once nothing
/// holds it.
pub(crate) fn pooled_buffer(
    global: &Arc<Global>,
    device: &SharedDevice,
    size: u64,
    shared: bool,
) -> Result<Arc<PooledBuffer>> {
    pool(shared).take(global, device, size, shared)
}

fn pool(shared: bool) -> &'static Pool {
    static SHARED: OnceLock<Pool> = OnceLock::new();
    static PLAIN: OnceLock<Pool> = OnceLock::new();
    if shared {
        SHARED.get_or_init(Pool::default)
    } else {
        PLAIN.get_or_init(Pool::default)
    }
}

/// A tensor made on the device: the buffer holding it, and the value the
/// shared sync object reaches once the pass has written it.
pub struct DeviceTensor {
    global: Arc<Global>,
    device: Arc<SharedDevice>,
    pub buffer: Arc<PooledBuffer>,
    pub spec: TensorSpec,
    /// For a buffer on a shared heap: the sync object's value once written.
    pub written_at: Option<u64>,
}

impl DeviceTensor {
    /// The tensor's bytes read back to host memory, counted as a download.
    pub fn read_back(&self) -> Result<Vec<u8>> {
        let started = Instant::now();
        let bytes = read_buffer(
            &self.global,
            &self.device,
            self.buffer.id,
            self.spec.byte_len(),
        )?;
        PROFILE.download.add(bytes.len(), started);
        Ok(bytes)
    }
}

/// Makes the tensor of a frame held as `planes`, read as `source`, into a
/// buffer from the pool (`shared`: one on a shared heap, for another API to
/// import), with commands submitted here.
pub(super) fn make(
    global: &Arc<Global>,
    device: &Arc<SharedDevice>,
    planes: &[Plane],
    source: Source,
    spec: &TensorSpec,
    shared: bool,
) -> Result<DeviceTensor> {
    spec.check()?;
    if planes.len() != source.planes() {
        bail!(
            "a frame read as {source:?} has {} planes, and this one has {}",
            source.planes(),
            planes.len()
        );
    }
    let pipes = pipeline(global, device)?;
    let buffer = pool(shared).take(global, device, spec.byte_len(), shared)?;

    let (uniform, err) = global.device_create_buffer(
        device.device,
        &wgpu_types::BufferDescriptor {
            label: None,
            size: PARAMS_LEN as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
        None,
    );
    let mut views: Vec<TextureViewId> = Vec::new();
    let mut group = None;
    let result = (|| -> Result<()> {
        if let Some(e) = err {
            bail!("the pass's parameters: {e}");
        }
        global
            .queue_write_buffer(*device.queue, uniform, 0, &params(spec, source, planes))
            .map_err(|e| anyhow!("writing the pass's parameters: {e}"))?;
        for plane in planes {
            views.push(super::convert::view(global, plane)?);
        }
        // Unused bindings read the last plane; the pass never samples them.
        let at = |i: usize| views[i.min(views.len() - 1)];
        let entries = vec![
            wgpu_core::binding_model::BindGroupEntry {
                binding: 0,
                resource: wgpu_core::binding_model::BindingResource::TextureView(at(0)),
            },
            wgpu_core::binding_model::BindGroupEntry {
                binding: 1,
                resource: wgpu_core::binding_model::BindingResource::TextureView(at(1)),
            },
            wgpu_core::binding_model::BindGroupEntry {
                binding: 2,
                resource: wgpu_core::binding_model::BindingResource::TextureView(at(2)),
            },
            wgpu_core::binding_model::BindGroupEntry {
                binding: 3,
                resource: wgpu_core::binding_model::BindingResource::Buffer(
                    wgpu_core::binding_model::BufferBinding {
                        buffer: buffer.id,
                        offset: 0,
                        size: Some(spec.byte_len()),
                    },
                ),
            },
            wgpu_core::binding_model::BindGroupEntry {
                binding: 4,
                resource: wgpu_core::binding_model::BindingResource::Buffer(
                    wgpu_core::binding_model::BufferBinding {
                        buffer: uniform,
                        offset: 0,
                        size: None,
                    },
                ),
            },
        ];
        let (made, err) = global.device_create_bind_group(
            device.device,
            &wgpu_core::binding_model::BindGroupDescriptor {
                label: None,
                layout: pipes.layout,
                entries: Cow::Owned(entries),
            },
            None,
        );
        group = Some(made);
        if let Some(e) = err {
            bail!("the pass's bind group: {e}");
        }
        let (encoder, err) = global.device_create_command_encoder(
            device.device,
            &wgpu_types::CommandEncoderDescriptor { label: None },
            None,
        );
        if let Some(e) = err {
            bail!("a command encoder for the pass: {e}");
        }
        let (mut pass, err) = global.command_encoder_begin_compute_pass(
            encoder,
            &wgpu_core::command::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            },
        );
        if let Some(e) = err {
            bail!("beginning the pass: {e}");
        }
        global
            .compute_pass_set_pipeline(&mut pass, pipes.pipeline)
            .map_err(|e| anyhow!("{e}"))?;
        global
            .compute_pass_set_bind_group(&mut pass, 0, Some(made), &[])
            .map_err(|e| anyhow!("{e}"))?;
        global
            .compute_pass_dispatch_workgroups(
                &mut pass,
                spec.width.div_ceil(8),
                spec.height.div_ceil(8),
                1,
            )
            .map_err(|e| anyhow!("{e}"))?;
        global
            .compute_pass_end(&mut pass)
            .map_err(|e| anyhow!("ending the pass: {e}"))?;
        let (commands, err) = global.command_encoder_finish(
            encoder,
            &wgpu_types::CommandBufferDescriptor { label: None },
            None,
        );
        if let Some((_, e)) = err {
            bail!("finishing the pass: {e}");
        }
        global
            .queue_submit(*device.queue, &[commands])
            .map_err(|(_, e)| anyhow!("submitting the pass: {e}"))?;
        Ok(())
    })();
    if let Some(group) = group {
        global.bind_group_drop(group);
    }
    for view in views {
        global.texture_view_drop(view);
    }
    global.buffer_drop(uniform);
    result?;

    #[cfg(windows)]
    let written_at = if shared {
        Some(super::d3d12::SharedSync::get()?.signal(global, device)?)
    } else {
        None
    };
    #[cfg(not(windows))]
    let written_at = None;
    Ok(DeviceTensor {
        global: Arc::clone(global),
        device: Arc::clone(device),
        buffer,
        spec: *spec,
        written_at,
    })
}

/// The first `len` bytes of `buffer`, copied out on the device and read.
pub(crate) fn read_buffer(
    global: &Global,
    device: &SharedDevice,
    buffer: BufferId,
    len: u64,
) -> Result<Vec<u8>> {
    let (staging, err) = global.device_create_buffer(
        device.device,
        &wgpu_types::BufferDescriptor {
            label: Some(Cow::Borrowed("ffrwd tensor readback")),
            size: len,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        },
        None,
    );
    let result = (|| -> Result<Vec<u8>> {
        if let Some(e) = err {
            bail!("a {len}-byte readback buffer: {e}");
        }
        let (encoder, err) = global.device_create_command_encoder(
            device.device,
            &wgpu_types::CommandEncoderDescriptor { label: None },
            None,
        );
        if let Some(e) = err {
            bail!("a command encoder for a readback: {e}");
        }
        global
            .command_encoder_copy_buffer_to_buffer(encoder, buffer, 0, staging, 0, Some(len))
            .map_err(|e| anyhow!("copying a tensor out: {e}"))?;
        let (commands, err) = global.command_encoder_finish(
            encoder,
            &wgpu_types::CommandBufferDescriptor { label: None },
            None,
        );
        if let Some((_, e)) = err {
            bail!("finishing a readback's commands: {e}");
        }
        global
            .queue_submit(*device.queue, &[commands])
            .map_err(|(_, e)| anyhow!("submitting a readback: {e}"))?;
        let (sender, receiver) = mpsc::channel();
        let op = wgpu_core::resource::BufferMapOperation {
            host: wgpu_core::device::HostMap::Read,
            callback: Some(Box::new(move |result| {
                let _ = sender.send(result);
            })),
        };
        global
            .buffer_map_async(staging, 0, None, op)
            .map_err(|e| anyhow!("mapping a readback: {e}"))?;
        loop {
            if let Ok(result) = receiver.try_recv() {
                result.map_err(|e| anyhow!("mapping a readback: {e}"))?;
                break;
            }
            global
                .poll_all_devices(true)
                .map_err(|e| anyhow!("waiting for a readback: {e}"))?;
        }
        let (ptr, mapped) = global
            .buffer_get_mapped_range(staging, 0, None)
            .map_err(|e| anyhow!("reading a readback: {e}"))?;
        // SAFETY: the mapping is the buffer's own memory, valid until
        // unmapped below.
        let bytes = unsafe { std::slice::from_raw_parts(ptr.as_ptr(), mapped as usize) }.to_vec();
        global
            .buffer_unmap(staging)
            .map_err(|e| anyhow!("unmapping a readback: {e}"))?;
        Ok(bytes)
    })();
    global.buffer_drop(staging);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::frames::GpuFrame;
    use crate::gpu::shared_device;

    fn spec(width: u32, height: u32, layout: Layout, fit: Fit) -> TensorSpec {
        TensorSpec {
            width,
            height,
            layout,
            order: Order::Rgb,
            fit,
            scale: 1.0,
            mean: [0.0; 3],
            std: [1.0; 3],
            yuv: Yuv::DEFAULT,
        }
    }

    #[test]
    fn a_letterbox_centres_the_picture_and_keeps_its_shape() {
        let s = spec(100, 100, Layout::Nchw, Fit::Letterbox);
        assert_eq!(s.placement(200, 100), [0, 25, 100, 50]);
        assert_eq!(s.placement(100, 200), [25, 0, 50, 100]);
        assert_eq!(
            spec(100, 100, Layout::Nchw, Fit::Stretch).placement(7, 3),
            [0, 0, 100, 100]
        );
    }

    #[test]
    fn a_grey_frame_is_grey_in_every_channel_and_black_outside_a_letterbox() {
        // Limited-range mid grey: Y 126 (that is, (126 - 16) / 219), chroma
        // at the centre.
        let (w, h) = (8u32, 4u32);
        let mut bytes = vec![126u8; (w * h) as usize];
        bytes.extend(vec![128u8; 2 * (w / 2 * h / 2) as usize]);
        let s = spec(4, 4, Layout::Nchw, Fit::Letterbox);
        let out = host_tensor(&bytes, "yuv420p", w, h, &s).unwrap();
        let grey = (126.0 * 1.0 / 255.0 * 255.0 - 16.0) / 219.0;
        for c in 0..3 {
            for y in 0..4 {
                for x in 0..4 {
                    let v = out[c * 16 + y * 4 + x];
                    if y == 0 || y == 3 {
                        assert_eq!(v, 0.0, "the bars are black");
                    } else {
                        assert!((v - grey).abs() < 1e-5, "{v} at {x},{y},{c}");
                    }
                }
            }
        }
    }

    #[test]
    fn the_device_pass_makes_what_the_cpu_makes() {
        let Some(device) = shared_device() else {
            eprintln!("SKIPPED: this machine has no GPU adapter.");
            return;
        };
        let (w, h) = (37u32, 23u32);
        let rgba: Vec<u8> = (0..w * h * 4).map(|i| (i * 13 % 251) as u8).collect();
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let yuv: Vec<u8> = (0..w * h + 2 * cw * ch)
            .map(|i| (i * 7 % 253) as u8)
            .collect();
        let mut cases = Vec::new();
        for layout in [Layout::Nchw, Layout::Nhwc] {
            for fit in [Fit::Stretch, Fit::Letterbox] {
                let mut s = spec(16, 12, layout, fit);
                s.order = Order::Bgr;
                s.mean = [0.485, 0.456, 0.406];
                s.std = [0.229, 0.224, 0.225];
                cases.push(s);
            }
        }
        let mut full = spec(9, 30, Layout::Nchw, Fit::Stretch);
        full.yuv = Yuv {
            matrix: Matrix::Bt601,
            full_range: true,
        };
        full.scale = 255.0;
        cases.push(full);
        for (pix_fmt, bytes) in [("rgba", &rgba), ("yuv420p", &yuv)] {
            let frame = GpuFrame::upload(pix_fmt, w, h, Arc::new(bytes.clone())).unwrap();
            for s in &cases {
                let cpu = host_tensor(bytes, pix_fmt, w, h, s).unwrap();
                let made = frame.tensor(s, false).unwrap();
                let read = made.read_back().unwrap();
                let gpu: Vec<f32> = read
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect();
                assert_eq!(gpu.len(), cpu.len());
                // Relative past 1: `scale` 255 makes values in the hundreds.
                let worst = cpu
                    .iter()
                    .zip(&gpu)
                    .map(|(a, b)| (a - b).abs() / a.abs().max(1.0))
                    .fold(0f32, f32::max);
                assert!(worst < 1e-4, "{pix_fmt} {s:?}: apart by up to {worst}");
            }
        }
        let _ = device;
    }
}
