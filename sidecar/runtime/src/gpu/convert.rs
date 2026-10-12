//! The passes the host runs on a frame's planes on the device.
//!
//! A decoder writes NV12 (luma, then chroma interleaved at half size), and
//! a module reads a frame's planes as the wire lays them out (`ffrwd:gpu`
//! hands yuv420p over as three `r8unorm` textures): [`yuv420p_from_nv12`]
//! copies the luma plane and splits the chroma plane, texel for texel, so
//! the bytes are the decoder's.
//!
//! An encoder on the device reads NV12 and nothing else. [`nv12_from_rgba`]
//! converts a frame a module rendered with the matrix and range [`Yuv`]
//! names: luma at full size, and chroma at half size the way ffmpeg's
//! scale filter makes it by default, each pair of pixels averaged across
//! and a bicubic filter of eight rows down the column, so the result is
//! ffmpeg's conversion to within a level or two. The rgba is
//! premultiplied, as a module renders it, and is read as it is, so a
//! transparent pixel is black. [`nv12_from_yuv420p`] interleaves the
//! chroma of a yuv420p frame, texel for texel.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use anyhow::{anyhow, bail, Result};
use wasi_webgpu_wasmtime::reexports::{wgpu_core, wgpu_types};
use wasi_webgpu_wasmtime::SharedDevice;
use wgpu_core::global::Global;
use wgpu_core::id::{BindGroupId, TextureViewId};
use wgpu_core::id::{BindGroupLayoutId, PipelineLayoutId, RenderPipelineId, ShaderModuleId};
use wgpu_types::TextureFormat;

use super::frames::{create_plane, Plane};

const SPLIT: &str = r#"
@group(0) @binding(0) var source: texture_2d<f32>;

@vertex
fn whole(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(i & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn luma(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(textureLoad(source, vec2<i32>(at.xy), 0).r, 0.0, 0.0, 1.0);
}

struct Split {
    @location(0) u: vec4<f32>,
    @location(1) v: vec4<f32>,
}

@fragment
fn chroma(@builtin(position) at: vec4<f32>) -> Split {
    let uv = textureLoad(source, vec2<i32>(at.xy), 0);
    return Split(vec4<f32>(uv.r, 0.0, 0.0, 1.0), vec4<f32>(uv.g, 0.0, 0.0, 1.0));
}
"#;

/// rgba to NV12, and yuv420p's chroma interleaved. `kr` and `kb` are the
/// matrix's red and blue weights; `full` is 1 for full range, 0 for
/// limited (luma 16 to 235, chroma 16 to 240).
const PACK: &str = r#"
override kr: f32 = 0.2126;
override kb: f32 = 0.0722;
override full: f32 = 0.0;

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var second: texture_2d<f32>;

@vertex
fn whole(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(i & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

fn luma_of(c: vec3<f32>) -> f32 {
    return kr * c.r + (1.0 - kr - kb) * c.g + kb * c.b;
}

fn coded_luma(y: f32) -> f32 {
    return select((16.0 + 219.0 * y) / 255.0, y, full > 0.5);
}

fn coded_chroma(c: f32) -> f32 {
    return select((128.0 + 224.0 * c) / 255.0, (128.0 + 255.0 * c) / 255.0, full > 0.5);
}

@fragment
fn rgba_luma(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
    let c = textureLoad(source, vec2<i32>(at.xy), 0).rgb;
    return vec4<f32>(coded_luma(luma_of(c)), 0.0, 0.0, 1.0);
}

// The pair of pixels at `x` and `x + 1` on row `y`, averaged, the edges
// repeated past the picture's.
fn pair(x: i32, y: i32, last: vec2<i32>) -> vec3<f32> {
    let row = clamp(y, 0, last.y);
    let a = textureLoad(source, vec2<i32>(min(x, last.x), row), 0).rgb;
    let b = textureLoad(source, vec2<i32>(min(x + 1, last.x), row), 0).rgb;
    return (a + b) * 0.5;
}

// A bicubic filter (a = -0.6) halving a column, from the centre between
// two rows outwards: what ffmpeg's scale filter applies by default.
const TAPS = array<f32, 4>(0.4359375, 0.1203125, -0.0421875, -0.0140625);

@fragment
fn rgba_chroma(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
    let last = vec2<i32>(textureDimensions(source)) - vec2<i32>(1, 1);
    let p = vec2<i32>(at.xy) * 2;
    var c = vec3<f32>(0.0);
    for (var i = 0; i < 4; i++) {
        c += TAPS[i] * (pair(p.x, p.y - i, last) + pair(p.x, p.y + 1 + i, last));
    }
    let y = luma_of(c);
    let u = (c.b - y) / (2.0 * (1.0 - kb));
    let v = (c.r - y) / (2.0 * (1.0 - kr));
    return vec4<f32>(coded_chroma(u), coded_chroma(v), 0.0, 1.0);
}

@fragment
fn interleave(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
    let xy = vec2<i32>(at.xy);
    return vec4<f32>(textureLoad(source, xy, 0).r, textureLoad(second, xy, 0).r, 0.0, 1.0);
}
"#;

/// The matrix an rgba frame is converted to YUV with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Matrix {
    /// BT.709: what HD and every newer stream uses.
    Bt709,
    /// BT.601 (`bt470bg`, `smpte170m`): standard definition.
    Bt601,
}

/// The YUV an rgba frame becomes ahead of an encoder: a matrix and a range.
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

    /// The matrix's red and blue weights.
    fn weights(self) -> (f64, f64) {
        match self.matrix {
            Matrix::Bt709 => (0.2126, 0.0722),
            Matrix::Bt601 => (0.299, 0.114),
        }
    }
}

/// The packing shader, and the layouts of its one- and two-texture
/// passes.
struct Pack {
    module: ShaderModuleId,
    one: (BindGroupLayoutId, PipelineLayoutId),
    two: (BindGroupLayoutId, PipelineLayoutId),
}

/// The pipelines, made once a process on its one device.
struct Pipelines {
    split_layout: BindGroupLayoutId,
    luma: RenderPipelineId,
    chroma: RenderPipelineId,
    pack: Pack,
    interleave: RenderPipelineId,
    /// rgba's luma and chroma pipelines, one pair per [`Yuv`] asked for.
    rgba: Mutex<HashMap<Yuv, (RenderPipelineId, RenderPipelineId)>>,
}

fn pipelines(global: &Global, device: &SharedDevice) -> Result<&'static Pipelines> {
    static MADE: OnceLock<std::result::Result<Pipelines, String>> = OnceLock::new();
    MADE.get_or_init(|| make_pipelines(global, device).map_err(|e| format!("{e:#}")))
        .as_ref()
        .map_err(|e| anyhow!("the frame conversion pipelines: {e}"))
}

fn shader(
    global: &Global,
    device: &SharedDevice,
    label: &'static str,
    source: &'static str,
) -> Result<ShaderModuleId> {
    let (module, err) = global.device_create_shader_module(
        device.device,
        &wgpu_core::pipeline::ShaderModuleDescriptor {
            label: Some(Cow::Borrowed(label)),
            runtime_checks: wgpu_types::ShaderRuntimeChecks::checked(),
        },
        wgpu_core::pipeline::ShaderModuleSource::Wgsl(Cow::Borrowed(source)),
        None,
    );
    if let Some(e) = err {
        bail!("compiling the {label} shader: {e}");
    }
    Ok(module)
}

/// A bind group layout of `count` sampled textures at bindings 0 on, and
/// a pipeline layout over it.
fn layouts(
    global: &Global,
    device: &SharedDevice,
    count: u32,
) -> Result<(BindGroupLayoutId, PipelineLayoutId)> {
    let entries = (0..count)
        .map(|binding| wgpu_types::BindGroupLayoutEntry {
            binding,
            visibility: wgpu_types::ShaderStages::FRAGMENT,
            ty: wgpu_types::BindingType::Texture {
                sample_type: wgpu_types::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu_types::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        })
        .collect();
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
    Ok((layout, pipeline_layout))
}

/// A pipeline drawing one triangle over the targets with `entry`, the
/// overridable constants set to `constants`.
fn pipeline(
    global: &Global,
    device: &SharedDevice,
    module: ShaderModuleId,
    layout: PipelineLayoutId,
    entry: &'static str,
    targets: &[TextureFormat],
    constants: &[(&str, f64)],
) -> Result<RenderPipelineId> {
    let stage = |name: &'static str, constants: &[(&str, f64)]| {
        let mut set = wgpu_core::pipeline::ProgrammableStageDescriptor {
            module,
            entry_point: Some(Cow::Borrowed(name)),
            constants: Default::default(),
            zero_initialize_workgroup_memory: true,
        };
        for (key, value) in constants {
            set.constants.insert((*key).to_string(), *value);
        }
        set
    };
    let targets = targets
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
            layout: Some(layout),
            vertex: wgpu_core::pipeline::VertexState {
                stage: stage("whole", &[]),
                buffers: Cow::Borrowed(&[]),
            },
            primitive: wgpu_types::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu_types::MultisampleState::default(),
            fragment: Some(wgpu_core::pipeline::FragmentState {
                stage: stage(entry, constants),
                targets: Cow::Owned(targets),
            }),
            multiview_mask: None,
            cache: None,
        },
        None,
    );
    match err {
        Some(e) => Err(anyhow!("the {entry} pipeline: {e}")),
        None => Ok(id),
    }
}

fn make_pipelines(global: &Global, device: &SharedDevice) -> Result<Pipelines> {
    let r8 = TextureFormat::R8Unorm;
    let split = shader(global, device, "ffrwd nv12 repack", SPLIT)?;
    let (split_layout, split_pipeline_layout) = layouts(global, device, 1)?;
    let luma = pipeline(
        global,
        device,
        split,
        split_pipeline_layout,
        "luma",
        &[r8],
        &[],
    )?;
    let chroma = pipeline(
        global,
        device,
        split,
        split_pipeline_layout,
        "chroma",
        &[r8, r8],
        &[],
    )?;
    let pack = Pack {
        module: shader(global, device, "ffrwd nv12 pack", PACK)?,
        one: layouts(global, device, 1)?,
        two: layouts(global, device, 2)?,
    };
    let interleave = pipeline(
        global,
        device,
        pack.module,
        pack.two.1,
        "interleave",
        &[TextureFormat::Rg8Unorm],
        &[],
    )?;
    Ok(Pipelines {
        split_layout,
        luma,
        chroma,
        pack,
        interleave,
        rgba: Mutex::new(HashMap::new()),
    })
}

impl Pipelines {
    /// rgba's luma and chroma pipelines for `yuv`, made the first time it
    /// is asked for.
    fn rgba(
        &self,
        global: &Global,
        device: &SharedDevice,
        yuv: Yuv,
    ) -> Result<(RenderPipelineId, RenderPipelineId)> {
        let mut made = self.rgba.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(pair) = made.get(&yuv) {
            return Ok(*pair);
        }
        let (kr, kb) = yuv.weights();
        let full = if yuv.full_range { 1.0 } else { 0.0 };
        let constants = [("kr", kr), ("kb", kb), ("full", full)];
        let (module, layout) = (self.pack.module, self.pack.one.1);
        let luma = pipeline(
            global,
            device,
            module,
            layout,
            "rgba_luma",
            &[TextureFormat::R8Unorm],
            &constants,
        )?;
        let chroma = pipeline(
            global,
            device,
            module,
            layout,
            "rgba_chroma",
            &[TextureFormat::Rg8Unorm],
            &constants,
        )?;
        made.insert(yuv, (luma, chroma));
        Ok((luma, chroma))
    }
}

/// A view of one plane as its own format, the aspect it is in.
fn view(global: &Global, plane: &Plane) -> Result<TextureViewId> {
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

/// One render pass: `pipeline` reading `sources` through a bind group of
/// `layout`, writing `targets` (indexes into the planes made).
struct Pass<'a> {
    pipeline: RenderPipelineId,
    layout: BindGroupLayoutId,
    sources: Vec<&'a Plane>,
    targets: Vec<usize>,
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
            let mut entries = Vec::with_capacity(pass.sources.len());
            for (binding, source) in pass.sources.iter().enumerate() {
                let v = view(global, source)?;
                views.push(v);
                entries.push(wgpu_core::binding_model::BindGroupEntry {
                    binding: binding as u32,
                    resource: wgpu_core::binding_model::BindingResource::TextureView(v),
                });
            }
            let (group, err) = global.device_create_bind_group(
                device.device,
                &wgpu_core::binding_model::BindGroupDescriptor {
                    label: None,
                    layout: pass.layout,
                    entries: Cow::Owned(entries),
                },
                None,
            );
            groups.push(group);
            if let Some(e) = err {
                bail!("a bind group over the planes: {e}");
            }
            recorded.push(group);
        }

        let (encoder, err) = global.device_create_command_encoder(
            device.device,
            &wgpu_types::CommandEncoderDescriptor { label: None },
            None,
        );
        if let Some(e) = err {
            bail!("a command encoder for {what}: {e}");
        }
        for (pass, group) in passes.iter().zip(recorded) {
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
                .render_pass_set_pipeline(&mut rpass, pass.pipeline)
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
                global.texture_drop(plane.texture);
            }
            Err(e)
        }
    }
}

/// The yuv420p planes of a `width` x `height` NV12 picture whose luma is
/// `y` (one channel) and chroma `uv` (two), as three new `r8unorm`
/// textures the caller owns, written by commands submitted here.
pub(super) fn yuv420p_from_nv12(
    global: &Global,
    device: &SharedDevice,
    y: &Plane,
    uv: &Plane,
    width: u32,
    height: u32,
) -> Result<Vec<Plane>> {
    let pipes = pipelines(global, device)?;
    let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
    let r8 = TextureFormat::R8Unorm;
    render(
        global,
        device,
        "the NV12 repacking",
        &[(width, height, r8, 1), (cw, ch, r8, 1), (cw, ch, r8, 1)],
        vec![
            Pass {
                pipeline: pipes.luma,
                layout: pipes.split_layout,
                sources: vec![y],
                targets: vec![0],
            },
            Pass {
                pipeline: pipes.chroma,
                layout: pipes.split_layout,
                sources: vec![uv],
                targets: vec![1, 2],
            },
        ],
    )
}

/// The NV12 planes of a `width` x `height` rgba frame, converted with
/// `yuv`: a new `r8unorm` luma texture and a new `rg8unorm` chroma one at
/// half size, which the caller owns, written by commands submitted here.
pub(super) fn nv12_from_rgba(
    global: &Global,
    device: &SharedDevice,
    rgba: &Plane,
    yuv: Yuv,
) -> Result<Vec<Plane>> {
    let pipes = pipelines(global, device)?;
    let (luma, chroma) = pipes.rgba(global, device, yuv)?;
    let (width, height) = (rgba.width, rgba.height);
    let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
    let layout = pipes.pack.one.0;
    render(
        global,
        device,
        "the rgba to NV12 conversion",
        &[
            (width, height, TextureFormat::R8Unorm, 1),
            (cw, ch, TextureFormat::Rg8Unorm, 2),
        ],
        vec![
            Pass {
                pipeline: luma,
                layout,
                sources: vec![rgba],
                targets: vec![0],
            },
            Pass {
                pipeline: chroma,
                layout,
                sources: vec![rgba],
                targets: vec![1],
            },
        ],
    )
}

/// The NV12 planes of a yuv420p frame's three: its luma copied and its
/// chroma interleaved, into new textures the caller owns.
pub(super) fn nv12_from_yuv420p(
    global: &Global,
    device: &SharedDevice,
    planes: &[Plane],
) -> Result<Vec<Plane>> {
    let [y, u, v] = planes else {
        bail!(
            "a yuv420p frame has three planes, and this one has {}",
            planes.len()
        );
    };
    let pipes = pipelines(global, device)?;
    render(
        global,
        device,
        "the yuv420p to NV12 interleave",
        &[
            (y.width, y.height, TextureFormat::R8Unorm, 1),
            (u.width, u.height, TextureFormat::Rg8Unorm, 2),
        ],
        vec![
            Pass {
                pipeline: pipes.luma,
                layout: pipes.split_layout,
                sources: vec![y],
                targets: vec![0],
            },
            Pass {
                pipeline: pipes.interleave,
                layout: pipes.pack.two.0,
                sources: vec![u, v],
                targets: vec![1],
            },
        ],
    )
}
