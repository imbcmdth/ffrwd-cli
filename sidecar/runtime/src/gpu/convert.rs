//! The one pass the host runs on a decoder's picture: its NV12 planes
//! repacked as the wire's three yuv420p planes, on the device.
//!
//! A module reads a frame's planes as the wire lays them out (`ffrwd:gpu`
//! hands yuv420p over as three `r8unorm` textures), and a decoder writes
//! NV12, whose chroma is one interleaved plane. This copies the luma plane
//! and splits the chroma plane, texel for texel, in two render passes;
//! nothing is filtered or converted, so the bytes are the decoder's.

use std::borrow::Cow;
use std::sync::OnceLock;

use anyhow::{anyhow, bail, Result};
use wasi_webgpu_wasmtime::reexports::{wgpu_core, wgpu_types};
use wasi_webgpu_wasmtime::SharedDevice;
use wgpu_core::global::Global;
use wgpu_core::id::{BindGroupLayoutId, RenderPipelineId, TextureViewId};
use wgpu_types::TextureFormat;

use super::frames::{create_plane, Plane};

const SHADER: &str = r#"
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

/// The two pipelines, made once a process on its one device.
struct Pipelines {
    layout: BindGroupLayoutId,
    luma: RenderPipelineId,
    chroma: RenderPipelineId,
}

fn pipelines(global: &Global, device: &SharedDevice) -> Result<&'static Pipelines> {
    static MADE: OnceLock<std::result::Result<Pipelines, String>> = OnceLock::new();
    MADE.get_or_init(|| make_pipelines(global, device).map_err(|e| format!("{e:#}")))
        .as_ref()
        .map_err(|e| anyhow!("the NV12 repacking pipelines: {e}"))
}

fn make_pipelines(global: &Global, device: &SharedDevice) -> Result<Pipelines> {
    let (module, err) = global.device_create_shader_module(
        device.device,
        &wgpu_core::pipeline::ShaderModuleDescriptor {
            label: Some(Cow::Borrowed("ffrwd nv12 repack")),
            runtime_checks: wgpu_types::ShaderRuntimeChecks::checked(),
        },
        wgpu_core::pipeline::ShaderModuleSource::Wgsl(Cow::Borrowed(SHADER)),
        None,
    );
    if let Some(e) = err {
        bail!("compiling the shader: {e}");
    }
    let (layout, err) = global.device_create_bind_group_layout(
        device.device,
        &wgpu_core::binding_model::BindGroupLayoutDescriptor {
            label: None,
            entries: Cow::Owned(vec![wgpu_types::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu_types::ShaderStages::FRAGMENT,
                ty: wgpu_types::BindingType::Texture {
                    sample_type: wgpu_types::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu_types::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }]),
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
    let target = Some(wgpu_types::ColorTargetState {
        format: TextureFormat::R8Unorm,
        blend: None,
        write_mask: wgpu_types::ColorWrites::ALL,
    });
    let pipeline = |entry: &'static str, targets: Vec<Option<wgpu_types::ColorTargetState>>| {
        let stage = |name: &'static str| wgpu_core::pipeline::ProgrammableStageDescriptor {
            module,
            entry_point: Some(Cow::Borrowed(name)),
            constants: Default::default(),
            zero_initialize_workgroup_memory: true,
        };
        let (id, err) = global.device_create_render_pipeline(
            device.device,
            &wgpu_core::pipeline::RenderPipelineDescriptor {
                label: Some(Cow::Borrowed(entry)),
                layout: Some(pipeline_layout),
                vertex: wgpu_core::pipeline::VertexState {
                    stage: stage("whole"),
                    buffers: Cow::Borrowed(&[]),
                },
                primitive: wgpu_types::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu_types::MultisampleState::default(),
                fragment: Some(wgpu_core::pipeline::FragmentState {
                    stage: stage(entry),
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
    };
    let luma = pipeline("luma", vec![target.clone()])?;
    let chroma = pipeline("chroma", vec![target.clone(), target])?;
    Ok(Pipelines {
        layout,
        luma,
        chroma,
    })
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
    let mut made: Vec<Plane> = Vec::with_capacity(3);
    let mut views: Vec<TextureViewId> = Vec::new();
    let mut groups = Vec::new();
    let result = (|| -> Result<()> {
        for (w, h) in [(width, height), (cw, ch), (cw, ch)] {
            made.push(create_plane(
                global,
                device,
                w,
                h,
                TextureFormat::R8Unorm,
                1,
            )?);
        }
        let mut sources = Vec::with_capacity(2);
        for plane in [y, uv] {
            let v = view(global, plane)?;
            views.push(v);
            sources.push(v);
        }
        let mut targets = Vec::with_capacity(3);
        for plane in &made {
            let v = view(global, plane)?;
            views.push(v);
            targets.push(v);
        }
        for source in &sources {
            let (group, err) = global.device_create_bind_group(
                device.device,
                &wgpu_core::binding_model::BindGroupDescriptor {
                    label: None,
                    layout: pipes.layout,
                    entries: Cow::Owned(vec![wgpu_core::binding_model::BindGroupEntry {
                        binding: 0,
                        resource: wgpu_core::binding_model::BindingResource::TextureView(*source),
                    }]),
                },
                None,
            );
            groups.push(group);
            if let Some(e) = err {
                bail!("a bind group over a plane: {e}");
            }
        }

        let (encoder, err) = global.device_create_command_encoder(
            device.device,
            &wgpu_types::CommandEncoderDescriptor { label: None },
            None,
        );
        if let Some(e) = err {
            bail!("a command encoder for the NV12 repacking: {e}");
        }
        let attachment = |view: TextureViewId| {
            Some(wgpu_core::command::RenderPassColorAttachment {
                view,
                resolve_target: None,
                load_op: wgpu_core::command::LoadOp::Clear(wgpu_types::Color::BLACK),
                store_op: wgpu_core::command::StoreOp::Store,
                depth_slice: None,
            })
        };
        let passes = [
            (pipes.luma, groups[0], vec![attachment(targets[0])]),
            (
                pipes.chroma,
                groups[1],
                vec![attachment(targets[1]), attachment(targets[2])],
            ),
        ];
        for (pipeline, group, attachments) in passes {
            let (mut pass, err) = global.command_encoder_begin_render_pass(
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
                bail!("beginning the NV12 repacking: {e}");
            }
            global
                .render_pass_set_pipeline(&mut pass, pipeline)
                .map_err(|e| anyhow!("{e}"))?;
            global
                .render_pass_set_bind_group(&mut pass, 0, Some(group), &[])
                .map_err(|e| anyhow!("{e}"))?;
            global
                .render_pass_draw(&mut pass, 3, 1, 0, 0)
                .map_err(|e| anyhow!("{e}"))?;
            global
                .render_pass_end(&mut pass)
                .map_err(|e| anyhow!("ending the NV12 repacking: {e}"))?;
        }
        let (commands, err) = global.command_encoder_finish(
            encoder,
            &wgpu_types::CommandBufferDescriptor { label: None },
            None,
        );
        if let Some((_, e)) = err {
            bail!("finishing the NV12 repacking: {e}");
        }
        global
            .queue_submit(*device.queue, &[commands])
            .map_err(|(_, e)| anyhow!("submitting the NV12 repacking: {e}"))?;
        Ok(())
    })();
    for group in groups {
        global.bind_group_drop(group);
    }
    for v in views {
        global.texture_view_drop(v);
    }
    match result {
        Ok(()) => Ok(made),
        Err(e) => {
            for plane in &made {
                global.texture_drop(plane.texture);
            }
            Err(e)
        }
    }
}
