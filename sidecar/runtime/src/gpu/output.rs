//! A model's output held as a texture: an `r32float` texture of the
//! process's device, one fp32 element a texel, row by row.
//!
//! Where the session writes device memory, it writes a buffer of the
//! device (on a shared heap, which CUDA imports, or one DirectML reads as
//! its own), and a compute pass copies the buffer into the texture: the
//! input tensor's pass run the other way. Where it does not, the elements
//! come back to host memory and are uploaded into the texture once.

use std::borrow::Cow;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, bail, Result};
use wasi_webgpu_wasmtime::reexports::{wgpu_core, wgpu_types};
use wasi_webgpu_wasmtime::SharedDevice;
use wgpu_core::global::Global;
use wgpu_core::id::{BindGroupId, BindGroupLayoutId, BufferId, ComputePipelineId, TextureId};
use wgpu_types::{BufferUsages, Extent3d, TextureFormat, TextureUsages};

use super::commands::Commands;
use super::tensor::{pooled_buffer, read_buffer, PooledBuffer};
use super::{shared, shared_device};
use crate::profile::PROFILE;

/// The pass: the buffer's elements into the texture, one a texel.
const PASS: &str = r#"
struct Size {
    w: u32,
    h: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> size: Size;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= size.w || id.y >= size.h) {
        return;
    }
    let v = src[id.y * size.w + id.x];
    textureStore(dst, vec2<i32>(i32(id.x), i32(id.y)), vec4<f32>(v, 0.0, 0.0, 1.0));
}
"#;

/// The texture's format.
pub const FORMAT: TextureFormat = TextureFormat::R32Float;

/// What the module may do with the texture.
pub const MODULE_USAGE: TextureUsages =
    TextureUsages::TEXTURE_BINDING.union(TextureUsages::COPY_SRC);

struct Pipeline {
    layout: BindGroupLayoutId,
    pipeline: ComputePipelineId,
}

fn pipeline(global: &Global, device: &SharedDevice) -> Result<&'static Pipeline> {
    static MADE: std::sync::OnceLock<std::result::Result<Pipeline, String>> =
        std::sync::OnceLock::new();
    MADE.get_or_init(|| make_pipeline(global, device).map_err(|e| format!("{e:#}")))
        .as_ref()
        .map_err(|e| anyhow!("the output pipeline: {e}"))
}

fn make_pipeline(global: &Global, device: &SharedDevice) -> Result<Pipeline> {
    let (module, err) = global.device_create_shader_module(
        device.device,
        &wgpu_core::pipeline::ShaderModuleDescriptor {
            label: Some(Cow::Borrowed("ffrwd output texture")),
            runtime_checks: wgpu_types::ShaderRuntimeChecks::checked(),
        },
        wgpu_core::pipeline::ShaderModuleSource::Wgsl(Cow::Borrowed(PASS)),
        None,
    );
    if let Some(e) = err {
        bail!("compiling the shader: {e}");
    }
    let entries = vec![
        wgpu_types::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu_types::ShaderStages::COMPUTE,
            ty: wgpu_types::BindingType::Buffer {
                ty: wgpu_types::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu_types::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu_types::ShaderStages::COMPUTE,
            ty: wgpu_types::BindingType::StorageTexture {
                access: wgpu_types::StorageTextureAccess::WriteOnly,
                format: FORMAT,
                view_dimension: wgpu_types::TextureViewDimension::D2,
            },
            count: None,
        },
        wgpu_types::BindGroupLayoutEntry {
            binding: 2,
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
            label: Some(Cow::Borrowed("ffrwd output texture")),
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

/// The pass's fixed parts for one texture and its buffer.
struct Copy {
    group: BindGroupId,
    view: wgpu_core::id::TextureViewId,
    uniform: BufferId,
}

/// One output's texture, and the device buffer a session writes it
/// through where it writes device memory.
pub struct OutputTexture {
    global: Arc<Global>,
    device: Arc<SharedDevice>,
    pub texture: TextureId,
    pub width: u32,
    pub height: u32,
    /// The buffer the session writes, on a shared heap.
    pub buffer: Option<Arc<PooledBuffer>>,
    copy: Option<Copy>,
}

// SAFETY: wgpu-core ids and the device are used from any thread; the
// buffer's handles are free-threaded.
unsafe impl Send for OutputTexture {}
unsafe impl Sync for OutputTexture {}

impl OutputTexture {
    /// A `width` x `height` texture, with a buffer of the device for a
    /// session to write where `buffer` asks.
    pub fn new(width: u32, height: u32, buffer: bool) -> Result<OutputTexture> {
        if width == 0 || height == 0 {
            bail!("a {width}x{height} texture has no texels");
        }
        let device = shared_device().ok_or_else(|| anyhow!("this machine has no GPU adapter"))?;
        let global = Arc::clone(shared());
        let desc = wgpu_types::TextureDescriptor {
            label: Some(Cow::Borrowed("ffrwd output")),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu_types::TextureDimension::D2,
            format: FORMAT,
            usage: MODULE_USAGE | TextureUsages::STORAGE_BINDING | TextureUsages::COPY_DST,
            view_formats: Vec::new(),
        };
        let (texture, err) = global.device_create_texture(device.device, &desc, None);
        if let Some(e) = err {
            global.texture_drop(texture);
            bail!("a {width}x{height} {FORMAT:?} texture: {e}");
        }
        let mut made = OutputTexture {
            global,
            device: Arc::clone(device),
            texture,
            width,
            height,
            buffer: None,
            copy: None,
        };
        if buffer {
            let size = 4 * u64::from(width) * u64::from(height);
            let pooled = pooled_buffer(&made.global, &made.device, size, cfg!(windows))?;
            made.copy = Some(made.copy_parts(pooled.id, size)?);
            made.buffer = Some(pooled);
        }
        Ok(made)
    }

    fn copy_parts(&self, buffer: BufferId, size: u64) -> Result<Copy> {
        let (global, device) = (&self.global, &self.device);
        let pipes = pipeline(global, device)?;
        let (view, err) = global.texture_create_view(
            self.texture,
            &wgpu_core::resource::TextureViewDescriptor {
                label: None,
                format: Some(FORMAT),
                dimension: Some(wgpu_types::TextureViewDimension::D2),
                usage: None,
                range: wgpu_types::ImageSubresourceRange::default(),
            },
            None,
        );
        if let Some(e) = err {
            global.texture_view_drop(view);
            bail!("a view of the output texture: {e}");
        }
        let (uniform, err) = global.device_create_buffer(
            device.device,
            &wgpu_types::BufferDescriptor {
                label: Some(Cow::Borrowed("ffrwd output size")),
                size: 16,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
            None,
        );
        let fail = |what: String| {
            global.texture_view_drop(view);
            global.buffer_drop(uniform);
            anyhow!(what)
        };
        if let Some(e) = err {
            return Err(fail(format!("the output pass's size: {e}")));
        }
        let mut words = [0u8; 16];
        words[..4].copy_from_slice(&self.width.to_le_bytes());
        words[4..8].copy_from_slice(&self.height.to_le_bytes());
        if let Err(e) = global.queue_write_buffer(*device.queue, uniform, 0, &words) {
            return Err(fail(format!("writing the output pass's size: {e}")));
        }
        let entries = vec![
            wgpu_core::binding_model::BindGroupEntry {
                binding: 0,
                resource: wgpu_core::binding_model::BindingResource::Buffer(
                    wgpu_core::binding_model::BufferBinding {
                        buffer,
                        offset: 0,
                        size: Some(size),
                    },
                ),
            },
            wgpu_core::binding_model::BindGroupEntry {
                binding: 1,
                resource: wgpu_core::binding_model::BindingResource::TextureView(view),
            },
            wgpu_core::binding_model::BindGroupEntry {
                binding: 2,
                resource: wgpu_core::binding_model::BindingResource::Buffer(
                    wgpu_core::binding_model::BufferBinding {
                        buffer: uniform,
                        offset: 0,
                        size: None,
                    },
                ),
            },
        ];
        let (group, err) = global.device_create_bind_group(
            device.device,
            &wgpu_core::binding_model::BindGroupDescriptor {
                label: None,
                layout: pipes.layout,
                entries: Cow::Owned(entries),
            },
            None,
        );
        if let Some(e) = err {
            global.bind_group_drop(group);
            return Err(fail(format!("the output pass's bind group: {e}")));
        }
        Ok(Copy {
            group,
            view,
            uniform,
        })
    }

    /// Gives the queue the pass that copies the buffer into the texture.
    pub fn copy_in(&self) -> Result<()> {
        let (global, device) = (&self.global, &self.device);
        let copy = self
            .copy
            .as_ref()
            .ok_or_else(|| anyhow!("the output has no buffer to copy from"))?;
        let pipes = pipeline(global, device)?;
        let commands = Commands::open(global, device, "the output pass")?;
        let encoder = commands.encoder();
        let (mut pass, err) = global.command_encoder_begin_compute_pass(
            encoder,
            &wgpu_core::command::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            },
        );
        if let Some(e) = err {
            bail!("beginning the output pass: {e}");
        }
        global
            .compute_pass_set_pipeline(&mut pass, pipes.pipeline)
            .map_err(|e| anyhow!("{e}"))?;
        global
            .compute_pass_set_bind_group(&mut pass, 0, Some(copy.group), &[])
            .map_err(|e| anyhow!("{e}"))?;
        global
            .compute_pass_dispatch_workgroups(
                &mut pass,
                self.width.div_ceil(8),
                self.height.div_ceil(8),
                1,
            )
            .map_err(|e| anyhow!("{e}"))?;
        global
            .compute_pass_end(&mut pass)
            .map_err(|e| anyhow!("ending the output pass: {e}"))?;
        commands.submit(*device.queue)?;
        Ok(())
    }

    /// Writes `bytes`, the elements in host memory, into the texture.
    pub fn upload(&self, bytes: &[u8]) -> Result<()> {
        let len = 4 * self.width as usize * self.height as usize;
        if bytes.len() != len {
            bail!(
                "the output has {} bytes, and a {}x{} texture takes {len}",
                bytes.len(),
                self.width,
                self.height
            );
        }
        let started = Instant::now();
        self.global
            .queue_write_texture(
                *self.device.queue,
                &wgpu_types::TexelCopyTextureInfo {
                    texture: self.texture,
                    mip_level: 0,
                    origin: wgpu_types::Origin3d::ZERO,
                    aspect: wgpu_types::TextureAspect::All,
                },
                bytes,
                &wgpu_types::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * self.width),
                    rows_per_image: Some(self.height),
                },
                &Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: 1,
                },
            )
            .map_err(|e| anyhow!("uploading the output: {e}"))?;
        PROFILE.upload.add(len, started);
        Ok(())
    }

    /// The buffer's elements in host memory, read back through the device.
    pub fn read_buffer(&self) -> Result<Vec<u8>> {
        let buffer = self
            .buffer
            .as_ref()
            .ok_or_else(|| anyhow!("the output has no buffer to read"))?;
        let started = Instant::now();
        let len = 4 * u64::from(self.width) * u64::from(self.height);
        let bytes = read_buffer(&self.global, &self.device, buffer.id, len)?;
        PROFILE.download.add(bytes.len(), started);
        Ok(bytes)
    }
}

impl Drop for OutputTexture {
    fn drop(&mut self) {
        if let Some(copy) = self.copy.take() {
            self.global.bind_group_drop(copy.group);
            self.global.texture_view_drop(copy.view);
            self.global.buffer_drop(copy.uniform);
        }
        self.global.texture_drop(self.texture);
    }
}
