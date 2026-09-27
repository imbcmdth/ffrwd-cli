use core::slice;
use std::{borrow::Cow, collections::HashMap, num::NonZeroU64, sync::Arc};

use wasmtime::component::Resource;
use wasmtime::error::bail;

use crate::{
    graphics_context::{AbstractBuffer, GraphicsContext},
    to_core_conversions::ToCore,
    wasi::{io::poll, webgpu::webgpu},
    wrapper_types::{
        Buffer, CommandEncoder, ComputePassEncoder, ComputePipeline, Device, ErrorHandler,
        Listener, RenderBundleEncoder, WgslLanguageFeatures, RenderBundleEncoderInner, RenderPassEncoder,
        RenderPipeline, Texture,
    },
    WasiWebGpuCtx, PREFERRED_CANVAS_FORMAT,
};

impl webgpu::Host for WasiWebGpuCtx<'_> {
    fn get_gpu(&mut self) -> wasmtime::Result<Resource<webgpu::Gpu>> {
        Ok(Resource::new_own(0))
    }
}

impl webgpu::HostGpuColorWrite for WasiWebGpuCtx<'_> {
    fn red(&mut self) -> wasmtime::Result<webgpu::GpuFlagsConstant> {
        Ok(wgpu_types::ColorWrites::RED.bits())
    }

    fn green(&mut self) -> wasmtime::Result<webgpu::GpuFlagsConstant> {
        Ok(wgpu_types::ColorWrites::GREEN.bits())
    }

    fn blue(&mut self) -> wasmtime::Result<webgpu::GpuFlagsConstant> {
        Ok(wgpu_types::ColorWrites::BLUE.bits())
    }

    fn alpha(&mut self) -> wasmtime::Result<webgpu::GpuFlagsConstant> {
        Ok(wgpu_types::ColorWrites::ALPHA.bits())
    }

    fn all(&mut self) -> wasmtime::Result<webgpu::GpuFlagsConstant> {
        Ok(wgpu_types::ColorWrites::ALL.bits())
    }

    fn drop(&mut self, _self_: Resource<webgpu::GpuColorWrite>) -> wasmtime::Result<()> {
        unreachable!()
    }
}

impl webgpu::HostRecordGpuPipelineConstantValue for WasiWebGpuCtx<'_> {
    fn new(&mut self) -> wasmtime::Result<Resource<webgpu::RecordGpuPipelineConstantValue>> {
        Ok(self.table.push(HashMap::new())?)
    }

    fn add(
        &mut self,
        record: Resource<webgpu::RecordGpuPipelineConstantValue>,
        key: String,
        value: webgpu::GpuPipelineConstantValue,
    ) -> wasmtime::Result<()> {
        let record = self.table.get_mut(&record)?;
        record.insert(key, value);
        Ok(())
    }

    fn get(
        &mut self,
        record: Resource<webgpu::RecordGpuPipelineConstantValue>,
        key: String,
    ) -> wasmtime::Result<Option<webgpu::GpuPipelineConstantValue>> {
        let record = self.table.get(&record)?;
        let value = record.get(&key).copied();
        Ok(value)
    }

    fn has(
        &mut self,
        record: Resource<webgpu::RecordGpuPipelineConstantValue>,
        key: String,
    ) -> wasmtime::Result<bool> {
        let record = self.table.get(&record)?;
        Ok(record.contains_key(&key))
    }

    fn remove(
        &mut self,
        record: Resource<webgpu::RecordGpuPipelineConstantValue>,
        key: String,
    ) -> wasmtime::Result<()> {
        let record = self.table.get_mut(&record)?;
        record.remove(&key);
        Ok(())
    }

    fn keys(
        &mut self,
        record: Resource<webgpu::RecordGpuPipelineConstantValue>,
    ) -> wasmtime::Result<Vec<String>> {
        let record = self.table.get(&record)?;
        let keys = record.keys().cloned().collect();
        Ok(keys)
    }

    fn values(
        &mut self,
        record: Resource<webgpu::RecordGpuPipelineConstantValue>,
    ) -> wasmtime::Result<Vec<webgpu::GpuPipelineConstantValue>> {
        let record = self.table.get(&record)?;
        let values = record.values().copied().collect();
        Ok(values)
    }

    fn entries(
        &mut self,
        record: Resource<webgpu::RecordGpuPipelineConstantValue>,
    ) -> wasmtime::Result<Vec<(String, webgpu::GpuPipelineConstantValue)>> {
        let record = self.table.get(&record)?;
        let entries = record.iter().map(|(k, v)| (k.clone(), *v)).collect();
        Ok(entries)
    }

    fn drop(
        &mut self,
        record: Resource<webgpu::RecordGpuPipelineConstantValue>,
    ) -> wasmtime::Result<()> {
        self.table.delete(record)?;
        Ok(())
    }
}

impl webgpu::HostGpuShaderStage for WasiWebGpuCtx<'_> {
    fn vertex(&mut self) -> wasmtime::Result<webgpu::GpuFlagsConstant> {
        Ok(wgpu_types::ShaderStages::VERTEX.bits())
    }

    fn fragment(&mut self) -> wasmtime::Result<webgpu::GpuFlagsConstant> {
        Ok(wgpu_types::ShaderStages::FRAGMENT.bits())
    }

    fn compute(&mut self) -> wasmtime::Result<webgpu::GpuFlagsConstant> {
        Ok(wgpu_types::ShaderStages::COMPUTE.bits())
    }

    fn drop(&mut self, _: Resource<webgpu::GpuShaderStage>) -> wasmtime::Result<()> {
        unreachable!()
    }
}
impl webgpu::HostGpuTextureUsage for WasiWebGpuCtx<'_> {
    fn copy_src(&mut self) -> wasmtime::Result<webgpu::GpuFlagsConstant> {
        Ok(wgpu_types::TextureUsages::COPY_SRC.bits())
    }
    fn copy_dst(&mut self) -> wasmtime::Result<webgpu::GpuFlagsConstant> {
        Ok(wgpu_types::TextureUsages::COPY_DST.bits())
    }
    fn texture_binding(&mut self) -> wasmtime::Result<webgpu::GpuFlagsConstant> {
        Ok(wgpu_types::TextureUsages::TEXTURE_BINDING.bits())
    }
    fn storage_binding(&mut self) -> wasmtime::Result<webgpu::GpuFlagsConstant> {
        Ok(wgpu_types::TextureUsages::STORAGE_BINDING.bits())
    }
    fn render_attachment(&mut self) -> wasmtime::Result<webgpu::GpuFlagsConstant> {
        Ok(wgpu_types::TextureUsages::RENDER_ATTACHMENT.bits())
    }
    fn drop(&mut self, _rep: Resource<webgpu::GpuTextureUsage>) -> wasmtime::Result<()> {
        unreachable!()
    }
}
impl webgpu::HostGpuMapMode for WasiWebGpuCtx<'_> {
    fn read(&mut self) -> wasmtime::Result<webgpu::GpuFlagsConstant> {
        // https://www.w3.org/TR/webgpu/#buffer-mapping
        Ok(0x0001)
    }
    fn write(&mut self) -> wasmtime::Result<webgpu::GpuFlagsConstant> {
        // https://www.w3.org/TR/webgpu/#buffer-mapping
        Ok(0x0002)
    }
    fn drop(&mut self, _rep: Resource<webgpu::GpuMapMode>) -> wasmtime::Result<()> {
        unreachable!()
    }
}
impl webgpu::HostGpuBufferUsage for WasiWebGpuCtx<'_> {
    fn map_read(&mut self) -> wasmtime::Result<webgpu::GpuFlagsConstant> {
        Ok(wgpu_types::BufferUsages::MAP_READ.bits())
    }
    fn map_write(&mut self) -> wasmtime::Result<webgpu::GpuFlagsConstant> {
        Ok(wgpu_types::BufferUsages::MAP_WRITE.bits())
    }
    fn copy_src(&mut self) -> wasmtime::Result<webgpu::GpuFlagsConstant> {
        Ok(wgpu_types::BufferUsages::COPY_SRC.bits())
    }
    fn copy_dst(&mut self) -> wasmtime::Result<webgpu::GpuFlagsConstant> {
        Ok(wgpu_types::BufferUsages::COPY_DST.bits())
    }
    fn index(&mut self) -> wasmtime::Result<webgpu::GpuFlagsConstant> {
        Ok(wgpu_types::BufferUsages::INDEX.bits())
    }
    fn vertex(&mut self) -> wasmtime::Result<webgpu::GpuFlagsConstant> {
        Ok(wgpu_types::BufferUsages::VERTEX.bits())
    }
    fn uniform(&mut self) -> wasmtime::Result<webgpu::GpuFlagsConstant> {
        Ok(wgpu_types::BufferUsages::UNIFORM.bits())
    }
    fn storage(&mut self) -> wasmtime::Result<webgpu::GpuFlagsConstant> {
        Ok(wgpu_types::BufferUsages::STORAGE.bits())
    }
    fn indirect(&mut self) -> wasmtime::Result<webgpu::GpuFlagsConstant> {
        Ok(wgpu_types::BufferUsages::INDIRECT.bits())
    }
    fn query_resolve(&mut self) -> wasmtime::Result<webgpu::GpuFlagsConstant> {
        Ok(wgpu_types::BufferUsages::QUERY_RESOLVE.bits())
    }
    fn drop(&mut self, _rep: Resource<webgpu::GpuBufferUsage>) -> wasmtime::Result<()> {
        unreachable!()
    }
}

impl webgpu::HostRecordOptionGpuSize64 for WasiWebGpuCtx<'_> {
    fn new(&mut self) -> wasmtime::Result<Resource<webgpu::RecordOptionGpuSize64>> {
        let record = std::collections::HashMap::new();
        Ok(self.table.push(record)?)
    }
    fn add(
        &mut self,
        record: Resource<webgpu::RecordOptionGpuSize64>,
        key: String,
        value: Option<webgpu::GpuSize64>,
    ) -> wasmtime::Result<()> {
        let record = self.table.get_mut(&record)?;
        record.insert(key, value);
        Ok(())
    }
    fn get(
        &mut self,
        record: Resource<webgpu::RecordOptionGpuSize64>,
        key: String,
    ) -> wasmtime::Result<Option<Option<webgpu::GpuSize64>>> {
        let record = self.table.get(&record)?;
        Ok(record.get(&key).copied())
    }
    fn has(
        &mut self,
        record: Resource<webgpu::RecordOptionGpuSize64>,
        key: String,
    ) -> wasmtime::Result<bool> {
        let record = self.table.get(&record)?;
        Ok(record.contains_key(&key))
    }
    fn remove(
        &mut self,
        record: Resource<webgpu::RecordOptionGpuSize64>,
        key: String,
    ) -> wasmtime::Result<()> {
        let record = self.table.get_mut(&record)?;
        record.remove(&key);
        Ok(())
    }
    fn keys(
        &mut self,
        record: Resource<webgpu::RecordOptionGpuSize64>,
    ) -> wasmtime::Result<Vec<String>> {
        let record = self.table.get(&record)?;
        Ok(record.keys().cloned().collect())
    }
    fn values(
        &mut self,
        record: Resource<webgpu::RecordOptionGpuSize64>,
    ) -> wasmtime::Result<Vec<Option<webgpu::GpuSize64>>> {
        let record = self.table.get(&record)?;
        Ok(record.values().cloned().collect())
    }
    fn entries(
        &mut self,
        record: Resource<webgpu::RecordOptionGpuSize64>,
    ) -> wasmtime::Result<Vec<(String, Option<webgpu::GpuSize64>)>> {
        let record = self.table.get(&record)?;
        Ok(record.iter().map(|(k, v)| (k.clone(), *v)).collect())
    }
    fn drop(&mut self, record: Resource<webgpu::RecordOptionGpuSize64>) -> wasmtime::Result<()> {
        self.table.delete(record)?;
        Ok(())
    }
}

// impl webgpu::HostNonStandardBuffer for WasiWebGpuCtx<'_> {
//     fn get(&mut self, buffer: Resource<webgpu::NonStandardBuffer>) -> Vec<u8> {
//         let buffer = self.table.get_mut(&buffer)?;
//         buffer.slice_mut().to_vec()
//     }

//     fn set(&mut self, buffer: Resource<webgpu::NonStandardBuffer>, val: Vec<u8>) {
//         let buffer = self.table.get_mut(&buffer)?;
//         buffer.slice_mut().copy_from_slice(&val);
//     }

//     fn drop(&mut self, buffer: Resource<webgpu::NonStandardBuffer>) -> wasmtime::Result<()> {
//         self.table.delete(buffer)?;
//         Ok(())
//     }
// }

impl webgpu::HostGpuDevice for WasiWebGpuCtx<'_> {
    fn connect_graphics_context(
        &mut self,
        _device: Resource<Device>,
        _context: Resource<GraphicsContext>,
    ) -> wasmtime::Result<()> {
        bail!("wasi:webgpu surfaces are not available: this host runs compute only")
    }

    // fn configure(
    //     &mut self,
    //     _device: Resource<Device>,
    //     _descriptor: webgpu::GpuDeviceConfiguration,
    // ) {
    //     bail!("wasi:webgpu: GpuDevice.configure is not implemented by this host")
    // }

    fn adapter_info(
        &mut self,
        device: Resource<Device>,
    ) -> wasmtime::Result<Resource<webgpu::GpuAdapterInfo>> {
        let adapter_id = *self.table.get(&device)?.adapter;
        let info = self.instance.adapter_get_info(adapter_id);
        let info = self.table.push(info)?;
        Ok(info)
    }

    fn create_command_encoder(
        &mut self,
        device: Resource<Device>,
        descriptor: Option<webgpu::GpuCommandEncoderDescriptor>,
    ) -> wasmtime::Result<Resource<CommandEncoder>> {
        let device = self.table.get(&device)?;
        let device_id = device.device;
        let error_handler = Arc::clone(&device.error_handler);

        let (command_encoder_id, err) = self.instance.device_create_command_encoder(
            device_id,
            &descriptor
                .map(|d| d.to_core(self.table))
                .unwrap_or(wgpu_types::CommandEncoderDescriptor::default()),
            None,
        );

        error_handler.handle_possible_error(err);

        let command_encoder = self.table.push(CommandEncoder {
            command_encoder_id,
            error_handler,
        })?;
        Ok(command_encoder)
    }

    fn create_shader_module(
        &mut self,
        device: Resource<Device>,
        descriptor: webgpu::GpuShaderModuleDescriptor,
    ) -> wasmtime::Result<Resource<webgpu::GpuShaderModule>> {
        let device = self.table.get(&device)?;
        let device_id = device.device;
        let error_handler = Arc::clone(&device.error_handler);

        let code =
            wgpu_core::pipeline::ShaderModuleSource::Wgsl(Cow::Owned(descriptor.code.to_owned()));
        let (shader, err) = self.instance.device_create_shader_module(
            device_id,
            &descriptor.to_core(self.table),
            code,
            None,
        );

        error_handler.handle_possible_error(err);

        Ok(self.table.push(shader)?)
    }

    fn create_render_pipeline(
        &mut self,
        device: Resource<Device>,
        descriptor: webgpu::GpuRenderPipelineDescriptor,
    ) -> wasmtime::Result<Resource<RenderPipeline>> {
        let device = self.table.get(&device)?;
        let device_id = device.device;
        let error_handler = Arc::clone(&device.error_handler);

        let (render_pipeline_id, err) = self.instance.device_create_render_pipeline(
            device_id,
            &descriptor.to_core(self.table),
            None,
        );

        error_handler.handle_possible_error(err);

        let render_pipeline = self.table.push(RenderPipeline {
            render_pipeline_id,
            error_handler,
        })?;
        Ok(render_pipeline)
    }

    fn queue(&mut self, device: Resource<Device>) -> wasmtime::Result<Resource<webgpu::GpuQueue>> {
        let queue = Arc::clone(&self.table.get(&device)?.queue);
        Ok(self.table.push(queue)?)
    }

    fn features(
        &mut self,
        device: Resource<webgpu::GpuDevice>,
    ) -> wasmtime::Result<Resource<webgpu::GpuSupportedFeatures>> {
        let device = self.table.get(&device)?.device;
        let features = self.instance.device_features(device);
        Ok(self.table.push(features)?)
    }

    fn limits(
        &mut self,
        device: Resource<webgpu::GpuDevice>,
    ) -> wasmtime::Result<Resource<webgpu::GpuSupportedLimits>> {
        let device = self.table.get(&device)?.device;
        let limits = self.instance.device_limits(device);
        Ok(self.table.push(limits)?)
    }

    fn destroy(&mut self, device: Resource<webgpu::GpuDevice>) -> wasmtime::Result<()> {
        let device_id = self.table.get(&device)?.device;
        self.instance.device_destroy(device_id);
        Ok(())
    }

    fn create_buffer(
        &mut self,
        device: Resource<webgpu::GpuDevice>,
        descriptor: webgpu::GpuBufferDescriptor,
    ) -> wasmtime::Result<Resource<webgpu::GpuBuffer>> {
        let device = self.table.get(&device)?;
        let device_id = device.device;
        let error_handler = Arc::clone(&device.error_handler);
        let descriptor = descriptor.to_core(self.table);

        let size = descriptor.size;
        let usage = descriptor.usage;
        let map_state = match descriptor.mapped_at_creation {
            true => webgpu::GpuBufferMapState::Mapped,
            false => webgpu::GpuBufferMapState::Unmapped,
        };

        let (buffer_id, err) = self
            .instance
            .device_create_buffer(device_id, &descriptor, None);

        error_handler.handle_possible_error(err);

        let buffer = Buffer {
            buffer_id,
            size,
            usage,
            map_state,
        };

        Ok(self.table.push(buffer)?)
    }

    fn create_texture(
        &mut self,
        device: Resource<webgpu::GpuDevice>,
        descriptor: webgpu::GpuTextureDescriptor,
    ) -> wasmtime::Result<Resource<webgpu::GpuTexture>> {
        let device = self.table.get(&device)?;
        let device_id = device.device;
        let error_handler = Arc::clone(&device.error_handler);

        let (texture_id, err) = self.instance.device_create_texture(
            device_id,
            &descriptor.to_core(self.table),
            None,
        );

        error_handler.handle_possible_error(err);

        Ok(self.table.push(Texture {
            texture_id,
            error_handler,
        })?)
    }

    fn create_sampler(
        &mut self,
        device: Resource<webgpu::GpuDevice>,
        descriptor: Option<webgpu::GpuSamplerDescriptor>,
    ) -> wasmtime::Result<Resource<webgpu::GpuSampler>> {
        let device = self.table.get(&device)?;
        let device_id = device.device;
        let error_handler = Arc::clone(&device.error_handler);

        let descriptor = descriptor
            .map(|d| d.to_core(self.table))
            // https://www.w3.org/TR/webgpu/#dictdef-gpusamplerdescriptor
            .unwrap_or_else(|| wgpu_core::resource::SamplerDescriptor {
                label: None,
                address_modes: [wgpu_types::AddressMode::ClampToEdge; 3],
                mag_filter: wgpu_types::FilterMode::Nearest,
                min_filter: wgpu_types::FilterMode::Nearest,
                mipmap_filter: wgpu_types::MipmapFilterMode::Nearest,
                lod_min_clamp: 0.0,
                lod_max_clamp: 32.0,
                compare: None,
                // TODO: make sure that anisotropy_clamp actually corresponds to maxAnisotropy
                anisotropy_clamp: 1,
                // border_color is not present in WebGPU
                border_color: None,
            });

        let (sampler, err) = self
            .instance
            .device_create_sampler(device_id, &descriptor, None);

        error_handler.handle_possible_error(err);

        Ok(self.table.push(sampler)?)
    }

    fn create_bind_group_layout(
        &mut self,
        device: Resource<webgpu::GpuDevice>,
        descriptor: webgpu::GpuBindGroupLayoutDescriptor,
    ) -> wasmtime::Result<Resource<webgpu::GpuBindGroupLayout>> {
        // https://www.w3.org/TR/webgpu/#bind-group-layout-creation: each
        // entry names exactly one of buffer, sampler, texture and
        // storage-texture. The conversion cannot say so, so refuse here.
        for entry in &descriptor.entries {
            let kinds = [
                entry.buffer.is_some(),
                entry.sampler.is_some(),
                entry.texture.is_some(),
                entry.storage_texture.is_some(),
            ];
            if kinds.iter().filter(|set| **set).count() != 1 {
                bail!(
                    "bind group layout entry {} must name exactly one binding type",
                    entry.binding
                );
            }
        }
        let device = self.table.get(&device)?;
        let device_id = device.device;
        let error_handler = Arc::clone(&device.error_handler);

        let (bind_group_layout, err) = self.instance.device_create_bind_group_layout(
            device_id,
            &descriptor.to_core(self.table),
            None,
        );

        error_handler.handle_possible_error(err);

        Ok(self.table.push(bind_group_layout)?)
    }

    fn create_pipeline_layout(
        &mut self,
        device: Resource<webgpu::GpuDevice>,
        descriptor: webgpu::GpuPipelineLayoutDescriptor,
    ) -> wasmtime::Result<Resource<webgpu::GpuPipelineLayout>> {
        let device = self.table.get(&device)?;
        let device_id = device.device;
        let error_handler = Arc::clone(&device.error_handler);

        let (pipeline_layout, err) = self.instance.device_create_pipeline_layout(
            device_id,
            &descriptor.to_core(self.table),
            None,
        );

        error_handler.handle_possible_error(err);

        Ok(self.table.push(pipeline_layout)?)
    }

    fn create_bind_group(
        &mut self,
        device: Resource<webgpu::GpuDevice>,
        descriptor: webgpu::GpuBindGroupDescriptor,
    ) -> wasmtime::Result<Resource<webgpu::GpuBindGroup>> {
        let device = self.table.get(&device)?;
        let device_id = device.device;
        let error_handler = Arc::clone(&device.error_handler);

        let (bind_group, err) = self.instance.device_create_bind_group(
            device_id,
            &descriptor.to_core(self.table),
            None,
        );

        error_handler.handle_possible_error(err);

        Ok(self.table.push(bind_group)?)
    }

    fn create_compute_pipeline(
        &mut self,
        device: Resource<webgpu::GpuDevice>,
        descriptor: webgpu::GpuComputePipelineDescriptor,
    ) -> wasmtime::Result<Resource<webgpu::GpuComputePipeline>> {
        let device = self.table.get(&device)?;
        let device_id = device.device;
        let error_handler = Arc::clone(&device.error_handler);

        let (compute_pipeline_id, err) = self.instance.device_create_compute_pipeline(
            device_id,
            &descriptor.to_core(self.table),
            None,
        );

        error_handler.handle_possible_error(err);

        Ok(self.table.push(ComputePipeline {
            compute_pipeline_id,
            error_handler,
        })?)
    }

    fn create_compute_pipeline_async(
        &mut self,
        device: Resource<webgpu::GpuDevice>,
        descriptor: webgpu::GpuComputePipelineDescriptor,
    ) -> wasmtime::Result<Result<Resource<webgpu::GpuComputePipeline>, webgpu::CreatePipelineError>>
    {
        Ok(Ok(self.create_compute_pipeline(device, descriptor)?))
    }

    fn create_render_pipeline_async(
        &mut self,
        device: Resource<webgpu::GpuDevice>,
        descriptor: webgpu::GpuRenderPipelineDescriptor,
    ) -> wasmtime::Result<Result<Resource<webgpu::GpuRenderPipeline>, webgpu::CreatePipelineError>>
    {
        Ok(Ok(self.create_render_pipeline(device, descriptor)?))
    }

    fn create_render_bundle_encoder(
        &mut self,
        device: Resource<webgpu::GpuDevice>,
        descriptor: webgpu::GpuRenderBundleEncoderDescriptor,
    ) -> wasmtime::Result<Resource<webgpu::GpuRenderBundleEncoder>> {
        let device = self.table.get(&device)?;
        let device_id = device.device;
        let error_handler = Arc::clone(&device.error_handler);
        let render_bundle_encoder = wgpu_core::command::RenderBundleEncoder::new(
            &descriptor.to_core(self.table),
            device_id,
        )?;
        let render_bundle_encoder =
            self.table
                .push(RenderBundleEncoder::new(RenderBundleEncoderInner {
                    render_bundle_encoder,
                    error_handler,
                }))?;
        Ok(render_bundle_encoder)
    }

    fn create_query_set(
        &mut self,
        device: Resource<webgpu::GpuDevice>,
        descriptor: webgpu::GpuQuerySetDescriptor,
    ) -> wasmtime::Result<Result<Resource<webgpu::GpuQuerySet>, webgpu::CreateQuerySetError>> {
        let device = self.table.get(&device)?;
        let device_id = device.device;
        let error_handler = Arc::clone(&device.error_handler);

        let (query_set, err) = self.instance.device_create_query_set(
            device_id,
            &descriptor.to_core(self.table),
            None,
        );

        error_handler.handle_possible_error(err);

        Ok(Ok(self.table.push(query_set)?))
    }

    fn label(&mut self, _device: Resource<webgpu::GpuDevice>) -> wasmtime::Result<String> {
        bail!("wasi:webgpu: GpuDevice.label is not implemented by this host")
    }

    fn set_label(
        &mut self,
        _device: Resource<webgpu::GpuDevice>,
        _label: String,
    ) -> wasmtime::Result<()> {
        bail!("wasi:webgpu: GpuDevice.set_label is not implemented by this host")
    }

    fn lost(
        &mut self,
        _device: Resource<webgpu::GpuDevice>,
    ) -> wasmtime::Result<Resource<webgpu::GpuDeviceLostInfo>> {
        bail!("wasi:webgpu: GpuDevice.lost is not implemented by this host")
    }

    fn push_error_scope(
        &mut self,
        device: Resource<webgpu::GpuDevice>,
        filter: webgpu::GpuErrorFilter,
    ) -> wasmtime::Result<()> {
        let device = self.table.get(&device)?;
        device.error_handler.push_scope(filter);
        Ok(())
    }

    fn pop_error_scope(
        &mut self,
        device: Resource<webgpu::GpuDevice>,
    ) -> wasmtime::Result<Result<Option<Resource<webgpu::GpuError>>, webgpu::PopErrorScopeError>>
    {
        let device = self.table.get(&device)?;
        let error = device.error_handler.pop_scope();

        Ok(match error {
            Ok(Some(error)) => Ok(Some(self.table.push(error)?)),
            Ok(None) => Ok(None),
            Err(e) => Err(e),
        })
    }

    fn onuncapturederror_subscribe(
        &mut self,
        device: Resource<webgpu::GpuDevice>,
    ) -> wasmtime::Result<Resource<poll::Pollable>> {
        let device = self.table.get_mut(&device)?;
        let receiver = device.error_handler.new_error_receiver();
        // 0.0.1 has no way to hand the error itself to the guest: the
        // pollable only says one happened. The host prints each one as it is
        // raised (see ErrorHandler::handle_possible_error).
        let listener = self.table.push(Listener::new(receiver, move |_data| {}))?;
        wasmtime_wasi_io::poll::subscribe(self.table, listener)
    }

    fn drop(&mut self, device: Resource<webgpu::GpuDevice>) -> wasmtime::Result<()> {
        let device = self.table.delete(device)?;
        self.instance.device_drop(device.device);
        if let Some(adapter_id) = Arc::into_inner(device.adapter) {
            self.instance.adapter_drop(adapter_id);
        }
        if let Some(queue_id) = Arc::into_inner(device.queue) {
            self.instance.queue_drop(queue_id);
        }
        Ok(())
    }
}

impl webgpu::HostGpuTexture for WasiWebGpuCtx<'_> {
    fn from_graphics_buffer(
        &mut self,
        _buffer: Resource<AbstractBuffer>,
    ) -> wasmtime::Result<Resource<Texture>> {
        bail!("wasi:webgpu surfaces are not available: this host runs compute only")
    }

    fn create_view(
        &mut self,
        texture: Resource<Texture>,
        descriptor: Option<webgpu::GpuTextureViewDescriptor>,
    ) -> wasmtime::Result<Resource<wgpu_core::id::TextureViewId>> {
        let texture = self.table.get(&texture)?;
        let texture_id = texture.texture_id;
        let error_handler = Arc::clone(&texture.error_handler);
        let (texture_view, err) = self.instance.texture_create_view(
            texture_id,
            &descriptor
                .map(|d| d.to_core(self.table))
                .unwrap_or(wgpu_core::resource::TextureViewDescriptor::default()),
            None,
        );
        error_handler.handle_possible_error(err);
        Ok(self.table.push(texture_view)?)
    }

    fn destroy(&mut self, texture: Resource<webgpu::GpuTexture>) -> wasmtime::Result<()> {
        let texture = self.table.get(&texture)?.texture_id;
        self.instance.texture_destroy(texture);
        Ok(())
    }

    fn width(
        &mut self,
        _self_: Resource<webgpu::GpuTexture>,
    ) -> wasmtime::Result<webgpu::GpuIntegerCoordinateOut> {
        bail!("wasi:webgpu: GpuTexture.width is not implemented by this host")
    }

    fn height(
        &mut self,
        _self_: Resource<webgpu::GpuTexture>,
    ) -> wasmtime::Result<webgpu::GpuIntegerCoordinateOut> {
        bail!("wasi:webgpu: GpuTexture.height is not implemented by this host")
    }

    fn depth_or_array_layers(
        &mut self,
        _self_: Resource<webgpu::GpuTexture>,
    ) -> wasmtime::Result<webgpu::GpuIntegerCoordinateOut> {
        bail!("wasi:webgpu: GpuTexture.depth_or_array_layers is not implemented by this host")
    }

    fn mip_level_count(
        &mut self,
        _self_: Resource<webgpu::GpuTexture>,
    ) -> wasmtime::Result<webgpu::GpuIntegerCoordinateOut> {
        bail!("wasi:webgpu: GpuTexture.mip_level_count is not implemented by this host")
    }

    fn sample_count(
        &mut self,
        _self_: Resource<webgpu::GpuTexture>,
    ) -> wasmtime::Result<webgpu::GpuSize32Out> {
        bail!("wasi:webgpu: GpuTexture.sample_count is not implemented by this host")
    }

    fn dimension(
        &mut self,
        _self_: Resource<webgpu::GpuTexture>,
    ) -> wasmtime::Result<webgpu::GpuTextureDimension> {
        bail!("wasi:webgpu: GpuTexture.dimension is not implemented by this host")
    }

    fn format(
        &mut self,
        _self_: Resource<webgpu::GpuTexture>,
    ) -> wasmtime::Result<webgpu::GpuTextureFormat> {
        bail!("wasi:webgpu: GpuTexture.format is not implemented by this host")
    }

    fn usage(
        &mut self,
        _self_: Resource<webgpu::GpuTexture>,
    ) -> wasmtime::Result<webgpu::GpuFlagsConstant> {
        bail!("wasi:webgpu: GpuTexture.usage is not implemented by this host")
    }

    fn label(&mut self, _self_: Resource<webgpu::GpuTexture>) -> wasmtime::Result<String> {
        // Labels are not kept.
        Ok(String::new())
    }

    fn set_label(
        &mut self,
        _self_: Resource<webgpu::GpuTexture>,
        _label: String,
    ) -> wasmtime::Result<()> {
        // Labels are not kept.
        Ok(())
    }

    fn drop(&mut self, texture: Resource<webgpu::GpuTexture>) -> wasmtime::Result<()> {
        let texture = self.table.delete(texture)?;
        self.instance.texture_drop(texture.texture_id);
        Ok(())
    }
}

impl webgpu::HostGpuTextureView for WasiWebGpuCtx<'_> {
    fn label(
        &mut self,
        _self_: Resource<wgpu_core::id::TextureViewId>,
    ) -> wasmtime::Result<String> {
        // Labels are not kept.
        Ok(String::new())
    }

    fn set_label(
        &mut self,
        _self_: Resource<wgpu_core::id::TextureViewId>,
        _label: String,
    ) -> wasmtime::Result<()> {
        // Labels are not kept.
        Ok(())
    }

    fn drop(&mut self, view: Resource<wgpu_core::id::TextureViewId>) -> wasmtime::Result<()> {
        let view_id = self.table.delete(view)?;
        self.instance.texture_view_drop(view_id);
        Ok(())
    }
}

impl webgpu::HostGpuCommandBuffer for WasiWebGpuCtx<'_> {
    fn label(&mut self, _self_: Resource<webgpu::GpuCommandBuffer>) -> wasmtime::Result<String> {
        // Labels are not kept.
        Ok(String::new())
    }

    fn set_label(
        &mut self,
        _self_: Resource<webgpu::GpuCommandBuffer>,
        _label: String,
    ) -> wasmtime::Result<()> {
        // Labels are not kept.
        Ok(())
    }

    fn drop(&mut self, command_buffer: Resource<webgpu::GpuCommandBuffer>) -> wasmtime::Result<()> {
        let command_buffer_id = self.table.delete(command_buffer)?;
        self.instance.command_buffer_drop(command_buffer_id);
        Ok(())
    }
}

impl webgpu::HostGpuShaderModule for WasiWebGpuCtx<'_> {
    fn get_compilation_info(
        &mut self,
        _self_: Resource<webgpu::GpuShaderModule>,
    ) -> wasmtime::Result<Resource<webgpu::GpuCompilationInfo>> {
        bail!("wasi:webgpu: GpuShaderModule.get_compilation_info is not implemented by this host")
    }

    fn label(&mut self, _self_: Resource<webgpu::GpuShaderModule>) -> wasmtime::Result<String> {
        // Labels are not kept.
        Ok(String::new())
    }

    fn set_label(
        &mut self,
        _self_: Resource<webgpu::GpuShaderModule>,
        _label: String,
    ) -> wasmtime::Result<()> {
        // Labels are not kept.
        Ok(())
    }

    fn drop(&mut self, shader: Resource<webgpu::GpuShaderModule>) -> wasmtime::Result<()> {
        let shader_id = self.table.delete(shader)?;
        self.instance.shader_module_drop(shader_id);
        Ok(())
    }
}

impl webgpu::HostGpuRenderPipeline for WasiWebGpuCtx<'_> {
    fn label(&mut self, _self_: Resource<RenderPipeline>) -> wasmtime::Result<String> {
        // Labels are not kept.
        Ok(String::new())
    }

    fn set_label(
        &mut self,
        _self_: Resource<RenderPipeline>,
        _label: String,
    ) -> wasmtime::Result<()> {
        // Labels are not kept.
        Ok(())
    }

    fn get_bind_group_layout(
        &mut self,
        pipeline: Resource<RenderPipeline>,
        index: u32,
    ) -> wasmtime::Result<Resource<webgpu::GpuBindGroupLayout>> {
        let pipeline = self.table.get(&pipeline)?;
        let pipeline_id = pipeline.render_pipeline_id;
        let error_handler = Arc::clone(&pipeline.error_handler);
        let (layout, err) =
            self.instance
                .render_pipeline_get_bind_group_layout(pipeline_id, index, None);
        error_handler.handle_possible_error(err);
        Ok(self.table.push(layout)?)
    }

    fn drop(&mut self, pipeline: Resource<webgpu::GpuRenderPipeline>) -> wasmtime::Result<()> {
        let pipeline = self.table.delete(pipeline)?;
        self.instance
            .render_pipeline_drop(pipeline.render_pipeline_id);
        Ok(())
    }
}

impl webgpu::HostGpuAdapter for WasiWebGpuCtx<'_> {
    fn request_device(
        &mut self,
        adapter: Resource<webgpu::GpuAdapter>,
        descriptor: Option<webgpu::GpuDeviceDescriptor>,
    ) -> wasmtime::Result<Result<Resource<webgpu::GpuDevice>, webgpu::RequestDeviceError>> {
        let adapter = Arc::clone(self.table.get(&adapter)?);

        if let Some(limits) = descriptor.as_ref().and_then(|d| d.required_limits.as_ref()) {
            if let Some(key) = crate::to_core_conversions::unknown_limit_key(self.table.get(limits)?)
            {
                // https://www.w3.org/TR/webgpu/#dom-gpuadapter-requestdevice
                return Ok(Err(webgpu::RequestDeviceError {
                    kind: webgpu::RequestDeviceErrorKind::OperationError,
                    message: format!("required-limits names no limit called {key}"),
                }));
            }
        }

        let device_queue_result = self.instance.adapter_request_device(
            *adapter,
            &descriptor
                .map(|d| d.to_core(self.table))
                .unwrap_or(wgpu_types::DeviceDescriptor::default()),
            None,
            None,
        );

        Ok(match device_queue_result {
            Ok((device_id, queue_id)) => {
                let device = self.table.push(Device {
                    device: device_id,
                    queue: Arc::new(queue_id),
                    adapter,
                    error_handler: Arc::new(ErrorHandler::default()),
                })?;
                Ok(device)
            }

            Err(err) => {
                let message = err.to_string();
                // https://www.w3.org/TR/webgpu/#dom-gpuadapter-requestdevice:
                // a feature the adapter lacks rejects with a TypeError, a
                // limit better than the adapter's with an OperationError, and
                // anything else wgpu refuses is taken as the latter.
                let kind = match err {
                    wgpu_core::instance::RequestDeviceError::UnsupportedFeature(_) => {
                        webgpu::RequestDeviceErrorKind::TypeError
                    }
                    _ => webgpu::RequestDeviceErrorKind::OperationError,
                };
                Err(webgpu::RequestDeviceError { kind, message })
            }
        })
    }

    fn features(
        &mut self,
        adapter: Resource<webgpu::GpuAdapter>,
    ) -> wasmtime::Result<Resource<webgpu::GpuSupportedFeatures>> {
        let adapter = *(*self.table.get(&adapter)?);
        let features = self.instance.adapter_features(adapter);
        Ok(self.table.push(features)?)
    }

    fn limits(
        &mut self,
        adapter: Resource<webgpu::GpuAdapter>,
    ) -> wasmtime::Result<Resource<webgpu::GpuSupportedLimits>> {
        let adapter = *(*self.table.get(&adapter)?);
        let limits = self.instance.adapter_limits(adapter);
        Ok(self.table.push(limits)?)
    }

    fn is_fallback_adapter(
        &mut self,
        adapter: Resource<webgpu::GpuAdapter>,
    ) -> wasmtime::Result<bool> {
        let adapter = *(*self.table.get(&adapter)?);
        let device_type = self.instance.adapter_get_info(adapter).device_type;
        // wgpu in browser treats only cpu as fallback
        // https://github.com/gfx-rs/wgpu/blob/0d32f7e75604feeff976445576c234da377fa3df/wgpu/src/backend/webgpu.rs#L889-L893
        let is_fallback = match device_type {
            wgpu_types::DeviceType::IntegratedGpu
            | wgpu_types::DeviceType::DiscreteGpu
            | wgpu_types::DeviceType::VirtualGpu
            | wgpu_types::DeviceType::Other => false,
            wgpu_types::DeviceType::Cpu => true,
        };
        Ok(is_fallback)
    }

    fn info(
        &mut self,
        adapter: Resource<webgpu::GpuAdapter>,
    ) -> wasmtime::Result<Resource<webgpu::GpuAdapterInfo>> {
        let adapter_id = *(*self.table.get(&adapter)?);
        let info = self.instance.adapter_get_info(adapter_id);
        Ok(self.table.push(info)?)
    }

    fn drop(&mut self, adapter: Resource<webgpu::GpuAdapter>) -> wasmtime::Result<()> {
        let adapter_id = self.table.delete(adapter)?;
        if let Some(adapter_id) = Arc::into_inner(adapter_id) {
            self.instance.adapter_drop(adapter_id);
        }
        Ok(())
    }
}

impl webgpu::HostGpuQueue for WasiWebGpuCtx<'_> {
    fn submit(
        &mut self,
        queue: Resource<webgpu::GpuQueue>,
        val: Vec<Resource<webgpu::GpuCommandBuffer>>,
    ) -> wasmtime::Result<()> {
        let command_buffers = val
            .into_iter()
            .map(|buffer| Ok(*self.table.get(&buffer)?))
            .collect::<wasmtime::Result<Vec<_>>>()?;
        let queue = *(*self.table.get(&queue)?);
        if let Err((_, e)) = self.instance.queue_submit(queue, &command_buffers) {
            bail!("queue submit failed: {e}");
        }
        Ok(())
    }

    fn on_submitted_work_done(&mut self, queue: Resource<webgpu::GpuQueue>) -> wasmtime::Result<()> {
        // Synchronous, as map-async is: returns once the queue has drained.
        let queue = *(*self.table.get(&queue)?);
        let (sender, receiver) = std::sync::mpsc::channel();
        self.instance.queue_on_submitted_work_done(
            queue,
            Box::new(move || {
                let _ = sender.send(());
            }),
        );
        while receiver.try_recv().is_err() {
            // TODO: only poll this queue's device.
            self.instance.poll_all_devices(true)?;
        }
        Ok(())
    }

    fn write_buffer_with_copy(
        &mut self,
        queue: Resource<webgpu::GpuQueue>,
        buffer: Resource<webgpu::GpuBuffer>,
        buffer_offset: webgpu::GpuSize64,
        data: Vec<u8>,
        data_offset: Option<webgpu::GpuSize64>,
        size: Option<webgpu::GpuSize64>,
    ) -> wasmtime::Result<Result<(), webgpu::WriteBufferError>> {
        let queue = *(*self.table.get(&queue)?);
        let buffer_id = self.table.get(&buffer)?.buffer_id;
        let mut data = &data[..];
        if let Some(data_offset) = data_offset {
            let data_offset = data_offset as usize;
            data = &data[data_offset..];
        }
        if let Some(size) = size {
            let size = size as usize;
            data = &data[..size];
        }
        self.instance
            .queue_write_buffer(queue, buffer_id, buffer_offset, data)?;
        Ok(Ok(()))
    }

    fn write_texture_with_copy(
        &mut self,
        queue: Resource<webgpu::GpuQueue>,
        destination: webgpu::GpuTexelCopyTextureInfo,
        data: Vec<u8>,
        data_layout: webgpu::GpuTexelCopyBufferLayout,
        size: webgpu::GpuExtent3D,
    ) -> wasmtime::Result<()> {
        let queue = *(*self.table.get(&queue)?);
        self.instance.queue_write_texture(
            queue,
            &destination.to_core(self.table),
            &data,
            &data_layout.to_core(self.table),
            &size.to_core(self.table),
        )?;
        Ok(())
    }

    fn label(&mut self, _self_: Resource<webgpu::GpuQueue>) -> wasmtime::Result<String> {
        // Labels are not kept.
        Ok(String::new())
    }

    fn set_label(
        &mut self,
        _self_: Resource<webgpu::GpuQueue>,
        _label: String,
    ) -> wasmtime::Result<()> {
        // Labels are not kept.
        Ok(())
    }

    fn drop(&mut self, queue: Resource<webgpu::GpuQueue>) -> wasmtime::Result<()> {
        let queue_id = self.table.delete(queue)?;
        if let Some(queue_id) = Arc::into_inner(queue_id) {
            self.instance.queue_drop(queue_id);
        }
        Ok(())
    }
}

impl webgpu::HostGpuCommandEncoder for WasiWebGpuCtx<'_> {
    fn begin_render_pass(
        &mut self,
        command_encoder: Resource<CommandEncoder>,
        descriptor: webgpu::GpuRenderPassDescriptor,
    ) -> wasmtime::Result<Resource<webgpu::GpuRenderPassEncoder>> {
        let command_encoder = self.table.get(&command_encoder)?;
        let command_encoder_id = command_encoder.command_encoder_id;
        let error_handler = Arc::clone(&command_encoder.error_handler);
        let timestamp_writes = descriptor
            .timestamp_writes
            .map(|tw| tw.to_core(self.table));
        // can't use to_core because depth_stencil_attachment is Option<&x>.
        let depth_stencil_attachment = descriptor
            .depth_stencil_attachment
            .map(|d| d.to_core(self.table));
        let descriptor = wgpu_core::command::RenderPassDescriptor {
            label: descriptor.label.map(|l| l.into()),
            color_attachments: descriptor
                .color_attachments
                .into_iter()
                .map(|c| c.map(|c| c.to_core(self.table)))
                .collect::<Vec<_>>()
                .into(),
            depth_stencil_attachment: depth_stencil_attachment.as_ref(),
            timestamp_writes: timestamp_writes.as_ref(),
            occlusion_query_set: descriptor
                .occlusion_query_set
                .map(|oqs| oqs.to_core(self.table)),
            // multiview_mask is not present in WebGPU
            multiview_mask: None,
            // TODO: self.max_draw_count not used
        };
        let (render_pass, err) = self
            .instance
            .command_encoder_begin_render_pass(command_encoder_id, &descriptor);

        error_handler.handle_possible_error(err);

        Ok(self.table.push(RenderPassEncoder::new(render_pass))?)
    }

    fn finish(
        &mut self,
        command_encoder: Resource<CommandEncoder>,
        descriptor: Option<webgpu::GpuCommandBufferDescriptor>,
    ) -> wasmtime::Result<Resource<webgpu::GpuCommandBuffer>> {
        let command_encoder = self.table.get(&command_encoder)?;
        let command_encoder_id = command_encoder.command_encoder_id;
        let error_handler = Arc::clone(&command_encoder.error_handler);
        let (command_buffer, err) = self.instance.command_encoder_finish(
            command_encoder_id,
            &descriptor
                .map(|d| d.to_core(self.table))
                .unwrap_or(wgpu_types::CommandBufferDescriptor::default()),
            None,
        );
        // dropping the label.
        // TODO: reconsider when implementing real labels.
        let err = err.map(|(_label, err)| err);
        error_handler.handle_possible_error(err);
        Ok(self.table.push(command_buffer)?)
    }

    fn begin_compute_pass(
        &mut self,
        command_encoder: Resource<CommandEncoder>,
        descriptor: Option<webgpu::GpuComputePassDescriptor>,
    ) -> wasmtime::Result<Resource<webgpu::GpuComputePassEncoder>> {
        let command_encoder = self.table.get(&command_encoder)?;
        let command_encoder_id = command_encoder.command_encoder_id;
        let error_handler = Arc::clone(&command_encoder.error_handler);
        let (compute_pass, err) = self.instance.command_encoder_begin_compute_pass(
            command_encoder_id,
            // can't use to_core because timestamp_writes is Option<&x>.
            &wgpu_core::command::ComputePassDescriptor {
                // TODO: can we get rid of the clone here?
                label: descriptor
                    .as_ref()
                    .and_then(|d| d.label.clone().map(|l| l.into())),
                timestamp_writes: descriptor
                    .and_then(|d| d.timestamp_writes.map(|tw| tw.to_core(self.table))),
            },
        );
        error_handler.handle_possible_error(err);
        Ok(self.table.push(ComputePassEncoder::new(compute_pass))?)
    }

    fn copy_buffer_to_buffer(
        &mut self,
        command_encoder: Resource<CommandEncoder>,
        source: Resource<webgpu::GpuBuffer>,
        source_offset: webgpu::GpuSize64,
        destination: Resource<webgpu::GpuBuffer>,
        destination_offset: webgpu::GpuSize64,
        size: webgpu::GpuSize64,
    ) -> wasmtime::Result<()> {
        let command_encoder = self.table.get(&command_encoder)?.command_encoder_id;
        let source = self.table.get(&source)?.buffer_id;
        let destination = self.table.get(&destination)?.buffer_id;
        self.instance.command_encoder_copy_buffer_to_buffer(
            command_encoder,
            source,
            source_offset,
            destination,
            destination_offset,
            Some(size),
        )?;
        Ok(())
    }

    fn copy_buffer_to_texture(
        &mut self,
        command_encoder: Resource<CommandEncoder>,
        source: webgpu::GpuTexelCopyBufferInfo,
        destination: webgpu::GpuTexelCopyTextureInfo,
        copy_size: webgpu::GpuExtent3D,
    ) -> wasmtime::Result<()> {
        let command_encoder = self.table.get(&command_encoder)?.command_encoder_id;
        self.instance.command_encoder_copy_buffer_to_texture(
            command_encoder,
            &source.to_core(self.table),
            &destination.to_core(self.table),
            &copy_size.to_core(self.table),
        )?;
        Ok(())
    }

    fn copy_texture_to_buffer(
        &mut self,
        command_encoder: Resource<CommandEncoder>,
        source: webgpu::GpuTexelCopyTextureInfo,
        destination: webgpu::GpuTexelCopyBufferInfo,
        copy_size: webgpu::GpuExtent3D,
    ) -> wasmtime::Result<()> {
        let command_encoder = self.table.get(&command_encoder)?.command_encoder_id;
        self.instance.command_encoder_copy_texture_to_buffer(
            command_encoder,
            &source.to_core(self.table),
            &destination.to_core(self.table),
            &copy_size.to_core(self.table),
        )?;
        Ok(())
    }

    fn copy_texture_to_texture(
        &mut self,
        command_encoder: Resource<CommandEncoder>,
        source: webgpu::GpuTexelCopyTextureInfo,
        destination: webgpu::GpuTexelCopyTextureInfo,
        copy_size: webgpu::GpuExtent3D,
    ) -> wasmtime::Result<()> {
        let command_encoder = self.table.get(&command_encoder)?.command_encoder_id;
        self.instance.command_encoder_copy_texture_to_texture(
            command_encoder,
            &source.to_core(self.table),
            &destination.to_core(self.table),
            &copy_size.to_core(self.table),
        )?;
        Ok(())
    }

    fn clear_buffer(
        &mut self,
        command_encoder: Resource<CommandEncoder>,
        buffer: Resource<webgpu::GpuBuffer>,
        offset: Option<webgpu::GpuSize64>,
        size: Option<webgpu::GpuSize64>,
    ) -> wasmtime::Result<()> {
        let buffer_id = self.table.get(&buffer)?.buffer_id;
        let command_encoder = self.table.get(&command_encoder)?.command_encoder_id;
        // https://www.w3.org/TR/webgpu/#gpucommandencoder
        self.instance.command_encoder_clear_buffer(
            command_encoder,
            buffer_id,
            offset.unwrap_or(0),
            size,
        )?;
        Ok(())
    }

    fn resolve_query_set(
        &mut self,
        command_encoder: Resource<CommandEncoder>,
        query_set: Resource<webgpu::GpuQuerySet>,
        first_query: webgpu::GpuSize32,
        query_count: webgpu::GpuSize32,
        destination: Resource<webgpu::GpuBuffer>,
        destination_offset: webgpu::GpuSize64,
    ) -> wasmtime::Result<()> {
        let query_set_id = *self.table.get(&query_set)?;
        let destination = self.table.get(&destination)?.buffer_id;
        let command_encoder = self.table.get(&command_encoder)?.command_encoder_id;
        self.instance.command_encoder_resolve_query_set(
            command_encoder,
            query_set_id,
            first_query,
            query_count,
            destination,
            destination_offset,
        )?;
        Ok(())
    }

    fn label(&mut self, command_encoder: Resource<CommandEncoder>) -> wasmtime::Result<String> {
        let _command_encoder = self.table.get(&command_encoder)?;
        // TODO: return real label
        Ok(String::new())
    }

    fn set_label(
        &mut self,
        _self_: Resource<CommandEncoder>,
        _label: String,
    ) -> wasmtime::Result<()> {
        // Labels are not kept.
        Ok(())
    }

    fn push_debug_group(
        &mut self,
        command_encoder: Resource<CommandEncoder>,
        group_label: String,
    ) -> wasmtime::Result<()> {
        let command_encoder = self.table.get(&command_encoder)?.command_encoder_id;
        self.instance
            .command_encoder_push_debug_group(command_encoder, &group_label)?;
        Ok(())
    }

    fn pop_debug_group(
        &mut self,
        command_encoder: Resource<CommandEncoder>,
    ) -> wasmtime::Result<()> {
        let command_encoder = self.table.get(&command_encoder)?.command_encoder_id;
        self.instance
            .command_encoder_pop_debug_group(command_encoder)?;
        Ok(())
    }

    fn insert_debug_marker(
        &mut self,
        command_encoder: Resource<CommandEncoder>,
        marker_label: String,
    ) -> wasmtime::Result<()> {
        let command_encoder = self.table.get(&command_encoder)?.command_encoder_id;
        self.instance
            .command_encoder_insert_debug_marker(command_encoder, &marker_label)?;
        Ok(())
    }

    fn drop(&mut self, command_encoder: Resource<CommandEncoder>) -> wasmtime::Result<()> {
        let command_encoder = self.table.delete(command_encoder)?;
        self.instance
            .command_encoder_drop(command_encoder.command_encoder_id);
        Ok(())
    }
}

impl webgpu::HostGpuRenderPassEncoder for WasiWebGpuCtx<'_> {
    fn set_pipeline(
        &mut self,
        render_pass: Resource<RenderPassEncoder>,
        pipeline: Resource<webgpu::GpuRenderPipeline>,
    ) -> wasmtime::Result<()> {
        let pipeline_id = self.table.get(&pipeline)?.render_pipeline_id;
        let instance = Arc::clone(self.instance);
        let mut render_pass = self.table.get_mut(&render_pass)?.lock();
        let render_pass = render_pass.as_mut().ok_or_else(|| wasmtime::format_err!("the render pass has already ended"))?;
        instance.render_pass_set_pipeline(render_pass, pipeline_id)?;
        Ok(())
    }

    fn draw(
        &mut self,
        render_pass: Resource<RenderPassEncoder>,
        vertex_count: webgpu::GpuSize32,
        instance_count: Option<webgpu::GpuSize32>,
        first_vertex: Option<webgpu::GpuSize32>,
        first_instance: Option<webgpu::GpuSize32>,
    ) -> wasmtime::Result<()> {
        let instance = Arc::clone(self.instance);
        let mut render_pass = self.table.get_mut(&render_pass)?.lock();
        let render_pass = render_pass.as_mut().ok_or_else(|| wasmtime::format_err!("the render pass has already ended"))?;
        // https://www.w3.org/TR/webgpu/#gpurendercommandsmixin
        instance.render_pass_draw(
            render_pass,
            vertex_count,
            instance_count.unwrap_or(1),
            first_vertex.unwrap_or(0),
            first_instance.unwrap_or(0),
        )?;
        Ok(())
    }

    fn end(&mut self, render_pass: Resource<RenderPassEncoder>) -> wasmtime::Result<()> {
        let instance = Arc::clone(self.instance);
        let mut render_pass = self.table.get_mut(&render_pass)?.lock();
        let mut render_pass = render_pass.take().ok_or_else(|| wasmtime::format_err!("the render pass has already ended"))?;
        instance.render_pass_end(&mut render_pass)?;
        Ok(())
    }

    fn set_viewport(
        &mut self,
        render_pass: Resource<RenderPassEncoder>,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) -> wasmtime::Result<()> {
        let instance = Arc::clone(self.instance);
        let mut render_pass = self.table.get_mut(&render_pass)?.lock();
        let render_pass = render_pass.as_mut().ok_or_else(|| wasmtime::format_err!("the render pass has already ended"))?;
        instance.render_pass_set_viewport(
            render_pass,
            x,
            y,
            width,
            height,
            min_depth,
            max_depth,
        )?;
        Ok(())
    }

    fn set_scissor_rect(
        &mut self,
        render_pass: Resource<RenderPassEncoder>,
        x: webgpu::GpuIntegerCoordinate,
        y: webgpu::GpuIntegerCoordinate,
        width: webgpu::GpuIntegerCoordinate,
        height: webgpu::GpuIntegerCoordinate,
    ) -> wasmtime::Result<()> {
        let instance = Arc::clone(self.instance);
        let mut render_pass = self.table.get_mut(&render_pass)?.lock();
        let render_pass = render_pass.as_mut().ok_or_else(|| wasmtime::format_err!("the render pass has already ended"))?;
        instance.render_pass_set_scissor_rect(render_pass, x, y, width, height)?;
        Ok(())
    }

    fn set_blend_constant(
        &mut self,
        render_pass: Resource<RenderPassEncoder>,
        color: webgpu::GpuColor,
    ) -> wasmtime::Result<()> {
        let instance = Arc::clone(self.instance);
        let mut render_pass = self.table.get_mut(&render_pass)?.lock();
        let render_pass = render_pass.as_mut().ok_or_else(|| wasmtime::format_err!("the render pass has already ended"))?;
        instance.render_pass_set_blend_constant(render_pass, color.into())?;
        Ok(())
    }

    fn set_stencil_reference(
        &mut self,
        render_pass: Resource<RenderPassEncoder>,
        reference: webgpu::GpuStencilValue,
    ) -> wasmtime::Result<()> {
        let instance = Arc::clone(self.instance);
        let mut render_pass = self.table.get_mut(&render_pass)?.lock();
        let render_pass = render_pass.as_mut().ok_or_else(|| wasmtime::format_err!("the render pass has already ended"))?;
        instance.render_pass_set_stencil_reference(render_pass, reference)?;
        Ok(())
    }

    fn begin_occlusion_query(
        &mut self,
        render_pass: Resource<RenderPassEncoder>,
        query_index: webgpu::GpuSize32,
    ) -> wasmtime::Result<()> {
        let instance = Arc::clone(self.instance);
        let mut render_pass = self.table.get_mut(&render_pass)?.lock();
        let render_pass = render_pass.as_mut().ok_or_else(|| wasmtime::format_err!("the render pass has already ended"))?;
        instance.render_pass_begin_occlusion_query(render_pass, query_index)?;
        Ok(())
    }

    fn end_occlusion_query(
        &mut self,
        render_pass: Resource<RenderPassEncoder>,
    ) -> wasmtime::Result<()> {
        let instance = Arc::clone(self.instance);
        let mut render_pass = self.table.get_mut(&render_pass)?.lock();
        let render_pass = render_pass.as_mut().ok_or_else(|| wasmtime::format_err!("the render pass has already ended"))?;
        instance.render_pass_end_occlusion_query(render_pass)?;
        Ok(())
    }

    fn execute_bundles(
        &mut self,
        render_pass: Resource<RenderPassEncoder>,
        bundles: Vec<Resource<webgpu::GpuRenderBundle>>,
    ) -> wasmtime::Result<()> {
        let instance = Arc::clone(self.instance);
        let render_bundle_ids = bundles
            .iter()
            .map(|bundle| Ok(*self.table.get(bundle)?))
            .collect::<wasmtime::Result<Vec<_>>>()?;
        let mut render_pass = self.table.get_mut(&render_pass)?.lock();
        let render_pass = render_pass.as_mut().ok_or_else(|| wasmtime::format_err!("the render pass has already ended"))?;
        instance.render_pass_execute_bundles(render_pass, &render_bundle_ids)?;
        Ok(())
    }

    fn label(&mut self, _self_: Resource<RenderPassEncoder>) -> wasmtime::Result<String> {
        // Labels are not kept.
        Ok(String::new())
    }

    fn set_label(
        &mut self,
        _self_: Resource<RenderPassEncoder>,
        _label: String,
    ) -> wasmtime::Result<()> {
        // Labels are not kept.
        Ok(())
    }

    fn push_debug_group(
        &mut self,
        render_pass: Resource<RenderPassEncoder>,
        group_label: String,
    ) -> wasmtime::Result<()> {
        let instance = Arc::clone(self.instance);
        let mut render_pass = self.table.get_mut(&render_pass)?.lock();
        let render_pass = render_pass.as_mut().ok_or_else(|| wasmtime::format_err!("the render pass has already ended"))?;
        instance.render_pass_push_debug_group(render_pass, &group_label, 0)?;
        Ok(())
    }

    fn pop_debug_group(
        &mut self,
        render_pass: Resource<RenderPassEncoder>,
    ) -> wasmtime::Result<()> {
        let instance = Arc::clone(self.instance);
        let mut render_pass = self.table.get_mut(&render_pass)?.lock();
        let render_pass = render_pass.as_mut().ok_or_else(|| wasmtime::format_err!("the render pass has already ended"))?;
        instance.render_pass_pop_debug_group(render_pass)?;
        Ok(())
    }

    fn insert_debug_marker(
        &mut self,
        render_pass: Resource<RenderPassEncoder>,
        marker_label: String,
    ) -> wasmtime::Result<()> {
        let instance = Arc::clone(self.instance);
        let mut render_pass = self.table.get_mut(&render_pass)?.lock();
        let render_pass = render_pass.as_mut().ok_or_else(|| wasmtime::format_err!("the render pass has already ended"))?;
        instance.render_pass_insert_debug_marker(render_pass, &marker_label, 0)?;
        Ok(())
    }

    fn set_bind_group(
        &mut self,
        render_pass: Resource<RenderPassEncoder>,
        index: webgpu::GpuIndex32,
        bind_group: Option<Resource<webgpu::GpuBindGroup>>,
        dynamic_offsets_data: Option<Vec<webgpu::GpuBufferDynamicOffset>>,
        dynamic_offsets_data_start: Option<webgpu::GpuSize64>,
        dynamic_offsets_data_length: Option<webgpu::GpuSize32>,
    ) -> wasmtime::Result<Result<(), webgpu::SetBindGroupError>> {
        let instance = Arc::clone(self.instance);
        let bind_group = bind_group.map(|bind_group| self.table.get(&bind_group).copied()).transpose()?;
        let mut render_pass = self.table.get_mut(&render_pass)?.lock();
        let render_pass = render_pass.as_mut().ok_or_else(|| wasmtime::format_err!("the render pass has already ended"))?;
        // https://www.w3.org/TR/webgpu/#programmable-passes
        let dynamic_offsets = &dynamic_offsets_data.unwrap_or(vec![]);
        let mut dynamic_offsets = &dynamic_offsets[..];
        if let Some(dynamic_offsets_data_start) = dynamic_offsets_data_start {
            let dynamic_offsets_data_start = dynamic_offsets_data_start as usize;
            dynamic_offsets = &dynamic_offsets[dynamic_offsets_data_start..];
        }
        if let Some(dynamic_offsets_data_length) = dynamic_offsets_data_length {
            let dynamic_offsets_data_length = dynamic_offsets_data_length as usize;
            dynamic_offsets = &dynamic_offsets[..dynamic_offsets_data_length];
        }

        instance.render_pass_set_bind_group(render_pass, index, bind_group, dynamic_offsets)?;
        Ok(Ok(()))
    }

    fn set_index_buffer(
        &mut self,
        render_pass: Resource<RenderPassEncoder>,
        buffer: Resource<webgpu::GpuBuffer>,
        index_format: webgpu::GpuIndexFormat,
        offset: Option<webgpu::GpuSize64>,
        size: Option<webgpu::GpuSize64>,
    ) -> wasmtime::Result<()> {
        let instance = Arc::clone(self.instance);
        let buffer_id = self.table.get(&buffer)?.buffer_id;
        let mut render_pass = self.table.get_mut(&render_pass)?.lock();
        let render_pass = render_pass.as_mut().ok_or_else(|| wasmtime::format_err!("the render pass has already ended"))?;
        instance.render_pass_set_index_buffer(
            render_pass,
            buffer_id,
            index_format.into(),
            // https://www.w3.org/TR/webgpu/#gpurendercommandsmixin
            offset.unwrap_or(0),
            size.map(|s| NonZeroU64::new(s).expect("Size can't be zero")),
        )?;
        Ok(())
    }

    fn set_vertex_buffer(
        &mut self,
        render_pass: Resource<RenderPassEncoder>,
        slot: webgpu::GpuIndex32,
        buffer: Option<Resource<webgpu::GpuBuffer>>,
        offset: Option<webgpu::GpuSize64>,
        size: Option<webgpu::GpuSize64>,
    ) -> wasmtime::Result<()> {
        let instance = Arc::clone(self.instance);
        let buffer_id = self
            .table
            .get(&buffer.expect("TODO: deal null buffers"))?
            .buffer_id;
        let mut render_pass = self.table.get_mut(&render_pass)?.lock();
        let render_pass = render_pass.as_mut().ok_or_else(|| wasmtime::format_err!("the render pass has already ended"))?;
        instance.render_pass_set_vertex_buffer(
            render_pass,
            slot,
            buffer_id,
            // https://www.w3.org/TR/webgpu/#gpurendercommandsmixin
            offset.unwrap_or(0),
            size.map(|s| NonZeroU64::new(s).expect("Size can't be zero")),
        )?;
        Ok(())
    }

    fn draw_indexed(
        &mut self,
        render_pass: Resource<RenderPassEncoder>,
        index_count: webgpu::GpuSize32,
        instance_count: Option<webgpu::GpuSize32>,
        first_index: Option<webgpu::GpuSize32>,
        base_vertex: Option<webgpu::GpuSignedOffset32>,
        first_instance: Option<webgpu::GpuSize32>,
    ) -> wasmtime::Result<()> {
        let instance = Arc::clone(self.instance);
        let mut render_pass = self.table.get_mut(&render_pass)?.lock();
        let render_pass = render_pass.as_mut().ok_or_else(|| wasmtime::format_err!("the render pass has already ended"))?;
        instance.render_pass_draw_indexed(
            render_pass,
            index_count,
            // https://www.w3.org/TR/webgpu/#gpurendercommandsmixin
            instance_count.unwrap_or(1),
            first_index.unwrap_or(0),
            base_vertex.unwrap_or(0),
            first_instance.unwrap_or(0),
        )?;
        Ok(())
    }

    fn draw_indirect(
        &mut self,
        render_pass: Resource<RenderPassEncoder>,
        indirect_buffer: Resource<webgpu::GpuBuffer>,
        indirect_offset: webgpu::GpuSize64,
    ) -> wasmtime::Result<()> {
        let instance = Arc::clone(self.instance);
        let indirect_buffer = self.table.get(&indirect_buffer)?.buffer_id;
        let mut render_pass = self.table.get_mut(&render_pass)?.lock();
        let render_pass = render_pass.as_mut().ok_or_else(|| wasmtime::format_err!("the render pass has already ended"))?;
        instance.render_pass_draw_indirect(render_pass, indirect_buffer, indirect_offset)?;
        Ok(())
    }

    fn draw_indexed_indirect(
        &mut self,
        render_pass: Resource<RenderPassEncoder>,
        indirect_buffer: Resource<webgpu::GpuBuffer>,
        indirect_offset: webgpu::GpuSize64,
    ) -> wasmtime::Result<()> {
        let instance = Arc::clone(self.instance);
        let indirect_buffer = self.table.get(&indirect_buffer)?.buffer_id;
        let mut render_pass = self.table.get_mut(&render_pass)?.lock();
        let render_pass = render_pass.as_mut().ok_or_else(|| wasmtime::format_err!("the render pass has already ended"))?;
        instance.render_pass_draw_indexed_indirect(
            render_pass,
            indirect_buffer,
            indirect_offset,
        )?;
        Ok(())
    }

    fn drop(&mut self, render_pass: Resource<RenderPassEncoder>) -> wasmtime::Result<()> {
        self.table.delete(render_pass)?;
        Ok(())
    }
}

impl webgpu::HostGpuUncapturedErrorEvent for WasiWebGpuCtx<'_> {
    // fn new(
    //     &mut self,
    //     _type_: String,
    //     _gpu_uncaptured_error_event_init_dict: webgpu::GpuUncapturedErrorEventInit,
    // ) -> Resource<webgpu::GpuUncapturedErrorEvent> {
    //     bail!("wasi:webgpu: GpuUncapturedErrorEvent.new is not implemented by this host")
    // }

    fn error(
        &mut self,
        _self_: Resource<webgpu::GpuUncapturedErrorEvent>,
    ) -> wasmtime::Result<Resource<webgpu::GpuError>> {
        bail!("wasi:webgpu: GpuUncapturedErrorEvent.error is not implemented by this host")
    }

    fn drop(&mut self, _error: Resource<webgpu::GpuUncapturedErrorEvent>) -> wasmtime::Result<()> {
        bail!("wasi:webgpu: GpuUncapturedErrorEvent.drop is not implemented by this host")
    }
}
// impl webgpu::HostGpuInternalError for WasiWebGpuCtx<'_> {
//     fn new(&mut self, _message: String) -> Resource<webgpu::GpuInternalError> {
//         bail!("wasi:webgpu: GpuUncapturedErrorEvent.new is not implemented by this host")
//     }

//     fn message(&mut self, _self_: Resource<webgpu::GpuInternalError>) -> String {
//         bail!("wasi:webgpu: GpuUncapturedErrorEvent.message is not implemented by this host")
//     }

//     fn drop(&mut self, error: Resource<webgpu::GpuInternalError>) -> wasmtime::Result<()> {
//         self.table.delete(error)?;
//         Ok(())
//     }
// }
// impl webgpu::HostGpuOutOfMemoryError for WasiWebGpuCtx<'_> {
//     fn new(&mut self, _message: String) -> Resource<webgpu::GpuOutOfMemoryError> {
//         bail!("wasi:webgpu: GpuUncapturedErrorEvent.new is not implemented by this host")
//     }

//     fn message(&mut self, _self_: Resource<webgpu::GpuOutOfMemoryError>) -> String {
//         bail!("wasi:webgpu: GpuUncapturedErrorEvent.message is not implemented by this host")
//     }

//     fn drop(&mut self, error: Resource<webgpu::GpuOutOfMemoryError>) -> wasmtime::Result<()> {
//         self.table.delete(error)?;
//         Ok(())
//     }
// }
// impl webgpu::HostGpuValidationError for WasiWebGpuCtx<'_> {
//     fn new(&mut self, _message: String) -> Resource<webgpu::GpuValidationError> {
//         bail!("wasi:webgpu: GpuUncapturedErrorEvent.new is not implemented by this host")
//     }

//     fn message(&mut self, _self_: Resource<webgpu::GpuValidationError>) -> String {
//         bail!("wasi:webgpu: GpuUncapturedErrorEvent.message is not implemented by this host")
//     }

//     fn drop(&mut self, error: Resource<webgpu::GpuValidationError>) -> wasmtime::Result<()> {
//         self.table.delete(error)?;
//         Ok(())
//     }
// }
impl webgpu::HostGpuError for WasiWebGpuCtx<'_> {
    fn message(&mut self, error: Resource<webgpu::GpuError>) -> wasmtime::Result<String> {
        let error = self.table.get(&error)?;
        Ok(error.message.clone())
    }

    fn kind(
        &mut self,
        error: Resource<webgpu::GpuError>,
    ) -> wasmtime::Result<webgpu::GpuErrorKind> {
        let error = self.table.get(&error)?;
        Ok(error.kind)
    }

    fn drop(&mut self, _error: Resource<webgpu::GpuError>) -> wasmtime::Result<()> {
        self.table.delete(_error)?;
        Ok(())
    }
}
impl webgpu::HostGpuDeviceLostInfo for WasiWebGpuCtx<'_> {
    fn reason(
        &mut self,
        _self_: Resource<webgpu::GpuDeviceLostInfo>,
    ) -> wasmtime::Result<webgpu::GpuDeviceLostReason> {
        bail!("wasi:webgpu: GpuDeviceLostInfo.reason is not implemented by this host")
    }

    fn message(&mut self, _self_: Resource<webgpu::GpuDeviceLostInfo>) -> wasmtime::Result<String> {
        bail!("wasi:webgpu: GpuDeviceLostInfo.message is not implemented by this host")
    }

    fn drop(&mut self, _info: Resource<webgpu::GpuDeviceLostInfo>) -> wasmtime::Result<()> {
        bail!("wasi:webgpu: GpuDeviceLostInfo.drop is not implemented by this host")
    }
}
impl webgpu::HostGpuCanvasContext for WasiWebGpuCtx<'_> {
    fn configure(
        &mut self,
        _self_: Resource<webgpu::GpuCanvasContext>,
        _configuration: webgpu::GpuCanvasConfiguration,
    ) -> wasmtime::Result<()> {
        bail!("wasi:webgpu canvases are not available: this host runs compute only")
    }

    fn get_configuration(
        &mut self,
        _self_: Resource<webgpu::GpuCanvasContext>,
    ) -> wasmtime::Result<Option<webgpu::GpuCanvasConfigurationOwned>> {
        bail!("wasi:webgpu canvases are not available: this host runs compute only")
    }

    fn unconfigure(&mut self, _self_: Resource<webgpu::GpuCanvasContext>) -> wasmtime::Result<()> {
        bail!("wasi:webgpu canvases are not available: this host runs compute only")
    }

    fn get_current_texture(
        &mut self,
        _self_: Resource<webgpu::GpuCanvasContext>,
    ) -> wasmtime::Result<Resource<webgpu::GpuTexture>> {
        bail!("wasi:webgpu canvases are not available: this host runs compute only")
    }

    fn drop(&mut self, rep: Resource<webgpu::GpuCanvasContext>) -> wasmtime::Result<()> {
        self.table.delete(rep)?;
        Ok(())
    }
}
impl webgpu::HostGpuRenderBundle for WasiWebGpuCtx<'_> {
    fn label(&mut self, _self_: Resource<webgpu::GpuRenderBundle>) -> wasmtime::Result<String> {
        // Labels are not kept.
        Ok(String::new())
    }

    fn set_label(
        &mut self,
        _self_: Resource<webgpu::GpuRenderBundle>,
        _label: String,
    ) -> wasmtime::Result<()> {
        // Labels are not kept.
        Ok(())
    }

    fn drop(&mut self, _bundle: Resource<webgpu::GpuRenderBundle>) -> wasmtime::Result<()> {
        let bundle_id = self.table.delete(_bundle)?;
        self.instance.render_bundle_drop(bundle_id);
        Ok(())
    }
}
impl webgpu::HostGpuComputePassEncoder for WasiWebGpuCtx<'_> {
    fn set_pipeline(
        &mut self,
        compute_pass: Resource<webgpu::GpuComputePassEncoder>,
        pipeline: Resource<webgpu::GpuComputePipeline>,
    ) -> wasmtime::Result<()> {
        let instance = Arc::clone(self.instance);
        let pipeline = self.table.get(&pipeline)?.compute_pipeline_id;
        let mut compute_pass = self.table.get_mut(&compute_pass)?.lock();
        let compute_pass = compute_pass.as_mut().ok_or_else(|| wasmtime::format_err!("the compute pass has already ended"))?;
        instance.compute_pass_set_pipeline(compute_pass, pipeline)?;
        Ok(())
    }

    fn dispatch_workgroups(
        &mut self,
        compute_pass: Resource<webgpu::GpuComputePassEncoder>,
        workgroup_count_x: webgpu::GpuSize32,
        workgroup_count_y: Option<webgpu::GpuSize32>,
        workgroup_count_z: Option<webgpu::GpuSize32>,
    ) -> wasmtime::Result<()> {
        let instance = Arc::clone(self.instance);
        let mut compute_pass = self.table.get_mut(&compute_pass)?.lock();
        let compute_pass = compute_pass.as_mut().ok_or_else(|| wasmtime::format_err!("the compute pass has already ended"))?;
        // https://www.w3.org/TR/webgpu/#gpucomputepassencoder
        instance.compute_pass_dispatch_workgroups(
            compute_pass,
            workgroup_count_x,
            workgroup_count_y.unwrap_or(1),
            workgroup_count_z.unwrap_or(1),
        )?;
        Ok(())
    }

    fn dispatch_workgroups_indirect(
        &mut self,
        compute_pass: Resource<webgpu::GpuComputePassEncoder>,
        indirect_buffer: Resource<webgpu::GpuBuffer>,
        indirect_offset: webgpu::GpuSize64,
    ) -> wasmtime::Result<()> {
        let instance = Arc::clone(self.instance);
        let indirect_buffer = self.table.get(&indirect_buffer)?.buffer_id;
        let mut compute_pass = self.table.get_mut(&compute_pass)?.lock();
        let compute_pass = compute_pass.as_mut().ok_or_else(|| wasmtime::format_err!("the compute pass has already ended"))?;
        instance.compute_pass_dispatch_workgroups_indirect(
            compute_pass,
            indirect_buffer,
            indirect_offset,
        )?;
        Ok(())
    }

    fn end(
        &mut self,
        compute_pass: Resource<webgpu::GpuComputePassEncoder>,
    ) -> wasmtime::Result<()> {
        let instance = Arc::clone(self.instance);
        let mut compute_pass = self.table.get_mut(&compute_pass)?.lock();
        let mut compute_pass = compute_pass.take().ok_or_else(|| wasmtime::format_err!("the compute pass has already ended"))?;
        instance.compute_pass_end(&mut compute_pass)?;
        Ok(())
    }

    fn label(
        &mut self,
        _self_: Resource<webgpu::GpuComputePassEncoder>,
    ) -> wasmtime::Result<String> {
        // Labels are not kept.
        Ok(String::new())
    }

    fn set_label(
        &mut self,
        _self_: Resource<webgpu::GpuComputePassEncoder>,
        _label: String,
    ) -> wasmtime::Result<()> {
        // Labels are not kept.
        Ok(())
    }

    fn push_debug_group(
        &mut self,
        compute_pass: Resource<webgpu::GpuComputePassEncoder>,
        group_label: String,
    ) -> wasmtime::Result<()> {
        let instance = Arc::clone(self.instance);
        let mut compute_pass = self.table.get_mut(&compute_pass)?.lock();
        let compute_pass = compute_pass.as_mut().ok_or_else(|| wasmtime::format_err!("the compute pass has already ended"))?;
        instance.compute_pass_push_debug_group(compute_pass, &group_label, 0)?;
        Ok(())
    }

    fn pop_debug_group(
        &mut self,
        compute_pass: Resource<webgpu::GpuComputePassEncoder>,
    ) -> wasmtime::Result<()> {
        let instance = Arc::clone(self.instance);
        let mut compute_pass = self.table.get_mut(&compute_pass)?.lock();
        let compute_pass = compute_pass.as_mut().ok_or_else(|| wasmtime::format_err!("the compute pass has already ended"))?;
        instance.compute_pass_pop_debug_group(compute_pass)?;
        Ok(())
    }

    fn insert_debug_marker(
        &mut self,
        compute_pass: Resource<webgpu::GpuComputePassEncoder>,
        label: String,
    ) -> wasmtime::Result<()> {
        let instance = Arc::clone(self.instance);
        let mut compute_pass = self.table.get_mut(&compute_pass)?.lock();
        let compute_pass = compute_pass.as_mut().ok_or_else(|| wasmtime::format_err!("the compute pass has already ended"))?;
        instance.compute_pass_insert_debug_marker(compute_pass, &label, 0)?;
        Ok(())
    }

    fn set_bind_group(
        &mut self,
        compute_pass: Resource<webgpu::GpuComputePassEncoder>,
        index: webgpu::GpuIndex32,
        bind_group: Option<Resource<webgpu::GpuBindGroup>>,
        dynamic_offsets_data: Option<Vec<webgpu::GpuBufferDynamicOffset>>,
        dynamic_offsets_data_start: Option<webgpu::GpuSize64>,
        dynamic_offsets_data_length: Option<webgpu::GpuSize32>,
    ) -> wasmtime::Result<Result<(), webgpu::SetBindGroupError>> {
        let instance = Arc::clone(self.instance);
        let bind_group = bind_group.map(|bind_group| self.table.get(&bind_group).copied()).transpose()?;
        let mut compute_pass = self.table.get_mut(&compute_pass)?.lock();
        let compute_pass = compute_pass.as_mut().ok_or_else(|| wasmtime::format_err!("the compute pass has already ended"))?;
        // https://www.w3.org/TR/webgpu/#programmable-passes

        let dynamic_offsets = &dynamic_offsets_data.unwrap_or(vec![]);
        let mut dynamic_offsets = &dynamic_offsets[..];
        if let Some(dynamic_offsets_data_start) = dynamic_offsets_data_start {
            let dynamic_offsets_data_start = dynamic_offsets_data_start as usize;
            dynamic_offsets = &dynamic_offsets[dynamic_offsets_data_start..];
        }
        if let Some(dynamic_offsets_data_length) = dynamic_offsets_data_length {
            let dynamic_offsets_data_length = dynamic_offsets_data_length as usize;
            dynamic_offsets = &dynamic_offsets[..dynamic_offsets_data_length];
        }

        instance.compute_pass_set_bind_group(compute_pass, index, bind_group, dynamic_offsets)?;
        Ok(Ok(()))
    }

    fn drop(
        &mut self,
        compute_pass: Resource<webgpu::GpuComputePassEncoder>,
    ) -> wasmtime::Result<()> {
        self.table.delete(compute_pass)?;
        Ok(())
    }
}
// impl webgpu::HostGpuPipelineError for WasiWebGpuCtx<'_> {
//     fn new(
//         &mut self,
//         _message: Option<String>,
//         _options: webgpu::GpuPipelineErrorInit,
//     ) -> Resource<webgpu::GpuPipelineError> {
//         bail!("wasi:webgpu: GpuComputePassEncoder.new is not implemented by this host")
//     }

//     fn reason(
//         &mut self,
//         _self_: Resource<webgpu::GpuPipelineError>,
//     ) -> webgpu::GpuPipelineErrorReason {
//         bail!("wasi:webgpu: GpuComputePassEncoder.reason is not implemented by this host")
//     }

//     fn drop(&mut self, error: Resource<webgpu::GpuPipelineError>) -> wasmtime::Result<()> {
//         self.table.delete(error)?;
//         Ok(())
//     }
// }
impl webgpu::HostGpuCompilationMessage for WasiWebGpuCtx<'_> {
    fn message(
        &mut self,
        _self_: Resource<webgpu::GpuCompilationMessage>,
    ) -> wasmtime::Result<String> {
        bail!("wasi:webgpu: GpuCompilationMessage.message is not implemented by this host")
    }

    fn type_(
        &mut self,
        _self_: Resource<webgpu::GpuCompilationMessage>,
    ) -> wasmtime::Result<webgpu::GpuCompilationMessageType> {
        bail!("wasi:webgpu: GpuCompilationMessage.type_ is not implemented by this host")
    }

    fn line_num(
        &mut self,
        _self_: Resource<webgpu::GpuCompilationMessage>,
    ) -> wasmtime::Result<u64> {
        bail!("wasi:webgpu: GpuCompilationMessage.line_num is not implemented by this host")
    }

    fn line_pos(
        &mut self,
        _self_: Resource<webgpu::GpuCompilationMessage>,
    ) -> wasmtime::Result<u64> {
        bail!("wasi:webgpu: GpuCompilationMessage.line_pos is not implemented by this host")
    }

    fn offset(&mut self, _self_: Resource<webgpu::GpuCompilationMessage>) -> wasmtime::Result<u64> {
        bail!("wasi:webgpu: GpuCompilationMessage.offset is not implemented by this host")
    }

    fn length(&mut self, _self_: Resource<webgpu::GpuCompilationMessage>) -> wasmtime::Result<u64> {
        bail!("wasi:webgpu: GpuCompilationMessage.length is not implemented by this host")
    }

    fn drop(&mut self, _cm: Resource<webgpu::GpuCompilationMessage>) -> wasmtime::Result<()> {
        bail!("wasi:webgpu: GpuCompilationMessage.drop is not implemented by this host")
    }
}
impl webgpu::HostGpuCompilationInfo for WasiWebGpuCtx<'_> {
    fn messages(
        &mut self,
        _self_: Resource<webgpu::GpuCompilationInfo>,
    ) -> wasmtime::Result<Vec<Resource<webgpu::GpuCompilationMessage>>> {
        bail!("wasi:webgpu: GpuCompilationInfo.messages is not implemented by this host")
    }

    fn drop(&mut self, _info: Resource<webgpu::GpuCompilationInfo>) -> wasmtime::Result<()> {
        bail!("wasi:webgpu: GpuCompilationInfo.drop is not implemented by this host")
    }
}
impl webgpu::HostGpuQuerySet for WasiWebGpuCtx<'_> {
    fn destroy(&mut self, _self_: Resource<webgpu::GpuQuerySet>) -> wasmtime::Result<()> {
        // https://github.com/gfx-rs/wgpu/issues/6495
        bail!("wasi:webgpu: GpuQuerySet.destroy is not implemented by this host")
    }

    fn type_(
        &mut self,
        _self_: Resource<webgpu::GpuQuerySet>,
    ) -> wasmtime::Result<webgpu::GpuQueryType> {
        bail!("wasi:webgpu: GpuQuerySet.type_ is not implemented by this host")
    }

    fn count(
        &mut self,
        _self_: Resource<webgpu::GpuQuerySet>,
    ) -> wasmtime::Result<webgpu::GpuSize32Out> {
        bail!("wasi:webgpu: GpuQuerySet.count is not implemented by this host")
    }

    fn label(&mut self, _self_: Resource<webgpu::GpuQuerySet>) -> wasmtime::Result<String> {
        // Labels are not kept.
        Ok(String::new())
    }

    fn set_label(
        &mut self,
        _self_: Resource<webgpu::GpuQuerySet>,
        _label: String,
    ) -> wasmtime::Result<()> {
        // Labels are not kept.
        Ok(())
    }

    fn drop(&mut self, _query_set: Resource<webgpu::GpuQuerySet>) -> wasmtime::Result<()> {
        let query_set_id = self.table.delete(_query_set)?;
        self.instance.query_set_drop(query_set_id);
        Ok(())
    }
}
impl webgpu::HostGpuRenderBundleEncoder for WasiWebGpuCtx<'_> {
    fn finish(
        &mut self,
        bundle_encoder: Resource<webgpu::GpuRenderBundleEncoder>,
        descriptor: Option<webgpu::GpuRenderBundleDescriptor>,
    ) -> wasmtime::Result<Resource<webgpu::GpuRenderBundle>> {
        let instance = Arc::clone(self.instance);
        let descriptor = descriptor
            .map(|d| d.to_core(self.table))
            .unwrap_or(wgpu_types::RenderBundleDescriptor::default());
        let mut bundle_encoder_lock = self.table.get_mut(&bundle_encoder)?.lock();
        let bundle_encoder = bundle_encoder_lock.take().ok_or_else(|| wasmtime::format_err!("the render bundle encoder has already finished"))?;
        drop(bundle_encoder_lock);
        let (render_bundle, err) = instance.render_bundle_encoder_finish(
            bundle_encoder.render_bundle_encoder,
            &descriptor,
            None,
        );
        bundle_encoder.error_handler.handle_possible_error(err);
        Ok(self.table.push(render_bundle)?)
    }

    fn label(
        &mut self,
        _self_: Resource<webgpu::GpuRenderBundleEncoder>,
    ) -> wasmtime::Result<String> {
        // Labels are not kept.
        Ok(String::new())
    }

    fn set_label(
        &mut self,
        _self_: Resource<webgpu::GpuRenderBundleEncoder>,
        _label: String,
    ) -> wasmtime::Result<()> {
        // Labels are not kept.
        Ok(())
    }

    fn push_debug_group(
        &mut self,
        _bundle_encoder: Resource<webgpu::GpuRenderBundleEncoder>,
        _group_label: String,
    ) -> wasmtime::Result<()> {
        bail!("wasi:webgpu: GpuRenderBundleEncoder.push_debug_group is not implemented by this host")
    }

    fn pop_debug_group(
        &mut self,
        _bundle_encoder: Resource<webgpu::GpuRenderBundleEncoder>,
    ) -> wasmtime::Result<()> {
        bail!("wasi:webgpu: GpuRenderBundleEncoder.pop_debug_group is not implemented by this host")
    }

    fn insert_debug_marker(
        &mut self,
        _bundle_encoder: Resource<webgpu::GpuRenderBundleEncoder>,
        _marker_label: String,
    ) -> wasmtime::Result<()> {
        bail!("wasi:webgpu: GpuRenderBundleEncoder.insert_debug_marker is not implemented by this host")
    }

    fn set_bind_group(
        &mut self,
        bundle_encoder: Resource<webgpu::GpuRenderBundleEncoder>,
        index: webgpu::GpuIndex32,
        bind_group: Option<Resource<webgpu::GpuBindGroup>>,
        dynamic_offsets_data: Option<Vec<webgpu::GpuBufferDynamicOffset>>,
        dynamic_offsets_data_start: Option<webgpu::GpuSize64>,
        dynamic_offsets_data_length: Option<webgpu::GpuSize32>,
    ) -> wasmtime::Result<Result<(), webgpu::SetBindGroupError>> {
        let bind_group_id = bind_group.map(|bind_group| self.table.get(&bind_group).copied()).transpose()?;
        let mut bundle_encoder = self.table.get_mut(&bundle_encoder)?.lock();
        let bundle_encoder = bundle_encoder.as_mut().ok_or_else(|| wasmtime::format_err!("the render bundle encoder has already ended"))?;

        // https://www.w3.org/TR/webgpu/#programmable-passes
        let dynamic_offsets = &dynamic_offsets_data.unwrap_or(vec![]);
        let mut dynamic_offsets = &dynamic_offsets[..];
        if let Some(dynamic_offsets_data_start) = dynamic_offsets_data_start {
            let dynamic_offsets_data_start = dynamic_offsets_data_start as usize;
            dynamic_offsets = &dynamic_offsets[dynamic_offsets_data_start..];
        }
        if let Some(dynamic_offsets_data_length) = dynamic_offsets_data_length {
            let dynamic_offsets_data_length = dynamic_offsets_data_length as usize;
            dynamic_offsets = &dynamic_offsets[..dynamic_offsets_data_length];
        }

        unsafe {
            wgpu_core::command::bundle_ffi::wgpu_render_bundle_set_bind_group(
                &mut bundle_encoder.render_bundle_encoder,
                index,
                bind_group_id,
                dynamic_offsets.as_ptr(),
                dynamic_offsets.len(),
            )
        };
        Ok(Ok(()))
    }

    fn set_pipeline(
        &mut self,
        bundle_encoder: Resource<webgpu::GpuRenderBundleEncoder>,
        pipeline: Resource<RenderPipeline>,
    ) -> wasmtime::Result<()> {
        let pipeline = self.table.get(&pipeline)?;
        let pipeline_id = pipeline.render_pipeline_id;
        let mut bundle_encoder = self.table.get_mut(&bundle_encoder)?.lock();
        let bundle_encoder = bundle_encoder.as_mut().ok_or_else(|| wasmtime::format_err!("the render bundle encoder has already ended"))?;
        wgpu_core::command::bundle_ffi::wgpu_render_bundle_set_pipeline(
            &mut bundle_encoder.render_bundle_encoder,
            pipeline_id,
        );
        Ok(())
    }

    fn set_index_buffer(
        &mut self,
        bundle_encoder: Resource<webgpu::GpuRenderBundleEncoder>,
        buffer: Resource<webgpu::GpuBuffer>,
        index_format: webgpu::GpuIndexFormat,
        offset: Option<webgpu::GpuSize64>,
        size: Option<webgpu::GpuSize64>,
    ) -> wasmtime::Result<()> {
        let buffer_id = self.table.get(&buffer)?.buffer_id;
        let mut bundle_encoder = self.table.get_mut(&bundle_encoder)?.lock();
        let bundle_encoder = bundle_encoder.as_mut().ok_or_else(|| wasmtime::format_err!("the render bundle encoder has already ended"))?;
        // https://www.w3.org/TR/webgpu/#gpurendercommandsmixin
        wgpu_core::command::bundle_ffi::wgpu_render_bundle_set_index_buffer(
            &mut bundle_encoder.render_bundle_encoder,
            buffer_id,
            index_format.into(),
            offset.unwrap_or(0),
            size.map(|s| NonZeroU64::new(s).expect("Size can't be zero")),
        );
        Ok(())
    }

    fn set_vertex_buffer(
        &mut self,
        bundle_encoder: Resource<webgpu::GpuRenderBundleEncoder>,
        slot: webgpu::GpuIndex32,
        buffer: Option<Resource<webgpu::GpuBuffer>>,
        offset: Option<webgpu::GpuSize64>,
        size: Option<webgpu::GpuSize64>,
    ) -> wasmtime::Result<()> {
        let buffer = buffer.expect("TODO: Null buffers not yet supported in wgpu");
        let buffer_id = self.table.get(&buffer)?.buffer_id;
        let mut bundle_encoder = self.table.get_mut(&bundle_encoder)?.lock();
        let bundle_encoder = bundle_encoder.as_mut().ok_or_else(|| wasmtime::format_err!("the render bundle encoder has already ended"))?;
        // https://www.w3.org/TR/webgpu/#gpurendercommandsmixin
        wgpu_core::command::bundle_ffi::wgpu_render_bundle_set_vertex_buffer(
            &mut bundle_encoder.render_bundle_encoder,
            slot,
            buffer_id,
            offset.unwrap_or(0),
            size.map(|s| NonZeroU64::new(s).expect("Size can't be zero")),
        );
        Ok(())
    }

    fn draw(
        &mut self,
        bundle_encoder: Resource<webgpu::GpuRenderBundleEncoder>,
        vertex_count: webgpu::GpuSize32,
        instance_count: Option<webgpu::GpuSize32>,
        first_vertex: Option<webgpu::GpuSize32>,
        first_instance: Option<webgpu::GpuSize32>,
    ) -> wasmtime::Result<()> {
        let mut bundle_encoder = self.table.get_mut(&bundle_encoder)?.lock();
        let bundle_encoder = bundle_encoder.as_mut().ok_or_else(|| wasmtime::format_err!("the render bundle encoder has already ended"))?;
        // https://www.w3.org/TR/webgpu/#gpurendercommandsmixin
        wgpu_core::command::bundle_ffi::wgpu_render_bundle_draw(
            &mut bundle_encoder.render_bundle_encoder,
            vertex_count,
            instance_count.unwrap_or(1),
            first_vertex.unwrap_or(0),
            first_instance.unwrap_or(0),
        );
        Ok(())
    }

    fn draw_indexed(
        &mut self,
        bundle_encoder: Resource<webgpu::GpuRenderBundleEncoder>,
        index_count: webgpu::GpuSize32,
        instance_count: Option<webgpu::GpuSize32>,
        first_index: Option<webgpu::GpuSize32>,
        base_vertex: Option<webgpu::GpuSignedOffset32>,
        first_instance: Option<webgpu::GpuSize32>,
    ) -> wasmtime::Result<()> {
        let mut bundle_encoder = self.table.get_mut(&bundle_encoder)?.lock();
        let bundle_encoder = bundle_encoder.as_mut().ok_or_else(|| wasmtime::format_err!("the render bundle encoder has already ended"))?;
        // https://www.w3.org/TR/webgpu/#gpurendercommandsmixin
        wgpu_core::command::bundle_ffi::wgpu_render_bundle_draw_indexed(
            &mut bundle_encoder.render_bundle_encoder,
            index_count,
            instance_count.unwrap_or(1),
            first_index.unwrap_or(0),
            base_vertex.unwrap_or(0),
            first_instance.unwrap_or(0),
        );
        Ok(())
    }

    fn draw_indirect(
        &mut self,
        bundle_encoder: Resource<webgpu::GpuRenderBundleEncoder>,
        indirect_buffer: Resource<webgpu::GpuBuffer>,
        indirect_offset: webgpu::GpuSize64,
    ) -> wasmtime::Result<()> {
        let indirect_buffer = self.table.get(&indirect_buffer)?.buffer_id;
        let mut bundle_encoder = self.table.get_mut(&bundle_encoder)?.lock();
        let bundle_encoder = bundle_encoder.as_mut().ok_or_else(|| wasmtime::format_err!("the render bundle encoder has already ended"))?;
        wgpu_core::command::bundle_ffi::wgpu_render_bundle_draw_indirect(
            &mut bundle_encoder.render_bundle_encoder,
            indirect_buffer,
            indirect_offset,
        );
        Ok(())
    }

    fn draw_indexed_indirect(
        &mut self,
        bundle_encoder: Resource<webgpu::GpuRenderBundleEncoder>,
        indirect_buffer: Resource<webgpu::GpuBuffer>,
        indirect_offset: webgpu::GpuSize64,
    ) -> wasmtime::Result<()> {
        let indirect_buffer = self.table.get(&indirect_buffer)?.buffer_id;
        let mut bundle_encoder = self.table.get_mut(&bundle_encoder)?.lock();
        let bundle_encoder = bundle_encoder.as_mut().ok_or_else(|| wasmtime::format_err!("the render bundle encoder has already ended"))?;
        wgpu_core::command::bundle_ffi::wgpu_render_bundle_draw_indexed_indirect(
            &mut bundle_encoder.render_bundle_encoder,
            indirect_buffer,
            indirect_offset,
        );
        Ok(())
    }

    fn drop(&mut self, encoder: Resource<webgpu::GpuRenderBundleEncoder>) -> wasmtime::Result<()> {
        self.table.delete(encoder)?;
        Ok(())
    }
}
impl webgpu::HostGpuComputePipeline for WasiWebGpuCtx<'_> {
    fn label(&mut self, _self_: Resource<webgpu::GpuComputePipeline>) -> wasmtime::Result<String> {
        // Labels are not kept.
        Ok(String::new())
    }

    fn set_label(
        &mut self,
        _self_: Resource<webgpu::GpuComputePipeline>,
        _label: String,
    ) -> wasmtime::Result<()> {
        // Labels are not kept.
        Ok(())
    }

    fn get_bind_group_layout(
        &mut self,
        compute_pipeline: Resource<webgpu::GpuComputePipeline>,
        index: u32,
    ) -> wasmtime::Result<Resource<webgpu::GpuBindGroupLayout>> {
        let pipeline = self.table.get(&compute_pipeline)?;
        let pipeline_id = pipeline.compute_pipeline_id;
        let error_handler = Arc::clone(&pipeline.error_handler);
        let (bind_group_layout, err) =
            self.instance
                .compute_pipeline_get_bind_group_layout(pipeline_id, index, None);
        error_handler.handle_possible_error(err);
        Ok(self.table.push(bind_group_layout)?)
    }

    fn drop(&mut self, pipeline: Resource<webgpu::GpuComputePipeline>) -> wasmtime::Result<()> {
        let pipeline = self.table.delete(pipeline)?;
        self.instance
            .compute_pipeline_drop(pipeline.compute_pipeline_id);
        Ok(())
    }
}
impl webgpu::HostGpuBindGroup for WasiWebGpuCtx<'_> {
    fn label(&mut self, _self_: Resource<webgpu::GpuBindGroup>) -> wasmtime::Result<String> {
        // Labels are not kept.
        Ok(String::new())
    }

    fn set_label(
        &mut self,
        _self_: Resource<webgpu::GpuBindGroup>,
        _label: String,
    ) -> wasmtime::Result<()> {
        // Labels are not kept.
        Ok(())
    }

    fn drop(&mut self, bind_group: Resource<webgpu::GpuBindGroup>) -> wasmtime::Result<()> {
        let bind_group_id = self.table.delete(bind_group)?;
        self.instance.bind_group_drop(bind_group_id);
        Ok(())
    }
}
impl webgpu::HostGpuPipelineLayout for WasiWebGpuCtx<'_> {
    fn label(&mut self, _self_: Resource<webgpu::GpuPipelineLayout>) -> wasmtime::Result<String> {
        // Labels are not kept.
        Ok(String::new())
    }

    fn set_label(
        &mut self,
        _self_: Resource<webgpu::GpuPipelineLayout>,
        _label: String,
    ) -> wasmtime::Result<()> {
        // Labels are not kept.
        Ok(())
    }

    fn drop(&mut self, layout: Resource<webgpu::GpuPipelineLayout>) -> wasmtime::Result<()> {
        let layout_id = self.table.delete(layout)?;
        self.instance.pipeline_layout_drop(layout_id);
        Ok(())
    }
}
impl webgpu::HostGpuBindGroupLayout for WasiWebGpuCtx<'_> {
    fn label(&mut self, _self_: Resource<webgpu::GpuBindGroupLayout>) -> wasmtime::Result<String> {
        // Labels are not kept.
        Ok(String::new())
    }

    fn set_label(
        &mut self,
        _self_: Resource<webgpu::GpuBindGroupLayout>,
        _label: String,
    ) -> wasmtime::Result<()> {
        // Labels are not kept.
        Ok(())
    }

    fn drop(&mut self, layout: Resource<webgpu::GpuBindGroupLayout>) -> wasmtime::Result<()> {
        let layout_id = self.table.delete(layout)?;
        self.instance.bind_group_layout_drop(layout_id);
        Ok(())
    }
}

impl webgpu::HostGpuSampler for WasiWebGpuCtx<'_> {
    fn label(&mut self, _self_: Resource<webgpu::GpuSampler>) -> wasmtime::Result<String> {
        // Labels are not kept.
        Ok(String::new())
    }

    fn set_label(
        &mut self,
        _self_: Resource<webgpu::GpuSampler>,
        _label: String,
    ) -> wasmtime::Result<()> {
        // Labels are not kept.
        Ok(())
    }

    fn drop(&mut self, sampler: Resource<webgpu::GpuSampler>) -> wasmtime::Result<()> {
        let sampler_id = self.table.delete(sampler)?;
        self.instance.sampler_drop(sampler_id);
        Ok(())
    }
}

impl webgpu::HostGpuBuffer for WasiWebGpuCtx<'_> {
    fn size(
        &mut self,
        buffer: Resource<webgpu::GpuBuffer>,
    ) -> wasmtime::Result<webgpu::GpuSize64Out> {
        let buffer = self.table.get(&buffer)?;
        Ok(buffer.size)
    }

    fn usage(
        &mut self,
        buffer: Resource<webgpu::GpuBuffer>,
    ) -> wasmtime::Result<webgpu::GpuFlagsConstant> {
        let buffer = self.table.get(&buffer)?;
        Ok(buffer.usage.bits())
    }

    fn map_state(
        &mut self,
        buffer: Resource<webgpu::GpuBuffer>,
    ) -> wasmtime::Result<webgpu::GpuBufferMapState> {
        let buffer = self.table.get(&buffer)?;
        Ok(buffer.map_state)
    }

    fn map_async(
        &mut self,
        buffer: Resource<webgpu::GpuBuffer>,
        mode: webgpu::GpuMapModeFlags,
        offset: Option<webgpu::GpuSize64>,
        size: Option<webgpu::GpuSize64>,
    ) -> wasmtime::Result<Result<(), webgpu::MapAsyncError>> {
        // Synchronous: the call returns once the map has landed or failed.
        // A WASI 0.2 guest has no other way to be told, and the store stays
        // a synchronous one.
        let instance = Arc::clone(self.instance);
        let buffer = self.table.get_mut(&buffer)?;
        let buffer_id = buffer.buffer_id;
        // source: https://www.w3.org/TR/webgpu/#typedefdef-gpumapmodeflags
        let host = match mode {
            1 => wgpu_core::device::HostMap::Read,
            2 => wgpu_core::device::HostMap::Write,
            _ => {
                return Ok(Err(webgpu::MapAsyncError {
                    kind: webgpu::MapAsyncErrorKind::OperationError,
                    message: format!("map mode {mode} is neither READ nor WRITE"),
                }))
            }
        };
        let (sender, receiver) = std::sync::mpsc::channel();
        let op = wgpu_core::resource::BufferMapOperation {
            host,
            callback: Some(Box::new(move |result| {
                let _ = sender.send(result);
            })),
        };
        // https://www.w3.org/TR/webgpu/#gpubuffer
        if let Err(e) = instance.buffer_map_async(buffer_id, offset.unwrap_or(0), size, op) {
            return Ok(Err(webgpu::MapAsyncError {
                kind: webgpu::MapAsyncErrorKind::OperationError,
                message: e.to_string(),
            }));
        }
        buffer.map_state = webgpu::GpuBufferMapState::Pending;
        let result = loop {
            if let Ok(result) = receiver.try_recv() {
                break result;
            }
            // TODO: only poll this buffer's device.
            if let Err(e) = instance.poll_all_devices(true) {
                buffer.map_state = webgpu::GpuBufferMapState::Unmapped;
                return Ok(Err(webgpu::MapAsyncError {
                    kind: webgpu::MapAsyncErrorKind::AbortError,
                    message: e.to_string(),
                }));
            }
        };
        match result {
            Ok(()) => {
                buffer.map_state = webgpu::GpuBufferMapState::Mapped;
                Ok(Ok(()))
            }
            Err(e) => {
                buffer.map_state = webgpu::GpuBufferMapState::Unmapped;
                Ok(Err(webgpu::MapAsyncError {
                    kind: webgpu::MapAsyncErrorKind::OperationError,
                    message: e.to_string(),
                }))
            }
        }
    }

    fn get_mapped_range_get_with_copy(
        &mut self,
        buffer: Resource<webgpu::GpuBuffer>,
        offset: Option<webgpu::GpuSize64>,
        size: Option<webgpu::GpuSize64>,
    ) -> wasmtime::Result<Result<Vec<u8>, webgpu::GetMappedRangeError>> {
        let buffer = self.table.get(&buffer)?;
        if buffer.map_state != webgpu::GpuBufferMapState::Mapped {
            // https://www.w3.org/TR/webgpu/#dom-gpubuffer-getmappedrange
            return Ok(Err(webgpu::GetMappedRangeError {
                kind: webgpu::GetMappedRangeErrorKind::OperationError,
                message: "the buffer is not mapped".to_string(),
            }));
        }
        let buffer_id = buffer.buffer_id;
        let (ptr, len) = self
            .instance
            // https://www.w3.org/TR/webgpu/#gpubuffer
            .buffer_get_mapped_range(buffer_id, offset.unwrap_or(0), size)?;
        let data = unsafe { slice::from_raw_parts(ptr.as_ptr(), len as usize) };
        let mut output = vec![0; len as usize];
        output.copy_from_slice(data);
        Ok(Ok(output))
    }

    fn get_mapped_range_set_with_copy(
        &mut self,
        buffer: Resource<webgpu::GpuBuffer>,
        data: Vec<u8>,
        offset: Option<webgpu::GpuSize64>,
        size: Option<webgpu::GpuSize64>,
    ) -> wasmtime::Result<Result<(), webgpu::GetMappedRangeError>> {
        let buffer = self.table.get(&buffer)?;
        if buffer.map_state != webgpu::GpuBufferMapState::Mapped {
            // https://www.w3.org/TR/webgpu/#dom-gpubuffer-getmappedrange
            return Ok(Err(webgpu::GetMappedRangeError {
                kind: webgpu::GetMappedRangeErrorKind::OperationError,
                message: "the buffer is not mapped".to_string(),
            }));
        }
        let buffer_id = buffer.buffer_id;
        let (ptr, len) = self
            .instance
            // https://www.w3.org/TR/webgpu/#gpubuffer
            .buffer_get_mapped_range(buffer_id, offset.unwrap_or(0), size)?;
        let buffer = unsafe { slice::from_raw_parts_mut(ptr.as_ptr(), len as usize) };
        buffer.copy_from_slice(&data);
        Ok(Ok(()))
    }

    fn unmap(
        &mut self,
        buffer: Resource<webgpu::GpuBuffer>,
    ) -> wasmtime::Result<Result<(), webgpu::UnmapError>> {
        let instance = Arc::clone(self.instance);
        let buffer = self.table.get_mut(&buffer)?;
        instance.buffer_unmap(buffer.buffer_id)?;
        buffer.map_state = webgpu::GpuBufferMapState::Unmapped;
        Ok(Ok(()))
    }

    fn destroy(&mut self, buffer: Resource<webgpu::GpuBuffer>) -> wasmtime::Result<()> {
        let buffer_id = self.table.get_mut(&buffer)?.buffer_id;
        self.instance.buffer_destroy(buffer_id);
        Ok(())
    }

    fn label(&mut self, _self_: Resource<webgpu::GpuBuffer>) -> wasmtime::Result<String> {
        // Labels are not kept.
        Ok(String::new())
    }

    fn set_label(
        &mut self,
        _self_: Resource<webgpu::GpuBuffer>,
        _label: String,
    ) -> wasmtime::Result<()> {
        // Labels are not kept.
        Ok(())
    }

    fn drop(&mut self, buffer: Resource<webgpu::GpuBuffer>) -> wasmtime::Result<()> {
        let buffer = self.table.delete(buffer)?;
        self.instance.buffer_drop(buffer.buffer_id);
        Ok(())
    }
}
impl webgpu::HostGpu for WasiWebGpuCtx<'_> {
    fn request_adapter(
        &mut self,
        _self_: Resource<webgpu::Gpu>,
        options: Option<webgpu::GpuRequestAdapterOptions>,
    ) -> wasmtime::Result<Option<Resource<webgpu::GpuAdapter>>> {
        let options = options
            .map(|o| o.to_core(self.table))
            .unwrap_or(wgpu_types::RequestAdapterOptions::default());
        // Vulkan and Metal before D3D12 and GL: on Windows the same GPU shows
        // up under Vulkan and D3D12, and wgpu's D3D12 reports no subgroups
        // unless it can load DXC. A backend the instance was not created
        // with finds nothing and the next is tried.
        let preference = [
            wgpu_types::Backends::VULKAN,
            wgpu_types::Backends::METAL,
            wgpu_types::Backends::DX12,
            wgpu_types::Backends::GL,
        ];
        let mut adapter = self.instance.request_adapter(&options, preference[0], None);
        for backends in &preference[1..] {
            if adapter.is_ok() {
                break;
            }
            adapter = self.instance.request_adapter(&options, *backends, None);
        }
        match adapter {
            Ok(adapter) => {
                let adapter = Arc::new(adapter);
                let adapter = self.table.push(adapter)?;
                Ok(Some(adapter))
            }
            Err(wgpu_types::RequestAdapterError::NotFound { .. }) => {
                log::warn!("GPU adapter not found");
                Ok(None)
            }
            Err(e) => {
                log::warn!("Failed to get gpu adapter: {e:?}");
                bail!("Error when trying to get GPU adapter");
            }
        }
    }

    fn get_preferred_canvas_format(
        &mut self,
        _gpu: Resource<webgpu::Gpu>,
    ) -> wasmtime::Result<webgpu::GpuTextureFormat> {
        Ok(PREFERRED_CANVAS_FORMAT)
    }

    fn wgsl_language_features(
        &mut self,
        _self_: Resource<webgpu::Gpu>,
    ) -> wasmtime::Result<Resource<webgpu::WgslLanguageFeatures>> {
        Ok(self.table.push(WgslLanguageFeatures::new())?)
    }

    fn drop(&mut self, _gpu: Resource<webgpu::Gpu>) -> wasmtime::Result<()> {
        // not actually a resource in the table
        Ok(())
    }
}
impl webgpu::HostGpuAdapterInfo for WasiWebGpuCtx<'_> {
    // TODO: more real values here
    // take ideas from https://bugzilla.mozilla.org/show_bug.cgi?id=1831994
    // keep an eye on https://github.com/gfx-rs/wgpu/issues/8649
    fn vendor(
        &mut self,
        adapter_info: Resource<webgpu::GpuAdapterInfo>,
    ) -> wasmtime::Result<String> {
        // WebGPU's vendor is a name; wgpu knows the PCI vendor id.
        let adapter_info = self.table.get(&adapter_info)?;
        Ok(format!("0x{:04x}", adapter_info.vendor))
    }

    fn architecture(
        &mut self,
        _adapter_info: Resource<webgpu::GpuAdapterInfo>,
    ) -> wasmtime::Result<String> {
        // TODO: implement real architecture
        Ok(String::new())
    }

    fn device(
        &mut self,
        adapter_info: Resource<webgpu::GpuAdapterInfo>,
    ) -> wasmtime::Result<String> {
        // The adapter's name, which is what WebGPU's `device` describes;
        // wgpu's `device` is the PCI device id.
        let adapter_info = self.table.get(&adapter_info)?;
        Ok(adapter_info.name.clone())
    }

    fn description(
        &mut self,
        adapter_info: Resource<webgpu::GpuAdapterInfo>,
    ) -> wasmtime::Result<String> {
        // Free text in WebGPU: the name, the backend wgpu drives it through,
        // and the driver.
        let info = self.table.get(&adapter_info)?;
        Ok(format!(
            "{} ({:?}, {} {})",
            info.name, info.backend, info.driver, info.driver_info
        ))
    }

    fn subgroup_min_size(
        &mut self,
        adapter_info: Resource<webgpu::GpuAdapterInfo>,
    ) -> wasmtime::Result<u32> {
        let adapter_info = self.table.get(&adapter_info)?;
        Ok(adapter_info.subgroup_min_size)
    }

    fn subgroup_max_size(
        &mut self,
        adapter_info: Resource<webgpu::GpuAdapterInfo>,
    ) -> wasmtime::Result<u32> {
        let adapter_info = self.table.get(&adapter_info)?;
        Ok(adapter_info.subgroup_max_size)
    }

    fn drop(&mut self, info: Resource<webgpu::GpuAdapterInfo>) -> wasmtime::Result<()> {
        self.table.delete(info)?;
        Ok(())
    }
}
impl webgpu::HostWgslLanguageFeatures for WasiWebGpuCtx<'_> {
    fn has(
        &mut self,
        features: Resource<webgpu::WgslLanguageFeatures>,
        key: String,
    ) -> wasmtime::Result<bool> {
        Ok(self.table.get(&features)?.has(&key))
    }

    fn drop(&mut self, features: Resource<webgpu::WgslLanguageFeatures>) -> wasmtime::Result<()> {
        self.table.delete(features)?;
        Ok(())
    }
}
impl webgpu::HostGpuSupportedFeatures for WasiWebGpuCtx<'_> {
    fn has(
        &mut self,
        features: Resource<webgpu::GpuSupportedFeatures>,
        query: String,
    ) -> wasmtime::Result<bool> {
        let features = self.table.get(&features)?;
        // TODO: disable the ones not present in the wgpu yet.
        Ok(match query.as_str() {
            "core-features-and-limits" => {
                // TODO: enable once wgpu does
                // features.contains(wgpu_types::Features::CORE_FEATURES_AND_LIMITS)
                false
            }
            "depth-clip-control" => features.contains(wgpu_types::Features::DEPTH_CLIP_CONTROL),
            "depth32float-stencil8" => {
                features.contains(wgpu_types::Features::DEPTH32FLOAT_STENCIL8)
            }
            "texture-compression-bc" => {
                features.contains(wgpu_types::Features::TEXTURE_COMPRESSION_BC)
            }
            "texture-compression-bc-sliced-3d" => {
                features.contains(wgpu_types::Features::TEXTURE_COMPRESSION_BC_SLICED_3D)
            }
            "texture-compression-etc2" => {
                features.contains(wgpu_types::Features::TEXTURE_COMPRESSION_ETC2)
            }
            "texture-compression-astc" => {
                features.contains(wgpu_types::Features::TEXTURE_COMPRESSION_ASTC)
            }
            "texture-compression-astc-sliced-3d" => {
                features.contains(wgpu_types::Features::TEXTURE_COMPRESSION_ASTC_SLICED_3D)
            }
            "timestamp-query" => features.contains(wgpu_types::Features::TIMESTAMP_QUERY),
            "indirect-first-instance" => {
                features.contains(wgpu_types::Features::INDIRECT_FIRST_INSTANCE)
            }
            "shader-f16" => features.contains(wgpu_types::Features::SHADER_F16),
            "rg11b10ufloat-renderable" => {
                features.contains(wgpu_types::Features::RG11B10UFLOAT_RENDERABLE)
            }
            "bgra8unorm-storage" => features.contains(wgpu_types::Features::BGRA8UNORM_STORAGE),
            "float32-filterable" => features.contains(wgpu_types::Features::FLOAT32_FILTERABLE),
            "float32-blendable" => features.contains(wgpu_types::Features::FLOAT32_BLENDABLE),
            "clip-distances" => features.contains(wgpu_types::Features::CLIP_DISTANCES),
            "dual-source-blending" => features.contains(wgpu_types::Features::DUAL_SOURCE_BLENDING),
            // wgpu's own SUBGROUP stands in for the standard feature it does
            // not have yet; see the device descriptor conversion.
            "subgroups" => features.features_wgpu.contains(wgpu_types::FeaturesWGPU::SUBGROUP),
            "texture-formats-tier1" => {
                // TODO: enable once wgpu does
                // features.contains(wgpu_types::Features::TEXTURE_FORMATS_TIER1)
                false
            }
            "texture-formats-tier2" => {
                // TODO: enable once wgpu does
                // features.contains(wgpu_types::Features::TEXTURE_FORMATS_TIER2)
                false
            }
            "primitive-index" => features.contains(wgpu_types::Features::PRIMITIVE_INDEX),
            "texture-component-swizzle" => {
                // TODO: enable once wgpu does
                // features.contains(wgpu_types::Features::TEXTURE_COMPONENT_SWIZZLE)
                false
            }
            name => {
                log::warn!("unknown feature name: {}", name);
                false
            }
        })
    }

    fn drop(&mut self, features: Resource<webgpu::GpuSupportedFeatures>) -> wasmtime::Result<()> {
        self.table.delete(features)?;
        Ok(())
    }
}
impl webgpu::HostGpuSupportedLimits for WasiWebGpuCtx<'_> {
    fn max_texture_dimension1_d(
        &mut self,
        limits: Resource<webgpu::GpuSupportedLimits>,
    ) -> wasmtime::Result<u32> {
        let limits = self.table.get(&limits)?;
        Ok(limits.max_texture_dimension_1d)
    }

    fn max_texture_dimension2_d(
        &mut self,
        limits: Resource<webgpu::GpuSupportedLimits>,
    ) -> wasmtime::Result<u32> {
        let limits = self.table.get(&limits)?;
        Ok(limits.max_texture_dimension_2d)
    }

    fn max_texture_dimension3_d(
        &mut self,
        limits: Resource<webgpu::GpuSupportedLimits>,
    ) -> wasmtime::Result<u32> {
        let limits = self.table.get(&limits)?;
        Ok(limits.max_texture_dimension_3d)
    }

    fn max_texture_array_layers(
        &mut self,
        limits: Resource<webgpu::GpuSupportedLimits>,
    ) -> wasmtime::Result<u32> {
        let limits = self.table.get(&limits)?;
        Ok(limits.max_texture_array_layers)
    }

    fn max_bind_groups(
        &mut self,
        limits: Resource<webgpu::GpuSupportedLimits>,
    ) -> wasmtime::Result<u32> {
        let limits = self.table.get(&limits)?;
        Ok(limits.max_bind_groups)
    }

    fn max_bind_groups_plus_vertex_buffers(
        &mut self,
        _limits: Resource<webgpu::GpuSupportedLimits>,
    ) -> wasmtime::Result<u32> {
        // Not present in wgpu yet so rely on spec default
        // https://www.w3.org/TR/webgpu/#dom-supported-limits-maxbindgroupsplusvertexbuffers
        // TODO: take value from wgpu once implemented there
        Ok(24)
    }

    fn max_bindings_per_bind_group(
        &mut self,
        limits: Resource<webgpu::GpuSupportedLimits>,
    ) -> wasmtime::Result<u32> {
        let limits = self.table.get(&limits)?;
        Ok(limits.max_bindings_per_bind_group)
    }

    fn max_dynamic_uniform_buffers_per_pipeline_layout(
        &mut self,
        limits: Resource<webgpu::GpuSupportedLimits>,
    ) -> wasmtime::Result<u32> {
        let limits = self.table.get(&limits)?;
        Ok(limits.max_dynamic_uniform_buffers_per_pipeline_layout)
    }

    fn max_dynamic_storage_buffers_per_pipeline_layout(
        &mut self,
        limits: Resource<webgpu::GpuSupportedLimits>,
    ) -> wasmtime::Result<u32> {
        let limits = self.table.get(&limits)?;
        Ok(limits.max_dynamic_storage_buffers_per_pipeline_layout)
    }

    fn max_sampled_textures_per_shader_stage(
        &mut self,
        limits: Resource<webgpu::GpuSupportedLimits>,
    ) -> wasmtime::Result<u32> {
        let limits = self.table.get(&limits)?;
        Ok(limits.max_sampled_textures_per_shader_stage)
    }

    fn max_samplers_per_shader_stage(
        &mut self,
        limits: Resource<webgpu::GpuSupportedLimits>,
    ) -> wasmtime::Result<u32> {
        let limits = self.table.get(&limits)?;
        Ok(limits.max_samplers_per_shader_stage)
    }

    fn max_storage_buffers_per_shader_stage(
        &mut self,
        limits: Resource<webgpu::GpuSupportedLimits>,
    ) -> wasmtime::Result<u32> {
        let limits = self.table.get(&limits)?;
        Ok(limits.max_storage_buffers_per_shader_stage)
    }

    fn max_storage_textures_per_shader_stage(
        &mut self,
        limits: Resource<webgpu::GpuSupportedLimits>,
    ) -> wasmtime::Result<u32> {
        let limits = self.table.get(&limits)?;
        Ok(limits.max_storage_textures_per_shader_stage)
    }

    fn max_uniform_buffers_per_shader_stage(
        &mut self,
        limits: Resource<webgpu::GpuSupportedLimits>,
    ) -> wasmtime::Result<u32> {
        let limits = self.table.get(&limits)?;
        Ok(limits.max_uniform_buffers_per_shader_stage)
    }

    fn max_uniform_buffer_binding_size(
        &mut self,
        limits: Resource<webgpu::GpuSupportedLimits>,
    ) -> wasmtime::Result<u64> {
        let limits = self.table.get(&limits)?;
        Ok(limits.max_uniform_buffer_binding_size)
    }

    fn max_storage_buffer_binding_size(
        &mut self,
        limits: Resource<webgpu::GpuSupportedLimits>,
    ) -> wasmtime::Result<u64> {
        let limits = self.table.get(&limits)?;
        Ok(limits.max_storage_buffer_binding_size)
    }

    fn min_uniform_buffer_offset_alignment(
        &mut self,
        limits: Resource<webgpu::GpuSupportedLimits>,
    ) -> wasmtime::Result<u32> {
        let limits = self.table.get(&limits)?;
        Ok(limits.min_uniform_buffer_offset_alignment)
    }

    fn min_storage_buffer_offset_alignment(
        &mut self,
        limits: Resource<webgpu::GpuSupportedLimits>,
    ) -> wasmtime::Result<u32> {
        let limits = self.table.get(&limits)?;
        Ok(limits.min_storage_buffer_offset_alignment)
    }

    fn max_vertex_buffers(
        &mut self,
        limits: Resource<webgpu::GpuSupportedLimits>,
    ) -> wasmtime::Result<u32> {
        let limits = self.table.get(&limits)?;
        Ok(limits.max_vertex_buffers)
    }

    fn max_buffer_size(
        &mut self,
        limits: Resource<webgpu::GpuSupportedLimits>,
    ) -> wasmtime::Result<u64> {
        let limits = self.table.get(&limits)?;
        Ok(limits.max_buffer_size)
    }

    fn max_vertex_attributes(
        &mut self,
        limits: Resource<webgpu::GpuSupportedLimits>,
    ) -> wasmtime::Result<u32> {
        let limits = self.table.get(&limits)?;
        Ok(limits.max_vertex_attributes)
    }

    fn max_vertex_buffer_array_stride(
        &mut self,
        limits: Resource<webgpu::GpuSupportedLimits>,
    ) -> wasmtime::Result<u32> {
        let limits = self.table.get(&limits)?;
        Ok(limits.max_vertex_buffer_array_stride)
    }

    fn max_inter_stage_shader_variables(
        &mut self,
        limits: Resource<webgpu::GpuSupportedLimits>,
    ) -> wasmtime::Result<u32> {
        let limits = self.table.get(&limits)?;
        Ok(limits.max_inter_stage_shader_variables)
    }

    fn max_color_attachments(
        &mut self,
        limits: Resource<webgpu::GpuSupportedLimits>,
    ) -> wasmtime::Result<u32> {
        let limits = self.table.get(&limits)?;
        Ok(limits.max_color_attachments)
    }

    fn max_color_attachment_bytes_per_sample(
        &mut self,
        limits: Resource<webgpu::GpuSupportedLimits>,
    ) -> wasmtime::Result<u32> {
        let limits = self.table.get(&limits)?;
        Ok(limits.max_color_attachment_bytes_per_sample)
    }

    fn max_compute_workgroup_storage_size(
        &mut self,
        limits: Resource<webgpu::GpuSupportedLimits>,
    ) -> wasmtime::Result<u32> {
        let limits = self.table.get(&limits)?;
        Ok(limits.max_compute_workgroup_storage_size)
    }

    fn max_compute_invocations_per_workgroup(
        &mut self,
        limits: Resource<webgpu::GpuSupportedLimits>,
    ) -> wasmtime::Result<u32> {
        let limits = self.table.get(&limits)?;
        Ok(limits.max_compute_invocations_per_workgroup)
    }

    fn max_compute_workgroup_size_x(
        &mut self,
        limits: Resource<webgpu::GpuSupportedLimits>,
    ) -> wasmtime::Result<u32> {
        let limits = self.table.get(&limits)?;
        Ok(limits.max_compute_workgroup_size_x)
    }

    fn max_compute_workgroup_size_y(
        &mut self,
        limits: Resource<webgpu::GpuSupportedLimits>,
    ) -> wasmtime::Result<u32> {
        let limits = self.table.get(&limits)?;
        Ok(limits.max_compute_workgroup_size_y)
    }

    fn max_compute_workgroup_size_z(
        &mut self,
        limits: Resource<webgpu::GpuSupportedLimits>,
    ) -> wasmtime::Result<u32> {
        let limits = self.table.get(&limits)?;
        Ok(limits.max_compute_workgroup_size_z)
    }

    fn max_compute_workgroups_per_dimension(
        &mut self,
        limits: Resource<webgpu::GpuSupportedLimits>,
    ) -> wasmtime::Result<u32> {
        let limits = self.table.get(&limits)?;
        Ok(limits.max_compute_workgroups_per_dimension)
    }

    fn drop(&mut self, limits: Resource<webgpu::GpuSupportedLimits>) -> wasmtime::Result<()> {
        self.table.delete(limits)?;
        Ok(())
    }
}
