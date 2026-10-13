//! `ffrwd:gpu`'s `encoder` and `decoder`: a codec package's halves reading
//! and writing their pictures as textures of the process's device.
//!
//! A module exporting them exports `ffrwd:av`'s `encoder` and `decoder` too,
//! which describe it and open it; these replace `encode` and `decode` once
//! it is open. A picture is handed over as one texture per plane in the wire's
//! order, and nothing crosses host memory but the packets.

use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use wasi_webgpu_wasmtime::reexports::wgpu_types;
use wasmtime::component::Instance;
use wasmtime::Store;

use super::{wasm_err, Host, Packet};
use crate::gpu::frames::{layout, GpuFrame, Plane, Yuv};

pub(super) mod encoder {
    wasmtime::component::bindgen!({
        path: "../wit-gpu",
        world: "encoder-module",
        require_store_data_send: true,
        with: {
            "wasi:io": wasmtime_wasi_io::bindings::wasi::io,
            "wasi:webgpu/webgpu": wasi_webgpu_wasmtime::wasi::webgpu::webgpu,
            "wasi:graphics-context/graphics-context": wasi_webgpu_wasmtime::wasi::graphics_context::graphics_context,
        },
    });
}

pub(super) mod decoder {
    wasmtime::component::bindgen!({
        path: "../wit-gpu",
        world: "decoder-module",
        require_store_data_send: true,
        with: {
            "wasi:io": wasmtime_wasi_io::bindings::wasi::io,
            "wasi:webgpu/webgpu": wasi_webgpu_wasmtime::wasi::webgpu::webgpu,
            "wasi:graphics-context/graphics-context": wasi_webgpu_wasmtime::wasi::graphics_context::graphics_context,
            "ffrwd:gpu/codec-types": super::encoder::ffrwd::gpu::codec_types,
        },
    });
}

use encoder::ffrwd::gpu::codec_types as wit;

/// A picture on the device, at `pts` for `duration` in the stream's time
/// base.
pub struct GpuPicture {
    pub pts: i64,
    pub duration: Option<i64>,
    pub frame: Arc<GpuFrame>,
}

/// The halves of `ffrwd:gpu`'s codec a module exports, bound to its
/// instance.
#[derive(Default)]
pub(super) struct GpuHalves {
    pub encoder: Option<encoder::EncoderModule>,
    pub decoder: Option<decoder::DecoderModule>,
}

impl GpuHalves {
    /// The halves `instance` exports, by `exports` naming which.
    pub fn bind(
        store: &mut Store<Host>,
        instance: &Instance,
        (encodes, decodes): (bool, bool),
    ) -> Result<GpuHalves> {
        let encoder = if encodes {
            Some(encoder::EncoderModule::new(&mut *store, instance).map_err(wasm_err)?)
        } else {
            None
        };
        let decoder = if decodes {
            Some(decoder::DecoderModule::new(&mut *store, instance).map_err(wasm_err)?)
        } else {
            None
        };
        Ok(GpuHalves { encoder, decoder })
    }
}

/// `pictures` through the encoder's `ffrwd:gpu/encoder.encode`, each
/// handed over as its wire planes. `name` is the module's, for messages.
pub(super) fn encode(
    store: &mut Store<Host>,
    module: &encoder::EncoderModule,
    name: &str,
    pictures: &[GpuPicture],
    last: bool,
) -> Result<Vec<Packet>> {
    let mut carried = Vec::with_capacity(pictures.len());
    for picture in pictures {
        let planes = picture.frame.wire_planes()?;
        let mut handles = Vec::with_capacity(planes.len());
        for plane in planes {
            handles.push(store.data_mut().table.push(shared(&plane))?);
        }
        carried.push(wit::Picture {
            pts: picture.pts,
            duration: picture.duration,
            planes: handles,
        });
    }
    let packets = module
        .ffrwd_gpu_encoder()
        .call_encode(&mut *store, &carried, last)
        .map_err(wasm_err)?
        .map_err(|e| anyhow!("{name}: {e}"))?;
    Ok(packets
        .into_iter()
        .map(|p| Packet {
            pts: p.pts,
            dts: p.dts,
            duration: p.duration,
            keyframe: p.keyframe,
            data: p.data,
        })
        .collect())
}

/// `packets` through the decoder's `ffrwd:gpu/decoder.decode`, each picture
/// taken over as a frame of `pix_fmt` at `width` x `height` whose YUV is read
/// in `yuv`. `name` is the module's, for messages.
#[allow(clippy::too_many_arguments)]
pub(super) fn decode(
    store: &mut Store<Host>,
    module: &decoder::DecoderModule,
    name: &str,
    packets: &[Packet],
    last: bool,
    (pix_fmt, width, height): (&str, u32, u32),
    yuv: Yuv,
) -> Result<Vec<GpuPicture>> {
    let carried: Vec<_> = packets
        .iter()
        .map(|p| wit::Packet {
            pts: p.pts,
            dts: p.dts,
            duration: p.duration,
            keyframe: p.keyframe,
            data: p.data.clone(),
        })
        .collect();
    let pictures = module
        .ffrwd_gpu_decoder()
        .call_decode(&mut *store, &carried, last)
        .map_err(wasm_err)?
        .map_err(|e| anyhow!("{name}: {e}"))?;
    let specs = layout(pix_fmt, width, height)
        .ok_or_else(|| anyhow!("{name}: {pix_fmt} is not a format held on the GPU"))?;
    let mut out = Vec::with_capacity(pictures.len());
    for picture in pictures {
        let mut taken = Vec::with_capacity(picture.planes.len());
        for handle in picture.planes {
            taken.push(store.data_mut().table.delete(handle)?);
        }
        let planes = adopted(name, picture.pts, &taken, &specs)?;
        let frame = GpuFrame::adopt_planes(pix_fmt, width, height, yuv, planes)?;
        out.push(GpuPicture {
            pts: picture.pts,
            duration: picture.duration,
            frame,
        });
    }
    Ok(out)
}

/// A plane of the host's as a guest handle: read-only, and the host's to
/// drop.
fn shared(plane: &Plane) -> wasi_webgpu_wasmtime::Texture {
    wasi_webgpu_wasmtime::Texture::shared(
        plane.texture,
        wgpu_types::Extent3d {
            width: plane.width,
            height: plane.height,
            depth_or_array_layers: 1,
        },
        plane.format,
        wgpu_types::TextureUsages::TEXTURE_BINDING | wgpu_types::TextureUsages::COPY_SRC,
    )
}

/// The textures a decoder handed over for the picture at `pts`, checked
/// against the planes `specs` lays out: as many, each the plane's size and
/// format, single-level, single-sample and 2D, readable and copyable, and
/// the module's own.
fn adopted(
    name: &str,
    pts: i64,
    taken: &[wasi_webgpu_wasmtime::Texture],
    specs: &[(u32, u32, wgpu_types::TextureFormat, u32)],
) -> Result<Vec<Plane>> {
    if taken.len() != specs.len() {
        bail!(
            "{name}: the picture at pts {pts} has {} planes, and its format has {}",
            taken.len(),
            specs.len()
        );
    }
    let needed = wgpu_types::TextureUsages::TEXTURE_BINDING | wgpu_types::TextureUsages::COPY_SRC;
    let mut planes = Vec::with_capacity(taken.len());
    for (index, (texture, &(width, height, format, bytes_per_pixel))) in
        taken.iter().zip(specs).enumerate()
    {
        let size = texture.size();
        let why = if texture.is_shared() {
            Some("is one the host handed in".to_string())
        } else if !texture.usage().contains(needed) {
            Some(format!(
                "was made with usage {:?}; a plane needs texture-binding and copy-src",
                texture.usage()
            ))
        } else if texture.mip_level_count() != 1
            || texture.sample_count() != 1
            || texture.dimension() != wgpu_types::TextureDimension::D2
            || size.depth_or_array_layers != 1
        {
            Some("is not a single-level, single-sample 2D texture".to_string())
        } else if (size.width, size.height, texture.format()) != (width, height, format) {
            Some(format!(
                "is {}x{} {:?}, and the plane is {width}x{height} {format:?}",
                size.width,
                size.height,
                texture.format()
            ))
        } else {
            None
        };
        if let Some(why) = why {
            bail!("{name}: plane {index} of the picture at pts {pts} {why}");
        }
        planes.push(Plane {
            texture: texture.texture_id(),
            aspect: wgpu_types::TextureAspect::All,
            width,
            height,
            format,
            bytes_per_pixel,
        });
    }
    Ok(planes)
}
