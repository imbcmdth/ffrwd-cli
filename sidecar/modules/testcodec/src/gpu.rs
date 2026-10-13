//! The GPU build's halves: ffrwd:gpu's encoder and decoder, coding exactly
//! what ffrwd:av's do. The encoder reads each plane it is handed back from
//! the device and codes the bytes; the decoder writes each picture's planes
//! into textures of its own and hands them over. Either way the packets are
//! the ones the host build makes of the same pictures.

use std::cell::RefCell;

use crate::exports::ffrwd::gpu::{decoder, encoder};
use crate::ffrwd::gpu::codec_types::{Packet, Picture};
use crate::wasi::webgpu::webgpu::{
    get_gpu, GpuBufferDescriptor, GpuBufferUsage, GpuDevice, GpuExtent3D, GpuMapMode,
    GpuTexelCopyBufferInfo, GpuTexelCopyBufferLayout, GpuTexelCopyTextureInfo, GpuTexture,
    GpuTextureDescriptor, GpuTextureFormat, GpuTextureUsage,
};
use crate::{
    compress, expand, frame_len, Opened, TestCodec, DECODING, ENCODING, HEADER_LEN, MAGIC, NAME,
};

thread_local! {
    static DEVICE: RefCell<Option<GpuDevice>> = const { RefCell::new(None) };
}

/// The device every texture is on, opened the first time it is asked for.
fn with_device<T>(f: impl FnOnce(&GpuDevice) -> Result<T, String>) -> Result<T, String> {
    DEVICE.with(|d| {
        let mut device = d.borrow_mut();
        if device.is_none() {
            let adapter = get_gpu()
                .request_adapter(None)
                .ok_or_else(|| format!("{NAME}: no GPU adapter"))?;
            *device = Some(
                adapter
                    .request_device(None)
                    .map_err(|_| format!("{NAME}: the adapter gave no device"))?,
            );
        }
        f(device.as_ref().expect("opened above"))
    })
}

/// The size of each plane of a picture of `opened`, in the wire's order.
fn planes(opened: Opened) -> Result<Vec<(u32, u32)>, String> {
    let (w, h) = (opened.width, opened.height);
    match opened.pix {
        0 => Ok(vec![(w, h), (w / 2, h / 2), (w / 2, h / 2)]),
        2 => Ok(vec![(w, h); 3]),
        3 => Ok(vec![(w, h), (w / 2, h), (w / 2, h)]),
        _ => Err(format!("{NAME}: gray is not held on the GPU")),
    }
}

/// `texture`, `width` x `height` r8unorm, read back.
fn read_plane(
    device: &GpuDevice,
    texture: &GpuTexture,
    width: u32,
    height: u32,
    out: &mut Vec<u8>,
) -> Result<(), String> {
    let padded = width.div_ceil(256) * 256;
    let buffer = device.create_buffer(&GpuBufferDescriptor {
        size: u64::from(padded * height),
        usage: GpuBufferUsage::copy_dst() | GpuBufferUsage::map_read(),
        mapped_at_creation: None,
        label: None,
    });
    let encoder = device.create_command_encoder(None);
    encoder.copy_texture_to_buffer(
        &GpuTexelCopyTextureInfo {
            texture,
            mip_level: None,
            origin: None,
            aspect: None,
        },
        &GpuTexelCopyBufferInfo {
            buffer: &buffer,
            offset: None,
            bytes_per_row: Some(padded),
            rows_per_image: Some(height),
        },
        GpuExtent3D {
            width,
            height: Some(height),
            depth_or_array_layers: None,
        },
    );
    device.queue().submit(&[&encoder.finish(None)]);
    buffer
        .map_async(GpuMapMode::read(), None, None)
        .map_err(|_| format!("{NAME}: mapping a plane's readback"))?;
    let mapped = buffer
        .get_mapped_range_get_with_copy(None, None)
        .map_err(|_| format!("{NAME}: reading a plane's readback"))?;
    for row in 0..height as usize {
        let at = row * padded as usize;
        out.extend_from_slice(&mapped[at..at + width as usize]);
    }
    buffer
        .unmap()
        .map_err(|_| format!("{NAME}: unmapping a plane's readback"))
}

/// A new r8unorm texture holding `data`, `width` x `height`.
fn write_plane(device: &GpuDevice, data: &[u8], width: u32, height: u32) -> GpuTexture {
    let extent = GpuExtent3D {
        width,
        height: Some(height),
        depth_or_array_layers: None,
    };
    let texture = device.create_texture(&GpuTextureDescriptor {
        size: extent,
        mip_level_count: None,
        sample_count: None,
        dimension: None,
        format: GpuTextureFormat::R8unorm,
        usage: GpuTextureUsage::copy_src()
            | GpuTextureUsage::copy_dst()
            | GpuTextureUsage::texture_binding(),
        view_formats: None,
        label: None,
    });
    device.queue().write_texture_with_copy(
        &GpuTexelCopyTextureInfo {
            texture: &texture,
            mip_level: None,
            origin: None,
            aspect: None,
        },
        data,
        GpuTexelCopyBufferLayout {
            offset: None,
            bytes_per_row: Some(width),
            rows_per_image: Some(height),
        },
        extent,
    );
    texture
}

impl encoder::Guest for TestCodec {
    fn encode(pictures: Vec<Picture>, _last: bool) -> Result<Vec<Packet>, String> {
        let opened = ENCODING
            .with(|e| *e.borrow())
            .ok_or_else(|| format!("{NAME}: encode before init"))?;
        let sizes = planes(opened)?;
        let len = frame_len(opened);
        let mut packets = Vec::with_capacity(pictures.len());
        for picture in pictures {
            if picture.planes.len() != sizes.len() {
                return Err(format!(
                    "{NAME}: a picture of {} planes, not the {} of its format",
                    picture.planes.len(),
                    sizes.len()
                ));
            }
            let mut frame = Vec::with_capacity(len);
            with_device(|device| {
                for (texture, (w, h)) in picture.planes.iter().zip(&sizes) {
                    read_plane(device, texture, *w, *h, &mut frame)?;
                }
                Ok(())
            })?;
            let mut data = Vec::with_capacity(HEADER_LEN + len / 4);
            data.extend_from_slice(MAGIC);
            data.extend_from_slice(&(opened.width as u16).to_le_bytes());
            data.extend_from_slice(&(opened.height as u16).to_le_bytes());
            data.push(opened.pix);
            compress(&frame, &mut data);
            packets.push(Packet {
                pts: picture.pts,
                dts: Some(picture.pts),
                duration: picture.duration,
                keyframe: true,
                data,
            });
        }
        Ok(packets)
    }
}

impl decoder::Guest for TestCodec {
    fn decode(packets: Vec<Packet>, _last: bool) -> Result<Vec<Picture>, String> {
        let opened = DECODING
            .with(|d| *d.borrow())
            .ok_or_else(|| format!("{NAME}: decode before init"))?;
        let sizes = planes(opened)?;
        let len = frame_len(opened);
        let mut pictures = Vec::with_capacity(packets.len());
        for packet in packets {
            let data = &packet.data;
            if data.len() < HEADER_LEN || &data[..2] != MAGIC || data[6] != opened.pix {
                return Err(format!(
                    "{NAME}: a packet at pts {} is not one of this stream's",
                    packet.pts
                ));
            }
            let frame = expand(&data[HEADER_LEN..], len)?;
            let planes = with_device(|device| {
                let mut at = 0;
                Ok(sizes
                    .iter()
                    .map(|&(w, h)| {
                        let end = at + (w * h) as usize;
                        let texture = write_plane(device, &frame[at..end], w, h);
                        at = end;
                        texture
                    })
                    .collect())
            })?;
            pictures.push(Picture {
                pts: packet.pts,
                duration: packet.duration,
                planes,
            });
        }
        Ok(pictures)
    }
}
