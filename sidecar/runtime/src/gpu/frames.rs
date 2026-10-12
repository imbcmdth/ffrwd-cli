//! Frames the host holds on the GPU: one texture per plane on the process's
//! device, in the layout the wire's pixel formats have.
//!
//! A frame gets here two ways. The host uploads one it holds in host memory
//! (a decoder's output, a port feed's picture) with [`GpuFrame::upload`],
//! keeping the bytes beside the textures so nothing is ever read back for
//! them. Or a module hands over a texture it rendered on the same device
//! and the host takes it with [`GpuFrame::adopt`]; reading that one's bytes
//! is a download, done once and kept ([`GpuFrame::bytes`]).
//!
//! Every texture is on the one device [`super::shared_device`] opens, which
//! is the device every module's `wasi:webgpu` answers, so a module reads a
//! frame's planes as textures of its own device and copies nothing.

use std::borrow::Cow;
use std::fmt;
use std::sync::{mpsc, Arc, OnceLock};

use anyhow::{anyhow, bail, Result};
use wasi_webgpu_wasmtime::reexports::{wgpu_core, wgpu_types};
use wasi_webgpu_wasmtime::SharedDevice;
use wgpu_core::global::Global;
use wgpu_core::id::TextureId;
use wgpu_types::{Extent3d, TextureFormat};

use super::{shared, shared_device};

/// One plane of a frame, as one texture.
#[derive(Debug, Clone, Copy)]
pub struct Plane {
    pub texture: TextureId,
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    pub bytes_per_pixel: u32,
}

impl Plane {
    fn wire_len(&self) -> usize {
        self.width as usize * self.height as usize * self.bytes_per_pixel as usize
    }

    fn size(&self) -> Extent3d {
        Extent3d {
            width: self.width,
            height: self.height,
            depth_or_array_layers: 1,
        }
    }
}

/// A frame on the GPU.
pub struct GpuFrame {
    global: Arc<Global>,
    device: Arc<SharedDevice>,
    width: u32,
    height: u32,
    pix_fmt: String,
    planes: Vec<Plane>,
    /// The frame's bytes as the wire lays them out: what it was uploaded
    /// from, or what reading it back gave.
    bytes: OnceLock<Arc<Vec<u8>>>,
}

fn chroma(n: u32) -> u32 {
    n.div_ceil(2)
}

/// The planes a wire format is held as, each as width, height, texture
/// format and bytes per pixel, in the wire's order. None for a format
/// frames are not held on the GPU in.
pub fn layout(
    pix_fmt: &str,
    width: u32,
    height: u32,
) -> Option<Vec<(u32, u32, TextureFormat, u32)>> {
    let (w, h) = (width, height);
    let r8 = TextureFormat::R8Unorm;
    Some(match pix_fmt {
        "rgba" => vec![(w, h, TextureFormat::Rgba8Unorm, 4)],
        "yuv420p" => vec![
            (w, h, r8, 1),
            (chroma(w), chroma(h), r8, 1),
            (chroma(w), chroma(h), r8, 1),
        ],
        "yuv422p" => vec![(w, h, r8, 1), (chroma(w), h, r8, 1), (chroma(w), h, r8, 1)],
        "yuv444p" => vec![(w, h, r8, 1), (w, h, r8, 1), (w, h, r8, 1)],
        _ => return None,
    })
}

/// Whether frames of this wire format can be held on the GPU.
pub fn holds(pix_fmt: &str) -> bool {
    layout(pix_fmt, 2, 2).is_some()
}

const USAGE: wgpu_types::TextureUsages = wgpu_types::TextureUsages::TEXTURE_BINDING
    .union(wgpu_types::TextureUsages::COPY_SRC)
    .union(wgpu_types::TextureUsages::COPY_DST)
    .union(wgpu_types::TextureUsages::RENDER_ATTACHMENT);

fn device() -> Result<(Arc<Global>, Arc<SharedDevice>)> {
    let device = shared_device()
        .ok_or_else(|| anyhow!("no GPU: this machine has no adapter the host can open"))?;
    Ok((Arc::clone(shared()), Arc::clone(device)))
}

impl GpuFrame {
    /// Puts a frame the host holds in host memory, packed as the wire lays
    /// `pix_fmt` out, on the GPU. The bytes are kept, so reading them back
    /// costs nothing.
    pub fn upload(
        pix_fmt: &str,
        width: u32,
        height: u32,
        bytes: Arc<Vec<u8>>,
    ) -> Result<Arc<GpuFrame>> {
        let specs = layout(pix_fmt, width, height)
            .ok_or_else(|| anyhow!("{pix_fmt} is not a format frames are held on the GPU in"))?;
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
        let (global, device) = device()?;
        let mut planes: Vec<Plane> = Vec::with_capacity(specs.len());
        let mut offset = 0usize;
        for (w, h, format, bpp) in specs {
            let plane = match create_plane(&global, &device, w, h, format, bpp) {
                Ok(plane) => plane,
                Err(e) => {
                    for made in &planes {
                        global.texture_drop(made.texture);
                    }
                    return Err(e);
                }
            };
            let len = plane.wire_len();
            let written = global.queue_write_texture(
                *device.queue,
                &wgpu_types::TexelCopyTextureInfo {
                    texture: plane.texture,
                    mip_level: 0,
                    origin: wgpu_types::Origin3d::ZERO,
                    aspect: wgpu_types::TextureAspect::All,
                },
                &bytes[offset..offset + len],
                &wgpu_types::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(w * bpp),
                    rows_per_image: Some(h),
                },
                &plane.size(),
            );
            planes.push(plane);
            if let Err(e) = written {
                for made in &planes {
                    global.texture_drop(made.texture);
                }
                bail!("uploading a {w}x{h} {format:?} plane: {e}");
            }
            offset += len;
        }
        let frame = GpuFrame {
            global,
            device,
            width,
            height,
            pix_fmt: pix_fmt.to_string(),
            planes,
            bytes: OnceLock::new(),
        };
        let _ = frame.bytes.set(bytes);
        Ok(Arc::new(frame))
    }

    /// Takes over a texture a module made on the process's device: an
    /// `rgba8unorm` texture of `width` x `height`, which becomes an `rgba`
    /// frame. The caller has removed the module's handle to it; the frame
    /// now owns the texture and drops it.
    pub fn adopt(
        texture: TextureId,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> Result<Arc<GpuFrame>> {
        if format != TextureFormat::Rgba8Unorm {
            bail!("a frame handed back is an rgba8unorm texture, and this one is {format:?}");
        }
        let (global, device) = device()?;
        Ok(Arc::new(GpuFrame {
            global,
            device,
            width,
            height,
            pix_fmt: "rgba".to_string(),
            planes: vec![Plane {
                texture,
                width,
                height,
                format,
                bytes_per_pixel: 4,
            }],
            bytes: OnceLock::new(),
        }))
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// The wire format the frame is in.
    pub fn pix_fmt(&self) -> &str {
        &self.pix_fmt
    }

    /// The textures, one per plane, in the wire's plane order.
    pub fn planes(&self) -> &[Plane] {
        &self.planes
    }

    /// How many bytes the frame is on the wire.
    pub fn wire_len(&self) -> usize {
        self.planes.iter().map(Plane::wire_len).sum()
    }

    /// Whether the bytes are already in host memory.
    pub fn has_bytes(&self) -> bool {
        self.bytes.get().is_some()
    }

    /// The bytes as the wire lays them out: a download the first time for
    /// a frame a module rendered, nothing after.
    pub fn bytes(&self) -> Result<Arc<Vec<u8>>> {
        if let Some(bytes) = self.bytes.get() {
            return Ok(Arc::clone(bytes));
        }
        let bytes = self.download()?;
        let _ = self.bytes.set(bytes);
        Ok(Arc::clone(self.bytes.get().expect("just set")))
    }

    fn download(&self) -> Result<Arc<Vec<u8>>> {
        let global = &self.global;
        let device = &self.device;
        let mut out = Vec::with_capacity(self.wire_len());
        for plane in &self.planes {
            let bytes_per_row = plane.width * plane.bytes_per_pixel;
            let padded = bytes_per_row.div_ceil(wgpu_types::COPY_BYTES_PER_ROW_ALIGNMENT)
                * wgpu_types::COPY_BYTES_PER_ROW_ALIGNMENT;
            let size = padded as u64 * plane.height as u64;
            let (buffer, err) = global.device_create_buffer(
                device.device,
                &wgpu_types::BufferDescriptor {
                    label: Some(Cow::Borrowed("ffrwd frame readback")),
                    size,
                    usage: wgpu_types::BufferUsages::COPY_DST | wgpu_types::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                },
                None,
            );
            if let Some(e) = err {
                global.buffer_drop(buffer);
                bail!("a {size}-byte readback buffer: {e}");
            }
            let result = copy_out(
                global,
                device,
                plane,
                buffer,
                padded,
                bytes_per_row,
                &mut out,
            );
            global.buffer_drop(buffer);
            result?;
        }
        Ok(Arc::new(out))
    }
}

fn create_plane(
    global: &Global,
    device: &SharedDevice,
    width: u32,
    height: u32,
    format: TextureFormat,
    bytes_per_pixel: u32,
) -> Result<Plane> {
    let desc = wgpu_types::TextureDescriptor {
        label: Some(Cow::Borrowed("ffrwd frame")),
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu_types::TextureDimension::D2,
        format,
        usage: USAGE,
        view_formats: Vec::new(),
    };
    let (texture, err) = global.device_create_texture(device.device, &desc, None);
    if let Some(e) = err {
        global.texture_drop(texture);
        bail!("a {width}x{height} {format:?} texture: {e}");
    }
    Ok(Plane {
        texture,
        width,
        height,
        format,
        bytes_per_pixel,
    })
}

/// Copies one plane into `buffer`, waits for it, and appends its rows,
/// unpadded, to `out`.
fn copy_out(
    global: &Global,
    device: &SharedDevice,
    plane: &Plane,
    buffer: wgpu_core::id::BufferId,
    padded: u32,
    bytes_per_row: u32,
    out: &mut Vec<u8>,
) -> Result<()> {
    let (encoder, err) = global.device_create_command_encoder(
        device.device,
        &wgpu_types::CommandEncoderDescriptor { label: None },
        None,
    );
    if let Some(e) = err {
        bail!("a command encoder for a readback: {e}");
    }
    global
        .command_encoder_copy_texture_to_buffer(
            encoder,
            &wgpu_types::TexelCopyTextureInfo {
                texture: plane.texture,
                mip_level: 0,
                origin: wgpu_types::Origin3d::ZERO,
                aspect: wgpu_types::TextureAspect::All,
            },
            &wgpu_types::TexelCopyBufferInfo {
                buffer,
                layout: wgpu_types::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(plane.height),
                },
            },
            &plane.size(),
        )
        .map_err(|e| anyhow!("copying a plane out: {e}"))?;
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
        .buffer_map_async(buffer, 0, None, op)
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
    let (ptr, len) = global
        .buffer_get_mapped_range(buffer, 0, None)
        .map_err(|e| anyhow!("reading a readback: {e}"))?;
    // The mapping is the buffer's own memory, valid until unmapped below.
    let mapped = unsafe { std::slice::from_raw_parts(ptr.as_ptr(), len as usize) };
    for row in 0..plane.height as usize {
        let start = row * padded as usize;
        out.extend_from_slice(&mapped[start..start + bytes_per_row as usize]);
    }
    global
        .buffer_unmap(buffer)
        .map_err(|e| anyhow!("unmapping a readback: {e}"))?;
    Ok(())
}

impl Drop for GpuFrame {
    fn drop(&mut self) {
        for plane in &self.planes {
            self.global.texture_drop(plane.texture);
        }
    }
}

impl fmt::Debug for GpuFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}x{} {} on {}, {} plane(s){}",
            self.width,
            self.height,
            self.pix_fmt,
            self.device.info.name,
            self.planes.len(),
            if self.has_bytes() { ", bytes held" } else { "" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_wire_formats_have_a_layout_and_nothing_else_does() {
        assert_eq!(layout("rgba", 1920, 1080).unwrap().len(), 1);
        let planes = layout("yuv420p", 1919, 1079).unwrap();
        assert_eq!(planes.len(), 3);
        assert_eq!((planes[1].0, planes[1].1), (960, 540));
        assert_eq!(layout("yuv422p", 1920, 1080).unwrap()[1].1, 1080);
        assert!(layout("nv12", 2, 2).is_none());
        assert!(holds("yuv444p"));
        assert!(!holds("gray"));
    }
}
