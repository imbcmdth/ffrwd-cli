//! Frames the host holds on the GPU: one texture per plane on the process's
//! device, in the layout the wire's pixel formats have.
//!
//! A frame gets here three ways. The host uploads one it holds in host
//! memory (a software decoder's output, a port feed's picture) with
//! [`GpuFrame::upload`], keeping the bytes beside the textures so nothing is
//! ever read back for them. A module hands over a texture it rendered on the
//! same device and the host takes it with [`GpuFrame::adopt`]. Or a decoder
//! opened on the device leaves its picture there, and the host takes it as
//! it is with [`DecodedPictures::hold`]. Reading the bytes of either of the
//! last two is a download, done once and kept ([`GpuFrame::bytes`]).
//!
//! A frame is held in one format and read in another where the two differ.
//! A decoder's picture is held as the decoder wrote it, NV12 (luma, then
//! interleaved chroma at half size); [`GpuFrame::pix_fmt`] says `nv12`. Its
//! wire format, [`GpuFrame::wire_pix_fmt`], is yuv420p: what its bytes are
//! packed as, and the planes a module is handed ([`GpuFrame::wire_planes`]),
//! repacked on the device once, the first time anything asks.
//!
//! An encoder on the device reads NV12, and [`GpuFrame::nv12_planes`] hands
//! a frame over so: a decoder's picture as it is held, an rgba frame
//! converted on the device with the stream's matrix and range, made once
//! and kept like the wire's planes.
//!
//! Every texture is on the one device [`super::shared_device`] opens, which
//! is the device every module's `wasi:webgpu` answers, so a module reads a
//! frame's planes as textures of its own device and copies nothing.

use std::borrow::Cow;
use std::fmt;
use std::sync::{mpsc, Arc, Mutex, MutexGuard, OnceLock};
use std::time::Instant;

use crate::profile::PROFILE;

use anyhow::{anyhow, bail, Result};
use wasi_webgpu_wasmtime::reexports::{wgpu_core, wgpu_types};
use wasi_webgpu_wasmtime::SharedDevice;
use wgpu_core::global::Global;
use wgpu_core::id::TextureId;
use wgpu_types::{Extent3d, TextureAspect, TextureFormat};

use super::{shared, shared_device};
use crate::av::DeviceFrame;

pub use super::convert::{Matrix, Yuv};

/// One plane of a frame: a texture, or one aspect of a multi-planar one
/// (the luma or chroma of an NV12 texture), read as `format`.
#[derive(Debug, Clone, Copy)]
pub struct Plane {
    pub texture: TextureId,
    pub aspect: TextureAspect,
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
    /// What the planes are held as: a wire format, or `nv12`.
    pix_fmt: String,
    /// What the bytes are packed as, and the planes a module reads.
    wire_pix_fmt: &'static str,
    planes: Vec<Plane>,
    /// The textures the frame drops: every plane's, except a decoder's
    /// texture, which is handed back instead.
    owned: Vec<TextureId>,
    /// The planes as the wire lays them out, where they are not the held
    /// ones: made the first time something asks, and owned by the frame.
    wire: Mutex<Option<Vec<Plane>>>,
    /// The frame as NV12 for an encoder on the device, where it is not
    /// held so, one entry per [`Yuv`] an rgba frame was converted with:
    /// made the first time something asks, and owned by the frame.
    nv12: Mutex<Vec<(Yuv, Vec<Plane>)>>,
    /// The frame's bytes as the wire lays them out: what it was uploaded
    /// from, or what reading it back gave.
    bytes: OnceLock<Arc<Vec<u8>>>,
    /// The decoder's own texture, where the frame is it.
    #[cfg(windows)]
    wrapped: Mutex<Option<super::d3d12::Wrapped>>,
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

/// The wire format a frame held as `pix_fmt` is read as: itself for a wire
/// format, yuv420p for `nv12`.
fn wire_of(pix_fmt: &str) -> Option<&'static str> {
    match pix_fmt {
        "nv12" => Some("yuv420p"),
        "rgba" => Some("rgba"),
        "yuv420p" => Some("yuv420p"),
        "yuv422p" => Some("yuv422p"),
        "yuv444p" => Some("yuv444p"),
        _ => None,
    }
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
        let started = Instant::now();
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
        let wire_pix_fmt = wire_of(pix_fmt).expect("a format with a layout is a wire format");
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
        let owned = planes.iter().map(|p| p.texture).collect();
        let frame = GpuFrame {
            global,
            device,
            width,
            height,
            pix_fmt: pix_fmt.to_string(),
            wire_pix_fmt,
            planes,
            owned,
            wire: Mutex::new(None),
            nv12: Mutex::new(Vec::new()),
            bytes: OnceLock::new(),
            #[cfg(windows)]
            wrapped: Mutex::new(None),
        };
        PROFILE.upload.add(bytes.len(), started);
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
            wire_pix_fmt: "rgba",
            planes: vec![Plane {
                texture,
                aspect: TextureAspect::All,
                width,
                height,
                format,
                bytes_per_pixel: 4,
            }],
            owned: vec![texture],
            wire: Mutex::new(None),
            nv12: Mutex::new(Vec::new()),
            bytes: OnceLock::new(),
            #[cfg(windows)]
            wrapped: Mutex::new(None),
        }))
    }

    /// The backend of the device the frame is on.
    pub fn backend(&self) -> wgpu_types::Backend {
        self.device.info.backend
    }

    /// Whether [`nv12_planes`](GpuFrame::nv12_planes) can make NV12 of
    /// this frame: a decoder's picture, an rgba frame, or a yuv420p one.
    pub fn has_nv12(&self) -> bool {
        matches!(self.pix_fmt.as_str(), "nv12" | "rgba" | "yuv420p")
    }

    /// The frame as NV12 on the device: its luma, and its chroma
    /// interleaved at half size. A decoder's picture is held so already;
    /// an rgba frame is converted with `yuv`'s matrix and range, and a
    /// yuv420p one has its chroma interleaved, once, the first time
    /// anything asks, and the result is kept with the frame. `yuv` is read
    /// for an rgba frame only.
    pub fn nv12_planes(&self, yuv: Yuv) -> Result<Vec<Plane>> {
        let key = match self.pix_fmt.as_str() {
            "nv12" => return Ok(self.planes.clone()),
            "rgba" => yuv,
            "yuv420p" => Yuv::DEFAULT,
            other => bail!("a {other} frame is not made NV12 on the device"),
        };
        let mut made = self.nv12.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, planes)) = made.iter().find(|(k, _)| *k == key) {
            return Ok(planes.clone());
        }
        let planes = if self.pix_fmt == "rgba" {
            let [rgba] = self.planes.as_slice() else {
                bail!("an rgba frame has one plane");
            };
            super::convert::nv12_from_rgba(&self.global, &self.device, rgba, yuv)?
        } else {
            super::convert::nv12_from_yuv420p(&self.global, &self.device, &self.planes)?
        };
        made.push((key, planes.clone()));
        Ok(planes)
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// What the frame is held as on the GPU: its wire format, or `nv12`
    /// for a decoder's picture. See [`wire_pix_fmt`](GpuFrame::wire_pix_fmt)
    /// for what its bytes are.
    pub fn pix_fmt(&self) -> &str {
        &self.pix_fmt
    }

    /// The wire format the frame is read as: what [`bytes`](GpuFrame::bytes)
    /// packs and [`wire_planes`](GpuFrame::wire_planes) hands over.
    pub fn wire_pix_fmt(&self) -> &'static str {
        self.wire_pix_fmt
    }

    /// The planes as held, in [`pix_fmt`](GpuFrame::pix_fmt)'s order: for
    /// `nv12`, the luma and the interleaved chroma, which may be two
    /// aspects of one texture.
    pub fn planes(&self) -> &[Plane] {
        &self.planes
    }

    /// The planes as the wire lays them out, one texture each: the held
    /// ones for a frame held in its wire format, else made from them on
    /// the device the first time anything asks, and kept.
    pub fn wire_planes(&self) -> Result<Vec<Plane>> {
        if self.pix_fmt == self.wire_pix_fmt {
            return Ok(self.planes.clone());
        }
        let mut wire = self.lock_wire();
        if let Some(planes) = wire.as_ref() {
            return Ok(planes.clone());
        }
        let [y, uv] = self.planes.as_slice() else {
            bail!("a {} frame has two planes", self.pix_fmt);
        };
        let made = super::convert::yuv420p_from_nv12(
            &self.global,
            &self.device,
            y,
            uv,
            self.width,
            self.height,
        )?;
        *wire = Some(made.clone());
        Ok(made)
    }

    fn lock_wire(&self) -> MutexGuard<'_, Option<Vec<Plane>>> {
        self.wire.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// How many bytes the frame is on the wire.
    pub fn wire_len(&self) -> usize {
        layout(self.wire_pix_fmt, self.width, self.height)
            .map(|specs| {
                specs
                    .iter()
                    .map(|(w, h, _, bpp)| *w as usize * *h as usize * *bpp as usize)
                    .sum()
            })
            .unwrap_or(0)
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
        let started = Instant::now();
        let bytes = self.download()?;
        PROFILE.download.add(bytes.len(), started);
        let _ = self.bytes.set(bytes);
        Ok(Arc::clone(self.bytes.get().expect("just set")))
    }

    fn download(&self) -> Result<Arc<Vec<u8>>> {
        let mut out = Vec::with_capacity(self.wire_len());
        for plane in &self.wire_planes()? {
            read_plane(&self.global, &self.device, plane, &mut out)?;
        }
        Ok(Arc::new(out))
    }
}

/// Reads one plane back, appending its rows, unpadded, to `out`.
fn read_plane(
    global: &Global,
    device: &SharedDevice,
    plane: &Plane,
    out: &mut Vec<u8>,
) -> Result<()> {
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
    let result = copy_out(global, device, plane, buffer, padded, bytes_per_row, out);
    global.buffer_drop(buffer);
    result
}

pub(super) fn create_plane(
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
        aspect: TextureAspect::All,
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
                aspect: plane.aspect,
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
        for texture in &self.owned {
            self.global.texture_drop(*texture);
        }
        if let Some(planes) = self.lock_wire().take() {
            for plane in planes {
                self.global.texture_drop(plane.texture);
            }
        }
        let nv12 = std::mem::take(&mut *self.nv12.lock().unwrap_or_else(|e| e.into_inner()));
        for (_, planes) in nv12 {
            for plane in planes {
                self.global.texture_drop(plane.texture);
            }
        }
        #[cfg(windows)]
        if let Some(wrapped) = self
            .wrapped
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        {
            wrapped.release(&self.global, &self.device);
        }
    }
}

/// How one decoder's pictures become frames: taken on the device as they
/// are (see [`super::d3d12`]), with one line on stderr saying how, and
/// again if that changes.
pub struct DecodedPictures {
    codec: String,
    device_type: &'static str,
    said: Option<String>,
    #[cfg(windows)]
    taker: super::d3d12::Taker,
}

impl DecodedPictures {
    /// For the decoder of `codec` opened on the process's device as
    /// `device_type` (`d3d12va`).
    pub fn new(codec: &str, device_type: &'static str) -> DecodedPictures {
        DecodedPictures {
            codec: codec.to_string(),
            device_type,
            said: None,
            #[cfg(windows)]
            taker: Default::default(),
        }
    }

    /// `picture` as a frame, held as the decoder wrote it and read as
    /// `wire_pix_fmt`. Says on stderr, the first time and whenever it
    /// changes, how the decoder's pictures are taken:
    /// `[gpu] decode h264: d3d12va, frames wrapped`, or
    /// `[gpu] decode h264: d3d12va, frames copied (<why not wrapped>)`.
    pub fn hold(&mut self, picture: DeviceFrame, wire_pix_fmt: &str) -> Result<Arc<GpuFrame>> {
        if wire_of(picture.pix_fmt) != Some(wire_pix_fmt) {
            bail!(
                "a {} picture is read as {}, and {wire_pix_fmt} was asked for",
                picture.pix_fmt,
                wire_of(picture.pix_fmt).unwrap_or("nothing")
            );
        }
        self.take(picture)
    }

    #[cfg(windows)]
    fn take(&mut self, picture: DeviceFrame) -> Result<Arc<GpuFrame>> {
        let (global, device) = device()?;
        let (width, height, pix_fmt) = (picture.width, picture.height, picture.pix_fmt);
        let taken = self.taker.take(&global, &device, picture)?;
        let how = match (taken.path, &taken.why) {
            (super::d3d12::Path::Wrapped, _) => "frames wrapped".to_string(),
            (super::d3d12::Path::Copied, Some(why)) => format!("frames copied ({why})"),
            (super::d3d12::Path::Copied, None) => "frames copied".to_string(),
        };
        if self.said.as_deref() != Some(how.as_str()) {
            eprintln!("[gpu] decode {}: {}, {how}", self.codec, self.device_type);
            self.said = Some(how);
        }
        Ok(Arc::new(GpuFrame {
            global,
            device,
            width,
            height,
            pix_fmt: pix_fmt.to_string(),
            wire_pix_fmt: wire_of(pix_fmt).expect("checked by hold"),
            planes: taken.planes,
            owned: taken.owned,
            wire: Mutex::new(None),
            nv12: Mutex::new(Vec::new()),
            bytes: OnceLock::new(),
            wrapped: Mutex::new(taken.wrapped),
        }))
    }

    #[cfg(not(windows))]
    fn take(&mut self, picture: DeviceFrame) -> Result<Arc<GpuFrame>> {
        let _ = (&self.codec, &self.device_type, &self.said, picture);
        bail!("a decoder's pictures are taken on the device on Windows only, so far")
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
        assert!(!holds("nv12"), "nv12 is held, never asked for");
        assert_eq!(wire_of("nv12"), Some("yuv420p"));
    }

    /// ffmpeg's own conversion of `rgba`, a `width` x `height` picture, to
    /// NV12 with `yuv`'s matrix and range; None where ffmpeg is not on the
    /// path.
    fn ffmpeg_nv12(rgba: &[u8], width: u32, height: u32, yuv: Yuv) -> Option<Vec<u8>> {
        use std::io::Write;
        use std::process::{Command, Stdio};
        let mut child = Command::new("ffmpeg")
            .args(["-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s"])
            .arg(format!("{width}x{height}"))
            .args([
                "-i",
                "-",
                "-vf",
                &format!(
                    "scale=out_color_matrix={}:out_range={},format=nv12",
                    match yuv.matrix {
                        Matrix::Bt709 => "bt709",
                        Matrix::Bt601 => "bt601",
                    },
                    if yuv.full_range { "pc" } else { "tv" }
                ),
                "-f",
                "rawvideo",
                "-",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .ok()?;
        let mut stdin = child.stdin.take()?;
        let input = rgba.to_vec();
        let writer = std::thread::spawn(move || stdin.write_all(&input));
        let output = child.wait_with_output().ok()?;
        let _ = writer.join();
        assert!(
            output.status.success(),
            "ffmpeg: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        Some(output.stdout)
    }

    /// A picture with smooth ramps, hard edges, a red block and a
    /// transparent (premultiplied, so all zero) corner.
    fn test_picture(width: u32, height: u32) -> Vec<u8> {
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                let pixel = if x < 8 && y < 8 {
                    [0, 0, 0, 0]
                } else if (20..36).contains(&x) && (10..26).contains(&y) {
                    [255, 0, 0, 255]
                } else {
                    [
                        (x * 255 / (width - 1)) as u8,
                        (y * 255 / (height - 1)) as u8,
                        (((x / 8 + y / 8) % 2) * 200 + 20) as u8,
                        255,
                    ]
                };
                rgba.extend_from_slice(&pixel);
            }
        }
        rgba
    }

    /// The largest and the mean difference between two planes.
    fn difference(a: &[u8], b: &[u8]) -> (u8, f64) {
        assert_eq!(a.len(), b.len());
        let max = a
            .iter()
            .zip(b)
            .map(|(x, y)| x.abs_diff(*y))
            .max()
            .unwrap_or(0);
        let sum: u64 = a
            .iter()
            .zip(b)
            .map(|(x, y)| u64::from(x.abs_diff(*y)))
            .sum();
        (max, sum as f64 / a.len() as f64)
    }

    #[test]
    fn a_yuv420p_frame_is_made_nv12_texel_for_texel() {
        let Some(device) = shared_device() else {
            eprintln!("SKIPPED: this machine has no GPU adapter.");
            return;
        };
        let (width, height) = (6u32, 4u32);
        let (luma, chroma) = ((width * height) as usize, (3 * 2) as usize);
        let bytes: Vec<u8> = (0..luma + 2 * chroma).map(|i| (i * 7) as u8).collect();
        let frame =
            GpuFrame::upload("yuv420p", width, height, Arc::new(bytes.clone())).expect("upload");
        let mut ours = Vec::new();
        for plane in &frame.nv12_planes(Yuv::DEFAULT).expect("interleave") {
            read_plane(shared(), device, plane, &mut ours).expect("read back");
        }
        let mut wanted = bytes[..luma].to_vec();
        for i in 0..chroma {
            wanted.push(bytes[luma + i]);
            wanted.push(bytes[luma + chroma + i]);
        }
        assert_eq!(ours, wanted);
    }

    #[test]
    fn rgba_is_made_nv12_on_the_device_as_ffmpeg_makes_it() {
        let (width, height) = (64u32, 48u32);
        let rgba = test_picture(width, height);
        if ffmpeg_nv12(&rgba, width, height, Yuv::DEFAULT).is_none() {
            eprintln!("SKIPPED: put ffmpeg on PATH to compare the conversion with its own.");
            return;
        }
        let Some(device) = shared_device() else {
            eprintln!("SKIPPED: this machine has no GPU adapter.");
            return;
        };
        let frame =
            GpuFrame::upload("rgba", width, height, Arc::new(rgba.clone())).expect("upload");
        let full = |matrix| Yuv {
            matrix,
            full_range: true,
        };
        for yuv in [Yuv::DEFAULT, full(Matrix::Bt709), full(Matrix::Bt601)] {
            let theirs = ffmpeg_nv12(&rgba, width, height, yuv).expect("ffmpeg ran once");
            let planes = frame.nv12_planes(yuv).expect("convert");
            assert_eq!(planes.len(), 2);
            let mut ours = Vec::new();
            for plane in &planes {
                read_plane(shared(), device, plane, &mut ours).expect("read back");
            }
            let luma = (width * height) as usize;
            assert_eq!(ours.len(), theirs.len());
            let (y_max, y_mean) = difference(&ours[..luma], &theirs[..luma]);
            let (c_max, c_mean) = difference(&ours[luma..], &theirs[luma..]);
            eprintln!(
                "rgba to NV12 ({yuv:?}) against ffmpeg: luma at most {y_max} apart \
                 (mean {y_mean:.3}), chroma at most {c_max} apart (mean {c_mean:.3})"
            );
            assert!(y_max <= 2, "{yuv:?}: luma differs by up to {y_max}");
            assert!(c_max <= 2, "{yuv:?}: chroma differs by up to {c_max}");
            // The transparent corner is black.
            let black = if yuv.full_range { 0 } else { 16 };
            assert_eq!(ours[0], black, "{yuv:?}");
            assert_eq!(&ours[luma..luma + 2], &[128, 128], "{yuv:?}");
        }
    }
}
