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
//! A frame is held in one layout and read in another where the two differ.
//! A decoder's picture is held as the decoder wrote it, NV12 (luma, then
//! interleaved chroma at half size); [`GpuFrame::pix_fmt`] says `nv12`. Its
//! wire format, [`GpuFrame::wire_pix_fmt`], is what its stream carries,
//! yuv420p or rgba: what its bytes are packed as, and the planes a module is
//! handed ([`GpuFrame::wire_planes`]), made on the device once, the first
//! time anything asks, in the colour of the stream it was decoded for.
//!
//! Every other layout and size is [`GpuFrame::converted`]: NV12 for an
//! encoder on the device, rgba for a reader that asks for it, another size
//! for an input that follows another's. Each is made by one conversion
//! (`super::convert`) the first time anything asks and kept with the frame,
//! so every reader asking for the same thing shares it.
//! A model reads a tensor, and [`GpuFrame::tensor`] makes one of the
//! planes as held, kept the same way.
//!
//! Every texture is on the one device [`super::shared_device`] opens, which
//! is the device every module's `wasi:webgpu` answers, so a module reads a
//! frame's planes as textures of its own device and copies nothing.

use std::borrow::Cow;
use std::fmt;
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::time::Instant;

use crate::profile::PROFILE;

use anyhow::{anyhow, bail, Result};
use wasi_webgpu_wasmtime::reexports::{wgpu_core, wgpu_types};
use wasi_webgpu_wasmtime::SharedDevice;
use wgpu_core::global::Global;
use wgpu_core::id::TextureId;
use wgpu_types::{Extent3d, TextureAspect, TextureFormat};

use super::convert;
use super::tensor::{DeviceTensor, Source, TensorSpec};
use super::{shared, shared_device};
use crate::av::DeviceFrame;

pub use super::convert::{converts, Layout, Matrix, Target, Yuv};

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
    /// What the planes are held as.
    layout: Layout,
    /// What the bytes are packed as, and the planes a module reads: a wire
    /// format.
    wire: Layout,
    /// The colour the frame's YUV is read in, where it is read as rgba.
    yuv: Yuv,
    planes: Vec<Plane>,
    /// The textures the frame drops: every plane's, except a decoder's
    /// texture, which is handed back instead.
    owned: Vec<TextureId>,
    /// What the frame was converted to, by target (see [`Target::key`]):
    /// made the first time something asks, and kept while the frame is.
    converted: Mutex<Vec<(Target, Arc<GpuFrame>)>>,
    /// The frame made into model inputs, one entry per spec and per kind of
    /// buffer (on a shared heap or not): made the first time something
    /// asks, and kept while the frame is.
    tensors: Mutex<Vec<(TensorSpec, bool, Arc<DeviceTensor>)>>,
    /// The frame's bytes as the wire lays them out: what it was uploaded
    /// from, or what reading it back gave.
    bytes: OnceLock<Arc<Vec<u8>>>,
    /// The decoder's own texture, where the frame is it.
    #[cfg(windows)]
    wrapped: Mutex<Option<super::d3d12::Wrapped>>,
}

/// The planes a wire format is held as, each as width, height, texture
/// format and bytes per pixel, in the wire's order. None for a format
/// frames are not held on the GPU in.
pub fn layout(
    pix_fmt: &str,
    width: u32,
    height: u32,
) -> Option<Vec<(u32, u32, TextureFormat, u32)>> {
    Layout::of(pix_fmt)
        .filter(|l| l.wire() == *l)
        .map(|l| l.planes(width, height))
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
    /// A frame of `planes` the host made on the device, held and read as
    /// `layout` (NV12 read as yuv420p), owning every plane.
    fn made(
        global: Arc<Global>,
        device: Arc<SharedDevice>,
        layout: Layout,
        (width, height): (u32, u32),
        yuv: Yuv,
        planes: Vec<Plane>,
    ) -> GpuFrame {
        GpuFrame {
            global,
            device,
            width,
            height,
            layout,
            wire: layout.wire(),
            yuv,
            owned: planes.iter().map(|p| p.texture).collect(),
            planes,
            converted: Mutex::new(Vec::new()),
            tensors: Mutex::new(Vec::new()),
            bytes: OnceLock::new(),
            #[cfg(windows)]
            wrapped: Mutex::new(None),
        }
    }

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
        let held = Layout::of(pix_fmt).expect("a format with a layout is a wire format");
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
        let frame = GpuFrame::made(global, device, held, (width, height), Yuv::DEFAULT, planes);
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
        let plane = Plane {
            texture,
            aspect: TextureAspect::All,
            width,
            height,
            format,
            bytes_per_pixel: 4,
        };
        Ok(Arc::new(GpuFrame::made(
            global,
            device,
            Layout::Rgba,
            (width, height),
            Yuv::DEFAULT,
            vec![plane],
        )))
    }

    /// The backend of the device the frame is on.
    pub fn backend(&self) -> wgpu_types::Backend {
        self.device.info.backend
    }

    /// Whether [`converted`](GpuFrame::converted) can make this frame
    /// `layout`.
    pub fn converts_to(&self, layout: Layout) -> bool {
        converts(self.layout, layout)
    }

    /// The frame as `target`: itself where it is held so already, else made
    /// on the device once, the first time anything asks, and kept with the
    /// frame, so every reader asking for the same target shares one
    /// conversion. `target.yuv` is the colour of the YUV side of a
    /// conversion between YUV and rgba, and is not read otherwise.
    pub fn converted(self: &Arc<Self>, target: Target) -> Result<Arc<GpuFrame>> {
        if target.layout == self.layout && (target.width, target.height) == self.size() {
            return Ok(Arc::clone(self));
        }
        self.conversion(target)
    }

    /// The frame as its wire format at another size, in its own colour:
    /// what an input that follows another input's size is handed.
    pub fn resized(self: &Arc<Self>, width: u32, height: u32) -> Result<Arc<GpuFrame>> {
        if (width, height) == self.size() {
            return Ok(Arc::clone(self));
        }
        self.converted(Target::new(self.wire, width, height, self.yuv))
    }

    /// The conversion to `target`, made once and kept.
    fn conversion(&self, target: Target) -> Result<Arc<GpuFrame>> {
        let key = target.key(self.layout);
        let mut made = self.converted.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, frame)) = made.iter().find(|(k, _)| *k == key) {
            return Ok(Arc::clone(frame));
        }
        let planes = convert::convert(&self.global, &self.device, self.layout, &self.planes, key)?;
        // YUV made from rgba is in the colour it was written in; anything
        // else keeps the source's.
        let yuv = if self.layout.is_rgb() && !key.layout.is_rgb() {
            key.yuv
        } else {
            self.yuv
        };
        let frame = Arc::new(GpuFrame::made(
            Arc::clone(&self.global),
            Arc::clone(&self.device),
            key.layout,
            (key.width, key.height),
            yuv,
            planes,
        ));
        made.push((key, Arc::clone(&frame)));
        Ok(frame)
    }

    fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// The frame made into a model's input tensor as `spec` says, on the
    /// device, into a buffer of its own (`shared`: on a shared heap, for
    /// another API to import). Made once per spec, the first time anything
    /// asks, and kept with the frame.
    pub fn tensor(&self, spec: &TensorSpec, shared: bool) -> Result<Arc<DeviceTensor>> {
        let mut made = self.tensors.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, _, tensor)) = made.iter().find(|(s, k, _)| s == spec && *k == shared) {
            return Ok(Arc::clone(tensor));
        }
        let source = Source::of(self.pix_fmt())
            .ok_or_else(|| anyhow!("a {} frame is not made into a tensor", self.pix_fmt()))?;
        let tensor = Arc::new(super::tensor::make(
            &self.global,
            &self.device,
            &self.planes,
            source,
            spec,
            shared,
        )?);
        made.push((*spec, shared, Arc::clone(&tensor)));
        Ok(tensor)
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
    pub fn pix_fmt(&self) -> &'static str {
        self.layout.name()
    }

    /// The wire format the frame is read as: what [`bytes`](GpuFrame::bytes)
    /// packs and [`wire_planes`](GpuFrame::wire_planes) hands over.
    pub fn wire_pix_fmt(&self) -> &'static str {
        self.wire.name()
    }

    /// The colour the frame's YUV is read in, where it is read as rgba.
    pub fn yuv(&self) -> Yuv {
        self.yuv
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
        if self.layout == self.wire {
            return Ok(self.planes.clone());
        }
        let target = Target::new(self.wire, self.width, self.height, self.yuv);
        Ok(self.conversion(target)?.planes.clone())
    }

    /// How many bytes the frame is on the wire.
    pub fn wire_len(&self) -> usize {
        self.wire.len(self.width, self.height)
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
    /// `wire_pix_fmt`, its YUV in the colour `yuv` names where it is read
    /// as rgba. Says on stderr, the first time and whenever it changes, how
    /// the decoder's pictures are taken:
    /// `[gpu] decode h264: d3d12va, frames wrapped`, or
    /// `[gpu] decode h264: d3d12va, frames copied (<why not wrapped>)`.
    pub fn hold(
        &mut self,
        picture: DeviceFrame,
        wire_pix_fmt: &str,
        yuv: Yuv,
    ) -> Result<Arc<GpuFrame>> {
        let held = Layout::of(picture.pix_fmt).ok_or_else(|| {
            anyhow!(
                "a decoder's {} picture is not held on the GPU",
                picture.pix_fmt
            )
        })?;
        let wire = Layout::of(wire_pix_fmt).filter(|w| w.wire() == *w && converts(held, *w));
        let Some(wire) = wire else {
            bail!(
                "a {} picture is not read as {wire_pix_fmt} on the device",
                picture.pix_fmt
            );
        };
        self.take(picture, held, wire, yuv)
    }

    #[cfg(windows)]
    fn take(
        &mut self,
        picture: DeviceFrame,
        held: Layout,
        wire: Layout,
        yuv: Yuv,
    ) -> Result<Arc<GpuFrame>> {
        let (global, device) = device()?;
        let (width, height) = (picture.width, picture.height);
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
            layout: held,
            wire,
            yuv,
            planes: taken.planes,
            owned: taken.owned,
            converted: Mutex::new(Vec::new()),
            tensors: Mutex::new(Vec::new()),
            bytes: OnceLock::new(),
            wrapped: Mutex::new(taken.wrapped),
        }))
    }

    #[cfg(not(windows))]
    fn take(
        &mut self,
        picture: DeviceFrame,
        held: Layout,
        wire: Layout,
        yuv: Yuv,
    ) -> Result<Arc<GpuFrame>> {
        let _ = (&self.codec, &self.device_type, &self.said, picture);
        let _ = (held, wire, yuv);
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
            self.layout.name(),
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
        assert_eq!(Layout::Nv12.wire(), Layout::Yuv420p);
        assert!(converts(Layout::Nv12, Layout::Rgba));
        assert!(converts(Layout::Rgba, Layout::Yuv420p));
        assert!(!converts(Layout::Rgba, Layout::Yuv444p));
    }

    #[test]
    fn a_target_counts_its_colour_only_between_rgba_and_yuv() {
        let bt601 = Yuv {
            matrix: Matrix::Bt601,
            full_range: true,
        };
        let target = Target::new(Layout::Yuv420p, 8, 8, bt601);
        assert_eq!(target.key(Layout::Rgba).yuv, bt601);
        assert_eq!(target.key(Layout::Nv12).yuv, Yuv::DEFAULT);
        let rgba = Target::new(Layout::Rgba, 8, 8, bt601);
        assert_eq!(rgba.key(Layout::Nv12).yuv, bt601);
        assert_eq!(rgba.key(Layout::Rgba).yuv, Yuv::DEFAULT);
        assert_eq!(
            Yuv::read_as("unknown", "unknown").unwrap().matrix,
            Matrix::Bt601
        );
        assert_eq!(
            Yuv::read_as("bt709", "pc").unwrap(),
            Yuv {
                matrix: Matrix::Bt709,
                full_range: true
            }
        );
        assert!(Yuv::read_as("bt2020nc", "tv").is_none());
    }

    /// The matrix and range as ffmpeg's scale filter names them.
    fn names(yuv: Yuv) -> (&'static str, &'static str) {
        (
            match yuv.matrix {
                Matrix::Bt709 => "bt709",
                Matrix::Bt601 => "bt601",
            },
            if yuv.full_range { "pc" } else { "tv" },
        )
    }

    /// ffmpeg's own conversion of a `width` x `height` `from` picture to a
    /// `to` picture through `-vf filter`; None where ffmpeg is not on the
    /// path.
    fn ffmpeg(
        picture: &[u8],
        from: &str,
        (width, height): (u32, u32),
        filter: &str,
    ) -> Option<Vec<u8>> {
        use std::io::Write;
        use std::process::{Command, Stdio};
        let mut child = Command::new("ffmpeg")
            .args(["-v", "error", "-f", "rawvideo", "-pix_fmt", from, "-s"])
            .arg(format!("{width}x{height}"))
            .args(["-i", "-", "-vf", filter, "-f", "rawvideo", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .ok()?;
        let mut stdin = child.stdin.take()?;
        let input = picture.to_vec();
        let writer = std::thread::spawn(move || stdin.write_all(&input));
        let output = child.wait_with_output().ok()?;
        let _ = writer.join();
        assert!(
            output.status.success(),
            "ffmpeg -vf {filter}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        Some(output.stdout)
    }

    fn ffmpeg_on_path() -> bool {
        std::process::Command::new("ffmpeg")
            .arg("-version")
            .output()
            .is_ok_and(|o| o.status.success())
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

    /// The largest and the mean difference between two pictures.
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

    /// Every plane of `frame` read back, packed.
    fn read_back(frame: &GpuFrame) -> Vec<u8> {
        let device = shared_device().expect("a device made the frame");
        let mut out = Vec::new();
        for plane in frame.planes() {
            read_plane(shared(), device, plane, &mut out).expect("read back");
        }
        out
    }

    /// The test picture as `pix_fmt`, as ffmpeg makes it with `yuv`, and
    /// as a frame held so on the device (NV12 interleaved there from
    /// yuv420p); None where ffmpeg is not on the path.
    fn source(pix_fmt: &str, size: (u32, u32), yuv: Yuv) -> Option<(Vec<u8>, Arc<GpuFrame>)> {
        let rgba = test_picture(size.0, size.1);
        if pix_fmt == "rgba" {
            let frame =
                GpuFrame::upload("rgba", size.0, size.1, Arc::new(rgba.clone())).expect("upload");
            return Some((rgba, frame));
        }
        let (matrix, range) = names(yuv);
        let wire = if pix_fmt == "nv12" {
            "yuv420p"
        } else {
            pix_fmt
        };
        let bytes = ffmpeg(
            &rgba,
            "rgba",
            size,
            &format!("scale=out_color_matrix={matrix}:out_range={range},format={wire}"),
        )?;
        let frame =
            GpuFrame::upload(wire, size.0, size.1, Arc::new(bytes.clone())).expect("upload");
        if pix_fmt != "nv12" {
            return Some((bytes, frame));
        }
        let nv12 = frame
            .converted(Target::new(Layout::Nv12, size.0, size.1, yuv))
            .expect("interleave");
        let luma = (size.0 * size.1) as usize;
        let chroma = (bytes.len() - luma) / 2;
        let mut packed = bytes[..luma].to_vec();
        for i in 0..chroma {
            packed.push(bytes[luma + i]);
            packed.push(bytes[luma + chroma + i]);
        }
        Some((packed, nv12))
    }

    /// Converts the test picture, made `from` by ffmpeg, to `to` at `out`
    /// on the device and with ffmpeg's scale filter (`flags`, the matrix
    /// and range `yuv` names on both sides), and returns the largest and
    /// mean difference, said on stderr. None, having said why, where
    /// ffmpeg or a GPU is missing.
    fn against_ffmpeg(
        from: &str,
        to: &str,
        size: (u32, u32),
        out: (u32, u32),
        yuv: Yuv,
        flags: &str,
    ) -> Option<(u8, f64)> {
        if !ffmpeg_on_path() {
            eprintln!("SKIPPED: put ffmpeg on PATH to compare the conversion with its own.");
            return None;
        }
        if shared_device().is_none() {
            eprintln!("SKIPPED: this machine has no GPU adapter.");
            return None;
        }
        let (picture, frame) = source(from, size, yuv)?;
        let (matrix, range) = names(yuv);
        let theirs = ffmpeg(
            &picture,
            from,
            size,
            &format!(
                "scale={}:{}:flags={flags}:in_color_matrix={matrix}:in_range={range}:\
                 out_color_matrix={matrix}:out_range={range},format={to}",
                out.0, out.1
            ),
        )
        .expect("ffmpeg ran once");
        let layout = Layout::of(to).expect("a layout");
        let converted = frame
            .converted(Target::new(layout, out.0, out.1, yuv))
            .expect("convert");
        assert_eq!(converted.pix_fmt(), to);
        let ours = read_back(&converted);
        assert_eq!(ours.len(), theirs.len(), "{from} to {to}");
        let (max, mean) = difference(&ours, &theirs);
        eprintln!(
            "{from} {}x{} to {to} {}x{} ({yuv:?}) against ffmpeg's scale with \
             flags={flags}: at most {max} apart, mean {mean:.3}",
            size.0, size.1, out.0, out.1
        );
        Some((max, mean))
    }

    const SIZE: (u32, u32) = (64, 48);

    /// ffmpeg's bilinear filter with rgba's chroma kept at full size on
    /// the way in and through, which it otherwise halves across.
    const FULL_BILINEAR: &str = "bilinear+full_chroma_inp+full_chroma_int";

    fn bt601_full() -> Yuv {
        Yuv {
            matrix: Matrix::Bt601,
            full_range: true,
        }
    }

    #[test]
    fn a_yuv420p_frame_is_made_nv12_texel_for_texel() {
        if shared_device().is_none() {
            eprintln!("SKIPPED: this machine has no GPU adapter.");
            return;
        }
        let (width, height) = (6u32, 4u32);
        let (luma, chroma) = ((width * height) as usize, (3 * 2) as usize);
        let bytes: Vec<u8> = (0..luma + 2 * chroma).map(|i| (i * 7) as u8).collect();
        let frame =
            GpuFrame::upload("yuv420p", width, height, Arc::new(bytes.clone())).expect("upload");
        let nv12 = frame
            .converted(Target::new(Layout::Nv12, width, height, Yuv::DEFAULT))
            .expect("interleave");
        let mut wanted = bytes[..luma].to_vec();
        for i in 0..chroma {
            wanted.push(bytes[luma + i]);
            wanted.push(bytes[luma + chroma + i]);
        }
        assert_eq!(read_back(&nv12), wanted);
        // Made NV12, it reads as the yuv420p it came from.
        assert_eq!(nv12.wire_pix_fmt(), "yuv420p");
        assert_eq!(*nv12.bytes().expect("bytes"), bytes);
        let again = frame
            .converted(Target::new(Layout::Nv12, width, height, Yuv::DEFAULT))
            .expect("kept");
        assert!(Arc::ptr_eq(&nv12, &again), "one conversion, shared");
    }

    #[test]
    fn rgba_is_made_nv12_on_the_device_as_ffmpeg_makes_it() {
        let (width, height) = SIZE;
        let rgba = test_picture(width, height);
        if !ffmpeg_on_path() {
            eprintln!("SKIPPED: put ffmpeg on PATH to compare the conversion with its own.");
            return;
        }
        if shared_device().is_none() {
            eprintln!("SKIPPED: this machine has no GPU adapter.");
            return;
        }
        let frame =
            GpuFrame::upload("rgba", width, height, Arc::new(rgba.clone())).expect("upload");
        let full = |matrix| Yuv {
            matrix,
            full_range: true,
        };
        for yuv in [Yuv::DEFAULT, full(Matrix::Bt709), full(Matrix::Bt601)] {
            let (matrix, range) = names(yuv);
            let theirs = ffmpeg(
                &rgba,
                "rgba",
                SIZE,
                &format!("scale=out_color_matrix={matrix}:out_range={range},format=nv12"),
            )
            .expect("ffmpeg ran once");
            let nv12 = frame
                .converted(Target::new(Layout::Nv12, width, height, yuv))
                .expect("convert");
            assert_eq!(nv12.planes().len(), 2);
            let ours = read_back(&nv12);
            let luma = (width * height) as usize;
            assert_eq!(ours.len(), theirs.len());
            let (y_max, y_mean) = difference(&ours[..luma], &theirs[..luma]);
            let (c_max, c_mean) = difference(&ours[luma..], &theirs[luma..]);
            eprintln!(
                "rgba to NV12 ({yuv:?}) against ffmpeg: luma at most {y_max} apart \
                 (mean {y_mean:.3}), chroma at most {c_max} apart (mean {c_mean:.3})"
            );
            // Measured on an RTX 4090: at most 1 apart; the means at most
            // 0.035 for luma and 0.488 for full-range chroma, which ffmpeg
            // rounds half a level apart from this.
            assert!(y_max <= 1, "{yuv:?}: luma differs by up to {y_max}");
            assert!(c_max <= 1, "{yuv:?}: chroma differs by up to {c_max}");
            assert!(y_mean < 0.04, "{yuv:?}: luma mean {y_mean}");
            assert!(c_mean < 0.5, "{yuv:?}: chroma mean {c_mean}");
            // The transparent corner is black.
            let black = if yuv.full_range { 0 } else { 16 };
            assert_eq!(ours[0], black, "{yuv:?}");
            assert_eq!(&ours[luma..luma + 2], &[128, 128], "{yuv:?}");
        }
    }

    /// Every YUV layout to rgba, at the picture's size and at a smaller
    /// and a larger one, in both colours, against ffmpeg's bilinear filter
    /// with the chroma interpolated to full size (`full_chroma_int`) and
    /// rounded accurately: without those, ffmpeg repeats each chroma sample
    /// across a pair of pixels rather than filtering it.
    #[test]
    fn yuv_is_made_rgba_on_the_device_as_ffmpeg_makes_it() {
        let flags = "bilinear+full_chroma_int+accurate_rnd";
        for yuv in [Yuv::DEFAULT, bt601_full()] {
            for from in ["nv12", "yuv420p", "yuv422p", "yuv444p"] {
                for out in [SIZE, (40, 30), (96, 72)] {
                    let Some((max, mean)) = against_ffmpeg(from, "rgba", SIZE, out, yuv, flags)
                    else {
                        return;
                    };
                    // Measured on an RTX 4090: at most 1 apart everywhere,
                    // the mean at most 0.026.
                    assert!(max <= 1, "{from} to rgba {out:?}, {yuv:?}: {max} apart");
                    assert!(mean < 0.03, "{from} to rgba {out:?}, {yuv:?}: mean {mean}");
                }
            }
        }
    }

    /// rgba to yuv420p at its own size, against ffmpeg's default filter,
    /// whose chroma the conversion follows; and at other sizes against its
    /// bilinear filter with rgba's chroma kept at full size on the way in.
    #[test]
    fn rgba_is_made_yuv420p_on_the_device_as_ffmpeg_makes_it() {
        for yuv in [Yuv::DEFAULT, bt601_full()] {
            for (out, flags) in [
                (SIZE, "bicubic"),
                ((40, 30), "bilinear+full_chroma_inp+accurate_rnd"),
                ((32, 24), "bilinear+full_chroma_inp+accurate_rnd"),
            ] {
                let Some((max, mean)) = against_ffmpeg("rgba", "yuv420p", SIZE, out, yuv, flags)
                else {
                    return;
                };
                // Measured on an RTX 4090: at most 1 apart, the mean at most
                // 0.196 (full range, where ffmpeg rounds chroma half a level
                // apart from this).
                assert!(max <= 1, "rgba to yuv420p {out:?}, {yuv:?}: {max} apart");
                assert!(mean < 0.21, "rgba to yuv420p {out:?}, {yuv:?}: mean {mean}");
            }
        }
    }

    /// Each layout resized smaller, by half, larger, and to an odd size,
    /// and NV12 and yuv420p made one another and yuv444p, against ffmpeg's
    /// bilinear filter.
    #[test]
    fn a_frame_is_resized_on_the_device_as_ffmpeg_resizes_it() {
        let mut cases: Vec<(&str, &str, (u32, u32))> = Vec::new();
        for from in ["rgba", "yuv420p", "nv12", "yuv444p"] {
            for out in [(40, 30), (32, 24), (96, 72), (17, 13)] {
                cases.push((from, from, out));
            }
        }
        cases.extend([
            ("nv12", "yuv420p", SIZE),
            ("yuv420p", "nv12", SIZE),
            ("nv12", "yuv444p", SIZE),
            ("nv12", "yuv420p", (40, 30)),
        ]);
        for (from, to, out) in cases {
            let Some((max, mean)) =
                against_ffmpeg(from, to, SIZE, out, Yuv::DEFAULT, FULL_BILINEAR)
            else {
                return;
            };
            // Measured on an RTX 4090: at most 1 apart, the mean at most
            // 0.096; NV12 and yuv420p made one another are exact.
            assert!(max <= 1, "{from} to {to} {out:?}: {max} apart");
            assert!(mean < 0.1, "{from} to {to} {out:?}: mean {mean}");
            if out == SIZE && from != "rgba" && to != "yuv444p" {
                assert_eq!(max, 0, "{from} to {to} is texel for texel");
            }
        }
    }
}
