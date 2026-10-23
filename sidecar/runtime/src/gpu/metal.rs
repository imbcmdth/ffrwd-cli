//! The Metal objects under the process's wgpu device, when it opened on
//! Metal: how a picture VideoToolbox decoded becomes a frame's planes.
//!
//! A VideoToolbox decoder writes each picture into an IOSurface-backed
//! `CVPixelBuffer`, NV12 as two planes: luma, then interleaved chroma at
//! half size. CoreVideo's texture cache makes each plane a Metal texture
//! over the same surface, and each is handed to wgpu as a texture of the
//! process's device (`create_texture_from_hal`). Nothing is copied.
//!
//! libavcodec hands a picture over once VideoToolbox has written it, so
//! nothing waits before reading it. The picture and its textures are kept
//! after the frame is dropped until every command queued by then has run,
//! so the decoder's pool does not write the surface again while a pass
//! still reads it.
//!
//! A VideoToolbox encoder opened on a pool of IOSurface-backed pictures
//! ([`EncoderFeed`]) is written the same way the other way round: the
//! picture's planes are made Metal textures, the frame's NV12 planes are
//! copied into them on the device, and the copy is waited for before the
//! picture is sent, since VideoToolbox reads it with nothing to wait on.

use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use objc2_core_foundation::CFRetained;
use objc2_core_video::{
    kCVReturnSuccess, CVMetalTexture, CVMetalTextureCache, CVMetalTextureGetTexture, CVPixelBuffer,
    CVPixelBufferGetHeightOfPlane, CVPixelBufferGetPixelFormatType, CVPixelBufferGetPlaneCount,
    CVPixelBufferGetWidthOfPlane,
};
use objc2_metal::{MTLPixelFormat, MTLTextureType};
use std::ptr::NonNull;
use wasi_webgpu_wasmtime::reexports::{wgpu_core, wgpu_types};
use wasi_webgpu_wasmtime::SharedDevice;
use wgpu_core::global::Global;
use wgpu_core::id::TextureId;
use wgpu_hal::api::Metal;
use wgpu_types::{Extent3d, TextureAspect, TextureFormat};

use super::commands::Commands;
use super::frames::{GpuFrame, Layout, Plane, Target, Yuv};
use super::{shared, shared_device};
use crate::av::{self, DeviceFrame};

/// `'420v'` and `'420f'`: 8-bit NV12 in video and in full range.
const NV12_VIDEO: u32 = u32::from_be_bytes(*b"420v");
const NV12_FULL: u32 = u32::from_be_bytes(*b"420f");

/// Whether the process's device opened on Metal.
pub fn on_metal() -> bool {
    shared_device().is_some_and(|d| d.info.backend == wgpu_types::Backend::Metal)
}

/// The process's device as a decoder is opened on it; None unless it
/// opened on Metal.
pub fn av_device() -> Option<av::Shared> {
    on_metal().then_some(av::Shared::VideoToolbox)
}

/// A picture and the textures made over it, held until the queue no longer
/// reads them.
struct Held {
    _picture: DeviceFrame,
    _planes: Vec<CFRetained<CVMetalTexture>>,
}

// SAFETY: CoreVideo's objects are reference counted across threads, and the
// picture is only moved; nothing here is touched again until it is dropped.
unsafe impl Send for Held {}

/// A decoder's picture wrapped as a frame's planes: let go once the queue
/// has run every command queued while the frame was alive.
pub(super) struct Wrapped {
    textures: Vec<TextureId>,
    held: Held,
}

/// What a taken picture is: its planes, and the picture they are over.
pub(super) struct Taken {
    pub planes: Vec<Plane>,
    pub owned: Vec<TextureId>,
    pub wrapped: Option<Wrapped>,
}

impl Taken {
    /// What the line on stderr says of how pictures are taken.
    pub(super) fn how(&self) -> String {
        "frames wrapped".to_string()
    }
}

/// The texture cache one decoder's pictures are made textures through.
#[derive(Default)]
pub struct Taker {
    cache: Option<CFRetained<CVMetalTextureCache>>,
}

// SAFETY: a texture cache is used from one thread at a time, which `&mut
// self` on `take` guarantees, and is reference counted across threads.
unsafe impl Send for Taker {}

/// `cache`, made on the process's Metal device the first time.
fn cache<'a>(
    cache: &'a mut Option<CFRetained<CVMetalTextureCache>>,
    global: &Global,
    device: &SharedDevice,
) -> Result<&'a CVMetalTextureCache> {
    if cache.is_none() {
        // SAFETY: the device is only referenced, never destroyed here.
        let hal = unsafe { global.device_as_hal::<Metal>(device.device) }
            .ok_or_else(|| anyhow!("the process's device is not a Metal one"))?;
        let mut out: *mut CVMetalTextureCache = std::ptr::null_mut();
        // SAFETY: a live Metal device, and an out-pointer of ours.
        let status = unsafe {
            CVMetalTextureCache::create(None, None, hal.raw_device(), None, NonNull::from(&mut out))
        };
        let made = NonNull::new(out)
            .filter(|_| status == kCVReturnSuccess)
            .ok_or_else(|| anyhow!("CVMetalTextureCacheCreate failed ({status})"))?;
        // SAFETY: a Create function's result is owned by the caller.
        *cache = Some(unsafe { CFRetained::from_raw(made) });
    }
    Ok(cache.as_deref().expect("made above"))
}

/// The `CVPixelBuffer` a picture holds, checked to be NV12 in two planes.
fn nv12_buffer<'a>(picture: &'a DeviceFrame, whose: &str) -> Result<&'a CVPixelBuffer> {
    let raw = picture
        .cv_pixel_buffer()
        .ok_or_else(|| anyhow!("the {whose} picture is not a VideoToolbox one"))?;
    // SAFETY: the picture holds a reference to the pixel buffer, alive while
    // it is.
    let buffer: &CVPixelBuffer = unsafe { &*(raw as *const CVPixelBuffer) };
    let format = CVPixelBufferGetPixelFormatType(buffer);
    if format != NV12_VIDEO && format != NV12_FULL {
        bail!(
            "the {whose} picture is {}, and only NV12 ('420v' or '420f') is taken",
            fourcc_text(format)
        );
    }
    if CVPixelBufferGetPlaneCount(buffer) != 2 {
        bail!("the {whose} NV12 picture does not have two planes");
    }
    Ok(buffer)
}

impl Taker {
    /// Takes `picture`, an NV12 VideoToolbox picture.
    pub(super) fn take(
        &mut self,
        global: &Global,
        device: &SharedDevice,
        picture: DeviceFrame,
    ) -> Result<Taken> {
        let buffer = nv12_buffer(&picture, "decoder's")?;
        let cache = cache(&mut self.cache, global, device)?;
        let (width, height) = (picture.width, picture.height);
        let specs = [
            (
                0,
                MTLPixelFormat::R8Unorm,
                TextureFormat::R8Unorm,
                width,
                height,
                1,
            ),
            (
                1,
                MTLPixelFormat::RG8Unorm,
                TextureFormat::Rg8Unorm,
                width.div_ceil(2),
                height.div_ceil(2),
                2,
            ),
        ];
        let mut planes = Vec::with_capacity(2);
        let mut textures = Vec::with_capacity(2);
        let mut kept = Vec::with_capacity(2);
        for (index, mtl, format, w, h, bpp) in specs {
            let made = plane_texture(global, device, cache, buffer, index, mtl, format, READ);
            let (texture, cv) = match made {
                Ok(made) => made,
                Err(e) => {
                    for texture in textures {
                        global.texture_drop(texture);
                    }
                    return Err(e);
                }
            };
            textures.push(texture);
            kept.push(cv);
            planes.push(Plane {
                texture,
                aspect: TextureAspect::All,
                width: w,
                height: h,
                format,
                bytes_per_pixel: bpp,
            });
        }
        Ok(Taken {
            planes,
            owned: Vec::new(),
            wrapped: Some(Wrapped {
                textures,
                held: Held {
                    _picture: picture,
                    _planes: kept,
                },
            }),
        })
    }
}

/// What a decoder's plane is used for: read, and copied from.
const READ: wgpu_types::TextureUsages =
    wgpu_types::TextureUsages::TEXTURE_BINDING.union(wgpu_types::TextureUsages::COPY_SRC);

/// What an encoder's plane is used for: copied into.
const WRITE: wgpu_types::TextureUsages = wgpu_types::TextureUsages::COPY_DST;

/// Plane `index` of `buffer` as a Metal texture over the same surface, and
/// that texture as a wgpu texture of the process's device in `format`, for
/// `usage`.
#[allow(clippy::too_many_arguments)]
fn plane_texture(
    global: &Global,
    device: &SharedDevice,
    cache: &CVMetalTextureCache,
    buffer: &CVPixelBuffer,
    index: usize,
    mtl: MTLPixelFormat,
    format: TextureFormat,
    usage: wgpu_types::TextureUsages,
) -> Result<(TextureId, CFRetained<CVMetalTexture>)> {
    let width = CVPixelBufferGetWidthOfPlane(buffer, index);
    let height = CVPixelBufferGetHeightOfPlane(buffer, index);
    let mut out: *mut CVMetalTexture = std::ptr::null_mut();
    // SAFETY: a live cache and pixel buffer, and an out-pointer of ours.
    let status = unsafe {
        CVMetalTextureCache::create_texture_from_image(
            None,
            cache,
            buffer,
            None,
            mtl,
            width,
            height,
            index,
            NonNull::from(&mut out),
        )
    };
    let cv = NonNull::new(out)
        .filter(|_| status == kCVReturnSuccess)
        .ok_or_else(|| {
            anyhow!("plane {index}: CVMetalTextureCacheCreateTextureFromImage failed ({status})")
        })?;
    // SAFETY: a Create function's result is owned by the caller.
    let cv = unsafe { CFRetained::from_raw(cv) };
    let raw = CVMetalTextureGetTexture(&cv)
        .ok_or_else(|| anyhow!("plane {index}: the CoreVideo texture has no Metal texture"))?;
    let size = Extent3d {
        width: u32::try_from(width)?,
        height: u32::try_from(height)?,
        depth_or_array_layers: 1,
    };
    // SAFETY: a 2D, single-level, single-sample texture of this device in
    // `format`: a decoder's, written before the picture was handed over, or
    // an encoder's, which reads it only once the copy into it has run.
    let hal = unsafe {
        wgpu_hal::metal::Device::texture_from_raw(
            raw,
            format,
            MTLTextureType::Type2D,
            1,
            1,
            wgpu_hal::CopyExtent {
                width: size.width,
                height: size.height,
                depth: 1,
            },
        )
    };
    let (texture, err) = unsafe {
        global.create_texture_from_hal(
            Box::new(hal),
            device.device,
            &wgpu_types::TextureDescriptor {
                label: Some(std::borrow::Cow::Borrowed("ffrwd VideoToolbox plane")),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu_types::TextureDimension::D2,
                format,
                usage,
                view_formats: Vec::new(),
            },
            None,
        )
    };
    if let Some(e) = err {
        global.texture_drop(texture);
        bail!("plane {index}: wgpu refused the texture: {e}");
    }
    Ok((texture, cv))
}

impl Wrapped {
    /// Drops the frame's textures, and lets the picture go once the queue
    /// has run every command queued so far.
    pub(super) fn release(self, global: &Global, device: &SharedDevice) {
        let Wrapped { textures, held } = self;
        for texture in textures {
            global.texture_drop(texture);
        }
        global.queue_on_submitted_work_done(*device.queue, Box::new(move || drop(held)));
    }
}

/// Writes frames on the process's device into the pictures of a
/// VideoToolbox encoder's pool.
#[derive(Default)]
pub struct EncoderFeed {
    cache: Option<CFRetained<CVMetalTextureCache>>,
}

// SAFETY: as for `Taker`.
unsafe impl Send for EncoderFeed {}

impl EncoderFeed {
    /// Writes `frame` into `picture`, a picture of a VideoToolbox encoder's
    /// pool: the frame as NV12 ([`GpuFrame::converted`], an rgba frame made
    /// NV12 in `yuv` first), its two planes copied into the picture's on
    /// the device, and the copy waited for.
    pub fn feed(
        &mut self,
        frame: &Arc<GpuFrame>,
        yuv: Yuv,
        picture: &mut DeviceFrame,
    ) -> Result<()> {
        let (global, device) = (
            shared(),
            shared_device().ok_or_else(|| anyhow!("no GPU: the process has no device"))?,
        );
        if frame.backend() != wgpu_types::Backend::Metal {
            bail!(
                "the frame is on a {:?} device, not a Metal one",
                frame.backend()
            );
        }
        let nv12 = frame.converted(Target::new(
            Layout::Nv12,
            frame.width(),
            frame.height(),
            yuv,
        ))?;
        let [y, uv] = nv12.planes() else {
            bail!(
                "an NV12 frame has two planes, and this one has {}",
                nv12.planes().len()
            );
        };
        let buffer = nv12_buffer(picture, "encoder's")?;
        let cache = cache(&mut self.cache, global, device)?;
        let specs = [
            (y, 0, MTLPixelFormat::R8Unorm, TextureFormat::R8Unorm),
            (uv, 1, MTLPixelFormat::RG8Unorm, TextureFormat::Rg8Unorm),
        ];
        let mut targets = Vec::with_capacity(2);
        let mut kept = Vec::with_capacity(2);
        let written = (|| -> Result<()> {
            for (_, index, mtl, format) in specs {
                let (texture, cv) =
                    plane_texture(global, device, cache, buffer, index, mtl, format, WRITE)?;
                targets.push(texture);
                kept.push(cv);
            }
            let commands = Commands::open(global, device, "the copy")?;
            let encoder = commands.encoder();
            for ((plane, ..), target) in specs.iter().zip(&targets) {
                global
                    .command_encoder_copy_texture_to_texture(
                        encoder,
                        &wgpu_types::TexelCopyTextureInfo {
                            texture: plane.texture,
                            mip_level: 0,
                            origin: wgpu_types::Origin3d::ZERO,
                            aspect: plane.aspect,
                        },
                        &wgpu_types::TexelCopyTextureInfo {
                            texture: *target,
                            mip_level: 0,
                            origin: wgpu_types::Origin3d::ZERO,
                            aspect: TextureAspect::All,
                        },
                        &Extent3d {
                            width: plane.width,
                            height: plane.height,
                            depth_or_array_layers: 1,
                        },
                    )
                    .map_err(|e| anyhow!("copying a plane into the encoder's picture: {e}"))?;
            }
            let index = commands.submit(*device.queue)?;
            global
                .device_poll(
                    device.device,
                    wgpu_types::PollType::Wait {
                        submission_index: Some(index),
                        timeout: None,
                    },
                )
                .map_err(|e| anyhow!("waiting for the copy: {e}"))?;
            Ok(())
        })();
        for texture in targets {
            global.texture_drop(texture);
        }
        drop(kept);
        written
    }
}

/// A four-character code as text, `'420v'`, or its number where it is not
/// printable.
fn fourcc_text(code: u32) -> String {
    let bytes = code.to_be_bytes();
    if bytes.iter().all(|b| b.is_ascii_graphic() || *b == b' ') {
        format!("'{}'", String::from_utf8_lossy(&bytes))
    } else {
        format!("pixel format {code:#x}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_nv12_codes_are_corevideos() {
        assert_eq!(NV12_VIDEO, 0x3432_3076);
        assert_eq!(NV12_FULL, 0x3432_3066);
        assert_eq!(fourcc_text(NV12_VIDEO), "'420v'");
        assert_eq!(fourcc_text(0x7834_3230), "'x420'");
    }
}
