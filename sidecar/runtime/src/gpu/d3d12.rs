//! The D3D12 objects under the process's wgpu device, when it opened on
//! D3D12: the device a decoder is opened on, so the pictures it writes are
//! textures of the device every module uses, and how such a picture
//! becomes a frame's planes.
//!
//! A picture is taken one of two ways. Wrapped: the decoder's own NV12
//! texture is handed to wgpu as it is (`create_texture_from_hal`), one view
//! per plane, and nothing is copied; the decoder gets the texture back when
//! the frame is dropped. Copied: where wgpu cannot take the texture (the
//! device has no NV12 textures, or the decoder writes into a texture array),
//! its two planes are copied on the device into two textures the host
//! allocates, and the decoder gets its texture back at once.
//!
//! Either way the queue every module submits to waits on the decoder's sync
//! object first, so nothing reads a picture before it is written, and
//! signals it again once the host is done, so the decoder does not write
//! over a picture still being read.
//!
//! An encoder opened on the same device reads pictures of its own pool,
//! and a frame reaches one by a copy on the device ([`EncoderFeed`]): the
//! frame as NV12 (a decoder's picture as it is, a module's rgba converted
//! first) copied into the pool's texture, after which the queue signals
//! the picture's sync object, which the encoder waits on before reading.

use std::collections::VecDeque;
use std::mem::{transmute_copy, ManuallyDrop};
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use wasi_webgpu_wasmtime::reexports::{wgpu_core, wgpu_types};
use wasi_webgpu_wasmtime::SharedDevice;
use wgpu_core::global::Global;
use wgpu_core::id::TextureId;
use wgpu_hal::api::Dx12;
use wgpu_types::{Extent3d, TextureAspect, TextureFormat};
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
// A D3D12 sync object: what the decoder signals and the queue waits on.
use windows::Win32::Graphics::Direct3D12::ID3D12Fence as SyncObject;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::frames::{GpuFrame, Plane, Yuv};
use super::{shared, shared_device};
use crate::av::{self, DeviceFrame};

/// The process's `ID3D12Device`; None unless its device opened on D3D12.
pub fn device() -> Option<ID3D12Device> {
    let shared_dev = shared_device()?;
    if shared_dev.info.backend != wgpu_types::Backend::Dx12 {
        return None;
    }
    // SAFETY: the device is only referenced, never destroyed here.
    let hal = unsafe { shared().device_as_hal::<Dx12>(shared_dev.device) }?;
    Some(hal.raw_device().clone())
}

/// The process's device as a decoder or an encoder is opened on it; None
/// unless it opened on D3D12.
pub fn av_device() -> Option<av::Shared> {
    let device = device()?;
    // SAFETY: `into_raw` hands over the reference `device` held.
    Some(av::Shared::D3d12(unsafe {
        av::ComDevice::from_raw(device.into_raw())
    }))
}

/// The queue every module and the host submit to.
fn queue(global: &Global, device: &SharedDevice) -> Result<ID3D12CommandQueue> {
    // SAFETY: the queue is only referenced, never destroyed here.
    let hal = unsafe { global.queue_as_hal::<Dx12>(*device.queue) }
        .ok_or_else(|| anyhow!("the process's device is not a D3D12 one"))?;
    Ok(hal.as_raw().clone())
}

/// A COM object behind a raw pointer, with a reference of its own.
///
/// # Safety
/// `raw` is a live `T`.
unsafe fn borrowed<T: Interface + Clone>(raw: *mut std::os::raw::c_void) -> Result<T> {
    T::from_raw_borrowed(&raw).cloned().ok_or_else(|| {
        anyhow!(
            "the decoder's picture has a null {}",
            std::any::type_name::<T>()
        )
    })
}

/// How a picture was taken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Path {
    Wrapped,
    Copied,
}

/// A decoder's texture wrapped as a frame's: handed back when the frame
/// is dropped.
pub(super) struct Wrapped {
    texture: TextureId,
    picture: DeviceFrame,
}

/// What a taken picture is: its planes, the textures the frame owns, and
/// the decoder's texture where it was wrapped.
pub(super) struct Taken {
    pub planes: Vec<Plane>,
    pub owned: Vec<TextureId>,
    pub wrapped: Option<Wrapped>,
    pub path: Path,
    /// Why the picture was copied rather than wrapped.
    pub why: Option<String>,
}

/// One decoder's way of taking its pictures: whether wrapping still
/// works, and the command lists copies are recorded into.
pub struct Taker {
    wrap: Result<(), String>,
    copier: Option<Copier>,
}

impl Default for Taker {
    fn default() -> Taker {
        Taker {
            wrap: Ok(()),
            copier: None,
        }
    }
}

impl Taker {
    /// Takes `picture`, an NV12 picture of the process's device.
    pub(super) fn take(
        &mut self,
        global: &Global,
        device: &SharedDevice,
        mut picture: DeviceFrame,
    ) -> Result<Taken> {
        let raw = picture
            .d3d12()
            .ok_or_else(|| anyhow!("the decoder's picture is not a D3D12 one"))?;
        // SAFETY: the picture holds the texture and the sync object.
        let (resource, sync) = unsafe {
            (
                borrowed::<ID3D12Resource>(raw.texture)?,
                borrowed::<SyncObject>(raw.sync)?,
            )
        };
        let queue = queue(global, device)?;
        // SAFETY: COM calls on live objects.
        let desc = unsafe { resource.GetDesc() };
        if desc.Format != DXGI_FORMAT_NV12 {
            bail!(
                "the decoder's picture is DXGI format {}, and only NV12 is taken",
                desc.Format.0
            );
        }
        // Nothing the queue runs from here reads the picture before the
        // decoder has written it.
        // SAFETY: as above.
        unsafe { queue.Wait(&sync, raw.sync_value) }.context("waiting on the decoder")?;
        let (width, height) = (picture.width, picture.height);

        if self.wrap.is_ok() {
            if !device
                .features
                .contains(wgpu_types::Features::TEXTURE_FORMAT_NV12)
            {
                self.wrap = Err("the device has no NV12 textures".to_string());
            } else if raw.slice != 0 || desc.DepthOrArraySize != 1 {
                self.wrap = Err("the decoder writes a texture array".to_string());
            }
        }
        if self.wrap.is_ok() {
            match wrap(global, device, &resource, &desc) {
                Ok(texture) => {
                    let plane = |aspect, format, w, h, bpp| Plane {
                        texture,
                        aspect,
                        width: w,
                        height: h,
                        format,
                        bytes_per_pixel: bpp,
                    };
                    return Ok(Taken {
                        planes: vec![
                            plane(
                                TextureAspect::Plane0,
                                TextureFormat::R8Unorm,
                                width,
                                height,
                                1,
                            ),
                            plane(
                                TextureAspect::Plane1,
                                TextureFormat::Rg8Unorm,
                                width.div_ceil(2),
                                height.div_ceil(2),
                                2,
                            ),
                        ],
                        owned: Vec::new(),
                        wrapped: Some(Wrapped { texture, picture }),
                        path: Path::Wrapped,
                        why: None,
                    });
                }
                Err(e) => self.wrap = Err(format!("wgpu refused the texture: {e:#}")),
            }
        }

        let copier = match &mut self.copier {
            Some(copier) => copier,
            None => self.copier.insert(Copier::new(device_of(&resource)?)?),
        };
        let (y, uv) = copier.copy(&queue, &resource, raw.slice, &desc, width, height)?;
        // The copy is on the queue; the decoder may write the texture again
        // once it has run.
        let next = raw.sync_value + 1;
        // SAFETY: as above.
        unsafe { queue.Signal(&sync, next) }.context("signalling the decoder")?;
        picture.read_until(next);
        drop(picture);

        let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
        let y = adopt(global, device, y, TextureFormat::R8Unorm, width, height)?;
        let uv = match adopt(global, device, uv, TextureFormat::Rg8Unorm, cw, ch) {
            Ok(uv) => uv,
            Err(e) => {
                global.texture_drop(y);
                return Err(e);
            }
        };
        let plane = |texture, format, w, h, bpp| Plane {
            texture,
            aspect: TextureAspect::All,
            width: w,
            height: h,
            format,
            bytes_per_pixel: bpp,
        };
        Ok(Taken {
            planes: vec![
                plane(y, TextureFormat::R8Unorm, width, height, 1),
                plane(uv, TextureFormat::Rg8Unorm, cw, ch, 2),
            ],
            owned: vec![y, uv],
            wrapped: None,
            path: Path::Copied,
            why: self.wrap.clone().err(),
        })
    }
}

fn device_of(resource: &ID3D12Resource) -> Result<ID3D12Device> {
    let mut device: Option<ID3D12Device> = None;
    // SAFETY: a live resource.
    unsafe { resource.GetDevice(&mut device) }.context("the picture's device")?;
    device.ok_or_else(|| anyhow!("the picture has no device"))
}

/// The decoder's NV12 texture as a wgpu texture of the process's device,
/// sampled through one view per plane.
fn wrap(
    global: &Global,
    device: &SharedDevice,
    resource: &ID3D12Resource,
    desc: &D3D12_RESOURCE_DESC,
) -> Result<TextureId> {
    let size = Extent3d {
        width: u32::try_from(desc.Width).context("the picture's width")?,
        height: desc.Height,
        depth_or_array_layers: 1,
    };
    // SAFETY: the resource is a 2D, single-level, single-sample NV12
    // texture of this device, written by the decoder before any command
    // that reads it runs (the queue waits on its sync object).
    let hal = unsafe {
        wgpu_hal::dx12::Device::texture_from_raw(
            resource.clone(),
            TextureFormat::NV12,
            wgpu_types::TextureDimension::D2,
            size,
            1,
            1,
        )
    };
    let (texture, err) = unsafe {
        global.create_texture_from_hal(
            Box::new(hal),
            device.device,
            &wgpu_types::TextureDescriptor {
                label: Some(std::borrow::Cow::Borrowed("ffrwd decoded picture")),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu_types::TextureDimension::D2,
                format: TextureFormat::NV12,
                usage: wgpu_types::TextureUsages::TEXTURE_BINDING,
                view_formats: Vec::new(),
            },
            None,
        )
    };
    if let Some(e) = err {
        global.texture_drop(texture);
        bail!("{e}");
    }
    Ok(texture)
}

/// A texture the host made on the device, handed to wgpu as `format`.
fn adopt(
    global: &Global,
    device: &SharedDevice,
    resource: ID3D12Resource,
    format: TextureFormat,
    width: u32,
    height: u32,
) -> Result<TextureId> {
    let size = Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    // SAFETY: a 2D, single-level texture of this device in `format`, in
    // the common state, written by a copy already on the queue.
    let hal = unsafe {
        wgpu_hal::dx12::Device::texture_from_raw(
            resource,
            format,
            wgpu_types::TextureDimension::D2,
            size,
            1,
            1,
        )
    };
    let (texture, err) = unsafe {
        global.create_texture_from_hal(
            Box::new(hal),
            device.device,
            &wgpu_types::TextureDescriptor {
                label: Some(std::borrow::Cow::Borrowed("ffrwd decoded plane")),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu_types::TextureDimension::D2,
                format,
                usage: wgpu_types::TextureUsages::TEXTURE_BINDING
                    | wgpu_types::TextureUsages::COPY_SRC,
                view_formats: Vec::new(),
            },
            None,
        )
    };
    if let Some(e) = err {
        global.texture_drop(texture);
        bail!("a copied {format:?} plane: {e}");
    }
    Ok(texture)
}

impl Wrapped {
    /// Hands the decoder's texture back: returned to the common state the
    /// decoder expects it in, and its sync object signalled once every
    /// command the host and the modules queued has run.
    pub(super) fn release(mut self, global: &Global, device: &SharedDevice) {
        let result = (|| -> Result<()> {
            let (encoder, err) = global.device_create_command_encoder(
                device.device,
                &wgpu_types::CommandEncoderDescriptor { label: None },
                None,
            );
            if let Some(e) = err {
                bail!("{e}");
            }
            global
                .command_encoder_transition_resources(
                    encoder,
                    std::iter::empty(),
                    std::iter::once(wgpu_types::TextureTransition {
                        texture: self.texture,
                        selector: None,
                        state: wgpu_types::TextureUses::PRESENT,
                    }),
                )
                .map_err(|e| anyhow!("{e}"))?;
            let (commands, err) = global.command_encoder_finish(
                encoder,
                &wgpu_types::CommandBufferDescriptor { label: None },
                None,
            );
            if let Some((_, e)) = err {
                bail!("{e}");
            }
            global
                .queue_submit(*device.queue, &[commands])
                .map_err(|(_, e)| anyhow!("{e}"))?;
            let raw = self
                .picture
                .d3d12()
                .ok_or_else(|| anyhow!("not a D3D12 picture"))?;
            // SAFETY: the picture still holds its sync object.
            let sync = unsafe { borrowed::<SyncObject>(raw.sync)? };
            let next = raw.sync_value + 1;
            // SAFETY: COM calls on live objects.
            unsafe { queue(global, device)?.Signal(&sync, next) }?;
            self.picture.read_until(next);
            Ok(())
        })();
        if let Err(e) = result {
            eprintln!("[gpu] handing a decoded picture back to the decoder: {e:#}");
        }
        global.texture_drop(self.texture);
    }
}

/// A command list and its allocator, free once the copier's sync object
/// reaches `done_at`, and the frames its copies read, kept until then.
struct Recorder {
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    done_at: u64,
    held: Vec<Arc<GpuFrame>>,
}

/// Records and submits copies on the device: of a decoder's pictures wgpu
/// cannot wrap, and of frames into an encoder's pictures.
struct Copier {
    device: ID3D12Device,
    sync: SyncObject,
    submitted: u64,
    busy: VecDeque<Recorder>,
}

// SAFETY: D3D12 objects are free-threaded; the copier is used from one
// thread at a time, the decoder's.
unsafe impl Send for Copier {}

/// A transition of one subresource, read while `resource` lives.
fn barrier(
    resource: &ID3D12Resource,
    subresource: u32,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer, no reference taken or dropped.
                pResource: unsafe { transmute_copy(resource) },
                Subresource: subresource,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

fn location(resource: &ID3D12Resource, subresource: u32) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as in `barrier`.
        pResource: unsafe { transmute_copy(resource) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: subresource,
        },
    }
}

impl Copier {
    fn new(device: ID3D12Device) -> Result<Copier> {
        // SAFETY: a COM call on a live device.
        let sync: SyncObject = unsafe { device.CreateFence(0, Default::default()) }
            .context("a sync object for copies")?;
        Ok(Copier {
            device,
            sync,
            submitted: 0,
            busy: VecDeque::new(),
        })
    }

    /// A command list ready to record: one whose last copies have run, or
    /// a new one.
    fn recorder(&mut self) -> Result<Recorder> {
        // SAFETY: COM calls on live objects.
        unsafe {
            let done = self.sync.GetCompletedValue();
            for recorder in self.busy.iter_mut().filter(|r| r.done_at <= done) {
                recorder.held.clear();
            }
            if let Some(recorder) = self.busy.pop_front_if(|r| r.done_at <= done) {
                recorder.allocator.Reset()?;
                recorder.list.Reset(&recorder.allocator, None)?;
                return Ok(recorder);
            }
            let allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            Ok(Recorder {
                allocator,
                list,
                done_at: 0,
                held: Vec::new(),
            })
        }
    }

    /// Closes `recorder`'s list, runs it on `queue`, and keeps it busy
    /// until it has run.
    fn submit(&mut self, queue: &ID3D12CommandQueue, mut recorder: Recorder) -> Result<()> {
        // SAFETY: COM calls on live objects; every resource the list names
        // outlives its run (the caller keeps them in `recorder.held` or
        // owns them past it).
        unsafe {
            recorder.list.Close()?;
            let runnable: ID3D12CommandList = recorder.list.cast()?;
            queue.ExecuteCommandLists(&[Some(runnable)]);
            self.submitted += 1;
            queue.Signal(&self.sync, self.submitted)?;
        }
        recorder.done_at = self.submitted;
        self.busy.push_back(recorder);
        Ok(())
    }

    /// A new texture in the common state, for one plane.
    fn plane(&self, format: DXGI_FORMAT, width: u32, height: u32) -> Result<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Alignment: 0,
            Width: u64::from(width),
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };
        let mut made: Option<ID3D12Resource> = None;
        // SAFETY: a COM call on a live device with valid descriptions.
        unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut made,
            )
        }
        .with_context(|| format!("a {width}x{height} plane texture"))?;
        made.ok_or_else(|| anyhow!("CreateCommittedResource made nothing"))
    }

    /// Copies the visible `width` x `height` of the NV12 picture in array
    /// slice `slice` of `source` into two new textures, luma and chroma,
    /// on `queue`, leaving every texture in the common state.
    fn copy(
        &mut self,
        queue: &ID3D12CommandQueue,
        source: &ID3D12Resource,
        slice: u32,
        desc: &D3D12_RESOURCE_DESC,
        width: u32,
        height: u32,
    ) -> Result<(ID3D12Resource, ID3D12Resource)> {
        let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
        let y = self.plane(DXGI_FORMAT_R8_UNORM, width, height)?;
        let uv = self.plane(DXGI_FORMAT_R8G8_UNORM, cw, ch)?;
        let mips = u32::from(desc.MipLevels.max(1));
        let layers = u32::from(desc.DepthOrArraySize.max(1));
        let source_plane = |plane: u32| slice * mips + plane * mips * layers;
        let recorder = self.recorder()?;
        let list = &recorder.list;
        let common = D3D12_RESOURCE_STATE_COMMON;
        let copy_src = D3D12_RESOURCE_STATE_COPY_SOURCE;
        let copy_dst = D3D12_RESOURCE_STATE_COPY_DEST;
        let before = [
            barrier(source, source_plane(0), common, copy_src),
            barrier(source, source_plane(1), common, copy_src),
            barrier(&y, 0, common, copy_dst),
            barrier(&uv, 0, common, copy_dst),
        ];
        let after = [
            barrier(source, source_plane(0), copy_src, common),
            barrier(source, source_plane(1), copy_src, common),
            barrier(&y, 0, copy_dst, common),
            barrier(&uv, 0, copy_dst, common),
        ];
        let region = |w: u32, h: u32| D3D12_BOX {
            left: 0,
            top: 0,
            front: 0,
            right: w,
            bottom: h,
            back: 1,
        };
        // SAFETY: COM calls on live objects; every resource named outlives
        // the recording, and the list is closed and submitted below.
        unsafe {
            list.ResourceBarrier(&before);
            list.CopyTextureRegion(
                &location(&y, 0),
                0,
                0,
                0,
                &location(source, source_plane(0)),
                Some(&region(width, height)),
            );
            list.CopyTextureRegion(
                &location(&uv, 0),
                0,
                0,
                0,
                &location(source, source_plane(1)),
                Some(&region(cw, ch)),
            );
            list.ResourceBarrier(&after);
        }
        self.submit(queue, recorder)?;
        Ok((y, uv))
    }
}

impl Drop for Copier {
    /// Waits for the copies still running, so nothing they read or write
    /// is released under them.
    fn drop(&mut self) {
        // SAFETY: COM calls on live objects; a null event blocks until the
        // value is reached.
        unsafe {
            if self.sync.GetCompletedValue() < self.submitted {
                let _ = self.sync.SetEventOnCompletion(
                    self.submitted,
                    windows::Win32::Foundation::HANDLE::default(),
                );
            }
        }
    }
}

/// Writes frames on the process's device into an encoder's pictures, one
/// copy on the device per frame and no byte through the host.
#[derive(Default)]
pub struct EncoderFeed {
    copier: Option<Copier>,
}

/// How a frame was copied into an encoder's picture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Copied {
    /// The decoder's NV12 texture as a whole, onto one of the same size.
    Whole,
    /// Each plane into the picture's own.
    Planes,
}

impl EncoderFeed {
    /// Copies `frame` into `picture`, a picture of an encoder's pool on the
    /// process's device: as it is held for a decoder's picture, else made
    /// NV12 first ([`GpuFrame::nv12_planes`], an rgba frame with `yuv`).
    /// The queue every module submits to runs the copy after what it has
    /// already been given, then signals the picture's sync object, and the
    /// picture says the value it waits for. The frame is kept until the
    /// copy has run.
    pub fn feed(
        &mut self,
        frame: &Arc<GpuFrame>,
        yuv: Yuv,
        picture: &mut DeviceFrame,
    ) -> Result<Copied> {
        let (global, device) = (
            shared(),
            shared_device().ok_or_else(|| anyhow!("no GPU: the process has no device"))?,
        );
        if frame.backend() != wgpu_types::Backend::Dx12 {
            bail!(
                "the frame is on a {:?} device, not a D3D12 one",
                frame.backend()
            );
        }
        let planes = frame.nv12_planes(yuv)?;
        let [y, uv] = planes.as_slice() else {
            bail!(
                "an NV12 frame has two planes, and this one has {}",
                planes.len()
            );
        };
        let raw = picture
            .d3d12()
            .ok_or_else(|| anyhow!("the encoder's picture is not a D3D12 one"))?;
        // SAFETY: the picture holds its texture and sync object.
        let (target, sync) = unsafe {
            (
                borrowed::<ID3D12Resource>(raw.texture)?,
                borrowed::<SyncObject>(raw.sync)?,
            )
        };
        // SAFETY: a COM call on a live resource.
        let target_desc = unsafe { target.GetDesc() };
        if target_desc.Format != DXGI_FORMAT_NV12 || raw.slice != 0 {
            bail!("the encoder's picture is not a texture of its own in NV12");
        }

        // wgpu leaves the planes in the common state the copy starts from,
        // after every command already given that writes them.
        let textures: Vec<TextureId> = if y.texture == uv.texture {
            vec![y.texture]
        } else {
            vec![y.texture, uv.texture]
        };
        hand_over(global, device, &textures)?;
        let resource = |texture: TextureId| -> Result<ID3D12Resource> {
            // SAFETY: the texture is the frame's, alive while it is.
            let hal = unsafe { global.texture_as_hal::<Dx12>(texture) }
                .ok_or_else(|| anyhow!("a plane of the frame is not a D3D12 texture"))?;
            // SAFETY: only referenced; the frame keeps the texture alive.
            Ok(unsafe { hal.raw_resource() }.clone())
        };
        let y_source = resource(y.texture)?;
        let uv_source = if y.texture == uv.texture {
            None
        } else {
            Some(resource(uv.texture)?)
        };

        let queue = queue(global, device)?;
        // The encoder is done with the texture once its own signal of it
        // has been reached.
        // SAFETY: COM calls on live objects.
        unsafe { queue.Wait(&sync, raw.sync_value) }.context("waiting on the encoder")?;
        let copier = match &mut self.copier {
            Some(copier) => copier,
            None => self.copier.insert(Copier::new(device_of(&target)?)?),
        };
        let mut recorder = copier.recorder()?;
        let common = D3D12_RESOURCE_STATE_COMMON;
        let copy_src = D3D12_RESOURCE_STATE_COPY_SOURCE;
        let copy_dst = D3D12_RESOURCE_STATE_COPY_DEST;
        let all = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        let (width, height) = (picture.width, picture.height);
        let region = |w: u32, h: u32| D3D12_BOX {
            left: 0,
            top: 0,
            front: 0,
            right: w,
            bottom: h,
            back: 1,
        };
        let copied = {
            let list = &recorder.list;
            let same_size = || {
                // SAFETY: a COM call on a live resource.
                let source = unsafe { y_source.GetDesc() };
                source.Width == target_desc.Width
                    && source.Height == target_desc.Height
                    && source.Format == target_desc.Format
                    && source.MipLevels == target_desc.MipLevels
                    && source.DepthOrArraySize == target_desc.DepthOrArraySize
            };
            match &uv_source {
                None if same_size() => {
                    // SAFETY: COM calls on live objects; both resources
                    // outlive the list's run (the frame is held below, the
                    // picture until the encoder is done with it).
                    unsafe {
                        list.ResourceBarrier(&[
                            barrier(&y_source, all, common, copy_src),
                            barrier(&target, all, common, copy_dst),
                        ]);
                        list.CopyResource(&target, &y_source);
                        list.ResourceBarrier(&[
                            barrier(&y_source, all, copy_src, common),
                            barrier(&target, all, copy_dst, common),
                        ]);
                    }
                    Copied::Whole
                }
                _ => {
                    // Luma from plane 0 (or a texture of its own), chroma
                    // from plane 1 (or its own), into the picture's two.
                    let (uv_resource, uv_sub) = match &uv_source {
                        Some(r) => (r, 0),
                        None => (&y_source, 1),
                    };
                    let mut before = vec![
                        barrier(&y_source, all, common, copy_src),
                        barrier(&target, all, common, copy_dst),
                    ];
                    let mut after = vec![
                        barrier(&y_source, all, copy_src, common),
                        barrier(&target, all, copy_dst, common),
                    ];
                    if let Some(r) = &uv_source {
                        before.push(barrier(r, all, common, copy_src));
                        after.push(barrier(r, all, copy_src, common));
                    }
                    // SAFETY: as above.
                    unsafe {
                        list.ResourceBarrier(&before);
                        list.CopyTextureRegion(
                            &location(&target, 0),
                            0,
                            0,
                            0,
                            &location(&y_source, 0),
                            Some(&region(width, height)),
                        );
                        list.CopyTextureRegion(
                            &location(&target, 1),
                            0,
                            0,
                            0,
                            &location(uv_resource, uv_sub),
                            Some(&region(width.div_ceil(2), height.div_ceil(2))),
                        );
                        list.ResourceBarrier(&after);
                    }
                    Copied::Planes
                }
            }
        };
        recorder.held.push(Arc::clone(frame));
        copier.submit(&queue, recorder)?;
        let next = raw.sync_value + 1;
        // SAFETY: COM calls on live objects.
        unsafe { queue.Signal(&sync, next) }.context("signalling the encoder")?;
        picture.written_at(next);
        Ok(copied)
    }
}

/// Puts `textures` in the common state on the queue, after every command
/// already submitted that uses them, for D3D12 commands of the host's own
/// to read next.
fn hand_over(global: &Global, device: &SharedDevice, textures: &[TextureId]) -> Result<()> {
    let (encoder, err) = global.device_create_command_encoder(
        device.device,
        &wgpu_types::CommandEncoderDescriptor { label: None },
        None,
    );
    if let Some(e) = err {
        bail!("a command encoder for the copy: {e}");
    }
    global
        .command_encoder_transition_resources(
            encoder,
            std::iter::empty(),
            textures
                .iter()
                .map(|&texture| wgpu_types::TextureTransition {
                    texture,
                    selector: None,
                    state: wgpu_types::TextureUses::PRESENT,
                }),
        )
        .map_err(|e| anyhow!("{e}"))?;
    let (commands, err) = global.command_encoder_finish(
        encoder,
        &wgpu_types::CommandBufferDescriptor { label: None },
        None,
    );
    if let Some((_, e)) = err {
        bail!("{e}");
    }
    global
        .queue_submit(*device.queue, &[commands])
        .map_err(|(_, e)| anyhow!("{e}"))?;
    Ok(())
}
