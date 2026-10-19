//! The Vulkan objects under the process's wgpu device on Linux: the device
//! itself, opened here so that it carries what libav's Vulkan decoder and the
//! route to NVENC need, and how the pictures they read and write become, and
//! are made from, a frame's planes.
//!
//! The device. wgpu opens a device with the extensions its own features
//! need and one queue. A decoder on the same device needs the video
//! extensions, timeline semaphores, synchronization2 and queues of the
//! video decode family, and libav must have queues no one else submits to.
//! So the device is created here with `ash` on the physical device wgpu's
//! adapter is, with wgpu's extensions and features and those, and handed to
//! wgpu-hal (`Adapter::device_from_raw`) and then to wgpu-core
//! (`create_device_from_hal`). wgpu submits to the first queue of the
//! graphics family; libav is given one queue of each of the transfer,
//! compute, video decode and video encode families. The graphics family is
//! listed to libav too, with no use it would pick it for, because libav
//! shares its images among exactly the families it is given. A card
//! without the video extensions still gets the device, with no decoder on
//! it.
//!
//! Decoding. A picture the decoder wrote is copied on the device into two
//! textures of the device, luma and chroma, on wgpu's queue, so the copy
//! runs in order with what reads the frame. The textures are made by
//! wgpu-hal and handed to wgpu-core, which counts such a texture as
//! written; one wgpu-core made itself would be cleared before its first
//! read, over what the copy wrote.
//! The picture is not wrapped as it is: wgpu-core takes a texture made from
//! a raw image as uninitialised, and its first barrier would start from
//! `VK_IMAGE_LAYOUT_UNDEFINED`, which lets the driver discard the picture;
//! and the decoder keeps reading a picture it holds as a reference, in its
//! own layout and on its own queue, while the frame would be read on
//! wgpu's. The copy reads the picture in the layout libav left it in and
//! leaves it in the copy's, says so to libav, and moves the picture's
//! timeline semaphore on by one once it has run. wgpu's submit waits on no
//! semaphore, and a semaphore it is asked to signal is taken up by
//! whichever submit comes next, which may be another thread's; so the copy
//! is a submission of the host's own on wgpu's queue ([`submit_raw`]),
//! made while wgpu-core's lock on its submissions is held. It waits on the
//! device for the decoder's write and signals the picture's semaphore
//! itself, and wgpu's later submissions run after it. The decoder waits
//! for that value before it writes the picture again; the copy's targets
//! are kept until it has run ([`Signaller`]).
//!
//! Encoding. NVENC reads CUDA pictures. An encoder fed from this device is
//! given a pool of Vulkan pictures, each plane an image of its own, which
//! wgpu writes: the image is wrapped as a texture (its contents are
//! overwritten, so starting from an undefined layout loses nothing), the
//! frame's NV12 planes are copied into it, its layout and access are told
//! to libav, and its semaphore is signalled once the copy has run, as for
//! the decoder. libavutil then copies the picture into one of the
//! encoder's CUDA pictures on the card, the two APIs sharing the image's
//! memory and semaphore.

use std::collections::VecDeque;
use std::ffi::{c_void, CStr, CString};
use std::sync::{mpsc, Arc, Mutex, OnceLock};

use anyhow::{anyhow, bail, Context, Result};
use ash::vk;
use ash::vk::Handle;
use wasi_webgpu_wasmtime::reexports::{wgpu_core, wgpu_types};
use wasi_webgpu_wasmtime::SharedDevice;
use wgpu_core::global::Global;
use wgpu_core::id::{AdapterId, DeviceId, QueueId, TextureId};
use wgpu_hal::api::Vulkan;
use wgpu_types::{Extent3d, TextureAspect, TextureFormat};

use super::frames::{GpuFrame, Layout, Plane, Target, Yuv};
use super::{shared, shared_device};
use crate::av::{self, DeviceFrame};

/// What the device was opened with, kept for libav and for the copies.
struct Opened {
    instance: ash::Instance,
    device: ash::Device,
    physical: vk::PhysicalDevice,
    get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
    instance_extensions: Vec<CString>,
    device_extensions: Vec<CString>,
    /// The features the device was created with, for libav: a
    /// `VkPhysicalDeviceFeatures2` whose chain lives as long as the process.
    features: *const c_void,
    /// The families libav is given; empty where it is given none.
    families: Vec<av::QueueFamily>,
    /// Why libav is given none, where it is not.
    no_codec: Option<String>,
    /// The family of wgpu's queue, which the host's own command buffers
    /// are made for.
    graphics: u32,
}

// SAFETY: the handles are Vulkan's, usable from any thread; the features
// pointer names leaked, never-written memory.
unsafe impl Send for Opened {}
unsafe impl Sync for Opened {}

static OPENED: OnceLock<Opened> = OnceLock::new();

/// The video extensions a decoder or an encoder on the device uses, enabled
/// where the card has them.
const VIDEO_EXTENSIONS: &[&CStr] = &[
    ash::khr::video_queue::NAME,
    ash::khr::video_decode_queue::NAME,
    ash::khr::video_decode_h264::NAME,
    ash::khr::video_decode_h265::NAME,
    ash::khr::video_decode_av1::NAME,
    ash::khr::video_encode_queue::NAME,
    ash::khr::video_encode_h264::NAME,
    ash::khr::video_encode_h265::NAME,
    ash::khr::video_maintenance1::NAME,
];

/// The other extensions libav uses, enabled where the card has them: the
/// memory and semaphore exports CUDA imports, and push descriptors.
const OTHER_EXTENSIONS: &[&CStr] = &[
    ash::khr::external_memory_fd::NAME,
    ash::khr::external_semaphore_fd::NAME,
    ash::ext::external_memory_dma_buf::NAME,
    ash::khr::push_descriptor::NAME,
    ash::khr::synchronization2::NAME,
];

/// The queues of one family to create, and what libav is told of it.
struct FamilyPlan {
    index: u32,
    queues: u32,
    /// What libav may use the family for; None where libav is not told of
    /// it.
    for_libav: Option<(vk::QueueFlags, vk::VideoCodecOperationFlagsKHR)>,
}

/// Which families get queues: wgpu's, and one each for libav.
fn plan_families(
    families: &[(vk::QueueFlags, u32, vk::VideoCodecOperationFlagsKHR)],
    video: bool,
) -> Result<(u32, Vec<FamilyPlan>, Option<String>)> {
    let graphics = families
        .iter()
        .position(|(flags, ..)| flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE))
        .ok_or_else(|| anyhow!("the adapter has no queue family for graphics and compute"))?
        as u32;
    let find = |want: vk::QueueFlags, without: vk::QueueFlags| {
        families
            .iter()
            .enumerate()
            .position(|(i, (flags, count, _))| {
                i as u32 != graphics
                    && *count > 0
                    && flags.contains(want)
                    && !flags.intersects(without)
            })
            .map(|i| i as u32)
    };
    let video_flags = vk::QueueFlags::VIDEO_DECODE_KHR | vk::QueueFlags::VIDEO_ENCODE_KHR;
    let compute = find(
        vk::QueueFlags::COMPUTE,
        vk::QueueFlags::GRAPHICS | video_flags,
    );
    let transfer = find(
        vk::QueueFlags::TRANSFER,
        vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | video_flags,
    );
    let decode = find(vk::QueueFlags::VIDEO_DECODE_KHR, vk::QueueFlags::empty());
    let encode = find(vk::QueueFlags::VIDEO_ENCODE_KHR, vk::QueueFlags::empty());

    let mut plans = vec![FamilyPlan {
        index: graphics,
        queues: 1,
        for_libav: None,
    }];
    let no_codec = if !video {
        Some("the device has no video extensions".to_string())
    } else if decode.is_none() {
        Some("the device has no video decode queue".to_string())
    } else if compute.is_none() {
        Some("the device has no compute queue but wgpu's".to_string())
    } else {
        None
    };
    if no_codec.is_some() {
        return Ok((graphics, plans, no_codec));
    }
    let caps = |i: u32| families[i as usize].2;
    // libav picks the first family listed for each use, so the transfer
    // family comes before the compute family, which also transfers.
    if let Some(t) = transfer {
        plans.push(FamilyPlan {
            index: t,
            queues: 1,
            for_libav: Some((
                vk::QueueFlags::TRANSFER,
                vk::VideoCodecOperationFlagsKHR::NONE,
            )),
        });
    }
    if let Some(c) = compute {
        plans.push(FamilyPlan {
            index: c,
            queues: 1,
            for_libav: Some((
                vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER,
                vk::VideoCodecOperationFlagsKHR::NONE,
            )),
        });
    }
    if let Some(d) = decode {
        plans.push(FamilyPlan {
            index: d,
            queues: 1,
            for_libav: Some((vk::QueueFlags::VIDEO_DECODE_KHR, caps(d))),
        });
    }
    if let Some(e) = encode.filter(|e| Some(*e) != decode) {
        plans.push(FamilyPlan {
            index: e,
            queues: 1,
            for_libav: Some((vk::QueueFlags::VIDEO_ENCODE_KHR, caps(e))),
        });
    }
    // Listed last and only as graphics, which libav looks for nothing by:
    // its images are then shared with the family wgpu's queue is in.
    plans[0].for_libav = Some((
        vk::QueueFlags::GRAPHICS,
        vk::VideoCodecOperationFlagsKHR::NONE,
    ));
    Ok((graphics, plans, None))
}

/// Opens the process's device on `adapter`, a Vulkan adapter of the shared
/// instance, with `features` and `limits`, as wgpu would and with what libav
/// needs besides.
pub fn open(
    global: &Global,
    adapter: AdapterId,
    desc: &wgpu_core::device::DeviceDescriptor<'_>,
) -> Result<(DeviceId, QueueId)> {
    // SAFETY: the instance and adapter are only referenced here.
    let hal_instance = unsafe { global.instance_as_hal::<Vulkan>() }
        .ok_or_else(|| anyhow!("the shared instance has no Vulkan backend"))?;
    let shared_instance = hal_instance.shared_instance();
    let instance = shared_instance.raw_instance().clone();
    let entry = shared_instance.entry().clone();
    let instance_extensions: Vec<CString> = shared_instance
        .extensions()
        .iter()
        .map(|e| (*e).to_owned())
        .collect();
    let hal_adapter = unsafe { global.adapter_as_hal::<Vulkan>(adapter) }
        .ok_or_else(|| anyhow!("the adapter is not a Vulkan one"))?;
    let physical = hal_adapter.raw_physical_device();
    let features = desc.required_features;

    // SAFETY: queries of a live physical device.
    let available: Vec<CString> =
        unsafe { instance.enumerate_device_extension_properties(physical) }
            .context("listing the device's extensions")?
            .iter()
            .filter_map(|e| e.extension_name_as_c_str().ok().map(CStr::to_owned))
            .collect();
    let has = |name: &CStr| available.iter().any(|a| a.as_c_str() == name);
    let mut extensions = hal_adapter.required_device_extensions(features);
    let video = has(ash::khr::video_queue::NAME) && has(ash::khr::video_decode_queue::NAME);
    for &name in OTHER_EXTENSIONS
        .iter()
        .chain(if video { VIDEO_EXTENSIONS } else { &[] })
    {
        if has(name) && !extensions.contains(&name) {
            extensions.push(name);
        }
    }
    let video_maintenance1 = extensions.contains(&ash::khr::video_maintenance1::NAME);

    // The queue families, with what each decodes or encodes.
    let families: Vec<(vk::QueueFlags, u32, vk::VideoCodecOperationFlagsKHR)> = unsafe {
        let count = instance.get_physical_device_queue_family_properties2_len(physical);
        let mut video_props = vec![vk::QueueFamilyVideoPropertiesKHR::default(); count];
        let mut props: Vec<vk::QueueFamilyProperties2> = video_props
            .iter_mut()
            .map(|v| {
                let p = vk::QueueFamilyProperties2::default();
                if video {
                    p.push_next(v)
                } else {
                    p
                }
            })
            .collect();
        instance.get_physical_device_queue_family_properties2(physical, &mut props);
        let flags: Vec<(vk::QueueFlags, u32)> = props
            .iter()
            .map(|p| {
                (
                    p.queue_family_properties.queue_flags,
                    p.queue_family_properties.queue_count,
                )
            })
            .collect();
        drop(props);
        flags
            .into_iter()
            .zip(video_props.iter().map(|v| v.video_codec_operations))
            .map(|((f, n), v)| (f, n, v))
            .collect()
    };
    let (graphics, plans, no_codec) = plan_families(&families, video)?;

    // The features libav needs beyond wgpu's, where the card has them.
    let (mut sync2, mut ycbcr, mut timeline) = (
        vk::PhysicalDeviceSynchronization2Features::default(),
        vk::PhysicalDeviceSamplerYcbcrConversionFeatures::default(),
        vk::PhysicalDeviceTimelineSemaphoreFeatures::default(),
    );
    let mut maintenance = vk::PhysicalDeviceVideoMaintenance1FeaturesKHR::default();
    unsafe {
        let mut query = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut sync2)
            .push_next(&mut ycbcr)
            .push_next(&mut timeline);
        instance.get_physical_device_features2(physical, &mut query);
        if video_maintenance1 {
            let mut query = vk::PhysicalDeviceFeatures2::default().push_next(&mut maintenance);
            instance.get_physical_device_features2(physical, &mut query);
        }
    }
    let mut phd = hal_adapter.physical_device_features(&extensions, features);
    let core = phd.get_core();

    const PRIORITIES: [f32; 2] = [1.0, 1.0];
    let queue_infos: Vec<vk::DeviceQueueCreateInfo> = plans
        .iter()
        .map(|p| {
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(p.index)
                .queue_priorities(&PRIORITIES[..p.queues as usize])
        })
        .collect();
    let names: Vec<*const std::os::raw::c_char> = extensions.iter().map(|e| e.as_ptr()).collect();
    // wgpu's own feature structs, then ours: none of ours is one of wgpu's.
    let mut sync2_on = vk::PhysicalDeviceSynchronization2Features::default()
        .synchronization2(sync2.synchronization2 == vk::TRUE);
    let mut ycbcr_on = vk::PhysicalDeviceSamplerYcbcrConversionFeatures::default()
        .sampler_ycbcr_conversion(ycbcr.sampler_ycbcr_conversion == vk::TRUE);
    let mut maintenance_on = vk::PhysicalDeviceVideoMaintenance1FeaturesKHR::default()
        .video_maintenance1(maintenance.video_maintenance1 == vk::TRUE);
    let mut info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&names);
    info = phd.add_to_device_create(info);
    info = info.push_next(&mut sync2_on).push_next(&mut ycbcr_on);
    if video_maintenance1 {
        info = info.push_next(&mut maintenance_on);
    }
    // SAFETY: a valid create info for this physical device, every extension
    // and feature in it one the device has.
    let device =
        unsafe { instance.create_device(physical, &info, None) }.context("vkCreateDevice")?;

    let limits = desc.required_limits.clone();
    // wgpu's default hint suballocates from blocks of 128 to 256 MiB, which
    // on a 3 GB card was a quarter of what a decode, a module and an encoder
    // held; this one uses blocks of 8 to 64 MiB.
    let desc = &wgpu_core::device::DeviceDescriptor {
        memory_hints: wgpu_types::MemoryHints::MemoryUsage,
        ..desc.clone()
    };
    // SAFETY: the device was created from this adapter with its required
    // extensions and features, and its queue (graphics, 0); wgpu-hal takes
    // ownership of it.
    let open = unsafe {
        hal_adapter.device_from_raw(
            device.clone(),
            None,
            &extensions,
            features,
            &limits,
            &desc.memory_hints,
            graphics,
            0,
        )
    }
    .map_err(|e| anyhow!("wgpu-hal took no device: {e}"))?;
    drop(hal_adapter);
    // SAFETY: the open device is of this adapter.
    let ids = unsafe { global.create_device_from_hal(adapter, open.into(), desc, None, None) }
        .map_err(|e| anyhow!("wgpu-core took no device: {e}"))?;

    // What libav reads of the features: a chain that lives for the process.
    let timeline_on: &'static mut _ = Box::leak(Box::new(
        vk::PhysicalDeviceTimelineSemaphoreFeatures::default()
            .timeline_semaphore(timeline.timeline_semaphore == vk::TRUE),
    ));
    let maintenance_kept: &'static mut _ = Box::leak(Box::new(
        vk::PhysicalDeviceVideoMaintenance1FeaturesKHR::default()
            .video_maintenance1(video_maintenance1 && maintenance.video_maintenance1 == vk::TRUE),
    ));
    let ycbcr_kept: &'static mut _ = Box::leak(Box::new(
        vk::PhysicalDeviceSamplerYcbcrConversionFeatures::default()
            .sampler_ycbcr_conversion(ycbcr.sampler_ycbcr_conversion == vk::TRUE),
    ));
    let sync2_kept: &'static mut _ = Box::leak(Box::new(
        vk::PhysicalDeviceSynchronization2Features::default()
            .synchronization2(sync2.synchronization2 == vk::TRUE),
    ));
    let mut kept = vk::PhysicalDeviceFeatures2::default()
        .features(core)
        .push_next(timeline_on)
        .push_next(ycbcr_kept)
        .push_next(sync2_kept);
    if video_maintenance1 {
        kept = kept.push_next(maintenance_kept);
    }
    let kept: &'static vk::PhysicalDeviceFeatures2<'static> = Box::leak(Box::new(kept));

    let no_codec = no_codec.or_else(|| {
        (sync2.synchronization2 != vk::TRUE || timeline.timeline_semaphore != vk::TRUE)
            .then(|| "the device has no synchronization2 or no timeline semaphores".to_string())
    });
    let families = if no_codec.is_some() {
        Vec::new()
    } else {
        // Listed in the order libav should prefer them: the graphics
        // family, wgpu's, last.
        let mut listed: Vec<av::QueueFamily> = plans[1..]
            .iter()
            .chain(std::iter::once(&plans[0]))
            .filter_map(|p| {
                p.for_libav.map(|(flags, caps)| av::QueueFamily {
                    index: p.index,
                    count: 1,
                    flags: flags.as_raw(),
                    video_caps: caps.as_raw(),
                })
            })
            .collect();
        listed.dedup_by_key(|f| f.index);
        listed
    };
    let _ = OPENED.set(Opened {
        instance,
        device,
        physical,
        get_instance_proc_addr: entry.static_fn().get_instance_proc_addr,
        instance_extensions,
        device_extensions: extensions.iter().map(|e| (*e).to_owned()).collect(),
        features: kept as *const _ as *const c_void,
        families,
        no_codec,
        graphics,
    });
    Ok(ids)
}

fn opened() -> Result<&'static Opened> {
    OPENED
        .get()
        .ok_or_else(|| anyhow!("the process's device was not opened through Vulkan here"))
}

/// The process's device as libav's `vulkan` device, made the first time a
/// decoder or an encoder asks; None, said once on stderr, where the device
/// was not opened here, has no decoder queue, or libav will not take it.
pub fn av_device() -> Option<av::Shared> {
    static MADE: OnceLock<Option<av::VulkanDevice>> = OnceLock::new();
    let made = MADE.get_or_init(|| {
        shared_device()?;
        let opened = OPENED.get()?;
        if let Some(why) = &opened.no_codec {
            eprintln!("[gpu] no decoder or encoder on the process's Vulkan device: {why}");
            return None;
        }
        let handles = av::VulkanHandles {
            get_instance_proc_addr: opened.get_instance_proc_addr as *const c_void,
            instance: opened.instance.handle().as_raw() as *mut c_void,
            physical_device: opened.physical.as_raw() as *mut c_void,
            device: opened.device.handle().as_raw() as *mut c_void,
            features: opened.features,
            instance_extensions: opened.instance_extensions.clone(),
            device_extensions: opened.device_extensions.clone(),
            queue_families: opened.families.clone(),
        };
        match av::VulkanDevice::new(&handles) {
            Ok(device) => Some(device),
            Err(e) => {
                eprintln!("[gpu] libav will not take the process's Vulkan device: {e}");
                None
            }
        }
    });
    made.as_ref().map(|d| av::Shared::Vulkan(d.share()))
}

/// The UUID of the card the process's device is on, as Vulkan and CUDA
/// both name it; None unless the device was opened here.
pub fn device_uuid() -> Option<[u8; 16]> {
    let opened = OPENED.get()?;
    let mut id = vk::PhysicalDeviceIDProperties::default();
    let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
    // SAFETY: a query of a live physical device.
    unsafe {
        opened
            .instance
            .get_physical_device_properties2(opened.physical, &mut props)
    };
    Some(id.device_uuid)
}

// Submissions of the host's own on wgpu's queue, and what they read and
// write kept alive until they have run.

/// Command buffers of the host's own, submitted on wgpu's queue between
/// wgpu's own submissions, for what wgpu's submit cannot do: wait on and
/// signal another API's timeline semaphores on the device. Each submission
/// also signals `timeline` with the next value, which says when it has run
/// and its command buffer is free.
struct Raw {
    timeline: vk::Semaphore,
    buffers: Mutex<Buffers>,
}

/// The command buffers of [`Raw`], used under its lock.
struct Buffers {
    pool: vk::CommandPool,
    /// The value the last submission signals.
    last: u64,
    /// Command buffers submitted, with the value their submission signals.
    in_flight: VecDeque<(u64, vk::CommandBuffer)>,
    free: Vec<vk::CommandBuffer>,
}

static RAW: OnceLock<std::result::Result<Raw, String>> = OnceLock::new();

fn raw(opened: &Opened) -> Result<&'static Raw> {
    let made = RAW.get_or_init(|| {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(opened.graphics);
        // SAFETY: a valid create info for a family the device has a queue
        // of.
        let pool = unsafe { opened.device.create_command_pool(&pool_info, None) }
            .map_err(|e| format!("a command pool: {e}"))?;
        let mut kind = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let info = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
        // SAFETY: a valid create info; timeline semaphores are on wherever
        // libav has the device, the only way here.
        let timeline = unsafe { opened.device.create_semaphore(&info, None) }
            .map_err(|e| format!("a timeline semaphore: {e}"))?;
        Ok(Raw {
            timeline,
            buffers: Mutex::new(Buffers {
                pool,
                last: 0,
                in_flight: VecDeque::new(),
                free: Vec::new(),
            }),
        })
    });
    made.as_ref().map_err(|e| anyhow!("{e}"))
}

/// Records commands with `record` into a command buffer of the host's own
/// and submits it on wgpu's queue: after every submission wgpu has made and
/// before every one it makes later, so wgpu's later reads and writes of
/// what it touched are ordered after it, as they are after a submission of
/// wgpu's. The submission waits on the device for each of `waits` and
/// signals each of `signals`, timeline semaphores at a value, once it has
/// run. Returns the value of the host's timeline it signals.
fn submit_raw(
    global: &Global,
    device: &SharedDevice,
    opened: &Opened,
    waits: &[(vk::Semaphore, u64)],
    signals: &[(vk::Semaphore, u64)],
    record: impl FnOnce(vk::CommandBuffer),
) -> Result<u64> {
    let raw = raw(opened)?;
    let timeline = raw.timeline;
    let mut buffers = raw.buffers.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: a live timeline semaphore of the device.
    let done = unsafe { opened.device.get_semaphore_counter_value(timeline) }
        .context("reading the host's timeline")?;
    while buffers
        .in_flight
        .front()
        .is_some_and(|(value, _)| *value <= done)
    {
        let (_, buffer) = buffers.in_flight.pop_front().expect("checked above");
        buffers.free.push(buffer);
    }
    let buffer = match buffers.free.pop() {
        Some(buffer) => buffer,
        None => {
            let info = vk::CommandBufferAllocateInfo::default()
                .command_pool(buffers.pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            // SAFETY: the pool is ours, used under the lock held.
            unsafe { opened.device.allocate_command_buffers(&info) }.context("a command buffer")?[0]
        }
    };
    let value = buffers.last + 1;
    let submitted = (|| -> Result<()> {
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: a command buffer of the pool, not in use: a free one has
        // run, and beginning it resets it.
        unsafe { opened.device.begin_command_buffer(buffer, &begin) }
            .context("beginning a command buffer")?;
        record(buffer);
        // SAFETY: as above, recorded by `record`.
        unsafe { opened.device.end_command_buffer(buffer) }.context("ending a command buffer")?;
        let semaphore = |(semaphore, value): &(vk::Semaphore, u64)| {
            vk::SemaphoreSubmitInfo::default()
                .semaphore(*semaphore)
                .value(*value)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        };
        let wait_infos: Vec<_> = waits.iter().map(semaphore).collect();
        let signal_infos: Vec<_> = signals
            .iter()
            .chain(std::iter::once(&(timeline, value)))
            .map(semaphore)
            .collect();
        let command_buffers = [vk::CommandBufferSubmitInfo::default().command_buffer(buffer)];
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&wait_infos)
            .command_buffer_infos(&command_buffers)
            .signal_semaphore_infos(&signal_infos);
        // wgpu-core makes each submission holding, for writing, the lock
        // that `device_fence_as_hal` takes for reading, so holding it keeps
        // wgpu's submissions off the queue while this one is made; the lock
        // on `buffers` keeps ours apart.
        // SAFETY: the handles are only used while the guards are held.
        let _submissions = unsafe { global.device_fence_as_hal::<Vulkan>(device.device) }
            .ok_or_else(|| anyhow!("the process's device is not a Vulkan one"))?;
        let queue = unsafe { global.queue_as_hal::<Vulkan>(*device.queue) }
            .ok_or_else(|| anyhow!("the process's queue is not a Vulkan one"))?;
        // SAFETY: wgpu's queue, which nothing else submits to while the
        // locks are held; every handle in the submission is alive.
        unsafe {
            opened
                .device
                .queue_submit2(queue.as_raw(), &[submit], vk::Fence::null())
        }
        .context("submitting on wgpu's queue")
    })();
    match submitted {
        Ok(()) => {
            buffers.last = value;
            buffers.in_flight.push_back((value, buffer));
            Ok(value)
        }
        Err(e) => {
            buffers.free.push(buffer);
            Err(e)
        }
    }
}

/// What a job waits for before it signals and lets go: a submission of
/// wgpu's, or one of the host's own, by the value it signals the host's
/// timeline with.
enum Until {
    Submission(wgpu_core::SubmissionIndex),
    Raw(u64),
}

/// The semaphores to signal once a submission has run, and what to keep
/// alive until then: the decoder's picture and the textures copied into, or
/// the frames copied from.
struct Job {
    until: Until,
    signals: Vec<(vk::Semaphore, u64)>,
    kept: Box<dyn std::any::Any + Send>,
}

/// The thread that signals pictures' semaphores from the host and lets go
/// of what submissions used, in the order the submissions were made, each
/// once its submission has run.
struct Signaller {
    jobs: Mutex<mpsc::Sender<Job>>,
}

fn signaller() -> &'static Signaller {
    static MADE: OnceLock<Signaller> = OnceLock::new();
    MADE.get_or_init(|| {
        let (sender, receiver) = mpsc::channel::<Job>();
        std::thread::Builder::new()
            .name("vk signal".to_string())
            .spawn(move || signal_loop(receiver))
            .expect("spawn the semaphore signaller");
        Signaller {
            jobs: Mutex::new(sender),
        }
    })
}

fn signal_loop(receiver: mpsc::Receiver<Job>) {
    let mut pending: VecDeque<Job> = VecDeque::new();
    // wgpu's fence, where it is a timeline semaphore whose value is the
    // submission index: waited on here directly, since wgpu-core's poll
    // holds the device's fence lock while it waits, and every submit from
    // another thread waits for that lock.
    let fence = shared_device().and_then(|device| {
        // SAFETY: the handle is only copied out, never destroyed here.
        let fence = unsafe { shared().device_fence_as_hal::<Vulkan>(device.device) }?;
        match &*fence {
            wgpu_hal::vulkan::Fence::TimelineSemaphore(semaphore) => Some(*semaphore),
            _ => None,
        }
    });
    loop {
        let job = match pending.pop_front() {
            Some(job) => job,
            None => match receiver.recv() {
                Ok(job) => job,
                Err(_) => return,
            },
        };
        let (Some(device), Ok(opened)) = (shared_device(), opened()) else {
            return;
        };
        let waited = match (&job.until, fence) {
            (Until::Raw(value), _) => {
                raw(opened).and_then(|raw| host_wait(&opened.device, raw.timeline, *value))
            }
            (Until::Submission(index), Some(semaphore)) => {
                host_wait(&opened.device, semaphore, *index)
            }
            (Until::Submission(index), None) => shared()
                .device_poll(
                    device.device,
                    wgpu_types::PollType::Wait {
                        submission_index: Some(*index),
                        timeout: None,
                    },
                )
                .map(|_| ())
                .map_err(|e| anyhow!("{e}")),
        };
        if let Err(e) = waited {
            eprintln!("[gpu] waiting for a copy of a picture: {e:#}");
        }
        for (semaphore, value) in &job.signals {
            let info = vk::SemaphoreSignalInfo::default()
                .semaphore(*semaphore)
                .value(*value);
            // SAFETY: a live timeline semaphore of the device, signalled
            // with the next value its picture was told of.
            if let Err(e) = unsafe { opened.device.signal_semaphore(&info) } {
                eprintln!("[gpu] signalling a picture's semaphore: {e}");
            }
        }
        drop(job.kept);
        while let Ok(job) = receiver.try_recv() {
            pending.push_back(job);
        }
    }
}

impl Signaller {
    fn send(&self, job: Job) -> Result<()> {
        self.jobs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .send(job)
            .map_err(|_| anyhow!("the semaphore signaller has stopped"))
    }
}

/// Waits on the host until `semaphore` reaches `value`.
fn host_wait(device: &ash::Device, semaphore: vk::Semaphore, value: u64) -> Result<()> {
    let semaphores = [semaphore];
    let values = [value];
    let info = vk::SemaphoreWaitInfo::default()
        .semaphores(&semaphores)
        .values(&values);
    // SAFETY: a live timeline semaphore of the device.
    unsafe { device.wait_semaphores(&info, u64::MAX) }.context("waiting on a picture's semaphore")
}

/// A command encoder of the device, for commands of the host's own.
fn command_encoder(
    global: &Global,
    device: &SharedDevice,
) -> Result<wgpu_core::id::CommandEncoderId> {
    let (encoder, err) = global.device_create_command_encoder(
        device.device,
        &wgpu_types::CommandEncoderDescriptor { label: None },
        None,
    );
    if let Some(e) = err {
        bail!("a command encoder: {e}");
    }
    Ok(encoder)
}

fn finish(
    global: &Global,
    encoder: wgpu_core::id::CommandEncoderId,
) -> Result<wgpu_core::id::CommandBufferId> {
    let (commands, err) = global.command_encoder_finish(
        encoder,
        &wgpu_types::CommandBufferDescriptor { label: None },
        None,
    );
    if let Some((_, e)) = err {
        bail!("finishing commands: {e}");
    }
    Ok(commands)
}

/// The raw image of a texture of the device.
fn raw_image(global: &Global, texture: TextureId) -> Result<vk::Image> {
    // SAFETY: the texture is alive while the caller holds it; the handle is
    // only used in commands recorded while it is.
    let hal = unsafe { global.texture_as_hal::<Vulkan>(texture) }
        .ok_or_else(|| anyhow!("a texture of the frame is not a Vulkan one"))?;
    Ok(unsafe { hal.raw_handle() })
}

/// `VK_FORMAT_G8_B8R8_2PLANE_420_UNORM`: NV12 in one multi-planar image.
const NV12: vk::Format = vk::Format::G8_B8R8_2PLANE_420_UNORM;

/// The usage of a plane a decoder's picture is copied into: a frame's.
const PLANE_USAGE: wgpu_types::TextureUsages = wgpu_types::TextureUsages::TEXTURE_BINDING
    .union(wgpu_types::TextureUsages::COPY_SRC)
    .union(wgpu_types::TextureUsages::COPY_DST)
    .union(wgpu_types::TextureUsages::RENDER_ATTACHMENT);

/// A texture of the device for one plane of a decoder's picture, made by
/// wgpu-hal and handed to wgpu-core, which counts a texture made that way
/// as written. One made by wgpu-core is counted as unwritten until a
/// command of its own writes it, and is cleared before it is first read,
/// which would clear what the copy wrote there.
fn copy_target(
    global: &Global,
    device: &SharedDevice,
    width: u32,
    height: u32,
    format: TextureFormat,
    bytes_per_pixel: u32,
) -> Result<Plane> {
    use wgpu_hal::Device as _;
    let size = Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let uses = wgpu_types::TextureUses::RESOURCE
        | wgpu_types::TextureUses::COPY_SRC
        | wgpu_types::TextureUses::COPY_DST
        | wgpu_types::TextureUses::COLOR_TARGET;
    let hal = {
        // SAFETY: the device is only referenced here.
        let hal_device = unsafe { global.device_as_hal::<Vulkan>(device.device) }
            .ok_or_else(|| anyhow!("the process's device is not a Vulkan one"))?;
        // SAFETY: a valid description of a 2D, single-level texture.
        unsafe {
            hal_device.create_texture(&wgpu_hal::TextureDescriptor {
                label: Some("ffrwd decoded plane"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu_types::TextureDimension::D2,
                format,
                usage: uses,
                memory_flags: wgpu_hal::MemoryFlags::empty(),
                view_formats: Vec::new(),
            })
        }
        .map_err(|e| anyhow!("a {width}x{height} {format:?} texture: {e}"))?
    };
    // SAFETY: made just above on this device from the same description;
    // it is written by the copy before anything reads it.
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
                usage: PLANE_USAGE,
                view_formats: Vec::new(),
            },
            None,
        )
    };
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

/// The copy targets a decoder keeps and copies its pictures into in turn:
/// a picture's frame holds its pair, and once every frame holding it is
/// gone the pair is free for a later picture. A later copy is queued on
/// wgpu's queue after every read of the pair queued so far, and wgpu
/// orders it after them, so a pair is never written while a read of the
/// picture before is still to run. Their wgpu ids are the same for the
/// run, as a pool of the decoder's own textures wrapped would be.
const SLOTS: usize = 32;

/// One pair of copy targets, its luma and its chroma. Dropping the last
/// reference drops the textures.
struct Slot {
    global: Arc<Global>,
    index: usize,
    y: Plane,
    uv: Plane,
}

impl Drop for Slot {
    fn drop(&mut self) {
        self.global.texture_drop(self.y.texture);
        self.global.texture_drop(self.uv.texture);
    }
}

/// What a decoder's picture taken here is.
pub(super) struct Taken {
    pub planes: Vec<Plane>,
    pub owned: Vec<TextureId>,
    pub wrapped: Option<Wrapped>,
}

impl Taken {
    /// What the line on stderr says of how pictures are taken.
    pub(super) fn how(&self) -> String {
        "frames copied (wgpu takes a raw image as uninitialised)".to_string()
    }
}

/// The pair of copy targets a frame's planes are: handed back to its
/// decoder's pool when the frame is dropped.
pub(super) struct Wrapped {
    slot: Arc<Slot>,
}

impl Wrapped {
    pub(super) fn release(self, _global: &Global, _device: &SharedDevice) {
        drop(self.slot);
    }
}

/// One decoder's way of taking its pictures: its pool of copy targets, and
/// where in it the next free pair is looked for.
#[derive(Default)]
pub struct Taker {
    slots: Vec<Arc<Slot>>,
    next: usize,
}

impl Taker {
    /// A free pair of copy targets for a `width` x `height` picture: the
    /// next free one in the pool after the last taken, so the pairs are
    /// used in turn; a new one where none is free, kept while the pool has
    /// room. Free pairs of another size are let go.
    fn slot(
        &mut self,
        global: &Arc<Global>,
        device: &SharedDevice,
        width: u32,
        height: u32,
    ) -> Result<Arc<Slot>> {
        let fits = |s: &Slot| (s.y.width, s.y.height) == (width, height);
        self.slots.retain(|s| Arc::strong_count(s) > 1 || fits(s));
        let count = self.slots.len();
        for k in 0..count {
            let at = (self.next + k) % count;
            let slot = &self.slots[at];
            if Arc::strong_count(slot) == 1 && fits(slot) {
                self.next = at + 1;
                return Ok(Arc::clone(slot));
            }
        }
        let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
        let y = copy_target(global, device, width, height, TextureFormat::R8Unorm, 1)?;
        let uv = match copy_target(global, device, cw, ch, TextureFormat::Rg8Unorm, 2) {
            Ok(uv) => uv,
            Err(e) => {
                global.texture_drop(y.texture);
                return Err(e);
            }
        };
        let pooled = count < SLOTS;
        let slot = Arc::new(Slot {
            global: Arc::clone(global),
            index: if pooled { count } else { usize::MAX },
            y,
            uv,
        });
        if pooled {
            self.slots.push(Arc::clone(&slot));
            self.next = count + 1;
        }
        Ok(slot)
    }

    /// Takes `picture`, an NV12 picture the decoder wrote on the process's
    /// device, by a copy into a pair of textures of wgpu's from the pool.
    pub(super) fn take(
        &mut self,
        global: &Arc<Global>,
        device: &SharedDevice,
        mut picture: DeviceFrame,
    ) -> Result<Taken> {
        let opened = opened()?;
        let slot = self.slot(global, device, picture.width, picture.height)?;
        let (y, uv) = (slot.y, slot.uv);
        let copied = copy_picture(global, device, opened, &mut picture, &y, &uv)?;
        if crate::trace::on() {
            crate::trace::line(format_args!(
                "taker: picture pts {} into slot {} (y {:?}, uv {:?})",
                picture.pts,
                if slot.index == usize::MAX {
                    "none, the pool being full".to_string()
                } else {
                    slot.index.to_string()
                },
                y.texture,
                uv.texture
            ));
        }
        // wgpu counts the targets used by its own submission before the
        // copy, which may finish first: the pair is kept, and out of the
        // pool, until the copy has run.
        signaller().send(Job {
            until: Until::Raw(copied),
            signals: Vec::new(),
            kept: Box::new((picture, Arc::clone(&slot))),
        })?;
        Ok(Taken {
            planes: vec![y, uv],
            owned: Vec::new(),
            wrapped: Some(Wrapped { slot }),
        })
    }
}

/// Copies the decoder's `picture` into `y` and `uv`, by a submission of the
/// host's own on wgpu's queue that waits for the decoder's write and moves
/// the picture's semaphore on by one once it has run. Returns the value of
/// the host's timeline the submission signals.
fn copy_picture(
    global: &Global,
    device: &SharedDevice,
    opened: &Opened,
    picture: &mut DeviceFrame,
    y: &Plane,
    uv: &Plane,
) -> Result<u64> {
    let (width, height) = (y.width, y.height);
    let mut lock = picture
        .vulkan()
        .ok_or_else(|| anyhow!("the decoder's picture is not a Vulkan one"))?;
    let images = lock.images();
    let [image] = images.as_slice() else {
        bail!(
            "the decoder's picture is {} images, and one multi-planar image is taken",
            images.len()
        );
    };
    if vk::Format::from_raw(image.format as i32) != NV12 {
        bail!(
            "the decoder's picture is VkFormat {}, and only NV12 (G8_B8R8_2PLANE_420) is taken",
            image.format
        );
    }
    if image.queue_family != vk::QUEUE_FAMILY_IGNORED {
        bail!("the decoder's picture belongs to one queue family, not to the device's");
    }
    if vk::ImageUsageFlags::from_raw(lock.usage()) & vk::ImageUsageFlags::TRANSFER_SRC
        != vk::ImageUsageFlags::TRANSFER_SRC
    {
        bail!("the decoder's pictures cannot be copied from");
    }
    let source = vk::Image::from_raw(image.image);
    let semaphore = vk::Semaphore::from_raw(image.semaphore);

    // The targets in the layout the copy writes, by wgpu, so that wgpu's
    // next use of them starts from that layout.
    transition(
        global,
        device,
        &[y.texture, uv.texture],
        wgpu_types::TextureUses::COPY_DST,
    )?;
    let (y_image, uv_image) = (
        raw_image(global, y.texture)?,
        raw_image(global, uv.texture)?,
    );
    let next = image.value + 1;
    let copied = submit_raw(
        global,
        device,
        opened,
        &[(semaphore, image.value)],
        &[(semaphore, next)],
        |commands| {
            let barrier = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .src_access_mask(vk::AccessFlags2::from_raw(image.access))
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .old_layout(vk::ImageLayout::from_raw(image.layout as i32))
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(source)
                .subresource_range(COLOR_RANGE)];
            // After an earlier copy into the same targets that nothing of
            // wgpu's came between.
            let after_copies = [copies_after_copies()];
            // SAFETY: commands of the device into a command buffer being
            // recorded; the picture is alive and locked, the targets alive
            // and in the layout wgpu moved them to.
            unsafe {
                opened.device.cmd_pipeline_barrier2(
                    commands,
                    &vk::DependencyInfo::default()
                        .memory_barriers(&after_copies)
                        .image_memory_barriers(&barrier),
                );
                opened.device.cmd_copy_image(
                    commands,
                    source,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    y_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region(
                        vk::ImageAspectFlags::PLANE_0,
                        vk::ImageAspectFlags::COLOR,
                        width,
                        height,
                    )],
                );
                opened.device.cmd_copy_image(
                    commands,
                    source,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    uv_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region(
                        vk::ImageAspectFlags::PLANE_1,
                        vk::ImageAspectFlags::COLOR,
                        uv.width,
                        uv.height,
                    )],
                );
            }
        },
    )?;
    lock.used(
        0,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL.as_raw() as u32,
        vk::AccessFlags2::TRANSFER_READ.as_raw(),
        next,
    );
    Ok(copied)
}

/// The colour aspect of a single-level, single-layer image.
const COLOR_RANGE: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_mip_level: 0,
    level_count: 1,
    base_array_layer: 0,
    layer_count: 1,
};

/// A dependency of copies on the copies before them on the queue: wgpu
/// moves a texture already in the layout a copy writes nowhere, and orders
/// nothing between two writes of it it did not make.
fn copies_after_copies() -> vk::MemoryBarrier2<'static> {
    vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COPY)
        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::COPY)
        .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
}

/// A copy of `width` x `height` texels from the `from` aspect of one image
/// to the `to` aspect of another, both single-level.
fn region(
    from: vk::ImageAspectFlags,
    to: vk::ImageAspectFlags,
    width: u32,
    height: u32,
) -> vk::ImageCopy {
    let layers = |aspect_mask| vk::ImageSubresourceLayers {
        aspect_mask,
        mip_level: 0,
        base_array_layer: 0,
        layer_count: 1,
    };
    vk::ImageCopy {
        src_subresource: layers(from),
        src_offset: vk::Offset3D::default(),
        dst_subresource: layers(to),
        dst_offset: vk::Offset3D::default(),
        extent: vk::Extent3D {
            width,
            height,
            depth: 1,
        },
    }
}

/// Moves `textures` into the layout of `state` by a submission of wgpu's,
/// for a submission of the host's own that uses them in it next.
fn transition(
    global: &Global,
    device: &SharedDevice,
    textures: &[TextureId],
    state: wgpu_types::TextureUses,
) -> Result<()> {
    let encoder = command_encoder(global, device)?;
    global
        .command_encoder_transition_resources(
            encoder,
            std::iter::empty(),
            textures
                .iter()
                .map(|&texture| wgpu_types::TextureTransition {
                    texture,
                    selector: None,
                    state,
                }),
        )
        .map_err(|e| anyhow!("{e}"))?;
    let commands = finish(global, encoder)?;
    global
        .queue_submit(*device.queue, &[commands])
        .map_err(|(_, e)| anyhow!("submitting a transition: {e}"))?;
    Ok(())
}

/// Writes frames on the process's device into the Vulkan pictures an
/// encoder reaching CUDA is fed from.
#[derive(Default)]
pub struct EncoderFeed;

impl EncoderFeed {
    /// Writes `frame` into `picture`, a Vulkan picture of an encoder's
    /// feeding pool, one image per plane: the frame as NV12
    /// ([`GpuFrame::converted`], an rgba frame made NV12 in `yuv` first),
    /// each plane copied into the picture's image on the device. libav is
    /// told the images' layout and the semaphore value they are written at,
    /// which is signalled once the copy has run.
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
        let opened = opened()?;
        if frame.backend() != wgpu_types::Backend::Vulkan {
            bail!(
                "the frame is on a {:?} device, not a Vulkan one",
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
        let mut lock = picture
            .vulkan()
            .ok_or_else(|| anyhow!("the encoder's picture is not a Vulkan one"))?;
        let images = lock.images();
        let formats = [vk::Format::R8_UNORM, vk::Format::R8G8_UNORM];
        if images.len() != 2
            || images
                .iter()
                .zip(formats)
                .any(|(i, f)| vk::Format::from_raw(i.format as i32) != f)
        {
            bail!("the encoder's picture is not NV12 as two images, R8 and R8G8");
        }
        // Whatever read the picture last is done with it.
        for image in &images {
            host_wait(
                &opened.device,
                vk::Semaphore::from_raw(image.semaphore),
                image.value,
            )?;
        }
        let mut targets = Vec::with_capacity(2);
        let written = (|| -> Result<wgpu_core::SubmissionIndex> {
            for (image, plane) in images.iter().zip([y, uv]) {
                targets.push(wrap_target(global, device, image.image, plane)?);
            }
            let encoder = command_encoder(global, device)?;
            for (plane, target) in [y, uv].iter().zip(&targets) {
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
            let commands = finish(global, encoder)?;
            global
                .queue_submit(*device.queue, &[commands])
                .map_err(|(_, e)| anyhow!("submitting the copy: {e}"))
        })();
        for texture in targets {
            global.texture_drop(texture);
        }
        let submission = written?;
        // The copy leaves each image as wgpu's copy writes it.
        let mut signals = Vec::with_capacity(2);
        for (index, image) in images.iter().enumerate() {
            let next = image.value + 1;
            lock.used(
                index,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL.as_raw() as u32,
                vk::AccessFlags2::TRANSFER_WRITE.as_raw(),
                next,
            );
            signals.push((vk::Semaphore::from_raw(image.semaphore), next));
        }
        drop(lock);
        signaller().send(Job {
            until: Until::Submission(submission),
            signals,
            kept: Box::new(vec![Arc::clone(frame), nv12]),
        })
    }
}

/// An image of an encoder's picture as a texture of the device that copies
/// write: wgpu neither owns it nor its memory, and starts from an undefined
/// layout, which loses nothing since the copy writes every texel.
fn wrap_target(
    global: &Global,
    device: &SharedDevice,
    image: u64,
    plane: &Plane,
) -> Result<TextureId> {
    let size = Extent3d {
        width: plane.width,
        height: plane.height,
        depth_or_array_layers: 1,
    };
    // SAFETY: the device is only referenced here.
    let hal_device = unsafe { global.device_as_hal::<Vulkan>(device.device) }
        .ok_or_else(|| anyhow!("the process's device is not a Vulkan one"))?;
    // SAFETY: a 2D, single-level, single-sample image of this device in the
    // plane's format, alive while the encoder's picture is, which outlives
    // every command using it (the copy is waited for before the picture is
    // read, and libav frees a picture only once its semaphore is reached).
    let hal = unsafe {
        hal_device.texture_from_raw(
            vk::Image::from_raw(image),
            &wgpu_hal::TextureDescriptor {
                label: Some("ffrwd encoder picture"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu_types::TextureDimension::D2,
                format: plane.format,
                usage: wgpu_types::TextureUses::COPY_DST,
                memory_flags: wgpu_hal::MemoryFlags::empty(),
                view_formats: Vec::new(),
            },
            Some(Box::new(|| {})),
            wgpu_hal::vulkan::TextureMemory::External,
        )
    };
    drop(hal_device);
    let (texture, err) = unsafe {
        global.create_texture_from_hal(
            Box::new(hal),
            device.device,
            &wgpu_types::TextureDescriptor {
                label: Some(std::borrow::Cow::Borrowed("ffrwd encoder picture")),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu_types::TextureDimension::D2,
                format: plane.format,
                usage: wgpu_types::TextureUsages::COPY_DST,
                view_formats: Vec::new(),
            },
            None,
        )
    };
    if let Some(e) = err {
        global.texture_drop(texture);
        bail!("wgpu refused the encoder's picture: {e}");
    }
    Ok(texture)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn family(
        flags: vk::QueueFlags,
        count: u32,
    ) -> (vk::QueueFlags, u32, vk::VideoCodecOperationFlagsKHR) {
        (flags, count, vk::VideoCodecOperationFlagsKHR::NONE)
    }

    #[test]
    fn nvidia_s_families_give_libav_a_queue_of_each_and_wgpu_the_graphics_one() {
        use vk::QueueFlags as Q;
        let families = [
            family(Q::GRAPHICS | Q::COMPUTE | Q::TRANSFER, 16),
            family(Q::TRANSFER, 2),
            family(Q::COMPUTE | Q::TRANSFER, 8),
            (
                Q::TRANSFER | Q::VIDEO_DECODE_KHR,
                1,
                vk::VideoCodecOperationFlagsKHR::DECODE_H264,
            ),
            (
                Q::TRANSFER | Q::VIDEO_ENCODE_KHR,
                1,
                vk::VideoCodecOperationFlagsKHR::ENCODE_H264,
            ),
        ];
        let (graphics, plans, no_codec) = plan_families(&families, true).unwrap();
        assert_eq!(graphics, 0);
        assert_eq!(no_codec, None);
        let listed: Vec<(u32, Option<vk::QueueFlags>)> = plans
            .iter()
            .map(|p| (p.index, p.for_libav.map(|(f, _)| f)))
            .collect();
        assert_eq!(
            listed,
            vec![
                (0, Some(Q::GRAPHICS)),
                (1, Some(Q::TRANSFER)),
                (2, Some(Q::COMPUTE | Q::TRANSFER)),
                (3, Some(Q::VIDEO_DECODE_KHR)),
                (4, Some(Q::VIDEO_ENCODE_KHR)),
            ]
        );
        assert!(plans.iter().all(|p| p.queues == 1));
    }

    #[test]
    fn one_family_and_no_video_gives_libav_nothing() {
        let families = [family(
            vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER,
            1,
        )];
        let (graphics, plans, no_codec) = plan_families(&families, false).unwrap();
        assert_eq!(graphics, 0);
        assert_eq!(plans.len(), 1);
        assert!(plans[0].for_libav.is_none());
        assert_eq!(
            no_codec.as_deref(),
            Some("the device has no video extensions")
        );
    }
}
