//! A Vulkan device the caller made, as libav's `vulkan` device: what a
//! decoder is opened on so its pictures are images of that device, and what
//! an encoder reaching CUDA is opened from.
//!
//! The caller hands over the raw handles ([`VulkanHandles`]); one
//! [`VulkanDevice`] is made of them for the process and shared by every
//! decoder and encoder, so libav's locks on the queues it was given are one
//! set. A picture of the device is reached through [`VkPictureLock`], which
//! holds libav's lock on the picture while its layout, access and timeline
//! semaphore are read and updated, as libav's own users do.

use std::ffi::CString;
use std::os::raw::c_void;
use std::sync::{Arc, OnceLock};

use super::error::{Error, Result};
use super::ffi::BufferRef;
use super::sys::{AVHWFramesContext, AVVkFrame, AVVulkanFramesContext};

/// One queue family libav may submit to: `count` queues from index 0,
/// used for what `flags` (`VkQueueFlagBits`) says, decoding or encoding
/// what `video_caps` (`VkVideoCodecOperationFlagBitsKHR`) says.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QueueFamily {
    pub index: u32,
    pub count: u32,
    pub flags: u32,
    pub video_caps: u32,
}

/// A Vulkan device the caller made and keeps for the life of the process.
pub struct VulkanHandles {
    /// `PFN_vkGetInstanceProcAddr` of the loader the instance came from.
    pub get_instance_proc_addr: *const c_void,
    /// `VkInstance`, created for Vulkan 1.3 or later.
    pub instance: *mut c_void,
    /// `VkPhysicalDevice`.
    pub physical_device: *mut c_void,
    /// `VkDevice`.
    pub device: *mut c_void,
    /// `const VkPhysicalDeviceFeatures2 *`: the features the device was
    /// created with, and a `pNext` chain that lives as long as the device.
    pub features: *const c_void,
    pub instance_extensions: Vec<CString>,
    pub device_extensions: Vec<CString>,
    /// The families libav may submit to, in the order it should prefer
    /// them, each of whose queues no one else submits to; and every family
    /// the device's images are used on, since libav shares its images
    /// among exactly the families listed.
    pub queue_families: Vec<QueueFamily>,
}

/// The process's Vulkan device as libav has it, and the CUDA device on the
/// same card, made the first time an encoder asks.
pub struct VulkanDevice {
    inner: Arc<Inner>,
}

struct Inner {
    device: BufferRef,
    cuda: OnceLock<std::result::Result<BufferRef, String>>,
}

// SAFETY: libav's buffer references are counted atomically, and the device
// contexts they hold are used under libav's own locks.
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

impl VulkanDevice {
    /// libav's `vulkan` device around `handles`.
    pub fn new(handles: &VulkanHandles) -> Result<VulkanDevice> {
        let lib = super::runtime()?;
        let device = lib.vulkan_device(handles).map_err(|e| Error::Call {
            call: "av_hwdevice_ctx_init",
            codec: "vulkan".to_string(),
            why: format!("the host's device: {}", e.why),
        })?;
        Ok(VulkanDevice {
            inner: Arc::new(Inner {
                device,
                cuda: OnceLock::new(),
            }),
        })
    }

    /// Another handle to the same device.
    pub fn share(&self) -> VulkanDevice {
        VulkanDevice {
            inner: Arc::clone(&self.inner),
        }
    }

    pub(super) fn device(&self) -> &BufferRef {
        &self.inner.device
    }

    /// The CUDA device on the same card, made once.
    pub(super) fn cuda(&self) -> std::result::Result<&BufferRef, String> {
        let made = self.inner.cuda.get_or_init(|| {
            let lib = super::runtime().map_err(|e| e.to_string())?;
            lib.cuda_from_vulkan(&self.inner.device)
                .map_err(|e| format!("no CUDA device on the same card: {}", e.why))
        });
        made.as_ref().map_err(Clone::clone)
    }
}

/// One image of a picture as libav last left it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VkImageState {
    /// `VkImage`.
    pub image: u64,
    /// `VkFormat`, as the picture's frames context has it.
    pub format: u32,
    /// `VkImageLayout`.
    pub layout: u32,
    /// `VkAccessFlags2`.
    pub access: u64,
    /// The image's timeline `VkSemaphore`.
    pub semaphore: u64,
    /// The value it reaches once the image is ready.
    pub value: u64,
    /// `VK_QUEUE_FAMILY_IGNORED` where the image is shared by its families.
    pub queue_family: u32,
}

/// A Vulkan picture held under libav's lock on it: its images' state is
/// read and updated with the lock held, and the lock is let go when this is
/// dropped. Taken just before a submission that uses the picture and
/// dropped just after it, as libav asks of its users.
pub struct VkPictureLock<'a> {
    frame: *mut AVVkFrame,
    frames: *mut AVHWFramesContext,
    _picture: std::marker::PhantomData<&'a mut ()>,
}

impl<'a> VkPictureLock<'a> {
    /// Locks the picture `frame` of `frames`.
    ///
    /// # Safety
    /// Both are a live Vulkan picture and its frames context, alive for
    /// `'a`.
    pub(super) unsafe fn lock(
        frame: *mut AVVkFrame,
        frames: *mut AVHWFramesContext,
    ) -> VkPictureLock<'a> {
        // SAFETY: a vulkan frames context's hwctx is its
        // AVVulkanFramesContext; libav sets its lock functions at init.
        unsafe {
            let hw = (*frames).hwctx as *const AVVulkanFramesContext;
            if let Some(lock) = (*hw).lock_frame {
                lock(frames, frame);
            }
        }
        VkPictureLock {
            frame,
            frames,
            _picture: std::marker::PhantomData,
        }
    }

    /// `VkImageUsageFlags` every image of the pool was made with.
    pub fn usage(&self) -> u32 {
        // SAFETY: as in `lock`.
        unsafe { (*((*self.frames).hwctx as *const AVVulkanFramesContext)).usage }
    }

    /// The picture's images, one multi-planar image or one per plane.
    pub fn images(&self) -> Vec<VkImageState> {
        // SAFETY: the picture is alive and locked.
        let (f, hw) = unsafe {
            (
                &*self.frame,
                &*((*self.frames).hwctx as *const AVVulkanFramesContext),
            )
        };
        (0..f.img.len())
            .take_while(|&i| f.img[i] != 0)
            .map(|i| VkImageState {
                image: f.img[i],
                format: hw.format[i],
                layout: f.layout[i],
                access: f.access[i],
                semaphore: f.sem[i],
                value: f.sem_value[i],
                queue_family: f.queue_family[i],
            })
            .collect()
    }

    /// Records a submission that used image `index`: it left the image in
    /// `layout` with `access`, and signals the image's semaphore with
    /// `value`, one past what it waited for.
    pub fn used(&mut self, index: usize, layout: u32, access: u64, value: u64) {
        // SAFETY: the picture is alive and locked.
        let f = unsafe { &mut *self.frame };
        f.layout[index] = layout;
        f.access[index] = access;
        f.sem_value[index] = value;
    }
}

impl Drop for VkPictureLock<'_> {
    fn drop(&mut self) {
        // SAFETY: as in `lock`; the lock taken there is let go.
        unsafe {
            let hw = (*self.frames).hwctx as *const AVVulkanFramesContext;
            if let Some(unlock) = (*hw).unlock_frame {
                unlock(self.frames, self.frame);
            }
        }
    }
}
