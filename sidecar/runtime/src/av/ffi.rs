//! The unsafe half of the shim: loading the libraries, resolving their
//! functions, checking that the structs in `sys` match what was loaded, and
//! owning everything libav allocates.
//!
//! Everything this module hands out is safe to use. Each handle owns one
//! libav allocation, made by libav's own allocator and freed by its
//! counterpart in `Drop`, and every read of a picture's bytes is bounded by
//! the geometry of a format this module knows the layout of.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;
use std::ptr;

use libloading::Library as Dl;

use super::error::{Error, Result};
use super::picture::{self, WIRE_PIX_FMTS};
use super::sys::*;
use super::Library;

/// One loaded library, and its file name for a refusal to name.
struct Loaded {
    dl: Dl,
    file: String,
}

/// The four libraries, loaded and kept loaded for the life of the process.
struct Libs {
    avutil: Loaded,
    swresample: Loaded,
    avcodec: Loaded,
    swscale: Loaded,
}

/// Resolves one function by name, as the type the headers declare for it.
fn symbol<T: Copy>(lib: &Loaded, name: &'static str) -> Result<T> {
    let mut cname = Vec::with_capacity(name.len() + 1);
    cname.extend_from_slice(name.as_bytes());
    cname.push(0);
    // SAFETY: `T` is the signature the pinned major's headers give `name`;
    // the pointer is copied out and the library outlives it (`Lib` keeps it).
    match unsafe { lib.dl.get::<T>(&cname) } {
        Ok(found) => Ok(*found),
        Err(e) => Err(Error::Symbol {
            library: lib.file.clone(),
            symbol: name,
            why: e.to_string(),
        }),
    }
}

macro_rules! functions {
    ($fns:ident($libs:ty) $($lib:ident { $($name:ident: $ty:ty,)* })*) => {
        /// Functions the shim calls, resolved once when their libraries
        /// load. A missing one refuses the load, naming it and its library.
        struct $fns {
            $($($name: $ty,)*)*
        }

        impl $fns {
            fn resolve(libs: &$libs) -> Result<$fns> {
                Ok($fns {
                    $($($name: symbol::<$ty>(&libs.$lib, stringify!($name))?,)*)*
                })
            }
        }
    };
}

/// libavfilter, which loads after the four and may be missing: a build
/// from before it was bundled runs everything but `filter`.
struct FilterLibs {
    avfilter: Loaded,
}

functions! {
    FilterFns(FilterLibs)
    avfilter {
        avfilter_version: FnVersion,
        av_filter_iterate: FnFilterIterate,
        avfilter_get_by_name: FnFilterByName,
        avfilter_graph_alloc: FnGraphAlloc,
        avfilter_graph_free: FnGraphFree,
        avfilter_graph_alloc_filter: FnGraphAllocFilter,
        avfilter_init_str: FnFilterInitStr,
        avfilter_graph_parse_ptr: FnGraphParsePtr,
        avfilter_graph_config: FnGraphConfig,
        avfilter_inout_alloc: FnInOutAlloc,
        avfilter_inout_free: FnInOutFree,
        av_buffersrc_parameters_alloc: FnSrcParamsAlloc,
        av_buffersrc_parameters_set: FnSrcParamsSet,
        av_buffersrc_add_frame_flags: FnSrcAddFrame,
        av_buffersink_get_frame: FnSinkGetFrame,
        av_buffersink_get_w: FnSinkInt,
        av_buffersink_get_h: FnSinkInt,
        av_buffersink_get_format: FnSinkInt,
        av_buffersink_get_time_base: FnSinkRational,
        av_buffersink_get_frame_rate: FnSinkRational,
        av_buffersink_get_hw_frames_ctx: FnSinkHwFrames,
    }
}

functions! {
    Fns(Libs)
    avutil {
        avutil_version: FnVersion,
        av_version_info: FnStr,
        av_frame_alloc: FnFrameAlloc,
        av_frame_free: FnFrameFree,
        av_frame_unref: FnFrameUnref,
        av_frame_get_buffer: FnFrameGetBuffer,
        av_frame_copy_props: FnFrameCopyProps,
        av_strerror: FnStrerror,
        av_get_pix_fmt: FnFromName,
        av_get_pix_fmt_name: FnEnumName,
        av_pix_fmt_desc_get: FnPixDesc,
        av_hwdevice_iterate_types: FnHwIterate,
        av_hwdevice_get_type_name: FnEnumName,
        av_hwdevice_find_type_by_name: FnFromName,
        av_hwdevice_ctx_create: FnHwDeviceCreate,
        av_hwdevice_ctx_alloc: FnHwDeviceAlloc,
        av_hwdevice_ctx_init: FnHwDeviceInit,
        av_hwframe_ctx_alloc: FnHwFramesAlloc,
        av_hwframe_ctx_init: FnHwFramesInit,
        av_hwframe_get_buffer: FnHwFrameGetBuffer,
        av_buffer_unref: FnBufferUnref,
        av_frame_ref: FnFrameRef,
        av_hwframe_transfer_data: FnHwTransfer,
        av_opt_set: FnOptSet,
        av_opt_find: FnOptFind,
        av_mallocz: FnMallocz,
        av_log_set_level: FnLogSetLevel,
        av_color_range_name: FnEnumName,
        av_color_primaries_name: FnEnumName,
        av_color_transfer_name: FnEnumName,
        av_color_space_name: FnEnumName,
        av_color_range_from_name: FnFromName,
        av_color_primaries_from_name: FnFromName,
        av_color_transfer_from_name: FnFromName,
        av_color_space_from_name: FnFromName,
        av_strdup: FnStrdup,
        av_free: FnFree,
    }
    swresample {
        swresample_version: FnVersion,
    }
    avcodec {
        avcodec_version: FnVersion,
        avcodec_license: FnStr,
        av_codec_iterate: FnCodecIterate,
        av_codec_is_encoder: FnCodecIs,
        av_codec_is_decoder: FnCodecIs,
        avcodec_find_decoder_by_name: FnFindByName,
        avcodec_find_encoder_by_name: FnFindByName,
        avcodec_find_decoder: FnFindById,
        avcodec_find_encoder: FnFindById,
        avcodec_descriptor_get_by_name: FnDescriptorByName,
        avcodec_get_name: FnEnumName,
        avcodec_alloc_context3: FnContextAlloc,
        avcodec_free_context: FnContextFree,
        avcodec_parameters_alloc: FnParamsAlloc,
        avcodec_parameters_free: FnParamsFree,
        avcodec_parameters_to_context: FnParamsToContext,
        avcodec_parameters_from_context: FnParamsFromContext,
        avcodec_open2: FnOpen2,
        avcodec_send_packet: FnSendPacket,
        avcodec_receive_frame: FnReceiveFrame,
        avcodec_send_frame: FnSendFrame,
        avcodec_receive_packet: FnReceivePacket,
        av_packet_alloc: FnPacketAlloc,
        av_packet_free: FnPacketFree,
        av_packet_unref: FnPacketUnref,
        av_new_packet: FnNewPacket,
        avcodec_get_hw_config: FnGetHwConfig,
        avcodec_get_supported_config: FnSupportedConfig,
    }
    swscale {
        swscale_version: FnVersion,
        sws_alloc_context: FnSwsAlloc,
        sws_free_context: FnSwsFree,
        sws_scale_frame: FnSwsScaleFrame,
    }
}

/// A C string libav owns, as text; None for a null pointer.
///
/// # Safety
/// `p` is null or points at a NUL-terminated string that outlives the call.
unsafe fn text(p: *const c_char) -> Option<String> {
    if p.is_null() {
        None
    } else {
        Some(CStr::from_ptr(p).to_string_lossy().into_owned())
    }
}

/// libav's answer to a call that failed: its code, and its words for it.
#[derive(Clone, Debug)]
pub(super) struct AvErr {
    pub code: c_int,
    pub why: String,
}

impl AvErr {
    /// Whether libav said the option does not exist, rather than that its
    /// value was wrong.
    pub fn is_option_not_found(&self) -> bool {
        self.code == AVERROR_OPTION_NOT_FOUND
    }
}

/// What a send or receive came back with.
pub(super) enum Status {
    Done,
    /// The other direction has to move first.
    Again,
    /// Flushed, and drained.
    Eof,
    Failed(AvErr),
}

/// A picture's colour, as libav's enum values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RawColor {
    pub range: c_int,
    pub primaries: c_int,
    pub trc: c_int,
    pub space: c_int,
}

impl RawColor {
    /// Nothing said about it.
    pub const UNSPECIFIED: RawColor = RawColor {
        range: AVCOL_RANGE_UNSPECIFIED,
        primaries: AVCOL_PRI_UNSPECIFIED,
        trc: AVCOL_TRC_UNSPECIFIED,
        space: AVCOL_SPC_UNSPECIFIED,
    };

    /// The same picture, tagged as rgb: full range, no matrix. Primaries and
    /// transfer stay, so a conversion changes the matrix and nothing else.
    pub fn as_rgb(self) -> RawColor {
        RawColor {
            range: AVCOL_RANGE_JPEG,
            space: AVCOL_SPC_RGB,
            ..self
        }
    }
}

/// What an opened encoder settled on, read back through
/// `avcodec_parameters_from_context`.
pub(super) struct CodedParams {
    pub extradata: Vec<u8>,
    pub video_delay: c_int,
    pub codec_tag: u32,
}

/// The loaded libraries and their functions.
pub(super) struct Lib {
    f: Fns,
    /// The wire's pixel formats, by libavutil's numbering.
    wire: Vec<(&'static str, c_int)>,
    /// libavfilter, or why it is not loaded.
    filters: std::result::Result<FilterLib, Error>,
    _libs: Libs,
}

/// libavfilter, loaded, its functions resolved and its structs checked.
pub(super) struct FilterLib {
    f: FilterFns,
    _libs: FilterLibs,
}

impl Lib {
    /// Loads the four libraries from `dir`, by full path and in dependency
    /// order, refusing any of another major, then resolves every function and
    /// checks the struct layout against what was loaded. `from` says how the
    /// directory was chosen, for the refusal to name.
    pub fn load(dir: &Path, from: &'static str) -> Result<Lib> {
        let missing = |file: String, why: Option<String>| Error::Missing {
            dir: dir.to_path_buf(),
            from,
            file,
            why,
        };

        for lib in Library::LOAD_ORDER {
            if !dir.join(lib.file_name()).is_file() {
                return Err(missing(lib.file_name(), None));
            }
        }

        // The libraries' own imports - each other, and on Windows whatever
        // they link of the system's - resolve through the search path, and
        // only the directories named to it can answer.
        if let Err(e) = crate::nn::dlls::add_search_dir(dir) {
            return Err(missing(
                Library::Avutil.file_name(),
                Some(format!(
                    "adding its directory to the library search path: {e}"
                )),
            ));
        }

        let open = |lib: Library| -> Result<Loaded> {
            let file = lib.file_name();
            // SAFETY: loading runs the library's initialisers. These are
            // FFmpeg's own, named by full path, never found by bare name.
            let dl = unsafe { Dl::new(dir.join(&file)) }
                .map_err(|e| missing(file.clone(), Some(e.to_string())))?;
            let loaded = Loaded { dl, file };
            let version: FnVersion = symbol(&loaded, lib.version_symbol())?;
            // SAFETY: the version functions take nothing and only return.
            super::check_major(lib, unsafe { version() })?;
            Ok(loaded)
        };

        let libs = Libs {
            avutil: open(Library::Avutil)?,
            swresample: open(Library::Swresample)?,
            avcodec: open(Library::Avcodec)?,
            swscale: open(Library::Swscale)?,
        };
        let f = Fns::resolve(&libs)?;
        check_layout(&f)?;

        // libav's own log goes to stderr at its "info" level by default;
        // a decode warning there is noise beside the sidecar's own lines,
        // and every failure the shim sees comes back as an error anyway.
        // SAFETY: sets one global int.
        unsafe { (f.av_log_set_level)(AV_LOG_ERROR) };

        // libavfilter last, since it imports the others; its absence or
        // refusal is kept for `filter` to report and refuses nothing else.
        let filters = if dir.join(Library::Avfilter.file_name()).is_file() {
            open(Library::Avfilter).and_then(|avfilter| {
                let libs = FilterLibs { avfilter };
                let ff = FilterFns::resolve(&libs)?;
                check_filter_layout(&f, &ff)?;
                Ok(FilterLib { f: ff, _libs: libs })
            })
        } else {
            Err(missing(Library::Avfilter.file_name(), None))
        };

        let mut lib = Lib {
            f,
            wire: Vec::new(),
            filters,
            _libs: libs,
        };
        let mut wire = Vec::new();
        for &name in WIRE_PIX_FMTS {
            match lib.pix_fmt(name) {
                Some(fmt) => wire.push((name, fmt)),
                None => {
                    return Err(Error::Layout {
                        what: format!("libavutil does not know the pixel format {name}"),
                    })
                }
            }
        }
        lib.wire = wire;
        Ok(lib)
    }

    /// A library's version, as the int its `*_version()` returns; 0 for
    /// libavfilter where it is not loaded.
    pub fn version(&self, lib: Library) -> u32 {
        // SAFETY: the version functions take nothing and only return.
        unsafe {
            match lib {
                Library::Avutil => (self.f.avutil_version)(),
                Library::Swresample => (self.f.swresample_version)(),
                Library::Avcodec => (self.f.avcodec_version)(),
                Library::Swscale => (self.f.swscale_version)(),
                Library::Avfilter => self
                    .filters
                    .as_ref()
                    .map_or(0, |fl| (fl.f.avfilter_version)()),
            }
        }
    }

    /// libavfilter, or the refusal that says why it is not loaded.
    pub fn filter_lib(&self) -> Result<&FilterLib> {
        self.filters.as_ref().map_err(Clone::clone)
    }

    /// The names of the filters the loaded libavfilter has; empty where
    /// none is loaded.
    pub fn filter_names(&self) -> Vec<String> {
        let Ok(fl) = &self.filters else {
            return Vec::new();
        };
        let mut out = Vec::new();
        let mut opaque: *mut c_void = ptr::null_mut();
        loop {
            // SAFETY: `opaque` is the iteration state libavfilter owns,
            // started at null; the filters it returns are static.
            let filter = unsafe { (fl.f.av_filter_iterate)(&mut opaque) };
            if filter.is_null() {
                return out;
            }
            // SAFETY: a registered filter's name is a static string.
            if let Some(name) = unsafe { text((*filter).name) } {
                out.push(name);
            }
        }
    }

    /// A `d3d12va` frames context of `sw_format` textures of `width` x
    /// `height`, on a device built around `device` (an `ID3D12Device *`
    /// whose one reference moves in), or on a device of its own when
    /// `device` is null.
    pub fn d3d12_frames(
        &'static self,
        device_type: c_int,
        device: *mut c_void,
        sw_format: c_int,
        width: c_int,
        height: c_int,
    ) -> std::result::Result<BufferRef, AvErr> {
        let d3d12 = self.pix_fmt("d3d12").ok_or_else(|| AvErr {
            code: 0,
            why: "libavutil has no d3d12 pixel format".to_string(),
        })?;
        let device_ref = if device.is_null() {
            self.default_device(device_type)?
        } else {
            self.d3d12_device(device_type, device)?
        };
        self.frames_on(device_ref, d3d12, sw_format, width, height)
    }

    /// A frames context of `format` pictures holding `sw_format` of `width`
    /// x `height`, on a default device of `device_type` of its own: a
    /// `videotoolbox` one's pictures are IOSurface-backed `CVPixelBuffer`s.
    pub fn device_frames(
        &'static self,
        device_type: c_int,
        format: c_int,
        sw_format: c_int,
        width: c_int,
        height: c_int,
    ) -> std::result::Result<BufferRef, AvErr> {
        let device_ref = self.default_device(device_type)?;
        self.frames_on(device_ref, format, sw_format, width, height)
    }

    /// A default device of `device_type`, as a new reference.
    fn default_device(&self, device_type: c_int) -> std::result::Result<*mut AVBufferRef, AvErr> {
        let mut made: *mut AVBufferRef = ptr::null_mut();
        // SAFETY: the out-pointer is ours; null name and options ask for
        // the default device.
        let ret = unsafe {
            (self.f.av_hwdevice_ctx_create)(&mut made, device_type, ptr::null(), ptr::null_mut(), 0)
        };
        if ret < 0 || made.is_null() {
            return Err(self.err(ret));
        }
        Ok(made)
    }

    /// A frames context on `device_ref`, whose one reference moves in.
    fn frames_on(
        &'static self,
        mut device_ref: *mut AVBufferRef,
        format: c_int,
        sw_format: c_int,
        width: c_int,
        height: c_int,
    ) -> std::result::Result<BufferRef, AvErr> {
        let f = &self.f;
        // SAFETY: the frames reference av_hwframe_ctx_alloc returns has an
        // AVHWFramesContext as its data, holding its own reference to the
        // device; ours is given up either way. On failure the frames
        // reference is unreffed.
        unsafe {
            let mut frames = (f.av_hwframe_ctx_alloc)(device_ref);
            (f.av_buffer_unref)(&mut device_ref);
            if frames.is_null() {
                return Err(AvErr {
                    code: 0,
                    why: "av_hwframe_ctx_alloc allocated nothing".to_string(),
                });
            }
            let ctx = (*frames).data as *mut AVHWFramesContext;
            (*ctx).format = format;
            (*ctx).sw_format = sw_format;
            (*ctx).width = width;
            (*ctx).height = height;
            let ret = (f.av_hwframe_ctx_init)(frames);
            if ret < 0 {
                (f.av_buffer_unref)(&mut frames);
                return Err(self.err(ret));
            }
            Ok(BufferRef {
                ptr: frames,
                lib: self,
            })
        }
    }

    /// FFmpeg's own version string for the build, `n9.0-...`.
    pub fn version_info(&self) -> String {
        // SAFETY: returns a static string.
        unsafe { text((self.f.av_version_info)()) }.unwrap_or_default()
    }

    /// The licence libavcodec was built under.
    pub fn license(&self) -> String {
        // SAFETY: returns a static string.
        unsafe { text((self.f.avcodec_license)()) }.unwrap_or_default()
    }

    /// libav's words for an error code.
    pub fn strerror(&self, code: c_int) -> String {
        let mut buf = [0 as c_char; 256];
        // SAFETY: the buffer and its length go together, and av_strerror
        // always NUL-terminates what it writes.
        unsafe {
            (self.f.av_strerror)(code, buf.as_mut_ptr(), buf.len());
            text(buf.as_ptr())
        }
        .unwrap_or_else(|| format!("error {code}"))
    }

    fn err(&self, code: c_int) -> AvErr {
        AvErr {
            code,
            why: self.strerror(code),
        }
    }

    fn status(&self, code: c_int) -> Status {
        match code {
            c if c >= 0 => Status::Done,
            AVERROR_EAGAIN => Status::Again,
            AVERROR_EOF => Status::Eof,
            c => Status::Failed(self.err(c)),
        }
    }

    /// A `d3d12va` device reference built around `device`, an
    /// `ID3D12Device *` whose one reference moves in: the device context
    /// holds it from then on and releases it when freed, or here when the
    /// device cannot be made.
    fn d3d12_device(
        &self,
        device_type: c_int,
        device: *mut c_void,
    ) -> std::result::Result<*mut AVBufferRef, AvErr> {
        let f = &self.f;
        // SAFETY: the reference av_hwdevice_ctx_alloc returns has a
        // AVHWDeviceContext as its data, whose hwctx is the d3d12va device
        // context for a device of `device_type`; on failure the reference is
        // unreffed, which releases the device set into it.
        unsafe {
            let mut made = (f.av_hwdevice_ctx_alloc)(device_type);
            if made.is_null() {
                release_com(device);
                return Err(AvErr {
                    code: 0,
                    why: "av_hwdevice_ctx_alloc allocated nothing".to_string(),
                });
            }
            let ctx = (*made).data as *mut AVHWDeviceContext;
            let hw = (*ctx).hwctx as *mut AVD3D12VADeviceContext;
            (*hw).device = device;
            let ret = (f.av_hwdevice_ctx_init)(made);
            if ret < 0 {
                (f.av_buffer_unref)(&mut made);
                return Err(self.err(ret));
            }
            Ok(made)
        }
    }

    /// Every codec the build registered, with whether each decodes and
    /// whether each encodes.
    pub fn codecs(&'static self) -> Vec<(Codec, bool, bool)> {
        let mut out = Vec::new();
        let mut opaque: *mut c_void = ptr::null_mut();
        loop {
            // SAFETY: `opaque` is the iteration state libavcodec owns,
            // started at null; the codecs it returns are static.
            let codec = unsafe { (self.f.av_codec_iterate)(&mut opaque) };
            if codec.is_null() {
                return out;
            }
            // SAFETY: a codec av_codec_iterate returned.
            let (decodes, encodes) = unsafe {
                (
                    (self.f.av_codec_is_decoder)(codec) != 0,
                    (self.f.av_codec_is_encoder)(codec) != 0,
                )
            };
            out.push((
                Codec {
                    ptr: codec,
                    lib: self,
                },
                decodes,
                encodes,
            ));
        }
    }

    /// The codec id ffmpeg's descriptor table gives `name`, which is the
    /// codec's name rather than any one decoder's or encoder's.
    fn codec_id(&self, name: &str) -> Option<c_int> {
        let cname = CString::new(name).ok()?;
        // SAFETY: a NUL-terminated name; descriptors are static.
        unsafe {
            let desc = (self.f.avcodec_descriptor_get_by_name)(cname.as_ptr());
            (!desc.is_null()).then(|| (*desc).id)
        }
    }

    /// The decoder for `name`: libavcodec's preferred one for that codec
    /// when `name` is a codec (`av1` is libdav1d where the build has it,
    /// since the native one decodes only on hardware), else the decoder of
    /// that name (`libdav1d`).
    pub fn find_decoder(&'static self, name: &str) -> Option<Codec> {
        let found = match self.codec_id(name) {
            // SAFETY: an id from the descriptor table.
            Some(id) => unsafe { (self.f.avcodec_find_decoder)(id) },
            None => {
                let cname = CString::new(name).ok()?;
                // SAFETY: a NUL-terminated name.
                unsafe { (self.f.avcodec_find_decoder_by_name)(cname.as_ptr()) }
            }
        };
        (!found.is_null()).then_some(Codec {
            ptr: found,
            lib: self,
        })
    }

    /// Every decoder for the codec `codec` decodes, the preferred first.
    pub fn decoders_like(&'static self, codec: Codec) -> Vec<Codec> {
        let id = codec.id();
        let mut out = vec![codec];
        for (other, decodes, _) in self.codecs() {
            if decodes && other.id() == id && other.ptr != codec.ptr {
                out.push(other);
            }
        }
        out
    }

    /// The encoder called `name` (`libx264`, `ffv1`), else libavcodec's
    /// preferred encoder for the codec called `name` (`h264`).
    pub fn find_encoder(&'static self, name: &str) -> Option<Codec> {
        let cname = CString::new(name).ok()?;
        // SAFETY: a NUL-terminated name.
        let mut found = unsafe { (self.f.avcodec_find_encoder_by_name)(cname.as_ptr()) };
        if found.is_null() {
            if let Some(id) = self.codec_id(name) {
                // SAFETY: an id from the descriptor table.
                found = unsafe { (self.f.avcodec_find_encoder)(id) };
            }
        }
        (!found.is_null()).then_some(Codec {
            ptr: found,
            lib: self,
        })
    }

    /// The device type libavutil calls `name`, or None for one it does not
    /// know or this build does not have.
    pub fn hw_type(&self, name: &str) -> Option<c_int> {
        let cname = CString::new(name).ok()?;
        // SAFETY: a NUL-terminated name.
        let found = unsafe { (self.f.av_hwdevice_find_type_by_name)(cname.as_ptr()) };
        if found == AV_HWDEVICE_TYPE_NONE || !self.hw_types_raw().contains(&found) {
            return None;
        }
        Some(found)
    }

    fn hw_types_raw(&self) -> Vec<c_int> {
        let mut out = Vec::new();
        let mut at = AV_HWDEVICE_TYPE_NONE;
        loop {
            // SAFETY: iterates from NONE until it comes back to NONE.
            at = unsafe { (self.f.av_hwdevice_iterate_types)(at) };
            if at == AV_HWDEVICE_TYPE_NONE {
                return out;
            }
            out.push(at);
        }
    }

    /// The hardware device types this build of libavutil can create.
    pub fn hw_types(&self) -> Vec<String> {
        self.hw_types_raw()
            .into_iter()
            // SAFETY: a type the iteration returned; names are static.
            .filter_map(|t| unsafe { text((self.f.av_hwdevice_get_type_name)(t)) })
            .collect()
    }

    /// libavutil's number for a pixel format, by name.
    pub fn pix_fmt(&self, name: &str) -> Option<c_int> {
        let cname = CString::new(name).ok()?;
        // SAFETY: a NUL-terminated name.
        let fmt = unsafe { (self.f.av_get_pix_fmt)(cname.as_ptr()) };
        (fmt != AV_PIX_FMT_NONE).then_some(fmt)
    }

    /// The name of a pixel format, or `none`.
    pub fn pix_fmt_name(&self, fmt: c_int) -> String {
        // SAFETY: an out-of-range value returns null; names are static.
        unsafe { text((self.f.av_get_pix_fmt_name)(fmt)) }.unwrap_or_else(|| "none".to_string())
    }

    /// Whether a pixel format is a hardware device's, its pictures held
    /// there rather than in host memory (`d3d12`, `videotoolbox_vld`).
    pub fn pix_fmt_is_hw(&self, fmt: c_int) -> bool {
        // SAFETY: an out-of-range value returns null; descriptors are static.
        unsafe {
            let desc = (self.f.av_pix_fmt_desc_get)(fmt);
            !desc.is_null() && (*desc).flags & AV_PIX_FMT_FLAG_HWACCEL != 0
        }
    }

    /// Whether a pixel format is rgb rather than yuv.
    pub fn pix_fmt_is_rgb(&self, fmt: c_int) -> bool {
        // SAFETY: an out-of-range value returns null; descriptors are static.
        unsafe {
            let desc = (self.f.av_pix_fmt_desc_get)(fmt);
            !desc.is_null() && (*desc).flags & AV_PIX_FMT_FLAG_RGB != 0
        }
    }

    /// The number of one of the wire's own formats.
    pub fn wire_pix_fmt(&self, name: &str) -> Option<c_int> {
        self.wire.iter().find(|(n, _)| *n == name).map(|(_, f)| *f)
    }

    /// A colour's values by ffmpeg's names: `tv`, `bt709`, and so on.
    pub fn color_names(&self, c: RawColor) -> [String; 4] {
        // SAFETY: an out-of-range value returns null; names are static.
        let name = |f: FnEnumName, v: c_int| {
            unsafe { text(f(v)) }.unwrap_or_else(|| "unknown".to_string())
        };
        [
            name(self.f.av_color_range_name, c.range),
            name(self.f.av_color_primaries_name, c.primaries),
            name(self.f.av_color_transfer_name, c.trc),
            name(self.f.av_color_space_name, c.space),
        ]
    }

    /// A colour's values from ffmpeg's names. An empty name is left
    /// unspecified; one libavutil does not know is refused, naming which.
    pub fn color_values(
        &self,
        names: [&str; 4],
    ) -> std::result::Result<RawColor, (&'static str, String)> {
        const KEYS: [&str; 4] = ["color_range", "color_primaries", "color_trc", "colorspace"];
        let fns = [
            self.f.av_color_range_from_name,
            self.f.av_color_primaries_from_name,
            self.f.av_color_transfer_from_name,
            self.f.av_color_space_from_name,
        ];
        let u = RawColor::UNSPECIFIED;
        let mut values = [u.range, u.primaries, u.trc, u.space];
        for i in 0..4 {
            let name = names[i];
            if name.is_empty() {
                continue;
            }
            let cname = CString::new(name).map_err(|_| (KEYS[i], name.to_string()))?;
            // SAFETY: a NUL-terminated name.
            let value = unsafe { fns[i](cname.as_ptr()) };
            if value < 0 {
                return Err((KEYS[i], name.to_string()));
            }
            values[i] = value;
        }
        Ok(RawColor {
            range: values[0],
            primaries: values[1],
            trc: values[2],
            space: values[3],
        })
    }
}

/// The options the layout check reads offsets from, and the field each one
/// stores into. libavcodec's option table names where the loaded build keeps
/// each field, so comparing that with `offset_of!` checks the transcription
/// of `AVCodecContext` against the library actually loaded, from its first
/// fields to `extra_hw_frames`, the last one transcribed.
const CONTEXT_OPTIONS: &[(&str, &str, usize)] = {
    use std::mem::offset_of;
    type C = AVCodecContext;
    &[
        ("b", "bit_rate", offset_of!(C, bit_rate)),
        ("flags", "flags", offset_of!(C, flags)),
        ("flags2", "flags2", offset_of!(C, flags2)),
        ("time_base", "time_base", offset_of!(C, time_base)),
        ("pkt_timebase", "pkt_timebase", offset_of!(C, pkt_timebase)),
        ("video_size", "width", offset_of!(C, width)),
        (
            "sar",
            "sample_aspect_ratio",
            offset_of!(C, sample_aspect_ratio),
        ),
        ("pixel_format", "pix_fmt", offset_of!(C, pix_fmt)),
        (
            "color_primaries",
            "color_primaries",
            offset_of!(C, color_primaries),
        ),
        ("color_trc", "color_trc", offset_of!(C, color_trc)),
        ("colorspace", "colorspace", offset_of!(C, colorspace)),
        ("color_range", "color_range", offset_of!(C, color_range)),
        (
            "chroma_sample_location",
            "chroma_sample_location",
            offset_of!(C, chroma_sample_location),
        ),
        ("field_order", "field_order", offset_of!(C, field_order)),
        ("refs", "refs", offset_of!(C, refs)),
        ("bf", "max_b_frames", offset_of!(C, max_b_frames)),
        ("nssew", "nsse_weight", offset_of!(C, nsse_weight)),
        ("mbd", "mb_decision", offset_of!(C, mb_decision)),
        (
            "dc",
            "intra_dc_precision",
            offset_of!(C, intra_dc_precision),
        ),
        ("keyint_min", "keyint_min", offset_of!(C, keyint_min)),
        ("g", "gop_size", offset_of!(C, gop_size)),
        ("slices", "slices", offset_of!(C, slices)),
        ("ar", "sample_rate", offset_of!(C, sample_rate)),
        ("ch_layout", "ch_layout", offset_of!(C, ch_layout)),
        ("frame_size", "frame_size", offset_of!(C, frame_size)),
        ("cutoff", "cutoff", offset_of!(C, cutoff)),
        (
            "bt",
            "bit_rate_tolerance",
            offset_of!(C, bit_rate_tolerance),
        ),
        (
            "global_quality",
            "global_quality",
            offset_of!(C, global_quality),
        ),
        ("qcomp", "qcompress", offset_of!(C, qcompress)),
        ("bufsize", "rc_buffer_size", offset_of!(C, rc_buffer_size)),
        ("maxrate", "rc_max_rate", offset_of!(C, rc_max_rate)),
        ("minrate", "rc_min_rate", offset_of!(C, rc_min_rate)),
        ("trellis", "trellis", offset_of!(C, trellis)),
        (
            "strict",
            "strict_std_compliance",
            offset_of!(C, strict_std_compliance),
        ),
        ("debug", "debug", offset_of!(C, debug)),
        (
            "err_detect",
            "err_recognition",
            offset_of!(C, err_recognition),
        ),
        (
            "hwaccel_flags",
            "hwaccel_flags",
            offset_of!(C, hwaccel_flags),
        ),
        (
            "extra_hw_frames",
            "extra_hw_frames",
            offset_of!(C, extra_hw_frames),
        ),
    ]
};

/// Checks the structs in `sys` against the libraries loaded: the codec
/// context's fields against the offsets its own option table gives, and the
/// frame, packet and parameters structs against the defaults their
/// allocators are documented to write. Any difference refuses the load.
fn check_layout(f: &Fns) -> Result<()> {
    let mut wrong = Vec::new();

    // SAFETY: each struct comes from its own allocator, is read through the
    // transcription under test only where a mismatch would still be inside
    // the allocation (every field read lies before the struct's end in both
    // the header and the transcription), and is freed by its counterpart.
    unsafe {
        let mut ctx = (f.avcodec_alloc_context3)(ptr::null());
        if ctx.is_null() {
            return Err(Error::Layout {
                what: "avcodec_alloc_context3 allocated nothing".to_string(),
            });
        }
        for &(option, field, ours) in CONTEXT_OPTIONS {
            let cname = CString::new(option).expect("option names have no NUL");
            let opt = (f.av_opt_find)(ctx.cast(), cname.as_ptr(), ptr::null(), 0, 0);
            if opt.is_null() {
                wrong.push(format!("AVCodecContext has no option {option}"));
                continue;
            }
            let theirs = (*opt).offset as usize;
            if theirs != ours {
                wrong.push(format!(
                    "AVCodecContext.{field} is at {theirs}, and this build put it at {ours}"
                ));
            }
        }
        (f.avcodec_free_context)(&mut ctx);

        let mut frame = (f.av_frame_alloc)();
        if frame.is_null() {
            return Err(Error::Layout {
                what: "av_frame_alloc allocated nothing".to_string(),
            });
        }
        {
            let fr = &mut *frame;
            let unset = AVRational { num: 0, den: 1 };
            let fresh = fr.format == AV_PIX_FMT_NONE
                && fr.pts == AV_NOPTS_VALUE
                && fr.pkt_dts == AV_NOPTS_VALUE
                && fr.best_effort_timestamp == AV_NOPTS_VALUE
                && fr.extended_data == fr.data.as_mut_ptr()
                && fr.sample_aspect_ratio == unset
                && fr.time_base == unset
                && fr.color_primaries == AVCOL_PRI_UNSPECIFIED
                && fr.color_trc == AVCOL_TRC_UNSPECIFIED
                && fr.colorspace == AVCOL_SPC_UNSPECIFIED
                && fr.duration == 0
                && fr.hw_frames_ctx.is_null();
            if !fresh {
                wrong.push("a new AVFrame does not hold the defaults av_frame_alloc writes".into());
            } else {
                fr.width = 34;
                fr.height = 18;
                fr.format = (f.av_get_pix_fmt)(c"yuv420p".as_ptr());
                let ret = (f.av_frame_get_buffer)(frame, 0);
                let fr = &*frame;
                let planes_ok = ret >= 0
                    && (0..3).all(|i| !fr.data[i].is_null())
                    && !fr.buf[0].is_null()
                    && fr.linesize[0] >= 34
                    && fr.linesize[1] >= 17
                    && fr.data[3].is_null();
                if !planes_ok {
                    wrong.push("an AVFrame's planes are not where this build looks".into());
                }
            }
        }
        (f.av_frame_free)(&mut frame);

        let mut packet = (f.av_packet_alloc)();
        if packet.is_null() {
            return Err(Error::Layout {
                what: "av_packet_alloc allocated nothing".to_string(),
            });
        }
        {
            let p = &*packet;
            let fresh = p.pts == AV_NOPTS_VALUE
                && p.dts == AV_NOPTS_VALUE
                && p.pos == -1
                && p.data.is_null()
                && p.size == 0
                && p.time_base == AVRational { num: 0, den: 1 };
            let filled = fresh && (f.av_new_packet)(packet, 16) >= 0 && {
                let p = &*packet;
                p.size == 16 && !p.data.is_null() && !p.buf.is_null()
            };
            if !filled {
                wrong.push("an AVPacket's fields are not where this build looks".into());
            }
        }
        (f.av_packet_free)(&mut packet);

        let mut par = (f.avcodec_parameters_alloc)();
        if par.is_null() {
            return Err(Error::Layout {
                what: "avcodec_parameters_alloc allocated nothing".to_string(),
            });
        }
        {
            let p = &*par;
            let unset = AVRational { num: 0, den: 1 };
            let fresh = p.codec_type == -1
                && p.format == -1
                && p.profile == -99
                && p.level == -99
                && p.sample_aspect_ratio == unset
                && p.framerate == unset
                && p.color_primaries == AVCOL_PRI_UNSPECIFIED
                && p.color_trc == AVCOL_TRC_UNSPECIFIED
                && p.color_space == AVCOL_SPC_UNSPECIFIED
                && p.extradata.is_null();
            if !fresh {
                wrong.push(
                    "new AVCodecParameters do not hold the defaults their allocator writes".into(),
                );
            }
        }
        (f.avcodec_parameters_free)(&mut par);
    }

    if wrong.is_empty() {
        Ok(())
    } else {
        Err(Error::Layout {
            what: wrong.join("; "),
        })
    }
}

/// The options the filter layout check reads offsets from, as
/// `CONTEXT_OPTIONS` does for the codec context: a filter context's own
/// option table and a graph's name where the loaded build keeps each field.
const FILTER_CONTEXT_OPTIONS: &[(&str, &str, usize)] = {
    use std::mem::offset_of;
    type C = AVFilterContext;
    &[
        ("thread_type", "thread_type", offset_of!(C, thread_type)),
        ("enable", "enable_str", offset_of!(C, enable_str)),
        ("threads", "nb_threads", offset_of!(C, nb_threads)),
        (
            "extra_hw_frames",
            "extra_hw_frames",
            offset_of!(C, extra_hw_frames),
        ),
    ]
};
const GRAPH_OPTIONS: &[(&str, &str, usize)] = {
    use std::mem::offset_of;
    type G = AVFilterGraph;
    &[
        (
            "scale_sws_opts",
            "scale_sws_opts",
            offset_of!(G, scale_sws_opts),
        ),
        ("thread_type", "thread_type", offset_of!(G, thread_type)),
        ("threads", "nb_threads", offset_of!(G, nb_threads)),
    ]
};

/// Checks the libavfilter structs in `sys` against the library loaded: the
/// filter context's and the graph's fields against their option tables,
/// the graph's list of filters against a filter allocated into it, and the
/// source parameters against the defaults their allocator writes.
fn check_filter_layout(f: &Fns, ff: &FilterFns) -> Result<()> {
    let mut wrong = Vec::new();
    let offsets = |object: *mut c_void, of: &str, table: &[(&str, &str, usize)]| {
        let mut wrong = Vec::new();
        for &(option, field, ours) in table {
            let cname = CString::new(option).expect("option names have no NUL");
            // SAFETY: a live object whose AVClass is its first field.
            let opt = unsafe { (f.av_opt_find)(object, cname.as_ptr(), ptr::null(), 0, 0) };
            if opt.is_null() {
                wrong.push(format!("{of} has no option {option}"));
                continue;
            }
            // SAFETY: an option libavutil returned.
            let theirs = unsafe { (*opt).offset } as usize;
            if theirs != ours {
                wrong.push(format!(
                    "{of}.{field} is at {theirs}, and this build put it at {ours}"
                ));
            }
        }
        wrong
    };

    // SAFETY: the graph and its filter come from their allocators and are
    // freed by the graph's; every field read lies inside the structs in
    // both the headers and the transcription, and the parameters go back to
    // av_free.
    unsafe {
        let mut graph = (ff.avfilter_graph_alloc)();
        if graph.is_null() {
            return Err(Error::Layout {
                what: "avfilter_graph_alloc allocated nothing".to_string(),
            });
        }
        wrong.extend(offsets(graph.cast(), "AVFilterGraph", GRAPH_OPTIONS));
        let buffer = (ff.avfilter_get_by_name)(c"buffer".as_ptr());
        if buffer.is_null() {
            wrong.push("libavfilter has no buffer filter".to_string());
        } else {
            let ctx = (ff.avfilter_graph_alloc_filter)(graph, buffer, c"ffrwd_check".as_ptr());
            if ctx.is_null() {
                wrong.push("avfilter_graph_alloc_filter allocated nothing".to_string());
            } else {
                let g = &*graph;
                let c = &*ctx;
                let listed = g.nb_filters == 1 && !g.filters.is_null() && *g.filters == ctx;
                let named = text(c.name).as_deref() == Some("ffrwd_check")
                    && text((*c.filter).name).as_deref() == Some("buffer");
                let placed = c.filter == buffer
                    && c.graph == graph
                    && c.nb_inputs == 0
                    && c.nb_outputs == 1
                    && c.hw_device_ctx.is_null();
                if !(listed && named && placed) {
                    wrong.push(
                        "a filter allocated into a graph is not where this build looks".into(),
                    );
                }
                wrong.extend(offsets(
                    ctx.cast(),
                    "AVFilterContext",
                    FILTER_CONTEXT_OPTIONS,
                ));
            }
        }
        (ff.avfilter_graph_free)(&mut graph);

        let par = (ff.av_buffersrc_parameters_alloc)();
        if par.is_null() {
            return Err(Error::Layout {
                what: "av_buffersrc_parameters_alloc allocated nothing".to_string(),
            });
        }
        let p = &*par;
        let unset = AVRational { num: 0, den: 0 };
        let fresh = p.format == -1
            && p.time_base == unset
            && p.width == 0
            && p.height == 0
            && p.frame_rate == unset
            && p.hw_frames_ctx.is_null()
            && p.color_space == AVCOL_SPC_UNSPECIFIED
            && p.color_range == AVCOL_RANGE_UNSPECIFIED
            && p.side_data.is_null()
            && p.alpha_mode == 0;
        if !fresh {
            wrong.push(
                "new AVBufferSrcParameters do not hold the defaults their allocator writes".into(),
            );
        }
        (f.av_free)(par.cast());
    }

    if wrong.is_empty() {
        Ok(())
    } else {
        Err(Error::Layout {
            what: wrong.join("; "),
        })
    }
}

/// One reference to a libav buffer (a frames context), from whoever made it
/// to `av_buffer_unref`.
pub(super) struct BufferRef {
    ptr: *mut AVBufferRef,
    lib: &'static Lib,
}

// SAFETY: libav's buffer references are counted atomically; this handle is
// only moved.
unsafe impl Send for BufferRef {}

impl BufferRef {
    /// Gives the reference up to a struct that frees it itself.
    fn into_raw(self) -> *mut AVBufferRef {
        let raw = self.ptr;
        std::mem::forget(self);
        raw
    }

    /// A picture of this frames context's pool, in `frame`.
    pub fn get_buffer(&self, frame: &mut Frame) -> std::result::Result<(), AvErr> {
        frame.unref();
        // SAFETY: a live frames context and a live frame.
        let ret = unsafe { (self.lib.f.av_hwframe_get_buffer)(self.ptr, frame.ptr, 0) };
        if ret < 0 {
            return Err(self.lib.err(ret));
        }
        Ok(())
    }
}

impl Drop for BufferRef {
    fn drop(&mut self) {
        // SAFETY: the one reference this holds.
        unsafe { (self.lib.f.av_buffer_unref)(&mut self.ptr) };
    }
}

/// What a filter graph's `buffer` source is told about the pictures it is
/// handed.
pub(super) struct SourceParams<'a> {
    pub format: c_int,
    pub width: c_int,
    pub height: c_int,
    pub time_base: AVRational,
    pub frame_rate: Option<AVRational>,
    pub color: RawColor,
    /// The frames context the pictures come from, for a device format.
    pub frames: Option<&'a BufferRef>,
}

/// A filter graph, from `avfilter_graph_alloc` to `avfilter_graph_free`:
/// a `buffer` source, the graph a description parses to, and a
/// `buffersink`, configured.
pub(super) struct FilterGraph {
    ptr: *mut AVFilterGraph,
    src: *mut AVFilterContext,
    sink: *mut AVFilterContext,
    lib: &'static Lib,
}

// SAFETY: as for `CodecCtx`: one thread at a time, through `&mut self`.
unsafe impl Send for FilterGraph {}

impl FilterGraph {
    /// Parses `description` (libavfilter's own syntax, one input and one
    /// output) between a source told `params` and a sink, and configures
    /// it. The error names the call that failed.
    pub fn new(
        lib: &'static Lib,
        description: &str,
        params: &SourceParams,
    ) -> std::result::Result<FilterGraph, (&'static str, AvErr)> {
        let fl = lib.filter_lib().map_err(|e| {
            (
                "libavfilter",
                AvErr {
                    code: 0,
                    why: e.to_string(),
                },
            )
        })?;
        let (f, ff) = (&lib.f, &fl.f);
        let none = |call: &'static str, what: &str| {
            (
                call,
                AvErr {
                    code: 0,
                    why: format!("{what} allocated nothing"),
                },
            )
        };
        let described = CString::new(description).map_err(|_| {
            (
                "avfilter_graph_parse_ptr",
                AvErr {
                    code: 0,
                    why: "the graph holds a NUL byte".to_string(),
                },
            )
        })?;
        // SAFETY: a fresh graph; null is checked.
        let ptr = unsafe { (ff.avfilter_graph_alloc)() };
        if ptr.is_null() {
            return Err(none("avfilter_graph_alloc", "avfilter_graph_alloc"));
        }
        // The graph frees every filter in it, whatever happens below.
        let mut graph = FilterGraph {
            ptr,
            src: ptr::null_mut(),
            sink: ptr::null_mut(),
            lib,
        };
        let failed = |call: &'static str, code: c_int| (call, lib.err(code));
        // SAFETY: the filters are allocated into the live graph and
        // initialised once; the parameters go back to av_free; the in/out
        // lists are ours until parsed, and freed either way.
        unsafe {
            let buffer = (ff.avfilter_get_by_name)(c"buffer".as_ptr());
            let buffersink = (ff.avfilter_get_by_name)(c"buffersink".as_ptr());
            if buffer.is_null() || buffersink.is_null() {
                return Err((
                    "avfilter_get_by_name",
                    AvErr {
                        code: 0,
                        why: "libavfilter has no buffer or buffersink".to_string(),
                    },
                ));
            }
            graph.src = (ff.avfilter_graph_alloc_filter)(ptr, buffer, c"in".as_ptr());
            if graph.src.is_null() {
                return Err(none("avfilter_graph_alloc_filter", "buffer"));
            }
            let par = (ff.av_buffersrc_parameters_alloc)();
            if par.is_null() {
                return Err(none("av_buffersrc_parameters_alloc", "it"));
            }
            {
                let p = &mut *par;
                p.format = params.format;
                p.width = params.width;
                p.height = params.height;
                p.time_base = params.time_base;
                p.sample_aspect_ratio = AVRational { num: 1, den: 1 };
                if let Some(rate) = params.frame_rate {
                    p.frame_rate = rate;
                }
                p.color_space = params.color.space;
                p.color_range = params.color.range;
                p.hw_frames_ctx = params.frames.map_or(ptr::null_mut(), |r| r.ptr);
            }
            let ret = (ff.av_buffersrc_parameters_set)(graph.src, par);
            (f.av_free)(par.cast());
            if ret < 0 {
                return Err(failed("av_buffersrc_parameters_set", ret));
            }
            let ret = (ff.avfilter_init_str)(graph.src, ptr::null());
            if ret < 0 {
                return Err(failed("avfilter_init_str buffer", ret));
            }
            graph.sink = (ff.avfilter_graph_alloc_filter)(ptr, buffersink, c"out".as_ptr());
            if graph.sink.is_null() {
                return Err(none("avfilter_graph_alloc_filter", "buffersink"));
            }
            let ret = (ff.avfilter_init_str)(graph.sink, ptr::null());
            if ret < 0 {
                return Err(failed("avfilter_init_str buffersink", ret));
            }

            // The description's open input is the source's output, and its
            // open output the sink's input.
            let mut outputs = (ff.avfilter_inout_alloc)();
            let mut inputs = (ff.avfilter_inout_alloc)();
            if outputs.is_null() || inputs.is_null() {
                (ff.avfilter_inout_free)(&mut outputs);
                (ff.avfilter_inout_free)(&mut inputs);
                return Err(none("avfilter_inout_alloc", "avfilter_inout_alloc"));
            }
            (*outputs).name = (f.av_strdup)(c"in".as_ptr());
            (*outputs).filter_ctx = graph.src;
            (*inputs).name = (f.av_strdup)(c"out".as_ptr());
            (*inputs).filter_ctx = graph.sink;
            let ret = (ff.avfilter_graph_parse_ptr)(
                ptr,
                described.as_ptr(),
                &mut inputs,
                &mut outputs,
                ptr::null_mut(),
            );
            (ff.avfilter_inout_free)(&mut inputs);
            (ff.avfilter_inout_free)(&mut outputs);
            if ret < 0 {
                return Err(failed("avfilter_graph_parse_ptr", ret));
            }
            let ret = (ff.avfilter_graph_config)(ptr, ptr::null_mut());
            if ret < 0 {
                return Err(failed("avfilter_graph_config", ret));
            }
        }
        Ok(graph)
    }

    fn fns(&self) -> &FilterFns {
        &self
            .lib
            .filter_lib()
            .expect("a graph exists only where libavfilter loaded")
            .f
    }

    /// The names of the filters the description made, in the graph's
    /// order: every filter but the source and the sink, the ones libavfilter
    /// inserted to convert between two included.
    pub fn filter_names(&self) -> Vec<String> {
        let mut out = Vec::new();
        // SAFETY: a configured graph's list holds `nb_filters` live filter
        // contexts, each naming a static filter.
        unsafe {
            let g = &*self.ptr;
            for i in 0..g.nb_filters as usize {
                let ctx = *g.filters.add(i);
                if ctx == self.src || ctx == self.sink || ctx.is_null() {
                    continue;
                }
                if let Some(name) = text((*(*ctx).filter).name) {
                    out.push(name);
                }
            }
        }
        out
    }

    /// The sink's picture: width, height and pixel format.
    pub fn output(&self) -> (c_int, c_int, c_int) {
        let ff = self.fns();
        // SAFETY: a configured sink.
        unsafe {
            (
                (ff.av_buffersink_get_w)(self.sink),
                (ff.av_buffersink_get_h)(self.sink),
                (ff.av_buffersink_get_format)(self.sink),
            )
        }
    }

    /// The format a device picture leaving the sink holds its pixels in,
    /// from the sink's frames context; None for pictures in host memory.
    pub fn output_sw_format(&self) -> Option<c_int> {
        // SAFETY: a configured sink; its frames context, where it has one,
        // is an AVHWFramesContext alive while the graph is.
        unsafe {
            let frames = (self.fns().av_buffersink_get_hw_frames_ctx)(self.sink);
            if frames.is_null() {
                return None;
            }
            Some((*((*frames).data as *const AVHWFramesContext)).sw_format)
        }
    }

    /// The sink's time base, and its frame rate where it has one.
    pub fn timing(&self) -> (AVRational, Option<AVRational>) {
        let ff = self.fns();
        // SAFETY: a configured sink.
        let (base, rate) = unsafe {
            (
                (ff.av_buffersink_get_time_base)(self.sink),
                (ff.av_buffersink_get_frame_rate)(self.sink),
            )
        };
        (base, (rate.num > 0 && rate.den > 0).then_some(rate))
    }

    /// Hands the source a picture, moving it out of `frame`, or with None,
    /// the end of the stream.
    pub fn send(&mut self, frame: Option<&mut Frame>) -> Status {
        let p = frame.map_or(ptr::null_mut(), |f| f.ptr);
        // SAFETY: a configured source and a live frame, or null.
        let ret = unsafe { (self.fns().av_buffersrc_add_frame_flags)(self.src, p, 0) };
        self.lib.status(ret)
    }

    /// The next picture the sink has, in `frame`.
    pub fn receive(&mut self, frame: &mut Frame) -> Status {
        frame.unref();
        // SAFETY: a configured sink and a live frame.
        let ret = unsafe { (self.fns().av_buffersink_get_frame)(self.sink, frame.ptr) };
        self.lib.status(ret)
    }
}

impl Drop for FilterGraph {
    fn drop(&mut self) {
        let fl = self
            .lib
            .filter_lib()
            .expect("a graph exists only where libavfilter loaded");
        // SAFETY: allocated by avfilter_graph_alloc; this frees every
        // filter in it.
        unsafe { (fl.f.avfilter_graph_free)(&mut self.ptr) };
    }
}

/// A codec libavcodec registered. Static data inside the library, never
/// freed, and the same for every thread.
#[derive(Clone, Copy)]
pub(super) struct Codec {
    ptr: *const AVCodec,
    lib: &'static Lib,
}

// SAFETY: an AVCodec is immutable static data in a library that is never
// unloaded.
unsafe impl Send for Codec {}
unsafe impl Sync for Codec {}

impl Codec {
    /// The decoder's or encoder's own name: `h264`, `libdav1d`, `ffv1`.
    pub fn name(&self) -> String {
        // SAFETY: a registered codec's name is a static string.
        unsafe { text((*self.ptr).name) }.unwrap_or_default()
    }

    /// The codec's name, rather than this implementation's: `h264` for
    /// `libx264`.
    pub fn codec_name(&self) -> String {
        // SAFETY: names are static; an unknown id has one too.
        unsafe { text((self.lib.f.avcodec_get_name)(self.id())) }.unwrap_or_default()
    }

    fn id(&self) -> c_int {
        // SAFETY: a registered codec.
        unsafe { (*self.ptr).id }
    }

    /// Whether this is a hardware API's own decoder (`h264_cuvid`) rather
    /// than libavcodec's with a hwaccel, or one libavcodec marks
    /// experimental.
    pub fn is_wrapper_or_experimental(&self) -> bool {
        // SAFETY: a registered codec.
        let caps = unsafe { (*self.ptr).capabilities };
        caps & (AV_CODEC_CAP_HARDWARE | AV_CODEC_CAP_EXPERIMENTAL) != 0
    }

    /// Whether this decoder takes a device of `device_type` through the
    /// context's `hw_device_ctx`.
    pub fn takes_device(&self, device_type: c_int) -> bool {
        for i in 0.. {
            // SAFETY: indexes until libavcodec returns null; configs are
            // static.
            let config = unsafe { (self.lib.f.avcodec_get_hw_config)(self.ptr, i) };
            if config.is_null() {
                return false;
            }
            // SAFETY: a config libavcodec returned.
            let config = unsafe { &*config };
            if config.methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX != 0
                && config.device_type == device_type
            {
                return true;
            }
        }
        false
    }
}

/// An `AVCodecContext`, from `avcodec_alloc_context3` to
/// `avcodec_free_context`.
pub(super) struct CodecCtx {
    ptr: *mut AVCodecContext,
    codec: Codec,
    lib: &'static Lib,
}

// SAFETY: a codec context may move between threads; it is used from one at
// a time, which `&mut self` on every call that touches it guarantees.
unsafe impl Send for CodecCtx {}

impl CodecCtx {
    pub fn new(codec: Codec) -> std::result::Result<CodecCtx, AvErr> {
        let lib = codec.lib;
        // SAFETY: a registered codec; null is checked.
        let ptr = unsafe { (lib.f.avcodec_alloc_context3)(codec.ptr) };
        if ptr.is_null() {
            return Err(AvErr {
                code: 0,
                why: "avcodec_alloc_context3 allocated nothing".to_string(),
            });
        }
        Ok(CodecCtx { ptr, codec, lib })
    }

    pub fn codec(&self) -> Codec {
        self.codec
    }

    /// Describes the stream a decoder is about to read: video, this codec,
    /// the out-of-band header the container carried, and the colour it
    /// said, through `avcodec_parameters_to_context`. A colour the bitstream
    /// carries takes its place as the decoder reads it.
    pub fn set_stream(
        &mut self,
        extradata: &[u8],
        color: RawColor,
    ) -> std::result::Result<(), AvErr> {
        let f = &self.lib.f;
        let size = c_int::try_from(extradata.len()).map_err(|_| AvErr {
            code: 0,
            why: format!("{} bytes of extradata is too many", extradata.len()),
        })?;
        // SAFETY: the parameters come from their allocator and go back to
        // it, and the extradata buffer from av_mallocz with the padding
        // libavcodec reads past its end; avcodec_parameters_free frees it.
        unsafe {
            let mut par = (f.avcodec_parameters_alloc)();
            if par.is_null() {
                return Err(AvErr {
                    code: 0,
                    why: "avcodec_parameters_alloc allocated nothing".to_string(),
                });
            }
            (*par).codec_type = AVMEDIA_TYPE_VIDEO;
            (*par).codec_id = self.codec.id();
            (*par).color_range = color.range;
            (*par).color_primaries = color.primaries;
            (*par).color_trc = color.trc;
            (*par).color_space = color.space;
            if size > 0 {
                let buf = (f.av_mallocz)(extradata.len() + AV_INPUT_BUFFER_PADDING_SIZE) as *mut u8;
                if buf.is_null() {
                    (f.avcodec_parameters_free)(&mut par);
                    return Err(AvErr {
                        code: 0,
                        why: "av_mallocz allocated nothing".to_string(),
                    });
                }
                ptr::copy_nonoverlapping(extradata.as_ptr(), buf, extradata.len());
                (*par).extradata = buf;
                (*par).extradata_size = size;
            }
            let ret = (f.avcodec_parameters_to_context)(self.ptr, par);
            (f.avcodec_parameters_free)(&mut par);
            if ret < 0 {
                return Err(self.lib.err(ret));
            }
        }
        Ok(())
    }

    /// Creates a device of `device_type` and gives it to the context, which
    /// holds the reference from then on and releases it when freed.
    pub fn attach_device(&mut self, device_type: c_int) -> std::result::Result<(), AvErr> {
        let mut device: *mut AVBufferRef = ptr::null_mut();
        // SAFETY: the out-pointer is ours; null device name and options ask
        // for the default device.
        let ret = unsafe {
            (self.lib.f.av_hwdevice_ctx_create)(
                &mut device,
                device_type,
                ptr::null(),
                ptr::null_mut(),
                0,
            )
        };
        if ret < 0 || device.is_null() {
            return Err(self.lib.err(ret));
        }
        // SAFETY: a fresh context has no device; the reference moves in.
        unsafe { (*self.ptr).hw_device_ctx = device };
        Ok(())
    }

    /// Gives the context a `d3d12va` device built around `device`, an
    /// `ID3D12Device *` whose one reference moves in: the device context
    /// holds it from then on and releases it when freed, or here when the
    /// device cannot be made.
    pub fn attach_d3d12_device(
        &mut self,
        device_type: c_int,
        device: *mut c_void,
    ) -> std::result::Result<(), AvErr> {
        let made = self.lib.d3d12_device(device_type, device)?;
        // SAFETY: a fresh context has no device; the reference moves in.
        unsafe { (*self.ptr).hw_device_ctx = made };
        Ok(())
    }

    /// Gives an encoder that takes D3D12 frames the pool it reads them
    /// from: `sw_format` textures of `width` x `height` on a `d3d12va`
    /// device built around `device` (an `ID3D12Device *` whose one
    /// reference moves in), or on a device of its own when `device` is
    /// null. The context's pixel format becomes `d3d12`.
    pub fn attach_d3d12_frames(
        &mut self,
        device_type: c_int,
        device: *mut c_void,
        sw_format: c_int,
        width: c_int,
        height: c_int,
    ) -> std::result::Result<(), AvErr> {
        let frames = self
            .lib
            .d3d12_frames(device_type, device, sw_format, width, height)?;
        let d3d12 = self.lib.pix_fmt("d3d12").expect("d3d12_frames found it");
        // SAFETY: a fresh context has no frames context; the reference
        // moves in.
        unsafe {
            (*self.ptr).hw_frames_ctx = frames.into_raw();
            (*self.ptr).pix_fmt = d3d12;
            (*self.ptr).sw_pix_fmt = sw_format;
        }
        Ok(())
    }

    /// Gives an encoder the pool it reads its pictures from: `format`
    /// pictures holding `sw_format` of `width` x `height`, on a default
    /// device of `device_type`. The context's pixel format becomes
    /// `format`.
    pub fn attach_device_frames(
        &mut self,
        device_type: c_int,
        format: c_int,
        sw_format: c_int,
        width: c_int,
        height: c_int,
    ) -> std::result::Result<(), AvErr> {
        let frames = self
            .lib
            .device_frames(device_type, format, sw_format, width, height)?;
        // SAFETY: a fresh context has no frames context; the reference
        // moves in.
        unsafe {
            (*self.ptr).hw_frames_ctx = frames.into_raw();
            (*self.ptr).pix_fmt = format;
            (*self.ptr).sw_pix_fmt = sw_format;
        }
        Ok(())
    }

    /// A picture from the pool `attach_d3d12_frames` or
    /// `attach_device_frames` gave the context, in `frame`, which held
    /// nothing.
    pub fn pool_frame(&self, frame: &mut Frame) -> std::result::Result<(), AvErr> {
        frame.unref();
        // SAFETY: a live context and a live frame; the pool is the
        // context's own frames context, checked for null.
        unsafe {
            let pool = (*self.ptr).hw_frames_ctx;
            if pool.is_null() {
                return Err(AvErr {
                    code: 0,
                    why: "the encoder has no frames context".to_string(),
                });
            }
            let ret = (self.lib.f.av_hwframe_get_buffer)(pool, frame.ptr, 0);
            if ret < 0 {
                return Err(self.lib.err(ret));
            }
        }
        Ok(())
    }

    /// What an encoder is handed before it opens: the picture's geometry and
    /// pixel format, the time base its timestamps count in, and the frame
    /// rate where one is known.
    pub fn set_video(
        &mut self,
        width: c_int,
        height: c_int,
        pix_fmt: c_int,
        time_base: AVRational,
        framerate: Option<AVRational>,
    ) {
        // SAFETY: fields of a live context, before it opens.
        unsafe {
            let ctx = &mut *self.ptr;
            ctx.width = width;
            ctx.height = height;
            ctx.pix_fmt = pix_fmt;
            ctx.time_base = time_base;
            if let Some(rate) = framerate {
                ctx.framerate = rate;
            }
        }
    }

    pub fn set_color(&mut self, color: RawColor) {
        // SAFETY: fields of a live context, before it opens.
        unsafe {
            let ctx = &mut *self.ptr;
            ctx.color_range = color.range;
            ctx.color_primaries = color.primaries;
            ctx.color_trc = color.trc;
            ctx.colorspace = color.space;
        }
    }

    pub fn add_flags(&mut self, flags: c_int) {
        // SAFETY: a field of a live context, before it opens.
        unsafe { (*self.ptr).flags |= flags };
    }

    /// `av_opt_set` with `AV_OPT_SEARCH_CHILDREN`: the codec's own private
    /// options (`crf`, `preset`) are found first, then the generic ones.
    pub fn set_option(&mut self, key: &str, value: &str) -> std::result::Result<(), AvErr> {
        let invalid = |what: &str| AvErr {
            code: 0,
            why: format!("the {what} holds a NUL byte"),
        };
        let ckey = CString::new(key).map_err(|_| invalid("name"))?;
        let cvalue = CString::new(value).map_err(|_| invalid("value"))?;
        // SAFETY: a live context, whose AVClass is its first field.
        let ret = unsafe {
            (self.lib.f.av_opt_set)(
                self.ptr.cast(),
                ckey.as_ptr(),
                cvalue.as_ptr(),
                AV_OPT_SEARCH_CHILDREN,
            )
        };
        if ret < 0 {
            return Err(self.lib.err(ret));
        }
        Ok(())
    }

    pub fn open(&mut self) -> std::result::Result<(), AvErr> {
        // SAFETY: a live context, opened with the codec it was allocated for.
        let ret = unsafe { (self.lib.f.avcodec_open2)(self.ptr, self.codec.ptr, ptr::null_mut()) };
        if ret < 0 {
            return Err(self.lib.err(ret));
        }
        Ok(())
    }

    /// The pixel formats an encoder takes, in its order of preference;
    /// empty where it does not say.
    pub fn supported_pix_fmts(&self) -> Vec<c_int> {
        let mut out: *const c_void = ptr::null();
        let mut count: c_int = 0;
        // SAFETY: libavcodec writes a pointer to its own static list and its
        // length; `count` bounds the read.
        unsafe {
            let ret = (self.lib.f.avcodec_get_supported_config)(
                self.ptr,
                self.codec.ptr,
                AV_CODEC_CONFIG_PIX_FORMAT,
                0,
                &mut out,
                &mut count,
            );
            if ret < 0 || out.is_null() || count <= 0 {
                return Vec::new();
            }
            std::slice::from_raw_parts(out as *const c_int, count as usize).to_vec()
        }
    }

    /// What an opened encoder settled on.
    pub fn parameters(&self) -> std::result::Result<CodedParams, AvErr> {
        let f = &self.lib.f;
        // SAFETY: the parameters come from their allocator and go back to
        // it; the extradata is copied out before they do.
        unsafe {
            let mut par = (f.avcodec_parameters_alloc)();
            if par.is_null() {
                return Err(AvErr {
                    code: 0,
                    why: "avcodec_parameters_alloc allocated nothing".to_string(),
                });
            }
            let ret = (f.avcodec_parameters_from_context)(par, self.ptr);
            let out = if ret < 0 {
                Err(self.lib.err(ret))
            } else {
                let p = &*par;
                let extradata = if p.extradata.is_null() || p.extradata_size <= 0 {
                    Vec::new()
                } else {
                    std::slice::from_raw_parts(p.extradata, p.extradata_size as usize).to_vec()
                };
                Ok(CodedParams {
                    extradata,
                    video_delay: p.video_delay,
                    codec_tag: p.codec_tag,
                })
            };
            (f.avcodec_parameters_free)(&mut par);
            out
        }
    }

    pub fn max_b_frames(&self) -> c_int {
        // SAFETY: a field of a live context.
        unsafe { (*self.ptr).max_b_frames }
    }

    /// Hands the decoder a packet, or with None, the end of the stream.
    pub fn send_packet(&mut self, packet: Option<&Packet>) -> Status {
        let p = packet.map_or(ptr::null(), |p| p.ptr as *const AVPacket);
        // SAFETY: a live context and a live packet, or null.
        self.lib
            .status(unsafe { (self.lib.f.avcodec_send_packet)(self.ptr, p) })
    }

    pub fn receive_frame(&mut self, frame: &mut Frame) -> Status {
        // SAFETY: a live context and a live frame.
        self.lib
            .status(unsafe { (self.lib.f.avcodec_receive_frame)(self.ptr, frame.ptr) })
    }

    /// Hands the encoder a frame, or with None, the end of the stream.
    pub fn send_frame(&mut self, frame: Option<&Frame>) -> Status {
        let p = frame.map_or(ptr::null(), |f| f.ptr as *const AVFrame);
        // SAFETY: a live context and a live frame, or null.
        self.lib
            .status(unsafe { (self.lib.f.avcodec_send_frame)(self.ptr, p) })
    }

    pub fn receive_packet(&mut self, packet: &mut Packet) -> Status {
        // SAFETY: a live context and a live packet.
        self.lib
            .status(unsafe { (self.lib.f.avcodec_receive_packet)(self.ptr, packet.ptr) })
    }
}

impl Drop for CodecCtx {
    fn drop(&mut self) {
        // SAFETY: allocated by avcodec_alloc_context3; this frees the device
        // reference and extradata with it.
        unsafe { (self.lib.f.avcodec_free_context)(&mut self.ptr) };
    }
}

/// An `AVFrame`, from `av_frame_alloc` to `av_frame_free`.
pub(super) struct Frame {
    ptr: *mut AVFrame,
    lib: &'static Lib,
}

// SAFETY: a frame and the buffers it references may move between threads;
// every access goes through `&self` or `&mut self`.
unsafe impl Send for Frame {}

impl Frame {
    pub fn new(lib: &'static Lib) -> std::result::Result<Frame, AvErr> {
        // SAFETY: null is checked.
        let ptr = unsafe { (lib.f.av_frame_alloc)() };
        if ptr.is_null() {
            return Err(AvErr {
                code: 0,
                why: "av_frame_alloc allocated nothing".to_string(),
            });
        }
        Ok(Frame { ptr, lib })
    }

    fn get(&self) -> &AVFrame {
        // SAFETY: a live frame, owned by this handle.
        unsafe { &*self.ptr }
    }

    fn get_mut(&mut self) -> &mut AVFrame {
        // SAFETY: a live frame, owned by this handle.
        unsafe { &mut *self.ptr }
    }

    /// Releases the buffers and resets every field to its default.
    pub fn unref(&mut self) {
        // SAFETY: a live frame.
        unsafe { (self.lib.f.av_frame_unref)(self.ptr) };
    }

    pub fn width(&self) -> u32 {
        self.get().width.max(0) as u32
    }

    pub fn height(&self) -> u32 {
        self.get().height.max(0) as u32
    }

    pub fn format(&self) -> c_int {
        self.get().format
    }

    pub fn pts(&self) -> Option<i64> {
        Some(self.get().pts).filter(|&p| p != AV_NOPTS_VALUE)
    }

    pub fn best_effort_timestamp(&self) -> Option<i64> {
        Some(self.get().best_effort_timestamp).filter(|&p| p != AV_NOPTS_VALUE)
    }

    pub fn duration(&self) -> Option<i64> {
        Some(self.get().duration).filter(|&d| d > 0)
    }

    /// Whether the picture is in device memory.
    pub fn is_hardware(&self) -> bool {
        !self.get().hw_frames_ctx.is_null()
    }

    /// The format a picture in device memory holds its pixels in (`nv12`,
    /// `p010le`), from its frames context; None for one in host memory.
    pub fn sw_format(&self) -> Option<c_int> {
        let f = self.get();
        if f.hw_frames_ctx.is_null() {
            return None;
        }
        // SAFETY: a hardware frame's hw_frames_ctx is a reference to an
        // AVHWFramesContext, alive while the frame is.
        unsafe {
            let frames = (*f.hw_frames_ctx).data as *const AVHWFramesContext;
            Some((*frames).sw_format)
        }
    }

    /// The D3D12 picture a `d3d12` frame's first data pointer names; None
    /// for any other frame.
    pub fn d3d12(&self) -> Option<*mut AVD3D12VAFrame> {
        let f = self.get();
        let d3d12 = self.lib.pix_fmt("d3d12")?;
        (f.format == d3d12 && !f.data[0].is_null()).then(|| f.data[0] as *mut AVD3D12VAFrame)
    }

    /// The `CVPixelBufferRef` a `videotoolbox_vld` frame's fourth data
    /// pointer names; None for any other frame.
    pub fn videotoolbox(&self) -> Option<*mut c_void> {
        let f = self.get();
        let videotoolbox = self.lib.pix_fmt("videotoolbox_vld")?;
        (f.format == videotoolbox && !f.data[3].is_null()).then(|| f.data[3] as *mut c_void)
    }

    /// Moves the picture `from` holds into this frame, which held nothing,
    /// leaving `from` empty.
    pub fn take(&mut self, from: &mut Frame) -> std::result::Result<(), AvErr> {
        self.unref();
        // SAFETY: two live frames; this one gains references to `from`'s
        // buffers before `from` drops its own.
        let ret = unsafe { (self.lib.f.av_frame_ref)(self.ptr, from.ptr) };
        if ret < 0 {
            return Err(self.lib.err(ret));
        }
        from.unref();
        Ok(())
    }

    pub fn color(&self) -> RawColor {
        let f = self.get();
        RawColor {
            range: f.color_range,
            primaries: f.color_primaries,
            trc: f.color_trc,
            space: f.colorspace,
        }
    }

    pub fn set_color(&mut self, color: RawColor) {
        let f = self.get_mut();
        f.color_range = color.range;
        f.color_primaries = color.primaries;
        f.color_trc = color.trc;
        f.colorspace = color.space;
    }

    /// Sets the geometry and format of a frame that holds no picture yet,
    /// for `alloc` or for libswscale to allocate into.
    pub fn set_video(&mut self, width: u32, height: u32, format: c_int) {
        self.unref();
        let f = self.get_mut();
        f.width = width as c_int;
        f.height = height as c_int;
        f.format = format;
    }

    pub fn set_timing(&mut self, pts: i64, duration: Option<i64>) {
        let f = self.get_mut();
        f.pts = pts;
        f.duration = duration.unwrap_or(0);
    }

    /// Allocates the buffers `set_video` described.
    pub fn alloc(&mut self) -> std::result::Result<(), AvErr> {
        // SAFETY: a live frame with its geometry and format set.
        let ret = unsafe { (self.lib.f.av_frame_get_buffer)(self.ptr, 0) };
        if ret < 0 {
            return Err(self.lib.err(ret));
        }
        Ok(())
    }

    /// Copies a picture in device memory to this frame in host memory, in
    /// the first format the device offers, with its timestamps and colour.
    pub fn download(&mut self, device: &Frame) -> std::result::Result<(), AvErr> {
        self.unref();
        // SAFETY: two live frames; libavutil allocates this one's buffers.
        unsafe {
            let ret = (self.lib.f.av_hwframe_transfer_data)(self.ptr, device.ptr, 0);
            if ret < 0 {
                return Err(self.lib.err(ret));
            }
            let ret = (self.lib.f.av_frame_copy_props)(self.ptr, device.ptr);
            if ret < 0 {
                return Err(self.lib.err(ret));
            }
        }
        Ok(())
    }

    /// Copies `host`, a picture in host memory in the pool's own format,
    /// into this frame, one of a device's pool.
    pub fn upload(&mut self, host: &Frame) -> std::result::Result<(), AvErr> {
        // SAFETY: two live frames; this one's texture was allocated by its
        // pool for the geometry and format `host` has.
        let ret = unsafe { (self.lib.f.av_hwframe_transfer_data)(self.ptr, host.ptr, 0) };
        if ret < 0 {
            return Err(self.lib.err(ret));
        }
        Ok(())
    }

    /// The planes of this frame, when it is in the wire format `pix_fmt`
    /// and its buffers hold every row and byte the wire's layout reads.
    fn wire_planes(&self, pix_fmt: &str) -> Option<Vec<(picture::Plane, *mut u8, isize)>> {
        let f = self.get();
        if self.lib.wire_pix_fmt(pix_fmt)? != f.format || f.width <= 0 || f.height <= 0 {
            return None;
        }
        let planes = picture::planes(pix_fmt, f.width as u32, f.height as u32)?;
        let mut out = Vec::with_capacity(planes.len());
        for (i, plane) in planes.into_iter().enumerate() {
            let base = f.data[i];
            let stride = f.linesize[i] as isize;
            if base.is_null() || stride.unsigned_abs() < plane.row_bytes {
                return None;
            }
            out.push((plane, base, stride));
        }
        Some(out)
    }

    /// Appends the picture to `out` as the wire lays `pix_fmt` out: planes
    /// in order, rows tightly packed. False, with nothing appended, when the
    /// frame is not a `pix_fmt` picture.
    pub fn pack(&self, pix_fmt: &str, out: &mut Vec<u8>) -> bool {
        let Some(planes) = self.wire_planes(pix_fmt) else {
            return false;
        };
        for (plane, base, stride) in planes {
            for row in 0..plane.rows {
                // SAFETY: libav allocated this plane for this format and
                // geometry, so it holds `rows` rows of at least `row_bytes`.
                unsafe {
                    let start = base.offset(stride * row as isize);
                    out.extend_from_slice(std::slice::from_raw_parts(start, plane.row_bytes));
                }
            }
        }
        true
    }

    /// Copies a packed `pix_fmt` picture into this frame's buffers. False,
    /// with nothing written, when the frame is not an allocated `pix_fmt`
    /// picture or `data` is not one packed picture of its size.
    pub fn fill(&mut self, pix_fmt: &str, data: &[u8]) -> bool {
        let Some(planes) = self.wire_planes(pix_fmt) else {
            return false;
        };
        let total: usize = planes.iter().map(|(p, _, _)| p.row_bytes * p.rows).sum();
        if total != data.len() {
            return false;
        }
        let mut at = 0;
        for (plane, base, stride) in planes {
            for row in 0..plane.rows {
                // SAFETY: as in `pack`, and `data` holds `total` bytes.
                unsafe {
                    let start = base.offset(stride * row as isize);
                    ptr::copy_nonoverlapping(data[at..].as_ptr(), start, plane.row_bytes);
                }
                at += plane.row_bytes;
            }
        }
        true
    }
}

impl Drop for Frame {
    fn drop(&mut self) {
        // SAFETY: allocated by av_frame_alloc.
        unsafe { (self.lib.f.av_frame_free)(&mut self.ptr) };
    }
}

/// Drops one reference to a COM object: `IUnknown::Release`, the third
/// entry of every COM interface's table.
///
/// # Safety
/// `object` is null or a COM interface pointer the caller holds a
/// reference to, which this gives up.
pub(super) unsafe fn release_com(object: *mut c_void) {
    if object.is_null() {
        return;
    }
    type Release = unsafe extern "system" fn(*mut c_void) -> u32;
    let table = *(object as *const *const Release);
    (*table.add(2))(object);
}

/// Takes one more reference to a COM object: `IUnknown::AddRef`, the
/// second entry of every COM interface's table.
///
/// # Safety
/// `object` is a live COM interface pointer.
pub(super) unsafe fn add_ref_com(object: *mut c_void) {
    if object.is_null() {
        return;
    }
    type AddRef = unsafe extern "system" fn(*mut c_void) -> u32;
    let table = *(object as *const *const AddRef);
    (*table.add(1))(object);
}

/// An `AVPacket`, from `av_packet_alloc` to `av_packet_free`.
pub(super) struct Packet {
    ptr: *mut AVPacket,
    lib: &'static Lib,
}

// SAFETY: as for `Frame`.
unsafe impl Send for Packet {}

impl Packet {
    pub fn new(lib: &'static Lib) -> std::result::Result<Packet, AvErr> {
        // SAFETY: null is checked.
        let ptr = unsafe { (lib.f.av_packet_alloc)() };
        if ptr.is_null() {
            return Err(AvErr {
                code: 0,
                why: "av_packet_alloc allocated nothing".to_string(),
            });
        }
        Ok(Packet { ptr, lib })
    }

    fn get(&self) -> &AVPacket {
        // SAFETY: a live packet, owned by this handle.
        unsafe { &*self.ptr }
    }

    pub fn unref(&mut self) {
        // SAFETY: a live packet.
        unsafe { (self.lib.f.av_packet_unref)(self.ptr) };
    }

    /// Makes this packet a copy of `data`, in a buffer libavcodec allocated
    /// with the padding it reads past the end, with these timestamps.
    pub fn fill(
        &mut self,
        data: &[u8],
        pts: i64,
        dts: Option<i64>,
        duration: Option<i64>,
        keyframe: bool,
    ) -> std::result::Result<(), AvErr> {
        let size = c_int::try_from(data.len()).map_err(|_| AvErr {
            code: 0,
            why: format!("a packet of {} bytes is too large", data.len()),
        })?;
        self.unref();
        // SAFETY: av_new_packet allocates `size` bytes plus padding, which
        // the copy stays inside.
        unsafe {
            let ret = (self.lib.f.av_new_packet)(self.ptr, size);
            if ret < 0 {
                return Err(self.lib.err(ret));
            }
            let p = &mut *self.ptr;
            ptr::copy_nonoverlapping(data.as_ptr(), p.data, data.len());
            p.pts = pts;
            p.dts = dts.unwrap_or(AV_NOPTS_VALUE);
            p.duration = duration.unwrap_or(0);
            p.flags = if keyframe { AV_PKT_FLAG_KEY } else { 0 };
        }
        Ok(())
    }

    pub fn pts(&self) -> Option<i64> {
        Some(self.get().pts).filter(|&p| p != AV_NOPTS_VALUE)
    }

    pub fn dts(&self) -> Option<i64> {
        Some(self.get().dts).filter(|&p| p != AV_NOPTS_VALUE)
    }

    pub fn duration(&self) -> Option<i64> {
        Some(self.get().duration).filter(|&d| d > 0)
    }

    pub fn is_key(&self) -> bool {
        self.get().flags & AV_PKT_FLAG_KEY != 0
    }

    /// A copy of the packet's bytes.
    pub fn data(&self) -> Vec<u8> {
        let p = self.get();
        if p.data.is_null() || p.size <= 0 {
            return Vec::new();
        }
        // SAFETY: libavcodec's packet holds `size` bytes at `data`.
        unsafe { std::slice::from_raw_parts(p.data, p.size as usize).to_vec() }
    }
}

impl Drop for Packet {
    fn drop(&mut self) {
        // SAFETY: allocated by av_packet_alloc.
        unsafe { (self.lib.f.av_packet_free)(&mut self.ptr) };
    }
}

/// A libswscale context used through `sws_scale_frame`, which reads the
/// geometry, format and colour of both frames and sets itself up to match.
///
/// One side of every conversion is a packed wire picture in a Rust buffer,
/// read or written in place through `view`: a frame whose plane pointers
/// and strides point into that buffer and which owns no buffers itself. So
/// a conversion is the one pass over the picture, with no copy before or
/// after it.
pub(super) struct Scaler {
    ptr: *mut SwsContext,
    view: Frame,
    lib: &'static Lib,
}

// SAFETY: as for `CodecCtx`.
unsafe impl Send for Scaler {}

/// Bytes reserved past the end of a picture libswscale writes into, so a
/// write rounded up past the last row lands in memory the vector owns.
const WRITE_SLACK: usize = 64;

impl Scaler {
    pub fn new(lib: &'static Lib) -> std::result::Result<Scaler, AvErr> {
        let view = Frame::new(lib)?;
        // SAFETY: null is checked.
        let ptr = unsafe { (lib.f.sws_alloc_context)() };
        if ptr.is_null() {
            return Err(AvErr {
                code: 0,
                why: "sws_alloc_context allocated nothing".to_string(),
            });
        }
        Ok(Scaler { ptr, view, lib })
    }

    /// Points `view` at a packed `pix_fmt` picture of `len` bytes at `base`,
    /// planes one after another as the wire lays them out. False when the
    /// format is not the wire's or the bytes are not one picture.
    fn point_view(
        &mut self,
        pix_fmt: &str,
        width: u32,
        height: u32,
        color: RawColor,
        base: *mut u8,
        len: usize,
    ) -> bool {
        let Some(format) = self.lib.wire_pix_fmt(pix_fmt) else {
            return false;
        };
        let Some(planes) = picture::planes(pix_fmt, width, height) else {
            return false;
        };
        let total: usize = planes.iter().map(|p| p.row_bytes * p.rows).sum();
        if width == 0 || height == 0 || total != len {
            return false;
        }
        let (Ok(w), Ok(h)) = (c_int::try_from(width), c_int::try_from(height)) else {
            return false;
        };
        let mut strides = [0 as c_int; AV_NUM_DATA_POINTERS];
        for (i, plane) in planes.iter().enumerate() {
            match c_int::try_from(plane.row_bytes) {
                Ok(stride) => strides[i] = stride,
                Err(_) => return false,
            }
        }
        self.view.unref();
        self.view.set_color(color);
        let f = self.view.get_mut();
        f.width = w;
        f.height = h;
        f.format = format;
        let mut at = 0;
        for (i, plane) in planes.iter().enumerate() {
            f.data[i] = base.wrapping_add(at);
            f.linesize[i] = strides[i];
            at += plane.row_bytes * plane.rows;
        }
        true
    }

    /// Lets go of the buffer `view` pointed into. The view owns no buffer,
    /// so this frees nothing of it.
    fn release_view(&mut self) {
        let f = self.view.get_mut();
        f.data = [ptr::null_mut(); AV_NUM_DATA_POINTERS];
        f.linesize = [0; AV_NUM_DATA_POINTERS];
        self.view.unref();
    }

    /// Converts `src` into a packed `pix_fmt` picture appended to `out`,
    /// written there by libswscale directly. False, with nothing appended,
    /// when `pix_fmt` is not one of the wire's.
    pub fn scale_into_packed(
        &mut self,
        src: &Frame,
        pix_fmt: &str,
        color: RawColor,
        out: &mut Vec<u8>,
    ) -> std::result::Result<bool, AvErr> {
        let (width, height) = (src.width(), src.height());
        let Some(len) = picture::frame_len(pix_fmt, width, height) else {
            return Ok(false);
        };
        let start = out.len();
        out.reserve(len + WRITE_SLACK);
        // The spare capacity after `start`, which is at least `len` bytes.
        let base = out.spare_capacity_mut().as_mut_ptr() as *mut u8;
        if !self.point_view(pix_fmt, width, height, color, base, len) {
            return Ok(false);
        }
        // SAFETY: a live context and two live frames; the view's planes lie
        // inside the vector's spare capacity, which nothing else touches
        // until the call returns.
        let ret = unsafe { (self.lib.f.sws_scale_frame)(self.ptr, self.view.ptr, src.ptr) };
        let in_place = self.view.get().data[0] == base;
        self.release_view();
        if ret < 0 {
            return Err(self.lib.err(ret));
        }
        if !in_place {
            return Err(AvErr {
                code: 0,
                why: "libswscale wrote the picture somewhere other than the buffer it was                       handed"
                    .to_string(),
            });
        }
        // SAFETY: libswscale wrote every row of every plane, and the planes
        // are packed with no gaps, so all `len` bytes after `start` are
        // initialised.
        unsafe { out.set_len(start + len) };
        Ok(true)
    }

    /// Converts a packed `pix_fmt` picture, read in place from `data`, into
    /// `dst`, which is allocated (`Frame::set_video`, then `Frame::alloc`)
    /// in the format, geometry and colour wanted. False, with nothing
    /// written, when `data` is not one packed picture of that size.
    pub fn scale_from_packed(
        &mut self,
        dst: &mut Frame,
        pix_fmt: &str,
        width: u32,
        height: u32,
        color: RawColor,
        data: &[u8],
    ) -> std::result::Result<bool, AvErr> {
        // libswscale only reads its source, so the view's pointers being
        // mutable is the struct's type, not a licence to write.
        let base = data.as_ptr() as *mut u8;
        if !self.point_view(pix_fmt, width, height, color, base, data.len()) {
            return Ok(false);
        }
        // SAFETY: a live context and two live frames; `data` outlives the
        // call and is only read.
        let ret = unsafe { (self.lib.f.sws_scale_frame)(self.ptr, dst.ptr, self.view.ptr) };
        self.release_view();
        if ret < 0 {
            return Err(self.lib.err(ret));
        }
        Ok(true)
    }
}

impl Drop for Scaler {
    fn drop(&mut self) {
        // SAFETY: allocated by sws_alloc_context.
        unsafe { (self.lib.f.sws_free_context)(&mut self.ptr) };
    }
}
