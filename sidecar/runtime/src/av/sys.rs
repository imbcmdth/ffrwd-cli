//! The C side of libav, as the pinned majors declare it: the structs this
//! shim reads or writes, the constants it passes, and the signatures of the
//! functions it resolves.
//!
//! Every struct here is transcribed field by field from the headers of
//! FFmpeg 9.0 (libavcodec 63, libavutil 61, libswscale 10). A field out of
//! place is silent memory corruption rather than an error, so the offsets
//! are checked twice: at compile time against the offsets the headers give
//! on a 64-bit target, below, and at load time against what the loaded
//! libraries report for themselves (`ffi::check_layout`).
//!
//! Structs that libav allocates and this shim only reaches through a pointer
//! are transcribed only as far as the last field it touches; nothing here is
//! ever allocated on the Rust side.

#![allow(non_camel_case_types, dead_code)]

use std::os::raw::{c_char, c_float, c_int, c_uint, c_void};

/// `AVRational`: a fraction, as two ints.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AVRational {
    pub num: c_int,
    pub den: c_int,
}

/// `AVBufferRef` (libavutil/buffer.h), whole.
#[repr(C)]
pub struct AVBufferRef {
    pub buffer: *mut c_void,
    pub data: *mut u8,
    pub size: usize,
}

/// `AVHWDeviceContext` (libavutil/hwcontext.h), whole: what a device
/// reference's `data` points at.
#[repr(C)]
pub struct AVHWDeviceContext {
    pub av_class: *const AVClass,
    pub type_: c_int,
    pub hwctx: *mut c_void,
    pub free: *mut c_void,
    pub user_opaque: *mut c_void,
}

/// `AVHWFramesContext` (libavutil/hwcontext.h), through `height`: what a
/// hardware frame's `hw_frames_ctx` points at.
#[repr(C)]
pub struct AVHWFramesContext {
    pub av_class: *const AVClass,
    pub device_ref: *mut AVBufferRef,
    pub device_ctx: *mut AVHWDeviceContext,
    pub hwctx: *mut c_void,
    pub free: *mut c_void,
    pub user_opaque: *mut c_void,
    pub pool: *mut c_void,
    pub initial_pool_size: c_int,
    pub format: c_int,
    pub sw_format: c_int,
    pub width: c_int,
    pub height: c_int,
}

/// `AVD3D12VADeviceContext` (libavutil/hwcontext_d3d12va.h), through
/// `device`, the one field set before `av_hwdevice_ctx_init`.
#[repr(C)]
pub struct AVD3D12VADeviceContext {
    /// `ID3D12Device *`; the context releases it when freed.
    pub device: *mut c_void,
}

/// `AVD3D12VASyncContext` (libavutil/hwcontext_d3d12va.h), whole.
#[repr(C)]
pub struct AVD3D12VASyncContext {
    /// The D3D12 sync object.
    pub sync: *mut c_void,
    /// `HANDLE`.
    pub event: *mut c_void,
    /// The value the sync object reaches once the picture is written.
    pub sync_value: u64,
}

/// `AVD3D12VAFrame` (libavutil/hwcontext_d3d12va.h), whole: what a d3d12
/// frame's `data[0]` points at.
#[repr(C)]
pub struct AVD3D12VAFrame {
    /// `ID3D12Resource *`.
    pub texture: *mut c_void,
    pub subresource_index: c_int,
    pub sync_ctx: AVD3D12VASyncContext,
    pub flags: c_int,
}

/// `AVDictionary`, only ever handled by pointer.
#[repr(C)]
pub struct AVDictionary {
    _private: [u8; 0],
}

/// `SwsContext`, only ever handled by pointer; its options are set through
/// the frames it is handed instead.
#[repr(C)]
pub struct SwsContext {
    _private: [u8; 0],
}

/// `AVClass`, only ever handled by pointer.
#[repr(C)]
pub struct AVClass {
    _private: [u8; 0],
}

/// `AVChannelLayout` (libavutil/channel_layout.h). Present because `AVFrame`,
/// `AVCodecParameters` and `AVCodecContext` embed one ahead of fields read
/// here.
#[repr(C)]
pub struct AVChannelLayout {
    pub order: c_int,
    pub nb_channels: c_int,
    /// `union { uint64_t mask; AVChannelCustom *map; }`
    pub u: u64,
    pub opaque: *mut c_void,
}

/// `AVPacket` (libavcodec/packet.h), whole.
#[repr(C)]
pub struct AVPacket {
    pub buf: *mut AVBufferRef,
    pub pts: i64,
    pub dts: i64,
    pub data: *mut u8,
    pub size: c_int,
    pub stream_index: c_int,
    pub flags: c_int,
    pub side_data: *mut c_void,
    pub side_data_elems: c_int,
    pub duration: i64,
    pub pos: i64,
    pub opaque: *mut c_void,
    pub opaque_ref: *mut AVBufferRef,
    pub time_base: AVRational,
}

/// The number of data pointers an `AVFrame` carries.
pub const AV_NUM_DATA_POINTERS: usize = 8;

/// `AVFrame` (libavutil/frame.h), whole.
#[repr(C)]
pub struct AVFrame {
    pub data: [*mut u8; AV_NUM_DATA_POINTERS],
    pub linesize: [c_int; AV_NUM_DATA_POINTERS],
    pub extended_data: *mut *mut u8,
    pub width: c_int,
    pub height: c_int,
    pub nb_samples: c_int,
    pub format: c_int,
    pub pict_type: c_int,
    pub sample_aspect_ratio: AVRational,
    pub pts: i64,
    pub pkt_dts: i64,
    pub time_base: AVRational,
    pub quality: c_int,
    pub opaque: *mut c_void,
    pub repeat_pict: c_int,
    pub sample_rate: c_int,
    pub buf: [*mut AVBufferRef; AV_NUM_DATA_POINTERS],
    pub extended_buf: *mut *mut AVBufferRef,
    pub nb_extended_buf: c_int,
    pub side_data: *mut *mut c_void,
    pub nb_side_data: c_int,
    pub flags: c_int,
    pub color_range: c_int,
    pub color_primaries: c_int,
    pub color_trc: c_int,
    pub colorspace: c_int,
    pub chroma_location: c_int,
    pub best_effort_timestamp: i64,
    pub metadata: *mut AVDictionary,
    pub decode_error_flags: c_int,
    pub hw_frames_ctx: *mut AVBufferRef,
    pub opaque_ref: *mut AVBufferRef,
    pub crop_top: usize,
    pub crop_bottom: usize,
    pub crop_left: usize,
    pub crop_right: usize,
    pub private_ref: *mut c_void,
    pub ch_layout: AVChannelLayout,
    pub duration: i64,
    pub alpha_mode: c_int,
}

/// `AVCodecParameters` (libavcodec/codec_par.h), whole.
#[repr(C)]
pub struct AVCodecParameters {
    pub codec_type: c_int,
    pub codec_id: c_int,
    pub codec_tag: u32,
    pub extradata: *mut u8,
    pub extradata_size: c_int,
    pub coded_side_data: *mut c_void,
    pub nb_coded_side_data: c_int,
    pub format: c_int,
    pub bit_rate: i64,
    pub bits_per_coded_sample: c_int,
    pub bits_per_raw_sample: c_int,
    pub profile: c_int,
    pub level: c_int,
    pub width: c_int,
    pub height: c_int,
    pub sample_aspect_ratio: AVRational,
    pub framerate: AVRational,
    pub field_order: c_int,
    pub color_range: c_int,
    pub color_primaries: c_int,
    pub color_trc: c_int,
    pub color_space: c_int,
    pub chroma_location: c_int,
    pub video_delay: c_int,
    pub ch_layout: AVChannelLayout,
    pub sample_rate: c_int,
    pub block_align: c_int,
    pub frame_size: c_int,
    pub initial_padding: c_int,
    pub trailing_padding: c_int,
    pub seek_preroll: c_int,
    pub alpha_mode: c_int,
}

/// `AVCodecContext` (libavcodec/avcodec.h), from its first field through
/// `extra_hw_frames`. The struct goes on past that; nothing past it is
/// touched, and it is only ever reached through the pointer
/// `avcodec_alloc_context3` returns.
#[repr(C)]
pub struct AVCodecContext {
    pub av_class: *const AVClass,
    pub log_level_offset: c_int,
    pub codec_type: c_int,
    pub codec: *const AVCodec,
    pub codec_id: c_int,
    pub codec_tag: c_uint,
    pub priv_data: *mut c_void,
    pub internal: *mut c_void,
    pub opaque: *mut c_void,
    pub bit_rate: i64,
    pub flags: c_int,
    pub flags2: c_int,
    pub extradata: *mut u8,
    pub extradata_size: c_int,
    pub time_base: AVRational,
    pub pkt_timebase: AVRational,
    pub framerate: AVRational,
    pub delay: c_int,
    pub width: c_int,
    pub height: c_int,
    pub coded_width: c_int,
    pub coded_height: c_int,
    pub sample_aspect_ratio: AVRational,
    pub pix_fmt: c_int,
    pub sw_pix_fmt: c_int,
    pub color_primaries: c_int,
    pub color_trc: c_int,
    pub colorspace: c_int,
    pub color_range: c_int,
    pub chroma_sample_location: c_int,
    pub field_order: c_int,
    pub refs: c_int,
    pub has_b_frames: c_int,
    pub slice_flags: c_int,
    pub draw_horiz_band: *mut c_void,
    pub get_format: *mut c_void,
    pub max_b_frames: c_int,
    pub b_quant_factor: c_float,
    pub b_quant_offset: c_float,
    pub i_quant_factor: c_float,
    pub i_quant_offset: c_float,
    pub lumi_masking: c_float,
    pub temporal_cplx_masking: c_float,
    pub spatial_cplx_masking: c_float,
    pub p_masking: c_float,
    pub dark_masking: c_float,
    pub nsse_weight: c_int,
    pub me_cmp: c_int,
    pub me_sub_cmp: c_int,
    pub mb_cmp: c_int,
    pub ildct_cmp: c_int,
    pub dia_size: c_int,
    pub last_predictor_count: c_int,
    pub me_pre_cmp: c_int,
    pub pre_dia_size: c_int,
    pub me_subpel_quality: c_int,
    pub me_range: c_int,
    pub mb_decision: c_int,
    pub intra_matrix: *mut u16,
    pub inter_matrix: *mut u16,
    pub chroma_intra_matrix: *mut u16,
    /// Deprecated, and still present in major 63: `FF_API_INTRA_DC_PRECISION`
    /// is `LIBAVCODEC_VERSION_MAJOR < 64`.
    pub intra_dc_precision: c_int,
    pub mb_lmin: c_int,
    pub mb_lmax: c_int,
    pub bidir_refine: c_int,
    pub keyint_min: c_int,
    pub gop_size: c_int,
    pub mv0_threshold: c_int,
    pub slices: c_int,
    pub sample_rate: c_int,
    pub sample_fmt: c_int,
    pub ch_layout: AVChannelLayout,
    pub frame_size: c_int,
    pub block_align: c_int,
    pub cutoff: c_int,
    pub audio_service_type: c_int,
    pub request_sample_fmt: c_int,
    pub initial_padding: c_int,
    pub trailing_padding: c_int,
    pub seek_preroll: c_int,
    pub get_buffer2: *mut c_void,
    pub bit_rate_tolerance: c_int,
    pub global_quality: c_int,
    pub compression_level: c_int,
    pub qcompress: c_float,
    pub qblur: c_float,
    pub qmin: c_int,
    pub qmax: c_int,
    pub max_qdiff: c_int,
    pub rc_buffer_size: c_int,
    pub rc_override_count: c_int,
    pub rc_override: *mut c_void,
    pub rc_max_rate: i64,
    pub rc_min_rate: i64,
    pub rc_max_available_vbv_use: c_float,
    pub rc_min_vbv_overflow_use: c_float,
    pub rc_initial_buffer_occupancy: c_int,
    pub trellis: c_int,
    pub stats_out: *mut c_char,
    pub stats_in: *mut c_char,
    pub workaround_bugs: c_int,
    pub strict_std_compliance: c_int,
    pub error_concealment: c_int,
    pub debug: c_int,
    pub err_recognition: c_int,
    pub hwaccel: *const c_void,
    pub hwaccel_context: *mut c_void,
    pub hw_frames_ctx: *mut AVBufferRef,
    pub hw_device_ctx: *mut AVBufferRef,
    pub hwaccel_flags: c_int,
    pub extra_hw_frames: c_int,
}

/// `AVCodec` (libavcodec/codec.h), through `capabilities`: all this shim
/// reads of one.
#[repr(C)]
pub struct AVCodec {
    pub name: *const c_char,
    pub long_name: *const c_char,
    pub type_: c_int,
    pub id: c_int,
    pub capabilities: c_int,
}

/// `AVCodecDescriptor` (libavcodec/codec_desc.h), through `name`.
#[repr(C)]
pub struct AVCodecDescriptor {
    pub id: c_int,
    pub type_: c_int,
    pub name: *const c_char,
}

/// `AVCodecHWConfig` (libavcodec/codec.h), whole.
#[repr(C)]
pub struct AVCodecHWConfig {
    pub pix_fmt: c_int,
    pub methods: c_int,
    pub device_type: c_int,
}

/// `AVPixFmtDescriptor` (libavutil/pixdesc.h), through `flags`.
#[repr(C)]
pub struct AVPixFmtDescriptor {
    pub name: *const c_char,
    pub nb_components: u8,
    pub log2_chroma_w: u8,
    pub log2_chroma_h: u8,
    pub flags: u64,
}

/// `AV_PIX_FMT_FLAG_RGB`.
pub const AV_PIX_FMT_FLAG_RGB: u64 = 1 << 5;

/// `AVOption` (libavutil/opt.h), through `type`: read only to learn where
/// the loaded build keeps a field, for the layout check.
#[repr(C)]
pub struct AVOption {
    pub name: *const c_char,
    pub help: *const c_char,
    pub offset: c_int,
    pub type_: c_int,
}

/// `AVMEDIA_TYPE_VIDEO`.
pub const AVMEDIA_TYPE_VIDEO: c_int = 0;

/// `AV_PIX_FMT_NONE`.
pub const AV_PIX_FMT_NONE: c_int = -1;

/// `AV_HWDEVICE_TYPE_NONE`.
pub const AV_HWDEVICE_TYPE_NONE: c_int = 0;

/// `AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX`: the codec takes a device
/// through `hw_device_ctx`, which is the one way this shim offers one.
pub const AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX: c_int = 0x01;

/// `AV_CODEC_FLAG_GLOBAL_HEADER`.
pub const AV_CODEC_FLAG_GLOBAL_HEADER: c_int = 1 << 22;

/// `AV_CODEC_CAP_EXPERIMENTAL`.
pub const AV_CODEC_CAP_EXPERIMENTAL: c_int = 1 << 9;

/// `AV_CODEC_CAP_HARDWARE`: a decoder that is a hardware API wrapper
/// (`h264_cuvid`, `h264_qsv`) rather than libavcodec's own decoder with a
/// hwaccel.
pub const AV_CODEC_CAP_HARDWARE: c_int = 1 << 18;

/// `AV_PKT_FLAG_KEY`.
pub const AV_PKT_FLAG_KEY: c_int = 0x0001;

/// `AV_INPUT_BUFFER_PADDING_SIZE`: the zeroed bytes libav may read past the
/// end of extradata and packet data.
pub const AV_INPUT_BUFFER_PADDING_SIZE: usize = 64;

/// `AV_NOPTS_VALUE`.
pub const AV_NOPTS_VALUE: i64 = i64::MIN;

/// `AV_OPT_SEARCH_CHILDREN`.
pub const AV_OPT_SEARCH_CHILDREN: c_int = 1;

/// `AV_LOG_ERROR`.
pub const AV_LOG_ERROR: c_int = 16;

/// `AV_CODEC_CONFIG_PIX_FORMAT`, the first of `enum AVCodecConfig`.
pub const AV_CODEC_CONFIG_PIX_FORMAT: c_int = 0;

/// The colour enums' unspecified values, which `av_frame_alloc` writes.
pub const AVCOL_RANGE_UNSPECIFIED: c_int = 0;
pub const AVCOL_PRI_UNSPECIFIED: c_int = 2;
pub const AVCOL_TRC_UNSPECIFIED: c_int = 2;
pub const AVCOL_SPC_UNSPECIFIED: c_int = 2;

/// `AVCOL_SPC_RGB` and `AVCOL_RANGE_JPEG`: what an rgba picture is tagged.
pub const AVCOL_SPC_RGB: c_int = 0;
pub const AVCOL_RANGE_JPEG: c_int = 2;

/// `FFERRTAG`, the form libav's own error codes take.
const fn fferrtag(a: u8, b: u8, c: u8, d: u8) -> c_int {
    -((a as i32) | ((b as i32) << 8) | ((c as i32) << 16) | ((d as i32) << 24))
}

/// `AVERROR_EOF`.
pub const AVERROR_EOF: c_int = fferrtag(b'E', b'O', b'F', b' ');

/// `AVERROR_OPTION_NOT_FOUND`.
pub const AVERROR_OPTION_NOT_FOUND: c_int = fferrtag(0xF8, b'O', b'P', b'T');

/// `AVERROR(EAGAIN)`. `AVERROR` flips the sign of the C library's errno,
/// and EAGAIN is 35 on Apple's, and 11 on Linux's and on the mingw runtime
/// FFmpeg's Windows builds link.
#[cfg(target_os = "macos")]
pub const AVERROR_EAGAIN: c_int = -35;
#[cfg(not(target_os = "macos"))]
pub const AVERROR_EAGAIN: c_int = -11;

/// The signatures resolved from each library, as the headers declare them.
pub type FnVersion = unsafe extern "C" fn() -> c_uint;
pub type FnStr = unsafe extern "C" fn() -> *const c_char;
pub type FnFrameAlloc = unsafe extern "C" fn() -> *mut AVFrame;
pub type FnFrameFree = unsafe extern "C" fn(*mut *mut AVFrame);
pub type FnFrameUnref = unsafe extern "C" fn(*mut AVFrame);
pub type FnFrameGetBuffer = unsafe extern "C" fn(*mut AVFrame, c_int) -> c_int;
pub type FnFrameCopyProps = unsafe extern "C" fn(*mut AVFrame, *const AVFrame) -> c_int;
pub type FnStrerror = unsafe extern "C" fn(c_int, *mut c_char, usize) -> c_int;
/// A name to an enum value: `av_get_pix_fmt`, `av_color_*_from_name`,
/// `av_hwdevice_find_type_by_name`.
pub type FnFromName = unsafe extern "C" fn(*const c_char) -> c_int;
/// An enum value to its name: `av_get_pix_fmt_name`, `av_color_*_name`,
/// `av_hwdevice_get_type_name`, `avcodec_get_name`.
pub type FnEnumName = unsafe extern "C" fn(c_int) -> *const c_char;
pub type FnHwIterate = unsafe extern "C" fn(c_int) -> c_int;
pub type FnHwDeviceCreate = unsafe extern "C" fn(
    *mut *mut AVBufferRef,
    c_int,
    *const c_char,
    *mut AVDictionary,
    c_int,
) -> c_int;
pub type FnHwDeviceAlloc = unsafe extern "C" fn(c_int) -> *mut AVBufferRef;
pub type FnHwDeviceInit = unsafe extern "C" fn(*mut AVBufferRef) -> c_int;
/// `av_hwframe_ctx_alloc`: a frames context on the device a reference names.
pub type FnHwFramesAlloc = unsafe extern "C" fn(*mut AVBufferRef) -> *mut AVBufferRef;
pub type FnHwFramesInit = unsafe extern "C" fn(*mut AVBufferRef) -> c_int;
pub type FnHwFrameGetBuffer = unsafe extern "C" fn(*mut AVBufferRef, *mut AVFrame, c_int) -> c_int;
pub type FnBufferUnref = unsafe extern "C" fn(*mut *mut AVBufferRef);
pub type FnFrameRef = unsafe extern "C" fn(*mut AVFrame, *const AVFrame) -> c_int;
pub type FnHwTransfer = unsafe extern "C" fn(*mut AVFrame, *const AVFrame, c_int) -> c_int;
pub type FnOptSet = unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char, c_int) -> c_int;
pub type FnOptFind = unsafe extern "C" fn(
    *mut c_void,
    *const c_char,
    *const c_char,
    c_int,
    c_int,
) -> *const AVOption;
pub type FnMallocz = unsafe extern "C" fn(usize) -> *mut c_void;
pub type FnLogSetLevel = unsafe extern "C" fn(c_int);
pub type FnCodecIterate = unsafe extern "C" fn(*mut *mut c_void) -> *const AVCodec;
pub type FnCodecIs = unsafe extern "C" fn(*const AVCodec) -> c_int;
pub type FnFindByName = unsafe extern "C" fn(*const c_char) -> *const AVCodec;
pub type FnFindById = unsafe extern "C" fn(c_int) -> *const AVCodec;
pub type FnDescriptorByName = unsafe extern "C" fn(*const c_char) -> *const AVCodecDescriptor;
pub type FnContextAlloc = unsafe extern "C" fn(*const AVCodec) -> *mut AVCodecContext;
pub type FnContextFree = unsafe extern "C" fn(*mut *mut AVCodecContext);
pub type FnParamsAlloc = unsafe extern "C" fn() -> *mut AVCodecParameters;
pub type FnParamsFree = unsafe extern "C" fn(*mut *mut AVCodecParameters);
pub type FnParamsToContext =
    unsafe extern "C" fn(*mut AVCodecContext, *const AVCodecParameters) -> c_int;
pub type FnParamsFromContext =
    unsafe extern "C" fn(*mut AVCodecParameters, *const AVCodecContext) -> c_int;
pub type FnOpen2 =
    unsafe extern "C" fn(*mut AVCodecContext, *const AVCodec, *mut *mut AVDictionary) -> c_int;
pub type FnSendPacket = unsafe extern "C" fn(*mut AVCodecContext, *const AVPacket) -> c_int;
pub type FnReceiveFrame = unsafe extern "C" fn(*mut AVCodecContext, *mut AVFrame) -> c_int;
pub type FnSendFrame = unsafe extern "C" fn(*mut AVCodecContext, *const AVFrame) -> c_int;
pub type FnReceivePacket = unsafe extern "C" fn(*mut AVCodecContext, *mut AVPacket) -> c_int;
pub type FnPacketAlloc = unsafe extern "C" fn() -> *mut AVPacket;
pub type FnPacketFree = unsafe extern "C" fn(*mut *mut AVPacket);
pub type FnPacketUnref = unsafe extern "C" fn(*mut AVPacket);
pub type FnNewPacket = unsafe extern "C" fn(*mut AVPacket, c_int) -> c_int;
pub type FnGetHwConfig = unsafe extern "C" fn(*const AVCodec, c_int) -> *const AVCodecHWConfig;
pub type FnSupportedConfig = unsafe extern "C" fn(
    *const AVCodecContext,
    *const AVCodec,
    c_int,
    c_uint,
    *mut *const c_void,
    *mut c_int,
) -> c_int;
pub type FnPixDesc = unsafe extern "C" fn(c_int) -> *const AVPixFmtDescriptor;
pub type FnSwsAlloc = unsafe extern "C" fn() -> *mut SwsContext;
pub type FnSwsFree = unsafe extern "C" fn(*mut *mut SwsContext);
pub type FnSwsScaleFrame =
    unsafe extern "C" fn(*mut SwsContext, *mut AVFrame, *const AVFrame) -> c_int;

/// The offsets the 9.0 headers give on a 64-bit target, worked out by hand
/// from the declarations, so a transcription slip fails the build rather
/// than a run.
#[cfg(target_pointer_width = "64")]
mod offsets {
    use super::*;
    use std::mem::{offset_of, size_of};

    const _: () = assert!(size_of::<AVRational>() == 8);
    const _: () = assert!(size_of::<AVChannelLayout>() == 24);

    const _: () = assert!(offset_of!(AVPacket, pts) == 8);
    const _: () = assert!(offset_of!(AVPacket, data) == 24);
    const _: () = assert!(offset_of!(AVPacket, size) == 32);
    const _: () = assert!(offset_of!(AVPacket, flags) == 40);
    const _: () = assert!(offset_of!(AVPacket, side_data) == 48);
    const _: () = assert!(offset_of!(AVPacket, duration) == 64);
    const _: () = assert!(offset_of!(AVPacket, pos) == 72);
    const _: () = assert!(offset_of!(AVPacket, time_base) == 96);
    const _: () = assert!(size_of::<AVPacket>() == 104);

    const _: () = assert!(offset_of!(AVFrame, linesize) == 64);
    const _: () = assert!(offset_of!(AVFrame, extended_data) == 96);
    const _: () = assert!(offset_of!(AVFrame, width) == 104);
    const _: () = assert!(offset_of!(AVFrame, format) == 116);
    const _: () = assert!(offset_of!(AVFrame, sample_aspect_ratio) == 124);
    const _: () = assert!(offset_of!(AVFrame, pts) == 136);
    const _: () = assert!(offset_of!(AVFrame, time_base) == 152);
    const _: () = assert!(offset_of!(AVFrame, opaque) == 168);
    const _: () = assert!(offset_of!(AVFrame, buf) == 184);
    const _: () = assert!(offset_of!(AVFrame, side_data) == 264);
    const _: () = assert!(offset_of!(AVFrame, flags) == 276);
    const _: () = assert!(offset_of!(AVFrame, color_range) == 280);
    const _: () = assert!(offset_of!(AVFrame, chroma_location) == 296);
    const _: () = assert!(offset_of!(AVFrame, best_effort_timestamp) == 304);
    const _: () = assert!(offset_of!(AVFrame, hw_frames_ctx) == 328);
    const _: () = assert!(offset_of!(AVFrame, crop_top) == 344);
    const _: () = assert!(offset_of!(AVFrame, ch_layout) == 384);
    const _: () = assert!(offset_of!(AVFrame, duration) == 408);
    const _: () = assert!(offset_of!(AVFrame, alpha_mode) == 416);
    const _: () = assert!(size_of::<AVFrame>() == 424);

    const _: () = assert!(offset_of!(AVCodecParameters, extradata) == 16);
    const _: () = assert!(offset_of!(AVCodecParameters, extradata_size) == 24);
    const _: () = assert!(offset_of!(AVCodecParameters, bit_rate) == 48);
    const _: () = assert!(offset_of!(AVCodecParameters, width) == 72);
    const _: () = assert!(offset_of!(AVCodecParameters, color_range) == 100);
    const _: () = assert!(offset_of!(AVCodecParameters, video_delay) == 120);
    const _: () = assert!(offset_of!(AVCodecParameters, ch_layout) == 128);

    const _: () = assert!(offset_of!(AVCodecContext, codec_id) == 24);
    const _: () = assert!(offset_of!(AVCodecContext, bit_rate) == 56);
    const _: () = assert!(offset_of!(AVCodecContext, flags) == 64);
    const _: () = assert!(offset_of!(AVCodecContext, extradata) == 72);
    const _: () = assert!(offset_of!(AVCodecContext, time_base) == 84);
    const _: () = assert!(offset_of!(AVCodecContext, framerate) == 100);
    const _: () = assert!(offset_of!(AVCodecContext, width) == 112);
    const _: () = assert!(offset_of!(AVCodecContext, pix_fmt) == 136);
    const _: () = assert!(offset_of!(AVCodecContext, color_primaries) == 144);
    const _: () = assert!(offset_of!(AVCodecContext, color_range) == 156);
    const _: () = assert!(offset_of!(AVCodecContext, has_b_frames) == 172);
    const _: () = assert!(offset_of!(AVCodecContext, draw_horiz_band) == 184);
    const _: () = assert!(offset_of!(AVCodecContext, max_b_frames) == 200);
    const _: () = assert!(offset_of!(AVCodecContext, intra_matrix) == 288);
    const _: () = assert!(offset_of!(AVCodecContext, intra_dc_precision) == 312);
    const _: () = assert!(offset_of!(AVCodecContext, gop_size) == 332);
    const _: () = assert!(offset_of!(AVCodecContext, ch_layout) == 352);
    const _: () = assert!(offset_of!(AVCodecContext, get_buffer2) == 408);
    const _: () = assert!(offset_of!(AVCodecContext, rc_max_rate) == 464);
    const _: () = assert!(offset_of!(AVCodecContext, stats_out) == 496);
    const _: () = assert!(offset_of!(AVCodecContext, hwaccel) == 536);
    const _: () = assert!(offset_of!(AVCodecContext, hw_device_ctx) == 560);
    const _: () = assert!(offset_of!(AVCodecContext, hwaccel_flags) == 568);
    const _: () = assert!(offset_of!(AVCodecContext, extra_hw_frames) == 572);

    const _: () = assert!(size_of::<AVBufferRef>() == 24);
    const _: () = assert!(offset_of!(AVHWDeviceContext, hwctx) == 16);
    const _: () = assert!(offset_of!(AVHWFramesContext, initial_pool_size) == 56);
    const _: () = assert!(offset_of!(AVHWFramesContext, sw_format) == 64);
    const _: () = assert!(offset_of!(AVHWFramesContext, height) == 72);
    const _: () = assert!(offset_of!(AVD3D12VAFrame, subresource_index) == 8);
    const _: () = assert!(offset_of!(AVD3D12VAFrame, sync_ctx) == 16);
    const _: () = assert!(offset_of!(AVD3D12VAFrame, flags) == 40);
    const _: () = assert!(size_of::<AVD3D12VAFrame>() == 48);

    const _: () = assert!(offset_of!(AVOption, offset) == 16);
    const _: () = assert!(offset_of!(AVCodec, type_) == 16);
    const _: () = assert!(offset_of!(AVCodec, id) == 20);
    const _: () = assert!(offset_of!(AVCodecDescriptor, name) == 8);
    const _: () = assert!(offset_of!(AVPixFmtDescriptor, flags) == 16);
}
