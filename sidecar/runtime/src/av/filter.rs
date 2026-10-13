//! Filtering pictures through libavfilter, on a D3D12 device.
//!
//! A [`Filter`] is one graph in libavfilter's own syntax between a `buffer`
//! source and a `buffersink`. Its pictures enter as NV12 textures of a
//! `d3d12va` frames context the shim makes, on the caller's device
//! ([`Shared`]) or on one of its own, which is what the D3D12 filters
//! (`scale_d3d12`, `deinterlace_d3d12`) read. A picture already on the
//! caller's device is written into one of those textures there
//! ([`Filter::device_picture`], [`Filter::send_picture`]); one in host
//! memory is converted to NV12 and uploaded ([`Filter::send`]). What the
//! sink hands back stays on the device as a [`DeviceFrame`] where the device
//! is the caller's, and is read back otherwise.

use std::os::raw::c_int;
use std::time::Instant;

use super::decode::{DecodedFrame, DeviceFrame, Received, Shared};
use super::error::{Error, Result};
use super::ffi::{
    AvErr, BufferRef, FilterGraph, Frame, Lib, RawColor, Scaler, SourceParams, Status,
};
use super::picture::{self, WIRE_PIX_FMTS};
use super::sys::AVRational;
use super::Color;

/// The device type the filters run on.
const D3D12VA: &str = "d3d12va";

/// One filter graph on a D3D12 device.
///
/// Pictures go in with [`send_picture`](Filter::send_picture) or
/// [`send`](Filter::send); after each, call
/// [`receive_any`](Filter::receive_any) until it answers None. At the end
/// of the stream, [`flush`](Filter::flush), then `receive_any` until None
/// again.
pub struct Filter {
    lib: &'static Lib,
    description: String,
    graph: FilterGraph,
    /// The source's frames context: the textures pictures are written into.
    frames: BufferRef,
    /// Whether the device is the caller's, so pictures leave on it.
    shared: bool,
    width: u32,
    height: u32,
    out_width: u32,
    out_height: u32,
    time_base: AVRational,
    frame_rate: Option<AVRational>,
    names: Vec<String>,
    color: RawColor,
    nv12: c_int,
    /// A host picture converted to NV12, ahead of its upload.
    host: Frame,
    scaler: Option<Scaler>,
    /// What the sink handed back.
    out: Frame,
    /// A picture of the sink read back to host memory.
    down: Frame,
    out_scaler: Option<Scaler>,
    last: Option<(i64, Option<i64>)>,
}

/// Whether a colour name leaves the value unspecified.
fn unset(name: &str) -> bool {
    matches!(name, "" | "unknown" | "unspecified")
}

impl Filter {
    /// Opens `description`, a graph in libavfilter's syntax with one input
    /// and one output (`scale_d3d12=w=1280:h=720`), for `width` x `height`
    /// pictures timestamped in `time_base`, in `color`. The pictures enter as
    /// NV12 on `device`, or on a `d3d12va` device of the filter's own when
    /// None. The graph is configured here, so its output's size is known
    /// before the first picture.
    ///
    /// A directory without libavfilter refuses with the reinstall hint; a
    /// graph libavfilter will not parse or configure refuses naming the call.
    #[allow(clippy::too_many_arguments)]
    pub fn open(
        description: &str,
        width: u32,
        height: u32,
        time_base: (i64, i64),
        frame_rate: Option<(i64, i64)>,
        color: Option<&Color>,
        device: Option<Shared>,
    ) -> Result<Filter> {
        let lib = super::runtime()?;
        lib.filter_lib()?;
        let refuse = |call: &'static str, why: String| Error::Graph {
            graph: description.to_string(),
            call,
            why,
        };
        let Some(device_type) = lib.hw_type(D3D12VA) else {
            return Err(refuse(
                "av_hwdevice_ctx_create",
                format!("this libav cannot create a {D3D12VA} device, which the filters run on"),
            ));
        };
        if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err(refuse(
                "av_hwframe_ctx_init",
                format!(
                    "pictures enter as NV12 textures, which have an even size, and these are \
                     {width}x{height}"
                ),
            ));
        }
        let (Ok(w), Ok(h)) = (c_int::try_from(width), c_int::try_from(height)) else {
            return Err(refuse(
                "av_hwframe_ctx_init",
                format!("{width}x{height} does not fit"),
            ));
        };
        let rational = |(num, den): (i64, i64)| match (c_int::try_from(num), c_int::try_from(den)) {
            (Ok(num), Ok(den)) if num > 0 && den > 0 => Some(AVRational { num, den }),
            _ => None,
        };
        let Some(base) = rational(time_base) else {
            return Err(refuse(
                "av_buffersrc_parameters_set",
                format!("the time base {}/{} is not one", time_base.0, time_base.1),
            ));
        };
        let rate = frame_rate.and_then(rational);
        let (Some(nv12), Some(d3d12)) = (lib.pix_fmt("nv12"), lib.pix_fmt("d3d12")) else {
            return Err(refuse(
                "av_get_pix_fmt",
                "libavutil has no nv12 or d3d12 pixel format".to_string(),
            ));
        };
        let color = match color {
            None => RawColor::UNSPECIFIED,
            Some(c) => {
                let names = [&c.range, &c.primaries, &c.trc, &c.space].map(|n| {
                    if unset(n) {
                        ""
                    } else {
                        n.as_str()
                    }
                });
                lib.color_values(names).map_err(|(key, value)| {
                    refuse(
                        "av_buffersrc_parameters_set",
                        format!("libavutil has no {key} called {value}"),
                    )
                })?
            }
        };

        let shared = device.is_some();
        let raw = match device {
            Some(Shared::D3d12(device)) => device.into_raw(),
            Some(Shared::VideoToolbox) => {
                return Err(refuse(
                    "av_hwframe_ctx_init",
                    "the graph runs on a d3d12va device, and the device is a VideoToolbox one"
                        .to_string(),
                ))
            }
            None => std::ptr::null_mut(),
        };
        let frames = lib
            .d3d12_frames(device_type, raw, nv12, w, h)
            .map_err(|e| refuse("av_hwframe_ctx_init", e.why))?;
        let graph = FilterGraph::new(
            lib,
            description,
            &SourceParams {
                format: d3d12,
                width: w,
                height: h,
                time_base: base,
                frame_rate: rate,
                color,
                frames: Some(&frames),
            },
        )
        .map_err(|(call, e)| refuse(call, e.why))?;

        let (ow, oh, ofmt) = graph.output();
        let leaves = if ofmt == d3d12 {
            match graph.output_sw_format() {
                Some(f) if f == nv12 => None,
                Some(f) => Some(lib.pix_fmt_name(f)),
                None => Some("d3d12 with no frames context".to_string()),
            }
        } else if lib.pix_fmt_is_rgb(ofmt)
            || WIRE_PIX_FMTS.contains(&lib.pix_fmt_name(ofmt).as_str())
            || ofmt == nv12
        {
            None
        } else {
            Some(lib.pix_fmt_name(ofmt))
        };
        if let Some(format) = leaves {
            return Err(refuse(
                "avfilter_graph_config",
                format!(
                    "the graph's pictures leave as {format}; they are taken as NV12 on the \
                     device, or in host memory"
                ),
            ));
        }
        if ow <= 0 || oh <= 0 {
            return Err(refuse(
                "avfilter_graph_config",
                format!("the graph's pictures leave {ow}x{oh}"),
            ));
        }
        let (time_base, frame_rate) = graph.timing();
        let names = graph.filter_names();
        let call = |call: &'static str| move |e: AvErr| refuse(call, e.why);
        Ok(Filter {
            lib,
            description: description.to_string(),
            names,
            graph,
            frames,
            shared,
            width,
            height,
            out_width: ow as u32,
            out_height: oh as u32,
            time_base,
            frame_rate,
            color,
            nv12,
            host: Frame::new(lib).map_err(call("av_frame_alloc"))?,
            scaler: None,
            out: Frame::new(lib).map_err(call("av_frame_alloc"))?,
            down: Frame::new(lib).map_err(call("av_frame_alloc"))?,
            out_scaler: None,
            last: None,
        })
    }

    fn refuse(&self, call: &'static str) -> impl FnOnce(AvErr) -> Error + '_ {
        move |e| Error::Graph {
            graph: self.description.clone(),
            call,
            why: e.why,
        }
    }

    /// The filters the graph runs, by libavfilter's names, in its order.
    pub fn filters(&self) -> &[String] {
        &self.names
    }

    /// Whether the pictures leave on the caller's device.
    pub fn on_shared_device(&self) -> bool {
        self.shared
    }

    /// The size of the pictures the graph hands back.
    pub fn output_size(&self) -> (u32, u32) {
        (self.out_width, self.out_height)
    }

    /// The time base the pictures the graph hands back are timestamped in.
    pub fn time_base(&self) -> (i64, i64) {
        (i64::from(self.time_base.num), i64::from(self.time_base.den))
    }

    /// The frame rate the graph's output declares, where it does.
    pub fn frame_rate(&self) -> Option<(i64, i64)> {
        self.frame_rate
            .map(|r| (i64::from(r.num), i64::from(r.den)))
    }

    /// A picture of the source's pool, for the caller to write on the
    /// device: an NV12 texture of the input's size, whose D3D12 objects
    /// [`DeviceFrame::d3d12`] names. The caller records the write, has the
    /// sync object signalled once it has run, says the value with
    /// [`DeviceFrame::written_at`], and hands the picture to
    /// [`send_picture`](Filter::send_picture).
    pub fn device_picture(&mut self) -> Result<DeviceFrame> {
        let mut frame = Frame::new(self.lib).map_err(self.refuse("av_frame_alloc"))?;
        self.frames
            .get_buffer(&mut frame)
            .map_err(self.refuse("av_hwframe_get_buffer"))?;
        Ok(DeviceFrame {
            pts: 0,
            duration: None,
            width: self.width,
            height: self.height,
            pix_fmt: "nv12",
            frame,
        })
    }

    /// Hands the graph a picture of its source's pool, written on the
    /// device, timestamped in the input's time base.
    pub fn send_picture(
        &mut self,
        mut picture: DeviceFrame,
        pts: i64,
        duration: Option<i64>,
    ) -> Result<()> {
        if picture.frame.d3d12().is_none() {
            return Err(Error::Graph {
                graph: self.description.clone(),
                call: "av_buffersrc_add_frame_flags",
                why: "the picture is not one of a D3D12 pool".to_string(),
            });
        }
        picture.frame.set_color(self.color);
        picture.frame.set_timing(pts, duration);
        let status = self.graph.send(Some(&mut picture.frame));
        self.sent(status)
    }

    /// Hands the graph one picture in host memory, packed as the wire lays
    /// `pix_fmt` out: converted to NV12 and uploaded into a picture of the
    /// source's pool.
    pub fn send(
        &mut self,
        pix_fmt: &str,
        pts: i64,
        duration: Option<i64>,
        data: &[u8],
    ) -> Result<()> {
        let Some(input) = picture::wire_pix_fmt(pix_fmt) else {
            return Err(Error::PixelFormat {
                name: pix_fmt.to_string(),
                why: format!("a filter takes one of {}", WIRE_PIX_FMTS.join(", ")),
            });
        };
        let expected =
            picture::frame_len(input, self.width, self.height).expect("the input is a wire format");
        if data.len() != expected {
            return Err(Error::FrameSize {
                pix_fmt: input.to_string(),
                width: self.width,
                height: self.height,
                expected,
                got: data.len(),
            });
        }
        let started = Instant::now();
        let source_color = if picture::is_rgb(input) {
            self.color.as_rgb()
        } else {
            self.color
        };
        self.host.set_video(self.width, self.height, self.nv12);
        self.host.set_color(self.color);
        self.host
            .alloc()
            .map_err(self.refuse("av_frame_get_buffer"))?;
        if self.scaler.is_none() {
            self.scaler = Some(Scaler::new(self.lib).map_err(self.refuse("sws_alloc_context"))?);
        }
        let Filter {
            description,
            scaler,
            host,
            width,
            height,
            ..
        } = self;
        let written = scaler
            .as_mut()
            .expect("set just above")
            .scale_from_packed(host, input, *width, *height, source_color, data)
            .map_err(|e| Error::Graph {
                graph: description.clone(),
                call: "sws_scale_frame",
                why: e.why,
            })?;
        if !written {
            return Err(Error::Graph {
                graph: self.description.clone(),
                call: "sws_scale_frame",
                why: format!("the picture is not laid out as {input}"),
            });
        }
        let mut picture = self.device_picture()?;
        let uploaded = picture.frame.upload(&self.host);
        self.host.unref();
        uploaded.map_err(self.refuse("av_hwframe_transfer_data"))?;
        // What crossed is the NV12 picture.
        let nv12 = picture::frame_len("yuv420p", self.width, self.height).unwrap_or(expected);
        crate::profile::PROFILE.upload.add(nv12, started);
        self.send_picture(picture, pts, duration)
    }

    /// Ends the stream. The pictures the graph still holds come out of
    /// `receive_any` after this.
    pub fn flush(&mut self) -> Result<()> {
        let status = self.graph.send(None);
        match status {
            Status::Eof => Ok(()),
            other => self.sent(other),
        }
    }

    fn sent(&self, status: Status) -> Result<()> {
        match status {
            Status::Done | Status::Again => Ok(()),
            Status::Eof => Err(Error::Graph {
                graph: self.description.clone(),
                call: "av_buffersrc_add_frame_flags",
                why: "the graph was already flushed".to_string(),
            }),
            Status::Failed(e) => Err(self.refuse("av_buffersrc_add_frame_flags")(e)),
        }
    }

    /// The next picture the graph has ready, or None when it wants another
    /// (or, after `flush`, has none left): on the caller's device where the
    /// filter runs on it and the picture is an NV12 texture there, else read
    /// back and packed as the wire lays `want` out.
    pub fn receive_any(&mut self, want: &str) -> Result<Option<Received>> {
        match self.graph.receive(&mut self.out) {
            Status::Done => {}
            Status::Again | Status::Eof => return Ok(None),
            Status::Failed(e) => return Err(self.refuse("av_buffersink_get_frame")(e)),
        }
        let (width, height) = (self.out.width(), self.out.height());
        let duration = self.out.duration();
        let last = self.last;
        let pts = self
            .out
            .pts()
            .or_else(|| last.map(|(p, d)| p + d.unwrap_or(1)))
            .unwrap_or(0);
        self.last = Some((pts, duration));

        if self.shared && self.out.d3d12().is_some() && self.out.sw_format() == Some(self.nv12) {
            let mut frame = Frame::new(self.lib).map_err(self.refuse("av_frame_alloc"))?;
            frame
                .take(&mut self.out)
                .map_err(self.refuse("av_frame_ref"))?;
            return Ok(Some(Received::Device(DeviceFrame {
                pts,
                duration,
                width,
                height,
                pix_fmt: "nv12",
                frame,
            })));
        }

        let Some(want) = picture::wire_pix_fmt(want) else {
            self.out.unref();
            return Err(Error::PixelFormat {
                name: want.to_string(),
                why: format!("a filter writes one of {}", WIRE_PIX_FMTS.join(", ")),
            });
        };
        let started = Instant::now();
        let on_device = self.out.is_hardware();
        if on_device {
            let downloaded = self.down.download(&self.out);
            self.out.unref();
            downloaded.map_err(self.refuse("av_hwframe_transfer_data"))?;
        }
        let Filter {
            lib,
            description,
            out,
            down,
            out_scaler,
            ..
        } = self;
        let source: &Frame = if on_device { down } else { out };
        let mut data = Vec::new();
        let packed = if source.format() == lib.wire_pix_fmt(want).unwrap_or(-1) {
            source.pack(want, &mut data)
        } else {
            let mut color = source.color();
            if picture::is_rgb(want) {
                color = color.as_rgb();
            }
            if out_scaler.is_none() {
                *out_scaler = Some(Scaler::new(lib).map_err(|e| Error::Graph {
                    graph: description.clone(),
                    call: "sws_alloc_context",
                    why: e.why,
                })?);
            }
            out_scaler
                .as_mut()
                .expect("set just above")
                .scale_into_packed(source, want, color, &mut data)
                .map_err(|e| Error::Graph {
                    graph: description.clone(),
                    call: "sws_scale_frame",
                    why: e.why,
                })?
        };
        self.down.unref();
        self.out.unref();
        if !packed {
            return Err(Error::Graph {
                graph: self.description.clone(),
                call: "sws_scale_frame",
                why: format!("the {width}x{height} picture is not laid out as {want}"),
            });
        }
        if on_device {
            crate::profile::PROFILE.download.add(data.len(), started);
        }
        Ok(Some(Received::Host(DecodedFrame {
            pts,
            duration,
            width,
            height,
            data,
        })))
    }
}
