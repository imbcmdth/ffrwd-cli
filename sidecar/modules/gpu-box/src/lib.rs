//! A node that draws one solid box in the corner of each frame, either on
//! the GPU or in host memory, so the two paths can be checked against each
//! other byte for byte.
//!
//! `memory` picks the path. On `gpu`, both ports are held on the device:
//! each input frame reaches the module as a texture, the box is copied and
//! drawn with the queue, and the result is handed back as a texture. On
//! `host`, the frame's bytes are fetched, drawn on, and emitted as bytes.
//! Either way the output is the input with a red box of `size` pixels at
//! its top-left corner.
//!
//! The input is rgba or yuv420p. A texture handed back is rgba, so a
//! yuv420p input held on the GPU is written in host memory: `out=host`
//! keeps the output there, and the module reads its planes back from the
//! device itself, draws the box on the bytes, and emits those. In yuv420p
//! the box is BT.601 red, (81, 90, 240), over the chroma samples it
//! covers.

wit_bindgen::generate!({
    path: ["../../wit", "../../wit-gpu", "wit"],
    // Fully qualified: several packages are in scope, and each has worlds.
    world: "ffrwd:gpu-box/gpu-box",
    generate_all,
});

use std::cell::RefCell;

use exports::ffrwd::av::node::{Emission, Emitted, Guest, Payload};
use exports::ffrwd::gpu::ports::Guest as PortsGuest;
use ffrwd::av::node_tick::Tick;
use ffrwd::av::node_types::{
    Accepts, Binding, BoundStream, Clock, InputPort, LikeInput, NodeShape, OutputFormat,
    OutputPort, Pairing, PortKind, RowsUse,
};
use ffrwd::av::types::{Meta, RawFrame, Wants};
use ffrwd::gpu::frames::{emit, planes};
use ffrwd::gpu::types::{FrameMemory, PortMemory};
use serde::Deserialize;
use wasi::webgpu::webgpu::{
    get_gpu, GpuBufferDescriptor, GpuBufferUsage, GpuDevice, GpuExtent3D, GpuMapMode, GpuOrigin3D,
    GpuTexelCopyBufferInfo, GpuTexelCopyBufferLayout, GpuTexelCopyTextureInfo,
    GpuTextureDescriptor, GpuTextureFormat, GpuTextureUsage,
};

const SCHEMA: &str = r#"{"type":"object","properties":{"memory":{"type":"string","enum":["gpu","host"]},"out":{"type":"string","enum":["gpu","host"]},"size":{"type":"integer","minimum":1}},"additionalProperties":false}"#;

/// The box's colour, rgba.
const COLOUR: [u8; 4] = [255, 0, 0, 255];

/// The box's colour in yuv420p: Y, U, V.
const YUV: [u8; 3] = [81, 90, 240];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Params {
    #[serde(default = "gpu")]
    memory: String,
    /// Where the output is written; `memory`'s unless said.
    #[serde(default)]
    out: Option<String>,
    #[serde(default = "eight")]
    size: u32,
}

impl Params {
    fn out_on_gpu(&self) -> bool {
        self.out.as_deref().unwrap_or(&self.memory) == "gpu"
    }
}

fn gpu() -> String {
    "gpu".to_string()
}

fn eight() -> u32 {
    8
}

impl Default for Params {
    fn default() -> Self {
        Params {
            memory: gpu(),
            out: None,
            size: eight(),
        }
    }
}

fn parse(text: &str) -> Result<Params, String> {
    let params: Params = if text.trim().is_empty() {
        Params::default()
    } else {
        serde_json::from_str(text).map_err(|e| format!("params: {e}"))?
    };
    if params.memory != "gpu" && params.memory != "host" {
        return Err(format!("memory is gpu or host, not '{}'", params.memory));
    }
    if params.memory == "host" && params.out_on_gpu() {
        return Err("out=gpu takes memory=gpu".to_string());
    }
    if params.size == 0 {
        return Err("size is above zero".to_string());
    }
    Ok(params)
}

struct State {
    v: u32,
    width: u32,
    height: u32,
    size: u32,
    yuv: bool,
    on_gpu: bool,
    out_on_gpu: bool,
    device: Option<GpuDevice>,
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

/// The box as pixels: `side` by `side`, tightly packed rgba.
fn box_pixels(side: u32) -> Vec<u8> {
    COLOUR.repeat((side * side) as usize)
}

/// The box's side on a frame of `width` by `height`: `size`, clipped.
fn side(size: u32, width: u32, height: u32) -> u32 {
    size.min(width).min(height)
}

/// Draws the box onto `pixels`, a `width` by `height` rgba frame.
fn draw(pixels: &mut [u8], width: u32, height: u32, size: u32) {
    let side = side(size, width, height) as usize;
    for y in 0..side {
        for x in 0..side {
            let at = (y * width as usize + x) * 4;
            pixels[at..at + 4].copy_from_slice(&COLOUR);
        }
    }
}

/// The planes of a `width` by `height` yuv420p frame: width and height.
fn yuv_planes(width: u32, height: u32) -> [(u32, u32); 3] {
    let chroma = (width.div_ceil(2), height.div_ceil(2));
    [(width, height), chroma, chroma]
}

/// Draws the box onto `pixels`, a packed `width` by `height` yuv420p frame.
fn draw_yuv(pixels: &mut [u8], width: u32, height: u32, size: u32) {
    let side = side(size, width, height);
    let mut start = 0usize;
    for (plane, (w, h)) in yuv_planes(width, height).into_iter().enumerate() {
        let covered = if plane == 0 { side } else { side.div_ceil(2) } as usize;
        for y in 0..covered {
            let row = start + y * w as usize;
            pixels[row..row + covered].fill(YUV[plane]);
        }
        start += (w * h) as usize;
    }
}

/// Reads the yuv420p planes of frame `index` back from the device, packed
/// as the wire lays them out.
fn read_back(state: &State, index: u32) -> Result<Vec<u8>, String> {
    let device = state
        .device
        .as_ref()
        .ok_or("the device is opened at init")?;
    let handed = planes(state.v, index)?;
    let sizes = yuv_planes(state.width, state.height);
    if handed.len() != sizes.len() {
        return Err(format!(
            "a yuv420p frame has 3 planes, and {} came",
            handed.len()
        ));
    }
    let mut out = Vec::new();
    for (texture, (w, h)) in handed.iter().zip(sizes) {
        let padded = w.div_ceil(256) * 256;
        let buffer = device.create_buffer(&GpuBufferDescriptor {
            size: u64::from(padded * h),
            usage: GpuBufferUsage::copy_dst() | GpuBufferUsage::map_read(),
            mapped_at_creation: None,
            label: None,
        });
        let encoder = device.create_command_encoder(None);
        encoder.copy_texture_to_buffer(
            &GpuTexelCopyTextureInfo {
                texture,
                mip_level: None,
                origin: None,
                aspect: None,
            },
            &GpuTexelCopyBufferInfo {
                buffer: &buffer,
                offset: None,
                bytes_per_row: Some(padded),
                rows_per_image: Some(h),
            },
            GpuExtent3D {
                width: w,
                height: Some(h),
                depth_or_array_layers: None,
            },
        );
        device.queue().submit(&[&encoder.finish(None)]);
        buffer
            .map_async(GpuMapMode::read(), None, None)
            .map_err(|_| "mapping a plane's readback".to_string())?;
        let mapped = buffer
            .get_mapped_range_get_with_copy(None, None)
            .map_err(|_| "reading a plane's readback".to_string())?;
        for row in 0..h as usize {
            let at = row * padded as usize;
            out.extend_from_slice(&mapped[at..at + w as usize]);
        }
        buffer
            .unmap()
            .map_err(|_| "unmapping a plane's readback".to_string())?;
    }
    Ok(out)
}

/// Copies the frame's texture to a new one, draws the box on that with the
/// queue, and hands it to the host as the frame on `out`.
fn draw_on_gpu(state: &State, pts: i64, duration: Option<i64>, index: u32) -> Result<(), String> {
    let device = state
        .device
        .as_ref()
        .ok_or("the device is opened at init")?;
    let handed = planes(state.v, index)?;
    let source = handed.first().ok_or("an rgba frame has one plane")?;
    let (width, height) = (state.width, state.height);
    let extent = |w: u32, h: u32| GpuExtent3D {
        width: w,
        height: Some(h),
        depth_or_array_layers: None,
    };
    let out = device.create_texture(&GpuTextureDescriptor {
        size: extent(width, height),
        mip_level_count: None,
        sample_count: None,
        dimension: None,
        format: GpuTextureFormat::Rgba8unorm,
        usage: GpuTextureUsage::copy_src()
            | GpuTextureUsage::copy_dst()
            | GpuTextureUsage::texture_binding(),
        view_formats: None,
        label: None,
    });
    let whole = |texture| GpuTexelCopyTextureInfo {
        texture,
        mip_level: None,
        origin: None,
        aspect: None,
    };
    let encoder = device.create_command_encoder(None);
    encoder.copy_texture_to_texture(&whole(source), &whole(&out), extent(width, height));
    let commands = encoder.finish(None);
    let queue = device.queue();
    queue.submit(&[&commands]);
    let side = side(state.size, width, height);
    queue.write_texture_with_copy(
        &GpuTexelCopyTextureInfo {
            texture: &out,
            mip_level: None,
            origin: Some(GpuOrigin3D {
                x: Some(0),
                y: Some(0),
                z: None,
            }),
            aspect: None,
        },
        &box_pixels(side),
        GpuTexelCopyBufferLayout {
            offset: None,
            bytes_per_row: Some(side * 4),
            rows_per_image: None,
        },
        extent(side, side),
    );
    emit("out", pts, duration, out)
}

struct Node;

impl Guest for Node {
    fn describe() -> Meta {
        Meta {
            name: "gpu_box".to_string(),
            version: "0.1.0".to_string(),
            params_schema: SCHEMA.to_string(),
            rows_schema: String::new(),
            pixel_formats: Vec::new(),
            sample_formats: Vec::new(),
            sample_rates: Vec::new(),
            channel_counts: Vec::new(),
            rows_language: Vec::new(),
        }
    }

    fn shape(params: String, _bound: Vec<Binding>) -> Result<NodeShape, String> {
        parse(&params)?;
        let v = InputPort {
            name: "v".to_string(),
            kind: PortKind::Video,
            required: true,
            many: false,
            pairing: Pairing::Lockstep,
            rows: RowsUse::Ignore,
            window: 1,
            stride: 1,
            accepts: Accepts {
                pixel_formats: vec!["rgba".to_string(), "yuv420p".to_string()],
                sample_formats: Vec::new(),
                sample_rates: Vec::new(),
                channel_counts: Vec::new(),
                codecs: Vec::new(),
                wants: Wants::All,
                like: None,
            },
            schema: None,
        };
        let out = OutputPort {
            name: "out".to_string(),
            kind: PortKind::Video,
            format: Some(OutputFormat::Like(LikeInput {
                port: "v".to_string(),
                pixel_format: None,
                sample_format: None,
            })),
            time_base: None,
            latency: 0.0,
            schema: None,
            row: None,
        };
        Ok(NodeShape {
            inputs: vec![v],
            outputs: vec![out],
            clock: Clock::Input("v".to_string()),
            pure: true,
            one_to_one: true,
            bounded: true,
            relation: Vec::new(),
        })
    }

    fn init(bound: Vec<BoundStream>, _latched: Vec<String>, params: String) -> Result<(), String> {
        let params = parse(&params)?;
        let v = bound.iter().find(|b| b.port == "v").ok_or("v is bound")?;
        let Some(OutputFormat::Video(format)) = &v.format else {
            return Err("v is video".to_string());
        };
        let on_gpu = params.memory == "gpu";
        let out_on_gpu = params.out_on_gpu();
        let yuv = format.pix_fmt == "yuv420p";
        if yuv && out_on_gpu {
            return Err(
                "a texture handed back is rgba; a yuv420p input takes out=host".to_string(),
            );
        }
        let device = if on_gpu {
            let adapter = get_gpu()
                .request_adapter(None)
                .ok_or("no GPU adapter; memory=host runs without one")?;
            Some(
                adapter
                    .request_device(None)
                    .map_err(|_| "the adapter gave no device".to_string())?,
            )
        } else {
            None
        };
        STATE.with(|s| {
            *s.borrow_mut() = Some(State {
                v: v.id,
                width: format.width,
                height: format.height,
                size: params.size,
                yuv,
                on_gpu,
                out_on_gpu,
                device,
            })
        });
        Ok(())
    }

    fn set_params(_params: String) -> Result<(), String> {
        Ok(())
    }

    fn process(tick: &Tick) -> Result<Emitted, String> {
        STATE.with(|s| {
            let state = s.borrow();
            let state = state.as_ref().ok_or("process before init")?;
            let mut items = Vec::new();
            for frame in tick.frames(state.v) {
                if state.out_on_gpu {
                    draw_on_gpu(state, frame.pts, frame.duration, frame.index)?;
                    continue;
                }
                let mut pixels = if state.on_gpu {
                    read_back(state, frame.index)?
                } else {
                    tick.fetch(state.v, frame.index)
                };
                if state.yuv {
                    draw_yuv(&mut pixels, state.width, state.height, state.size);
                } else {
                    draw(&mut pixels, state.width, state.height, state.size);
                }
                items.push(Emission {
                    port: "out".to_string(),
                    payload: Payload::Frame(RawFrame {
                        pts: frame.pts,
                        duration: frame.duration,
                        data: pixels,
                    }),
                });
            }
            Ok(Emitted {
                items,
                rows: Vec::new(),
                finished: false,
            })
        })
    }
}

impl PortsGuest for Node {
    fn memory(params: String) -> Vec<PortMemory> {
        let Ok(params) = parse(&params) else {
            return Vec::new();
        };
        let held = [("v", params.memory == "gpu"), ("out", params.out_on_gpu())];
        held.into_iter()
            .filter(|(_, on_gpu)| *on_gpu)
            .map(|(port, _)| PortMemory {
                port: port.to_string(),
                memory: FrameMemory::Gpu,
            })
            .collect()
    }
}

export!(Node);
