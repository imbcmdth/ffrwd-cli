//! A module of the node world, driven straight through the runtime: its
//! shape asked for params and bound inputs and checked, an instance opened on
//! bound streams, a tick handed over as a borrowed resource, a frame handed
//! back with `same` resolved to the input's own buffer, and params refused
//! where they would change the shape.

use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, OnceLock};

use ffrwd_wasm_runtime::node::{
    BoundStream, Clock, Node, OutputFormat, Pairing, Payload, PortKind, StreamFormat, Tick,
    TickFrame, TickStream,
};
use ffrwd_wasm_runtime::runtime::{self, StreamInfo, TimeBase, VideoFormat, WitNode};

fn sidecar_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("runtime/ has a parent directory")
        .to_path_buf()
}

/// Builds the probe for wasm32-wasip2, once per test binary.
fn probe() -> &'static str {
    static BUILT: OnceLock<String> = OnceLock::new();
    BUILT.get_or_init(|| {
        let output = Command::new("cargo")
            .args([
                "build",
                "--release",
                "--target",
                "wasm32-wasip2",
                "-p",
                "shape-probe",
            ])
            .current_dir(sidecar_root().join("modules"))
            .output()
            .expect("spawn cargo build for shape-probe");
        assert!(
            output.status.success(),
            "building shape-probe failed (status {:?}):\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );
        sidecar_root()
            .join("modules/target/wasm32-wasip2/release/shape_probe.wasm")
            .to_str()
            .expect("a UTF-8 path")
            .to_string()
    })
}

const TENTHS: TimeBase = TimeBase { num: 1, den: 10 };

fn picture() -> BoundStream {
    BoundStream {
        port: "v".to_string(),
        id: 7,
        info: StreamInfo {
            index: 0,
            kind: "video".to_string(),
            codec: "rawvideo".to_string(),
            duration: None,
            tags: Vec::new(),
        },
        time_base: TENTHS,
        format: StreamFormat::Video(VideoFormat {
            width: 2,
            height: 2,
            pix_fmt: "rgba",
            frame_len: 16,
            color: None,
        }),
        rendition: Default::default(),
        row: None,
        decode_delay: 0,
        latency: None,
    }
}

fn tick(pts: i64, data: &Arc<Vec<u8>>, last: bool) -> Tick {
    Tick {
        pts,
        time_base: TENTHS,
        last,
        streams: vec![TickStream {
            id: 7,
            frames: vec![TickFrame {
                pts,
                duration: Some(1),
                data: Arc::clone(data),
                rows: Vec::new(),
            }],
            ..TickStream::default()
        }],
    }
}

#[test]
fn a_node_says_its_world_and_its_shape_for_what_a_call_binds() {
    let path = probe();
    assert!(runtime::exports_node(path).expect("read the exports"));
    let meta = runtime::describe_node(path).expect("describe");
    assert_eq!(meta.name, "shape_probe");
    assert!(
        meta.pixel_formats.is_empty(),
        "a node's ports say what they accept"
    );

    let bound = vec!["v".to_string()];
    let shape = runtime::node_shape(path, "", &bound).expect("a shape");
    assert_eq!(shape.clock, Clock::Input("v".into()));
    let names: Vec<&str> = shape.outputs.iter().map(|o| o.name.as_str()).collect();
    assert_eq!(names, ["mask", "copy", "spots"]);
    assert!(matches!(
        shape.outputs[0].format,
        Some(OutputFormat::Like(ref like)) if like.pixel_format.as_deref() == Some("gray")
    ));
    let words = shape.input("words").expect("declared");
    assert_eq!(words.kind, PortKind::Data);
    assert!(matches!(words.pairing, Pairing::Interval(ref i) if i.latency == Some(2.0)));

    let unbound = runtime::node_shape(path, "", &[]).expect("a shape");
    assert!(
        unbound.output_index("mask").is_none(),
        "an output like an unbound input is left out"
    );
}

#[test]
fn a_shape_the_wit_refuses_is_refused_naming_the_module_and_the_port() {
    let path = probe();
    let bound = vec!["v".to_string()];
    let refusal = runtime::node_shape(path, r#"{"refuse":"lockstep_on_rate"}"#, &bound)
        .expect_err("refused")
        .to_string();
    assert!(
        refusal.contains("shape_probe input 'v' is lockstep"),
        "{refusal}"
    );
    let refusal = runtime::node_shape(path, r#"{"nope":1}"#, &bound)
        .expect_err("refused")
        .to_string();
    assert!(
        refusal.contains("shape_probe refused the shape"),
        "{refusal}"
    );
}

#[test]
fn a_frame_handed_back_with_same_leaves_on_the_inputs_own_buffer() {
    let path = probe();
    let latched = vec!["copy".to_string(), "spots".to_string()];
    let mut node =
        WitNode::open(path, "", vec![picture()], &["v".to_string()], &latched).expect("open");
    let copy = node.shape().output_index("copy").expect("copy");
    let spots = node.shape().output_index("spots").expect("spots");

    let pixels = Arc::new(vec![9u8; 16]);
    let emitted = node.process(tick(3, &pixels, false)).expect("a tick");
    let mut saw_copy = false;
    let mut saw_spot = false;
    for item in emitted.items {
        match item.payload {
            Payload::Frame(frame) if item.port == copy => {
                assert_eq!(frame.pts, 3);
                assert!(
                    Arc::ptr_eq(&frame.data, &pixels),
                    "the input's buffer itself, not a copy"
                );
                saw_copy = true;
            }
            Payload::Message(message) if item.port == spots => {
                assert_eq!(message.pts, 300_000, "0.3 s in microseconds");
                assert_eq!(message.data, br#"{"start_t":0.3}"#);
                saw_spot = true;
            }
            _ => panic!("nothing else leaves"),
        }
    }
    assert!(saw_copy && saw_spot);

    let last = node
        .process(tick(4, &pixels, true))
        .expect("the final call");
    assert!(!last.items.is_empty());
    let again = node
        .process(tick(5, &pixels, false))
        .expect_err("a call after the last");
    assert!(
        again.to_string().contains("after the final call"),
        "{again}"
    );
}

#[test]
fn params_whose_shape_differs_are_refused_and_the_rest_reach_the_module() {
    let path = probe();
    let mut node = WitNode::open(path, "", vec![picture()], &["v".to_string()], &[]).expect("open");
    node.set_params("{}").expect("the same shape");
    let refusal = node
        .set_params(r#"{"canvas":{"width":4,"height":4}}"#)
        .expect_err("a canvas adds an output")
        .to_string();
    assert!(refusal.contains("shape differs"), "{refusal}");
}

#[test]
fn a_stream_the_shape_cannot_take_is_refused_before_init() {
    let path = probe();
    let mut misplaced = picture();
    misplaced.port = "words".to_string();
    misplaced.id = 8;
    let refusal = WitNode::open(
        path,
        "",
        vec![picture(), misplaced],
        &["v".to_string()],
        &[],
    )
    .err()
    .expect("a picture on a data port")
    .to_string();
    assert!(
        refusal.contains("input 'words' takes data and is bound a video stream"),
        "{refusal}"
    );
}
