//! The packet source driven straight through `ffrwd_wasm_runtime::runtime`,
//! with no NUT wire and no ffmpeg: the contract the runtime holds a caller
//! to, whatever drove it.
//!
//! `source_replay` is built four times: against the current world, against
//! the vendored `worlds/0.16.0` and `worlds/0.15.0` as `source-replay-0160`
//! and `source-replay-0150`, and against `worlds/0.13.0` as
//! `source-replay-0130`. The 0.16.0 and 0.15.0 builds are here to be HOSTED -
//! `open` is told which tracks to pull there as it is now, so a source built
//! before the 0.17.0 bump still runs - and the 0.13.0 build to be REFUSED,
//! since its `open` takes params alone and a source that cannot be told what
//! to pull is rebuilt rather than adapted.

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use ffrwd_wasm_runtime::runtime::{describe_packet_source, Catalog, PacketSource};

/// Repo root, the parent of `runtime/`.
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("runtime/ has a parent directory")
        .to_path_buf()
}

/// Absolute path to a module's built `.wasm` component, built once per test
/// binary. `modules/` is a separate cargo workspace with its own build lock,
/// so this does not deadlock against the `cargo test` run driving this
/// binary.
fn module_path(name: &str) -> PathBuf {
    static BUILT: OnceLock<()> = OnceLock::new();
    BUILT.get_or_init(|| {
        let output = Command::new("cargo")
            .args([
                "build",
                "--release",
                "--target",
                "wasm32-wasip2",
                "-p",
                "source-replay",
                "-p",
                "source-replay-0160",
                "-p",
                "source-replay-0150",
                "-p",
                "source-replay-0130",
            ])
            .current_dir(repo_root().join("modules"))
            .output()
            .expect("spawn cargo build for modules");
        assert!(
            output.status.success(),
            "building modules failed (status {:?}):\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );
    });
    repo_root()
        .join("modules/target/wasm32-wasip2/release")
        .join(format!("{name}.wasm"))
}

/// `source_replay`'s own packets, mirrored from
/// `modules/source-replay/src/lib.rs`. Pinned from the module's own
/// description, the same way `ffrwd-wasm/tests/packet_source.rs` pins it.
const EXPECTED: &[(i64, Option<i64>, bool)] = &[
    (0, None, true),
    (3, None, false),
    (1, Some(0), false),
    (2, Some(1), false),
    (6, Some(2), false),
    (4, Some(3), false),
    (5, Some(4), false),
];

const EXPECTED_EXTRADATA: &[u8] = &[
    0x00, 0x00, 0x00, 0x01, 0x67, 0x64, 0x00, 0x0a, 0xac, 0xd9, 0x49, 0x7e, 0x5c, 0x04, 0x40, 0x00,
    0x00, 0x03, 0x00, 0x40, 0x00, 0x00, 0x0c, 0x83, 0xc4, 0x89, 0x65, 0x80, 0x00, 0x00, 0x00, 0x01,
    0x68, 0xef, 0x8f, 0x2c, 0x8b,
];

/// `RAW` is the module's own checked-in fixture - see
/// `ffrwd-wasm/tests/packet_source.rs` for why the exact bytes are read back
/// rather than re-typed.
const RAW: &[u8] = include_bytes!("../../modules/source-replay/src/generated/packets.bin");
const EXTRADATA_LEN: usize = 37;
const PACKET_LENS: &[usize] = &[1077, 12, 12, 12, 13, 14, 12];

/// The real coded bytes for packet `index`, decode order, sliced out of
/// `RAW` the same way the module itself slices it.
fn expected_packet_bytes(index: usize) -> &'static [u8] {
    let start = EXTRADATA_LEN + PACKET_LENS[..index].iter().sum::<usize>();
    &RAW[start..start + PACKET_LENS[index]]
}

/// Opens `name` for a run and pulls every packet `next` produces, asserting
/// it answers none exactly once, at the end.
fn drain(mut source: PacketSource) -> Vec<(i64, Option<i64>, bool, Vec<u8>)> {
    let mut got = Vec::new();
    loop {
        let pads = source.next().expect("next");
        let Some(pads) = pads else { break };
        assert_eq!(pads.len(), 1, "source_replay publishes one track");
        for packet in &pads[0].packets {
            got.push((packet.pts, packet.dts, packet.keyframe, packet.data.clone()));
        }
    }
    got
}

/// The catalog either build's `open` reads: one h264 track, 32x24, the same
/// extradata/profile/level `probe` reports at compile time.
fn assert_catalog(catalog: &Catalog) {
    assert_eq!(catalog.tracks.len(), 1);
    assert!(catalog.bounded);
    let track = &catalog.tracks[0];
    assert_eq!(track.stream.codec, "h264");
    assert_eq!(track.stream.time_base.num, 1);
    assert_eq!(track.stream.time_base.den, 25);
    assert_eq!(track.stream.extradata, EXPECTED_EXTRADATA);
    assert_eq!(track.stream.profile, Some(0x64));
    assert_eq!(track.stream.level, Some(0x0a));
    assert_eq!(track.row, 0);
    assert!(track.rendition.name.is_none());
    assert!(track.rendition.bandwidth.is_none());
    assert!(track.rendition.codecs.is_none());
    assert!(track.rendition.language.is_none());
    match track.stream.format {
        ffrwd_wasm_runtime::runtime::CodedFormat::Video { width, height, .. } => {
            assert_eq!((width, height), (32, 24));
        }
        other => panic!("source_replay publishes video, not {}", other.kind()),
    }
}

#[test]
fn a_packet_source_built_against_an_older_world_is_refused_with_the_world_to_rebuild_against() {
    let module = module_path("source_replay_0130");
    let module_str = module.to_str().expect("module path is valid UTF-8");

    let err = describe_packet_source(module_str)
        .expect_err("0.13.0's open is not told which tracks to pull");
    let text = format!("{err:#}");
    assert!(text.contains("ffrwd:av/packet-source@0.13.0"), "{text}");
    assert!(
        text.contains("open is not told which tracks to pull"),
        "the refusal names why 0.13.0 cannot be adapted: {text}"
    );
    assert!(
        text.contains("rebuild it against ffrwd:av@0.17.0"),
        "{text}"
    );

    let Err(err) = PacketSource::open(module_str, "", &[0]) else {
        panic!("the same refusal is expected at open");
    };
    let text = err.to_string();
    assert!(
        text.contains("rebuild it against ffrwd:av@0.17.0"),
        "{text}"
    );
}

#[test]
fn the_current_build_opens_the_one_track_it_was_told_to_pull() {
    let module = module_path("source_replay");
    let module_str = module.to_str().expect("module path is valid UTF-8");

    let described = describe_packet_source(module_str).expect("describing source_replay");
    assert_eq!(described.world, "0.17.0");

    assert_replays(module_str);
}

/// The catalog and the packets, whichever world the module was built
/// against: what "still loads" has to mean for a released source.
fn assert_replays(module_str: &str) {
    let (source, catalog) = PacketSource::open(module_str, "", &[0]).expect("opening the source");
    assert_catalog(&catalog);

    let got = drain(source);
    assert_eq!(got.len(), EXPECTED.len());
    for (index, ((pts, dts, keyframe, data), (want_pts, want_dts, want_keyframe))) in
        got.iter().zip(EXPECTED).enumerate()
    {
        assert_eq!(pts, want_pts, "packet {index} pts");
        assert_eq!(dts, want_dts, "packet {index} dts");
        assert_eq!(keyframe, want_keyframe, "packet {index} keyframe");
        assert_eq!(
            data.as_slice(),
            expected_packet_bytes(index),
            "packet {index} data"
        );
    }
}

#[test]
fn sources_built_against_older_hosted_worlds_still_open_and_pull() {
    // The adapter paths for a packet source, over modules actually built
    // that way: `open` was told which tracks to pull in 0.15.0 and still is,
    // so a source published against either older world is hosted rather
    // than refused. Each answers the world it was BUILT against, not the
    // host's.
    for (name, world) in [
        ("source_replay_0160", "0.16.0"),
        ("source_replay_0150", "0.15.0"),
    ] {
        let module = module_path(name);
        let module_str = module.to_str().expect("module path is valid UTF-8");

        let described = describe_packet_source(module_str).expect("describing the source");
        assert_eq!(described.world, world, "{name}");

        let catalog = PacketSource::probe(module_str, "").expect("probing the source");
        assert_catalog(&catalog);

        assert_replays(module_str);
    }
}

#[test]
fn a_track_the_source_does_not_publish_is_refused_by_the_source_itself() {
    let module = module_path("source_replay");
    let Err(err) = PacketSource::open(module.to_str().unwrap(), "", &[1]) else {
        panic!("source_replay publishes one track, so track 1 is not one of them");
    };
    let text = err.to_string();
    assert!(text.contains("track 1"), "{text}");
    assert!(text.contains("publishes 1 track"), "{text}");
}
