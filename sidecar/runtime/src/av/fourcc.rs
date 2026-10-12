//! The four-character tags the sidecar's NUT wire names coded video by, and
//! the codec each names.
//!
//! `ffrwd-nut` knows h264, hevc and av1 by name. Any other codec is carried
//! under its own tag, exactly as ffmpeg's NUT muxer writes it (the riff and
//! bmp tags), so this is where a tag on the wire becomes a codec libavcodec
//! can open and back.

/// Each codec, by ffmpeg's name for it, and the tags it is carried under.
/// The first is the one ffmpeg's NUT muxer writes; the rest are accepted.
const TAGS: &[(&str, &[&[u8; 4]])] = &[
    ("h264", &[b"H264", b"h264", b"avc1", b"AVC1"]),
    ("hevc", &[b"HEVC", b"hevc", b"hev1", b"hvc1"]),
    ("av1", &[b"AV01", b"av01"]),
    ("ffv1", &[b"FFV1"]),
    ("mpeg4", &[b"FMP4", b"mp4v", b"XVID", b"DIVX"]),
    ("vp9", &[b"VP90"]),
    // ffmpeg writes MPEG-1 video as `mpg1`; its mpeg2video decoder reads
    // MPEG-1 as well, so the one name opens both.
    ("mpeg2video", &[b"mpg2", b"MPG2", b"mpg1"]),
    // One tag per profile; the encoder's own choice wins where it makes one
    // (`Encoder::fourcc`). `apch`, HQ, is prores_ks' default.
    (
        "prores",
        &[b"apch", b"apcn", b"apcs", b"apco", b"ap4h", b"ap4x"],
    ),
    ("mjpeg", &[b"MJPG"]),
];

/// The codec a tag on the wire names, by ffmpeg's name for it, or None for
/// a tag this table does not know.
pub fn codec_for_fourcc(tag: &[u8; 4]) -> Option<&'static str> {
    TAGS.iter()
        .find(|(_, tags)| tags.contains(&tag))
        .map(|(name, _)| *name)
}

/// The tag the wire carries a codec under, for ffmpeg's name for it: the
/// one ffmpeg's NUT muxer writes. None for a codec this table does not know.
pub fn fourcc_for_codec(name: &str) -> Option<[u8; 4]> {
    TAGS.iter()
        .find(|(codec, _)| *codec == name)
        .map(|(_, tags)| *tags[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tag_names_its_codec_and_the_first_comes_back() {
        for (codec, tags) in TAGS {
            for tag in *tags {
                assert_eq!(codec_for_fourcc(tag), Some(*codec));
            }
            assert_eq!(fourcc_for_codec(codec), Some(*tags[0]));
        }
    }

    #[test]
    fn an_unknown_tag_names_nothing() {
        assert_eq!(codec_for_fourcc(b"I420"), None);
        assert_eq!(fourcc_for_codec("rawvideo"), None);
    }
}
