//! ADTS: the header every AAC frame out of HLS or mpegts carries on itself,
//! and the AudioSpecificConfig it is built from when the stream around it
//! has none of its own.
//!
//! `-c copy` off an ADTS source leaves NUT's stream header extradata empty,
//! because ADTS repeats the same handful of fields on every frame instead of
//! once in the container. This module reads that copy back out - the way
//! ffmpeg's own `aac_adtstoasc` bitstream filter does - and removes it from
//! each frame's bytes, ahead of NUT, while the header is still attached to
//! the frame that carries it rather than side data a NUT muxer drops.

use anyhow::{anyhow, bail, Result};

/// The ADTS syncword: twelve set bits at the front of every header.
const SYNCWORD: u16 = 0x0FFF;

/// One ADTS header, parsed far enough to strip it from a packet and to
/// derive an AudioSpecificConfig.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    /// Header length on the wire: 7 bytes, or 9 when a CRC follows it
    /// (`protection_absent` clear).
    pub header_len: usize,
    /// This ADTS frame's length, header and payload together, as the header
    /// states it.
    pub frame_length: usize,
    /// `number_of_raw_data_blocks_in_frame`: 0 for the ordinary case of one
    /// AAC frame per ADTS header.
    pub raw_data_blocks: u8,
    audio_object_type: u8,
    sampling_frequency_index: u8,
    channel_configuration: u8,
}

impl Header {
    /// Reads the ADTS header at the front of `data`, or None when it does
    /// not start with the syncword.
    pub fn parse(data: &[u8]) -> Option<Header> {
        if data.len() < 7 {
            return None;
        }
        let sync = (u16::from(data[0]) << 4) | u16::from(data[1] >> 4);
        if sync != SYNCWORD {
            return None;
        }
        let protection_absent = data[1] & 0b1;
        let profile = (data[2] >> 6) & 0b11;
        let sampling_frequency_index = (data[2] >> 2) & 0b1111;
        let channel_configuration = ((data[2] & 0b1) << 2) | (data[3] >> 6);
        let frame_length = (usize::from(data[3] & 0b11) << 11)
            | (usize::from(data[4]) << 3)
            | (usize::from(data[5]) >> 5);
        let raw_data_blocks = data[6] & 0b11;
        let header_len = if protection_absent == 1 { 7 } else { 9 };
        if data.len() < header_len {
            return None;
        }
        Some(Header {
            header_len,
            frame_length,
            raw_data_blocks,
            audio_object_type: profile + 1,
            sampling_frequency_index,
            channel_configuration,
        })
    }

    /// The 2-byte AudioSpecificConfig this header codes: `audioObjectType`
    /// (5 bits), `samplingFrequencyIndex` (4 bits), `channelConfiguration`
    /// (4 bits, zero-extended from ADTS's 3), and the 3 GASpecificConfig
    /// bits AAC-LC leaves clear.
    pub fn audio_specific_config(&self) -> [u8; 2] {
        let a = self.audio_object_type;
        let f = self.sampling_frequency_index;
        let c = self.channel_configuration;
        [(a << 3) | (f >> 1), ((f & 0b1) << 7) | (c << 3)]
    }
}

/// Strips the ADTS header off the front of `data` in place, and returns it
/// parsed - the caller priming a stream also wants the AudioSpecificConfig
/// it carries.
///
/// Refuses a packet whose header does not account for the whole packet
/// rather than guessing at a split: `raw_data_blocks` nonzero packs more
/// than one AAC frame behind a single header, and a `frame_length` short of
/// `data.len()` means several ADTS-framed frames are concatenated in the one
/// packet NUT coded a single PTS for. Splitting either would mean
/// fabricating timestamps the wire never coded, so both are refused by name
/// instead.
pub fn strip(data: &mut Vec<u8>) -> Result<Header> {
    let header = Header::parse(data).ok_or_else(|| {
        anyhow!("ADTS packet does not start with the syncword its stream opened with")
    })?;
    if header.raw_data_blocks != 0 {
        bail!(
            "ADTS header packs {} AAC frames behind one header; splitting a packed frame is not \
             implemented",
            header.raw_data_blocks + 1
        );
    }
    if header.frame_length != data.len() {
        bail!(
            "ADTS packet is {} bytes but its header claims one frame of {}; a packet carrying \
             more than one ADTS frame is refused rather than split, since NUT coded it a single PTS",
            data.len(),
            header.frame_length
        );
    }
    data.drain(0..header.header_len);
    Ok(header)
}

/// Test support: one hand-crafted ADTS header for `profile` (0 Main, 1 LC, 2
/// SSR, 3 LTP), `sfi` (the sampling frequency index) and `channels`, ahead of
/// `payload_len` opaque bytes (`0xEE`, so a stripped payload is easy to spot
/// in an assertion). `crc` selects the 9-byte header that carries one.
/// Shared with `demux`'s tests, which build a whole NUT wire around it.
#[cfg(test)]
pub(crate) fn test_packet(
    profile: u8,
    sfi: u8,
    channels: u8,
    crc: bool,
    payload_len: usize,
) -> Vec<u8> {
    let header_len = if crc { 9 } else { 7 };
    let frame_length = header_len + payload_len;
    let mut packet = vec![0u8; header_len];
    packet[0] = 0xFF;
    packet[1] = 0xF0 | u8::from(!crc);
    packet[2] = (profile << 6) | (sfi << 2) | (channels >> 2);
    packet[3] = ((channels & 0b11) << 6) | ((frame_length >> 11) as u8 & 0b11);
    packet[4] = (frame_length >> 3) as u8;
    packet[5] = ((frame_length as u8 & 0b111) << 5) | 0b0001_1111;
    packet[6] = 0b1111_1100; // raw_data_blocks = 0
    packet.extend(std::iter::repeat_n(0xEE, payload_len));
    packet
}

#[cfg(test)]
mod tests {
    use super::test_packet as adts_packet;
    use super::*;

    #[test]
    fn lc_48000_stereo_derives_the_known_asc() {
        let mut packet = adts_packet(1, 3, 2, false, 4);
        let header = strip(&mut packet).expect("a well-formed ADTS header strips");
        assert_eq!(header.audio_specific_config(), [0x11, 0x90]);
        assert_eq!(packet, vec![0xEE; 4], "the payload is what remains");
    }

    #[test]
    fn lc_44100_mono_derives_the_known_asc() {
        let mut packet = adts_packet(1, 4, 1, false, 4);
        let header = strip(&mut packet).expect("a well-formed ADTS header strips");
        assert_eq!(header.audio_specific_config(), [0x12, 0x08]);
    }

    #[test]
    fn the_crc_variant_carries_a_nine_byte_header() {
        let mut packet = adts_packet(1, 3, 2, true, 4);
        let header = strip(&mut packet).expect("a checksummed header still parses");
        assert_eq!(header.header_len, 9);
        assert_eq!(packet, vec![0xEE; 4]);
    }

    #[test]
    fn bytes_without_the_syncword_are_not_adts() {
        assert!(Header::parse(&[0, 0, 0, 0, 0, 0, 0]).is_none());
    }

    #[test]
    fn a_short_buffer_is_not_adts() {
        assert!(Header::parse(&[0xFF, 0xF1, 0, 0, 0, 0]).is_none());
    }

    #[test]
    fn a_packet_carrying_two_frames_is_refused_not_split() {
        let mut packet = adts_packet(1, 3, 2, false, 4);
        packet.extend(adts_packet(1, 3, 2, false, 4));
        let err = strip(&mut packet).unwrap_err().to_string();
        assert!(err.contains("more than one ADTS frame"), "{err}");
    }

    #[test]
    fn several_raw_data_blocks_behind_one_header_is_refused() {
        let mut packet = adts_packet(1, 3, 2, false, 4);
        packet[6] |= 0b01; // raw_data_blocks = 1, so 2 frames share this header
        let err = strip(&mut packet).unwrap_err().to_string();
        assert!(err.contains("packs 2 AAC frames"), "{err}");
    }
}
