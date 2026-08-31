//! Cutting an arriving stream into the windows one module's `process`
//! receives.
//!
//! For video a window is frames, and a frame arrives whole: the cutter only
//! counts them. For audio a window is SAMPLES, and what arrives is whatever
//! packets the producer chose - 1024 samples is ffmpeg's own shape for pcm,
//! and nothing says a module's window is a multiple of it. So the samples are
//! buffered and re-cut: a window leaves here as one contiguous piece at the
//! timestamp of its first sample, however many packets it was spread over, and
//! a module never sees a packet boundary.
//!
//! Rows arrive on a packet, at that packet's timestamp. They are held until a
//! window whose span covers that timestamp is cut, and ride that window.
//! Whatever is still held when the stream ends rides the final call.

use std::sync::Arc;

use anyhow::{bail, Result};
use ffrwd_wasm_runtime::runtime::{AudioFormat, Format, Frame, Media, Shape, TimeBase};

/// Slices a stream into the windows one module is driven with.
pub enum Windows {
    Frames(FrameWindows),
    Samples(SampleWindows),
}

impl Windows {
    pub fn new(shape: Shape, format: &Format) -> Windows {
        match format.media {
            Media::Video(_) => Windows::Frames(FrameWindows {
                window: shape.window as usize,
                stride: shape.stride as usize,
                buffered: Vec::new(),
            }),
            Media::Audio(audio) => Windows::Samples(SampleWindows::new(shape, audio, format)),
        }
    }

    /// Every window this arrival completed, in the order they were cut. One
    /// packet may complete several windows, or none.
    pub fn push(&mut self, frame: Frame, module: &str) -> Result<Vec<Vec<Frame>>> {
        match self {
            Windows::Frames(w) => Ok(w.push(frame).into_iter().collect()),
            Windows::Samples(w) => w.push(frame, module),
        }
    }

    /// Whatever the last stride left over, for the final call. Empty when the
    /// windows came out even and no rows are still waiting.
    pub fn tail(&mut self) -> Vec<Frame> {
        match self {
            Windows::Frames(w) => w.tail(),
            Windows::Samples(w) => w.tail(),
        }
    }
}

/// Frames, counted.
pub struct FrameWindows {
    window: usize,
    stride: usize,
    buffered: Vec<Frame>,
}

impl FrameWindows {
    /// The window this frame completed, if it completed one.
    fn push(&mut self, frame: Frame) -> Option<Vec<Frame>> {
        self.buffered.push(frame);
        if self.buffered.len() < self.window {
            return None;
        }
        let window = self.buffered.clone();
        self.buffered.drain(..self.stride);
        Some(window)
    }

    fn tail(&mut self) -> Vec<Frame> {
        std::mem::take(&mut self.buffered)
    }
}

/// Samples, re-cut out of whatever packets arrived.
pub struct SampleWindows {
    /// Bytes one sample occupies, every channel included.
    sample_len: usize,
    /// Samples one call receives, and how many it consumes.
    window: usize,
    stride: usize,
    /// The samples not yet consumed, contiguous, the first of them at `pts`.
    buffered: Vec<u8>,
    pts: Option<i64>,
    /// Rows that arrived, each with the timestamp it arrived on, waiting for a
    /// window whose span covers it.
    pending: Vec<(i64, Vec<String>)>,
    /// Ticks per sample, as a ratio in lowest terms.
    ticks_num: i64,
    ticks_den: i64,
}

impl SampleWindows {
    fn new(shape: Shape, audio: AudioFormat, format: &Format) -> SampleWindows {
        let (ticks_num, ticks_den) = ticks_per_sample(audio, format.time_base);
        SampleWindows {
            sample_len: audio.sample_len(),
            window: shape.window as usize,
            stride: shape.stride as usize,
            buffered: Vec::new(),
            pts: None,
            pending: Vec::new(),
            ticks_num,
            ticks_den,
        }
    }

    /// Buffers one arriving packet and cuts every window it completed.
    fn push(&mut self, frame: Frame, module: &str) -> Result<Vec<Vec<Frame>>> {
        if !frame.data.len().is_multiple_of(self.sample_len) {
            bail!(
                "{module}: a packet at pts {} carries {} bytes, which is not a whole number of \
                 {}-byte samples",
                frame.pts,
                frame.data.len(),
                self.sample_len
            );
        }
        if !frame.rows.is_empty() {
            self.pending.push((frame.pts, frame.rows));
        }
        if self.buffered.is_empty() {
            self.pts = Some(frame.pts);
        }
        self.buffered.extend_from_slice(&frame.data);

        let mut cut = Vec::new();
        while self.buffered.len() >= self.window * self.sample_len {
            cut.push(vec![self.cut(module)?]);
        }
        Ok(cut)
    }

    /// One window off the front, advancing by the stride.
    fn cut(&mut self, module: &str) -> Result<Frame> {
        let pts = self.pts.expect("a window is only cut once samples arrived");
        let data = Arc::new(self.buffered[..self.window * self.sample_len].to_vec());
        let ends = pts + self.ticks(self.window, module)?;
        let rows = self.claim(pts, ends);
        self.buffered.drain(..self.stride * self.sample_len);
        self.pts = Some(pts + self.ticks(self.stride, module)?);
        Ok(Frame { pts, data, rows })
    }

    /// Whatever the strides left over, plus any rows no window claimed. Rows
    /// with nothing left to ride still travel, on a window of no samples.
    fn tail(&mut self) -> Vec<Frame> {
        let rows: Vec<String> = std::mem::take(&mut self.pending)
            .into_iter()
            .flat_map(|(_, rows)| rows)
            .collect();
        let data = std::mem::take(&mut self.buffered);
        if data.is_empty() && rows.is_empty() {
            return Vec::new();
        }
        vec![Frame {
            pts: self.pts.unwrap_or(0),
            data: Arc::new(data),
            rows,
        }]
    }

    /// The rows that arrived inside `[from, to)`, taken out of the queue.
    fn claim(&mut self, from: i64, to: i64) -> Vec<String> {
        let mut claimed = Vec::new();
        self.pending.retain_mut(|(pts, rows)| {
            if *pts < to && *pts >= from {
                claimed.append(rows);
                return false;
            }
            true
        });
        claimed
    }

    /// The span `samples` cover, in ticks.
    fn ticks(&self, samples: usize, module: &str) -> Result<i64> {
        let total = samples as i64 * self.ticks_num;
        if total % self.ticks_den != 0 {
            bail!(
                "{module}: {samples} samples do not land on a whole tick of this stream's time \
                 base, so a window has no timestamp to carry"
            );
        }
        Ok(total / self.ticks_den)
    }
}

/// Ticks of the stream's time base one sample covers, in lowest terms. The
/// natural base for audio is 1/sample-rate, where the answer is 1/1.
fn ticks_per_sample(audio: AudioFormat, time_base: TimeBase) -> (i64, i64) {
    let num = time_base.den as i64;
    let den = i64::from(audio.sample_rate) * time_base.num as i64;
    let divisor = gcd(num, den).max(1);
    (num / divisor, den / divisor)
}

fn gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffrwd_wasm_runtime::runtime::{Media, VideoFormat};

    /// 48 kHz mono f32: one sample is four bytes and one tick.
    fn audio_format() -> Format {
        Format {
            media: Media::Audio(AudioFormat {
                sample_rate: 48_000,
                channels: 1,
                sample_fmt: "f32",
            }),
            time_base: TimeBase {
                num: 1,
                den: 48_000,
            },
        }
    }

    fn video_format() -> Format {
        Format {
            media: Media::Video(VideoFormat {
                width: 2,
                height: 2,
                pix_fmt: "rgba",
                frame_len: 16,
            }),
            time_base: TimeBase { num: 1, den: 25 },
        }
    }

    fn shape(window: u32, stride: u32) -> Shape {
        Shape {
            window,
            stride,
            pure: true,
            one_to_one: true,
        }
    }

    /// One packet of `samples` samples at `pts`, each sample its own index so
    /// a window's contents say where it was cut from.
    fn packet(pts: i64, samples: usize, rows: Vec<String>) -> Frame {
        let mut data = Vec::with_capacity(samples * 4);
        for offset in 0..samples {
            data.extend_from_slice(&((pts as usize + offset) as f32).to_le_bytes());
        }
        Frame {
            pts,
            data: data.into(),
            rows,
        }
    }

    /// The sample values a window carries.
    fn values(frame: &Frame) -> Vec<f32> {
        let (whole, _) = frame.data.as_chunks::<4>();
        whole.iter().copied().map(f32::from_le_bytes).collect()
    }

    /// Every window `packets` produce, the final call included.
    fn drive(shape: Shape, packets: &[Frame]) -> Vec<Frame> {
        let mut windows = Windows::new(shape, &audio_format());
        let mut out = Vec::new();
        for packet in packets {
            for window in windows
                .push(packet.clone(), "cutter")
                .expect("the packets are whole samples")
            {
                out.extend(window);
            }
        }
        out.extend(windows.tail());
        out
    }

    #[test]
    fn one_packet_is_cut_into_several_windows() {
        // Windows smaller than the packet: the cutter empties it a window at a
        // time and keeps what is left for the next one.
        let cut = drive(shape(4, 4), &[packet(0, 10, vec![])]);
        assert_eq!(cut.len(), 3, "two full windows and the leftover");
        assert_eq!(values(&cut[0]), vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(cut[0].pts, 0);
        assert_eq!(values(&cut[1]), vec![4.0, 5.0, 6.0, 7.0]);
        assert_eq!(cut[1].pts, 4);
        assert_eq!(values(&cut[2]), vec![8.0, 9.0], "the final call");
        assert_eq!(cut[2].pts, 8);
    }

    #[test]
    fn several_packets_make_one_window() {
        // Windows wider than the packets: nothing comes out until enough has
        // arrived, and what does is contiguous across the packet boundaries.
        let packets = [
            packet(0, 3, vec![]),
            packet(3, 3, vec![]),
            packet(6, 3, vec![]),
        ];
        let cut = drive(shape(8, 8), &packets);
        assert_eq!(cut.len(), 2);
        assert_eq!(
            values(&cut[0]),
            (0..8).map(|s| s as f32).collect::<Vec<_>>()
        );
        assert_eq!(cut[0].pts, 0);
        assert_eq!(values(&cut[1]), vec![8.0], "the final call");
        assert_eq!(cut[1].pts, 8);
    }

    #[test]
    fn a_window_wider_than_a_packet_still_strides_shorter_than_one() {
        // The awkward middle: window > packet > stride, so the cutter is
        // buffering up and cutting several windows out of one arrival at once.
        let packets = [
            packet(0, 6, vec![]),
            packet(6, 6, vec![]),
            packet(12, 6, vec![]),
        ];
        let cut = drive(shape(8, 4), &packets);
        let starts: Vec<i64> = cut.iter().map(|f| f.pts).collect();
        assert_eq!(starts, vec![0, 4, 8, 12], "a window every stride");
        for window in &cut[..3] {
            assert_eq!(window.data.len(), 8 * 4, "every full window is the window");
            assert_eq!(
                values(window),
                (window.pts..window.pts + 8)
                    .map(|s| s as f32)
                    .collect::<Vec<_>>(),
                "and holds the samples its timestamp names"
            );
        }
        assert_eq!(values(&cut[3]), vec![12.0, 13.0, 14.0, 15.0, 16.0, 17.0]);
    }

    #[test]
    fn rows_ride_the_window_whose_span_covers_them() {
        let packets = [
            packet(0, 3, vec![r#"{"a":0}"#.to_string()]),
            packet(3, 3, vec![]),
            packet(6, 3, vec![r#"{"a":6}"#.to_string()]),
        ];
        let cut = drive(shape(4, 4), &packets);
        let spans: Vec<(i64, Vec<String>)> = cut.iter().map(|f| (f.pts, f.rows.clone())).collect();
        assert_eq!(
            spans,
            vec![
                (0, vec![r#"{"a":0}"#.to_string()]),
                (4, vec![r#"{"a":6}"#.to_string()]),
                (8, vec![]),
            ],
            "the row at pts 0 rides [0,4) and the one at pts 6 rides [4,8)"
        );
    }

    #[test]
    fn a_row_arriving_past_the_last_full_window_rides_the_final_call() {
        // The row comes in on the last packet, and the stream ends before the
        // window covering it fills. The final call is what carries it.
        let packets = [
            packet(0, 4, vec![]),
            packet(4, 2, vec![r#"{"a":4}"#.to_string()]),
        ];
        let cut = drive(shape(4, 4), &packets);
        assert_eq!(cut.len(), 2, "one full window and the final call");
        assert!(cut[0].rows.is_empty(), "nothing arrived inside [0,4)");
        assert_eq!(cut[1].pts, 4);
        assert_eq!(values(&cut[1]), vec![4.0, 5.0], "the samples left over");
        assert_eq!(cut[1].rows, vec![r#"{"a":4}"#.to_string()]);
    }

    #[test]
    fn a_packet_that_is_not_whole_samples_is_refused_naming_the_module() {
        let mut windows = Windows::new(shape(4, 4), &audio_format());
        let err = windows
            .push(
                Frame {
                    pts: 0,
                    data: vec![0u8; 7].into(),
                    rows: Vec::new(),
                },
                "again",
            )
            .expect_err("seven bytes is not whole four-byte samples");
        let message = err.to_string();
        assert!(message.contains("again"), "got: {message}");
        assert!(message.contains("7 bytes"), "got: {message}");
    }

    #[test]
    fn frames_are_still_counted_rather_than_re_cut() {
        let mut windows = Windows::new(shape(2, 2), &video_format());
        let frame = |pts: i64| Frame {
            pts,
            data: vec![0u8; 16].into(),
            rows: Vec::new(),
        };
        assert!(windows
            .push(frame(0), "invert")
            .expect("a frame")
            .is_empty());
        let cut = windows.push(frame(1), "invert").expect("a frame");
        assert_eq!(cut.len(), 1, "the second frame completes the window");
        assert_eq!(cut[0].len(), 2, "and the window holds both frames");
        assert!(windows.tail().is_empty(), "the windows came out even");
    }

    #[test]
    fn a_sample_is_a_tick_at_the_natural_time_base() {
        let audio = AudioFormat {
            sample_rate: 48_000,
            channels: 2,
            sample_fmt: "s16",
        };
        assert_eq!(
            ticks_per_sample(
                audio,
                TimeBase {
                    num: 1,
                    den: 48_000
                }
            ),
            (1, 1)
        );
        // A base counting milliseconds instead: 48 samples to the tick.
        assert_eq!(
            ticks_per_sample(audio, TimeBase { num: 1, den: 1_000 }),
            (1, 48)
        );
    }
}
