//! NUT's byte primitives: the variable-length integers every field is coded
//! in, and the CRC32 that closes every packet.

use std::io::{ErrorKind, Read};

use anyhow::{anyhow, bail, Result};

/// NUT's CRC polynomial.
const CRC_POLY: u32 = 0x04C1_1DB7;

const fn crc_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = (i as u32) << 24;
        let mut bit = 0;
        while bit < 8 {
            c = if c & 0x8000_0000 != 0 {
                (c << 1) ^ CRC_POLY
            } else {
                c << 1
            };
            bit += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
}

static CRC_TABLE: [u32; 256] = crc_table();

/// NUT's CRC32: polynomial 0x04C11DB7 taken most significant bit first,
/// starting at zero, with no final inversion.
pub fn crc32(data: &[u8]) -> u32 {
    let mut c: u32 = 0;
    for &b in data {
        c = (c << 8) ^ CRC_TABLE[usize::from(((c >> 24) as u8) ^ b)];
    }
    c
}

/// A variable-length integer never needs more than 10 bytes for 64 bits.
const MAX_V_BYTES: usize = 10;

/// Largest byte string a `vb` field may carry, so a damaged length cannot ask
/// for an unbounded allocation.
const MAX_VB_LEN: u64 = 1 << 20;

/// Reads NUT's primitives from a stream, counting bytes so packet positions
/// are known, and optionally recording what it read so a checksum can cover
/// it. Works over a pipe: it never seeks.
pub struct ByteReader<R> {
    inner: R,
    pos: u64,
    tap: Option<Vec<u8>>,
}

impl<R: Read> ByteReader<R> {
    pub fn new(inner: R) -> Self {
        ByteReader {
            inner,
            pos: 0,
            tap: None,
        }
    }

    /// Starts recording every byte read from here on, after `seed` - the
    /// bytes a caller had already consumed and still wants covered.
    pub fn start_tap(&mut self, seed: &[u8]) {
        let mut tap = Vec::with_capacity(16);
        tap.extend_from_slice(seed);
        self.tap = Some(tap);
    }

    /// Stops recording and returns what was read since `start_tap`.
    pub fn take_tap(&mut self) -> Vec<u8> {
        self.tap.take().unwrap_or_default()
    }

    /// One byte, or None when the stream ends cleanly right here.
    pub fn read_u8_or_eof(&mut self) -> Result<Option<u8>> {
        let mut b = [0u8; 1];
        loop {
            match self.inner.read(&mut b) {
                Ok(0) => return Ok(None),
                Ok(_) => break,
                Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.into()),
            }
        }
        self.pos += 1;
        if let Some(tap) = self.tap.as_mut() {
            tap.push(b[0]);
        }
        Ok(Some(b[0]))
    }

    pub fn read_u8(&mut self) -> Result<u8> {
        self.read_u8_or_eof()?
            .ok_or_else(|| anyhow!("NUT stream ends mid-field at byte {}", self.pos))
    }

    /// Fills as much of `buf` as the stream has left, returning how many bytes
    /// that was. Short only at end of stream.
    pub fn read_up_to(&mut self, buf: &mut [u8]) -> Result<usize> {
        let mut total = 0;
        while total < buf.len() {
            match self.inner.read(&mut buf[total..]) {
                Ok(0) => break,
                Ok(n) => total += n,
                Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.into()),
            }
        }
        self.pos += total as u64;
        if let Some(tap) = self.tap.as_mut() {
            tap.extend_from_slice(&buf[..total]);
        }
        Ok(total)
    }

    /// Fills `buf`, naming how much was missing if the stream ends first.
    pub fn read_exact(&mut self, buf: &mut [u8], what: &str) -> Result<()> {
        let want = buf.len();
        let got = self.read_up_to(buf)?;
        if got != want {
            bail!("NUT stream ends inside {what}: got {got} of {want} bytes");
        }
        Ok(())
    }

    /// Discards `count` bytes.
    pub fn skip(&mut self, count: u64, what: &str) -> Result<()> {
        let mut scratch = [0u8; 8192];
        let mut left = count;
        while left > 0 {
            let want = left.min(scratch.len() as u64) as usize;
            let got = self.read_up_to(&mut scratch[..want])?;
            if got == 0 {
                bail!("NUT stream ends inside {what}: {left} bytes short");
            }
            left -= got as u64;
        }
        Ok(())
    }

    /// A `v` field: 7 bits per byte, most significant group first, high bit
    /// set on every byte but the last.
    pub fn read_v(&mut self) -> Result<u64> {
        let mut value: u64 = 0;
        for _ in 0..MAX_V_BYTES {
            let b = self.read_u8()?;
            if value > (u64::MAX >> 7) {
                bail!(
                    "NUT variable-length integer ending at byte {} overflows 64 bits",
                    self.pos
                );
            }
            value = (value << 7) | u64::from(b & 0x7f);
            if b & 0x80 == 0 {
                return Ok(value);
            }
        }
        bail!(
            "NUT variable-length integer at byte {} runs past {MAX_V_BYTES} bytes",
            self.pos
        )
    }

    /// An `s` field: a `v` biased so both signs are cheap to code.
    pub fn read_s(&mut self) -> Result<i64> {
        let raw = self.read_v()?;
        if raw >= u64::MAX / 2 {
            bail!("NUT signed integer at byte {} overflows 64 bits", self.pos);
        }
        let v = raw + 1;
        Ok(if v & 1 != 0 {
            -((v >> 1) as i64)
        } else {
            (v >> 1) as i64
        })
    }

    /// A `vb` field: a length, then that many bytes.
    pub fn read_vb(&mut self, what: &str) -> Result<Vec<u8>> {
        let len = self.read_v()?;
        if len > MAX_VB_LEN {
            bail!("NUT {what} claims {len} bytes, more than {MAX_VB_LEN}");
        }
        let mut buf = vec![0u8; len as usize];
        self.read_exact(&mut buf, what)?;
        Ok(buf)
    }

    pub fn read_u32(&mut self) -> Result<u32> {
        let mut buf = [0u8; 4];
        self.read_exact(&mut buf, "a 32-bit field")?;
        Ok(u32::from_be_bytes(buf))
    }
}

/// Appends a `v` field.
pub fn put_v(out: &mut Vec<u8>, value: u64) {
    let mut groups = [0u8; MAX_V_BYTES];
    let mut count = 0;
    let mut left = value;
    loop {
        groups[count] = (left & 0x7f) as u8;
        count += 1;
        left >>= 7;
        if left == 0 {
            break;
        }
    }
    while count > 0 {
        count -= 1;
        out.push(groups[count] | if count > 0 { 0x80 } else { 0 });
    }
}

/// Appends an `s` field.
pub fn put_s(out: &mut Vec<u8>, value: i64) {
    let magnitude = value.unsigned_abs();
    put_v(out, magnitude.saturating_mul(2) - u64::from(value > 0));
}

/// Appends a `vb` field: the length, then the bytes.
pub fn put_vb(out: &mut Vec<u8>, bytes: &[u8]) {
    put_v(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

pub fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

pub fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_be_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip_v(value: u64) {
        let mut buf = Vec::new();
        put_v(&mut buf, value);
        let mut reader = ByteReader::new(&buf[..]);
        assert_eq!(reader.read_v().unwrap(), value, "v {value} round trip");
        // Nothing is left over: a second read runs straight into the end.
        assert!(
            reader.read_u8_or_eof().unwrap().is_none(),
            "v {value} length"
        );
    }

    #[test]
    fn variable_length_integers_round_trip() {
        for value in [0, 1, 127, 128, 255, 256, 16383, 16384, 65536, u64::MAX] {
            round_trip_v(value);
        }
    }

    #[test]
    fn variable_length_integers_match_ffmpeg_bytes() {
        // Taken from a NUT file ffmpeg wrote: 32767 and 65536 as they appear
        // in the main header.
        let mut buf = Vec::new();
        put_v(&mut buf, 32767);
        assert_eq!(buf, vec![0x81, 0xff, 0x7f]);
        buf.clear();
        put_v(&mut buf, 65536);
        assert_eq!(buf, vec![0x84, 0x80, 0x00]);
    }

    #[test]
    fn signed_integers_round_trip() {
        for value in [0i64, 1, -1, 2, -2, 2048, -2048, i32::MAX as i64] {
            let mut buf = Vec::new();
            put_s(&mut buf, value);
            let mut reader = ByteReader::new(&buf[..]);
            assert_eq!(reader.read_s().unwrap(), value, "s {value} round trip");
        }
    }

    #[test]
    fn byte_strings_round_trip() {
        let mut buf = Vec::new();
        put_vb(&mut buf, b"RGBA");
        assert_eq!(buf, vec![4, b'R', b'G', b'B', b'A']);
        let mut reader = ByteReader::new(&buf[..]);
        assert_eq!(reader.read_vb("fourcc").unwrap(), b"RGBA");
    }

    #[test]
    fn checksum_matches_a_packet_ffmpeg_wrote() {
        // The syncpoint body ffmpeg wrote for a stream starting at pts 0:
        // global_key_pts 0, back_ptr_div16 0. Its checksum is zero, which is
        // what the CRC of two zero bytes comes to.
        assert_eq!(crc32(&[0, 0]), 0);
        // A one-byte message the polynomial does move.
        assert_eq!(crc32(&[0x01]), 0x04C1_1DB7);
    }

    #[test]
    fn a_truncated_field_names_the_stream_end() {
        let mut reader = ByteReader::new(&[0x81u8][..]);
        let err = reader.read_v().unwrap_err().to_string();
        assert!(err.contains("ends mid-field"), "{err}");
    }
}
