//! The log-mel spectrogram whisper reads, computed on this thread alone.
//!
//! candle ships this, but its version always spawns at least two threads, and
//! wasip2 has none to give. The arithmetic here is candle's, down to the order
//! the mel sums are added in - floating-point addition does not associate, so
//! reordering it would move the last digits of every value and, now and then,
//! a token. What that version splits across threads by column, this walks in
//! one pass; the columns are independent, so the result is the same numbers.

use candle_transformers::models::whisper as m;

/// Frequency bins one spectrogram column holds, for whisper's 400-point
/// transform.
const N_FFT_BINS: usize = 1 + m::N_FFT / 2;

/// Real-input FFT, returning interleaved re/im pairs. Splits even from odd
/// while the length is even and falls back to the direct transform when it is
/// not.
fn fft(input: &[f32]) -> Vec<f32> {
    let n = input.len();
    if n == 1 {
        return vec![input[0], 0.0];
    }
    if n % 2 == 1 {
        return dft(input);
    }

    let mut even = Vec::with_capacity(n / 2);
    let mut odd = Vec::with_capacity(n / 2);
    for (i, &value) in input.iter().enumerate() {
        if i % 2 == 0 {
            even.push(value);
        } else {
            odd.push(value);
        }
    }
    let even_fft = fft(&even);
    let odd_fft = fft(&odd);

    let mut out = vec![0.0; n * 2];
    let two_pi = std::f32::consts::TAU;
    for k in 0..n / 2 {
        let theta = two_pi * k as f32 / n as f32;
        let re = theta.cos();
        let im = -theta.sin();

        let re_odd = odd_fft[2 * k];
        let im_odd = odd_fft[2 * k + 1];

        out[2 * k] = even_fft[2 * k] + re * re_odd - im * im_odd;
        out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;
        out[2 * (k + n / 2)] = even_fft[2 * k] - re * re_odd + im * im_odd;
        out[2 * (k + n / 2) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
    }
    out
}

/// The direct transform, for a length the halving cannot split.
fn dft(input: &[f32]) -> Vec<f32> {
    let n = input.len();
    let two_pi = std::f32::consts::TAU;
    let mut out = Vec::with_capacity(2 * n);
    for k in 0..n {
        let mut re = 0.0;
        let mut im = 0.0;
        for (j, &value) in input.iter().enumerate() {
            let angle = two_pi * k as f32 * j as f32 / n as f32;
            re += value * angle.cos();
            im -= value * angle.sin();
        }
        out.push(re);
        out.push(im);
    }
    out
}

/// One column of the spectrogram per hop, each the log of the sample's energy
/// through the mel filterbank.
fn columns(hann: &[f32], samples: &[f32], filters: &[f32], n_len: usize, n_mel: usize) -> Vec<f32> {
    let fft_size = m::N_FFT;
    let fft_step = m::HOP_LENGTH;
    let n_samples = samples.len();
    let end = (n_samples / fft_step + 1).min(n_len);

    let mut fft_in = vec![0.0f32; fft_size];
    let mut mel = vec![0.0f32; n_len * n_mel];

    for i in 0..end {
        let offset = i * fft_step;
        for j in 0..fft_size.min(n_samples - offset) {
            fft_in[j] = hann[j] * samples[offset + j];
        }
        if n_samples - offset < fft_size {
            fft_in[n_samples - offset..].fill(0.0);
        }

        let mut fft_out = fft(&fft_in);
        // Power, then fold the mirrored half onto the one that is kept.
        for j in 0..fft_size {
            fft_out[j] = fft_out[2 * j] * fft_out[2 * j] + fft_out[2 * j + 1] * fft_out[2 * j + 1];
        }
        for j in 1..fft_size / 2 {
            let v = fft_out[fft_size - j];
            fft_out[j] += v;
        }

        for j in 0..n_mel {
            let mut sum = 0.0f32;
            let mut k = 0;
            // Four at a time, which is how candle adds them up.
            while k < N_FFT_BINS.saturating_sub(3) {
                sum += fft_out[k] * filters[j * N_FFT_BINS + k]
                    + fft_out[k + 1] * filters[j * N_FFT_BINS + k + 1]
                    + fft_out[k + 2] * filters[j * N_FFT_BINS + k + 2]
                    + fft_out[k + 3] * filters[j * N_FFT_BINS + k + 3];
                k += 4;
            }
            while k < N_FFT_BINS {
                sum += fft_out[k] * filters[j * N_FFT_BINS + k];
                k += 1;
            }
            mel[j * n_len + i] = sum.max(1e-10).log10();
        }
    }
    mel
}

/// The spectrogram of one stretch of 16 kHz mono samples, laid out mel bin by
/// mel bin. The samples are padded out with silence first, so the result is
/// always at least one whole chunk longer than the audio given.
pub fn pcm_to_mel(n_mel: usize, samples: &[f32], filters: &[f32]) -> Vec<f32> {
    let fft_size = m::N_FFT;
    let fft_step = m::HOP_LENGTH;
    let two_pi = std::f32::consts::TAU;

    let hann: Vec<f32> = (0..fft_size)
        .map(|i| 0.5 * (1.0 - (two_pi * i as f32 / fft_size as f32).cos()))
        .collect();

    // Rounded up to a whole chunk, then one more chunk of silence on the end.
    let pad = 100 * m::CHUNK_LENGTH / 2;
    let n_len = samples.len() / fft_step;
    let n_len = if n_len.is_multiple_of(pad) {
        n_len
    } else {
        (n_len / pad + 1) * pad
    };
    let n_len = n_len + pad;

    let mut padded = samples.to_vec();
    padded.resize(n_len * fft_step, 0.0);

    let mut mel = columns(&hann, &padded, filters, n_len, n_mel);

    // Everything more than eight decades below the loudest is that quiet, and
    // the scale is squeezed into roughly -1 to 1.
    let mmax = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max) - 8.0;
    for value in mel.iter_mut() {
        *value = value.max(mmax) / 4.0 + 1.0;
    }
    mel
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The transform of a single spike is flat across every bin.
    #[test]
    fn a_spike_transforms_to_a_flat_spectrum() {
        let mut input = vec![0.0f32; 8];
        input[0] = 1.0;
        let out = fft(&input);
        for k in 0..8 {
            assert!((out[2 * k] - 1.0).abs() < 1e-6, "bin {k} real part");
            assert!(out[2 * k + 1].abs() < 1e-6, "bin {k} imaginary part");
        }
    }

    /// The halving path and the direct one must agree where both apply.
    #[test]
    fn the_split_transform_agrees_with_the_direct_one() {
        let input: Vec<f32> = (0..16).map(|i| (i as f32 * 0.7).sin()).collect();
        let split = fft(&input);
        let direct = dft(&input);
        for (i, (a, b)) in split.iter().zip(&direct).enumerate() {
            assert!((a - b).abs() < 1e-4, "slot {i}: {a} against {b}");
        }
    }

    #[test]
    fn the_hann_window_opens_and_closes_at_nothing() {
        let n = 400;
        let hann: Vec<f32> = (0..n)
            .map(|i| 0.5 * (1.0 - (std::f32::consts::TAU * i as f32 / n as f32).cos()))
            .collect();
        assert!(hann[0].abs() < 1e-6, "it opens at nothing");
        assert!((hann[n / 2] - 1.0).abs() < 1e-6, "and peaks in the middle");
    }

    /// A window's worth of samples pads out to one chunk past itself, which is
    /// what leaves room for the frames whisper is decoded over.
    #[test]
    fn the_spectrogram_is_padded_a_whole_chunk_past_the_audio() {
        let filters = vec![0.0f32; 80 * N_FFT_BINS];
        let mel = pcm_to_mel(80, &vec![0.0f32; 480_000], &filters);
        assert_eq!(mel.len() / 80, m::N_FRAMES + 1500);
    }
}
