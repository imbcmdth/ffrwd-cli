//! whisper base over one 30-second window, decoded greedily.
//!
//! The window arrives as 16 kHz mono samples, becomes a log-mel spectrogram,
//! and is decoded once. Timestamps are on, so the token stream carries the
//! within-window times the segments are cut at.
//!
//! The model is multilingual, so the decode opens on a three-token prompt -
//! start-of-transcript, the spoken language, the task - and the caller says
//! which language and which task.
//!
//! Decoding is greedy - the most likely token every step, temperature zero -
//! so a window always decodes to the same text. There is no temperature
//! fallback: a window the thresholds reject yields no segments rather than
//! being retried with sampling, which would need a random source and would
//! make the same audio decode differently on different runs.

use candle_core::{IndexOp, Tensor};
use candle_nn::ops::{log_softmax, softmax};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, model::Whisper, Config};

use crate::mel;
use tokenizers::Tokenizer;

/// Seconds one timestamp token step is worth.
const TIMESTAMP_STEP: f64 = 0.02;

/// The languages whisper was trained on, as ISO 639-1 codes. Each is a
/// `<|xx|>` token in the multilingual tokenizer.
pub const LANGUAGES: [&str; 99] = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su",
];

/// Whether whisper knows the language this code names.
pub fn is_language(code: &str) -> bool {
    LANGUAGES.contains(&code)
}

/// Which of whisper's two jobs a decode does.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Task {
    Transcribe,
    Translate,
}

/// One stretch of speech, in seconds from the start of the window it was
/// decoded from.
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// The model, its tokenizer, and the token ids the decoding rules are written
/// in terms of.
pub struct Transcriber {
    model: Whisper,
    tokenizer: Tokenizer,
    config: Config,
    mel_filters: Vec<f32>,
    suppress_tokens: Tensor,
    /// What every decode opens on: start-of-transcript, the language, the task.
    prompt: Vec<u32>,
    eot_token: u32,
    no_speech_token: u32,
    no_timestamps_token: u32,
}

/// What one greedy decode produced.
struct Decoded {
    tokens: Vec<u32>,
    avg_logprob: f64,
    no_speech_prob: f64,
}

/// The id of a special token, or an error naming the token that is missing.
fn token_id(tokenizer: &Tokenizer, token: &str) -> Result<u32, String> {
    tokenizer
        .token_to_id(token)
        .ok_or_else(|| format!("transcribe: the tokenizer has no {token} token"))
}

/// The mel filterbank as floats, from the little-endian f32 it is stored as.
fn mel_filters(bytes: &[u8]) -> Vec<f32> {
    let (whole, _) = bytes.as_chunks::<4>();
    whole.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

/// The three tokens a decode opens on, for one language and one task.
fn prompt(tokenizer: &Tokenizer, language: &str, task: Task) -> Result<Vec<u32>, String> {
    let task_token = match task {
        Task::Transcribe => m::TRANSCRIBE_TOKEN,
        Task::Translate => m::TRANSLATE_TOKEN,
    };
    Ok(vec![
        token_id(tokenizer, m::SOT_TOKEN)?,
        token_id(tokenizer, &format!("<|{language}|>"))?,
        token_id(tokenizer, task_token)?,
    ])
}

impl Transcriber {
    /// Loads the model from the bytes compiled into the module, decoding
    /// `language` for `task`.
    pub fn load(
        weights: &[u8],
        tokenizer_json: &[u8],
        config_json: &[u8],
        melfilters: &[u8],
        language: &str,
        task: Task,
    ) -> Result<Self, String> {
        // candle sizes gemm's thread pool from the processor count unless this
        // says otherwise, and wasip2 has no threads to spawn: without it the
        // first matrix multiply traps.
        std::env::set_var("RAYON_NUM_THREADS", "1");

        let config: Config = serde_json::from_slice(config_json)
            .map_err(|e| format!("transcribe: the whisper config does not parse: {e}"))?;
        let tokenizer = Tokenizer::from_bytes(tokenizer_json)
            .map_err(|e| format!("transcribe: the whisper tokenizer does not parse: {e}"))?;

        if config.num_mel_bins != 80 {
            return Err(format!(
                "transcribe: the model wants {} mel bins and the filterbank compiled in has 80",
                config.num_mel_bins
            ));
        }

        let device = candle_core::Device::Cpu;
        let vb = VarBuilder::from_slice_safetensors(weights, m::DTYPE, &device)
            .map_err(|e| format!("transcribe: the whisper weights do not load: {e}"))?;
        let model = Whisper::load(&vb, config.clone())
            .map_err(|e| format!("transcribe: the whisper weights do not load: {e}"))?;

        let no_timestamps_token = token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;
        // Timestamps are on, so the token that turns them off is suppressed
        // along with the ones the model itself names.
        let suppress: Vec<f32> = (0..config.vocab_size as u32)
            .map(|i| {
                if config.suppress_tokens.contains(&i) || i == no_timestamps_token {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let suppress_tokens =
            Tensor::new(suppress.as_slice(), &device).map_err(|e| format!("transcribe: {e}"))?;

        let no_speech_token = m::NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| token_id(&tokenizer, token).ok())
            .ok_or_else(|| {
                format!(
                    "transcribe: the tokenizer has none of {}",
                    m::NO_SPEECH_TOKENS.join(", ")
                )
            })?;

        Ok(Transcriber {
            prompt: prompt(&tokenizer, language, task)?,
            eot_token: token_id(&tokenizer, m::EOT_TOKEN)?,
            no_speech_token,
            no_timestamps_token,
            suppress_tokens,
            mel_filters: mel_filters(melfilters),
            model,
            tokenizer,
            config,
        })
    }

    /// Points the decoding at another language, or another task, from the next
    /// window on.
    pub fn retarget(&mut self, language: &str, task: Task) -> Result<(), String> {
        self.prompt = prompt(&self.tokenizer, language, task)?;
        Ok(())
    }

    /// The segments one window of 16 kHz mono samples decodes to. Empty when
    /// the model reports no speech.
    pub fn window(&mut self, pcm: &[f32]) -> Result<Vec<Segment>, String> {
        if pcm.is_empty() {
            return Ok(Vec::new());
        }
        let device = candle_core::Device::Cpu;
        let bins = self.config.num_mel_bins;

        // The spectrogram is padded out past the samples given, so only the
        // frames one window covers are decoded.
        let mel = mel::pcm_to_mel(bins, pcm, &self.mel_filters);
        let frames = mel.len() / bins;
        let mel = Tensor::from_vec(mel, (1, bins, frames), &device)
            .map_err(|e| format!("transcribe: {e}"))?;
        let frames = frames.min(m::N_FRAMES);
        let mel = mel
            .narrow(2, 0, frames)
            .map_err(|e| format!("transcribe: {e}"))?;

        let decoded = self.decode(&mel)?;
        // Whisper's own no-speech rule: both the probability and the
        // confidence must say so, since either alone is noisy.
        if decoded.no_speech_prob > m::NO_SPEECH_THRESHOLD
            && decoded.avg_logprob < m::LOGPROB_THRESHOLD
        {
            return Ok(Vec::new());
        }
        let window_seconds = pcm.len() as f64 / m::SAMPLE_RATE as f64;
        self.segments(&decoded.tokens, window_seconds)
    }

    /// One greedy pass over the decoder, from the start-of-transcript token to
    /// the end-of-text one.
    fn decode(&mut self, mel: &Tensor) -> Result<Decoded, String> {
        let err = |e: candle_core::Error| format!("transcribe: {e}");

        let audio_features = self.model.encoder.forward(mel, true).map_err(err)?;
        let sample_len = self.config.max_target_positions / 2;
        let mut tokens = self.prompt.clone();
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;

        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())
                .map_err(err)?
                .unsqueeze(0)
                .map_err(err)?;
            let ys = self
                .model
                .decoder
                .forward(&tokens_t, &audio_features, i == 0)
                .map_err(err)?;

            // The chance this window is silence, read off the first step.
            if i == 0 {
                let logits = self
                    .model
                    .decoder
                    .final_linear(&ys.i(..1).map_err(err)?)
                    .map_err(err)?
                    .i(0)
                    .map_err(err)?
                    .i(0)
                    .map_err(err)?;
                no_speech_prob = f64::from(
                    softmax(&logits, 0)
                        .map_err(err)?
                        .i(self.no_speech_token as usize)
                        .map_err(err)?
                        .to_scalar::<f32>()
                        .map_err(err)?,
                );
            }

            let (_, seq_len, _) = ys.dims3().map_err(err)?;
            let logits = self
                .model
                .decoder
                .final_linear(&ys.i((..1, seq_len - 1..)).map_err(err)?)
                .map_err(err)?
                .i(0)
                .map_err(err)?
                .i(0)
                .map_err(err)?;
            let logits = self.apply_timestamp_rules(&logits, &tokens)?;
            let logits = logits.broadcast_add(&self.suppress_tokens).map_err(err)?;

            let logits_v: Vec<f32> = logits.to_vec1().map_err(err)?;
            let next_token = logits_v
                .iter()
                .enumerate()
                .max_by(|(_, u), (_, v)| u.total_cmp(v))
                .map(|(i, _)| i as u32)
                .expect("the vocabulary is not empty");

            tokens.push(next_token);
            let prob = f64::from(
                softmax(&logits, candle_core::D::Minus1)
                    .map_err(err)?
                    .i(next_token as usize)
                    .map_err(err)?
                    .to_scalar::<f32>()
                    .map_err(err)?,
            );
            if next_token == self.eot_token || tokens.len() > self.config.max_target_positions {
                break;
            }
            sum_logprob += prob.ln();
        }

        Ok(Decoded {
            avg_logprob: sum_logprob / tokens.len() as f64,
            no_speech_prob,
            tokens,
        })
    }

    /// Whisper's constraints on where a timestamp token may appear: they come
    /// in pairs, they never go backwards, the transcript opens with one, and a
    /// step whose timestamps outweigh every word must produce one.
    fn apply_timestamp_rules(&self, input: &Tensor, tokens: &[u32]) -> Result<Tensor, String> {
        let err = |e: candle_core::Error| format!("transcribe: {e}");
        let device = input.device().clone();
        let timestamp_begin = self.no_timestamps_token + 1;
        let vocab_size = self.config.vocab_size as u32;
        // Past the start-of-transcript, language and task tokens, which are not
        // sampled.
        const SAMPLE_BEGIN: usize = 3;

        let sampled = if tokens.len() > SAMPLE_BEGIN {
            &tokens[SAMPLE_BEGIN..]
        } else {
            &[][..]
        };

        let mut logits = input.clone();
        let mut buffer = vec![0f32; vocab_size as usize];
        // Applies one mask, built by `keep` saying which ids stay reachable.
        let suppress = |logits: &Tensor,
                        buffer: &mut Vec<f32>,
                        keep: &dyn Fn(u32) -> bool|
         -> Result<Tensor, String> {
            for (i, slot) in buffer.iter_mut().enumerate() {
                *slot = if keep(i as u32) {
                    0.0
                } else {
                    f32::NEG_INFINITY
                };
            }
            let mask = Tensor::new(buffer.as_slice(), &device).map_err(err)?;
            logits.broadcast_add(&mask).map_err(err)
        };

        if let Some(&last) = sampled.last() {
            let last_was_timestamp = last >= timestamp_begin;
            let penultimate_was_timestamp =
                sampled.len() >= 2 && sampled[sampled.len() - 2] >= timestamp_begin;

            if last_was_timestamp {
                logits = if penultimate_was_timestamp {
                    // A closed pair: the next token must be a word.
                    suppress(&logits, &mut buffer, &|i| i < timestamp_begin)?
                } else {
                    // An open pair: the next token must close it or end.
                    suppress(&logits, &mut buffer, &|i| i >= self.eot_token)?
                };
            }

            let timestamps: Vec<u32> = sampled
                .iter()
                .copied()
                .filter(|&t| t >= timestamp_begin)
                .collect();
            if let Some(&latest) = timestamps.last() {
                // An open pair may close where it opened; a closed one must
                // move on.
                let floor = if last_was_timestamp && !penultimate_was_timestamp {
                    latest
                } else {
                    latest + 1
                };
                logits = suppress(&logits, &mut buffer, &|i| i < timestamp_begin || i >= floor)?;
            }
        }

        if tokens.len() == SAMPLE_BEGIN {
            // The transcript opens on a timestamp.
            logits = suppress(&logits, &mut buffer, &|i| i >= timestamp_begin)?;
        }

        // When the timestamps together outweigh the likeliest word, take a
        // timestamp.
        let log_probs = log_softmax(&logits, 0).map_err(err)?;
        let timestamp_log_probs = log_probs
            .narrow(
                0,
                timestamp_begin as usize,
                (vocab_size - timestamp_begin) as usize,
            )
            .map_err(err)?;
        let text_log_probs = log_probs
            .narrow(0, 0, timestamp_begin as usize)
            .map_err(err)?;

        let max_val = timestamp_log_probs.max(0).map_err(err)?;
        let timestamp_logprob = max_val
            .broadcast_add(
                &timestamp_log_probs
                    .broadcast_sub(&max_val)
                    .map_err(err)?
                    .exp()
                    .map_err(err)?
                    .sum(0)
                    .map_err(err)?
                    .log()
                    .map_err(err)?,
            )
            .map_err(err)?
            .to_scalar::<f32>()
            .map_err(err)?;
        let max_text_logprob = text_log_probs
            .max(0)
            .map_err(err)?
            .to_scalar::<f32>()
            .map_err(err)?;

        if timestamp_logprob > max_text_logprob {
            logits = suppress(&logits, &mut buffer, &|i| i >= timestamp_begin)?;
        }
        Ok(logits)
    }

    /// The decoded tokens cut into segments at the timestamp tokens. A run of
    /// words that the model never closed runs to the end of the window.
    fn segments(&self, tokens: &[u32], window_seconds: f64) -> Result<Vec<Segment>, String> {
        let mut segments = Vec::new();
        let mut pending: Vec<u32> = Vec::new();
        let mut start = 0f64;

        // Flushes the words gathered so far as one segment ending at `end`.
        let flush = |pending: &mut Vec<u32>,
                     start: f64,
                     end: f64,
                     segments: &mut Vec<Segment>|
         -> Result<(), String> {
            if pending.is_empty() {
                return Ok(());
            }
            let text = self
                .tokenizer
                .decode(pending, true)
                .map_err(|e| format!("transcribe: the tokens do not decode: {e}"))?;
            pending.clear();
            let text = text.trim().to_string();
            if !text.is_empty() && end > start {
                segments.push(Segment { start, end, text });
            }
            Ok(())
        };

        for &token in tokens {
            if token > self.no_timestamps_token {
                let at = f64::from(token - self.no_timestamps_token - 1) * TIMESTAMP_STEP;
                flush(&mut pending, start, at, &mut segments)?;
                start = at;
            } else if token >= self.eot_token {
                // The prompt's own tokens and end-of-text, which are not words.
                continue;
            } else {
                pending.push(token);
            }
        }
        flush(&mut pending, start, window_seconds, &mut segments)?;
        Ok(segments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_filterbank_reads_back_as_little_endian_floats() {
        let bytes: Vec<u8> = [1.0f32, -0.5, 0.25]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        assert_eq!(mel_filters(&bytes), vec![1.0, -0.5, 0.25]);
        assert_eq!(mel_filters(&[]), Vec::<f32>::new());
    }

    #[test]
    fn a_timestamp_token_is_two_hundredths_of_a_second_past_the_one_before() {
        // The first timestamp token sits one past the no-timestamps token and
        // means zero seconds; each one after is a fiftieth of a second later.
        let no_timestamps = 50_363u32;
        let at = |token: u32| f64::from(token - no_timestamps - 1) * TIMESTAMP_STEP;
        assert!(
            at(50_364).abs() < 1e-12,
            "the first one is the window start"
        );
        assert!((at(50_365) - 0.02).abs() < 1e-12);
        assert!((at(50_453) - 1.78).abs() < 1e-12);
    }

    #[test]
    fn every_language_whisper_knows_is_a_two_or_three_letter_code() {
        assert_eq!(LANGUAGES.len(), 99);
        for code in LANGUAGES {
            assert!(
                (2..=3).contains(&code.len()) && code.chars().all(|c| c.is_ascii_lowercase()),
                "{code} is not a language code"
            );
        }
        assert!(is_language("es"));
        assert!(is_language("cy"));
        assert!(!is_language("EN"));
        assert!(!is_language("klingon"));
    }
}
