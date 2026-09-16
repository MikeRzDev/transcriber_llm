//! Stateful, CPU-only microphone cleanup at 16 kHz mono.
//!
//! Mild mode applies an 80 Hz high-pass and SpeexDSP adaptive suppression,
//! capped at 12 dB. AGC and the legacy Speex VAD are disabled. Arbitrary input
//! packets are assembled into 20 ms frames; the one-frame overlap delay is
//! removed once and drained on finish, preserving sample count and timestamps.
use std::ffi::{c_int, c_void};
use std::ptr::NonNull;

use anyhow::{ensure, Context, Result};
use clap::ValueEnum;

const FRAME: usize = 320;
const SAMPLE_RATE: c_int = 16_000;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum NoiseSuppression {
    #[default]
    Off,
    Mild,
}

impl NoiseSuppression {
    pub fn key(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Mild => "mild",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "Off",
            Self::Mild => "Mild (SpeexDSP)",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "off" => Some(Self::Off),
            "mild" => Some(Self::Mild),
            _ => None,
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::Off => Self::Mild,
            Self::Mild => Self::Off,
        }
    }
}

// ABI from vendor/speexdsp/include/speex/speex_preprocess.h. The opaque
// allocation is owned exclusively by Speex and never shared across threads.
unsafe extern "C" {
    fn speex_preprocess_state_init(frame_size: c_int, sampling_rate: c_int) -> *mut c_void;
    fn speex_preprocess_state_destroy(state: *mut c_void);
    fn speex_preprocess_ctl(state: *mut c_void, request: c_int, value: *mut c_void) -> c_int;
    fn speex_preprocess_run(state: *mut c_void, frame: *mut i16) -> c_int;
}

struct Speex(NonNull<c_void>);

impl Speex {
    fn new() -> Result<Self> {
        // SAFETY: fixed, positive frame size and supported sample rate.
        let state = Self(
            NonNull::new(unsafe { speex_preprocess_state_init(FRAME as c_int, SAMPLE_RATE) })
                .context("Cannot initialize microphone noise suppression")?,
        );
        // SET_DENOISE, SET_AGC, SET_NOISE_SUPPRESS. VAD defaults to disabled.
        for (request, mut value) in [(0, 1_i32), (2, 0), (18, -12)] {
            // SAFETY: state is live; these requests take a pointer to an int32.
            let result = unsafe {
                speex_preprocess_ctl(state.0.as_ptr(), request, (&mut value as *mut i32).cast())
            };
            ensure!(result == 0, "Cannot configure microphone noise suppression");
        }
        Ok(state)
    }

    fn process(&mut self, frame: &mut [i16; FRAME]) {
        // SAFETY: exclusive, live state and exactly FRAME writable samples.
        unsafe {
            speex_preprocess_run(self.0.as_ptr(), frame.as_mut_ptr());
        }
    }
}

impl Drop for Speex {
    fn drop(&mut self) {
        // SAFETY: this is the sole owner and destruction occurs exactly once.
        unsafe {
            speex_preprocess_state_destroy(self.0.as_ptr());
        }
    }
}

pub struct NoiseFilter {
    speex: Option<Speex>,
    pending: [i16; FRAME],
    pending_len: usize,
    skip: usize,
    input_len: usize,
    output_len: usize,
    previous_input: f32,
    previous_output: f32,
    high_pass_alpha: f32,
    finished: bool,
}

impl NoiseFilter {
    pub fn new(mode: NoiseSuppression) -> Result<Self> {
        Ok(Self {
            speex: match mode {
                NoiseSuppression::Off => None,
                NoiseSuppression::Mild => Some(Speex::new()?),
            },
            pending: [0; FRAME],
            pending_len: 0,
            skip: FRAME,
            input_len: 0,
            output_len: 0,
            previous_input: 0.0,
            previous_output: 0.0,
            high_pass_alpha: 1.0 / (1.0 + std::f32::consts::TAU * 80.0 / SAMPLE_RATE as f32),
            finished: false,
        })
    }

    /// Off is an exact bypass. Call only before finish().
    pub fn push(&mut self, samples: &[f32]) -> Vec<f32> {
        assert!(
            !self.finished,
            "cannot push audio after NoiseFilter::finish"
        );
        self.input_len += samples.len();
        if self.speex.is_none() {
            self.output_len += samples.len();
            return samples.to_vec();
        }
        let mut output = Vec::with_capacity(samples.len() + FRAME);
        for &sample in samples {
            let sample = if sample.is_finite() {
                sample.clamp(-1.0, 1.0)
            } else {
                0.0
            };
            let filtered =
                self.high_pass_alpha * (self.previous_output + sample - self.previous_input);
            self.previous_input = sample;
            self.previous_output = filtered;
            self.pending[self.pending_len] = (filtered.clamp(-1.0, 1.0) * 32768.0)
                .round()
                .clamp(-32768.0, 32767.0) as i16;
            self.pending_len += 1;
            if self.pending_len == FRAME {
                self.frame(&mut output);
            }
        }
        output
    }

    fn frame(&mut self, output: &mut Vec<f32>) {
        self.speex.as_mut().unwrap().process(&mut self.pending);
        let skip = self.skip.min(FRAME);
        self.skip -= skip;
        let take = (FRAME - skip).min(self.input_len - self.output_len);
        output.extend(
            self.pending[skip..skip + take]
                .iter()
                .map(|&s| s as f32 / 32768.0),
        );
        self.output_len += take;
        self.pending.fill(0);
        self.pending_len = 0;
    }

    /// Flush the partial frame and Speex overlap, without returning padding.
    /// Repeated calls are harmless. Start a new filter for each recording.
    pub fn finish(&mut self) -> Vec<f32> {
        if self.finished {
            return Vec::new();
        }
        self.finished = true;
        let mut output = Vec::new();
        if self.speex.is_some() && self.output_len < self.input_len {
            if self.pending_len > 0 {
                self.frame(&mut output);
            }
            self.frame(&mut output);
        }
        debug_assert_eq!(self.input_len, self.output_len);
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filter(input: &[f32], chunk: usize, mode: NoiseSuppression) -> Vec<f32> {
        let mut filter = NoiseFilter::new(mode).unwrap();
        let mut out = Vec::new();
        for packet in input.chunks(chunk) {
            out.extend(filter.push(packet));
        }
        out.extend(filter.finish());
        assert!(filter.finish().is_empty());
        out
    }

    fn noise(len: usize) -> Vec<f32> {
        let mut state = 12345_u32;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32 * 0.04
            })
            .collect()
    }

    fn energy(input: &[f32]) -> f32 {
        input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32
    }

    #[test]
    fn packets_and_flush_preserve_duration_and_tail() {
        for len in [0, 1, 137, 319, 320, 321, 640, 17337] {
            let mut input = noise(len);
            if len > 0 {
                input[len - 1] = 0.8;
            }
            let expected = filter(&input, 1600, NoiseSuppression::Mild);
            assert_eq!(expected.len(), len);
            for chunk in [1, 137, 320, 1601] {
                assert_eq!(filter(&input, chunk, NoiseSuppression::Mild), expected);
            }
            if len > 1 {
                assert!(expected[len - 1].abs() > 0.05, "lost final sample at {len}");
            }
        }
    }

    #[test]
    fn off_is_exact_and_silence_stays_silent() {
        let input = noise(17337);
        assert_eq!(filter(&input, 137, NoiseSuppression::Off), input);
        assert_eq!(
            filter(&[0.0; 3333], 137, NoiseSuppression::Mild),
            vec![0.0; 3333]
        );
        let output = filter(
            &[f32::NAN, f32::INFINITY, -2.0, 2.0],
            1,
            NoiseSuppression::Mild,
        );
        assert!(output.iter().all(|x| x.is_finite() && x.abs() <= 1.0));
    }

    #[test]
    fn suppresses_stationary_noise_and_keeps_a_new_voiced_signal() {
        let mut input = noise(16000 * 5);
        // Noise-only lead-in, then a speech-like harmonic signal.
        let start = 16000 * 3;
        for (i, sample) in input.iter_mut().enumerate().skip(start) {
            let phase = std::f32::consts::TAU * 180.0 * i as f32 / 16000.0;
            *sample += 0.15 * phase.sin() + 0.07 * (phase * 2.0).sin();
        }
        let out = filter(&input, 137, NoiseSuppression::Mild);
        let ratio = energy(&out[16000..start]) / energy(&input[16000..start]);
        assert!(ratio < 0.5, "noise energy ratio {ratio}");
        let voiced_ratio = energy(&out[start + 1600..]) / energy(&input[start + 1600..]);
        assert!(voiced_ratio > 0.35, "voiced energy ratio {voiced_ratio}");
    }
}
