//! Microphone capture lives on its own thread so inference cannot stall the
//! level meter or stopping the device. The audio queue is bounded: falling
//! behind fails visibly rather than silently losing speech or growing RAM.
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, sync_channel, Receiver, Sender, SyncSender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

use crate::transcribe::Event;

/// Input-capable CoreAudio devices include built-in, USB and connected
/// Bluetooth microphones. Enumeration does not start a recording stream.
pub fn input_devices() -> Result<Vec<String>> {
    let mut names = cpal::default_host()
        .input_devices()
        .context("Cannot list microphone inputs")?
        .filter_map(|device| device.name().ok())
        .collect::<Vec<_>>();
    names.sort();
    Ok(names)
}

fn input_index(names: &[String], requested: &str) -> Result<usize> {
    let matches: Vec<_> = names
        .iter()
        .enumerate()
        .filter(|(_, name)| *name == requested)
        .map(|(i, _)| i)
        .collect();
    match matches.as_slice() {
        [index] => Ok(*index),
        [] => bail!("Microphone '{requested}' is unavailable. Connect it (including Bluetooth), then press a to refresh and select an input"),
        _ => bail!("More than one microphone is named '{requested}'. Select it as the macOS Sound input and choose System default in the app"),
    }
}

fn resolve_input(requested: Option<&str>) -> Result<cpal::Device> {
    let host = cpal::default_host();
    let Some(requested) = requested else {
        return host.default_input_device().context(
            "No microphone found. Connect a microphone and select it in macOS Sound settings",
        );
    };
    let mut devices: Vec<_> = host
        .input_devices()?
        .filter_map(|device| device.name().ok().map(|name| (name, device)))
        .collect();
    let names = devices
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    let index = input_index(&names, requested)?;
    Ok(devices.swap_remove(index).1)
}

pub struct Capture {
    pub packets: Receiver<Vec<f32>>,
    pub sample_rate: u32,
    pub device: String,
    error: Arc<Mutex<Option<String>>>,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Capture {
    pub fn open(
        events: Sender<Event>,
        cancel: Arc<AtomicBool>,
        stop: Arc<AtomicBool>,
        input_device: Option<String>,
    ) -> Result<Self> {
        let (tx, packets) = sync_channel(128); // 12.8 seconds at 100 ms per packet
        let (ready_tx, ready_rx) = channel();
        let error = Arc::new(Mutex::new(None));
        let worker_error = error.clone();
        let worker_stop = stop.clone();
        let handle = std::thread::spawn(move || {
            let result = (|| -> Result<()> {
                let device = resolve_input(input_device.as_deref())?;
                let supported = device.default_input_config().context("Cannot open microphone. Allow your terminal in macOS Privacy & Security → Microphone")?;
                let config: cpal::StreamConfig = supported.clone().into();
                let name = device
                    .name()
                    .unwrap_or_else(|_| "Default microphone".into());
                let args = Callback {
                    tx,
                    events: events.clone(),
                    error: worker_error.clone(),
                    stop: worker_stop.clone(),
                    cancel: cancel.clone(),
                    channels: config.channels as usize,
                    rate: config.sample_rate.0,
                    pending: Vec::new(),
                    frames: 0,
                };
                let err = worker_error.clone();
                let on_error = move |e: cpal::StreamError| {
                    *err.lock().unwrap() = Some(format!("Microphone disconnected or failed: {e}"));
                };
                let stream = match supported.sample_format() {
                    cpal::SampleFormat::F32 => build::<f32>(&device, &config, args, on_error),
                    cpal::SampleFormat::I16 => build::<i16>(&device, &config, args, on_error),
                    cpal::SampleFormat::U16 => build::<u16>(&device, &config, args, on_error),
                    cpal::SampleFormat::I32 => build::<i32>(&device, &config, args, on_error),
                    cpal::SampleFormat::F64 => build::<f64>(&device, &config, args, on_error),
                    format => bail!("Unsupported microphone sample format: {format}"),
                }.context("Cannot start microphone. Check your terminal's Microphone permission in macOS Privacy & Security")?;
                stream.play()?;
                let _ = ready_tx.send(Ok((config.sample_rate.0, name)));
                while !worker_stop.load(Ordering::Relaxed)
                    && !cancel.load(Ordering::Relaxed)
                    && worker_error.lock().unwrap().is_none()
                {
                    std::thread::sleep(Duration::from_millis(20));
                }
                drop(stream);
                Ok(())
            })();
            if let Err(e) = result {
                let message = format!("{e:#}");
                *worker_error.lock().unwrap() = Some(message.clone());
                let _ = ready_tx.send(Err(message));
            }
            let _ = events.send(Event::RecordingStopped);
        });
        let mut capture = Self {
            packets,
            sample_rate: 0,
            device: String::new(),
            error,
            stop,
            handle: Some(handle),
        };
        match ready_rx.recv().context("Microphone setup thread stopped")? {
            Ok((rate, device)) => {
                capture.sample_rate = rate;
                capture.device = device;
                Ok(capture)
            }
            Err(e) => bail!(e),
        }
    }

    pub fn check_error(&self) -> Result<()> {
        if let Some(e) = self.error.lock().unwrap().as_ref() {
            bail!("{e}");
        }
        Ok(())
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

struct Callback {
    tx: SyncSender<Vec<f32>>,
    events: Sender<Event>,
    error: Arc<Mutex<Option<String>>>,
    stop: Arc<AtomicBool>,
    cancel: Arc<AtomicBool>,
    channels: usize,
    rate: u32,
    pending: Vec<f32>,
    frames: usize,
}

impl Callback {
    fn packet(&mut self) {
        let samples = std::mem::take(&mut self.pending);
        self.frames += samples.len();
        let (rms, peak) = levels(&samples);
        let _ = self.events.send(Event::RecordingLevel {
            rms,
            peak,
            seconds: self.frames as f32 / self.rate as f32,
        });
        if matches!(
            self.tx.try_send(samples),
            Err(std::sync::mpsc::TrySendError::Full(_))
        ) && !self.cancel.load(Ordering::Relaxed)
        {
            *self.error.lock().unwrap() = Some("Live model cannot keep up: microphone queue exceeded 12 seconds. Stop and select a smaller model".into());
        }
    }
}

impl Drop for Callback {
    fn drop(&mut self) {
        // Preserve the last partial packet when stopping between callbacks.
        if !self.pending.is_empty() {
            self.packet();
        }
    }
}

fn build<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    mut state: Callback,
    on_error: impl FnMut(cpal::StreamError) + Send + 'static,
) -> Result<cpal::Stream>
where
    T: cpal::SizedSample,
    f32: cpal::FromSample<T>,
{
    Ok(device.build_input_stream(
        config,
        move |data: &[T], _| {
            if state.stop.load(Ordering::Relaxed) || state.cancel.load(Ordering::Relaxed) {
                return;
            }
            for frame in data.chunks_exact(state.channels) {
                let mono = frame
                    .iter()
                    .map(|v| <f32 as cpal::FromSample<T>>::from_sample_(*v))
                    .sum::<f32>()
                    / state.channels as f32;
                state.pending.push(if mono.is_finite() {
                    mono.clamp(-1.0, 1.0)
                } else {
                    0.0
                });
                if state.pending.len() >= (state.rate / 10) as usize {
                    state.packet();
                }
            }
        },
        on_error,
        None,
    )?)
}

pub fn levels(samples: &[f32]) -> (f32, f32) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }
    let sum = samples.iter().map(|s| s * s).sum::<f32>();
    (
        (sum / samples.len() as f32).sqrt(),
        samples.iter().map(|s| s.abs()).fold(0.0, f32::max),
    )
}

/// Stateful sinc resampling; removes filter delay once, flushes once, and
/// preserves the exact recording duration across packet boundaries.
pub struct Resample16k {
    inner: SincFixedIn<f32>,
    rate: u32,
    skip: usize,
    input_len: usize,
    output_len: usize,
}

impl Resample16k {
    pub fn new(rate: u32) -> Result<Self> {
        let inner = SincFixedIn::new(
            16_000.0 / rate as f64,
            1.0,
            SincInterpolationParameters {
                sinc_len: 128,
                f_cutoff: 0.95,
                interpolation: SincInterpolationType::Linear,
                oversampling_factor: 128,
                window: WindowFunction::BlackmanHarris2,
            },
            (rate / 10) as usize,
            1,
        )?;
        let skip = inner.output_delay();
        Ok(Self {
            inner,
            rate,
            skip,
            input_len: 0,
            output_len: 0,
        })
    }
    fn trim(&mut self, mut out: Vec<f32>) -> Vec<f32> {
        let skip = self.skip.min(out.len());
        out.drain(..skip);
        self.skip -= skip;
        let expected = (self.input_len as f64 * 16_000.0 / self.rate as f64).round() as usize;
        out.truncate(expected.saturating_sub(self.output_len));
        self.output_len += out.len();
        out
    }
    pub fn push(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        self.input_len += input.len();
        let mut out = if input.len() == self.inner.input_frames_next() {
            self.inner.process(&[input], None)?
        } else {
            self.inner.process_partial(Some(&[input]), None)?
        };
        Ok(self.trim(out.remove(0)))
    }
    pub fn finish(&mut self) -> Result<Vec<f32>> {
        let mut out = self.inner.process_partial::<&[f32]>(None, None)?;
        Ok(self.trim(out.remove(0)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_microphone_never_falls_back_to_another_input() {
        let names = vec!["MacBook Microphone".into(), "AirPods Microphone".into()];
        assert_eq!(input_index(&names, "AirPods Microphone").unwrap(), 1);
        assert!(input_index(&names[..1], "AirPods Microphone")
            .unwrap_err()
            .to_string()
            .contains("unavailable"));
        assert!(
            input_index(&["Same name".into(), "Same name".into()], "Same name")
                .unwrap_err()
                .to_string()
                .contains("More than one")
        );
    }

    #[test]
    fn callback_drop_flushes_last_samples_and_overflow_is_visible() {
        for capacity in [0, 2] {
            let (tx, rx) = sync_channel(capacity);
            let (events, _) = channel();
            let error = Arc::new(Mutex::new(None));
            let callback = Callback {
                tx,
                events,
                error: error.clone(),
                stop: Arc::new(AtomicBool::new(true)),
                cancel: Arc::new(AtomicBool::new(false)),
                channels: 1,
                rate: 16000,
                pending: vec![0.1, 0.2, 0.3],
                frames: 0,
            };
            drop(callback);
            if capacity == 0 {
                assert!(error
                    .lock()
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .contains("cannot keep up"));
            } else {
                assert_eq!(rx.recv().unwrap(), vec![0.1, 0.2, 0.3]);
                assert!(error.lock().unwrap().is_none());
            }
        }
    }

    #[test]
    fn meter_tracks_signal_and_silence() {
        assert_eq!(levels(&[]), (0.0, 0.0));
        assert_eq!(levels(&[0.0; 64]), (0.0, 0.0));
        assert_eq!(levels(&[0.5, -0.5]), (0.5, 0.5));
    }
    #[test]
    fn continuous_resampling_preserves_tail_and_duration() {
        for rate in [8_000, 16_000, 24_000, 32_000, 44_100, 48_000] {
            let count = rate as usize + 137;
            let input: Vec<_> = (0..count)
                .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / rate as f32).sin() * 0.5)
                .collect();
            let mut r = Resample16k::new(rate).unwrap();
            let mut out = Vec::new();
            for packet in input.chunks((rate / 10) as usize) {
                out.extend(r.push(packet).unwrap());
            }
            out.extend(r.finish().unwrap());
            assert_eq!(
                out.len(),
                (count as f64 * 16_000.0 / rate as f64).round() as usize
            );
            assert!(levels(&out[200..out.len() - 200]).0 > 0.3);
        }
    }
}
