//! Live MLX transcription: persistent model process, bounded microphone
//! capture, native streaming where available, speech windows otherwise.
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::{Event, Segment};
use crate::audio::capture::{levels, Capture, Resample16k};
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

pub struct LiveJob {
    pub model: PathBuf,
    pub language: Option<String>,
    pub input_device: Option<String>,
}

/// Reject incompatible conversions before loading weights or opening the mic.
pub fn unavailable_reason(model: &Path) -> Option<String> {
    if !model.is_dir() {
        return Some("Live mode needs an MLX ASR model folder (Qwen3-ASR or Voxtral Realtime). Press m to select one".into());
    }
    if model
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_lowercase()
        .contains("forcedaligner")
    {
        return Some("Forced aligners require an existing transcript; select a Qwen3-ASR or Voxtral Realtime model for recording".into());
    }
    let config: Value = match std::fs::read(model.join("config.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
    {
        Some(config) => config,
        None => return Some("Live model has no readable config.json".into()),
    };
    match config["model_type"].as_str() {
        Some("qwen3_forced_aligner") => Some("Forced aligners cannot transcribe microphone audio; select an ASR model".into()),
        None if config.get("multimodal").is_some() => Some("This Voxtral conversion uses voxmlx, not mlx-audio. Select the Voxtral Realtime 2602 4-bit or fp16 folder".into()),
        _ => None,
    }
}

struct Runner {
    child: Child,
    messages: Receiver<String>,
    readers: Vec<std::thread::JoinHandle<()>>,
    native: bool,
    language: Option<String>,
}

impl Runner {
    fn start(
        job: &LiveJob,
        events: &Sender<Event>,
        cancel: &Arc<AtomicBool>,
    ) -> Result<Option<Self>> {
        let Some(python) = super::backend::live_python(events, cancel)? else {
            return Ok(None);
        };
        let child = Command::new(python)
            .args(["-u", "-c", include_str!("realtime.py")])
            .arg(&job.model)
            .args(super::mlx_language(&job.model, job.language.as_deref()))
            .env("HF_HUB_OFFLINE", "1")
            .env("TOKENIZERS_PARALLELISM", "false")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("Starting live MLX engine")?;
        let (tx, messages) = channel();
        let mut runner = Self {
            child,
            messages,
            readers: Vec::new(),
            native: false,
            language: job.language.clone(),
        };
        let stdout = runner.child.stdout.take().unwrap();
        runner.readers.push(std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if tx.send(line).is_err() {
                    break;
                }
            }
        }));
        let stderr = runner.child.stderr.take().unwrap();
        let log = events.clone();
        runner.readers.push(std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                let clean: String = line
                    .chars()
                    .filter(|c| !c.is_control())
                    .take(2000)
                    .collect();
                if !clean.trim().is_empty() {
                    let _ = log.send(Event::EngineLog(clean));
                }
            }
        }));
        if !runner.wait_for("ready", events, cancel)? {
            return Ok(None);
        }
        Ok(Some(runner))
    }

    fn wait_for(
        &mut self,
        target: &str,
        events: &Sender<Event>,
        cancel: &Arc<AtomicBool>,
    ) -> Result<bool> {
        let started = Instant::now();
        let mut heartbeat = Instant::now();
        loop {
            if cancel.load(Ordering::SeqCst) {
                return Ok(false);
            }
            match self.messages.recv_timeout(Duration::from_millis(50)) {
                Ok(line) => {
                    let msg: Value =
                        serde_json::from_str(&line).context("Invalid live engine response")?;
                    let kind = msg["type"].as_str().unwrap_or_default();
                    match kind {
                        "error" => bail!(
                            "{}",
                            msg["message"].as_str().unwrap_or("Live inference failed")
                        ),
                        "ready" => {
                            self.native = msg["native"].as_bool().unwrap_or(false);
                            // Native Voxtral auto-detects and does not expose
                            // language metadata; do not export an ignored hint.
                            if self.native {
                                self.language = None;
                            }
                            let _ = events.send(Event::ModelReady {
                                load_secs: msg["load_secs"].as_f64().unwrap_or(0.0) as f32,
                            });
                        }
                        "partial" | "segment" => {
                            let seg = Segment {
                                start_ms: (msg["start"].as_f64().unwrap_or(0.0) * 1000.0) as i64,
                                end_ms: (msg["end"].as_f64().unwrap_or(0.0) * 1000.0) as i64,
                                text: msg["text"].as_str().unwrap_or_default().into(),
                                speaker: None,
                            };
                            let _ = events.send(if kind == "partial" {
                                Event::LivePartial(seg)
                            } else {
                                Event::Segment(seg)
                            });
                        }
                        "ack" => {
                            if let Some(language) = msg["language"].as_str() {
                                self.language = Some(language.into());
                            }
                        }
                        _ => bail!("Unknown live engine message: {kind}"),
                    }
                    if kind == target {
                        return Ok(true);
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    if heartbeat.elapsed() >= Duration::from_secs(5) {
                        let _ = events.send(Event::EngineHeartbeat(started.elapsed().as_secs()));
                        heartbeat = Instant::now();
                    }
                }
                Err(RecvTimeoutError::Disconnected) => {
                    bail!("Live MLX engine exited unexpectedly; see the job log (l) for details")
                }
            }
        }
    }

    fn audio(
        &mut self,
        samples: &[f32],
        start: f64,
        end: f64,
        finish: bool,
        events: &Sender<Event>,
        cancel: &Arc<AtomicBool>,
    ) -> Result<bool> {
        if cancel.load(Ordering::SeqCst) {
            return Ok(false);
        }
        let stdin = self
            .child
            .stdin
            .as_mut()
            .context("Live engine input closed")?;
        serde_json::to_writer(
            &mut *stdin,
            &json!({"samples": samples, "start": start, "end": end, "finish": finish}),
        )?;
        stdin.write_all(b"\n")?;
        stdin.flush()?;
        let complete = self.wait_for("ack", events, cancel)?;
        if complete {
            let _ = events.send(Event::LiveProgress {
                seconds: end as f32,
            });
        }
        Ok(complete)
    }
}

impl Drop for Runner {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        for reader in self.readers.drain(..) {
            let _ = reader.join();
        }
    }
}

/// A pause ends an utterance after 500 ms; continuous speech is bounded
/// at four seconds. All-silence windows skip inference to avoid hallucinations.
#[derive(Default)]
struct SpeechWindow {
    samples: Vec<f32>,
    voiced: bool,
    quiet: usize,
    start: usize,
}

impl SpeechWindow {
    fn push(&mut self, samples: &[f32]) -> bool {
        let speech = levels(samples).0 >= 0.003;
        self.voiced |= speech;
        self.quiet = if speech {
            0
        } else {
            self.quiet + samples.len()
        };
        self.samples.extend_from_slice(samples);
        self.samples.len() >= 64_000
            || (self.voiced && self.samples.len() >= 16_000 && self.quiet >= 8_000)
    }
    fn take(&mut self) -> (Vec<f32>, f64, f64, bool) {
        let samples = std::mem::take(&mut self.samples);
        let start = self.start;
        self.start += samples.len();
        let voiced = self.voiced;
        self.voiced = false;
        self.quiet = 0;
        (
            samples,
            start as f64 / 16_000.0,
            self.start as f64 / 16_000.0,
            voiced,
        )
    }
}

pub(super) fn run(
    job: &LiveJob,
    events: &Sender<Event>,
    cancel: &Arc<AtomicBool>,
    stop: &Arc<AtomicBool>,
) -> Result<()> {
    if let Some(reason) = unavailable_reason(&job.model) {
        bail!(reason);
    }
    let _ = events.send(Event::LoadingModel(
        job.model
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into(),
    ));
    let Some(mut runner) = Runner::start(job, events, cancel)? else {
        let _ = events.send(Event::Cancelled);
        return Ok(());
    };
    if cancel.load(Ordering::SeqCst) || stop.load(Ordering::SeqCst) {
        let _ = events.send(Event::Cancelled);
        return Ok(());
    }
    let started = Instant::now();
    let capture = Capture::open(
        events.clone(),
        cancel.clone(),
        stop.clone(),
        job.input_device.clone(),
    )?;
    let mode = if runner.native {
        "continuous streaming"
    } else {
        "speech windows · up to 4s"
    };
    let _ = events.send(Event::RecordingStarted {
        device: capture.device.clone(),
        mode: mode.into(),
    });
    let mut resampler = Resample16k::new(capture.sample_rate)?;
    let mut window = SpeechWindow::default();
    let mut count = 0usize;
    let mut native_audio = Vec::new();
    let mut last_input = Instant::now();
    let result = (|| -> Result<()> {
        loop {
            if cancel.load(Ordering::SeqCst) {
                return Ok(());
            }
            capture.check_error()?;
            match capture.packets.recv_timeout(Duration::from_millis(50)) {
                Ok(input) => {
                    last_input = Instant::now();
                    let audio = resampler.push(&input)?;
                    count += audio.len();
                    if runner.native {
                        native_audio.extend(audio);
                        // A 500 ms feed amortizes encoder overhead; the level
                        // meter still updates independently every 100 ms.
                        if native_audio.len() < 8000 {
                            continue;
                        }
                        if !runner.audio(
                            &native_audio,
                            (count - native_audio.len()) as f64 / 16_000.0,
                            count as f64 / 16_000.0,
                            false,
                            events,
                            cancel,
                        )? {
                            return Ok(());
                        }
                        native_audio.clear();
                    } else if window.push(&audio) {
                        let (audio, start, end, voiced) = window.take();
                        if voiced && !runner.audio(&audio, start, end, false, events, cancel)? {
                            return Ok(());
                        }
                        if !voiced {
                            let _ = events.send(Event::LiveProgress {
                                seconds: end as f32,
                            });
                        }
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    if last_input.elapsed() > Duration::from_secs(10) {
                        bail!("No audio received from the microphone for 10 seconds. Check macOS Microphone permission and the selected input device");
                    }
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
        capture.check_error()?;
        let tail = resampler.finish()?;
        count += tail.len();
        if runner.native {
            native_audio.extend(tail);
            runner.audio(
                &native_audio,
                (count - native_audio.len()) as f64 / 16_000.0,
                count as f64 / 16_000.0,
                true,
                events,
                cancel,
            )?;
        } else {
            window.push(&tail);
            let (audio, start, end, voiced) = window.take();
            runner.audio(
                if voiced { &audio } else { &[] },
                start,
                end,
                true,
                events,
                cancel,
            )?;
        }
        Ok(())
    })();
    drop(capture); // Always release mic before completion or error reaches UI.
    result?;
    if cancel.load(Ordering::SeqCst) {
        let _ = events.send(Event::Cancelled);
    } else {
        let _ = events.send(Event::Done {
            elapsed_secs: started.elapsed().as_secs_f32(),
            audio_secs: count as f32 / 16_000.0,
            language: runner.language.clone(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn model_compatibility_explains_aligner_and_voxmlx_formats() {
        let root = std::env::temp_dir().join(format!("live-model-check-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let qwen = root.join("Qwen3-ASR-0.6B");
        let aligner = root.join("Qwen3-ForcedAligner-0.6B");
        let vox = root.join("Voxtral-6bit");
        for dir in [&qwen, &aligner, &vox] {
            std::fs::create_dir_all(dir).unwrap();
        }
        std::fs::write(qwen.join("config.json"), r#"{"model_type":"qwen3_asr"}"#).unwrap();
        std::fs::write(aligner.join("config.json"), r#"{"model_type":"qwen3_asr"}"#).unwrap();
        std::fs::write(vox.join("config.json"), r#"{"multimodal":{}}"#).unwrap();
        assert!(unavailable_reason(&qwen).is_none());
        assert!(unavailable_reason(&aligner)
            .unwrap()
            .contains("existing transcript"));
        assert!(unavailable_reason(&vox).unwrap().contains("voxmlx"));
        assert!(unavailable_reason(&root.join("whisper.bin"))
            .unwrap()
            .contains("MLX"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn speech_window_keeps_final_audio_and_absolute_timestamps() {
        let mut w = SpeechWindow::default();
        for _ in 0..9 {
            assert!(!w.push(&[0.05; 1600]));
        }
        for _ in 0..4 {
            assert!(!w.push(&[0.0; 1600]));
        }
        assert!(w.push(&[0.0; 1600]));
        let (audio, start, end, voiced) = w.take();
        assert_eq!(audio.len(), 22400);
        assert_eq!(start, 0.0);
        assert_eq!(end, 1.4);
        assert!(voiced);
        w.push(&[0.1; 800]);
        let (tail, start, end, voiced) = w.take();
        assert_eq!(tail.len(), 800);
        assert_eq!(start, 1.4);
        assert_eq!(end, 1.45);
        assert!(voiced);
    }
    #[test]
    fn silence_is_bounded_and_does_not_invoke_inference() {
        let mut w = SpeechWindow::default();
        for _ in 0..39 {
            assert!(!w.push(&[0.0; 1600]));
        }
        assert!(w.push(&[0.0; 1600]));
        assert!(!w.take().3);
    }
}
