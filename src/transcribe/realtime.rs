//! Live MLX transcription: persistent model process, bounded microphone
//! capture, native streaming where available, speech windows otherwise.
use std::collections::VecDeque;
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
    speed: InferenceSpeed,
    /// Native feed size selected to meet the live text latency target.
    feed_samples: Option<usize>,
    last_ack: Value,
    last_metrics_log: Instant,
    capture_started: Option<Instant>,
}

#[derive(Default)]
struct InferenceSpeed {
    recent: VecDeque<(f64, f64)>,
    last_seconds: f64,
    observed_audio: f64,
    best_batch: Option<(f64, f64)>, // audio seconds, measured RTF
}

impl InferenceSpeed {
    fn observe(&mut self, audio_seconds: f64, inference_seconds: f64) -> f32 {
        self.last_seconds = inference_seconds;
        // Initial native prefill can be unusually cheap or expensive. Compare
        // batch efficiency only after at least four seconds have been fed.
        if self.observed_audio >= 4.0 {
            let rtf = inference_seconds / audio_seconds;
            if self.best_batch.is_none_or(|(_, best)| rtf < best) {
                self.best_batch = Some((audio_seconds, rtf));
            }
        }
        self.observed_audio += audio_seconds;
        self.recent.push_back((audio_seconds, inference_seconds));
        if self.recent.len() > 8 {
            self.recent.pop_front();
        }
        let (audio, elapsed) = self
            .recent
            .iter()
            .fold((0.0, 0.0), |(a, t), (da, dt)| (a + da, t + dt));
        (elapsed / audio) as f32
    }

    fn batch_samples(&self) -> usize {
        if let (Some((best_seconds, best_rtf)), Some((audio, elapsed))) =
            (self.best_batch, self.recent.back())
        {
            // Bigger batches can be slower (notably Voxtral); fall back to
            // measured efficient chunks instead of expanding without limit.
            if elapsed / audio > best_rtf * 1.1 {
                return (best_seconds * 16_000.0).clamp(8000.0, 64_000.0) as usize;
            }
        }
        // Allow 20% headroom relative to the last observed inference cost.
        // Bounds control latency/memory; they are not model speed estimates.
        (self.last_seconds * 1.25 * 16_000.0).clamp(8000.0, 64_000.0) as usize
    }
}

fn catch_up_audio(
    mut packets: impl Iterator<Item = Vec<f32>>,
    resampler: &mut Resample16k,
    audio: &mut Vec<f32>,
    max_samples: usize,
) -> Result<()> {
    // Coalesce queued packets without waiting for new input. Leave room for
    // one 100 ms packet so each native request stays at or below four seconds.
    while audio.len() + 1600 <= max_samples {
        let Some(input) = packets.next() else { break };
        audio.extend(resampler.push(&input)?);
    }
    Ok(())
}

fn native_send_len(buffered: usize, target: Option<usize>) -> Option<usize> {
    match target {
        // The first resampled packet is slightly short due to filter delay.
        // Send it now rather than adding a full 100ms packet of avoidable lag.
        Some(samples) if samples <= 1600 => (buffered > 0).then_some(buffered),
        Some(samples) => (buffered >= samples).then_some(samples),
        None => (buffered >= 8000).then_some(buffered),
    }
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
            .env("TRANSCRIBE_STT_LIVE_METRICS", "1")
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
            speed: InferenceSpeed::default(),
            feed_samples: None,
            last_ack: Value::Null,
            last_metrics_log: Instant::now(),
            capture_started: None,
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
        if runner.native {
            // Throughput-optimal 1–4s batches cannot satisfy sub-second text
            // latency. Live native feeds have a 100ms latency ceiling.
            runner.feed_samples = Some(1600);
            if let Some(samples) = runner.feed_samples {
                let _ = events.send(Event::EngineLog(format!(
                    "Using low-latency live feed: {}s",
                    samples as f64 / 16_000.0,
                )));
            }
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
                            let _ = events.send(Event::EngineLog(format!(
                                "Live engine: runtime={} · layout={} · context limit={}s · model delay={}ms",
                                msg["runtime"],
                                msg["optimization"]["layout"]
                                    .as_str()
                                    .unwrap_or("unchanged"),
                                msg["session_limit_seconds"],
                                msg["transcription_delay_ms"],
                            )));
                            // Native Voxtral auto-detects and does not expose
                            // language metadata; do not export an ignored hint.
                            if self.native {
                                self.language = None;
                            }
                            let _ = events.send(Event::ModelReady {
                                load_secs: msg["load_secs"].as_f64().unwrap_or(0.0) as f32,
                            });
                        }
                        "speech_start" => {
                            if let (Some(capture), Some(seconds)) = (
                                self.capture_started,
                                msg["start"].as_f64().filter(|s| s.is_finite() && *s >= 0.0),
                            ) {
                                if let Ok(offset) = Duration::try_from_secs_f64(seconds) {
                                    if let Some(at) = capture.checked_add(offset) {
                                        let _ = events.send(Event::LiveSpeechStarted { at });
                                    }
                                }
                            }
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
                            self.last_ack = msg.clone();
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
        let started = Instant::now();
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
            if self.last_metrics_log.elapsed() >= Duration::from_secs(10) || finish {
                let engine = self.last_ack["inference_seconds"].as_f64().unwrap_or(0.0);
                let memory = self.last_ack["active_gpu_bytes"].as_u64().unwrap_or(0);
                let _ = events.send(Event::EngineLog(format!(
                    "Live timing at {end:.1}s: {:.2}s audio · {:.3}s round trip · {engine:.3}s engine · context {}s · resets {} · active GPU {:.0} MiB",
                    samples.len() as f64 / 16_000.0,
                    started.elapsed().as_secs_f64(),
                    self.last_ack["session_seconds"],
                    self.last_ack["closed_sessions"],
                    memory as f64 / 1_048_576.0,
                )));
                self.last_metrics_log = Instant::now();
            }
            let inference_active = self.last_ack["inference_active"].as_bool().unwrap_or(true);
            if !inference_active {
                self.speed = InferenceSpeed::default();
            }
            let rtf = if !finish && !samples.is_empty() && inference_active {
                Some(self.speed.observe(
                    samples.len() as f64 / 16_000.0,
                    started.elapsed().as_secs_f64(),
                ))
            } else {
                None
            };
            let _ = events.send(Event::LiveProgress {
                seconds: end as f32,
                rtf,
                idle: self.last_ack["idle"].as_bool().unwrap_or(false),
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
    run_with_capture(job, events, cancel, stop, || {
        Capture::open(
            events.clone(),
            cancel.clone(),
            stop.clone(),
            job.input_device.clone(),
        )
    })
}

fn run_with_capture(
    job: &LiveJob,
    events: &Sender<Event>,
    cancel: &Arc<AtomicBool>,
    stop: &Arc<AtomicBool>,
    open_capture: impl FnOnce() -> Result<Capture>,
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
    let capture = open_capture()?;
    runner.capture_started = Some(capture.started_at);
    let mode = if let Some(samples) = runner.feed_samples {
        format!("streaming · {}s feed", samples as f64 / 16_000.0)
    } else if runner.native {
        "continuous streaming · adaptive".into()
    } else {
        "speech windows · up to 4s".into()
    };
    let _ = events.send(Event::RecordingStarted {
        device: capture.device.clone(),
        mode,
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
                        // Forward short resampled packets immediately in low-latency mode.
                        if native_send_len(native_audio.len(), runner.feed_samples).is_none() {
                            continue;
                        }
                        let before = native_audio.len();
                        catch_up_audio(
                            std::iter::from_fn(|| capture.packets.try_recv().ok()),
                            &mut resampler,
                            &mut native_audio,
                            runner
                                .feed_samples
                                .unwrap_or_else(|| runner.speed.batch_samples()),
                        )?;
                        count += native_audio.len() - before;
                        let send_len =
                            native_send_len(native_audio.len(), runner.feed_samples).unwrap();
                        let start = count - native_audio.len();
                        if !runner.audio(
                            &native_audio[..send_len],
                            start as f64 / 16_000.0,
                            (start + send_len) as f64 / 16_000.0,
                            false,
                            events,
                            cancel,
                        )? {
                            return Ok(());
                        }
                        native_audio.drain(..send_len);
                    } else if window.push(&audio) {
                        let (audio, start, end, voiced) = window.take();
                        if voiced && !runner.audio(&audio, start, end, false, events, cancel)? {
                            return Ok(());
                        }
                        if !voiced {
                            let _ = events.send(Event::LiveProgress {
                                seconds: end as f32,
                                rtf: None,
                                idle: true,
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
    #[ignore = "requires MLX, local model/fixture, GPU and paced idle interval"]
    fn paced_idle_first_render() {
        use crate::app::{App, LiveState};
        use ratatui::{backend::TestBackend, Terminal};
        use std::sync::mpsc::sync_channel;
        let model = PathBuf::from(std::env::var("TRANSCRIBE_TEST_MODEL").expect("model path"));
        let audio = PathBuf::from(std::env::var("TRANSCRIBE_TEST_AUDIO").expect("fixture path"));
        let idle: u64 = std::env::var("TRANSCRIBE_TEST_IDLE_SECS")
            .unwrap_or("480".into())
            .parse()
            .unwrap();
        let report_path =
            PathBuf::from(std::env::var("TRANSCRIBE_TEST_REPORT").expect("report path"));
        let mut speech = crate::audio::load_media(&audio).unwrap().samples;
        speech.truncate(24 * 16000);
        let onset = speech
            .chunks(320)
            .position(|frame| levels(frame).0 >= 0.003)
            .unwrap();
        let expected_onset = idle as f64 + onset as f64 * 0.02;
        let scratch =
            std::env::temp_dir().join(format!("transcribe-render-check-{}", std::process::id()));
        std::fs::create_dir_all(&scratch).unwrap();
        std::env::set_var("TRANSCRIBE_STT_CONFIG", scratch.join("config.toml"));
        std::env::set_var("TRANSCRIBE_STT_BENCHMARK", "1");
        let (tx, rx) = channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        let worker_stop = stop.clone();
        let capture_start = Arc::new(std::sync::Mutex::new(None::<Instant>));
        let worker_start = capture_start.clone();
        let worker = std::thread::spawn(move || {
            let capture_events = tx.clone();
            let job = LiveJob {
                model,
                language: Some("es".into()),
                input_device: None,
            };
            run_with_capture(&job, &tx, &worker_cancel, &worker_stop, || {
                let (packets_tx, packets_rx) = sync_channel(1200);
                let started = Instant::now();
                *worker_start.lock().unwrap() = Some(started);
                let producer_stop = worker_stop.clone();
                let producer_cancel = worker_cancel.clone();
                let handle = std::thread::spawn(move || {
                    let count = idle as usize * 10 + speech.len().div_ceil(1600);
                    for packet in 0..count {
                        let deadline =
                            started + Duration::from_secs_f64((packet + 1) as f64 / 10.0);
                        std::thread::sleep(deadline.saturating_duration_since(Instant::now()));
                        if producer_stop.load(Ordering::Relaxed)
                            || producer_cancel.load(Ordering::Relaxed)
                        {
                            break;
                        }
                        let input = if packet < idle as usize * 10 {
                            vec![0.0; 4800]
                        } else {
                            let first = (packet - idle as usize * 10) * 1600;
                            speech[first..(first + 1600).min(speech.len())]
                                .iter()
                                .flat_map(|sample| [*sample; 3])
                                .collect::<Vec<_>>()
                        };
                        let (rms, peak) = levels(&input);
                        let _ = capture_events.send(Event::RecordingLevel {
                            rms,
                            peak,
                            seconds: (packet + 1) as f32 / 10.0,
                        });
                        if packets_tx.try_send(input).is_err() {
                            break;
                        }
                    }
                    let _ = capture_events.send(Event::RecordingStopped);
                });
                Ok(Capture::scripted(
                    packets_rx,
                    48000,
                    started,
                    worker_stop.clone(),
                    handle,
                ))
            })
        });
        let (unused_tx, _) = channel();
        let mut app = App::new(scratch.clone(), None, crate::transcribe::spawn(unused_tx));
        app.output_dir = scratch.join("out");
        app.live = LiveState {
            active: true,
            visible: true,
            ..Default::default()
        };
        app.show_log = false;
        app.transcript.begin(
            "synthetic-live-input".into(),
            "Voxtral render check".into(),
            None,
        );
        let mut terminal = Terminal::new(TestBackend::new(120, 40)).unwrap();
        let started = Instant::now();
        let wall_started = std::time::SystemTime::now();
        let mut next_progress = 60;
        let mut last_draw = Instant::now();
        let mut measured = None;
        let mut display_text = String::new();
        while started.elapsed() < Duration::from_secs(idle + 120)
            && wall_started.elapsed().unwrap_or_default() < Duration::from_secs(idle + 120)
        {
            let mut dirty = false;
            while let Ok(event) = rx.try_recv() {
                app.handle_event(event);
                dirty = true;
            }
            if dirty || last_draw.elapsed() >= Duration::from_millis(100) {
                crate::ui::clamp_transcript(&mut app, ratatui::layout::Rect::new(0, 0, 120, 40));
                terminal.draw(|frame| crate::ui::draw(frame, &app)).unwrap();
                if app.live_text_rendered() {
                    let rendered = Instant::now();
                    let origin = capture_start.lock().unwrap().unwrap();
                    measured = Some(rendered.duration_since(origin).as_secs_f64() - expected_onset);
                    display_text = app
                        .transcript
                        .partial
                        .as_ref()
                        .or_else(|| app.transcript.segments.last())
                        .map(|s| s.text.trim().to_string())
                        .unwrap_or_default();
                    let screen = terminal
                        .backend()
                        .buffer()
                        .content
                        .iter()
                        .map(|c| c.symbol())
                        .collect::<String>();
                    assert!(screen.contains(&display_text));
                    break;
                }
                last_draw = Instant::now();
            }
            app.stats.refresh_if_due();
            if app.live.seconds as u64 >= next_progress {
                println!(
                    "{}s capture: idle={}, text={}",
                    next_progress,
                    app.live.inference_idle,
                    app.transcript.segments.len()
                );
                next_progress += 60;
            }
            if worker.is_finished() {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        cancel.store(true, Ordering::SeqCst);
        stop.store(true, Ordering::SeqCst);
        let worker_result = worker.join().unwrap();
        app.transcriber.shutdown();
        let report = json!({"test": "paced 48kHz capture -> Rust resampler -> Python Voxtral -> Rust UI -> terminal test-backend draw",
            "idle_seconds": idle, "first_text": display_text, "onset_to_completed_draw_seconds": measured,
            "ui_reported_first_text_ms": app.live.first_text_ms, "passed": measured.is_some_and(|s| s <= 0.9),
            "worker_result": format!("{worker_result:?}"), "job_log": app.job_log.lines});
        std::fs::write(&report_path, serde_json::to_string_pretty(&report).unwrap()).unwrap();
        println!(
            "First rendered text: {display_text:?}; onset-to-draw: {measured:?}s; report: {}",
            report_path.display()
        );
        std::env::remove_var("TRANSCRIBE_STT_CONFIG");
        std::env::remove_var("TRANSCRIBE_STT_BENCHMARK");
        std::fs::remove_dir_all(scratch).unwrap();
        worker_result.unwrap();
        assert!(
            measured.is_some_and(|s| s <= 0.9),
            "render latency exceeded 0.9s; see {}",
            report_path.display()
        );
    }

    #[test]
    fn low_latency_feed_sends_the_first_short_resampled_packet_immediately() {
        for rate in [16000, 48000] {
            let mut resampler = Resample16k::new(rate).unwrap();
            let audio = resampler.push(&vec![0.1; rate as usize / 10]).unwrap();
            assert!(!audio.is_empty());
            assert!(audio.len() < 1600);
            assert_eq!(native_send_len(audio.len(), Some(1600)), Some(audio.len()));
            assert_eq!(native_send_len(audio.len(), Some(16000)), None);
        }
        assert_eq!(native_send_len(0, Some(1600)), None);
    }

    #[test]
    fn catch_up_batches_preserve_audio_and_bound_each_request() {
        let mut packets = (0..100).map(|_| vec![0.1; 1600]).peekable();
        let mut resampler = Resample16k::new(16000).unwrap();
        let mut total = 0;
        let mut requests = 0;
        while packets.peek().is_some() {
            let mut audio = Vec::new();
            catch_up_audio(&mut packets, &mut resampler, &mut audio, 64000).unwrap();
            assert!(!audio.is_empty());
            assert!(audio.len() <= 64000);
            total += audio.len();
            requests += 1;
        }
        total += resampler.finish().unwrap().len();
        assert_eq!(total, 160000);
        assert_eq!(requests, 3);
    }

    #[test]
    fn inference_speed_weights_audio_duration_and_recovers_after_stall() {
        let mut speed = InferenceSpeed::default();
        assert_eq!(speed.batch_samples(), 8000);
        assert_eq!(speed.observe(0.5, 4.0), 8.0);
        assert_eq!(speed.batch_samples(), 64000);
        assert!((speed.observe(4.0, 2.0) - 6.0 / 4.5).abs() < 0.001);
        assert_eq!(speed.batch_samples(), 40000);
        for _ in 0..8 {
            speed.observe(4.0, 0.2);
        }
        assert_eq!(speed.batch_samples(), 8000);
        assert!((speed.observe(4.0, 0.2) - 0.05).abs() < 0.001);
    }

    #[test]
    fn larger_slower_batches_fall_back_to_measured_efficient_size() {
        let mut speed = InferenceSpeed::default();
        speed.observe(4.0, 0.1); // Exclude startup from batch comparisons.
        speed.observe(1.0, 0.95);
        speed.observe(4.0, 4.8);
        assert_eq!(speed.batch_samples(), 16000);
        speed.observe(1.0, 0.4);
        assert_eq!(speed.batch_samples(), 8000);
    }

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
