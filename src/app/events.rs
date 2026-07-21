//! Reactions to worker-thread events: the WorkState machine, the job
//! log (every event leaves a line), and the auto-export on completion.

use crate::app::{App, WorkState};
use crate::format::clock_time;
use crate::transcribe::Event;

/// One updating log line per phase: `label: [#####·····] 47%`.
fn pct_line(label: &str, p: i32) -> String {
    const WIDTH: i32 = 20;
    let filled = (p.clamp(0, 100) * WIDTH / 100) as usize;
    format!(
        "{label}: [{}{}] {p}%",
        "#".repeat(filled),
        "·".repeat(WIDTH as usize - filled)
    )
}

impl App {
    pub fn handle_event(&mut self, event: Event) {
        match event {
            Event::LoadingModel(name) => {
                self.job_log.push(format!("loading model {name}"));
                self.status = format!("Loading {name}…");
                self.work = WorkState::LoadingModel { name, progress: 0 };
            }
            Event::LoadProgress(p) => {
                if p >= 0 {
                    self.job_log
                        .push_tagged("load-pct", &pct_line("loading model", p));
                }
                if let WorkState::LoadingModel { progress, .. } = &mut self.work {
                    *progress = p;
                }
            }
            Event::ModelReady { load_secs } => {
                self.job_log
                    .push(format!("model ready in {load_secs:.1}s"));
                self.status = format!("Model loaded in {load_secs:.1}s");
            }
            Event::Unloading => {
                self.job_log.push("releasing model memory");
                self.work = WorkState::UnloadingModel;
                self.status = "Releasing model memory…".into();
            }
            Event::Unloaded => {
                self.job_log.push("model unloaded — memory freed");
                self.work = WorkState::Idle;
                self.status = "Model unloaded — memory freed".into();
            }
            Event::Decoding => {
                self.work = WorkState::Decoding { progress: -1 };
                let is_video = self
                    .transcript
                    .source
                    .as_deref()
                    .map(crate::audio::is_video_file)
                    .unwrap_or(false);
                self.status = if is_video {
                    "Extracting audio from video…".into()
                } else {
                    "Decoding audio…".into()
                };
                self.job_log.push(self.status.clone());
            }
            Event::DecodeProgress(p) => {
                if p >= 0 {
                    self.job_log
                        .push_tagged("decode-pct", &pct_line("extracting audio", p));
                }
                if let WorkState::Decoding { progress } = &mut self.work {
                    *progress = p;
                }
            }
            Event::AudioInfo { duration_secs } => {
                self.job_log
                    .push(format!("audio duration: {duration_secs:.1}s"));
                self.transcript.duration_secs = Some(duration_secs);
                self.work = WorkState::Transcribing { progress: 0 };
                self.status = format!("Transcribing {duration_secs:.0}s of audio…");
            }
            Event::Progress(p) => {
                if p >= 0 {
                    self.job_log
                        .push_tagged("transcribe-pct", &pct_line("transcribing", p));
                }
                if let WorkState::Transcribing { progress } = &mut self.work {
                    *progress = p;
                }
            }
            Event::EngineLog(line) => {
                self.job_log.push(&line);
                // The status line mirrors the latest line while work is
                // in flight
                if self.busy() {
                    self.status = line;
                }
            }
            Event::EngineHeartbeat(secs) => {
                // Liveness while a silent subprocess works; consecutive
                // beats coalesce into one updating line
                let msg = format!("engine working… {secs}s elapsed (model computing)");
                self.job_log.push_tagged("engine-heartbeat", &msg);
                if self.busy() {
                    self.status = msg;
                }
            }
            Event::Segment(seg) => {
                self.job_log.push(format!(
                    "segment [{} → {}] {}",
                    clock_time(seg.start_ms),
                    clock_time(seg.end_ms),
                    seg.text.trim()
                ));
                self.transcript.segments.push(seg);
            }
            Event::SegmentsFinal(segments) => {
                self.job_log.push(format!(
                    "diarization: speaker labels applied to {} segments",
                    segments.len()
                ));
                self.transcript.segments = segments;
            }
            Event::Done {
                elapsed_secs,
                audio_secs,
                language,
            } => {
                self.work = WorkState::Idle;
                let rtf = elapsed_secs / audio_secs.max(0.001);
                let lang = language.as_deref().unwrap_or("?").to_string();
                self.transcript.language = language;
                let spoken = if self.transcript.segments.iter().any(|s| s.speaker.is_some()) {
                    " · 2 speakers labeled"
                } else {
                    ""
                };
                let base = format!(
                    "Done in {elapsed_secs:.1}s ({rtf:.2}× realtime, lang: {lang}{spoken})"
                );
                // Exports run automatically after every successful transcription
                self.status = if self.transcript.segments.is_empty() {
                    format!("{base} — no speech found, nothing to export")
                } else {
                    match self.export() {
                        Ok(folder) => format!("{base} — exported to {}", folder.display()),
                        Err(e) => format!("{base} — export FAILED: {e}"),
                    }
                };
                self.job_log.push(self.status.clone());
            }
            Event::Cancelled => {
                self.job_log.push("job cancelled");
                self.work = WorkState::Idle;
                self.status = "Cancelled".into();
            }
            Event::Error(msg) => {
                self.job_log.push(format!("ERROR: {msg}"));
                self.work = WorkState::Idle;
                self.status = format!("Error: {msg}");
            }
        }
    }
}
