//! Reactions to worker-thread events: the WorkState machine and the
//! auto-export on completion.

use crate::app::{App, WorkState};
use crate::transcribe::Event;

impl App {
    pub fn handle_event(&mut self, event: Event) {
        match event {
            Event::LoadingModel(name) => {
                self.status = format!("Loading {name} (Metal)…");
                self.work = WorkState::LoadingModel { name, progress: 0 };
            }
            Event::LoadProgress(p) => {
                if let WorkState::LoadingModel { progress, .. } = &mut self.work {
                    *progress = p;
                }
            }
            Event::ModelReady { load_secs } => {
                self.status = format!("Model loaded in {load_secs:.1}s");
            }
            Event::Unloading => {
                self.work = WorkState::UnloadingModel;
                self.status = "Releasing model memory…".into();
            }
            Event::Unloaded => {
                self.work = WorkState::Idle;
                self.status = "Model unloaded — memory freed".into();
            }
            Event::Decoding => {
                self.work = WorkState::Decoding;
                self.status = "Decoding audio…".into();
            }
            Event::AudioInfo { duration_secs } => {
                self.transcript.duration_secs = Some(duration_secs);
                self.work = WorkState::Transcribing { progress: 0 };
                self.status = format!("Transcribing {duration_secs:.0}s of audio…");
            }
            Event::Progress(p) => {
                if let WorkState::Transcribing { progress } = &mut self.work {
                    *progress = p;
                }
            }
            Event::Segment(seg) => {
                self.transcript.segments.push(seg);
            }
            Event::SegmentsFinal(segments) => {
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
            }
            Event::Cancelled => {
                self.work = WorkState::Idle;
                self.status = "Cancelled".into();
            }
            Event::Error(msg) => {
                self.work = WorkState::Idle;
                self.status = format!("Error: {msg}");
            }
        }
    }
}
