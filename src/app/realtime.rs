use super::{App, Focus, WorkState};
use crate::transcribe::realtime::{unavailable_reason, LiveJob};
use std::collections::VecDeque;
use std::path::PathBuf;

#[derive(Default)]
pub struct LiveState {
    pub visible: bool,
    pub active: bool,
    pub recording: bool,
    pub stopping: bool,
    pub quit_after: bool,
    pub device: String,
    pub mode: String,
    pub levels: VecDeque<f32>,
    pub rms: f32,
    pub peak: f32,
    pub seconds: f32,
    pub decoded_seconds: f32,
    pub inference_rtf: Option<f32>,
    pub inference_idle: bool,
    pub pending_speech_started: Option<std::time::Instant>,
    pub first_text_pending_render: Option<std::time::Instant>,
    pub first_text_ms: Option<f32>,
}

impl LiveState {
    pub fn level(&mut self, rms: f32, peak: f32, seconds: f32) {
        self.rms = rms;
        self.peak = peak;
        self.seconds = seconds;
        self.levels.push_back(peak);
        if self.levels.len() > 120 {
            self.levels.pop_front();
        }
    }
    pub fn finish(&mut self) {
        self.active = false;
        self.recording = false;
        self.stopping = false;
    }
}

impl App {
    /// Called only after a successful terminal draw of the following transcript.
    pub fn live_text_rendered(&mut self) -> bool {
        if !self.live.visible
            || !self.transcript.follow
            || self.show_log
            || self.settings.open
            || self.picker.open
            || self.hub.open
            || self.audio_input.open
            || self.start_prompt.is_some()
            || self.settings.dir_picker.is_some()
            || self.settings.move_prompt.is_some()
            || self.speakers_input.is_some()
            || self.language_input.is_some()
            || self.naming.is_some()
            || self.tdrz_prompt.is_some()
        {
            return false;
        }
        let Some(started) = self.live.first_text_pending_render.take() else {
            return false;
        };
        let ms = started.elapsed().as_secs_f32() * 1000.0;
        self.live.first_text_ms = Some(ms);
        self.job_log.push(format!(
            "First text rendered after detected speech: {:.3}s{}",
            ms / 1000.0,
            if ms > 900.0 {
                " — 0.900s target exceeded"
            } else {
                ""
            }
        ));
        true
    }

    pub fn start_live(&mut self) {
        if self.busy() {
            self.status = "A job is already running — stop or cancel it first".into();
            return;
        }
        let Some(model) = self.library.selected.clone() else {
            self.status = "No live model selected — press m, or set the models folder in s".into();
            return;
        };
        if let Some(reason) = unavailable_reason(&model.path) {
            self.status = reason;
            return;
        }
        self.live = LiveState {
            visible: true,
            active: true,
            device: self.audio_input.selected.clone().unwrap_or_default(),
            ..Default::default()
        };
        self.show_log = false;
        self.focus = Focus::Transcript;
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        self.transcript.begin(
            PathBuf::from(format!("microphone-{stamp}")),
            model.name.clone(),
            None,
        );
        self.work = WorkState::LoadingModel {
            name: model.name.clone(),
            progress: -1,
        };
        self.status = format!("Preparing {} for microphone transcription…", model.name);
        self.job_log.push(format!(
            "live session: {} · language {} · microphone {} · diarization off",
            model.path.display(),
            self.config.language.as_deref().unwrap_or("auto"),
            self.audio_input
                .selected
                .as_deref()
                .unwrap_or("System default")
        ));
        self.transcriber.submit_live(LiveJob {
            model: model.path,
            language: self.config.language.clone(),
            input_device: self.audio_input.selected.clone(),
            noise_suppression: self.config.noise_suppression,
        });
    }

    pub fn stop_live(&mut self) {
        if !self.live.active || self.live.stopping {
            return;
        }
        self.live.stopping = true;
        self.transcriber.stop_recording();
        if !self.live.recording && matches!(self.work, WorkState::LoadingModel { .. }) {
            self.transcriber.request_cancel();
        }
        self.status = "Stopping microphone and finishing remaining speech…".into();
        self.job_log
            .push("recording stop requested — drain remaining audio and export");
    }

    pub fn request_quit(&mut self) {
        if self.live.active {
            self.live.quit_after = true;
            self.stop_live();
        } else {
            self.should_quit = true;
        }
    }
}
