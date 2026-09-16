//! Application state and update logic. `App` owns the sub-state for each
//! UI concern; the per-concern logic lives in the submodules below, and
//! `ui` renders it all read-only.

mod audio_input;
mod browser;
mod drop;
mod events;
mod hub_state;
mod job_log;
mod keys;
mod library;
mod naming;
mod realtime;
mod settings;
mod transcript;

pub use audio_input::AudioInputState;
pub(crate) use browser::file_name;
pub use browser::{scroll_window, FileBrowser, FileEntry};
pub use drop::DropDetector;
pub use hub_state::{DefaultEntry, Download, EntryKind, HubList, HubState, RepoView};
pub use job_log::JobLog;
pub use library::{ModelLibrary, ModelPicker};
pub use naming::{NamingRow, SpeakerNaming};
pub use realtime::LiveState;
pub use settings::{DirPicker, DirRow, DirTarget, MovePrompt, SettingsRow, SettingsUi};
pub use transcript::TranscriptState;

use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};

use anyhow::Result;

use crate::config::{self, Config};
use crate::diarize::{self, DiarizeMethod};
use crate::export::TranscriptDoc;
use crate::hub::HubEvent;
use crate::models::{self, ModelFile};
use crate::stats::ProcStats;
use crate::transcribe::{Job, Transcriber};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    Files,
    Transcript,
}

#[derive(Clone, Debug, PartialEq)]
pub enum WorkState {
    Idle,
    Recording,
    LoadingModel {
        name: String,
        progress: i32,
    },
    /// Resident model being released after a model switch
    UnloadingModel,
    Decoding {
        progress: i32,
    },
    Transcribing {
        progress: i32,
    },
}

/// Yes/No dialog offering to download the tdrz model: opened whenever a
/// tdrz-based strategy is selected (or a job needs it) while the model
/// is missing. "Yes" opens Model management with the download running.
pub struct TdrzDownloadPrompt {
    pub yes_selected: bool,
}

/// Yes/No dialog shown before a transcription job starts.
pub struct StartPrompt {
    pub audio: PathBuf,
    pub yes_selected: bool,
    /// Diarization method the job will use, resolved when the prompt
    /// opens (the requirement check may probe the Python runtime, so it
    /// must not run per rendered frame).
    pub diarize: DiarizeMethod,
    /// What diarization still needs downloaded; None = ready (or off)
    pub diarize_note: Option<String>,
}

pub struct App {
    pub should_quit: bool,
    pub focus: Focus,
    /// Render-loop counter driving the loading spinner
    pub tick: usize,
    pub status: String,
    pub work: WorkState,
    pub stats: ProcStats,
    /// Persisted settings — the single source for diarize/language/split
    pub config: Config,
    /// Where transcript exports are written
    pub output_dir: PathBuf,

    pub browser: FileBrowser,
    pub library: ModelLibrary,
    pub picker: ModelPicker,
    pub settings: SettingsUi,
    pub transcript: TranscriptState,
    pub hub: HubState,
    /// Some while asking "transcribe this file?" — every job start goes
    /// through this confirmation
    pub start_prompt: Option<StartPrompt>,
    /// Some while the post-transcription speaker-naming dialog is open (`n`)
    pub naming: Option<SpeakerNaming>,
    /// Some(text) while the diarization speaker count is being edited (`p`)
    pub speakers_input: Option<String>,
    /// Some(text) while the transcription language is being edited (`i`)
    pub language_input: Option<String>,
    /// Some while offering to download the missing tdrz model
    pub tdrz_prompt: Option<TdrzDownloadPrompt>,
    /// Timestamped record of everything since the first file was loaded
    pub job_log: JobLog,
    /// Right pane shows the job log instead of the transcript (`l`)
    pub show_log: bool,
    pub live: LiveState,
    pub audio_input: AudioInputState,
    pub stream_server: Option<crate::stream_service::StreamServer>,

    pub transcriber: Transcriber,
    hub_tx: Sender<HubEvent>,
    hub_rx: Receiver<HubEvent>,
}

impl App {
    pub fn new(start_dir: PathBuf, model: Option<PathBuf>, transcriber: Transcriber) -> Self {
        let app_config = config::load();
        config::apply_hf_token(&app_config);
        let models_dir = config::resolve_models_dir(&app_config);
        let output_dir = config::resolve_output_dir(&app_config);
        let models_list = models::scan_models(&models_dir);
        let selected_model = model
            .map(|p| ModelFile {
                name: file_name(&p),
                size_bytes: if p.is_dir() {
                    models::dir_size(&p)
                } else {
                    std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0)
                },
                is_dir: p.is_dir(),
                path: p,
            })
            .or_else(|| {
                models::pick_default(&models_list, app_config.default_model.as_deref()).cloned()
            });

        let (hub_tx, hub_rx) = channel();
        Self {
            should_quit: false,
            focus: Focus::Files,
            tick: 0,
            status: String::from(
                "Select a file and press Enter, or drop an audio/video file onto the window",
            ),
            work: WorkState::Idle,
            stats: ProcStats::new(),
            config: app_config,
            output_dir,
            browser: FileBrowser::new(start_dir),
            library: ModelLibrary {
                dir: models_dir,
                models: models_list,
                selected: selected_model,
            },
            picker: ModelPicker::default(),
            settings: SettingsUi::new(),
            transcript: TranscriptState::new(),
            hub: HubState::new(),
            start_prompt: None,
            naming: None,
            speakers_input: None,
            language_input: None,
            tdrz_prompt: None,
            job_log: JobLog::new(),
            // The log view is the default; `l` switches to the transcript
            show_log: true,
            live: LiveState::default(),
            audio_input: AudioInputState::default(),
            stream_server: None,
            transcriber,
            hub_tx,
            hub_rx,
        }
    }

    pub fn busy(&self) -> bool {
        self.work != WorkState::Idle
    }

    pub fn start_stream_service(&mut self, port: u16) -> Result<()> {
        if self.stream_server.is_some() {
            return Ok(());
        }
        let server =
            crate::stream_service::StreamServer::start(port, self.transcriber.text_stream())?;
        self.status = format!(
            "Text service: http://{}/events · ws://{}/ws",
            server.address(),
            server.address()
        );
        self.job_log.push(&self.status);
        self.stream_server = Some(server);
        Ok(())
    }

    pub fn toggle_stream_service(&mut self) {
        if self.stream_server.take().is_some() {
            self.status = "Text service stopped".into();
            self.job_log.push(&self.status);
        } else if let Err(error) = self.start_stream_service(crate::stream_service::DEFAULT_PORT) {
            self.status = format!("Error starting text service: {error:#}");
            self.job_log.push(&self.status);
        }
    }

    /// The method the configured strategy resolves to for the current
    /// model selection — pure and cheap, safe per rendered frame.
    pub fn resolved_diarize_method(&self) -> DiarizeMethod {
        self.config
            .diarize
            .resolve(self.library.selected.as_ref().map(|m| m.name.as_str()))
    }

    /// The diarization plan for the current model selection: the method
    /// the configured strategy resolves to, plus what it still needs
    /// downloaded (None = ready). May probe the Python runtime — call on
    /// user actions, not per rendered frame.
    pub fn diarize_plan(&self) -> (DiarizeMethod, Option<String>) {
        let method = self.resolved_diarize_method();
        let note = diarize::download_note(
            method,
            &self.library.models,
            &self.library.dir,
            &self.config.diarize_models,
        );
        (method, note)
    }

    /// Every job start goes through here: opens the confirmation prompt
    /// instead of starting immediately. `start_transcription` runs once
    /// the prompt is confirmed.
    pub fn request_transcription(&mut self, audio: PathBuf) {
        if self.busy() && self.work != WorkState::UnloadingModel {
            self.status = "A job is already running — press c to cancel it first".into();
            return;
        }
        self.job_log
            .push(format!("file selected: {}", audio.display()));
        self.status = format!("Transcribe {}?", file_name(&audio));
        let (diarize, diarize_note) = self.diarize_plan();
        self.start_prompt = Some(StartPrompt {
            audio,
            yes_selected: true,
            diarize,
            diarize_note,
        });
    }

    pub fn start_transcription(&mut self, audio: PathBuf) {
        // A pending unload is fine — the job queues right behind it
        if self.busy() && self.work != WorkState::UnloadingModel {
            self.status = "A job is already running — press c to cancel it first".into();
            return;
        }
        // Resolve the diarization strategy against the selected model;
        // the tdrz method transcribes with the tdrz model itself
        let method = self.resolved_diarize_method();
        // The pyannote model is gated: without a token the job would only
        // fail later at the runner's token gate, so stop here, open the
        // consent page, and point at the settings row that stores a token.
        if method == DiarizeMethod::Pyannote && !crate::diarize::pyannote::hf_token_available() {
            crate::hub::open_consent_page(crate::diarize::pyannote::MODEL);
            self.status = format!(
                "{} is gated — accept its terms on the page just opened in your browser, \
                 then set your token in s → HF token (or press d for another strategy)",
                crate::diarize::pyannote::MODEL
            );
            self.job_log
                .push("pyannote blocked: no Hugging Face token — consent page opened");
            return;
        }
        let model = if method == DiarizeMethod::Tdrz {
            let selected_tdrz = self
                .library
                .selected
                .clone()
                .filter(|m| models::is_tdrz(&m.name));
            match selected_tdrz.or_else(|| self.find_tdrz_model()) {
                Some(m) => m,
                None => {
                    self.status = format!(
                        "Diarization (tinydiarize) needs the tdrz model — download {} ({}) \
                         via s → Model management, or press d for another strategy",
                        diarize::TDRZ_FILE,
                        diarize::TDRZ_SIZE
                    );
                    // Offer the download instead of dead-ending the job
                    self.tdrz_prompt = Some(TdrzDownloadPrompt { yes_selected: true });
                    return;
                }
            }
        } else {
            match self.library.selected.clone() {
                Some(m) => m,
                None => {
                    self.status = "No model found — download one via s → Model management".into();
                    return;
                }
            }
        };
        if let Some(note) = diarize::download_note(
            method,
            &self.library.models,
            &self.library.dir,
            &self.config.diarize_models,
        ) {
            self.job_log.push(format!("diarization {note}"));
        }
        self.live = LiveState::default();
        self.transcript.begin(
            audio.clone(),
            model.name.clone(),
            method.export_note().map(String::from),
        );
        self.work = WorkState::LoadingModel {
            name: model.name.clone(),
            progress: 0,
        };
        self.job_log.push(format!(
            "job start: {} · model {} · language {} · split {} · diarize {} · formats {}",
            file_name(&audio),
            model.name,
            self.config.language.as_deref().unwrap_or("auto"),
            self.config.split_mode.label(),
            method.label(),
            self.config
                .export_formats
                .iter()
                .map(|f| f.key())
                .collect::<Vec<_>>()
                .join("/"),
        ));
        self.status = format!("Starting {}", file_name(&audio));
        self.transcriber.submit(Job {
            model: model.path,
            audio,
            diarize: method,
            diarize_models: self.config.diarize_models.clone(),
            diarize_speakers: self.config.diarize_speakers,
            language: self.config.language.clone(),
            split_mode: self.config.split_mode,
        });
    }

    /// Wipe the job log (`x`). The cleared log starts with a marker line
    /// so the pane confirms what happened.
    pub fn clear_log(&mut self) {
        self.job_log.clear();
        self.job_log.push("log cleared");
        self.status = "Job log cleared".into();
    }

    /// Snapshot the job log to `<output_dir>/logs/log_<timestamp>.log`.
    /// Available at any time via `e`, including mid-job.
    pub fn export_log(&mut self) {
        match self.job_log.export_to(&self.output_dir.join("logs")) {
            Ok(path) => {
                self.job_log
                    .push(format!("log exported: {}", path.display()));
                self.status = format!("Log exported to {}", path.display());
            }
            Err(e) => self.status = format!("Log export failed: {e}"),
        }
    }

    /// Write the configured export formats into
    /// `<output_dir>/<source>_<timestamp>/` and return that folder. Runs
    /// automatically after each successful transcription.
    pub fn export(&mut self) -> Result<PathBuf> {
        let (Some(audio), false) = (
            self.transcript.source.clone(),
            self.transcript.segments.is_empty(),
        ) else {
            anyhow::bail!("nothing to export yet");
        };
        let doc = TranscriptDoc {
            segments: &self.transcript.segments,
            source_name: file_name(&audio),
            duration_secs: self.transcript.duration_secs,
            language: self.transcript.language.as_deref(),
            model_name: self.transcript.model_name.as_deref(),
            diarization: self.transcript.diarization.as_deref(),
            speaker_names: Some(&self.transcript.speaker_names),
        };
        let written = doc.write(&self.output_dir, &self.config.export_formats)?;
        for path in &written {
            self.job_log.push(format!("exported: {}", path.display()));
        }
        Ok(written
            .first()
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| self.output_dir.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hub::HubEvent;
    use crate::transcribe::{Event, Segment};
    use crossterm::event::KeyCode;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    /// Serializes tests that touch TRANSCRIBE_STT_CONFIG or HF_TOKEN:
    /// both are process-global, so concurrent setters would make one
    /// test's config::save land in another test's (soon-deleted) folder.
    /// Readers need it too — every App::new loads the config from that
    /// path and exports HF_TOKEN from it, so an unlocked test_app can
    /// pick up (and re-export) a locked test's freshly saved token.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn lock_env() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn tempdir() -> PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static UNIQUE: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "transcribe-stt-app-{}-{}",
            std::process::id(),
            UNIQUE.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn test_app(start_dir: PathBuf) -> App {
        let (tx, rx) = std::sync::mpsc::channel();
        std::mem::forget(rx); // keep worker sends from erroring
        App::new(start_dir, None, crate::transcribe::spawn(tx))
    }

    #[test]
    fn browser_lists_dirs_first_then_media_only() {
        let dir = tempdir();
        std::fs::create_dir(dir.join("z_subdir")).unwrap();
        std::fs::write(dir.join("b.wav"), b"x").unwrap();
        std::fs::write(dir.join("a.mp4"), b"x").unwrap();
        std::fs::write(dir.join("notes.txt"), b"x").unwrap();
        std::fs::write(dir.join(".hidden.wav"), b"x").unwrap();

        let _guard = lock_env(); // App::new reads the config path and exports HF_TOKEN
        let app = test_app(dir.clone());
        let labels: Vec<String> = app.browser.entries.iter().map(|e| e.label()).collect();
        assert_eq!(labels, vec!["../", "z_subdir/", "a.mp4 \u{29c9}", "b.wav"]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn every_worker_event_lands_in_the_job_log() {
        let dir = tempdir();
        let _guard = lock_env(); // App::new reads the config path and exports HF_TOKEN
        let mut app = test_app(dir.clone());
        app.handle_event(Event::LoadingModel("m.bin".into()));
        app.handle_event(Event::LoadProgress(50));
        app.handle_event(Event::ModelReady { load_secs: 1.2 });
        app.handle_event(Event::Decoding);
        app.handle_event(Event::DecodeProgress(40));
        app.handle_event(Event::AudioInfo { duration_secs: 5.0 });
        app.handle_event(Event::Progress(10));
        app.handle_event(Event::EngineLog("mlx: fetching model".into()));
        app.handle_event(Event::Segment(Segment {
            start_ms: 0,
            end_ms: 1000,
            text: "hi".into(),
            speaker: None,
        }));
        app.handle_event(Event::Error("boom".into()));

        let joined: Vec<&str> = app.job_log.lines.iter().map(|s| s.as_str()).collect();
        let joined = joined.join("\n");
        for expected in [
            "loading model m.bin",
            "loading model: [##########··········] 50%",
            "model ready in 1.2s",
            "Decoding audio…",
            "extracting audio: [########············] 40%",
            "audio duration: 5.0s",
            "transcribing: [##··················] 10%",
            "mlx: fetching model",
            "segment [00:00 → 00:01] hi",
            "ERROR: boom",
        ] {
            assert!(
                joined.contains(expected),
                "missing {expected:?} in:\n{joined}"
            );
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn transcription_asks_for_confirmation_first() {
        use crossterm::event::KeyModifiers;
        let dir = tempdir();
        std::fs::write(dir.join("clip.wav"), b"x").unwrap();
        let _guard = lock_env(); // App::new reads the config path and exports HF_TOKEN
        let mut app = test_app(dir.clone());

        app.request_transcription(dir.join("clip.wav"));
        let prompt = app.start_prompt.as_ref().expect("prompt opens");
        assert!(prompt.yes_selected); // starting is the default choice
        assert!(!app.busy(), "no job before confirmation");

        // n cancels without starting
        app.on_key(KeyCode::Char('n'), KeyModifiers::NONE);
        assert!(app.start_prompt.is_none());
        assert!(!app.busy());
        assert!(app.status.contains("cancelled"));

        // Enter on "No" also cancels
        app.request_transcription(dir.join("clip.wav"));
        app.on_key(KeyCode::Left, KeyModifiers::NONE); // flip to No
        app.on_key(KeyCode::Enter, KeyModifiers::NONE);
        assert!(app.start_prompt.is_none());
        assert!(!app.busy());

        // Enter on "Yes" proceeds into the normal start path (which
        // fails fast here: no model is selected)
        app.request_transcription(dir.join("clip.wav"));
        app.library.selected = None;
        app.config.diarize = crate::diarize::DiarizeStrategy::Off;
        app.on_key(KeyCode::Enter, KeyModifiers::NONE);
        assert!(app.start_prompt.is_none());
        assert!(app.status.contains("No model"), "status: {}", app.status);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn diarize_strategy_cycles_with_status_explaining_each() {
        use crate::diarize::DiarizeStrategy;
        let _guard = lock_env(); // cycling persists the config
        let dir = tempdir();
        std::env::set_var("TRANSCRIBE_STT_CONFIG", dir.join("config.toml"));
        let mut app = test_app(dir.clone());
        app.library.dir = dir.clone(); // empty: no tdrz model on disk
        app.library.models.clear();
        app.config.diarize = DiarizeStrategy::Off;

        app.cycle_diarize();
        assert_eq!(app.config.diarize, DiarizeStrategy::Auto);
        assert!(app.status.contains("Auto"), "{}", app.status);

        // tdrz with no tdrz model: the status names the exact download
        // and a dialog offers to fetch it right away
        app.cycle_diarize();
        assert_eq!(app.config.diarize, DiarizeStrategy::Tdrz);
        assert!(
            app.status.contains(crate::diarize::TDRZ_FILE),
            "{}",
            app.status
        );
        assert!(
            app.status.contains(crate::diarize::TDRZ_SIZE),
            "{}",
            app.status
        );
        assert!(app.tdrz_prompt.is_some(), "download offer should open");
        // declining leaves the strategy set but downloads nothing
        app.on_key(
            crossterm::event::KeyCode::Esc,
            crossterm::event::KeyModifiers::NONE,
        );
        assert!(app.tdrz_prompt.is_none());
        assert!(app.hub.download.is_none());
        assert!(app.status.contains("skipped"), "{}", app.status);

        app.cycle_diarize();
        assert_eq!(app.config.diarize, DiarizeStrategy::Embedding);
        assert!(app.status.contains("embeddings"), "{}", app.status);

        app.cycle_diarize();
        assert_eq!(app.config.diarize, DiarizeStrategy::Pyannote);
        assert!(app.status.contains("Pyannote"), "{}", app.status);

        app.cycle_diarize();
        assert_eq!(app.config.diarize, DiarizeStrategy::Off);
        assert!(app.status.contains("OFF"), "{}", app.status);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn start_prompt_reports_the_diarize_plan() {
        use crate::diarize::{DiarizeMethod, DiarizeStrategy};
        let dir = tempdir();
        std::fs::write(dir.join("clip.wav"), b"x").unwrap();
        let _guard = lock_env(); // App::new reads the config path and exports HF_TOKEN
        let mut app = test_app(dir.clone());
        app.library.dir = dir.clone();
        app.library.models.clear();

        app.config.diarize = DiarizeStrategy::Off;
        app.request_transcription(dir.join("clip.wav"));
        let prompt = app.start_prompt.as_ref().unwrap();
        assert_eq!(prompt.diarize, DiarizeMethod::None);
        assert!(prompt.diarize_note.is_none());

        // tdrz strategy without the tdrz model: the prompt warns exactly
        // what must be downloaded before this can run
        app.start_prompt = None;
        app.config.diarize = DiarizeStrategy::Tdrz;
        app.request_transcription(dir.join("clip.wav"));
        let prompt = app.start_prompt.as_ref().unwrap();
        assert_eq!(prompt.diarize, DiarizeMethod::Tdrz);
        let note = prompt.diarize_note.as_ref().expect("download note");
        assert!(note.contains(crate::diarize::TDRZ_FILE), "{note}");
        assert!(note.contains(crate::diarize::TDRZ_SIZE), "{note}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn diarize_speaker_count_validates_and_persists() {
        let _guard = lock_env();
        let dir = tempdir();
        std::env::set_var("TRANSCRIBE_STT_CONFIG", dir.join("config.toml"));
        let mut app = test_app(dir.clone());

        app.set_diarize_speakers("3");
        assert_eq!(app.config.diarize_speakers, Some(3));
        assert!(app.status.contains("exactly 3"), "{}", app.status);

        // empty and zero mean auto-detect
        app.set_diarize_speakers("");
        assert_eq!(app.config.diarize_speakers, None);
        app.set_diarize_speakers("0");
        assert_eq!(app.config.diarize_speakers, None);

        // out-of-range input is rejected and keeps the old value
        app.set_diarize_speakers("2");
        app.set_diarize_speakers("99");
        assert_eq!(app.config.diarize_speakers, Some(2));
        assert!(app.status.contains("1–26"), "{}", app.status);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn speaker_count_hotkey_gates_on_the_strategy() {
        use crate::diarize::DiarizeStrategy;
        use crossterm::event::KeyModifiers;
        let _guard = lock_env(); // Enter persists the count
        let dir = tempdir();
        std::env::set_var("TRANSCRIBE_STT_CONFIG", dir.join("config.toml"));
        let mut app = test_app(dir.clone());
        app.library.selected = None;

        // Off: the key explains instead of opening the input
        app.config.diarize = DiarizeStrategy::Off;
        app.on_key(KeyCode::Char('p'), KeyModifiers::NONE);
        assert!(app.speakers_input.is_none());
        assert!(app.status.contains("diarization"), "{}", app.status);

        // TinyDiarize is fixed at 2 speakers — no count to set
        app.config.diarize = DiarizeStrategy::Tdrz;
        app.on_key(KeyCode::Char('p'), KeyModifiers::NONE);
        assert!(app.speakers_input.is_none());
        assert!(app.status.contains("2 speakers"), "{}", app.status);

        // A clustering strategy opens the input; digits only, Enter saves
        app.config.diarize = DiarizeStrategy::Embedding;
        app.on_key(KeyCode::Char('p'), KeyModifiers::NONE);
        assert_eq!(app.speakers_input.as_deref(), Some(""));
        app.on_key(KeyCode::Char('x'), KeyModifiers::NONE); // not a digit
        assert_eq!(app.speakers_input.as_deref(), Some(""));
        app.on_key(KeyCode::Char('3'), KeyModifiers::NONE);
        app.on_key(KeyCode::Enter, KeyModifiers::NONE);
        assert!(app.speakers_input.is_none());
        assert_eq!(app.config.diarize_speakers, Some(3));

        // Reopening prefills the saved count; Esc keeps it untouched
        app.on_key(KeyCode::Char('p'), KeyModifiers::NONE);
        assert_eq!(app.speakers_input.as_deref(), Some("3"));
        app.on_key(KeyCode::Esc, KeyModifiers::NONE);
        assert!(app.speakers_input.is_none());
        assert_eq!(app.config.diarize_speakers, Some(3));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn settings_language_picker_saves_cancels_and_renders_on_small_terminals() {
        use crossterm::event::KeyModifiers;
        use ratatui::{backend::TestBackend, Terminal};
        let _guard = lock_env();
        let dir = tempdir();
        std::env::set_var("TRANSCRIBE_STT_CONFIG", dir.join("config.toml"));
        let mut app = test_app(dir.clone());
        app.on_key(KeyCode::Char('s'), KeyModifiers::NONE);
        app.on_key(KeyCode::Down, KeyModifiers::NONE);
        assert_eq!(app.settings.selected, SettingsRow::Language);

        for (index, (code, label)) in config::INPUT_LANGUAGES.iter().enumerate() {
            app.on_key(KeyCode::Enter, KeyModifiers::NONE);
            for _ in 0..config::INPUT_LANGUAGES.len() {
                app.on_key(KeyCode::Up, KeyModifiers::NONE);
            }
            for _ in 0..index {
                app.on_key(KeyCode::Down, KeyModifiers::NONE);
            }
            for (width, height) in [(100, 30), (80, 16)] {
                let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
                terminal.draw(|f| crate::ui::draw(f, &app)).unwrap();
                let screen = terminal
                    .backend()
                    .buffer()
                    .content
                    .iter()
                    .map(|c| c.symbol())
                    .collect::<String>();
                for (_, name) in config::INPUT_LANGUAGES {
                    assert!(screen.contains(name), "missing {name} at {width}x{height}");
                }
            }
            app.on_key(KeyCode::Enter, KeyModifiers::NONE);
            let expected = if *code == "auto" { None } else { Some(*code) };
            assert_eq!(app.config.language.as_deref(), expected);
            assert_eq!(config::load().language.as_deref(), expected);
            assert!(app.settings.language_cursor.is_none());
            assert!(app.settings.open);
            let mut terminal = Terminal::new(TestBackend::new(80, 16)).unwrap();
            terminal.draw(|f| crate::ui::draw(f, &app)).unwrap();
            let screen = terminal
                .backend()
                .buffer()
                .content
                .iter()
                .map(|c| c.symbol())
                .collect::<String>();
            assert!(screen.contains(&format!("Input language: {label}")));
            assert!(screen.contains("HF token:"));
        }

        // Reopening selects the saved choice; moving then cancelling does not save.
        app.on_key(KeyCode::Enter, KeyModifiers::NONE);
        assert_eq!(app.settings.language_cursor, Some(5));
        app.on_key(KeyCode::Up, KeyModifiers::NONE);
        app.on_key(KeyCode::Esc, KeyModifiers::NONE);
        assert_eq!(config::load().language.as_deref(), Some("fr"));
        // Auto clears a previously pinned language, including on disk.
        app.on_key(KeyCode::Enter, KeyModifiers::NONE);
        for _ in 0..5 {
            app.on_key(KeyCode::Up, KeyModifiers::NONE);
        }
        app.on_key(KeyCode::Enter, KeyModifiers::NONE);
        assert_eq!(config::load().language, None);
        // Custom codes from the existing shortcut survive opening/cancelling.
        app.set_language("ja");
        app.on_key(KeyCode::Enter, KeyModifiers::NONE);
        app.on_key(KeyCode::Esc, KeyModifiers::NONE);
        assert_eq!(config::load().language.as_deref(), Some("ja"));
        let reopened = test_app(dir.clone());
        assert_eq!(reopened.config.language.as_deref(), Some("ja"));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn language_hotkey_edits_and_persists_the_code() {
        use crossterm::event::KeyModifiers;
        let _guard = lock_env(); // Enter persists the language
        let dir = tempdir();
        std::env::set_var("TRANSCRIBE_STT_CONFIG", dir.join("config.toml"));
        let mut app = test_app(dir.clone());

        // `i` opens the input; letters only, Enter saves a valid code
        app.on_key(KeyCode::Char('i'), KeyModifiers::NONE);
        assert_eq!(app.language_input.as_deref(), Some(""));
        app.on_key(KeyCode::Char('3'), KeyModifiers::NONE); // not a letter
        assert_eq!(app.language_input.as_deref(), Some(""));
        for c in "en".chars() {
            app.on_key(KeyCode::Char(c), KeyModifiers::NONE);
        }
        app.on_key(KeyCode::Enter, KeyModifiers::NONE);
        assert!(app.language_input.is_none());
        assert_eq!(app.config.language.as_deref(), Some("en"));
        assert_eq!(config::load().language.as_deref(), Some("en"));

        // Reopening prefills the saved code; Esc keeps it untouched
        app.on_key(KeyCode::Char('i'), KeyModifiers::NONE);
        assert_eq!(app.language_input.as_deref(), Some("en"));
        app.on_key(KeyCode::Esc, KeyModifiers::NONE);
        assert!(app.language_input.is_none());
        assert_eq!(app.config.language.as_deref(), Some("en"));

        // "auto" clears the pin back to detection
        app.on_key(KeyCode::Char('i'), KeyModifiers::NONE);
        app.on_key(KeyCode::Backspace, KeyModifiers::NONE);
        app.on_key(KeyCode::Backspace, KeyModifiers::NONE);
        for c in "auto".chars() {
            app.on_key(KeyCode::Char(c), KeyModifiers::NONE);
        }
        app.on_key(KeyCode::Enter, KeyModifiers::NONE);
        assert_eq!(app.config.language, None);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn hf_token_setting_persists_and_reaches_the_environment() {
        let _guard = lock_env(); // both files and HF_TOKEN are process-global
        let dir = tempdir();
        std::env::set_var("TRANSCRIBE_STT_CONFIG", dir.join("config.toml"));
        let shell_token = std::env::var_os("HF_TOKEN");
        std::env::remove_var("HF_TOKEN");
        let mut app = test_app(dir.clone());

        // Enter on the row opens the input; typing + Enter saves it
        app.settings.open = true;
        app.settings.selected = SettingsRow::HfToken;
        app.settings_key(KeyCode::Enter);
        assert_eq!(app.settings.hf_token_input.as_deref(), Some(""));
        for c in "hf_abc123".chars() {
            app.settings_key(KeyCode::Char(c));
        }
        app.settings_key(KeyCode::Enter);
        assert!(app.settings.hf_token_input.is_none());
        assert_eq!(app.config.hf_token.as_deref(), Some("hf_abc123"));
        // exported for the pyannote runner and authenticated downloads
        assert_eq!(std::env::var("HF_TOKEN").as_deref(), Ok("hf_abc123"));
        // and persisted to the config file
        assert_eq!(config::load().hf_token.as_deref(), Some("hf_abc123"));

        // Reopening prefills the token; emptying it clears everywhere
        app.settings_key(KeyCode::Enter);
        assert_eq!(app.settings.hf_token_input.as_deref(), Some("hf_abc123"));
        for _ in 0.."hf_abc123".len() {
            app.settings_key(KeyCode::Backspace);
        }
        app.settings_key(KeyCode::Enter);
        assert_eq!(app.config.hf_token, None);
        assert!(std::env::var_os("HF_TOKEN").is_none());
        assert_eq!(config::load().hf_token, None);

        if let Some(token) = shell_token {
            std::env::set_var("HF_TOKEN", token);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn gated_download_failure_names_the_consent_page() {
        let dir = tempdir();
        let _guard = lock_env(); // App::new reads the config path and exports HF_TOKEN
        let mut app = test_app(dir.clone());
        app.library.dir = dir.clone();
        app.open_hub();

        // A 401 from a gated repo: the info line sends the user to the
        // consent page (opened in the browser) and the token setting
        let mut d = fake_download("model.bin", &dir, Arc::new(AtomicBool::new(false)));
        d.repo = "pyannote/speaker-diarization-community-1".into();
        app.hub.download = Some(d);
        app.handle_hub_event(HubEvent::Failed {
            file: "model.bin".into(),
            error: "status code 401 for https://huggingface.co/x".into(),
        });
        assert!(app.hub.download.is_none());
        assert!(
            app.hub
                .info
                .contains("hf.co/pyannote/speaker-diarization-community-1"),
            "{}",
            app.hub.info
        );
        assert!(app.hub.info.contains("HF token"), "{}", app.hub.info);

        // Any other failure keeps the plain error message
        app.hub.download = Some(fake_download(
            "m2.bin",
            &dir,
            Arc::new(AtomicBool::new(false)),
        ));
        app.handle_hub_event(HubEvent::Failed {
            file: "m2.bin".into(),
            error: "connection closed early".into(),
        });
        assert!(
            app.hub.info.contains("connection closed early"),
            "{}",
            app.hub.info
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn hub_tdrz_row_downloads_or_selects_the_strategy() {
        use crate::diarize::DiarizeStrategy;
        let _guard = lock_env(); // Enter and Done persist the strategy
        let dir = tempdir();
        std::env::set_var("TRANSCRIBE_STT_CONFIG", dir.join("config.toml"));
        let mut app = test_app(dir.clone());
        app.library.dir = dir.clone();
        app.config.diarize = DiarizeStrategy::Off;
        app.open_hub();

        // The tdrz offer's "Yes" path selects the row about to download
        app.hub_select_tdrz_row();
        let entries = app.hub_default_entries();
        let row = &entries[app.hub.selected];
        assert_eq!(row.kind, crate::app::EntryKind::Diarize);
        assert_eq!(row.file, crate::diarize::TDRZ_FILE);
        assert!(!row.installed);
        drop(entries);

        // A finished tdrz download makes TinyDiarize the strategy
        let path = dir.join(crate::diarize::TDRZ_FILE);
        std::fs::write(&path, b"weights").unwrap();
        app.hub.download = Some(fake_download(
            crate::diarize::TDRZ_FILE,
            &dir,
            Arc::new(AtomicBool::new(false)),
        ));
        app.handle_hub_event(HubEvent::Done {
            file: crate::diarize::TDRZ_FILE.into(),
            path,
        });
        assert_eq!(app.config.diarize, DiarizeStrategy::Tdrz);
        assert!(app.hub.info.contains("TinyDiarize"), "{}", app.hub.info);

        // Installed + active now: Enter re-selects the strategy after a
        // detour through another one
        app.config.diarize = DiarizeStrategy::Embedding;
        app.hub_select_tdrz_row();
        let entries = app.hub_default_entries();
        let row = &entries[app.hub.selected];
        assert!(row.installed && !row.active);
        drop(entries);
        app.hub_key(KeyCode::Enter);
        assert_eq!(app.config.diarize, DiarizeStrategy::Tdrz);
        assert!(
            app.hub.download.is_none(),
            "no download for an installed model"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn naming_dialog_renames_speakers_and_reexports_on_close() {
        use crossterm::event::KeyModifiers;
        let dir = tempdir();
        let _guard = lock_env(); // App::new reads the config path and exports HF_TOKEN
        let mut app = test_app(dir.clone());
        app.output_dir = dir.join("out");

        // No diarized transcript yet → the dialog refuses to open
        app.open_speaker_naming();
        assert!(app.naming.is_none());
        assert!(app.status.contains("No speaker labels"), "{}", app.status);

        // A diarized transcript: two speakers, distinct longest samples
        app.transcript.source = Some(dir.join("call.wav"));
        app.transcript.segments = vec![
            Segment {
                start_ms: 0,
                end_ms: 5000,
                text: "the long intro".into(),
                speaker: Some(0),
            },
            Segment {
                start_ms: 5000,
                end_ms: 6000,
                text: "reply".into(),
                speaker: Some(1),
            },
        ];
        app.open_speaker_naming();
        let naming = app.naming.as_ref().expect("dialog opens");
        assert_eq!(naming.rows.len(), 2);
        assert_eq!(naming.rows[0].sample_text, "the long intro");

        // Enter opens the input; type a name; Enter saves it
        app.on_key(KeyCode::Enter, KeyModifiers::NONE);
        for c in "George".chars() {
            app.on_key(KeyCode::Char(c), KeyModifiers::NONE);
        }
        app.on_key(KeyCode::Enter, KeyModifiers::NONE);
        assert_eq!(
            app.transcript.speaker_names.get(&0).map(String::as_str),
            Some("George")
        );

        // Esc closes and re-exports with the name applied
        app.on_key(KeyCode::Esc, KeyModifiers::NONE);
        assert!(app.naming.is_none());
        assert!(app.status.contains("re-exported"), "{}", app.status);
        let exported: Vec<_> = std::fs::read_dir(dir.join("out"))
            .unwrap()
            .flatten()
            .collect();
        assert_eq!(exported.len(), 1);
        let md = std::fs::read_to_string(exported[0].path().join("call.llm.md")).unwrap();
        assert!(md.contains("George: the long intro"), "{md}");
        assert!(md.contains("Speaker B: reply"), "{md}");
        assert!(md.contains("speakers: Speaker A = George"), "{md}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn tdrz_job_without_the_model_explains_the_download() {
        use crate::diarize::DiarizeStrategy;
        let dir = tempdir();
        let _guard = lock_env(); // App::new reads the config path and exports HF_TOKEN
        let mut app = test_app(dir.clone());
        app.library.dir = dir.clone();
        app.library.models.clear();
        app.library.selected = None;
        app.config.diarize = DiarizeStrategy::Tdrz;

        app.start_transcription(dir.join("clip.wav"));
        assert!(!app.busy(), "job must not start without the tdrz model");
        assert!(
            app.status.contains(crate::diarize::TDRZ_FILE),
            "{}",
            app.status
        );
        assert!(app.status.contains("Model management"), "{}", app.status);
        // and instead of dead-ending, the download offer opens
        assert!(app.tdrz_prompt.is_some());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn export_format_toggle_keeps_at_least_one() {
        use crate::export::ExportFormat;
        let _guard = lock_env(); // toggling persists the config
        let dir = tempdir();
        std::env::set_var("TRANSCRIBE_STT_CONFIG", dir.join("config.toml"));
        let mut app = test_app(dir.clone());

        app.config.export_formats = vec![ExportFormat::Srt];
        // the last selected format cannot be removed
        app.toggle_export_format(ExportFormat::Srt);
        assert_eq!(app.config.export_formats, vec![ExportFormat::Srt]);
        assert!(app.status.contains("At least one"));

        // re-adding rebuilds canonical order regardless of toggle order
        app.toggle_export_format(ExportFormat::LlmMd);
        assert_eq!(
            app.config.export_formats,
            vec![ExportFormat::LlmMd, ExportFormat::Srt]
        );
        // and only the selected formats are written on export
        app.output_dir = dir.join("out");
        app.transcript.source = Some(dir.join("clip.wav"));
        app.transcript.segments.push(Segment {
            start_ms: 0,
            end_ms: 1000,
            text: "hi".into(),
            speaker: None,
        });
        let folder = app.export().unwrap();
        assert!(folder.join("clip.llm.md").is_file());
        assert!(folder.join("clip.srt").is_file());
        assert!(!folder.join("clip.segments.json").exists());
        assert!(!folder.join("clip.txt").exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn worker_events_drive_state_machine_and_auto_export() {
        let app_dir = tempdir();
        let _guard = lock_env(); // App::new reads the config path and exports HF_TOKEN
        let mut app = test_app(app_dir.clone());
        app.output_dir = app_dir.join("out");
        app.transcript.source = Some(app_dir.join("clip.wav"));

        app.handle_event(Event::LoadingModel("m.bin".into()));
        assert!(app.busy());

        app.handle_event(Event::AudioInfo {
            duration_secs: 10.0,
        });
        assert_eq!(app.work, WorkState::Transcribing { progress: 0 });
        assert_eq!(app.transcript.duration_secs, Some(10.0));

        app.handle_event(Event::Progress(40));
        assert_eq!(app.work, WorkState::Transcribing { progress: 40 });

        app.handle_event(Event::Segment(Segment {
            start_ms: 0,
            end_ms: 1000,
            text: "hi".into(),
            speaker: None,
        }));
        assert_eq!(app.transcript.segments.len(), 1);

        app.handle_event(Event::Done {
            elapsed_secs: 2.0,
            audio_secs: 10.0,
            language: Some("en".into()),
        });
        assert!(!app.busy());
        assert_eq!(app.transcript.language.as_deref(), Some("en"));
        assert!(app.status.contains("lang: en"));
        // completion auto-exported into <output_dir>/clip_<timestamp>/
        assert!(app.status.contains("exported to"), "status: {}", app.status);
        let exported: Vec<_> = std::fs::read_dir(app_dir.join("out"))
            .unwrap()
            .flatten()
            .collect();
        assert_eq!(exported.len(), 1);
        assert!(exported[0]
            .file_name()
            .to_string_lossy()
            .starts_with("clip_"));
        assert!(exported[0].path().join("clip.llm.md").is_file());
        std::fs::remove_dir_all(&app_dir).unwrap();
    }

    #[test]
    fn rendered_text_over_budget_is_reported_without_hiding_the_text() {
        use ratatui::{backend::TestBackend, Terminal};
        let _guard = lock_env();
        let dir = tempdir();
        let mut app = test_app(dir.clone());
        app.live = LiveState { visible: true, active: true, recording: true, ..Default::default() };
        app.show_log = false;
        app.handle_event(Event::LiveSpeechStarted {
            at: std::time::Instant::now() - Duration::from_millis(950),
        });
        app.handle_event(Event::LivePartial(Segment {
            start_ms: 0, end_ms: 1000, text: "Visible speech".into(), speaker: None,
        }));
        let mut terminal = Terminal::new(TestBackend::new(120, 40)).unwrap();
        crate::ui::clamp_transcript(&mut app, ratatui::layout::Rect::new(0, 0, 120, 40));
        terminal.draw(|frame| crate::ui::draw(frame, &app)).unwrap();
        assert!(app.live_text_rendered());
        terminal.draw(|frame| crate::ui::draw(frame, &app)).unwrap();
        let screen = terminal.backend().buffer().content.iter().map(|c| c.symbol()).collect::<String>();
        assert!(screen.contains("Visible speech"));
        assert!(screen.contains("> 0.9s"));
        assert!(app.job_log.lines.iter().any(|s| s.contains("target exceeded")));
        app.transcriber.shutdown();
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn model_picker_shows_live_fit_without_blocking_selection() {
        use crate::live_benchmark::LiveFit;
        use crossterm::event::KeyModifiers;
        use ratatui::{backend::TestBackend, Terminal};
        let _guard = lock_env();
        let dir = tempdir();
        std::env::set_var("TRANSCRIBE_STT_CONFIG", dir.join("config.toml"));
        for name in ["Fast", "Borderline", "TooSlow", "Untested", "ForcedAligner"] {
            let model = dir.join(name);
            std::fs::create_dir_all(&model).unwrap();
            std::fs::write(model.join("config.json"), r#"{"model_type":"qwen3_asr"}"#).unwrap();
            std::fs::write(model.join("model.safetensors"), "weights").unwrap();
        }
        let mut app = test_app(dir.clone());
        app.library.dir = dir.clone();
        app.on_key(KeyCode::Char('m'), KeyModifiers::NONE);
        assert!(app.picker.open);
        app.picker.live_fit.insert(dir.join("Fast"), LiveFit::Good { rtf: 0.06, batch: 4.0 });
        app.picker.live_fit.insert(dir.join("Borderline"), LiveFit::Borderline { rtf: 0.95, batch: 1.0 });
        app.picker.live_fit.insert(dir.join("TooSlow"), LiveFit::TooSlow { rtf: 6.16, batch: 0.5 });
        let mut terminal = Terminal::new(TestBackend::new(120, 40)).unwrap();
        terminal.draw(|frame| crate::ui::draw(frame, &app)).unwrap();
        let screen = terminal.backend().buffer().content.iter().map(|c| c.symbol()).collect::<String>();
        for text in ["Live: good fit", "Live: borderline", "Live: too slow for this Mac", "Live: unavailable", "Live: not benchmarked on this Mac", "6.16s/audio s"] {
            assert!(screen.contains(text), "missing {text}");
        }
        app.picker.selected = app.library.models.iter().position(|m| m.name == "TooSlow").unwrap();
        app.on_key(KeyCode::Enter, KeyModifiers::NONE);
        assert!(!app.picker.open);
        assert_eq!(app.library.selected.as_ref().unwrap().name, "TooSlow");
        app.transcriber.shutdown();
        std::env::remove_var("TRANSCRIBE_STT_CONFIG");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn live_transcript_follows_partials_and_exports_on_stop() {
        use crossterm::event::KeyModifiers;
        use ratatui::{backend::TestBackend, Terminal};
        let dir = tempdir();
        let _guard = lock_env();
        let mut app = test_app(dir.clone());
        app.output_dir = dir.join("out");
        app.live = LiveState {
            visible: true,
            active: true,
            ..Default::default()
        };
        app.focus = Focus::Transcript;
        app.show_log = false;
        app.transcript
            .begin("microphone-test".into(), "Qwen".into(), None);
        app.handle_event(Event::RecordingStarted {
            device: "Test microphone".into(),
            mode: "continuous streaming".into(),
        });
        for i in 0..50 {
            app.handle_event(Event::RecordingLevel {
                rms: 0.05,
                peak: 0.2,
                seconds: i as f32,
            });
            app.handle_event(Event::Segment(Segment {
                start_ms: i * 1000,
                end_ms: (i + 1) * 1000,
                text: format!("Sentence {i}, a streamed transcription."),
                speaker: None,
            }));
        }
        let rect = ratatui::layout::Rect::new(0, 0, 120, 30);
        app.handle_event(Event::LiveSpeechStarted {
            at: std::time::Instant::now() - Duration::from_millis(250),
        });
        app.handle_event(Event::LivePartial(Segment {
            start_ms: 50000,
            end_ms: 51000,
            text: "LATEST words appear live".into(),
            speaker: None,
        }));
        crate::ui::clamp_transcript(&mut app, rect);
        assert!(app.transcript.scroll > 0);
        app.handle_event(Event::LiveProgress {
            seconds: 45.0,
            rtf: Some(1.5),
            idle: false,
        });
        let mut terminal = Terminal::new(TestBackend::new(120, 30)).unwrap();
        terminal.draw(|f| crate::ui::draw(f, &app)).unwrap();
        let screen = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect::<String>();
        assert!(screen.contains("LATEST words appear live"));
        assert!(screen.contains("RECORDING"));
        assert!(screen.contains("First text:"));
        assert!(app.live.first_text_ms.is_none(), "receive time is not render time");
        app.show_log = true;
        assert!(!app.live_text_rendered(), "hidden text must not count as rendered");
        app.show_log = false;
        assert!(app.live_text_rendered());
        assert!(app.live.first_text_ms.unwrap() >= 250.0);
        assert!(app.live.pending_speech_started.is_none());
        assert!(screen.contains("1.50s/audio s"));
        assert!(screen.contains("falling behind"));
        app.handle_event(Event::LiveProgress {
            seconds: 46.0,
            rtf: None,
            idle: true,
        });
        assert_eq!(app.live.inference_rtf, Some(1.5));
        assert!(app.live.inference_idle);
        terminal.draw(|f| crate::ui::draw(f, &app)).unwrap();
        let idle_screen = terminal.backend().buffer().content.iter().map(|c| c.symbol()).collect::<String>();
        assert!(idle_screen.contains("Waiting for speech"));
        app.on_key(KeyCode::PageUp, KeyModifiers::NONE);
        let paused = app.transcript.scroll;
        app.handle_event(Event::LivePartial(Segment {
            start_ms: 50000,
            end_ms: 52000,
            text: "LATEST complete words".into(),
            speaker: None,
        }));
        crate::ui::clamp_transcript(&mut app, rect);
        assert_eq!(app.transcript.scroll, paused);
        assert!(!app.transcript.follow);
        app.on_key(KeyCode::Char('G'), KeyModifiers::NONE);
        crate::ui::clamp_transcript(&mut app, rect);
        assert!(app.transcript.follow);
        app.handle_event(Event::Segment(Segment {
            start_ms: 50000,
            end_ms: 52000,
            text: "LATEST complete words".into(),
            speaker: None,
        }));
        assert!(app.transcript.partial.is_none());
        app.on_key(KeyCode::Char('R'), KeyModifiers::NONE);
        assert!(app.live.stopping);
        assert!(app.busy());
        app.handle_event(Event::RecordingStopped);
        assert!(!app.live.recording);
        assert!(
            app.busy(),
            "remaining inference must finish before a new job"
        );
        app.handle_event(Event::Done {
            elapsed_secs: 53.0,
            audio_secs: 52.0,
            language: Some("es".into()),
        });
        assert!(!app.live.active);
        assert!(!app.busy());
        assert!(app.status.contains("exported to"), "{}", app.status);
        let folder = std::fs::read_dir(&app.output_dir)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let text = std::fs::read_to_string(folder.join("microphone-test.txt")).unwrap();
        assert_eq!(text.matches("LATEST complete words").count(), 1);
        for (width, height) in [(60, 18), (24, 8), (1, 1)] {
            let mut small = Terminal::new(TestBackend::new(width, height)).unwrap();
            crate::ui::clamp_transcript(&mut app, ratatui::layout::Rect::new(0, 0, width, height));
            small.draw(|f| crate::ui::draw(f, &app)).unwrap();
        }
        app.transcriber.shutdown();
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn microphone_picker_selects_bluetooth_without_starting_recording() {
        use crossterm::event::KeyModifiers;
        use ratatui::{backend::TestBackend, Terminal};
        let dir = tempdir();
        let _guard = lock_env();
        let mut app = test_app(dir.clone());
        app.audio_input.open = true;
        app.audio_input.update_devices(vec![
            "MacBook Microphone".into(),
            "Bluetooth Headset".into(),
        ]);
        app.on_key(KeyCode::Down, KeyModifiers::NONE);
        app.on_key(KeyCode::Down, KeyModifiers::NONE);
        let mut terminal = Terminal::new(TestBackend::new(110, 30)).unwrap();
        terminal.draw(|f| crate::ui::draw(f, &app)).unwrap();
        let screen = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect::<String>();
        assert!(screen.contains("Bluetooth Headset"));
        app.on_key(KeyCode::Enter, KeyModifiers::NONE);
        assert_eq!(
            app.audio_input.selected.as_deref(),
            Some("Bluetooth Headset")
        );
        assert!(!app.audio_input.open);
        assert!(!app.busy());
        app.live = LiveState {
            active: true,
            recording: true,
            ..Default::default()
        };
        app.work = WorkState::Recording;
        app.on_key(KeyCode::Char('a'), KeyModifiers::NONE);
        assert!(
            !app.audio_input.open,
            "input selection cannot change during capture"
        );
        assert!(app.status.contains("stop recording"));
        app.handle_event(Event::Cancelled);
        assert_eq!(
            app.audio_input.selected.as_deref(),
            Some("Bluetooth Headset")
        );
        app.audio_input.open = true;
        app.audio_input.cursor = 0;
        app.on_key(KeyCode::Enter, KeyModifiers::NONE);
        assert!(app.audio_input.selected.is_none());
        app.audio_input.open = true;
        app.audio_input.cursor = 2;
        app.on_key(KeyCode::Esc, KeyModifiers::NONE);
        assert!(
            app.audio_input.selected.is_none(),
            "Esc does not select the highlighted input"
        );
        app.transcriber.shutdown();
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn live_cancel_and_error_release_busy_state_and_keep_text() {
        let dir = tempdir();
        let _guard = lock_env();
        let mut app = test_app(dir.clone());
        for terminal in [Event::Cancelled, Event::Error("device disconnected".into())] {
            app.live = LiveState {
                active: true,
                recording: true,
                ..Default::default()
            };
            app.work = WorkState::Recording;
            app.transcript.partial = Some(Segment {
                start_ms: 0,
                end_ms: 1000,
                text: "kept".into(),
                speaker: None,
            });
            app.handle_event(terminal);
            assert!(!app.busy());
            assert!(!app.live.active);
            assert!(!app.live.recording);
            assert_eq!(app.transcript.partial.as_ref().unwrap().text, "kept");
        }
        app.transcriber.shutdown();
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn error_event_resets_to_idle() {
        let dir = tempdir();
        let _guard = lock_env(); // App::new reads the config path and exports HF_TOKEN
        let mut app = test_app(dir.clone());
        app.handle_event(Event::Decoding);
        assert!(app.busy());
        app.handle_event(Event::Error("boom".into()));
        assert!(!app.busy());
        assert!(app.status.starts_with("Error: boom"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn dropped_dir_changes_cwd_and_bad_drops_report() {
        let dir = tempdir();
        let sub = dir.join("inner");
        std::fs::create_dir(&sub).unwrap();
        std::fs::write(dir.join("doc.txt"), b"x").unwrap();
        let _guard = lock_env(); // App::new reads the config path and exports HF_TOKEN
        let mut app = test_app(dir.clone());

        app.handle_dropped_text(&sub.display().to_string());
        assert_eq!(app.browser.cwd, sub);

        app.handle_dropped_text(&dir.join("doc.txt").display().to_string());
        assert!(app.status.contains("Unsupported file type"));

        app.handle_dropped_text("/nonexistent/x.wav");
        assert!(app.status.contains("Not found"));

        app.handle_dropped_text("random words");
        assert!(app.status.contains("doesn't look like a file path"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn hub_modal_typing_navigation_and_esc_layers() {
        let dir = tempdir();
        let _guard = lock_env(); // App::new reads the config path and exports HF_TOKEN
        let mut app = test_app(dir.clone());
        app.library.dir = dir.clone(); // empty models folder → default view is the 3 suggestions

        app.open_hub();
        assert!(app.hub.open);
        // the curated list is embedded and parsed at startup
        assert_eq!(app.hub.suggested.len(), 3);

        // typing edits the search bar and arms the debounce (no search yet)
        for c in "whisper".chars() {
            app.hub_key(KeyCode::Char(c));
        }
        assert_eq!(app.hub.input, "whisper");
        assert!(app.hub.last_edit.is_some());
        assert!(!app.hub.searching);

        // Down clamps to the visible list, Up back to the top
        let len = app.hub_visible_list().len();
        for _ in 0..len + 3 {
            app.hub_key(KeyCode::Down);
        }
        assert_eq!(app.hub.selected, len - 1);
        for _ in 0..len + 3 {
            app.hub_key(KeyCode::Up);
        }
        assert_eq!(app.hub.selected, 0);

        // Esc peels one layer at a time: clear input, then close
        app.hub_key(KeyCode::Esc);
        assert!(app.hub.input.is_empty());
        assert!(app.hub.open);
        app.hub_key(KeyCode::Esc);
        assert!(!app.hub.open);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn hub_enter_on_unsupported_suggestion_explains_instead_of_downloading() {
        let dir = tempdir();
        let _guard = lock_env(); // App::new reads the config path and exports HF_TOKEN
        let mut app = test_app(dir.clone());
        app.library.dir = dir.clone(); // empty folder → default rows match suggested order
        app.open_hub();

        // No engine runs this format, so Enter must explain, not download.
        app.hub.suggested.push(crate::hub::SuggestedModel {
            name: "some-nemo-model".into(),
            repo: "x/some-nemo-model".into(),
            file: String::new(),
            size: "1 GB".into(),
            format: "nemo".into(),
            note: String::new(),
        });
        let index = app
            .hub_default_entries()
            .iter()
            .position(|e| e.name == "some-nemo-model")
            .unwrap();
        app.hub.selected = index;
        app.hub_key(KeyCode::Enter);
        assert!(app.hub.download.is_none());
        assert!(app.hub.info.contains("no engine"), "{}", app.hub.info);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// On Apple Silicon the MLX suggestions are real downloads: the
    /// default rows mark them as directory models with the repo basename
    /// as their on-disk name.
    #[test]
    fn hub_mlx_suggestions_are_dir_models_named_after_the_repo() {
        let dir = tempdir();
        let _guard = lock_env(); // App::new reads the config path and exports HF_TOKEN
        let mut app = test_app(dir.clone());
        app.library.dir = dir.clone();
        app.open_hub();

        let entries = app.hub_default_entries();
        let parakeet = entries
            .iter()
            .find(|e| e.name == "parakeet-tdt-0.6b-v3")
            .unwrap();
        assert!(parakeet.is_dir);
        assert!(!parakeet.installed);
        assert_eq!(parakeet.file, "parakeet-tdt-0.6b-v3");
        assert_eq!(parakeet.supported, crate::hub::metal_available());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn hub_dir_download_done_scans_the_new_directory_model() {
        let dir = tempdir();
        let _guard = lock_env(); // App::new reads the config path and exports HF_TOKEN
        let mut app = test_app(dir.clone());
        app.library.dir = dir.clone();
        app.library.selected = None;
        app.open_hub();

        // The finished download sits on disk as a complete model folder
        let model = dir.join("parakeet-tdt-0.6b-v3");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::write(model.join("config.json"), b"{}").unwrap();
        std::fs::write(model.join("model.safetensors"), vec![0u8; 500]).unwrap();

        let mut d = fake_download(
            "parakeet-tdt-0.6b-v3",
            &dir,
            Arc::new(AtomicBool::new(false)),
        );
        d.is_dir = true;
        d.remote_file = String::new();
        app.hub.download = Some(d);
        app.handle_hub_event(HubEvent::Done {
            file: "parakeet-tdt-0.6b-v3".into(),
            path: model.clone(),
        });

        assert!(app.hub.download.is_none());
        let found = app
            .library
            .models
            .iter()
            .find(|m| m.name == "parakeet-tdt-0.6b-v3")
            .expect("dir model scanned");
        assert!(found.is_dir);
        assert_eq!(found.size_bytes, 502);
        // nothing was selected, so the arrival becomes the selection
        assert_eq!(
            app.library.selected.as_ref().map(|m| m.name.as_str()),
            Some("parakeet-tdt-0.6b-v3")
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn hub_file_view_lists_variants_before_files() {
        let dir = tempdir();
        let _guard = lock_env(); // App::new reads the config path and exports HF_TOKEN
        let mut app = test_app(dir.clone());
        app.library.dir = dir.clone();
        app.open_hub();

        app.hub.listing_repo = Some("owner/multi".into());
        app.handle_hub_event(HubEvent::Files {
            repo: "owner/multi".into(),
            files: vec![crate::hub::HubFile {
                name: "ggml-tiny.bin".into(),
                size_bytes: 75,
            }],
            variants: vec![crate::hub::DirVariant {
                subdir: "8bit".into(),
                size_bytes: 1000,
                file_count: 3,
            }],
        });
        let view = app.hub.files.as_ref().expect("file view opens");
        assert_eq!(view.variants.len(), 1);
        assert_eq!(view.files.len(), 1);
        assert_eq!(app.hub_visible_list().len(), 2);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn hub_default_view_lists_downloaded_models_with_folder_size() {
        let _guard = lock_env(); // Enter → choose_model persists config
        let dir = tempdir();
        std::env::set_var("TRANSCRIBE_STT_CONFIG", dir.join("config.toml"));
        // A model that isn't in the curated list, plus the suggested large-v3.
        std::fs::write(dir.join("ggml-tiny.bin"), vec![0u8; 2048]).unwrap();
        std::fs::write(dir.join("ggml-large-v3.bin"), vec![0u8; 4096]).unwrap();
        let mut app = test_app(dir.clone());
        app.library.dir = dir.clone();
        app.open_hub();

        let entries = app.hub_default_entries();
        // Downloaded models come first, checkmarked, with their real folder size.
        let tiny = entries.iter().find(|e| e.file == "ggml-tiny.bin").unwrap();
        assert!(tiny.installed);
        assert_eq!(tiny.name, "ggml-tiny.bin"); // not in the curated list → file name
        assert_eq!(tiny.size, "2 KB");
        // A suggestion that's on disk shows its friendly name, still installed.
        let large = entries
            .iter()
            .find(|e| e.file == "ggml-large-v3.bin")
            .unwrap();
        assert!(large.installed);
        assert_eq!(large.name, "whisper-large-v3");
        // The two downloaded models precede the two remaining suggestions
        // (the MLX ones), which are greyed out (not installed).
        assert!(entries[0].installed && entries[1].installed);
        assert_eq!(
            entries
                .iter()
                .filter(|e| e.kind == crate::app::EntryKind::Model && !e.installed)
                .count(),
            2
        );
        let first_file = entries[0].file.to_string();
        drop(entries);
        // Enter on a downloaded row selects it as the default model.
        app.hub.selected = 0;
        app.hub_key(KeyCode::Enter);
        assert_eq!(
            app.config.default_model.as_deref(),
            Some(first_file.as_str())
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn hub_diarization_category_lists_downloads_and_selects_active() {
        use crate::app::EntryKind;
        use crate::diarize::sherpa;
        let _guard = lock_env(); // Enter persists the choice
        let dir = tempdir();
        std::env::set_var("TRANSCRIBE_STT_CONFIG", dir.join("config.toml"));
        let mut app = test_app(dir.clone());
        app.library.dir = dir.clone();
        app.open_hub();

        // The section header precedes the tdrz row plus one row per
        // catalog component, and each role's default is marked active
        // before anything downloads.
        let entries = app.hub_default_entries();
        let section = entries
            .iter()
            .position(|e| e.kind == EntryKind::Section)
            .expect("section header present");
        let rows: Vec<_> = entries
            .iter()
            .filter(|e| e.kind == EntryKind::Diarize)
            .collect();
        assert_eq!(rows.len(), sherpa::CATALOG.len() + 1);
        // The tdrz row heads the section: not installed here, and not
        // active while the strategy is Off.
        assert!(rows[0].file == crate::diarize::TDRZ_FILE && !rows[0].installed);
        assert!(!rows[0].active);
        assert!(rows
            .iter()
            .any(|e| e.file == "pyannote-segmentation-3.onnx" && e.active && !e.installed));
        assert!(rows
            .iter()
            .any(|e| e.file == "nemo-titanet-small.onnx" && e.active && !e.installed));
        drop(rows);
        drop(entries);

        // Enter on the section header is inert.
        app.hub.selected = section;
        app.hub_key(KeyCode::Enter);
        assert!(app.hub.download.is_none());

        // Enter on an installed embedding component makes it active
        // (persisted); the default loses its marker.
        let ddir = sherpa::dir(&dir);
        std::fs::create_dir_all(&ddir).unwrap();
        std::fs::write(ddir.join("campplus-zh-en.onnx"), b"onnx").unwrap();
        let index = app
            .hub_default_entries()
            .iter()
            .position(|e| e.file == "campplus-zh-en.onnx")
            .unwrap();
        app.hub.selected = index;
        app.hub_key(KeyCode::Enter);
        assert_eq!(
            app.config.diarize_models.embedding.as_deref(),
            Some("campplus-zh-en.onnx")
        );
        let entries = app.hub_default_entries();
        let camp = entries
            .iter()
            .find(|e| e.file == "campplus-zh-en.onnx")
            .unwrap();
        assert!(camp.installed && camp.active);
        let titanet = entries
            .iter()
            .find(|e| e.file == "nemo-titanet-small.onnx")
            .unwrap();
        assert!(!titanet.active);
        drop(entries);

        // Del prompts and removes the component file.
        app.hub.selected = index;
        app.hub_key(KeyCode::Delete);
        let prompt = app.hub.delete_prompt.as_ref().expect("prompt opens");
        assert_eq!(prompt.name, "campplus-zh-en.onnx");
        app.hub_key(KeyCode::Char('y'));
        assert!(!ddir.join("campplus-zh-en.onnx").exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn hub_delete_prompts_then_removes_the_file_on_confirm() {
        let _guard = lock_env(); // deleting the default persists a new one
        let dir = tempdir();
        std::env::set_var("TRANSCRIBE_STT_CONFIG", dir.join("config.toml"));
        std::fs::write(dir.join("ggml-tiny.bin"), vec![0u8; 1024]).unwrap();
        let mut app = test_app(dir.clone());
        app.library.dir = dir.clone();
        app.open_hub();

        // The downloaded model heads the default view.
        app.hub.selected = 0;
        app.hub_key(KeyCode::Delete);
        let prompt = app.hub.delete_prompt.as_ref().expect("prompt opens");
        assert_eq!(prompt.name, "ggml-tiny.bin");
        assert!(!prompt.yes_selected); // destructive default is "No"
        assert!(dir.join("ggml-tiny.bin").exists()); // nothing deleted yet

        // Esc cancels without touching the file.
        app.hub_key(KeyCode::Esc);
        assert!(app.hub.delete_prompt.is_none());
        assert!(dir.join("ggml-tiny.bin").exists());

        // Re-open and confirm with the 'y' shortcut.
        app.hub_key(KeyCode::Delete);
        app.hub_key(KeyCode::Char('y'));
        assert!(app.hub.delete_prompt.is_none());
        assert!(!dir.join("ggml-tiny.bin").exists());
        assert!(app.hub.info.contains("Deleted"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Build an in-flight download entry for tests, sharing the pause flag so
    /// the test can observe pause requests.
    fn fake_download(file: &str, dest_dir: &std::path::Path, pause: Arc<AtomicBool>) -> Download {
        Download {
            file: file.into(),
            repo: "repo".into(),
            remote_file: file.into(),
            is_dir: false,
            subdir: String::new(),
            dest_dir: dest_dir.to_path_buf(),
            got: 0,
            total: 0,
            paused: false,
            cancel: Arc::new(AtomicBool::new(false)),
            pause,
        }
    }

    #[test]
    fn hub_download_events_update_state_and_model_list() {
        let dir = tempdir();
        let _guard = lock_env(); // App::new reads the config path and exports HF_TOKEN
        let mut app = test_app(dir.clone());
        app.library.dir = dir.clone();
        app.open_hub();

        // Progress only updates an already-running download.
        app.hub.download = Some(fake_download(
            "ggml-x.bin",
            &dir,
            Arc::new(AtomicBool::new(false)),
        ));
        app.handle_hub_event(HubEvent::Progress {
            file: "ggml-x.bin".into(),
            got: 5,
            total: 10,
        });
        let d = app.hub.download.as_ref().unwrap();
        assert_eq!((d.file.as_str(), d.got, d.total), ("ggml-x.bin", 5, 10));

        // Done rescans the models folder and selects the new model if none was
        let path = dir.join("ggml-x.bin");
        std::fs::write(&path, b"weights").unwrap();
        app.library.selected = None;
        app.handle_hub_event(HubEvent::Done {
            file: "ggml-x.bin".into(),
            path: path.clone(),
        });
        assert!(app.hub.download.is_none());
        assert_eq!(
            app.library.selected.as_ref().map(|m| m.path.clone()),
            Some(path)
        );

        app.handle_hub_event(HubEvent::Failed {
            file: "y.bin".into(),
            error: "boom".into(),
        });
        assert!(app.hub.info.contains("boom"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn hub_pause_then_cancel_clears_state_and_partial_file() {
        let dir = tempdir();
        let _guard = lock_env(); // App::new reads the config path and exports HF_TOKEN
        let mut app = test_app(dir.clone());
        app.library.dir = dir.clone();
        app.open_hub();

        let pause = Arc::new(AtomicBool::new(false));
        app.hub.download = Some(fake_download("ggml-x.bin", &dir, pause.clone()));

        // 'p' asks the worker to pause: the shared flag flips, state not yet paused.
        app.hub_key(KeyCode::Char('p'));
        assert!(pause.load(Ordering::Relaxed));
        assert!(!app.hub.download.as_ref().unwrap().paused);

        // The worker acknowledges with Paused (honoured only while the flag holds).
        app.handle_hub_event(HubEvent::Paused {
            file: "ggml-x.bin".into(),
            got: 4,
        });
        let d = app.hub.download.as_ref().unwrap();
        assert!(d.paused);
        assert_eq!(d.got, 4);

        // Cancelling a paused download clears state and removes the partial file.
        std::fs::write(dir.join("ggml-x.bin.part"), b"partial").unwrap();
        app.hub_key(KeyCode::Esc);
        assert!(app.hub.download.is_none());
        assert!(!dir.join("ggml-x.bin.part").exists());
        assert!(app.hub.info.contains("Cancelled"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn export_with_no_transcript_errors() {
        let dir = tempdir();
        let _guard = lock_env(); // App::new reads the config path and exports HF_TOKEN
        let mut app = test_app(dir.clone());
        let err = app.export().unwrap_err();
        assert!(err.to_string().contains("nothing to export"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn dir_picker_navigates_and_commits_both_targets() {
        let _guard = lock_env();
        let dir = tempdir();
        // keep config::save away from the user's real config file
        std::env::set_var("TRANSCRIBE_STT_CONFIG", dir.join("config.toml"));
        std::fs::create_dir(dir.join("aa")).unwrap();
        std::fs::create_dir(dir.join("bb")).unwrap();
        std::fs::create_dir(dir.join(".hidden")).unwrap();
        let mut app = test_app(dir.clone());

        // models target: descend into aa/, then choose it
        app.settings.dir_picker = Some(DirPicker::at(DirTarget::Models, &dir));
        let rows = app.settings.dir_picker.as_ref().unwrap().rows();
        assert_eq!(rows[0], DirRow::UseThis);
        assert_eq!(rows[1], DirRow::Parent);
        assert_eq!(
            rows[2..],
            [DirRow::Sub(dir.join("aa")), DirRow::Sub(dir.join("bb"))]
        );

        app.dir_picker_key(KeyCode::Down);
        app.dir_picker_key(KeyCode::Down);
        app.dir_picker_key(KeyCode::Enter); // enter aa/
        assert_eq!(
            app.settings.dir_picker.as_ref().unwrap().cwd,
            dir.join("aa")
        );
        app.dir_picker_key(KeyCode::Enter); // "use this folder"
        assert!(app.settings.dir_picker.is_none());
        assert_eq!(app.library.dir, dir.join("aa").canonicalize().unwrap());
        assert_eq!(app.config.models_dir, Some(app.library.dir.clone()));

        // output target reuses the same component
        app.settings.dir_picker = Some(DirPicker::at(DirTarget::Output, &dir.join("bb")));
        app.dir_picker_key(KeyCode::Enter);
        assert_eq!(app.output_dir, dir.join("bb").canonicalize().unwrap());
        assert_eq!(app.config.output_dir, Some(app.output_dir.clone()));

        // Esc closes without committing
        app.settings.dir_picker = Some(DirPicker::at(DirTarget::Models, &dir));
        app.dir_picker_key(KeyCode::Esc);
        assert!(app.settings.dir_picker.is_none());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn setting_models_folder_adopts_existing_models() {
        let _guard = lock_env();
        let dir = tempdir();
        std::env::set_var("TRANSCRIBE_STT_CONFIG", dir.join("config.toml"));
        let folder = dir.join("stash");
        std::fs::create_dir(&folder).unwrap();
        std::fs::write(folder.join("ggml-large-v3.bin"), b"weights").unwrap();
        std::fs::write(folder.join("other.gguf"), b"weights").unwrap();
        std::fs::write(folder.join("notes.txt"), b"not a model").unwrap();

        let mut app = test_app(dir.clone());
        app.library.selected = None;
        app.set_models_dir_path(folder.clone());

        let names: Vec<&str> = app.library.models.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, vec!["ggml-large-v3.bin", "other.gguf"]);
        // with nothing selected before, the scan adopts large-v3
        assert_eq!(
            app.library.selected.as_ref().map(|m| m.name.as_str()),
            Some("ggml-large-v3.bin")
        );
        assert!(app.status.contains("2 model(s) found"), "{}", app.status);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn changing_models_folder_offers_to_move_and_moves_on_yes() {
        let _guard = lock_env();
        let dir = tempdir();
        std::env::set_var("TRANSCRIBE_STT_CONFIG", dir.join("config.toml"));
        let old = dir.join("old");
        let new = dir.join("new");
        std::fs::create_dir(&old).unwrap();
        std::fs::create_dir(&new).unwrap();
        std::fs::write(old.join("ggml-a.bin"), b"weights").unwrap();
        std::fs::write(old.join("ggml-b.bin"), b"weights").unwrap();

        let mut app = test_app(dir.clone());
        app.set_models_dir_path(old.clone());
        // dismiss any prompt about the host machine's previous folder
        app.settings.move_prompt = None;

        app.set_models_dir_path(new.clone());
        let prompt = app
            .settings
            .move_prompt
            .as_ref()
            .expect("prompt should open");
        assert_eq!(prompt.count, 2);
        assert!(prompt.yes_selected);

        app.move_prompt_key(KeyCode::Enter); // Yes is preselected
        assert!(app.settings.move_prompt.is_none());
        // the move runs on a worker thread; wait for its completion event
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while !app.status.starts_with("Moved") {
            assert!(std::time::Instant::now() < deadline, "move never finished");
            app.hub_pump();
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(app.status.contains("Moved 2 model(s)"), "{}", app.status);
        assert!(app.library.dir.join("ggml-a.bin").is_file());
        assert!(!old.join("ggml-a.bin").exists());
        assert_eq!(app.library.models.len(), 2);
        // with nothing selected before, the arrival adopts a model
        assert!(app.library.selected.is_some());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn move_prompt_no_leaves_models_in_place() {
        let _guard = lock_env();
        let dir = tempdir();
        std::env::set_var("TRANSCRIBE_STT_CONFIG", dir.join("config.toml"));
        let old = dir.join("old");
        let new = dir.join("new");
        std::fs::create_dir(&old).unwrap();
        std::fs::create_dir(&new).unwrap();
        std::fs::write(old.join("ggml-a.bin"), b"weights").unwrap();

        let mut app = test_app(dir.clone());
        app.set_models_dir_path(old.clone());
        app.settings.move_prompt = None;
        app.set_models_dir_path(new.clone());
        assert!(app.settings.move_prompt.is_some());

        // toggle to No, confirm with Enter
        app.move_prompt_key(KeyCode::Left);
        app.move_prompt_key(KeyCode::Enter);
        assert!(app.settings.move_prompt.is_none());
        assert!(old.join("ggml-a.bin").is_file());
        assert!(!new.join("ggml-a.bin").exists());
        assert!(app.status.contains("left in the previous folder"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn load_and_unload_events_drive_the_progress_states() {
        let dir = tempdir();
        let _guard = lock_env(); // App::new reads the config path and exports HF_TOKEN
        let mut app = test_app(dir.clone());

        app.handle_event(Event::LoadingModel("m.bin".into()));
        assert_eq!(
            app.work,
            WorkState::LoadingModel {
                name: "m.bin".into(),
                progress: 0
            }
        );
        app.handle_event(Event::LoadProgress(42));
        assert_eq!(
            app.work,
            WorkState::LoadingModel {
                name: "m.bin".into(),
                progress: 42
            }
        );

        app.handle_event(Event::Unloading);
        assert_eq!(app.work, WorkState::UnloadingModel);
        app.handle_event(Event::Unloaded);
        assert!(!app.busy());
        assert!(app.status.contains("unloaded"), "{}", app.status);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn choosing_a_model_is_metadata_only_and_persists_the_default() {
        let _guard = lock_env();
        let dir = tempdir();
        std::env::set_var("TRANSCRIBE_STT_CONFIG", dir.join("config.toml"));
        let mut app = test_app(dir.clone());
        app.library.selected = None;

        let model = ModelFile {
            path: dir.join("ggml-x.bin"),
            name: "ggml-x.bin".into(),
            size_bytes: 4,
            is_dir: false,
        };
        app.choose_model(model.clone());
        assert_eq!(app.config.default_model.as_deref(), Some("ggml-x.bin"));
        assert_eq!(
            app.library.selected.as_ref().map(|m| m.name.as_str()),
            Some("ggml-x.bin")
        );
        // lazy by design: selection is metadata; loading happens at job time
        assert!(
            app.status
                .contains("loads when the next transcription starts"),
            "{}",
            app.status
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn split_mode_cycles_and_lands_in_the_job() {
        let _guard = lock_env();
        let dir = tempdir();
        std::env::set_var("TRANSCRIBE_STT_CONFIG", dir.join("config.toml"));
        let mut app = test_app(dir.clone());
        app.config.split_mode = crate::split::SplitMode::Auto;

        app.cycle_split_mode();
        assert_eq!(app.config.split_mode, crate::split::SplitMode::Silence);
        assert!(app.status.contains("speech pauses"), "{}", app.status);
        app.cycle_split_mode();
        assert_eq!(app.config.split_mode, crate::split::SplitMode::Fixed);
        app.cycle_split_mode();
        assert_eq!(app.config.split_mode, crate::split::SplitMode::Auto);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn dir_picker_falls_back_from_missing_start() {
        let dir = tempdir();
        let picker = DirPicker::at(DirTarget::Output, &dir.join("does/not/exist"));
        assert_eq!(picker.cwd, dir);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
