//! Application state and update logic. `App` owns the sub-state for each
//! UI concern; the per-concern logic lives in the submodules below, and
//! `ui` renders it all read-only.

mod browser;
mod drop;
mod events;
mod hub_state;
mod keys;
mod library;
mod settings;
mod transcript;

pub(crate) use browser::file_name;
pub use browser::{FileBrowser, FileEntry};
pub use drop::DropDetector;
pub use hub_state::{HubList, HubState};
pub use library::{ModelLibrary, ModelPicker};
pub use settings::{DirPicker, DirRow, DirTarget, MovePrompt, SettingsRow, SettingsUi};
pub use transcript::TranscriptState;

use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};

use anyhow::Result;

use crate::config::{self, Config};
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
    LoadingModel {
        name: String,
        progress: i32,
    },
    /// Resident model being released after a model switch
    UnloadingModel,
    Decoding,
    Transcribing {
        progress: i32,
    },
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

    pub transcriber: Transcriber,
    hub_tx: Sender<HubEvent>,
    hub_rx: Receiver<HubEvent>,
}

impl App {
    pub fn new(start_dir: PathBuf, model: Option<PathBuf>, transcriber: Transcriber) -> Self {
        let app_config = config::load();
        let models_dir = config::resolve_models_dir(&app_config);
        let output_dir = config::resolve_output_dir(&app_config);
        let models_list = models::scan_models(&models_dir);
        let selected_model = model
            .map(|p| ModelFile {
                name: file_name(&p),
                size_bytes: std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0),
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
            transcriber,
            hub_tx,
            hub_rx,
        }
    }

    pub fn busy(&self) -> bool {
        self.work != WorkState::Idle
    }

    pub fn start_transcription(&mut self, audio: PathBuf) {
        // A pending unload is fine — the job queues right behind it
        if self.busy() && self.work != WorkState::UnloadingModel {
            self.status = "A job is already running — press c to cancel it first".into();
            return;
        }
        // Diarization is decided before conversion: it needs a tdrz model
        let model = if self.config.diarize {
            match self.find_tdrz_model() {
                Some(m) => m,
                None => {
                    self.status = "Diarization is ON but no tdrz model found — \
                                   get ggml-small.en-tdrz.bin from huggingface.co/akashmjn/\
                                   tinydiarize-whisper.cpp (or press d to turn it off)"
                        .into();
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
        self.transcript.begin(audio.clone(), model.name.clone());
        self.work = WorkState::LoadingModel {
            name: model.name.clone(),
            progress: 0,
        };
        self.status = format!("Starting {}", file_name(&audio));
        self.transcriber.submit(Job {
            model: model.path,
            audio,
            diarize: self.config.diarize,
            language: self.config.language.clone(),
            split_mode: self.config.split_mode,
        });
    }

    /// Write every export format into `<output_dir>/<source>_<timestamp>/`
    /// and return that folder. Runs automatically after each transcription;
    /// `e` re-exports on demand.
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
        };
        let written = doc.write_all(&self.output_dir)?;
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
    use std::sync::Mutex;
    use std::time::Duration;

    /// Serializes tests that point TRANSCRIBE_STT_CONFIG at their tempdir:
    /// the variable is process-global, so concurrent setters would make one
    /// test's config::save land in another test's (soon-deleted) folder.
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

        let app = test_app(dir.clone());
        let labels: Vec<String> = app.browser.entries.iter().map(|e| e.label()).collect();
        assert_eq!(labels, vec!["../", "z_subdir/", "a.mp4 \u{29c9}", "b.wav"]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn worker_events_drive_state_machine_and_auto_export() {
        let app_dir = tempdir();
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
    fn error_event_resets_to_idle() {
        let dir = tempdir();
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
        let mut app = test_app(dir.clone());

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
        app.hub_key(KeyCode::Down);
        app.hub_key(KeyCode::Down);
        app.hub_key(KeyCode::Down);
        app.hub_key(KeyCode::Down);
        assert_eq!(app.hub.selected, 2);
        app.hub_key(KeyCode::Up);
        app.hub_key(KeyCode::Up);
        app.hub_key(KeyCode::Up);
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
    fn hub_enter_on_mlx_suggestion_explains_instead_of_downloading() {
        let dir = tempdir();
        let mut app = test_app(dir.clone());
        app.open_hub();

        let mlx_index = app
            .hub
            .suggested
            .iter()
            .position(|s| !s.supported())
            .unwrap();
        app.hub.selected = mlx_index;
        app.hub_key(KeyCode::Enter);
        assert!(app.hub.download.is_none());
        assert!(app.hub.info.contains("mlx-format"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn hub_download_events_update_state_and_model_list() {
        let dir = tempdir();
        let mut app = test_app(dir.clone());
        app.library.dir = dir.clone();
        app.open_hub();

        app.handle_hub_event(HubEvent::Progress {
            file: "ggml-x.bin".into(),
            got: 5,
            total: 10,
        });
        assert_eq!(app.hub.download, Some(("ggml-x.bin".into(), 5, 10)));

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
    fn export_with_no_transcript_errors() {
        let dir = tempdir();
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
