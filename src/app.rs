use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::event::KeyCode;

use crate::audio;
use crate::config::{self, Config};
use crate::export::TranscriptDoc;
use crate::hub::{self, HubEvent, HubFile, RepoHit, SuggestedModel};
use crate::models::{self, ModelFile};
use crate::stats::ProcStats;
use crate::transcribe::{Event, Job, Segment, Transcriber};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    Files,
    Transcript,
}

#[derive(Clone, Debug, PartialEq)]
pub enum WorkState {
    Idle,
    LoadingModel { name: String, progress: i32 },
    /// Resident model being released after a model switch
    UnloadingModel,
    Decoding,
    Transcribing { progress: i32 },
}

pub enum FileEntry {
    Parent,
    Dir(PathBuf),
    Media(PathBuf),
}

impl FileEntry {
    pub fn label(&self) -> String {
        match self {
            FileEntry::Parent => "../".into(),
            FileEntry::Dir(p) => format!("{}/", file_name(p)),
            FileEntry::Media(p) => {
                if audio::is_video_file(p) {
                    format!("{} ⧉", file_name(p))
                } else {
                    file_name(p)
                }
            }
        }
    }
}

fn file_name(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.display().to_string())
}

/// Which folder setting a DirPicker session is choosing for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DirTarget {
    Models,
    Output,
}

impl DirTarget {
    pub fn label(&self) -> &'static str {
        match self {
            DirTarget::Models => "models",
            DirTarget::Output => "output",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum DirRow {
    UseThis,
    Parent,
    Sub(PathBuf),
}

/// The reusable directory browser: any setting that needs a folder opens
/// one of these over the settings modal.
pub struct DirPicker {
    pub target: DirTarget,
    pub cwd: PathBuf,
    pub dirs: Vec<PathBuf>,
    pub selected: usize,
}

impl DirPicker {
    pub fn at(target: DirTarget, start: &Path) -> Self {
        // Walk up until something exists, falling back to $HOME
        let mut cwd = start.to_path_buf();
        while !cwd.is_dir() {
            match cwd.parent() {
                Some(p) if !p.as_os_str().is_empty() => cwd = p.to_path_buf(),
                _ => {
                    cwd = std::env::var_os("HOME")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| PathBuf::from("/"));
                    break;
                }
            }
        }
        let mut picker = Self {
            target,
            cwd,
            dirs: Vec::new(),
            selected: 0,
        };
        picker.refresh();
        picker
    }

    fn refresh(&mut self) {
        self.dirs = std::fs::read_dir(&self.cwd)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|entry| {
                let path = entry.path();
                let hidden = path
                    .file_name()
                    .map(|n| n.to_string_lossy().starts_with('.'))
                    .unwrap_or(true);
                (path.is_dir() && !hidden).then_some(path)
            })
            .collect();
        self.dirs.sort();
        self.selected = 0;
    }

    /// The rows as displayed: pick-here first, then up, then subfolders.
    pub fn rows(&self) -> Vec<DirRow> {
        let mut rows = vec![DirRow::UseThis];
        if self.cwd.parent().is_some() {
            rows.push(DirRow::Parent);
        }
        rows.extend(self.dirs.iter().cloned().map(DirRow::Sub));
        rows
    }
}

/// Yes/No dialog offered after the models folder changes while the old
/// folder still holds models.
pub struct MovePrompt {
    pub from: PathBuf,
    pub to: PathBuf,
    pub count: usize,
    pub yes_selected: bool,
}

/// State of the Model management modal (search + download from Hugging Face).
pub struct HubState {
    pub open: bool,
    /// Search bar contents; empty shows the suggested list
    pub input: String,
    pub selected: usize,
    pub searching: bool,
    /// Some(repo) while its file list is being fetched
    pub listing_repo: Option<String>,
    /// None → suggested list; Some → search results
    pub results: Option<Vec<RepoHit>>,
    /// Some((repo, files)) → file view of one repo
    pub files: Option<(String, Vec<HubFile>)>,
    pub suggested: Vec<SuggestedModel>,
    /// Some((file, got_bytes, total_bytes)) while a download runs
    pub download: Option<(String, u64, u64)>,
    pub cancel: Arc<AtomicBool>,
    /// Modal-local status/info line
    pub info: String,
    /// Debounce marker: search fires shortly after typing pauses
    last_edit: Option<Instant>,
}

impl HubState {
    fn new() -> Self {
        Self {
            open: false,
            input: String::new(),
            selected: 0,
            searching: false,
            listing_repo: None,
            results: None,
            files: None,
            suggested: hub::suggested_models(),
            download: None,
            cancel: Arc::new(AtomicBool::new(false)),
            info: String::new(),
            last_edit: None,
        }
    }
}

pub struct App {
    pub should_quit: bool,
    pub focus: Focus,

    // file browser
    pub cwd: PathBuf,
    pub entries: Vec<FileEntry>,
    pub file_selected: usize,

    // models
    pub models: Vec<ModelFile>,
    pub selected_model: Option<ModelFile>,
    pub model_picker_open: bool,
    pub model_picker_selected: usize,
    pub models_dir: PathBuf,

    // settings
    pub config: Config,
    pub settings_open: bool,
    pub settings_selected: usize,
    /// Some while a folder is being chosen via the directory browser
    pub dir_picker: Option<DirPicker>,
    /// Some while asking whether to move models into the new folder
    pub move_prompt: Option<MovePrompt>,
    /// Some(text) while the language code is being edited
    pub language_input: Option<String>,
    /// Where transcript exports are written
    pub output_dir: PathBuf,

    // transcript
    pub segments: Vec<Segment>,
    pub transcript_scroll: usize,
    pub follow: bool,
    pub current_audio: Option<PathBuf>,
    pub audio_duration: Option<f32>,
    pub language: Option<String>,
    pub used_model_name: Option<String>,

    pub work: WorkState,
    pub status: String,
    pub stats: ProcStats,
    /// Diarization toggle — resolved to a tdrz model when a job starts
    pub diarize: bool,
    /// Render-loop counter driving the loading spinner
    pub tick: usize,

    pub transcriber: Transcriber,

    pub hub: HubState,
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

        let diarize = app_config.diarize;
        let (hub_tx, hub_rx) = channel();
        let mut app = Self {
            should_quit: false,
            focus: Focus::Files,
            cwd: start_dir,
            entries: Vec::new(),
            file_selected: 0,
            models: models_list,
            selected_model,
            model_picker_open: false,
            model_picker_selected: 0,
            models_dir,
            config: app_config,
            settings_open: false,
            settings_selected: 0,
            dir_picker: None,
            move_prompt: None,
            language_input: None,
            output_dir,
            segments: Vec::new(),
            transcript_scroll: 0,
            follow: true,
            current_audio: None,
            audio_duration: None,
            language: None,
            used_model_name: None,
            work: WorkState::Idle,
            status: String::from(
                "Select a file and press Enter, or drop an audio/video file onto the window",
            ),
            stats: ProcStats::new(),
            diarize,
            tick: 0,
            transcriber,
            hub: HubState::new(),
            hub_tx,
            hub_rx,
        };
        app.refresh_entries();
        app
    }

    pub fn busy(&self) -> bool {
        self.work != WorkState::Idle
    }

    pub fn refresh_entries(&mut self) {
        let mut dirs: Vec<PathBuf> = Vec::new();
        let mut files: Vec<PathBuf> = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&self.cwd) {
            for entry in rd.flatten() {
                let path = entry.path();
                let name = file_name(&path);
                if name.starts_with('.') {
                    continue;
                }
                if path.is_dir() {
                    dirs.push(path);
                } else if audio::is_media_file(&path) {
                    files.push(path);
                }
            }
        }
        dirs.sort();
        files.sort();

        self.entries.clear();
        if self.cwd.parent().is_some() {
            self.entries.push(FileEntry::Parent);
        }
        self.entries.extend(dirs.into_iter().map(FileEntry::Dir));
        self.entries.extend(files.into_iter().map(FileEntry::Media));
        self.file_selected = self.file_selected.min(self.entries.len().saturating_sub(1));
    }

    pub fn refresh_models(&mut self) {
        self.models = models::scan_models(&self.models_dir);
        self.model_picker_selected = self
            .model_picker_selected
            .min(self.models.len().saturating_sub(1));
    }

    /// Toggle diarization and persist the choice.
    pub fn toggle_diarize(&mut self) {
        self.diarize = !self.diarize;
        self.config.diarize = self.diarize;
        let _ = config::save(&self.config);
        self.status = if self.diarize {
            match self.find_tdrz_model() {
                Some(m) => format!("Diarization ON — conversations will use {}", m.name),
                None => "Diarization ON, but no tdrz model found — get ggml-small.en-tdrz.bin from huggingface.co/akashmjn/tinydiarize-whisper.cpp into the models folder".into(),
            }
        } else {
            "Diarization OFF".into()
        };
    }

    /// Set the transcription language ("auto" or an ISO 639-1 code),
    /// validated against whisper's language list, and persist it.
    pub fn set_language(&mut self, input: &str) {
        let code = input.trim().to_ascii_lowercase();
        if code.is_empty() || code == "auto" {
            self.config.language = None;
            self.status = "Language: auto-detect".into();
        } else if whisper_rs::get_lang_id(&code).is_some() {
            self.config.language = Some(code.clone());
            self.status = format!("Language set to {code} (skips auto-detection)");
        } else {
            self.status = format!("Unknown language code: {code} (use e.g. en, es, de, or auto)");
            return;
        }
        let _ = config::save(&self.config);
    }

    /// Cycle the long-audio split strategy and persist it.
    pub fn cycle_split_mode(&mut self) {
        self.config.split_mode = self.config.split_mode.next();
        let _ = config::save(&self.config);
        self.status = format!("Split mode: {}", self.config.split_mode.label());
    }

    pub fn find_tdrz_model(&self) -> Option<ModelFile> {
        self.models
            .iter()
            .find(|m| models::is_tdrz(&m.name))
            .cloned()
    }

    /// Set a new default model and persist it. Models load lazily: the
    /// selection here is metadata only, and switching away from a loaded
    /// model releases its memory right away instead of at the next job.
    pub fn choose_model(&mut self, model: ModelFile) {
        let changed = self
            .selected_model
            .as_ref()
            .map(|m| m.path != model.path)
            .unwrap_or(true);
        if changed {
            self.transcriber.request_unload();
        }
        self.config.default_model = Some(model.name.clone());
        self.status = format!(
            "Selected {} (default) — loads when the next transcription starts",
            model.name
        );
        self.selected_model = Some(model);
        if let Err(e) = config::save(&self.config) {
            self.status = format!("Model selected, but saving config failed: {e}");
        }
    }

    /// Change the models folder, persist it, and rescan. If the previous
    /// folder still holds models, offer to move them over.
    pub fn set_models_dir_path(&mut self, path: PathBuf) {
        if let Err(e) = std::fs::create_dir_all(&path) {
            self.status = format!("Cannot use folder {}: {e}", path.display());
            return;
        }
        let old_dir = self.models_dir.clone();
        let old_count = self.models.len();
        let path = path.canonicalize().unwrap_or(path);
        self.models_dir = path.clone();
        self.config.models_dir = Some(path);
        // Scan the new folder: every supported model inside joins the list
        self.refresh_models();
        self.adopt_selection();
        if let Err(e) = config::save(&self.config) {
            self.status = format!("Saving config failed: {e}");
        } else {
            self.status = format!(
                "Models folder set to {} — {} model(s) found",
                self.models_dir.display(),
                self.models.len()
            );
        }
        if old_dir != self.models_dir && old_count > 0 {
            self.move_prompt = Some(MovePrompt {
                from: old_dir,
                to: self.models_dir.clone(),
                count: old_count,
                yes_selected: true,
            });
        }
    }

    /// Keep a still-valid selection; otherwise adopt what the folder has
    /// (configured default first, then large-v3, then whatever exists).
    fn adopt_selection(&mut self) {
        let selection_gone = self
            .selected_model
            .as_ref()
            .map(|m| !m.path.exists())
            .unwrap_or(true);
        if selection_gone {
            self.selected_model =
                models::pick_default(&self.models, self.config.default_model.as_deref()).cloned();
        }
    }

    pub fn move_prompt_key(&mut self, code: KeyCode) {
        let Some(prompt) = &mut self.move_prompt else {
            return;
        };
        match code {
            KeyCode::Esc | KeyCode::Char('n') => {
                self.move_prompt = None;
                self.status = "Models left in the previous folder".into();
            }
            KeyCode::Left
            | KeyCode::Right
            | KeyCode::Tab
            | KeyCode::Char('h')
            | KeyCode::Char('l') => prompt.yes_selected = !prompt.yes_selected,
            KeyCode::Char('y') => {
                let prompt = self.move_prompt.take().unwrap();
                self.start_move_models(prompt.from, prompt.to, prompt.count);
            }
            KeyCode::Enter => {
                let prompt = self.move_prompt.take().unwrap();
                if prompt.yes_selected {
                    self.start_move_models(prompt.from, prompt.to, prompt.count);
                } else {
                    self.status = "Models left in the previous folder".into();
                }
            }
            _ => {}
        }
    }

    /// Move the models on a worker thread — same-volume moves are instant
    /// renames, but cross-volume copies of multi-GB files must not freeze
    /// the render loop. Completion arrives as HubEvent::ModelsMoved.
    fn start_move_models(&mut self, from: PathBuf, to: PathBuf, count: usize) {
        self.status = format!("Moving {count} model(s) to {}…", to.display());
        let tx = self.hub_tx.clone();
        std::thread::spawn(move || {
            let (moved, skipped, failed) = models::move_models(&from, &to);
            let _ = tx.send(HubEvent::ModelsMoved {
                moved,
                skipped,
                failed,
            });
        });
    }

    /// Change the export output folder and persist it.
    pub fn set_output_dir_path(&mut self, path: PathBuf) {
        if let Err(e) = std::fs::create_dir_all(&path) {
            self.status = format!("Cannot use folder {}: {e}", path.display());
            return;
        }
        let path = path.canonicalize().unwrap_or(path);
        self.output_dir = path.clone();
        self.config.output_dir = Some(path);
        if let Err(e) = config::save(&self.config) {
            self.status = format!("Saving config failed: {e}");
        } else {
            self.status = format!("Output folder set to {}", self.output_dir.display());
        }
    }

    pub fn open_dir_picker(&mut self, target: DirTarget) {
        let start = match target {
            DirTarget::Models => self.models_dir.clone(),
            DirTarget::Output => self.output_dir.clone(),
        };
        self.dir_picker = Some(DirPicker::at(target, &start));
    }

    pub fn dir_picker_key(&mut self, code: KeyCode) {
        let Some(picker) = &mut self.dir_picker else {
            return;
        };
        match code {
            KeyCode::Esc => self.dir_picker = None,
            KeyCode::Up | KeyCode::Char('k') => picker.selected = picker.selected.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => {
                let len = picker.rows().len();
                if len > 0 {
                    picker.selected = (picker.selected + 1).min(len - 1);
                }
            }
            KeyCode::Enter => match picker.rows().get(picker.selected).cloned() {
                Some(DirRow::UseThis) => {
                    let (target, dir) = (picker.target, picker.cwd.clone());
                    self.dir_picker = None;
                    match target {
                        DirTarget::Models => self.set_models_dir_path(dir),
                        DirTarget::Output => self.set_output_dir_path(dir),
                    }
                }
                Some(DirRow::Parent) => {
                    if let Some(parent) = picker.cwd.parent() {
                        picker.cwd = parent.to_path_buf();
                        picker.refresh();
                    }
                }
                Some(DirRow::Sub(dir)) => {
                    picker.cwd = dir;
                    picker.refresh();
                }
                None => {}
            },
            _ => {}
        }
    }

    pub fn enter_selected(&mut self) {
        match self.entries.get(self.file_selected) {
            Some(FileEntry::Parent) => {
                if let Some(parent) = self.cwd.parent() {
                    self.cwd = parent.to_path_buf();
                    self.file_selected = 0;
                    self.refresh_entries();
                }
            }
            Some(FileEntry::Dir(p)) => {
                self.cwd = p.clone();
                self.file_selected = 0;
                self.refresh_entries();
            }
            Some(FileEntry::Media(p)) => {
                let path = p.clone();
                self.start_transcription(path);
            }
            None => {}
        }
    }

    pub fn start_transcription(&mut self, audio: PathBuf) {
        // A pending unload is fine — the job queues right behind it
        if self.busy() && self.work != WorkState::UnloadingModel {
            self.status = "A job is already running — press c to cancel it first".into();
            return;
        }
        // Diarization is decided before conversion: it needs a tdrz model
        let model = if self.diarize {
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
            match self.selected_model.clone() {
                Some(m) => m,
                None => {
                    self.status =
                        "No model found — download one via s → Model management".into();
                    return;
                }
            }
        };
        self.segments.clear();
        self.transcript_scroll = 0;
        self.follow = true;
        self.audio_duration = None;
        self.language = None;
        self.used_model_name = Some(model.name.clone());
        self.current_audio = Some(audio.clone());
        self.work = WorkState::LoadingModel {
            name: model.name.clone(),
            progress: 0,
        };
        self.status = format!("Starting {}", file_name(&audio));
        self.transcriber.submit(Job {
            model: model.path,
            audio,
            diarize: self.diarize,
            language: self.config.language.clone(),
            split_mode: self.config.split_mode,
        });
    }

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
                self.audio_duration = Some(duration_secs);
                self.work = WorkState::Transcribing { progress: 0 };
                self.status = format!("Transcribing {duration_secs:.0}s of audio…");
            }
            Event::Progress(p) => {
                if let WorkState::Transcribing { progress } = &mut self.work {
                    *progress = p;
                }
            }
            Event::Segment(seg) => {
                self.segments.push(seg);
            }
            Event::SegmentsFinal(segments) => {
                self.segments = segments;
            }
            Event::Done {
                elapsed_secs,
                audio_secs,
                language,
            } => {
                self.work = WorkState::Idle;
                let rtf = elapsed_secs / audio_secs.max(0.001);
                let lang = language.as_deref().unwrap_or("?").to_string();
                self.language = language;
                let spoken = if self.segments.iter().any(|s| s.speaker.is_some()) {
                    " · 2 speakers labeled"
                } else {
                    ""
                };
                let base =
                    format!("Done in {elapsed_secs:.1}s ({rtf:.2}× realtime, lang: {lang}{spoken})");
                // Exports run automatically after every successful transcription
                self.status = if self.segments.is_empty() {
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

    // --- Model management (Hugging Face hub) ---

    pub fn open_hub(&mut self) {
        self.refresh_models();
        self.hub.open = true;
        self.hub.selected = 0;
        self.hub.info.clear();
    }

    /// Called every render loop: fires the debounced search and drains
    /// events from hub worker threads.
    pub fn hub_pump(&mut self) {
        if let Some(t) = self.hub.last_edit {
            if t.elapsed() >= Duration::from_millis(450) {
                self.hub.last_edit = None;
                let query = self.hub.input.trim().to_string();
                if query.len() >= 2 && self.hub.files.is_none() {
                    self.hub.searching = true;
                    hub::search(query, self.hub_tx.clone());
                }
            }
        }
        let events: Vec<HubEvent> = std::iter::from_fn(|| self.hub_rx.try_recv().ok()).collect();
        for event in events {
            self.handle_hub_event(event);
        }
    }

    pub fn hub_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Esc => {
                if self.hub.download.is_some() {
                    self.hub.cancel.store(true, Ordering::Relaxed);
                    self.hub.info = "Cancelling download…".into();
                } else if self.hub.files.is_some() {
                    self.hub.files = None;
                    self.hub.selected = 0;
                } else if !self.hub.input.is_empty() {
                    self.hub.input.clear();
                    self.hub.results = None;
                    self.hub.searching = false;
                    self.hub.last_edit = None;
                    self.hub.selected = 0;
                    self.hub.info.clear();
                } else {
                    self.hub.open = false;
                }
            }
            KeyCode::Up => self.hub.selected = self.hub.selected.saturating_sub(1),
            KeyCode::Down => {
                let len = self.hub_list_len();
                if len > 0 {
                    self.hub.selected = (self.hub.selected + 1).min(len - 1);
                }
            }
            KeyCode::Backspace if self.hub.files.is_none() => {
                self.hub.input.pop();
                self.hub_input_edited();
            }
            KeyCode::Char(c) if self.hub.files.is_none() => {
                self.hub.input.push(c);
                self.hub_input_edited();
            }
            KeyCode::Enter => self.hub_enter(),
            _ => {}
        }
    }

    fn hub_input_edited(&mut self) {
        self.hub.last_edit = Some(Instant::now());
        self.hub.selected = 0;
        if self.hub.input.trim().is_empty() {
            self.hub.results = None;
            self.hub.searching = false;
        }
    }

    fn hub_list_len(&self) -> usize {
        if let Some((_, files)) = &self.hub.files {
            files.len()
        } else if let Some(results) = &self.hub.results {
            results.len()
        } else {
            self.hub.suggested.len()
        }
    }

    fn hub_enter(&mut self) {
        if self.hub.download.is_some() {
            self.hub.info = "A download is already running — Esc cancels it".into();
            return;
        }
        // File view: download the selected file
        if let Some((repo, files)) = &self.hub.files {
            if let Some(f) = files.get(self.hub.selected) {
                let (repo, file) = (repo.clone(), f.name.clone());
                self.hub_start_download(repo, file);
            }
            return;
        }
        // Search results: open the repo's file list
        if let Some(results) = &self.hub.results {
            if let Some(hit) = results.get(self.hub.selected) {
                let repo = hit.id.clone();
                self.hub.listing_repo = Some(repo.clone());
                self.hub.info = format!("Fetching file list for {repo}…");
                hub::list_files(repo, self.hub_tx.clone());
            }
            return;
        }
        // Suggested list: download directly (if the format is runnable)
        if let Some(s) = self.hub.suggested.get(self.hub.selected).cloned() {
            if !s.supported() {
                self.hub.info = format!(
                    "{} is {}-format — whisper.cpp can only run GGML/GGUF models",
                    s.name, s.format
                );
                return;
            }
            self.hub_start_download(s.repo, s.file);
        }
    }

    fn hub_start_download(&mut self, repo: String, file: String) {
        let Some(base) = hub::dest_name(&file) else {
            self.hub.info = "Bad file name".into();
            return;
        };
        if self.models.iter().any(|m| m.name == base) {
            self.hub.info = format!("{base} is already in the models folder");
            return;
        }
        self.hub.cancel = Arc::new(AtomicBool::new(false));
        self.hub.download = Some((base.clone(), 0, 0));
        self.hub.info = format!("Downloading {base}…");
        hub::download(
            repo,
            file,
            self.models_dir.clone(),
            self.hub.cancel.clone(),
            self.hub_tx.clone(),
        );
    }

    fn handle_hub_event(&mut self, event: HubEvent) {
        match event {
            HubEvent::SearchResults { query, hits } => {
                self.hub.searching = false;
                // Drop stale results the input has moved past
                if query == self.hub.input.trim() {
                    self.hub.info = if hits.is_empty() {
                        format!("No repos match '{query}'")
                    } else {
                        String::new()
                    };
                    self.hub.results = Some(hits);
                    self.hub.selected = 0;
                }
            }
            HubEvent::SearchFailed { query, error } => {
                self.hub.searching = false;
                self.hub.info = format!("Search '{query}' failed: {error}");
            }
            HubEvent::Files { repo, files } => {
                if self.hub.listing_repo.as_deref() == Some(repo.as_str()) {
                    self.hub.listing_repo = None;
                    if files.is_empty() {
                        self.hub.info = format!("No GGML/GGUF files in {repo}");
                    } else {
                        self.hub.files = Some((repo, files));
                        self.hub.selected = 0;
                        self.hub.info.clear();
                    }
                }
            }
            HubEvent::FilesFailed { repo, error } => {
                self.hub.listing_repo = None;
                self.hub.info = format!("Listing {repo} failed: {error}");
            }
            HubEvent::Progress { file, got, total } => {
                self.hub.download = Some((file, got, total));
            }
            HubEvent::Done { file, path } => {
                self.hub.download = None;
                self.refresh_models();
                self.hub.info = format!("Downloaded {file} ✓");
                self.status = format!("Downloaded {file} to {}", self.models_dir.display());
                if self.selected_model.is_none() {
                    self.selected_model = self.models.iter().find(|m| m.path == path).cloned();
                }
            }
            HubEvent::Cancelled { file } => {
                self.hub.download = None;
                self.hub.info = format!("Cancelled {file}");
            }
            HubEvent::Failed { file, error } => {
                self.hub.download = None;
                self.hub.info = format!("Download of {file} failed: {error}");
            }
            HubEvent::ModelsMoved {
                moved,
                skipped,
                failed,
            } => {
                self.refresh_models();
                self.adopt_selection();
                let mut parts = vec![format!(
                    "Moved {moved} model(s) to {}",
                    self.models_dir.display()
                )];
                if skipped > 0 {
                    parts.push(format!("{skipped} already existed"));
                }
                if failed > 0 {
                    parts.push(format!("{failed} FAILED"));
                }
                self.status = parts.join(" · ");
            }
        }
    }

    /// Handle a path dropped onto the terminal window (arrives as pasted
    /// text, possibly quoted or backslash-escaped by the terminal).
    pub fn handle_dropped_text(&mut self, text: &str) {
        let Some(path) = parse_dropped_path(text) else {
            self.status = "Dropped text doesn't look like a file path".into();
            return;
        };
        if path.is_dir() {
            self.cwd = path;
            self.file_selected = 0;
            self.refresh_entries();
            self.focus = Focus::Files;
            return;
        }
        if !path.exists() {
            self.status = format!("Not found: {}", path.display());
            return;
        }
        if !audio::is_media_file(&path) {
            self.status = format!("Unsupported file type: {}", file_name(&path));
            return;
        }
        self.start_transcription(path);
    }

    /// Write every export format into `<output_dir>/<source>_<timestamp>/`
    /// and return that folder. Runs automatically after each transcription;
    /// `e` re-exports on demand.
    pub fn export(&mut self) -> Result<PathBuf> {
        let (Some(audio), false) = (self.current_audio.clone(), self.segments.is_empty()) else {
            anyhow::bail!("nothing to export yet");
        };
        let doc = TranscriptDoc {
            segments: &self.segments,
            source_name: file_name(&audio),
            duration_secs: self.audio_duration,
            language: self.language.as_deref(),
            model_name: self.used_model_name.as_deref(),
        };
        let written = doc.write_all(&self.output_dir)?;
        Ok(written
            .first()
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| self.output_dir.clone()))
    }
}

/// Terminals paste dropped files as text: possibly 'single-quoted',
/// "double-quoted", or with backslash-escaped spaces and parens.
fn parse_dropped_path(text: &str) -> Option<PathBuf> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    let unquoted = trimmed
        .strip_prefix('\'')
        .and_then(|s| s.strip_suffix('\''))
        .or_else(|| trimmed.strip_prefix('"').and_then(|s| s.strip_suffix('"')))
        .unwrap_or(trimmed);

    // Some terminals paste drops as file:// URLs with percent-encoding
    let unescaped = if let Some(url_path) = unquoted.strip_prefix("file://") {
        percent_decode(url_path.trim_start_matches("localhost"))
    } else {
        // Undo backslash escaping (e.g. "My\ File.mp4")
        let mut plain = String::with_capacity(unquoted.len());
        let mut chars = unquoted.chars();
        while let Some(c) = chars.next() {
            if c == '\\' {
                if let Some(next) = chars.next() {
                    plain.push(next);
                }
            } else {
                plain.push(c);
            }
        }
        plain
    };

    let expanded = if let Some(rest) = unescaped.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            PathBuf::from(home).join(rest)
        } else {
            PathBuf::from(unescaped)
        }
    } else {
        PathBuf::from(unescaped)
    };

    if expanded.is_absolute() || expanded.exists() {
        Some(expanded)
    } else {
        None
    }
}

fn percent_decode(s: &str) -> String {
    // Malformed sequences pass through unchanged; invalid UTF-8 is lossy —
    // matching what terminals need for file:// drops.
    percent_encoding::percent_decode_str(s)
        .decode_utf8_lossy()
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcribe::Event;
    use std::path::PathBuf;

    #[test]
    fn plain_absolute_path() {
        assert_eq!(
            parse_dropped_path("/tmp/file.mp4"),
            Some(PathBuf::from("/tmp/file.mp4"))
        );
    }

    #[test]
    fn backslash_escaped_spaces() {
        assert_eq!(
            parse_dropped_path("/tmp/my\\ file\\ (1).mp4 "),
            Some(PathBuf::from("/tmp/my file (1).mp4"))
        );
    }

    #[test]
    fn quoted_paths() {
        assert_eq!(
            parse_dropped_path("'/tmp/my file.mp4'"),
            Some(PathBuf::from("/tmp/my file.mp4"))
        );
        assert_eq!(
            parse_dropped_path("\"/tmp/a.wav\""),
            Some(PathBuf::from("/tmp/a.wav"))
        );
    }

    #[test]
    fn tilde_expansion() {
        let home = std::env::var("HOME").unwrap();
        assert_eq!(
            parse_dropped_path("~/x.wav"),
            Some(PathBuf::from(home).join("x.wav"))
        );
    }

    #[test]
    fn rejects_non_paths() {
        assert_eq!(parse_dropped_path(""), None);
        assert_eq!(parse_dropped_path("   "), None);
        assert_eq!(parse_dropped_path("hello world"), None);
    }

    #[test]
    fn file_url_with_percent_encoding() {
        assert_eq!(
            parse_dropped_path("file:///tmp/my%20file.mp4"),
            Some(PathBuf::from("/tmp/my file.mp4"))
        );
        assert_eq!(
            parse_dropped_path("file://localhost/tmp/a.wav"),
            Some(PathBuf::from("/tmp/a.wav"))
        );
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
        let labels: Vec<String> = app.entries.iter().map(|e| e.label()).collect();
        assert_eq!(labels, vec!["../", "z_subdir/", "a.mp4 \u{29c9}", "b.wav"]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn worker_events_drive_state_machine_and_auto_export() {
        let app_dir = tempdir();
        let mut app = test_app(app_dir.clone());
        app.output_dir = app_dir.join("out");
        app.current_audio = Some(app_dir.join("clip.wav"));

        app.handle_event(Event::LoadingModel("m.bin".into()));
        assert!(app.busy());

        app.handle_event(Event::AudioInfo {
            duration_secs: 10.0,
        });
        assert_eq!(app.work, WorkState::Transcribing { progress: 0 });
        assert_eq!(app.audio_duration, Some(10.0));

        app.handle_event(Event::Progress(40));
        assert_eq!(app.work, WorkState::Transcribing { progress: 40 });

        app.handle_event(Event::Segment(Segment {
            start_ms: 0,
            end_ms: 1000,
            text: "hi".into(),
            speaker: None,
        }));
        assert_eq!(app.segments.len(), 1);

        app.handle_event(Event::Done {
            elapsed_secs: 2.0,
            audio_secs: 10.0,
            language: Some("en".into()),
        });
        assert!(!app.busy());
        assert_eq!(app.language.as_deref(), Some("en"));
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
        assert_eq!(app.cwd, sub);

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
        app.models_dir = dir.clone();
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
        app.selected_model = None;
        app.handle_hub_event(HubEvent::Done {
            file: "ggml-x.bin".into(),
            path: path.clone(),
        });
        assert!(app.hub.download.is_none());
        assert_eq!(app.selected_model.as_ref().map(|m| m.path.clone()), Some(path));

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
        let dir = tempdir();
        // keep config::save away from the user's real config file
        std::env::set_var("TRANSCRIBE_STT_CONFIG", dir.join("config.toml"));
        std::fs::create_dir(dir.join("aa")).unwrap();
        std::fs::create_dir(dir.join("bb")).unwrap();
        std::fs::create_dir(dir.join(".hidden")).unwrap();
        let mut app = test_app(dir.clone());

        // models target: descend into aa/, then choose it
        app.dir_picker = Some(DirPicker::at(DirTarget::Models, &dir));
        let rows = app.dir_picker.as_ref().unwrap().rows();
        assert_eq!(rows[0], DirRow::UseThis);
        assert_eq!(rows[1], DirRow::Parent);
        assert_eq!(rows[2..], [DirRow::Sub(dir.join("aa")), DirRow::Sub(dir.join("bb"))]);

        app.dir_picker_key(KeyCode::Down);
        app.dir_picker_key(KeyCode::Down);
        app.dir_picker_key(KeyCode::Enter); // enter aa/
        assert_eq!(app.dir_picker.as_ref().unwrap().cwd, dir.join("aa"));
        app.dir_picker_key(KeyCode::Enter); // "use this folder"
        assert!(app.dir_picker.is_none());
        assert_eq!(app.models_dir, dir.join("aa").canonicalize().unwrap());
        assert_eq!(app.config.models_dir, Some(app.models_dir.clone()));

        // output target reuses the same component
        app.dir_picker = Some(DirPicker::at(DirTarget::Output, &dir.join("bb")));
        app.dir_picker_key(KeyCode::Enter);
        assert_eq!(app.output_dir, dir.join("bb").canonicalize().unwrap());
        assert_eq!(app.config.output_dir, Some(app.output_dir.clone()));

        // Esc closes without committing
        app.dir_picker = Some(DirPicker::at(DirTarget::Models, &dir));
        app.dir_picker_key(KeyCode::Esc);
        assert!(app.dir_picker.is_none());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn setting_models_folder_adopts_existing_models() {
        let dir = tempdir();
        std::env::set_var("TRANSCRIBE_STT_CONFIG", dir.join("config.toml"));
        let folder = dir.join("stash");
        std::fs::create_dir(&folder).unwrap();
        std::fs::write(folder.join("ggml-large-v3.bin"), b"weights").unwrap();
        std::fs::write(folder.join("other.gguf"), b"weights").unwrap();
        std::fs::write(folder.join("notes.txt"), b"not a model").unwrap();

        let mut app = test_app(dir.clone());
        app.selected_model = None;
        app.set_models_dir_path(folder.clone());

        let names: Vec<&str> = app.models.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, vec!["ggml-large-v3.bin", "other.gguf"]);
        // with nothing selected before, the scan adopts large-v3
        assert_eq!(
            app.selected_model.as_ref().map(|m| m.name.as_str()),
            Some("ggml-large-v3.bin")
        );
        assert!(app.status.contains("2 model(s) found"), "{}", app.status);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn changing_models_folder_offers_to_move_and_moves_on_yes() {
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
        app.move_prompt = None;

        app.set_models_dir_path(new.clone());
        let prompt = app.move_prompt.as_ref().expect("prompt should open");
        assert_eq!(prompt.count, 2);
        assert!(prompt.yes_selected);

        app.move_prompt_key(KeyCode::Enter); // Yes is preselected
        assert!(app.move_prompt.is_none());
        // the move runs on a worker thread; wait for its completion event
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while !app.status.starts_with("Moved") {
            assert!(std::time::Instant::now() < deadline, "move never finished");
            app.hub_pump();
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(app.status.contains("Moved 2 model(s)"), "{}", app.status);
        assert!(app.models_dir.join("ggml-a.bin").is_file());
        assert!(!old.join("ggml-a.bin").exists());
        assert_eq!(app.models.len(), 2);
        // with nothing selected before, the arrival adopts a model
        assert!(app.selected_model.is_some());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn move_prompt_no_leaves_models_in_place() {
        let dir = tempdir();
        std::env::set_var("TRANSCRIBE_STT_CONFIG", dir.join("config.toml"));
        let old = dir.join("old");
        let new = dir.join("new");
        std::fs::create_dir(&old).unwrap();
        std::fs::create_dir(&new).unwrap();
        std::fs::write(old.join("ggml-a.bin"), b"weights").unwrap();

        let mut app = test_app(dir.clone());
        app.set_models_dir_path(old.clone());
        app.move_prompt = None;
        app.set_models_dir_path(new.clone());
        assert!(app.move_prompt.is_some());

        // toggle to No, confirm with Enter
        app.move_prompt_key(KeyCode::Left);
        app.move_prompt_key(KeyCode::Enter);
        assert!(app.move_prompt.is_none());
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
        let dir = tempdir();
        std::env::set_var("TRANSCRIBE_STT_CONFIG", dir.join("config.toml"));
        let mut app = test_app(dir.clone());
        app.selected_model = None;

        let model = ModelFile {
            path: dir.join("ggml-x.bin"),
            name: "ggml-x.bin".into(),
            size_bytes: 4,
        };
        app.choose_model(model.clone());
        assert_eq!(app.config.default_model.as_deref(), Some("ggml-x.bin"));
        assert_eq!(app.selected_model.as_ref().map(|m| m.name.as_str()), Some("ggml-x.bin"));
        // lazy by design: selection is metadata; loading happens at job time
        assert!(
            app.status.contains("loads when the next transcription starts"),
            "{}",
            app.status
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn split_mode_cycles_and_lands_in_the_job() {
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
