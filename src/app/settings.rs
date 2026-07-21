//! The settings modal: its rows, folder choices (directory picker), the
//! move-models prompt, and the toggles persisted to config.

use std::path::{Path, PathBuf};

use crossterm::event::KeyCode;

use crate::app::App;
use crate::config;
use crate::hub::HubEvent;
use crate::models;

/// The rows of the settings menu, in display order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SettingsRow {
    DefaultModel,
    ModelsFolder,
    OutputFolder,
    ModelManagement,
    Diarize,
    SplitMode,
    Language,
}

impl SettingsRow {
    pub const ALL: [Self; 7] = [
        Self::DefaultModel,
        Self::ModelsFolder,
        Self::OutputFolder,
        Self::ModelManagement,
        Self::Diarize,
        Self::SplitMode,
        Self::Language,
    ];

    pub fn next(self) -> Self {
        let i = Self::ALL.iter().position(|r| *r == self).unwrap_or(0);
        Self::ALL[(i + 1) % Self::ALL.len()]
    }

    pub fn prev(self) -> Self {
        let i = Self::ALL.iter().position(|r| *r == self).unwrap_or(0);
        Self::ALL[(i + Self::ALL.len() - 1) % Self::ALL.len()]
    }
}

/// State of the settings modal and the flows it can open on top of
/// itself (folder picker, move prompt, inline language entry).
pub struct SettingsUi {
    pub open: bool,
    pub selected: SettingsRow,
    /// Some(text) while the language code is being edited
    pub language_input: Option<String>,
    /// Some while a folder is being chosen via the directory browser
    pub dir_picker: Option<DirPicker>,
    /// Some while asking whether to move models into the new folder
    pub move_prompt: Option<MovePrompt>,
}

impl SettingsUi {
    pub(crate) fn new() -> Self {
        Self {
            open: false,
            selected: SettingsRow::DefaultModel,
            language_input: None,
            dir_picker: None,
            move_prompt: None,
        }
    }
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

impl App {
    pub(crate) fn settings_key(&mut self, code: KeyCode) {
        if let Some(input) = &mut self.settings.language_input {
            match code {
                KeyCode::Char(c) => input.push(c),
                KeyCode::Backspace => {
                    input.pop();
                }
                KeyCode::Enter => {
                    let text = self.settings.language_input.take().unwrap_or_default();
                    self.set_language(&text);
                }
                KeyCode::Esc => self.settings.language_input = None,
                _ => {}
            }
            return;
        }
        match code {
            KeyCode::Esc | KeyCode::Char('s') | KeyCode::Char('q') => {
                self.settings.open = false;
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.settings.selected = self.settings.selected.prev();
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.settings.selected = self.settings.selected.next();
            }
            KeyCode::Enter => match self.settings.selected {
                SettingsRow::DefaultModel => self.open_model_picker(),
                SettingsRow::ModelsFolder => self.open_dir_picker(DirTarget::Models),
                SettingsRow::OutputFolder => self.open_dir_picker(DirTarget::Output),
                SettingsRow::ModelManagement => {
                    self.settings.open = false;
                    self.open_hub();
                }
                SettingsRow::Diarize => self.toggle_diarize(),
                SettingsRow::SplitMode => self.cycle_split_mode(),
                SettingsRow::Language => {
                    self.settings.language_input =
                        Some(self.config.language.clone().unwrap_or_default())
                }
            },
            _ => {}
        }
    }

    /// Toggle diarization and persist the choice.
    pub fn toggle_diarize(&mut self) {
        self.config.diarize = !self.config.diarize;
        let _ = config::save(&self.config);
        self.status = if self.config.diarize {
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

    /// Change the models folder, persist it, and rescan. If the previous
    /// folder still holds models, offer to move them over.
    pub fn set_models_dir_path(&mut self, path: PathBuf) {
        if let Err(e) = std::fs::create_dir_all(&path) {
            self.status = format!("Cannot use folder {}: {e}", path.display());
            return;
        }
        let old_dir = self.library.dir.clone();
        let old_count = self.library.models.len();
        let path = path.canonicalize().unwrap_or(path);
        self.library.dir = path.clone();
        self.config.models_dir = Some(path);
        // Scan the new folder: every supported model inside joins the list
        self.refresh_models();
        self.adopt_selection();
        if let Err(e) = config::save(&self.config) {
            self.status = format!("Saving config failed: {e}");
        } else {
            self.status = format!(
                "Models folder set to {} — {} model(s) found",
                self.library.dir.display(),
                self.library.models.len()
            );
        }
        if old_dir != self.library.dir && old_count > 0 {
            self.settings.move_prompt = Some(MovePrompt {
                from: old_dir,
                to: self.library.dir.clone(),
                count: old_count,
                yes_selected: true,
            });
        }
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
            DirTarget::Models => self.library.dir.clone(),
            DirTarget::Output => self.output_dir.clone(),
        };
        self.settings.dir_picker = Some(DirPicker::at(target, &start));
    }

    pub fn dir_picker_key(&mut self, code: KeyCode) {
        let Some(picker) = &mut self.settings.dir_picker else {
            return;
        };
        match code {
            KeyCode::Esc => self.settings.dir_picker = None,
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
                    self.settings.dir_picker = None;
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

    pub fn move_prompt_key(&mut self, code: KeyCode) {
        let Some(prompt) = &mut self.settings.move_prompt else {
            return;
        };
        match code {
            KeyCode::Esc | KeyCode::Char('n') => {
                self.settings.move_prompt = None;
                self.status = "Models left in the previous folder".into();
            }
            KeyCode::Left
            | KeyCode::Right
            | KeyCode::Tab
            | KeyCode::Char('h')
            | KeyCode::Char('l') => prompt.yes_selected = !prompt.yes_selected,
            KeyCode::Char('y') => {
                let prompt = self.settings.move_prompt.take().unwrap();
                self.start_move_models(prompt.from, prompt.to, prompt.count);
            }
            KeyCode::Enter => {
                let prompt = self.settings.move_prompt.take().unwrap();
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
}
