//! Local model management: the models folder, the picker modal, and
//! keeping the default selection valid.

use std::path::PathBuf;

use crossterm::event::KeyCode;

use crate::app::App;
use crate::config;
use crate::models::{self, ModelFile};

/// The models on disk and the current default selection.
pub struct ModelLibrary {
    /// The models folder being scanned
    pub dir: PathBuf,
    pub models: Vec<ModelFile>,
    /// The default model for the next job (metadata only — loading is lazy)
    pub selected: Option<ModelFile>,
}

/// The model picker modal.
#[derive(Default)]
pub struct ModelPicker {
    pub open: bool,
    pub selected: usize,
}

impl App {
    /// Open the model picker preselected on the current default.
    pub fn open_model_picker(&mut self) {
        self.refresh_models();
        self.picker.selected = self
            .library
            .selected
            .as_ref()
            .and_then(|sel| self.library.models.iter().position(|m| m.path == sel.path))
            .unwrap_or(0);
        self.picker.open = true;
    }

    pub(crate) fn model_picker_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Esc | KeyCode::Char('m') | KeyCode::Char('q') => {
                self.picker.open = false;
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.picker.selected = self.picker.selected.saturating_sub(1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if !self.library.models.is_empty() {
                    self.picker.selected =
                        (self.picker.selected + 1).min(self.library.models.len() - 1);
                }
            }
            KeyCode::Enter => {
                if let Some(model) = self.library.models.get(self.picker.selected).cloned() {
                    self.choose_model(model);
                }
                self.picker.open = false;
            }
            _ => {}
        }
    }

    pub fn refresh_models(&mut self) {
        self.library.models = models::scan_models(&self.library.dir);
        self.picker.selected = self
            .picker
            .selected
            .min(self.library.models.len().saturating_sub(1));
    }

    pub fn find_tdrz_model(&self) -> Option<ModelFile> {
        self.library
            .models
            .iter()
            .find(|m| models::is_tdrz(&m.name))
            .cloned()
    }

    /// Set a new default model and persist it. Models load lazily: the
    /// selection here is metadata only, and switching away from a loaded
    /// model releases its memory right away instead of at the next job.
    pub fn choose_model(&mut self, model: ModelFile) {
        let changed = self
            .library
            .selected
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
        self.library.selected = Some(model);
        if let Err(e) = config::save(&self.config) {
            self.status = format!("Model selected, but saving config failed: {e}");
        }
    }

    /// Keep a still-valid selection; otherwise adopt what the folder has
    /// (configured default first, then large-v3, then whatever exists).
    pub(crate) fn adopt_selection(&mut self) {
        let selection_gone = self
            .library
            .selected
            .as_ref()
            .map(|m| !m.path.exists())
            .unwrap_or(true);
        if selection_gone {
            self.library.selected =
                models::pick_default(&self.library.models, self.config.default_model.as_deref())
                    .cloned();
        }
    }
}
