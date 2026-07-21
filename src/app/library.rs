//! Local model management: scanning the models folder and keeping the
//! default selection valid.

use crossterm::event::KeyCode;

use crate::app::App;
use crate::config;
use crate::models::{self, ModelFile};

impl App {
    /// Open the model picker preselected on the current default.
    pub fn open_model_picker(&mut self) {
        self.refresh_models();
        self.model_picker_selected = self
            .selected_model
            .as_ref()
            .and_then(|sel| self.models.iter().position(|m| m.path == sel.path))
            .unwrap_or(0);
        self.model_picker_open = true;
    }

    pub(crate) fn model_picker_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Esc | KeyCode::Char('m') | KeyCode::Char('q') => {
                self.model_picker_open = false;
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.model_picker_selected = self.model_picker_selected.saturating_sub(1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if !self.models.is_empty() {
                    self.model_picker_selected =
                        (self.model_picker_selected + 1).min(self.models.len() - 1);
                }
            }
            KeyCode::Enter => {
                if let Some(model) = self.models.get(self.model_picker_selected).cloned() {
                    self.choose_model(model);
                }
                self.model_picker_open = false;
            }
            _ => {}
        }
    }

    pub fn refresh_models(&mut self) {
        self.models = models::scan_models(&self.models_dir);
        self.model_picker_selected = self
            .model_picker_selected
            .min(self.models.len().saturating_sub(1));
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

    /// Keep a still-valid selection; otherwise adopt what the folder has
    /// (configured default first, then large-v3, then whatever exists).
    pub(crate) fn adopt_selection(&mut self) {
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
}
