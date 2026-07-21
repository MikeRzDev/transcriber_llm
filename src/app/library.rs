//! Local model management: scanning the models folder and keeping the
//! default selection valid.

use crate::app::App;
use crate::config;
use crate::models::{self, ModelFile};

impl App {
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
