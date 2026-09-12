//! Transcription backends. Every engine lives in its own submodule and
//! implements the common [`Engine`] interface; `Backends` owns one
//! instance of each and picks the right one for a job by the model's
//! on-disk shape. Adding a new backend means a new submodule plus a
//! routing arm in [`Backends::for_model`] — nothing outside this module
//! changes.

mod mlx;
mod whisper_metal;

pub(crate) use mlx::live_python;
pub use mlx::mlx_audio_available;

use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::Sender;
use std::sync::Arc;

use super::{Event, Job};

/// The interface every transcription backend implements. `run` executes
/// a whole job, streaming `Event`s and honouring `cancel`; the engine
/// decides how (and whether) to keep a model resident between jobs.
pub(crate) trait Engine {
    fn run(
        &mut self,
        job: &Job,
        events: &Sender<Event>,
        cancel: &Arc<AtomicBool>,
    ) -> anyhow::Result<()>;

    /// Whether model memory is resident right now.
    fn loaded(&self) -> bool;

    /// Release any resident model memory.
    fn unload(&mut self);
}

/// One instance of every backend, owned by the worker thread for the
/// whole session so residency (whisper's kept context) survives between
/// jobs.
pub(super) struct Backends {
    whisper_metal: whisper_metal::WhisperEngine,
    mlx: mlx::MlxEngine,
}

impl Backends {
    pub(super) fn new() -> Self {
        Self {
            whisper_metal: whisper_metal::WhisperEngine::new(),
            mlx: mlx::MlxEngine::new(),
        }
    }

    /// The engine that runs this model, decided by shape: a directory is
    /// an MLX model (config.json + weights), a single GGML/GGUF file is
    /// whisper.cpp's.
    pub(super) fn for_model(&mut self, model: &Path) -> &mut dyn Engine {
        if model.is_dir() {
            &mut self.mlx
        } else {
            &mut self.whisper_metal
        }
    }

    /// Whether any backend holds model memory.
    pub(super) fn loaded(&self) -> bool {
        self.whisper_metal.loaded() || self.mlx.loaded()
    }

    /// Release every backend's resident model memory.
    pub(super) fn unload_all(&mut self) {
        self.whisper_metal.unload();
        self.mlx.unload();
    }
}
