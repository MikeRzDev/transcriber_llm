//! Transcription: the job/segment/event types, the worker thread
//! (`worker`) that executes jobs, and the engine implementations behind
//! the common `backend::Engine` interface — `backend/whisper_metal`
//! drives whisper.cpp in-process via whisper-rs, `backend/mlx` runs
//! directory (safetensors) models by shelling out to mlx-audio.

mod backend;
mod worker;

pub use backend::mlx_audio_available;
pub use worker::{spawn, Transcriber};

use std::path::PathBuf;

use crate::diarize::{DiarizeMethod, DiarizeModelChoice};
use crate::split::SplitMode;

pub struct Job {
    pub model: PathBuf,
    pub audio: PathBuf,
    /// Resolved diarization method: tdrz runs inline in the whisper
    /// engine, the embedding pipeline as a worker post-pass over any
    /// engine's output
    pub diarize: DiarizeMethod,
    /// Which catalog models the embedding pipeline uses (defaults apply
    /// when unset); irrelevant for the other methods
    pub diarize_models: DiarizeModelChoice,
    /// Known speaker count for the embedding pipeline; None = auto-detect
    pub diarize_speakers: Option<u8>,
    /// ISO 639-1 hint; None = auto-detect
    pub language: Option<String>,
    /// Client-side chunking strategy for long audio
    pub split_mode: SplitMode,
}

#[derive(Debug, Clone)]
pub struct Segment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
    /// Speaker index (0 = A, 1 = B, …) from diarization; None when no
    /// diarizer labeled this segment.
    pub speaker: Option<u8>,
}

#[derive(Debug)]
pub enum Event {
    LoadingModel(String),
    /// 0–100 while the model file is read and initialized
    LoadProgress(i32),
    ModelReady {
        load_secs: f32,
    },
    /// The resident model is being released (user switched models)
    Unloading,
    Unloaded,
    Decoding,
    /// 0–100 while ffmpeg rips the audio track out of a video (or
    /// converts an exotic codec); negative means the total duration is
    /// unknown (UI shows an indeterminate spinner)
    DecodeProgress(i32),
    AudioInfo {
        duration_secs: f32,
    },
    /// 0–100 while transcribing; negative means the engine reports no
    /// fine-grained progress (UI shows an indeterminate spinner)
    Progress(i32),
    /// A line of live output from a subprocess engine (mlx-audio, pip
    /// during runtime setup) — surfaced in the status line so long
    /// otherwise-silent phases always show what is happening
    EngineLog(String),
    /// Liveness signal sent while a subprocess engine runs without
    /// printing anything (seconds since it started) — proves the job is
    /// alive even when the child is completely silent
    EngineHeartbeat(u64),
    Segment(Segment),
    /// Re-issued full transcript with speaker labels, sent after a
    /// diarization pass completes (streamed segments carry no speaker).
    SegmentsFinal(Vec<Segment>),
    Done {
        elapsed_secs: f32,
        audio_secs: f32,
        language: Option<String>,
    },
    Cancelled,
    Error(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::channel;

    /// Unload with nothing loaded is a no-op, and the worker must still
    /// shut down cleanly afterwards (no hang, no panic).
    #[test]
    fn unload_is_safe_with_no_model_loaded() {
        let (tx, _rx) = channel();
        let mut transcriber = spawn(tx);
        transcriber.request_unload();
        transcriber.request_unload();
        transcriber.shutdown();
    }
}
