use std::path::PathBuf;

use clap::Parser;

/// Terminal speech-to-text client for whisper.cpp (Metal) and MLX
/// (mlx-audio) voice models.
#[derive(Parser, Debug)]
#[command(name = "transcribe-stt", version)]
pub struct Args {
    /// Directory to browse, or audio file to transcribe
    pub path: Option<PathBuf>,

    /// Model to use: a .bin/.gguf file (whisper.cpp) or an MLX model
    /// folder; default: auto-detect in the models folder
    #[arg(short, long, value_name = "FILE|DIR")]
    pub model: Option<PathBuf>,

    /// Transcribe PATH without the TUI, print segments to stdout
    #[arg(long)]
    pub headless: bool,

    /// Diarization strategy: off, auto (recommended per model), tdrz
    /// (tinydiarize, 2 speakers, English), embedding (any model,
    /// multi-speaker), pyannote (community-1, needs HF token + PyTorch).
    /// Bare --diarize means auto; pass a strategy as --diarize=embedding
    /// (the = keeps the audio path unambiguous).
    #[arg(
        long,
        value_name = "STRATEGY",
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "auto"
    )]
    pub diarize: Option<String>,

    /// Known speaker count for embedding diarization (pins the
    /// clustering); default: auto-detect
    #[arg(long, value_name = "N")]
    pub speakers: Option<u8>,

    /// Language hint (ISO 639-1, e.g. en, es); default: auto-detect
    #[arg(short, long, value_name = "CODE")]
    pub language: Option<String>,

    /// Fetch the smallest whisper model (ggml-tiny, ~75 MB) into the
    /// models folder for quick testing
    #[arg(long)]
    pub download_test_model: bool,
}
