use std::path::PathBuf;

use clap::Parser;

/// Terminal speech-to-text client for whisper.cpp models (Metal).
#[derive(Parser, Debug)]
#[command(name = "transcribe-stt", version)]
pub struct Args {
    /// Directory to browse, or audio file to transcribe
    pub path: Option<PathBuf>,

    /// Model to use (.bin / .gguf); default: auto-detect in the models folder
    #[arg(short, long, value_name = "FILE")]
    pub model: Option<PathBuf>,

    /// Transcribe PATH without the TUI, print segments to stdout
    #[arg(long)]
    pub headless: bool,

    /// Label speaker turns (requires a tdrz model, English)
    #[arg(long)]
    pub diarize: bool,

    /// Language hint (ISO 639-1, e.g. en, es); default: auto-detect
    #[arg(short, long, value_name = "CODE")]
    pub language: Option<String>,

    /// Fetch the smallest whisper model (ggml-tiny, ~75 MB) into the
    /// models folder for quick testing
    #[arg(long)]
    pub download_test_model: bool,
}
