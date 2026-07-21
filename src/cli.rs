use std::path::PathBuf;

use anyhow::{bail, Result};

pub struct Args {
    pub path: Option<PathBuf>,
    pub model: Option<PathBuf>,
    pub headless: bool,
    pub diarize: bool,
    pub language: Option<String>,
    pub download_test_model: bool,
}

pub fn parse_args() -> Result<Args> {
    let mut args = Args {
        path: None,
        model: None,
        headless: false,
        diarize: false,
        language: None,
        download_test_model: false,
    };
    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-m" | "--model" => {
                args.model = Some(PathBuf::from(
                    iter.next()
                        .ok_or_else(|| anyhow::anyhow!("--model needs a path"))?,
                ));
            }
            "--headless" => args.headless = true,
            "--diarize" => args.diarize = true,
            "--download-test-model" => args.download_test_model = true,
            "-l" | "--language" => {
                args.language = Some(
                    iter.next()
                        .ok_or_else(|| anyhow::anyhow!("--language needs a code, e.g. en"))?,
                );
            }
            "-h" | "--help" => {
                println!(
                    "transcribe-stt — terminal speech-to-text client for whisper.cpp models (Metal)\n\n\
                     usage: transcribe-stt [OPTIONS] [PATH]\n\n\
                     PATH                 directory to browse, or audio file to transcribe\n\
                     -m, --model <FILE>   model to use (.bin / .gguf); default: auto-detect in the models folder\n\
                     --headless           transcribe PATH without the TUI, print segments to stdout\n\
                     --diarize            label speaker turns (requires a tdrz model, English)\n\
                     -l, --language <CODE> language hint (ISO 639-1, e.g. en, es); default: auto-detect\n\
                     --download-test-model fetch the smallest whisper model (ggml-tiny, ~75 MB)\n\
                     \x20                    into the models folder for quick testing"
                );
                std::process::exit(0);
            }
            other if !other.starts_with('-') => args.path = Some(PathBuf::from(other)),
            other => bail!("unknown flag: {other}"),
        }
    }
    Ok(args)
}
