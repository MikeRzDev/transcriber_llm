//! Terminal speech-to-text client for whisper.cpp models with Metal
//! acceleration. The binary in `main.rs` is a thin wrapper around this
//! library so integration tests can drive the internals directly.

pub mod app;
pub mod audio;
pub mod cli;
pub mod config;
pub mod diarize;
pub mod export;
pub mod format;
pub mod headless;
pub mod hub;
pub mod hw;
pub mod live_benchmark;
pub mod models;
pub mod pyrt;
pub mod split;
pub mod stats;
pub mod stream_service;
pub mod transcribe;
pub mod tui;
pub mod ui;

use anyhow::Result;

/// Dispatch a parsed command line to the right mode: one-shot model
/// download, headless transcription, or the interactive TUI.
pub fn run(args: cli::Args) -> Result<()> {
    if args.list_input_devices {
        println!("System default (omit --input-device)");
        for name in audio::capture::input_devices()? {
            println!("{name}");
        }
        return Ok(());
    }
    if args.download_test_model {
        return headless::download_test_model();
    }
    if args.headless {
        return headless::run(&args);
    }
    tui::run(args)
}
