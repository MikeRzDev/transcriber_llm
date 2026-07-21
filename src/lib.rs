//! Terminal speech-to-text client for whisper.cpp models with Metal
//! acceleration. The binary in `main.rs` is a thin wrapper around this
//! library so integration tests can drive the internals directly.

pub mod app;
pub mod audio;
pub mod config;
pub mod export;
pub mod hub;
pub mod models;
pub mod split;
pub mod stats;
pub mod transcribe;
pub mod ui;
