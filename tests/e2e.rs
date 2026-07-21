//! End-to-end test of the real pipeline: binary -> model load (Metal) ->
//! decode -> transcribe -> stdout. Needs the large-v3 model and the demo
//! sample, so it is #[ignore]d by default; run with `cargo test -- --ignored`.

use std::path::Path;
use std::process::Command;

#[test]
#[ignore = "requires models/ggml-large-v3.bin and samples/demo.wav (~10s on M-series)"]
fn headless_transcribes_demo_wav() {
    assert!(
        Path::new("models/ggml-large-v3.bin").exists(),
        "place ggml-large-v3.bin in ./models (download via Model management)"
    );
    assert!(Path::new("samples/demo.wav").exists());

    let output = Command::new(env!("CARGO_BIN_EXE_transcribe-stt"))
        .args(["--headless", "-m", "models/ggml-large-v3.bin", "samples/demo.wav"])
        .output()
        .expect("failed to run transcribe-stt");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout).to_lowercase();
    // The generated demo clip says these words
    assert!(stdout.contains("whisper"), "stdout: {stdout}");
    assert!(stdout.contains("quick brown fox"), "stdout: {stdout}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("language: en"), "stderr: {stderr}");
}

#[test]
#[ignore = "requires the model, samples/demo_video.mp4, and ffmpeg"]
fn headless_extracts_and_transcribes_video() {
    assert!(Path::new("models/ggml-large-v3.bin").exists());
    assert!(Path::new("samples/demo_video.mp4").exists());

    let output = Command::new(env!("CARGO_BIN_EXE_transcribe-stt"))
        .args([
            "--headless",
            "-m",
            "models/ggml-large-v3.bin",
            "samples/demo_video.mp4",
        ])
        .output()
        .expect("failed to run transcribe-stt");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout).to_lowercase();
    assert!(stdout.contains("quick brown fox"), "stdout: {stdout}");
}

#[test]
#[ignore = "requires models/ggml-small.en-tdrz.bin and samples/conversation.wav"]
fn headless_diarize_emits_labeled_transcript() {
    assert!(Path::new("models/ggml-small.en-tdrz.bin").exists());
    assert!(Path::new("samples/conversation.wav").exists());

    let output = Command::new(env!("CARGO_BIN_EXE_transcribe-stt"))
        .args([
            "--headless",
            "--diarize",
            "-m",
            "models/ggml-small.en-tdrz.bin",
            "samples/conversation.wav",
        ])
        .output()
        .expect("failed to run transcribe-stt");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    // The labeled final pass must be emitted with speaker prefixes;
    // how many distinct speakers get detected depends on the audio.
    assert!(stdout.contains("--- diarized ---"), "stdout: {stdout}");
    assert!(stdout.contains("Speaker A:"), "stdout: {stdout}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("language: en"), "stderr: {stderr}");
}
