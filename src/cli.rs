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

    /// Start live microphone transcription in the TUI (MLX ASR models)
    #[arg(long, conflicts_with_all = ["headless", "path", "download_test_model"])]
    pub realtime: bool,

    /// Model library folder for this session (e.g. an external drive)
    #[arg(long, value_name = "DIR")]
    pub models_dir: Option<PathBuf>,

    /// Microphone name (built-in, USB or connected Bluetooth); default: system input
    #[arg(long, value_name = "NAME", conflicts_with_all = ["headless", "download_test_model"])]
    pub input_device: Option<String>,

    /// List available microphone names and exit without recording
    #[arg(long)]
    pub list_input_devices: bool,

    /// Serve transcript updates over local SSE and WebSocket (default port: 8765)
    #[arg(long, value_name = "PORT", num_args = 0..=1, require_equals = true,
        default_missing_value = "8765", conflicts_with_all = ["download_test_model", "list_input_devices"])]
    pub serve: Option<u16>,

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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn text_service_is_opt_in_and_accepts_an_optional_port() {
        assert!(Args::try_parse_from(["transcribe-stt"])
            .unwrap()
            .serve
            .is_none());
        assert_eq!(
            Args::try_parse_from(["transcribe-stt", "--serve"])
                .unwrap()
                .serve,
            Some(8765)
        );
        assert_eq!(
            Args::try_parse_from(["transcribe-stt", "--serve=9000", "--realtime"])
                .unwrap()
                .serve,
            Some(9000)
        );
        assert_eq!(
            Args::try_parse_from(["transcribe-stt", "--headless", "--serve=9000", "clip.wav"])
                .unwrap()
                .serve,
            Some(9000)
        );
        assert!(Args::try_parse_from(["transcribe-stt", "--serve=70000"]).is_err());
        assert!(
            Args::try_parse_from(["transcribe-stt", "--serve", "--list-input-devices"]).is_err()
        );
    }

    #[test]
    fn live_flags_require_tui_and_accept_external_library() {
        let args = Args::try_parse_from([
            "transcribe-stt",
            "--realtime",
            "--models-dir",
            "/Volumes/models",
            "-l",
            "es",
            "--input-device",
            "AirPods Microphone",
        ])
        .unwrap();
        assert!(args.realtime);
        assert_eq!(args.input_device.as_deref(), Some("AirPods Microphone"));
        assert!(
            Args::try_parse_from(["transcribe-stt", "--list-input-devices"])
                .unwrap()
                .list_input_devices
        );
        assert_eq!(args.models_dir, Some(PathBuf::from("/Volumes/models")));
        assert_eq!(args.language.as_deref(), Some("es"));
        assert!(Args::try_parse_from(["transcribe-stt", "--realtime", "--headless"]).is_err());
        assert!(Args::try_parse_from(["transcribe-stt", "--realtime", "clip.wav"]).is_err());
    }
}
