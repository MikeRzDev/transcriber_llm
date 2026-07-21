use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::diarize::{DiarizeModelChoice, DiarizeStrategy};
use crate::export::ExportFormat;
use crate::split::SplitMode;

/// Persisted settings, stored as simple `key = value` lines in
/// ~/.config/transcribe-stt/config.toml
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub models_dir: Option<PathBuf>,
    pub output_dir: Option<PathBuf>,
    pub default_model: Option<String>,
    /// Diarization strategy applied to every transcription
    pub diarize: DiarizeStrategy,
    /// Which diarization catalog models the embedding pipeline uses
    /// (on-disk names; None = the role's default)
    pub diarize_models: DiarizeModelChoice,
    /// Known speaker count for the embedding pipeline; None = auto-detect.
    /// Fixing the count when it is known constrains the clustering and
    /// noticeably improves label quality.
    pub diarize_speakers: Option<u8>,
    /// ISO 639-1 code passed to whisper; None = auto-detect
    pub language: Option<String>,
    /// Hugging Face access token for gated models (pyannote community-1)
    /// and authenticated hub downloads; None = rely on the environment /
    /// `hf auth login`
    pub hf_token: Option<String>,
    /// Chunking strategy for long audio
    pub split_mode: SplitMode,
    /// Formats written after each transcription — never empty
    pub export_formats: Vec<ExportFormat>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            models_dir: None,
            output_dir: None,
            default_model: None,
            diarize: DiarizeStrategy::Off,
            diarize_models: DiarizeModelChoice::default(),
            diarize_speakers: None,
            language: None,
            hf_token: None,
            split_mode: SplitMode::default(),
            export_formats: ExportFormat::ALL.to_vec(),
        }
    }
}

/// Export the configured Hugging Face token into the process environment
/// (`HF_TOKEN`), where everything that needs it — the pyannote runner's
/// Python subprocess, the hub's authenticated downloads, the HF stack's
/// own token discovery — already looks. An HF_TOKEN set by the user's
/// shell wins over the config value.
pub fn apply_hf_token(config: &Config) {
    if std::env::var_os("HF_TOKEN").is_none() {
        if let Some(token) = &config.hf_token {
            std::env::set_var("HF_TOKEN", token);
        }
    }
}

/// Comma list → formats in canonical order; unknown tokens are dropped
/// and an empty result falls back to every format (the invariant is "at
/// least one", so a hand-emptied config key must not disable exporting).
fn parse_formats(value: &str) -> Vec<ExportFormat> {
    let formats: Vec<ExportFormat> = ExportFormat::ALL
        .into_iter()
        .filter(|f| value.split(',').any(|tok| ExportFormat::parse(tok) == Some(*f)))
        .collect();
    if formats.is_empty() {
        ExportFormat::ALL.to_vec()
    } else {
        formats
    }
}

fn render_formats(formats: &[ExportFormat]) -> String {
    formats
        .iter()
        .map(|f| f.key())
        .collect::<Vec<_>>()
        .join(",")
}

fn config_path() -> Option<PathBuf> {
    // Test isolation hook: point config at a scratch file
    if let Some(p) = std::env::var_os("TRANSCRIBE_STT_CONFIG") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var_os("HOME")?;
    Some(
        PathBuf::from(home)
            .join(".config")
            .join("transcribe-stt")
            .join("config.toml"),
    )
}

/// Both default folders live under one base: ~/Documents/llm_transcribe on
/// macOS, ~/llm_transcribe elsewhere.
fn default_base() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    if cfg!(target_os = "macos") {
        home.join("Documents").join("llm_transcribe")
    } else {
        home.join("llm_transcribe")
    }
}

pub fn default_models_dir() -> PathBuf {
    default_base().join("models")
}

pub fn default_output_dir() -> PathBuf {
    default_base().join("output")
}

/// Raw mirror of the on-disk file: every field optional and stringly so a
/// well-formed TOML file always deserializes regardless of which keys it
/// holds. Normalization (empty values, "auto" language, unknown split
/// modes) happens in `into_config`.
#[derive(Default, Deserialize, Serialize)]
struct ConfigToml {
    #[serde(skip_serializing_if = "Option::is_none")]
    models_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    default_model: Option<String>,
    /// Legacy on/off toggle from before strategies existed — read only
    /// (true meant tinydiarize), never written back.
    #[serde(skip_serializing_if = "Option::is_none")]
    diarize: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    diarize_strategy: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    diarize_segmentation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    diarize_embedding: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    diarize_speakers: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    split_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hf_token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    export_formats: Option<String>,
}

impl ConfigToml {
    fn into_config(self) -> Config {
        fn non_empty(value: Option<String>) -> Option<String> {
            value.filter(|v| !v.is_empty())
        }
        Config {
            models_dir: non_empty(self.models_dir).map(PathBuf::from),
            output_dir: non_empty(self.output_dir).map(PathBuf::from),
            default_model: non_empty(self.default_model),
            diarize: self
                .diarize_strategy
                .as_deref()
                .and_then(DiarizeStrategy::parse)
                .unwrap_or(match self.diarize {
                    Some(true) => DiarizeStrategy::Tdrz,
                    _ => DiarizeStrategy::Off,
                }),
            diarize_models: DiarizeModelChoice {
                segmentation: non_empty(self.diarize_segmentation),
                embedding: non_empty(self.diarize_embedding),
            },
            diarize_speakers: self.diarize_speakers.filter(|n| *n > 0),
            language: non_empty(self.language).filter(|lang| lang != "auto"),
            hf_token: non_empty(self.hf_token),
            split_mode: non_empty(self.split_mode)
                .and_then(|mode| SplitMode::parse(&mode))
                .unwrap_or_default(),
            export_formats: self
                .export_formats
                .as_deref()
                .map(parse_formats)
                .unwrap_or_else(|| ExportFormat::ALL.to_vec()),
        }
    }

    fn from_config(config: &Config) -> Self {
        Self {
            models_dir: config.models_dir.as_ref().map(|p| p.display().to_string()),
            output_dir: config.output_dir.as_ref().map(|p| p.display().to_string()),
            default_model: config.default_model.clone(),
            diarize: None,
            diarize_strategy: (config.diarize != DiarizeStrategy::Off)
                .then(|| config.diarize.key().to_string()),
            diarize_segmentation: config.diarize_models.segmentation.clone(),
            diarize_embedding: config.diarize_models.embedding.clone(),
            diarize_speakers: config.diarize_speakers,
            split_mode: (config.split_mode != SplitMode::Auto)
                .then(|| config.split_mode.as_str().to_string()),
            language: config.language.clone(),
            hf_token: config.hf_token.clone(),
            export_formats: (config.export_formats != ExportFormat::ALL)
                .then(|| render_formats(&config.export_formats)),
        }
    }
}

fn parse_str(contents: &str) -> Config {
    match toml::from_str::<ConfigToml>(contents) {
        Ok(raw) => raw.into_config(),
        // Legacy hand-edited files may hold unquoted values or stray lines
        // that strict TOML rejects; those keep loading via the original
        // line parser.
        Err(_) => parse_lenient(contents),
    }
}

fn parse_lenient(contents: &str) -> Config {
    let mut config = Config::default();
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let value = value.trim().trim_matches('"').trim_matches('\'');
        if value.is_empty() {
            continue;
        }
        match key.trim() {
            "models_dir" => config.models_dir = Some(PathBuf::from(value)),
            "output_dir" => config.output_dir = Some(PathBuf::from(value)),
            "default_model" => config.default_model = Some(value.to_string()),
            // legacy on/off toggle: true meant tinydiarize
            "diarize" if value == "true" => config.diarize = DiarizeStrategy::Tdrz,
            "diarize_strategy" => {
                if let Some(strategy) = DiarizeStrategy::parse(value) {
                    config.diarize = strategy;
                }
            }
            "diarize_segmentation" => {
                config.diarize_models.segmentation = Some(value.to_string())
            }
            "diarize_embedding" => config.diarize_models.embedding = Some(value.to_string()),
            "diarize_speakers" => {
                config.diarize_speakers = value.parse::<u8>().ok().filter(|n| *n > 0)
            }
            "split_mode" => {
                if let Some(mode) = SplitMode::parse(value) {
                    config.split_mode = mode;
                }
            }
            "language" if value != "auto" => config.language = Some(value.to_string()),
            "hf_token" => config.hf_token = Some(value.to_string()),
            "export_formats" => config.export_formats = parse_formats(value),
            _ => {}
        }
    }
    config
}

fn render(config: &Config) -> String {
    // Serializing a struct of scalars cannot fail in practice.
    let body = toml::to_string(&ConfigToml::from_config(config)).unwrap_or_default();
    format!("# transcribe-stt settings\n{body}")
}

pub fn load() -> Config {
    config_path()
        .and_then(|path| std::fs::read_to_string(path).ok())
        .map(|contents| parse_str(&contents))
        .unwrap_or_default()
}

pub fn save(config: &Config) -> anyhow::Result<()> {
    let Some(path) = config_path() else {
        return Ok(());
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, render(config))?;
    Ok(())
}

/// Resolution order: $TRANSCRIBE_STT_MODELS > config > platform default
pub fn resolve_models_dir(config: &Config) -> PathBuf {
    if let Some(env_dir) = std::env::var_os("TRANSCRIBE_STT_MODELS") {
        return PathBuf::from(env_dir);
    }
    config.models_dir.clone().unwrap_or_else(default_models_dir)
}

/// Resolution order: config > platform default
pub fn resolve_output_dir(config: &Config) -> PathBuf {
    config.output_dir.clone().unwrap_or_else(default_output_dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_render_round_trip() {
        let config = Config {
            models_dir: Some(PathBuf::from("/some/dir with spaces")),
            output_dir: Some(PathBuf::from("/somewhere/out")),
            default_model: Some("ggml-large-v3.bin".into()),
            diarize: DiarizeStrategy::Embedding,
            diarize_models: DiarizeModelChoice {
                segmentation: None,
                embedding: Some("campplus-zh-en.onnx".into()),
            },
            diarize_speakers: Some(3),
            language: Some("es".into()),
            hf_token: Some("hf_abc123".into()),
            split_mode: SplitMode::Silence,
            export_formats: vec![ExportFormat::LlmMd, ExportFormat::Srt],
        };
        assert_eq!(parse_str(&render(&config)), config);
    }

    #[test]
    fn legacy_diarize_bool_maps_to_the_tdrz_strategy() {
        // pre-strategy configs held a bool; true meant tinydiarize
        assert_eq!(
            parse_str("diarize = true").diarize,
            DiarizeStrategy::Tdrz
        );
        assert_eq!(parse_str("diarize = false").diarize, DiarizeStrategy::Off);
        // the new key wins over the legacy one
        assert_eq!(
            parse_str("diarize = true\ndiarize_strategy = \"auto\"").diarize,
            DiarizeStrategy::Auto
        );
        // saving never writes the legacy key back
        let config = Config {
            diarize: DiarizeStrategy::Auto,
            ..Config::default()
        };
        let rendered = render(&config);
        assert!(rendered.contains("diarize_strategy"), "{rendered}");
        assert!(!rendered.contains("diarize = "), "{rendered}");
    }

    #[test]
    fn export_formats_default_to_all_and_survive_garbage() {
        // absent key → all formats
        assert_eq!(parse_str("").export_formats, ExportFormat::ALL.to_vec());
        // unknown tokens dropped, known ones kept in canonical order
        assert_eq!(
            parse_str("export_formats = \"srt,bogus,json\"").export_formats,
            vec![ExportFormat::Json, ExportFormat::Srt]
        );
        // a hand-emptied key must not disable exporting entirely
        assert_eq!(
            parse_str("export_formats = \",,\"").export_formats,
            ExportFormat::ALL.to_vec()
        );
    }

    #[test]
    fn default_dirs_share_the_platform_base() {
        let models = default_models_dir();
        let output = default_output_dir();
        if cfg!(target_os = "macos") {
            assert!(models.ends_with("Documents/llm_transcribe/models"));
            assert!(output.ends_with("Documents/llm_transcribe/output"));
        } else {
            assert!(models.ends_with("llm_transcribe/models"));
            assert!(output.ends_with("llm_transcribe/output"));
        }
        assert_eq!(models.parent(), output.parent());
    }

    #[test]
    fn parse_tolerates_comments_quotes_and_garbage() {
        let config = parse_str(
            "# a comment\n\
             \n\
             models_dir = '/quoted/path'\n\
             default_model=unquoted.bin\n\
             not a key value line\n\
             unknown_key = whatever\n",
        );
        assert_eq!(config.models_dir, Some(PathBuf::from("/quoted/path")));
        assert_eq!(config.default_model, Some("unquoted.bin".into()));
    }

    #[test]
    fn parse_skips_empty_values() {
        let config = parse_str("models_dir = \"\"\ndefault_model =\n");
        assert_eq!(config, Config::default());
    }

    #[test]
    fn empty_config_renders_header_only() {
        assert_eq!(render(&Config::default()), "# transcribe-stt settings\n");
    }
}
