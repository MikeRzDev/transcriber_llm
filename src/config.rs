use std::path::PathBuf;

use crate::split::SplitMode;

/// Persisted settings, stored as simple `key = value` lines in
/// ~/.config/transcribe-stt/config.toml
#[derive(Default, Clone, Debug, PartialEq)]
pub struct Config {
    pub models_dir: Option<PathBuf>,
    pub output_dir: Option<PathBuf>,
    pub default_model: Option<String>,
    /// Label speakers via tinydiarize before transcription starts
    pub diarize: bool,
    /// ISO 639-1 code passed to whisper; None = auto-detect
    pub language: Option<String>,
    /// Chunking strategy for long audio
    pub split_mode: SplitMode,
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

fn parse_str(contents: &str) -> Config {
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
            "diarize" => config.diarize = value == "true",
            "split_mode" => {
                if let Some(mode) = SplitMode::parse(value) {
                    config.split_mode = mode;
                }
            }
            "language" if value != "auto" => config.language = Some(value.to_string()),
            _ => {}
        }
    }
    config
}

fn render(config: &Config) -> String {
    let mut out = String::from("# transcribe-stt settings\n");
    if let Some(dir) = &config.models_dir {
        out.push_str(&format!("models_dir = \"{}\"\n", dir.display()));
    }
    if let Some(dir) = &config.output_dir {
        out.push_str(&format!("output_dir = \"{}\"\n", dir.display()));
    }
    if let Some(model) = &config.default_model {
        out.push_str(&format!("default_model = \"{model}\"\n"));
    }
    if config.diarize {
        out.push_str("diarize = true\n");
    }
    if config.split_mode != SplitMode::Auto {
        out.push_str(&format!("split_mode = \"{}\"\n", config.split_mode.as_str()));
    }
    if let Some(lang) = &config.language {
        out.push_str(&format!("language = \"{lang}\"\n"));
    }
    out
}

pub fn load() -> Config {
    config_path()
        .and_then(|path| std::fs::read_to_string(path).ok())
        .map(|contents| parse_str(&contents))
        .unwrap_or_default()
}

pub fn save(config: &Config) -> std::io::Result<()> {
    let Some(path) = config_path() else {
        return Ok(());
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, render(config))
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
            diarize: true,
            language: Some("es".into()),
            split_mode: SplitMode::Silence,
        };
        assert_eq!(parse_str(&render(&config)), config);
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
