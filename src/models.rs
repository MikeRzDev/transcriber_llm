use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Integrity manifest written into each downloaded directory model: the
/// exact file list (with sizes) the download fetched. A later scan checks
/// it to catch missing or truncated files, and a re-download fetches only
/// the gaps. Dot-named so it can never collide with a repo file (dotfiles
/// are excluded from repo downloads).
pub const DIR_MANIFEST: &str = ".manifest.json";

#[derive(Debug, Serialize, Deserialize)]
pub struct DirManifest {
    pub repo: String,
    /// Repo-relative folder this model came from ("" = the repo root;
    /// non-empty for a variant subfolder download)
    #[serde(default)]
    pub subdir: String,
    pub files: Vec<ManifestEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ManifestEntry {
    pub name: String,
    pub size: u64,
}

impl DirManifest {
    pub fn load(dir: &Path) -> Option<Self> {
        let text = std::fs::read_to_string(dir.join(DIR_MANIFEST)).ok()?;
        serde_json::from_str(&text).ok()
    }

    pub fn write(&self, dir: &Path) -> anyhow::Result<()> {
        std::fs::write(dir.join(DIR_MANIFEST), serde_json::to_string_pretty(self)?)?;
        Ok(())
    }
}

/// Files the manifest promises that are missing or truncated on disk.
/// Empty means intact — or that the directory carries no manifest (a
/// hand-placed model makes no integrity claim). A size the manifest
/// doesn't know (0) only requires the file to exist.
pub fn manifest_gaps(dir: &Path) -> Vec<String> {
    let Some(manifest) = DirManifest::load(dir) else {
        return Vec::new();
    };
    manifest
        .files
        .iter()
        .filter(|f| {
            let path = dir.join(&f.name);
            if f.size > 0 {
                std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0) != f.size
            } else {
                !path.exists()
            }
        })
        .map(|f| f.name.clone())
        .collect()
}

#[derive(Clone, Debug)]
pub struct ModelFile {
    pub path: PathBuf,
    pub name: String,
    /// For directory models, the summed size of every file inside.
    pub size_bytes: u64,
    /// Directory-shaped model (MLX safetensors folder) rather than a
    /// single GGML/GGUF file.
    pub is_dir: bool,
}

impl ModelFile {
    pub fn size_human(&self) -> String {
        crate::format::human_size(self.size_bytes)
    }

    pub fn display_name(&self) -> String {
        display_name(&self.name)
    }
}

/// The model name a person recognizes, derived from whisper.cpp's file
/// naming: `ggml-large-v3-turbo-q5_0.bin` is "Whisper Large v3 Turbo
/// (q5_0)". Display only — the file name stays the model's identity in
/// config, matching, and job routing. Names that don't follow the
/// `ggml-*.bin` convention (gguf drops, MLX dirs, hand-renamed files)
/// pass through untouched rather than guessing.
pub fn display_name(name: &str) -> String {
    let stripped = name
        .strip_prefix("ggml-")
        .and_then(|r| r.strip_suffix(".bin").or_else(|| r.strip_suffix(".gguf")));
    let Some(stripped) = stripped else {
        return name.to_string();
    };

    let english = stripped.contains(".en");
    let tdrz = stripped.contains("-tdrz");
    let mut core = stripped.replace(".en", "").replace("-tdrz", "");

    // Trailing quantization tag: q + digit + [a-z0-9_]* (q5_0, q8_0, q4_k)
    let mut quant = None;
    if let Some((head, tail)) = core.rsplit_once('-') {
        let mut chars = tail.chars();
        if chars.next() == Some('q')
            && chars.next().is_some_and(|c| c.is_ascii_digit())
            && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        {
            quant = Some(tail.to_string());
            core = head.to_string();
        }
    }
    if core.is_empty() {
        return name.to_string();
    }

    // "large-v3-turbo" → "Large v3 Turbo": capitalize words, keep
    // version tags (v1, v2, v3…) as-is
    let core = core
        .split('-')
        .map(|word| {
            let is_version =
                word.len() >= 2 && word.starts_with('v') && word[1..].chars().all(|c| c.is_ascii_digit());
            if is_version {
                word.to_string()
            } else {
                let mut chars = word.chars();
                match chars.next() {
                    Some(first) => first.to_uppercase().chain(chars).collect(),
                    None => String::new(),
                }
            }
        })
        .collect::<Vec<String>>()
        .join(" ");

    let mut extras: Vec<&str> = Vec::new();
    if english {
        extras.push("English");
    }
    if tdrz {
        extras.push("tinydiarize");
    }
    if let Some(quant) = &quant {
        extras.push(quant);
    }
    if extras.is_empty() {
        format!("Whisper {core}")
    } else {
        format!("Whisper {core} ({})", extras.join(", "))
    }
}

/// tinydiarize builds carry speaker-turn tokens and are named `*-tdrz*`.
pub fn is_tdrz(name: &str) -> bool {
    name.contains("tdrz")
}

/// Why a models folder can't be used right now, if it can't. The classic
/// case: a configured folder on an external volume that isn't mounted —
/// creating it would mean writing into root-owned `/Volumes`, so
/// downloads die with a bare "permission denied" unless this explains it.
pub fn dir_unavailable(dir: &Path) -> Option<String> {
    if dir.exists() {
        return None;
    }
    use std::path::Component;
    let comps: Vec<Component> = dir.components().collect();
    if let [Component::RootDir, Component::Normal(volumes), Component::Normal(name), ..] =
        comps[..]
    {
        if volumes == "Volumes" {
            let volume_root = Path::new("/Volumes").join(name);
            if !volume_root.exists() {
                return Some(format!(
                    "the drive '{}' is not mounted — connect it, or change the models \
                     folder in settings (s)",
                    name.to_string_lossy()
                ));
            }
        }
    }
    None
}

/// English-only whisper builds are tagged `.en` in the file name.
pub fn is_english_only(name: &str) -> bool {
    name.contains(".en")
}

/// The model a job should use when none is explicitly selected: the
/// configured default first, then large-v3, then whatever exists.
pub fn pick_default<'a>(
    models: &'a [ModelFile],
    configured: Option<&str>,
) -> Option<&'a ModelFile> {
    configured
        .and_then(|name| models.iter().find(|m| m.name == name))
        .or_else(|| models.iter().find(|m| m.name.contains("large-v3")))
        .or_else(|| models.first())
}

/// whisper.cpp ships GGML-format .bin models; .gguf covers other
/// ggml-family voice models dropped into the same directory.
pub fn is_model_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("bin") | Some("gguf")
    )
}

/// Weights file of an MLX model dir. Safetensors only: it is the format
/// every mlx-audio loader globs for — legacy npz conversions (some old
/// Whisper-MLX repos) would download fine but cannot be run, so they are
/// not offered as models.
pub fn is_mlx_weights_file(name: &str) -> bool {
    name.ends_with(".safetensors")
}

/// MLX models are directories holding a `config.json` plus weights
/// (e.g. mlx-community/parakeet-tdt-0.6b-v3). Run by the MLX engine via
/// mlx-audio; a directory job routes there.
pub fn is_mlx_model_dir(path: &Path) -> bool {
    path.is_dir()
        && path.join("config.json").is_file()
        && std::fs::read_dir(path)
            .into_iter()
            .flatten()
            .flatten()
            .any(|e| is_mlx_weights_file(&e.file_name().to_string_lossy()))
}

/// Total size of every file under `dir`, recursively.
pub fn dir_size(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| {
            let path = entry.path();
            if path.is_dir() {
                dir_size(&path)
            } else {
                entry.metadata().map(|m| m.len()).unwrap_or(0)
            }
        })
        .sum()
}

/// Outcome of a bulk model move — a domain result, not an error: any
/// mix of moved/skipped/failed is a legitimate answer to report.
pub struct MoveReport {
    pub moved: usize,
    pub skipped: usize,
    pub failed: usize,
}

/// Move every supported model from one folder to another, skipping ones
/// that already exist at the destination.
pub fn move_models(from: &Path, to: &Path) -> MoveReport {
    let mut report = MoveReport {
        moved: 0,
        skipped: 0,
        failed: 0,
    };
    for model in scan_models(from) {
        let dest = to.join(&model.name);
        if dest.exists() {
            report.skipped += 1;
            continue;
        }
        let result = std::fs::rename(&model.path, &dest).or_else(|_| {
            // cross-volume fallback: copy, then delete the original
            if model.is_dir {
                copy_dir(&model.path, &dest)
                    .map(|_| {
                        let _ = std::fs::remove_dir_all(&model.path);
                    })
                    .inspect_err(|_| {
                        let _ = std::fs::remove_dir_all(&dest);
                    })
            } else {
                std::fs::copy(&model.path, &dest)
                    .map(|_| {
                        let _ = std::fs::remove_file(&model.path);
                    })
                    .inspect_err(|_| {
                        let _ = std::fs::remove_file(&dest);
                    })
            }
        });
        match result {
            Ok(()) => report.moved += 1,
            Err(_) => report.failed += 1,
        }
    }
    report
}

fn copy_dir(from: &Path, to: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let src = entry.path();
        let dst = to.join(entry.file_name());
        if src.is_dir() {
            copy_dir(&src, &dst)?;
        } else {
            std::fs::copy(&src, &dst)?;
        }
    }
    Ok(())
}

pub fn scan_models(dir: &Path) -> Vec<ModelFile> {
    let mut models: Vec<ModelFile> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            let name = path.file_name()?.to_string_lossy().into_owned();
            // In-flight downloads stay invisible until renamed into place
            if name.ends_with(".part") {
                return None;
            }
            let (size_bytes, is_dir) = if path.is_file() && is_model_file(&path) {
                (entry.metadata().ok()?.len(), false)
            } else if is_mlx_model_dir(&path) && manifest_gaps(&path).is_empty() {
                // A dir failing its integrity manifest is not offered as a
                // model; re-downloading it fetches just the missing files.
                (dir_size(&path), true)
            } else {
                return None;
            };
            // Skip zero-byte stubs left by interrupted writes
            if size_bytes == 0 {
                return None;
            }
            Some(ModelFile {
                name,
                size_bytes,
                path,
                is_dir,
            })
        })
        .collect();
    models.sort_by(|a, b| a.name.cmp(&b.name));
    models
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static UNIQUE: AtomicUsize = AtomicUsize::new(0);

    fn tempdir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "transcribe-stt-test-{}-{}",
            std::process::id(),
            UNIQUE.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Lay down a minimal MLX model directory (config.json + weights).
    fn write_mlx_dir(dir: &Path, name: &str) -> PathBuf {
        let model = dir.join(name);
        std::fs::create_dir_all(&model).unwrap();
        std::fs::write(model.join("config.json"), b"{}").unwrap();
        std::fs::write(model.join("model.safetensors"), vec![0u8; 1000]).unwrap();
        std::fs::write(model.join("vocab.txt"), vec![0u8; 24]).unwrap();
        model
    }

    #[test]
    fn scan_filters_sorts_and_skips_stubs() {
        let dir = tempdir();
        std::fs::write(dir.join("b.gguf"), b"data").unwrap();
        std::fs::write(dir.join("a.bin"), b"data").unwrap();
        std::fs::write(dir.join("empty.bin"), b"").unwrap(); // zero-byte stub
        std::fs::write(dir.join("readme.txt"), b"not a model").unwrap();
        std::fs::create_dir(dir.join("subdir.bin")).unwrap(); // dir, not file

        let names: Vec<String> = scan_models(&dir).into_iter().map(|m| m.name).collect();
        assert_eq!(names, vec!["a.bin", "b.gguf"]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn scan_includes_mlx_dirs_with_summed_size() {
        let dir = tempdir();
        std::fs::write(dir.join("a.bin"), b"data").unwrap();
        write_mlx_dir(&dir, "parakeet-tdt-0.6b-v3");
        // an in-flight dir download must stay invisible
        write_mlx_dir(&dir, "canary-1b.part");
        // a plain folder without model files is not a model
        std::fs::create_dir(dir.join("notes")).unwrap();

        let models = scan_models(&dir);
        let names: Vec<&str> = models.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, vec!["a.bin", "parakeet-tdt-0.6b-v3"]);
        let mlx = &models[1];
        assert!(mlx.is_dir);
        assert_eq!(mlx.size_bytes, 1000 + 2 + 24);
        assert!(!models[0].is_dir);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn mlx_dir_detection_needs_config_and_safetensors() {
        let dir = tempdir();
        let full = write_mlx_dir(&dir, "full");
        assert!(is_mlx_model_dir(&full));

        let no_weights = dir.join("no-weights");
        std::fs::create_dir(&no_weights).unwrap();
        std::fs::write(no_weights.join("config.json"), b"{}").unwrap();
        assert!(!is_mlx_model_dir(&no_weights));

        let no_config = dir.join("no-config");
        std::fs::create_dir(&no_config).unwrap();
        std::fs::write(no_config.join("model.safetensors"), b"w").unwrap();
        assert!(!is_mlx_model_dir(&no_config));

        // Legacy npz conversions can't be run by mlx-audio → not a model
        let npz = dir.join("whisper-tiny-mlx");
        std::fs::create_dir(&npz).unwrap();
        std::fs::write(npz.join("config.json"), b"{}").unwrap();
        std::fs::write(npz.join("weights.npz"), b"w").unwrap();
        assert!(!is_mlx_model_dir(&npz));

        assert!(!is_mlx_model_dir(&dir.join("missing")));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn manifest_gaps_flag_missing_and_truncated_files() {
        let dir = tempdir();
        let model = write_mlx_dir(&dir, "qwen3-asr");
        // No manifest → no integrity claim → intact
        assert!(manifest_gaps(&model).is_empty());

        let manifest = DirManifest {
            repo: "mlx-community/qwen3-asr".into(),
            subdir: String::new(),
            files: vec![
                ManifestEntry {
                    name: "config.json".into(),
                    size: 2,
                },
                ManifestEntry {
                    name: "model.safetensors".into(),
                    size: 1000,
                },
                ManifestEntry {
                    name: "vocab.txt".into(),
                    size: 24,
                },
            ],
        };
        manifest.write(&model).unwrap();
        assert!(manifest_gaps(&model).is_empty());
        assert_eq!(scan_models(&dir).len(), 1);

        // A deleted file and a truncated file both break integrity, and a
        // broken model disappears from the scan (so the hub re-downloads).
        std::fs::remove_file(model.join("vocab.txt")).unwrap();
        std::fs::write(model.join("model.safetensors"), b"short").unwrap();
        assert_eq!(
            manifest_gaps(&model),
            vec!["model.safetensors".to_string(), "vocab.txt".to_string()]
        );
        assert!(scan_models(&dir).is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn move_models_carries_dir_models_too() {
        let dir = tempdir();
        let from = dir.join("from");
        let to = dir.join("to");
        std::fs::create_dir_all(&from).unwrap();
        std::fs::create_dir_all(&to).unwrap();
        std::fs::write(from.join("a.bin"), b"data").unwrap();
        write_mlx_dir(&from, "parakeet");

        let report = move_models(&from, &to);
        assert_eq!((report.moved, report.skipped, report.failed), (2, 0, 0));
        assert!(to.join("a.bin").is_file());
        assert!(to.join("parakeet/model.safetensors").is_file());
        assert!(!from.join("parakeet").exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn scan_missing_dir_is_empty() {
        assert!(scan_models(Path::new("/nonexistent/nowhere")).is_empty());
    }

    #[test]
    fn dir_unavailable_spots_unmounted_volumes_only() {
        // a folder on a volume that isn't mounted names the drive
        let reason =
            dir_unavailable(Path::new("/Volumes/NoSuchDrive-xyz/models")).expect("unavailable");
        assert!(reason.contains("NoSuchDrive-xyz"), "{reason}");
        assert!(reason.contains("not mounted"), "{reason}");

        // an existing folder is fine
        assert_eq!(dir_unavailable(&std::env::temp_dir()), None);
        // a missing folder NOT under /Volumes is creatable — no complaint
        assert_eq!(
            dir_unavailable(&std::env::temp_dir().join("does-not-exist-yet")),
            None
        );
    }

    #[test]
    fn model_extension_check() {
        assert!(is_model_file(Path::new("x.bin")));
        assert!(is_model_file(Path::new("x.gguf")));
        assert!(!is_model_file(Path::new("x.txt")));
        assert!(!is_model_file(Path::new("noext")));
    }

    #[test]
    fn display_name_translates_whisper_cpp_files() {
        assert_eq!(display_name("ggml-large-v3.bin"), "Whisper Large v3");
        assert_eq!(
            display_name("ggml-large-v3-turbo-q5_0.bin"),
            "Whisper Large v3 Turbo (q5_0)"
        );
        assert_eq!(display_name("ggml-base.en.bin"), "Whisper Base (English)");
        assert_eq!(
            display_name("ggml-small.en-tdrz.bin"),
            "Whisper Small (English, tinydiarize)"
        );
        assert_eq!(display_name("ggml-tiny-q5_1.bin"), "Whisper Tiny (q5_1)");
        assert_eq!(
            display_name("ggml-distil-large-v3.bin"),
            "Whisper Distil Large v3"
        );
    }

    #[test]
    fn display_name_leaves_unrecognized_names_alone() {
        // no ggml- prefix, gguf drops, MLX dirs: shown as-is
        assert_eq!(display_name("large-v3.bin"), "large-v3.bin");
        assert_eq!(display_name("voice-model.gguf"), "voice-model.gguf");
        assert_eq!(display_name("parakeet-tdt-0.6b-v3"), "parakeet-tdt-0.6b-v3");
        assert_eq!(display_name("ggml-.bin"), "ggml-.bin");
    }

    #[test]
    fn size_human_units() {
        let m = |size_bytes| ModelFile {
            path: PathBuf::new(),
            name: String::new(),
            size_bytes,
            is_dir: false,
        };
        assert_eq!(m(3_100_000_000).size_human(), "3.1 GB");
        assert_eq!(m(488_000_000).size_human(), "488 MB");
        assert_eq!(m(12_000).size_human(), "12 KB");
    }
}
