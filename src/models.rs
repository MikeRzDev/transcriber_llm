use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct ModelFile {
    pub path: PathBuf,
    pub name: String,
    pub size_bytes: u64,
}

impl ModelFile {
    pub fn size_human(&self) -> String {
        crate::format::human_size(self.size_bytes)
    }
}

/// tinydiarize builds carry speaker-turn tokens and are named `*-tdrz*`.
pub fn is_tdrz(name: &str) -> bool {
    name.contains("tdrz")
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

/// Outcome of a bulk model move — a domain result, not an error: any
/// mix of moved/skipped/failed is a legitimate answer to report.
pub struct MoveReport {
    pub moved: usize,
    pub skipped: usize,
    pub failed: usize,
}

/// Move every supported model from one folder to another, skipping files
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
            std::fs::copy(&model.path, &dest)
                .map(|_| {
                    let _ = std::fs::remove_file(&model.path);
                })
                .inspect_err(|_| {
                    let _ = std::fs::remove_file(&dest);
                })
        });
        match result {
            Ok(()) => report.moved += 1,
            Err(_) => report.failed += 1,
        }
    }
    report
}

pub fn scan_models(dir: &Path) -> Vec<ModelFile> {
    let mut models: Vec<ModelFile> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            if !path.is_file() || !is_model_file(&path) {
                return None;
            }
            let meta = entry.metadata().ok()?;
            // Skip in-flight downloads (curl partials are fine, but zero-byte stubs are not)
            if meta.len() == 0 {
                return None;
            }
            Some(ModelFile {
                name: path.file_name()?.to_string_lossy().into_owned(),
                size_bytes: meta.len(),
                path,
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
    fn scan_missing_dir_is_empty() {
        assert!(scan_models(Path::new("/nonexistent/nowhere")).is_empty());
    }

    #[test]
    fn model_extension_check() {
        assert!(is_model_file(Path::new("x.bin")));
        assert!(is_model_file(Path::new("x.gguf")));
        assert!(!is_model_file(Path::new("x.txt")));
        assert!(!is_model_file(Path::new("noext")));
    }

    #[test]
    fn size_human_units() {
        let m = |size_bytes| ModelFile {
            path: PathBuf::new(),
            name: String::new(),
            size_bytes,
        };
        assert_eq!(m(3_100_000_000).size_human(), "3.1 GB");
        assert_eq!(m(488_000_000).size_human(), "488 MB");
        assert_eq!(m(12_000).size_human(), "12 KB");
    }
}
