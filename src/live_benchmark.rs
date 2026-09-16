//! Local measured throughput for the model picker. A result only applies to
//! matching hardware and unchanged model files; missing/stale data is unknown.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use serde::Deserialize;

use crate::models::ModelFile;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LiveFit {
    Good { rtf: f64, batch: f64 },
    Borderline { rtf: f64, batch: f64 },
    TooSlow { rtf: f64, batch: f64 },
}

impl LiveFit {
    pub fn feed_samples(self) -> usize {
        let batch = match self {
            Self::Good { batch, .. }
            | Self::Borderline { batch, .. }
            | Self::TooSlow { batch, .. } => batch,
        };
        (batch * 16_000.0).round() as usize
    }

    fn from_measurement(rtf: f64, worst: f64, batch: f64) -> Option<Self> {
        if !rtf.is_finite()
            || !worst.is_finite()
            || !batch.is_finite()
            || rtf <= 0.0
            || worst < rtf
            || !(0.1..=4.0).contains(&batch)
        {
            return None;
        }
        Some(if worst <= 0.8 {
            Self::Good { rtf, batch }
        } else if worst <= 1.0 {
            Self::Borderline { rtf, batch }
        } else {
            Self::TooSlow { rtf, batch }
        })
    }

    pub fn label(self) -> String {
        let (status, rtf, batch) = match self {
            Self::Good { rtf, batch } => ("good fit", rtf, batch),
            Self::Borderline { rtf, batch } => ("borderline", rtf, batch),
            Self::TooSlow { rtf, batch } => ("too slow for this Mac", rtf, batch),
        };
        format!("Live: {status} · {rtf:.2}s/audio s · {batch}s feed")
    }
}

#[derive(Deserialize)]
struct Report {
    hardware: BTreeMap<String, String>,
    models: Vec<Measurement>,
}

#[derive(Deserialize)]
struct Measurement {
    path: PathBuf,
    #[serde(default)]
    fingerprint: Vec<FileStamp>,
    recommendation: Option<Recommendation>,
}

#[derive(Deserialize)]
struct Recommendation {
    median_sustained_rtf: f64,
    worst_sustained_rtf: f64,
    batch_seconds: f64,
}

#[derive(Deserialize)]
struct FileStamp {
    name: PathBuf,
    size: u64,
    modified_ns: String,
}

impl Measurement {
    fn current(&self) -> bool {
        !self.fingerprint.is_empty()
            && self.fingerprint.iter().all(|stamp| {
                // Reports are data, never permission to inspect arbitrary paths.
                if stamp.name.is_absolute()
                    || stamp
                        .name
                        .components()
                        .any(|c| !matches!(c, std::path::Component::Normal(_)))
                {
                    return false;
                }
                let Ok(meta) = std::fs::metadata(self.path.join(&stamp.name)) else {
                    return false;
                };
                let modified = meta
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok());
                meta.len() == stamp.size
                    && modified.is_some_and(|t| t.as_nanos().to_string() == stamp.modified_ns)
            })
    }
}

const HARDWARE_KEYS: [&str; 3] = ["hw.memsize", "hw.model", "machdep.cpu.brand_string"];

fn hardware_matches(saved: &BTreeMap<String, String>, current: &BTreeMap<String, String>) -> bool {
    HARDWARE_KEYS.iter().all(|key| {
        current
            .get(*key)
            .is_some_and(|value| !value.is_empty() && saved.get(*key) == Some(value))
    })
}

pub fn load(models: &[ModelFile]) -> BTreeMap<PathBuf, LiveFit> {
    load_paths(
        &models
            .iter()
            .map(|model| model.path.as_path())
            .collect::<Vec<_>>(),
    )
}

/// Revalidate at session start, including sessions started without the picker.
pub fn for_model(path: &Path) -> Option<LiveFit> {
    load_paths(&[path]).remove(path)
}

fn load_paths(models: &[&Path]) -> BTreeMap<PathBuf, LiveFit> {
    let mut current = BTreeMap::new();
    for key in HARDWARE_KEYS {
        if let Ok(output) = std::process::Command::new("sysctl")
            .args(["-n", key])
            .output()
        {
            if output.status.success() {
                current.insert(
                    key.into(),
                    String::from_utf8_lossy(&output.stdout).trim().into(),
                );
            }
        }
    }
    crate::config::live_benchmark_path()
        .map(|path| load_file(&path, models, &current))
        .unwrap_or_default()
}

fn load_file(
    path: &Path,
    models: &[&Path],
    hardware: &BTreeMap<String, String>,
) -> BTreeMap<PathBuf, LiveFit> {
    let report = std::fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Report>(&bytes).ok());
    let Some(report) = report else {
        return BTreeMap::new();
    };
    if !hardware_matches(&report.hardware, hardware) {
        return BTreeMap::new();
    }
    report
        .models
        .iter()
        .filter_map(|result| {
            let model = models.iter().find(|model| {
                **model == result.path || model.canonicalize().ok().as_ref() == Some(&result.path)
            })?;
            if !result.current() {
                return None;
            }
            let score = result.recommendation.as_ref()?;
            let fit = LiveFit::from_measurement(
                score.median_sustained_rtf,
                score.worst_sustained_rtf,
                score.batch_seconds,
            )?;
            Some((model.to_path_buf(), fit))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn measured_fit_distinguishes_borderline_and_rejects_invalid_results() {
        assert!(matches!(
            LiveFit::from_measurement(0.06, 0.07, 4.0),
            Some(LiveFit::Good { .. })
        ));
        assert!(matches!(
            LiveFit::from_measurement(0.95, 0.96, 1.0),
            Some(LiveFit::Borderline { .. })
        ));
        assert!(matches!(
            LiveFit::from_measurement(6.16, 6.25, 0.5),
            Some(LiveFit::TooSlow { .. })
        ));
        assert!(matches!(
            LiveFit::from_measurement(0.9, 1.1, 1.0),
            Some(LiveFit::TooSlow { .. })
        ));
        assert!(LiveFit::from_measurement(f64::NAN, 1.0, 1.0).is_none());
        assert!(LiveFit::from_measurement(0.0, 0.0, 1.0).is_none());
    }
    #[test]
    fn benchmarks_do_not_transfer_to_unknown_or_different_hardware() {
        let saved = BTreeMap::from([
            ("hw.memsize".into(), "25769803776".into()),
            ("hw.model".into(), "Mac16,12".into()),
            ("machdep.cpu.brand_string".into(), "Apple M4".into()),
        ]);
        assert!(hardware_matches(&saved, &saved));
        assert!(!hardware_matches(&saved, &BTreeMap::new()));
        let mut other = saved.clone();
        other.insert("machdep.cpu.brand_string".into(), "Apple M3".into());
        assert!(!hardware_matches(&saved, &other));
    }
    #[test]
    fn replacing_model_files_invalidates_measurements() {
        let path =
            std::env::temp_dir().join(format!("live-benchmark-stamp-{}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        let file = path.join("config.json");
        std::fs::write(&file, "{}").unwrap();
        let meta = std::fs::metadata(&file).unwrap();
        let result = Measurement {
            path: path.clone(),
            recommendation: None,
            fingerprint: vec![FileStamp {
                name: "config.json".into(),
                size: meta.len(),
                modified_ns: meta
                    .modified()
                    .unwrap()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
                    .to_string(),
            }],
        };
        assert!(result.current());
        let hardware: BTreeMap<String, String> = HARDWARE_KEYS
            .iter()
            .map(|key| (key.to_string(), "test machine".into()))
            .collect();
        let report_path = path.join("benchmark.json");
        let report = serde_json::json!({
            "hardware": hardware,
            "models": [{
                "path": path,
                "fingerprint": [{"name": "config.json", "size": meta.len(),
                    "modified_ns": result.fingerprint[0].modified_ns}],
                "recommendation": {"median_sustained_rtf": 0.95,
                    "worst_sustained_rtf": 0.96, "batch_seconds": 1.0}
            }]
        });
        std::fs::write(&report_path, report.to_string()).unwrap();
        let fits = load_file(&report_path, &[path.as_path()], &hardware);
        assert_eq!(fits[&path].feed_samples(), 16000);
        assert!(load_file(&report_path, &[path.as_path()], &BTreeMap::new()).is_empty());
        std::fs::write(&file, "changed").unwrap();
        assert!(!result.current());
        assert!(load_file(&report_path, &[path.as_path()], &hardware).is_empty());
        std::fs::remove_dir_all(path).unwrap();
    }
}
