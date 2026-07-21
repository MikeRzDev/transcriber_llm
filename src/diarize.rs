//! Model-agnostic speaker diarization. The user picks a
//! [`DiarizeStrategy`] (persisted in config); per job it resolves —
//! against the model in use — into the [`DiarizeMethod`] that actually
//! executes: tinydiarize runs inline in the whisper engine, the
//! embedding pipeline (`sherpa`) runs as an engine-agnostic post-pass in
//! the worker over any engine's output. Each strategy reports its
//! [`Requirement`]s so the UI can tell the user exactly what still has
//! to be downloaded (and how big it is) before the strategy can run.

pub mod pyannote;
pub mod sherpa;

use std::collections::BTreeMap;
use std::path::Path;

use crate::models::{self, ModelFile};
use crate::transcribe::Segment;

/// The tdrz whisper build the tinydiarize strategy transcribes with.
pub const TDRZ_FILE: &str = "ggml-small.en-tdrz.bin";
pub const TDRZ_REPO: &str = "akashmjn/tinydiarize-whisper.cpp";
pub const TDRZ_SIZE: &str = "488 MB";

/// The user-selectable diarization strategies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DiarizeStrategy {
    /// No speaker labels
    #[default]
    Off,
    /// Use whatever `recommended_for` picks for the model in use
    Auto,
    /// whisper.cpp tinydiarize: speaker-turn tokens emitted while
    /// transcribing with the tdrz model (2 speakers, English only)
    Tdrz,
    /// pyannote segmentation + speaker-embedding clustering via
    /// sherpa-onnx: any model, any engine, any number of speakers
    Embedding,
    /// pyannote.audio speaker-diarization-community-1 via PyTorch: the
    /// open-accuracy SOTA — heavier (~2 GB) and gated (HF token)
    Pyannote,
}

impl DiarizeStrategy {
    /// Cycle order for the settings row and the `d` shortcut.
    pub const ALL: [Self; 5] = [
        Self::Off,
        Self::Auto,
        Self::Tdrz,
        Self::Embedding,
        Self::Pyannote,
    ];

    /// Config-file token.
    pub fn key(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Auto => "auto",
            Self::Tdrz => "tdrz",
            Self::Embedding => "embedding",
            Self::Pyannote => "pyannote",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim().to_ascii_lowercase();
        Self::ALL
            .into_iter()
            .find(|v| v.key() == s)
            .or(match s.as_str() {
                "tinydiarize" => Some(Self::Tdrz),
                "embeddings" | "sherpa" => Some(Self::Embedding),
                "community-1" | "pyannote-community-1" => Some(Self::Pyannote),
                _ => None,
            })
    }

    /// Human description for the settings row and status line.
    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "OFF",
            Self::Auto => "Auto (recommended per model)",
            Self::Tdrz => "TinyDiarize (2 speakers, English, uses the tdrz model)",
            Self::Embedding => "Speaker embeddings (any model, multi-speaker)",
            Self::Pyannote => "Pyannote community-1 (max quality, PyTorch + HF token)",
        }
    }

    pub fn next(self) -> Self {
        let i = Self::ALL.iter().position(|v| *v == self).unwrap_or(0);
        Self::ALL[(i + 1) % Self::ALL.len()]
    }

    /// The strategy recommended for a model: a tdrz build carries its own
    /// turn tokens (nothing extra to download), everything else is served
    /// engine-agnostically by the embedding pipeline.
    pub fn recommended_for(model_name: Option<&str>) -> Self {
        match model_name {
            Some(name) if models::is_tdrz(name) => Self::Tdrz,
            _ => Self::Embedding,
        }
    }

    /// What actually runs for a job using this strategy on this model.
    pub fn resolve(self, model_name: Option<&str>) -> DiarizeMethod {
        match self {
            Self::Off => DiarizeMethod::None,
            Self::Auto => match Self::recommended_for(model_name) {
                Self::Tdrz => DiarizeMethod::Tdrz,
                _ => DiarizeMethod::Embedding,
            },
            Self::Tdrz => DiarizeMethod::Tdrz,
            Self::Embedding => DiarizeMethod::Embedding,
            Self::Pyannote => DiarizeMethod::Pyannote,
        }
    }
}

/// The executable diarization method for one job, after `Auto` has been
/// resolved against the model in use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DiarizeMethod {
    #[default]
    None,
    Tdrz,
    Embedding,
    Pyannote,
}

impl DiarizeMethod {
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Tdrz => "tinydiarize",
            Self::Embedding => "speaker embeddings",
            Self::Pyannote => "pyannote community-1",
        }
    }

    /// Export-header description of how speaker labels were produced.
    pub fn export_note(self) -> Option<&'static str> {
        match self {
            Self::None => None,
            Self::Tdrz => Some("speaker-turn detection (tinydiarize), two-speaker labeling"),
            Self::Embedding => {
                Some("pyannote segmentation + speaker-embedding clustering (sherpa-onnx)")
            }
            Self::Pyannote => Some("pyannote.audio speaker-diarization-community-1"),
        }
    }
}

/// Which catalog models the embedding pipeline runs with, by their
/// on-disk names (`None` = the role's default). Persisted in config and
/// set from the hub's diarization category.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DiarizeModelChoice {
    pub segmentation: Option<String>,
    pub embedding: Option<String>,
}

/// One thing a diarization method needs on disk before it can run.
/// Satisfied pieces are already there; the rest is what must be
/// downloaded — each names its size so the user decides informed.
pub struct Requirement {
    pub name: String,
    pub size: &'static str,
    pub satisfied: bool,
}

/// Everything `method` needs, checked against the model library and the
/// models folder. Note: for `Embedding` this may probe the Python
/// runtime (a ~50 ms subprocess on the first call) — call it on user
/// actions, not per rendered frame.
pub fn requirements(
    method: DiarizeMethod,
    models: &[ModelFile],
    models_dir: &Path,
    choice: &DiarizeModelChoice,
) -> Vec<Requirement> {
    match method {
        DiarizeMethod::None => Vec::new(),
        DiarizeMethod::Tdrz => vec![Requirement {
            name: format!("tdrz model {TDRZ_FILE}"),
            size: TDRZ_SIZE,
            satisfied: models.iter().any(|m| models::is_tdrz(&m.name)),
        }],
        DiarizeMethod::Embedding => sherpa::requirements(models_dir, choice),
        DiarizeMethod::Pyannote => pyannote::requirements(),
    }
}

/// The unsatisfied requirements as one human line: `None` when
/// everything is in place, else what will be downloaded and how — the
/// embedding pieces self-download on first run, the tdrz model is a
/// Model-management download.
pub fn download_note(
    method: DiarizeMethod,
    models: &[ModelFile],
    models_dir: &Path,
    choice: &DiarizeModelChoice,
) -> Option<String> {
    let missing: Vec<String> = requirements(method, models, models_dir, choice)
        .iter()
        .filter(|r| !r.satisfied)
        .map(|r| format!("{} ({})", r.name, r.size))
        .collect();
    if missing.is_empty() {
        return None;
    }
    let missing = missing.join(" + ");
    Some(match method {
        DiarizeMethod::Tdrz => {
            format!("needs download: {missing} — Model management (s)")
        }
        DiarizeMethod::Pyannote => format!("first run sets up: {missing}"),
        _ => format!("first run downloads: {missing}"),
    })
}

/// One diarizer-attributed span of speech.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeakerTurn {
    pub start_ms: i64,
    pub end_ms: i64,
    pub speaker: u8,
}

/// Every diarizer runner writes the same JSON:
/// `[{"start": seconds, "end": seconds, "speaker": n}]`.
pub(crate) fn parse_turns(text: &str) -> anyhow::Result<Vec<SpeakerTurn>> {
    use anyhow::Context;
    #[derive(serde::Deserialize)]
    struct RawTurn {
        start: f64,
        end: f64,
        speaker: u32,
    }
    let raw: Vec<RawTurn> = serde_json::from_str(text).context("parsing diarizer JSON output")?;
    Ok(raw
        .into_iter()
        .map(|t| SpeakerTurn {
            start_ms: (t.start * 1000.0).round() as i64,
            end_ms: (t.end * 1000.0).round() as i64,
            speaker: t.speaker.min(u8::MAX as u32) as u8,
        })
        .collect())
}

/// Label each transcript segment with the speaker whose turns overlap it
/// most. A segment no turn touches keeps `None` (silence, music, or the
/// diarizer disagreeing about speech) rather than getting a guessed
/// label; overlap ties go to the lower speaker index for determinism.
pub fn assign_speakers(segments: &[Segment], turns: &[SpeakerTurn]) -> Vec<Segment> {
    segments
        .iter()
        .map(|seg| {
            let mut overlap: BTreeMap<u8, i64> = BTreeMap::new();
            for turn in turns {
                let ms = seg.end_ms.min(turn.end_ms) - seg.start_ms.max(turn.start_ms);
                if ms > 0 {
                    *overlap.entry(turn.speaker).or_default() += ms;
                }
            }
            let speaker = overlap
                .into_iter()
                .max_by_key(|&(s, ms)| (ms, std::cmp::Reverse(s)))
                .map(|(s, _)| s);
            Segment {
                speaker,
                ..seg.clone()
            }
        })
        .collect()
}

/// Assign alternating speaker indices from per-segment "next segment is a
/// new speaker" flags (tinydiarize). Assumes a two-person conversation:
/// every detected turn flips between speaker 0 and speaker 1.
pub fn alternate_speakers(turn_after: &[bool]) -> Vec<u8> {
    let mut current = 0u8;
    turn_after
        .iter()
        .map(|&turn| {
            let speaker = current;
            if turn {
                current = 1 - current;
            }
            speaker
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn seg(start_ms: i64, end_ms: i64) -> Segment {
        Segment {
            start_ms,
            end_ms,
            text: "x".into(),
            speaker: None,
        }
    }

    fn turn(start_ms: i64, end_ms: i64, speaker: u8) -> SpeakerTurn {
        SpeakerTurn {
            start_ms,
            end_ms,
            speaker,
        }
    }

    #[test]
    fn strategy_keys_round_trip_and_aliases_parse() {
        for s in DiarizeStrategy::ALL {
            assert_eq!(DiarizeStrategy::parse(s.key()), Some(s));
        }
        assert_eq!(
            DiarizeStrategy::parse("TinyDiarize"),
            Some(DiarizeStrategy::Tdrz)
        );
        assert_eq!(
            DiarizeStrategy::parse("sherpa"),
            Some(DiarizeStrategy::Embedding)
        );
        assert_eq!(
            DiarizeStrategy::parse("community-1"),
            Some(DiarizeStrategy::Pyannote)
        );
        assert_eq!(DiarizeStrategy::parse("bogus"), None);
    }

    #[test]
    fn cycling_visits_every_strategy_and_wraps() {
        let mut s = DiarizeStrategy::Off;
        let mut seen = Vec::new();
        for _ in 0..DiarizeStrategy::ALL.len() {
            seen.push(s);
            s = s.next();
        }
        assert_eq!(seen, DiarizeStrategy::ALL);
        assert_eq!(s, DiarizeStrategy::Off);
    }

    #[test]
    fn recommendation_prefers_tdrz_only_for_tdrz_models() {
        assert_eq!(
            DiarizeStrategy::recommended_for(Some("ggml-small.en-tdrz.bin")),
            DiarizeStrategy::Tdrz
        );
        assert_eq!(
            DiarizeStrategy::recommended_for(Some("ggml-large-v3.bin")),
            DiarizeStrategy::Embedding
        );
        assert_eq!(
            DiarizeStrategy::recommended_for(Some("parakeet-tdt-0.6b-v3")),
            DiarizeStrategy::Embedding
        );
        assert_eq!(
            DiarizeStrategy::recommended_for(None),
            DiarizeStrategy::Embedding
        );
    }

    #[test]
    fn auto_resolves_per_model_and_fixed_strategies_stay_fixed() {
        let auto = DiarizeStrategy::Auto;
        assert_eq!(
            auto.resolve(Some("ggml-small.en-tdrz.bin")),
            DiarizeMethod::Tdrz
        );
        assert_eq!(auto.resolve(Some("ggml-large-v3.bin")), DiarizeMethod::Embedding);
        assert_eq!(DiarizeStrategy::Off.resolve(None), DiarizeMethod::None);
        assert_eq!(
            DiarizeStrategy::Tdrz.resolve(Some("anything.bin")),
            DiarizeMethod::Tdrz
        );
        assert_eq!(
            DiarizeStrategy::Embedding.resolve(Some("ggml-small.en-tdrz.bin")),
            DiarizeMethod::Embedding
        );
    }

    #[test]
    fn tdrz_requirement_tracks_the_library() {
        let dir = PathBuf::from("/nonexistent");
        let choice = DiarizeModelChoice::default();
        let none = requirements(DiarizeMethod::Tdrz, &[], &dir, &choice);
        assert_eq!(none.len(), 1);
        assert!(!none[0].satisfied);
        assert!(none[0].name.contains(TDRZ_FILE));

        let with = requirements(
            DiarizeMethod::Tdrz,
            &[ModelFile {
                path: PathBuf::from("/m/ggml-small.en-tdrz.bin"),
                name: "ggml-small.en-tdrz.bin".into(),
                size_bytes: 1,
                is_dir: false,
            }],
            &dir,
            &choice,
        );
        assert!(with[0].satisfied);
        assert!(requirements(DiarizeMethod::None, &[], &dir, &choice).is_empty());
    }

    #[test]
    fn download_note_names_what_is_missing_and_how_it_arrives() {
        let dir = PathBuf::from("/nonexistent");
        let choice = DiarizeModelChoice::default();
        let note = download_note(DiarizeMethod::Tdrz, &[], &dir, &choice).unwrap();
        assert!(note.contains(TDRZ_FILE), "{note}");
        assert!(note.contains(TDRZ_SIZE), "{note}");
        assert!(note.contains("Model management"), "{note}");
        assert_eq!(download_note(DiarizeMethod::None, &[], &dir, &choice), None);

        let ready = download_note(
            DiarizeMethod::Tdrz,
            &[ModelFile {
                path: PathBuf::from("/m/ggml-small.en-tdrz.bin"),
                name: "ggml-small.en-tdrz.bin".into(),
                size_bytes: 1,
                is_dir: false,
            }],
            &dir,
            &choice,
        );
        assert_eq!(ready, None);
    }

    #[test]
    fn segments_take_the_speaker_with_most_overlap() {
        let turns = [turn(0, 1000, 0), turn(1000, 3000, 1)];
        let labeled = assign_speakers(&[seg(0, 900), seg(900, 2900), seg(5000, 6000)], &turns);
        assert_eq!(labeled[0].speaker, Some(0));
        // 100 ms of speaker 0 vs 1900 ms of speaker 1
        assert_eq!(labeled[1].speaker, Some(1));
        // outside every turn → honest None
        assert_eq!(labeled[2].speaker, None);
    }

    #[test]
    fn overlap_accumulates_across_turns_of_the_same_speaker() {
        // speaker 0 speaks twice for 300 ms each inside the segment;
        // speaker 1 once for 500 ms → 600 vs 500 → speaker 0
        let turns = [turn(0, 300, 0), turn(300, 800, 1), turn(800, 1100, 0)];
        let labeled = assign_speakers(&[seg(0, 1100)], &turns);
        assert_eq!(labeled[0].speaker, Some(0));
    }

    #[test]
    fn overlap_ties_go_to_the_lower_speaker_index() {
        let turns = [turn(0, 500, 1), turn(500, 1000, 0)];
        let labeled = assign_speakers(&[seg(0, 1000)], &turns);
        assert_eq!(labeled[0].speaker, Some(0));
    }

    #[test]
    fn parse_turns_maps_seconds_to_ms() {
        let turns = parse_turns(
            r#"[{"start": 0.5, "end": 2.25, "speaker": 0},
                {"start": 2.25, "end": 4.0, "speaker": 3}]"#,
        )
        .unwrap();
        assert_eq!(
            turns,
            vec![
                SpeakerTurn {
                    start_ms: 500,
                    end_ms: 2250,
                    speaker: 0
                },
                SpeakerTurn {
                    start_ms: 2250,
                    end_ms: 4000,
                    speaker: 3
                },
            ]
        );
        assert!(parse_turns("[]").unwrap().is_empty());
        assert!(parse_turns("not json").is_err());
    }

    #[test]
    fn no_turns_is_all_speaker_a() {
        assert_eq!(alternate_speakers(&[false, false, false]), vec![0, 0, 0]);
    }

    #[test]
    fn turns_alternate_between_two_speakers() {
        // turn flag means the NEXT segment has a new speaker
        assert_eq!(
            alternate_speakers(&[true, false, true, true, false]),
            vec![0, 1, 1, 0, 1]
        );
    }
}
