//! Post-transcription speaker naming: after a diarized run, `n` opens a
//! modal listing every detected speaker with a voice sample — their
//! longest utterance, playable through `afplay` — so a human can replace
//! the anonymous A/B/C labels with real names (George, Marco, Sarah…).
//! Names flow into the TUI transcript and every export; closing the
//! dialog re-exports automatically if anything changed.

use crossterm::event::KeyCode;

use crate::app::App;
use crate::audio;
use crate::export::speaker_label;
use crate::format::clock_time;
use crate::transcribe::Segment;

/// Longest a played voice sample gets — identification needs seconds,
/// not the whole monologue.
const MAX_SAMPLE_MS: i64 = 15_000;
/// Longest a typed speaker name gets.
const MAX_NAME_CHARS: usize = 40;

/// One detected speaker and the utterance that best represents their
/// voice (the longest one — most signal for "who is this?").
pub struct NamingRow {
    pub speaker: u8,
    pub sample_text: String,
    pub sample_start_ms: i64,
    pub sample_end_ms: i64,
}

/// State of the speaker-naming modal.
pub struct SpeakerNaming {
    pub rows: Vec<NamingRow>,
    pub selected: usize,
    /// Some(text) while a name is being typed for the selected row
    pub input: Option<String>,
    /// A name changed — closing re-exports the transcript
    pub dirty: bool,
}

/// The longest utterance of every speaker in the transcript, one row per
/// speaker, ordered by speaker index.
pub(crate) fn naming_rows(segments: &[Segment]) -> Vec<NamingRow> {
    let mut best: std::collections::BTreeMap<u8, &Segment> = std::collections::BTreeMap::new();
    for seg in segments {
        let Some(speaker) = seg.speaker else { continue };
        let longer = best
            .get(&speaker)
            .map(|b| seg.end_ms - seg.start_ms > b.end_ms - b.start_ms)
            .unwrap_or(true);
        if longer {
            best.insert(speaker, seg);
        }
    }
    best.into_iter()
        .map(|(speaker, seg)| NamingRow {
            speaker,
            sample_text: seg.text.trim().to_string(),
            sample_start_ms: seg.start_ms,
            sample_end_ms: seg.end_ms,
        })
        .collect()
}

impl App {
    /// Open the naming dialog (`n`). Needs a diarized transcript to name
    /// anyone in.
    pub fn open_speaker_naming(&mut self) {
        if self.busy() {
            self.status = "Wait for the running job to finish before naming speakers".into();
            return;
        }
        let rows = naming_rows(&self.transcript.segments);
        if rows.is_empty() {
            self.status =
                "No speaker labels to name — run a transcription with diarization on (d)".into();
            return;
        }
        self.naming = Some(SpeakerNaming {
            rows,
            selected: 0,
            input: None,
            dirty: false,
        });
        self.status = "Name the speakers: Enter types a name · p plays their voice sample".into();
    }

    pub(crate) fn naming_key(&mut self, code: KeyCode) {
        // Typing mode first: the text input owns every key.
        let mut apply: Option<(u8, String)> = None;
        {
            let Some(naming) = &mut self.naming else { return };
            if let Some(input) = &mut naming.input {
                match code {
                    KeyCode::Char(c) if !c.is_control() && input.len() < MAX_NAME_CHARS => {
                        input.push(c)
                    }
                    KeyCode::Backspace => {
                        input.pop();
                    }
                    KeyCode::Esc => naming.input = None,
                    KeyCode::Enter => {
                        let text = naming.input.take().unwrap_or_default();
                        if let Some(row) = naming.rows.get(naming.selected) {
                            naming.dirty = true;
                            apply = Some((row.speaker, text));
                        }
                    }
                    _ => {}
                }
                if let Some((speaker, text)) = apply {
                    let text = text.trim().to_string();
                    if text.is_empty() {
                        self.transcript.speaker_names.remove(&speaker);
                        self.status = format!("{} back to anonymous", speaker_label(speaker));
                    } else {
                        self.status = format!("{} is now \"{text}\"", speaker_label(speaker));
                        self.transcript.speaker_names.insert(speaker, text);
                    }
                }
                return;
            }
        }
        match code {
            KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('n') => self.close_naming(),
            KeyCode::Up | KeyCode::Char('k') => {
                if let Some(naming) = &mut self.naming {
                    naming.selected = naming.selected.saturating_sub(1);
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if let Some(naming) = &mut self.naming {
                    naming.selected = (naming.selected + 1).min(naming.rows.len() - 1);
                }
            }
            KeyCode::Enter => {
                let prefill = self.naming.as_ref().and_then(|naming| {
                    let row = naming.rows.get(naming.selected)?;
                    Some(
                        self.transcript
                            .speaker_names
                            .get(&row.speaker)
                            .cloned()
                            .unwrap_or_default(),
                    )
                });
                if let (Some(naming), Some(prefill)) = (&mut self.naming, prefill) {
                    naming.input = Some(prefill);
                }
            }
            KeyCode::Char('p') => self.play_speaker_sample(),
            _ => {}
        }
    }

    /// Close the dialog; if names changed, re-export so the files on
    /// disk carry them.
    fn close_naming(&mut self) {
        let dirty = self.naming.take().map(|n| n.dirty).unwrap_or(false);
        if !dirty {
            self.status = "Speaker naming closed".into();
            return;
        }
        match self.export() {
            Ok(folder) => {
                self.status = format!(
                    "Speaker names applied — re-exported to {}",
                    folder.display()
                );
            }
            Err(e) => self.status = format!("Speaker names applied — re-export failed: {e}"),
        }
        self.job_log.push(self.status.clone());
    }

    /// Play the selected speaker's voice sample: decode the source media
    /// on a thread, cut the sample span, and hand it to `afplay`
    /// (macOS). Fire-and-forget — a long file takes a moment to decode.
    fn play_speaker_sample(&mut self) {
        let Some(naming) = &self.naming else { return };
        let Some(row) = naming.rows.get(naming.selected) else {
            return;
        };
        let Some(source) = self.transcript.source.clone() else {
            self.status = "No source media to sample".into();
            return;
        };
        let (speaker, start_ms) = (row.speaker, row.sample_start_ms);
        let end_ms = row.sample_end_ms.min(start_ms + MAX_SAMPLE_MS);
        let label = self
            .transcript
            .speaker_names
            .get(&speaker)
            .cloned()
            .unwrap_or_else(|| speaker_label(speaker));
        self.status = format!(
            "Extracting {label}'s voice sample [{} → {}] — plays when ready…",
            clock_time(start_ms),
            clock_time(end_ms)
        );
        std::thread::spawn(move || {
            let Ok(decoded) = audio::load_media(&source) else {
                return;
            };
            let rate = audio::WHISPER_SAMPLE_RATE as i64;
            let clamp = |ms: i64| ((ms * rate / 1000).max(0) as usize).min(decoded.samples.len());
            let (a, b) = (clamp(start_ms), clamp(end_ms));
            if b <= a {
                return;
            }
            let path = std::env::temp_dir().join(format!(
                "transcribe-stt-sample-{}-{speaker}.wav",
                std::process::id()
            ));
            if audio::write_wav_16k_mono(&path, &decoded.samples[a..b]).is_ok() {
                let _ = std::process::Command::new("afplay").arg(&path).status();
                let _ = std::fs::remove_file(&path);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start_ms: i64, end_ms: i64, text: &str, speaker: Option<u8>) -> Segment {
        Segment {
            start_ms,
            end_ms,
            text: text.into(),
            speaker,
        }
    }

    #[test]
    fn rows_pick_the_longest_utterance_per_speaker() {
        let segments = vec![
            seg(0, 1000, "hi", Some(0)),
            seg(1000, 6000, "long monologue", Some(0)),
            seg(6000, 7000, "yes", Some(1)),
            seg(7000, 7200, "unlabeled", None),
        ];
        let rows = naming_rows(&segments);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].speaker, 0);
        assert_eq!(rows[0].sample_text, "long monologue");
        assert_eq!(rows[0].sample_start_ms, 1000);
        assert_eq!(rows[1].speaker, 1);
        assert_eq!(rows[1].sample_text, "yes");
    }

    #[test]
    fn no_labeled_segments_no_rows() {
        assert!(naming_rows(&[seg(0, 1000, "hi", None)]).is_empty());
        assert!(naming_rows(&[]).is_empty());
    }
}
