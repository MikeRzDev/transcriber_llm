//! Transcript output formats. The `.llm.md` format is the primary one:
//! it is designed for an LLM to interpret the conversation and act on it.

use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::format::{llm_time, srt_time};
use crate::transcribe::Segment;

/// Everything needed to render a transcript in any output format.
pub struct TranscriptDoc<'a> {
    pub segments: &'a [Segment],
    pub source_name: String,
    pub duration_secs: Option<f32>,
    pub language: Option<&'a str>,
    pub model_name: Option<&'a str>,
}

impl TranscriptDoc<'_> {
    /// Write all formats into `<out_base>/<source-stem>_<timestamp>/`.
    /// Returns the paths written.
    pub fn write_all(&self, out_base: &Path) -> Result<Vec<PathBuf>> {
        let stem = Path::new(&self.source_name)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "transcript".into());
        let dir = out_base.join(format!("{stem}_{}", local_timestamp()));
        std::fs::create_dir_all(&dir)?;
        let outputs = [
            (dir.join(format!("{stem}.llm.md")), self.llm_markdown()),
            (dir.join(format!("{stem}.segments.json")), self.json()),
            (dir.join(format!("{stem}.txt")), self.plain_text()),
            (dir.join(format!("{stem}.srt")), self.srt()),
        ];
        let mut written = Vec::with_capacity(outputs.len());
        for (path, contents) in outputs {
            std::fs::write(&path, contents)?;
            written.push(path);
        }
        Ok(written)
    }

    pub fn plain_text(&self) -> String {
        let mut out = String::new();
        for seg in self.segments {
            if let Some(speaker) = seg.speaker {
                out.push_str(&format!("{}: ", speaker_label(speaker)));
            }
            out.push_str(seg.text.trim());
            out.push('\n');
        }
        out
    }

    pub fn srt(&self) -> String {
        let mut out = String::new();
        for (i, seg) in self.segments.iter().enumerate() {
            let speaker = seg
                .speaker
                .map(|s| format!("{}: ", speaker_label(s)))
                .unwrap_or_default();
            out.push_str(&format!(
                "{}\n{} --> {}\n{}{}\n\n",
                i + 1,
                srt_time(seg.start_ms),
                srt_time(seg.end_ms),
                speaker,
                seg.text.trim()
            ));
        }
        out
    }

    /// Markdown optimized for LLM consumption: machine-readable metadata
    /// up front, then the conversation as timestamped paragraphs.
    pub fn llm_markdown(&self) -> String {
        let duration = self
            .duration_secs
            .map(|d| llm_time((d * 1000.0) as i64))
            .unwrap_or_else(|| "unknown".into());

        let mut out = String::new();
        out.push_str("---\n");
        out.push_str("type: conversation-transcript\n");
        out.push_str(&format!("source: {}\n", self.source_name));
        out.push_str(&format!("duration: {duration}\n"));
        out.push_str(&format!(
            "language: {}\n",
            self.language.unwrap_or("unknown")
        ));
        out.push_str(&format!(
            "model: {} (whisper.cpp, local)\n",
            self.model_name.unwrap_or("unknown")
        ));
        out.push_str(&format!("segments: {}\n", self.segments.len()));
        let diarized = self.segments.iter().any(|s| s.speaker.is_some());
        if diarized {
            out.push_str(
                "diarization: speaker-turn detection (tinydiarize), two-speaker labeling\n",
            );
        } else {
            out.push_str("diarization: none\n");
        }
        out.push_str("---\n\n");
        out.push_str(&format!("# Transcript: {}\n\n", self.source_name));
        if diarized {
            out.push_str(
                "> Verbatim automatic speech recognition output. `[HH:MM:SS]` anchors mark where\n\
                 > each paragraph starts in the source media. Speaker labels come from acoustic\n\
                 > turn detection assuming a two-person conversation: labels A/B are consistent\n\
                 > but arbitrary (A is whoever speaks first), and occasional turns may be missed\n\
                 > or split — treat labels as strong hints, not ground truth.\n\n",
            );
        } else {
            out.push_str(
                "> Verbatim automatic speech recognition output. `[HH:MM:SS]` anchors mark where\n\
                 > each paragraph starts in the source media. Paragraph breaks correspond to\n\
                 > pauses in speech and often indicate speaker turns; no speaker labels are\n\
                 > available, so infer speakers and conversational structure from context.\n\n",
            );
        }
        for para in paragraphs(self.segments) {
            match para.speaker {
                Some(speaker) => out.push_str(&format!(
                    "[{}] {}: {}\n\n",
                    llm_time(para.start_ms),
                    speaker_label(speaker),
                    para.text
                )),
                None => out.push_str(&format!("[{}] {}\n\n", llm_time(para.start_ms), para.text)),
            }
        }
        out
    }

    /// Segment-level JSON for programmatic pipelines.
    pub fn json(&self) -> String {
        let value = serde_json::json!({
            "source": self.source_name,
            "duration_seconds": self.duration_secs.unwrap_or(0.0),
            "language": self.language,
            "model": self.model_name,
            "segments": self.segments.iter().map(|seg| {
                serde_json::json!({
                    "start_ms": seg.start_ms,
                    "end_ms": seg.end_ms,
                    "speaker": seg.speaker.map(speaker_label),
                    "text": seg.text.trim(),
                })
            }).collect::<Vec<_>>(),
        });
        let mut out = serde_json::to_string_pretty(&value).unwrap_or_default();
        out.push('\n');
        out
    }
}

/// Local time as `YYYYMMDD_HHMMSS`, used to name export folders.
fn local_timestamp() -> String {
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    let t = unsafe { libc::time(std::ptr::null_mut()) };
    unsafe { libc::localtime_r(&t, &mut tm) };
    format!(
        "{:04}{:02}{:02}_{:02}{:02}{:02}",
        tm.tm_year + 1900,
        tm.tm_mon + 1,
        tm.tm_mday,
        tm.tm_hour,
        tm.tm_min,
        tm.tm_sec
    )
}

/// A paragraph of merged segments, optionally attributed to a speaker.
pub struct Paragraph {
    pub start_ms: i64,
    pub speaker: Option<u8>,
    pub text: String,
}

pub fn speaker_label(speaker: u8) -> String {
    format!("Speaker {}", (b'A' + speaker.min(25)) as char)
}

/// Merge raw ASR segments into readable paragraphs: a new paragraph starts
/// on a speaker change, a speech gap (pause), or when the current one gets
/// long.
pub fn paragraphs(segments: &[Segment]) -> Vec<Paragraph> {
    const GAP_MS: i64 = 1500;
    const MAX_CHARS: usize = 700;

    let mut paras: Vec<Paragraph> = Vec::new();
    let mut prev_end: i64 = 0;
    for seg in segments {
        let text = seg.text.trim();
        if text.is_empty() {
            continue;
        }
        let new_para = match paras.last() {
            None => true,
            Some(para) => {
                para.speaker != seg.speaker
                    || seg.start_ms - prev_end >= GAP_MS
                    || para.text.len() >= MAX_CHARS
            }
        };
        if new_para {
            paras.push(Paragraph {
                start_ms: seg.start_ms,
                speaker: seg.speaker,
                text: text.to_string(),
            });
        } else {
            let para = paras.last_mut().unwrap();
            para.text.push(' ');
            para.text.push_str(text);
        }
        prev_end = seg.end_ms;
    }
    paras
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start_ms: i64, end_ms: i64, text: &str) -> Segment {
        Segment {
            start_ms,
            end_ms,
            text: text.to_string(),
            speaker: None,
        }
    }

    fn spk(start_ms: i64, end_ms: i64, text: &str, speaker: u8) -> Segment {
        Segment {
            speaker: Some(speaker),
            ..seg(start_ms, end_ms, text)
        }
    }

    #[test]
    fn paragraphs_merge_and_split_on_gap() {
        let segments = vec![
            seg(0, 1000, "Hello there."),
            seg(1100, 2000, "How are you?"), // 100ms gap: same paragraph
            seg(5000, 6000, "Fine, thanks."), // 3s gap: new paragraph
        ];
        let paras = paragraphs(&segments);
        assert_eq!(paras.len(), 2);
        assert_eq!(paras[0].start_ms, 0);
        assert_eq!(paras[0].text, "Hello there. How are you?");
        assert_eq!(paras[1].start_ms, 5000);
        assert_eq!(paras[1].text, "Fine, thanks.");
    }

    #[test]
    fn paragraphs_skip_empty_segments() {
        let segments = vec![seg(0, 1000, "  "), seg(1000, 2000, "Text.")];
        assert_eq!(paragraphs(&segments).len(), 1);
    }

    #[test]
    fn json_is_valid_and_escapes() {
        let segments = vec![seg(0, 1000, "He said \"hi\"\n\tand left")];
        let doc = TranscriptDoc {
            segments: &segments,
            source_name: "a \"b\".mp4".into(),
            duration_secs: Some(1.0),
            language: Some("en"),
            model_name: None,
        };
        let parsed: serde_json::Value = serde_json::from_str(&doc.json()).unwrap();
        assert_eq!(parsed["source"], "a \"b\".mp4");
        assert_eq!(parsed["model"], serde_json::Value::Null);
        assert_eq!(parsed["segments"][0]["text"], "He said \"hi\"\n\tand left");
    }

    #[test]
    fn llm_markdown_has_frontmatter_and_anchors() {
        let segments = vec![seg(0, 1000, "Hello.")];
        let doc = TranscriptDoc {
            segments: &segments,
            source_name: "demo.wav".into(),
            duration_secs: Some(61.0),
            language: Some("en"),
            model_name: Some("ggml-large-v3.bin"),
        };
        let md = doc.llm_markdown();
        assert!(md.starts_with("---\ntype: conversation-transcript\n"));
        assert!(md.contains("duration: 00:01:01\n"));
        assert!(md.contains("language: en\n"));
        assert!(md.contains("[00:00:00] Hello."));
    }

    #[test]
    fn srt_output_is_exact() {
        let segments = vec![seg(0, 1500, " Hello. "), seg(1500, 3000, "Bye.")];
        let doc = TranscriptDoc {
            segments: &segments,
            source_name: "x.wav".into(),
            duration_secs: None,
            language: None,
            model_name: None,
        };
        assert_eq!(
            doc.srt(),
            "1\n00:00:00,000 --> 00:00:01,500\nHello.\n\n2\n00:00:01,500 --> 00:00:03,000\nBye.\n\n"
        );
    }

    #[test]
    fn paragraphs_split_when_too_long() {
        // continuous speech (no gaps) but > MAX_CHARS forces a split
        let long = "word ".repeat(100); // 500 chars per segment
        let segments = vec![
            seg(0, 1000, &long),
            seg(1000, 2000, &long),
            seg(2000, 3000, "tail"),
        ];
        let paras = paragraphs(&segments);
        assert_eq!(
            paras.len(),
            2,
            "second 500-char segment pushes body past 700"
        );
        assert_eq!(paras[1].start_ms, 2000);
    }

    #[test]
    fn write_all_creates_the_four_formats() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static UNIQUE: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "transcribe-stt-export-{}-{}",
            std::process::id(),
            UNIQUE.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&dir).unwrap();

        let segments = vec![seg(0, 1000, "Hello.")];
        let doc = TranscriptDoc {
            segments: &segments,
            source_name: "clip.wav".into(),
            duration_secs: Some(1.0),
            language: Some("en"),
            model_name: Some("m.bin"),
        };
        let written = doc.write_all(&dir).unwrap();
        assert_eq!(written.len(), 4);
        let out_dir = written[0].parent().unwrap();
        assert_eq!(out_dir.parent().unwrap(), dir);
        let folder = out_dir.file_name().unwrap().to_string_lossy();
        assert!(
            folder.starts_with("clip_") && folder.len() == "clip_YYYYMMDD_HHMMSS".len(),
            "unexpected folder name: {folder}"
        );
        for ext in ["llm.md", "segments.json", "txt", "srt"] {
            assert!(
                out_dir.join(format!("clip.{ext}")).is_file(),
                "missing .{ext}"
            );
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn speaker_change_starts_new_paragraph() {
        let segments = vec![
            spk(0, 1000, "Hi.", 0),
            spk(1100, 2000, "How are you?", 0), // same speaker, tiny gap: merges
            spk(2100, 3000, "Good, thanks.", 1), // speaker flips: new paragraph
        ];
        let paras = paragraphs(&segments);
        assert_eq!(paras.len(), 2);
        assert_eq!(paras[0].speaker, Some(0));
        assert_eq!(paras[0].text, "Hi. How are you?");
        assert_eq!(paras[1].speaker, Some(1));
    }

    #[test]
    fn speaker_labels_flow_into_all_formats() {
        let segments = vec![spk(0, 1000, "Hi.", 0), spk(1000, 2000, "Hello.", 1)];
        let doc = TranscriptDoc {
            segments: &segments,
            source_name: "call.wav".into(),
            duration_secs: Some(2.0),
            language: Some("en"),
            model_name: Some("ggml-small.en-tdrz.bin"),
        };
        let md = doc.llm_markdown();
        assert!(md.contains("diarization: speaker-turn detection"));
        assert!(md.contains("Speaker A: Hi."));
        assert!(md.contains("Speaker B: Hello."));

        assert!(doc.plain_text().contains("Speaker B: Hello."));
        assert!(doc.srt().contains("Speaker A: Hi."));

        let parsed: serde_json::Value = serde_json::from_str(&doc.json()).unwrap();
        assert_eq!(parsed["segments"][0]["speaker"], "Speaker A");
        assert_eq!(parsed["segments"][1]["speaker"], "Speaker B");
    }

    #[test]
    fn speaker_label_letters() {
        assert_eq!(speaker_label(0), "Speaker A");
        assert_eq!(speaker_label(1), "Speaker B");
    }
}
