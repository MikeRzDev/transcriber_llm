use ratatui::layout::{Alignment, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;

use crate::app::{App, Focus, TranscriptState, WorkState};
use crate::format::clock_time;
use crate::ui::layout::areas;
use crate::ui::theme::{ACCENT, DIM};

/// Pin the transcript scroll to its content for this frame. Runs in the
/// update phase (before `draw`) so rendering itself never mutates state.
pub fn clamp_transcript(app: &mut App, area: Rect) {
    if app.transcript.segments.is_empty() && app.transcript.partial.is_none() {
        return;
    }
    let key_rows = crate::ui::status::keys_rows(app, area.width);
    let transcript_area = areas(area, key_rows).transcript;
    // Block borders take one cell on each side
    let inner_width = transcript_area.width.saturating_sub(2) as usize;
    let viewport = transcript_area.height.saturating_sub(2) as usize;
    let text_width = inner_width.max(1);
    prepare_layout(&mut app.transcript, text_width);
    let total = app.transcript.layout.committed.len() + app.transcript.layout.partial.len();
    app.transcript.clamp(total, viewport);
}

/// Wrap one immutable segment; used once for history and when partial text changes.
fn segment_lines(
    transcript: &TranscriptState,
    seg: &crate::transcribe::Segment,
    partial: bool,
    text_width: usize,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let stamp = if text_width < 24 {
        String::new()
    } else {
        format!(
            "[{} → {}] ",
            clock_time(seg.start_ms),
            clock_time(seg.end_ms)
        )
    };
    let speaker = seg.speaker.map(|s| {
        // Assigned name if the user labeled this speaker, else the letter
        let tag = transcript
            .speaker_names
            .get(&s)
            .cloned()
            .unwrap_or_else(|| ((b'A' + s.min(25)) as char).to_string());
        (format!("{tag}: "), crate::ui::theme::speaker_color(s))
    });
    let prefix_len = Span::raw(stamp.clone()).width()
        + speaker
            .as_ref()
            .map(|(t, _)| Span::raw(t.clone()).width())
            .unwrap_or(0);
    let indent = " ".repeat(prefix_len);
    let wrapped = textwrap::wrap(
        seg.text.trim(),
        text_width.saturating_sub(prefix_len).max(1),
    );
    for (i, piece) in wrapped.iter().enumerate() {
        if i == 0 {
            let mut spans = vec![Span::styled(stamp.clone(), Style::default().fg(DIM))];
            if let Some((tag, color)) = &speaker {
                spans.push(Span::styled(
                    tag.clone(),
                    Style::default().fg(*color).add_modifier(Modifier::BOLD),
                ));
            }
            spans.push(Span::styled(
                piece.to_string(),
                if partial {
                    Style::default().fg(ratatui::style::Color::Yellow)
                } else {
                    Style::default()
                },
            ));
            lines.push(Line::from(spans));
        } else {
            lines.push(Line::from(vec![
                Span::raw(indent.clone()),
                Span::styled(
                    piece.to_string(),
                    if partial {
                        Style::default().fg(ratatui::style::Color::Yellow)
                    } else {
                        Style::default()
                    },
                ),
            ]));
        }
    }
    lines
}

fn prepare_layout(transcript: &mut TranscriptState, width: usize) {
    if transcript.layout.width != width
        || transcript.layout.names != transcript.speaker_names
        || transcript.layout.segment_count > transcript.segments.len()
    {
        transcript.invalidate_layout();
        transcript.layout.width = width;
        transcript.layout.names = transcript.speaker_names.clone();
    }
    for index in transcript.layout.segment_count..transcript.segments.len() {
        let lines = segment_lines(transcript, &transcript.segments[index], false, width);
        transcript.layout.committed.extend(lines);
        #[cfg(test)]
        {
            transcript.layout.wrapped_segments += 1;
        }
    }
    transcript.layout.segment_count = transcript.segments.len();
    if transcript.layout.partial_source != transcript.partial {
        transcript.layout.partial = transcript
            .partial
            .as_ref()
            .map(|seg| segment_lines(transcript, seg, true, width))
            .unwrap_or_default();
        transcript
            .layout
            .partial_source
            .clone_from(&transcript.partial);
    }
}

fn visible_cached_lines(transcript: &TranscriptState, height: usize) -> Vec<Line<'static>> {
    let cache = &transcript.layout;
    let start = transcript.scroll.min(cache.committed.len());
    let end = start.saturating_add(height).min(cache.committed.len());
    let mut visible = cache.committed[start..end].to_vec();
    let partial_start = transcript
        .scroll
        .saturating_sub(cache.committed.len())
        .min(cache.partial.len());
    let partial_end = partial_start
        .saturating_add(height - visible.len())
        .min(cache.partial.len());
    visible.extend_from_slice(&cache.partial[partial_start..partial_end]);
    visible
}

// Fallback for callers drawing without the normal pre-draw clamp/update phase.
fn transcript_lines(transcript: &TranscriptState, width: usize) -> Vec<Line<'static>> {
    transcript
        .segments
        .iter()
        .map(|s| (s, false))
        .chain(transcript.partial.iter().map(|s| (s, true)))
        .flat_map(|(segment, partial)| segment_lines(transcript, segment, partial, width))
        .collect()
}

pub(super) fn draw_transcript(frame: &mut Frame, area: Rect, app: &App) {
    let focused = app.focus == Focus::Transcript;
    let border_style = if focused {
        Style::default().fg(ACCENT)
    } else {
        Style::default().fg(DIM)
    };

    let title = if app.live.visible {
        format!(
            " Live transcript · {} ",
            if app.transcript.follow {
                "following"
            } else {
                "paused · G resumes"
            }
        )
    } else {
        app.transcript
            .source
            .as_ref()
            .and_then(|p| p.file_name())
            .map(|n| format!(" {} ", n.to_string_lossy()))
            .unwrap_or_else(|| " transcript ".into())
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border_style)
        .title(title);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if app.transcript.segments.is_empty() && app.transcript.partial.is_none() {
        let placeholder = match app.work {
            WorkState::Idle => {
                "No transcript yet.\n\nPick an audio file on the left and press Enter."
            }
            WorkState::Recording => "Listening…\n\nSpeak into your microphone.\nLive text appears here as it is decoded.",
            _ => "Waiting for first segment…",
        };
        frame.render_widget(
            Paragraph::new(placeholder)
                .style(Style::default().fg(DIM))
                .alignment(Alignment::Center)
                .wrap(Wrap { trim: true }),
            inner,
        );
        return;
    }

    let text_width = (inner.width as usize).max(1);
    let cache = &app.transcript.layout;
    let lines = if cache.width == text_width
        && cache.segment_count == app.transcript.segments.len()
        && cache.names == app.transcript.speaker_names
        && cache.partial_source == app.transcript.partial
    {
        visible_cached_lines(&app.transcript, inner.height as usize)
    } else {
        transcript_lines(&app.transcript, text_width)
            .into_iter()
            .skip(app.transcript.scroll)
            .take(inner.height as usize)
            .collect()
    };
    frame.render_widget(Paragraph::new(lines), inner);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcribe::Segment;

    fn segment(i: usize) -> Segment {
        Segment {
            start_ms: (i * 1000) as i64,
            end_ms: ((i + 1) * 1000) as i64,
            text: format!(
                "Sentence {i}: palabras con acentos, 中文, and enough words to wrap across lines."
            ),
            speaker: None,
        }
    }

    fn assert_matches_reference(t: &mut TranscriptState, width: usize) {
        prepare_layout(t, width);
        let expected = transcript_lines(t, width);
        let cached = t
            .layout
            .committed
            .iter()
            .chain(t.layout.partial.iter())
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(cached, expected);
        for scroll in [
            0,
            1,
            expected.len() / 2,
            expected.len().saturating_sub(1),
            expected.len() + 5,
        ] {
            t.scroll = scroll;
            assert_eq!(
                visible_cached_lines(t, 7),
                expected
                    .iter()
                    .skip(scroll)
                    .take(7)
                    .cloned()
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn cached_layout_matches_wrapping_scrolling_names_and_replacement() {
        let mut t = TranscriptState::new();
        t.segments = (0..5).map(segment).collect();
        t.segments[0].speaker = Some(0);
        t.partial = Some(segment(5));
        assert_matches_reference(&mut t, 60);
        t.partial
            .as_mut()
            .unwrap()
            .text
            .push_str(" latest partial text");
        assert_matches_reference(&mut t, 60);
        t.segments.push(t.partial.take().unwrap());
        assert_matches_reference(&mut t, 60);
        t.speaker_names.insert(0, "Alejandra".into());
        assert_matches_reference(&mut t, 60);
        assert_matches_reference(&mut t, 20);
        t.segments[0].text = "Replaced transcript".into();
        t.invalidate_layout();
        assert_matches_reference(&mut t, 20);
        t.begin("new session".into(), "model".into(), None);
        assert_matches_reference(&mut t, 60);
    }

    #[test]
    fn partial_updates_do_not_rewrap_recording_history() {
        let mut t = TranscriptState::new();
        t.segments = (0..10000).map(segment).collect();
        prepare_layout(&mut t, 60);
        assert_eq!(t.layout.wrapped_segments, 10000);
        for i in 0..100 {
            t.partial = Some(segment(10000 + i));
            prepare_layout(&mut t, 60);
            t.clamp(t.layout.committed.len() + t.layout.partial.len(), 25);
            assert!(visible_cached_lines(&t, 25).len() <= 25);
        }
        assert_eq!(
            t.layout.wrapped_segments, 10000,
            "history was rewrapped during streaming"
        );
        t.segments.push(segment(10100));
        t.partial = None;
        prepare_layout(&mut t, 60);
        assert_eq!(
            t.layout.wrapped_segments, 10001,
            "only the appended segment should be wrapped"
        );
    }

    #[test]
    #[ignore = "local CPU render-cost measurement; no model or microphone"]
    fn measure_history_render_cost() {
        use std::hint::black_box;
        use std::time::Instant;
        let mut rows = Vec::new();
        for size in [100, 10000] {
            let mut t = TranscriptState::new();
            t.segments = (0..size).map(segment).collect();
            prepare_layout(&mut t, 60);
            t.clamp(t.layout.committed.len(), 25);
            let started = Instant::now();
            black_box(transcript_lines(&t, 60));
            let old_ms = started.elapsed().as_secs_f64() * 1000.0;
            let started = Instant::now();
            for update in 0..100 {
                t.partial = Some(segment(size + update));
                prepare_layout(&mut t, 60);
                t.clamp(t.layout.committed.len() + t.layout.partial.len(), 25);
                black_box(visible_cached_lines(&t, 25));
            }
            let cached_ms = started.elapsed().as_secs_f64() * 1000.0 / 100.0;
            rows.push(serde_json::json!({"history_segments":size,"full_history_wrap_ms":old_ms,
                "cached_partial_update_and_viewport_ms":cached_ms,"wrapped_history_segments":t.layout.wrapped_segments}));
        }
        let report = serde_json::json!({"test":"steady-state transcript layout cost", "rows":rows,
            "notes":"CPU-only: cached timing includes changing the partial, layout preparation, scrolling and cloning visible lines. Full terminal/GPU latency is not measured here. Width/name changes reflow history once."});
        let path = std::env::var("TRANSCRIBE_RENDER_COST_REPORT").expect("report path");
        std::fs::write(&path, serde_json::to_string_pretty(&report).unwrap()).unwrap();
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    }
}
