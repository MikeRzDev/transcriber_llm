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
    let total = transcript_lines(&app.transcript, text_width).len();
    app.transcript.clamp(total, viewport);
}

/// Wrapped display lines: "[mm:ss → mm:ss] text", speaker-tagged when
/// diarized. Shared by rendering and the pre-draw scroll clamp.
fn transcript_lines(transcript: &TranscriptState, text_width: usize) -> Vec<Line<'static>> {
    let mut lines: Vec<Line> = Vec::new();
    for (seg, partial) in transcript
        .segments
        .iter()
        .map(|s| (s, false))
        .chain(transcript.partial.iter().map(|s| (s, true)))
    {
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
    }
    lines
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
    let lines = transcript_lines(&app.transcript, text_width);
    frame.render_widget(
        Paragraph::new(
            lines
                .into_iter()
                .skip(app.transcript.scroll)
                .take(inner.height as usize)
                .collect::<Vec<_>>(),
        ),
        inner,
    );
}
