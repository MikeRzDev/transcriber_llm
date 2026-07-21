use ratatui::layout::{Alignment, Rect};
use ratatui::style::{Color, Modifier, Style};
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
    if app.transcript.segments.is_empty() {
        return;
    }
    let transcript_area = areas(area).transcript;
    // Block borders take one cell on each side
    let inner_width = transcript_area.width.saturating_sub(2) as usize;
    let viewport = transcript_area.height.saturating_sub(2) as usize;
    let text_width = inner_width.saturating_sub(2).max(20);
    let total = transcript_lines(&app.transcript, text_width).len();
    app.transcript.clamp(total, viewport);
}

/// Wrapped display lines: "[mm:ss → mm:ss] text", speaker-tagged when
/// diarized. Shared by rendering and the pre-draw scroll clamp.
fn transcript_lines(transcript: &TranscriptState, text_width: usize) -> Vec<Line<'static>> {
    let mut lines: Vec<Line> = Vec::new();
    for seg in &transcript.segments {
        let stamp = format!(
            "[{} → {}] ",
            clock_time(seg.start_ms),
            clock_time(seg.end_ms)
        );
        let speaker = seg.speaker.map(|s| {
            (
                format!("{}: ", if s == 0 { "A" } else { "B" }),
                if s == 0 { Color::Cyan } else { Color::Magenta },
            )
        });
        let prefix_len = stamp.len() + speaker.as_ref().map(|(t, _)| t.len()).unwrap_or(0);
        let indent = " ".repeat(prefix_len);
        let wrapped = textwrap::wrap(
            seg.text.trim(),
            text_width.saturating_sub(prefix_len).max(10),
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
                spans.push(Span::raw(piece.to_string()));
                lines.push(Line::from(spans));
            } else {
                lines.push(Line::from(vec![
                    Span::raw(indent.clone()),
                    Span::raw(piece.to_string()),
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

    let title = app
        .transcript
        .source
        .as_ref()
        .and_then(|p| p.file_name())
        .map(|n| format!(" {} ", n.to_string_lossy()))
        .unwrap_or_else(|| " transcript ".into());

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border_style)
        .title(title);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if app.transcript.segments.is_empty() {
        let placeholder = match app.work {
            WorkState::Idle => {
                "No transcript yet.\n\nPick an audio file on the left and press Enter."
            }
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

    let text_width = (inner.width as usize).saturating_sub(2).max(20);
    let lines = transcript_lines(&app.transcript, text_width);
    frame.render_widget(
        Paragraph::new(lines).scroll((app.transcript.scroll as u16, 0)),
        inner,
    );
}
