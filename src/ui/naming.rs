//! The speaker-naming modal (`n`): every detected speaker with their
//! voice sample (longest utterance), an inline name editor, and playback
//! via afplay so a human can tell who is who.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;

use crate::app::App;
use crate::export::speaker_label;
use crate::format::clock_time;
use crate::ui::layout::centered_rect;
use crate::ui::theme::{highlight_style, speaker_color, ACCENT, DIM};

/// Sample text is trimmed to fit one dialog line.
const SAMPLE_CHARS: usize = 56;

pub(super) fn draw_speaker_naming(frame: &mut Frame, app: &App) {
    let Some(naming) = &app.naming else { return };
    let area = centered_rect(74, 70, frame.area());
    frame.render_widget(Clear, area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT))
        .title(" name the speakers ");

    let mut lines = vec![Line::raw("")];
    for (i, row) in naming.rows.iter().enumerate() {
        let selected = i == naming.selected;
        let letter = speaker_label(row.speaker);
        let color = speaker_color(row.speaker);

        // Line 1: "Speaker A → George" (or the live input while typing)
        let mut spans = vec![
            Span::raw(if selected { "  ▸ " } else { "    " }),
            Span::styled(
                format!("{letter:<10}"),
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            ),
            Span::styled(" → ", Style::default().fg(DIM)),
        ];
        if selected && naming.input.is_some() {
            let input = naming.input.as_deref().unwrap_or("");
            spans.push(Span::styled(
                format!("{input}▏"),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ));
        } else {
            match app.transcript.speaker_names.get(&row.speaker) {
                Some(name) => spans.push(Span::styled(
                    name.clone(),
                    if selected {
                        highlight_style()
                    } else {
                        Style::default().fg(Color::White)
                    },
                )),
                None => spans.push(Span::styled(
                    "unnamed — Enter to name",
                    if selected {
                        highlight_style()
                    } else {
                        Style::default().fg(DIM)
                    },
                )),
            }
        }
        lines.push(Line::from(spans));

        // Line 2: the voice sample this row is identified by
        let mut sample: String = row.sample_text.chars().take(SAMPLE_CHARS).collect();
        if row.sample_text.chars().count() > SAMPLE_CHARS {
            sample.push('…');
        }
        lines.push(Line::from(Span::styled(
            format!(
                "        sample [{} → {}] “{sample}”",
                clock_time(row.sample_start_ms),
                clock_time(row.sample_end_ms)
            ),
            Style::default().fg(DIM),
        )));
        lines.push(Line::raw(""));
    }

    lines.push(Line::from(Span::styled(
        if naming.input.is_some() {
            "  Type the name · Enter saves · empty clears · Esc keeps the old one"
        } else {
            "  ↑↓ select · Enter name · p play voice sample · Esc done (re-exports if changed)"
        },
        Style::default().fg(DIM),
    )));

    frame.render_widget(Paragraph::new(lines).block(block), area);
}
