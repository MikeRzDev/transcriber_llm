//! The diarization speaker-count input, opened with `p` on the base
//! screen while a clustering diarization strategy (embeddings/pyannote)
//! is selected. Empty means auto-detect; a fixed count pins the
//! clustering when the number of speakers is known.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;

use crate::app::App;
use crate::ui::layout::centered_rect_rows;
use crate::ui::theme::{ACCENT, DIM};

pub(super) fn draw_speakers_prompt(frame: &mut Frame, app: &App) {
    let Some(input) = &app.speakers_input else {
        return;
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT))
        .title(" number of speakers ");

    let lines = vec![
        Line::raw(""),
        Line::from(vec![
            Span::raw("  Speakers: "),
            Span::styled(
                format!("{input}▏"),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::raw(""),
        Line::from(Span::styled(
            "  Known count (1–26) pins the clustering — labels get noticeably better",
            Style::default().fg(DIM),
        )),
        Line::from(Span::styled(
            "  digits · Enter save · empty = auto-detect · Esc cancel",
            Style::default().fg(DIM),
        )),
    ];

    let area = centered_rect_rows(56, lines.len() as u16 + 2, frame.area());
    frame.render_widget(Clear, area);
    frame.render_widget(Paragraph::new(lines).block(block), area);
}
