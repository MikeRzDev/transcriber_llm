//! The transcription-language input, opened with `i` on the base
//! screen. Empty or "auto" means whisper detects the language; an ISO
//! 639-1 code skips detection and forces that language.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;

use crate::app::App;
use crate::ui::layout::centered_rect_rows;
use crate::ui::theme::{ACCENT, DIM};

pub(super) fn draw_language_prompt(frame: &mut Frame, app: &App) {
    let Some(input) = &app.language_input else {
        return;
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT))
        .title(" transcription language ");

    let lines = vec![
        Line::raw(""),
        Line::from(vec![
            Span::raw("  Language: "),
            Span::styled(
                format!("{input}▏"),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::raw(""),
        Line::from(Span::styled(
            "  ISO 639-1 code (en, es, de, fr…) skips auto-detection",
            Style::default().fg(DIM),
        )),
        Line::from(Span::styled(
            "  letters · Enter save · empty or auto = detect · Esc cancel",
            Style::default().fg(DIM),
        )),
    ];

    let area = centered_rect_rows(56, lines.len() as u16 + 2, frame.area());
    frame.render_widget(Clear, area);
    frame.render_widget(Paragraph::new(lines).block(block), area);
}
