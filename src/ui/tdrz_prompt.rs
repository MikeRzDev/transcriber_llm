//! The "download the tdrz model?" offer, shown when a tdrz-based
//! diarization strategy is selected (or a job needs it) while the model
//! is missing. Yes opens Model management with the download running.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;

use crate::app::App;
use crate::diarize::{TDRZ_FILE, TDRZ_REPO, TDRZ_SIZE};
use crate::ui::layout::centered_rect;
use crate::ui::theme::{highlight_style, ACCENT, DIM};

pub(super) fn draw_tdrz_prompt(frame: &mut Frame, app: &App) {
    let Some(prompt) = &app.tdrz_prompt else { return };
    let area = centered_rect(64, 36, frame.area());
    frame.render_widget(Clear, area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT))
        .title(" download diarization model? ");

    let option = |label: &str, selected: bool| {
        if selected {
            Span::styled(format!("[ {label} ]"), highlight_style())
        } else {
            Span::styled(format!("[ {label} ]"), Style::default().fg(DIM))
        }
    };

    let lines = vec![
        Line::raw(""),
        Line::raw("  TinyDiarize transcribes with a special whisper build that is"),
        Line::raw("  not installed yet."),
        Line::raw(""),
        Line::from(vec![
            Span::styled("  model: ", Style::default().fg(DIM)),
            Span::styled(TDRZ_FILE, Style::default().add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("  size:  ", Style::default().fg(DIM)),
            Span::raw(TDRZ_SIZE),
        ]),
        Line::from(vec![
            Span::styled("  from:  ", Style::default().fg(DIM)),
            Span::raw(format!("huggingface.co/{TDRZ_REPO}")),
        ]),
        Line::from(Span::styled(
            "  note:  English only, 2 speakers — the embeddings strategy has no such limits",
            Style::default().fg(Color::Yellow),
        )),
        Line::raw(""),
        Line::from(vec![
            Span::raw("  "),
            option("Yes, download", prompt.yes_selected),
            Span::raw("   "),
            option("No", !prompt.yes_selected),
        ]),
        Line::raw(""),
        Line::from(Span::styled(
            "  ←→ choose · Enter confirm · y / n shortcuts · Esc cancel",
            Style::default().fg(DIM),
        )),
    ];

    frame.render_widget(Paragraph::new(lines).block(block), area);
}
