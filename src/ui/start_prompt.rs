use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;

use crate::app::App;
use crate::ui::layout::centered_rect;
use crate::ui::theme::{highlight_style, ACCENT, DIM};

/// The "transcribe this file?" confirmation shown before every job.
pub(super) fn draw_start_prompt(frame: &mut Frame, app: &App) {
    let Some(prompt) = &app.start_prompt else {
        return;
    };
    let area = centered_rect(64, 30, frame.area());
    frame.render_widget(Clear, area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT))
        .title(" start transcription? ");

    let option = |label: &str, selected: bool| {
        if selected {
            Span::styled(format!("[ {label} ]"), highlight_style())
        } else {
            Span::styled(format!("[ {label} ]"), Style::default().fg(DIM))
        }
    };

    let model = app
        .library
        .selected
        .as_ref()
        .map(|m| m.name.clone())
        .unwrap_or_else(|| "none — pick one in settings".into());
    let formats = app
        .config
        .export_formats
        .iter()
        .map(|f| f.key())
        .collect::<Vec<_>>()
        .join(", ");
    let file_name = prompt
        .audio
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| prompt.audio.display().to_string());

    let lines = vec![
        Line::raw(""),
        Line::from(vec![
            Span::styled("  file:    ", Style::default().fg(DIM)),
            Span::styled(file_name, Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("  model:   ", Style::default().fg(DIM)),
            Span::raw(model),
        ]),
        Line::from(vec![
            Span::styled("  exports: ", Style::default().fg(DIM)),
            Span::raw(formats),
        ]),
        Line::raw(""),
        Line::from(vec![
            Span::raw("  "),
            option("Yes, transcribe", prompt.yes_selected),
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
