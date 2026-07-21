use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;

use crate::app::{App, SettingsRow};
use crate::ui::layout::centered_rect;
use crate::ui::theme::{highlight_style, ACCENT, DIM};

pub(super) fn draw_settings(frame: &mut Frame, app: &App) {
    let area = centered_rect(70, 55, frame.area());
    frame.render_widget(Clear, area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT))
        .title(" settings ");

    let model = app
        .library
        .selected
        .as_ref()
        .map(|m| format!("{} ({})", m.name, m.size_human()))
        .unwrap_or_else(|| "none".into());

    let row_style = |row: SettingsRow| {
        if app.settings.selected == row && app.settings.language_input.is_none() {
            highlight_style()
        } else {
            Style::default()
        }
    };

    let diarize_state = if app.config.diarize {
        "ON  (uses the tdrz model, English only)"
    } else {
        "OFF"
    };

    let language_line: Line = if let Some(input) = &app.settings.language_input {
        Line::from(vec![
            Span::raw("  Language:       "),
            Span::styled(
                format!("{input}▏"),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
        ])
    } else {
        Line::from(Span::styled(
            format!(
                "  Language:       {}",
                app.config.language.as_deref().unwrap_or("auto-detect")
            ),
            row_style(SettingsRow::Language),
        ))
    };

    let lines = vec![
        Line::raw(""),
        Line::from(Span::styled(
            format!("  Default model:  {model}"),
            row_style(SettingsRow::DefaultModel),
        )),
        Line::raw(""),
        Line::from(Span::styled(
            format!("  Models folder:  {}", app.library.dir.display()),
            row_style(SettingsRow::ModelsFolder),
        )),
        Line::raw(""),
        Line::from(Span::styled(
            format!("  Output folder:  {}", app.output_dir.display()),
            row_style(SettingsRow::OutputFolder),
        )),
        Line::raw(""),
        Line::from(Span::styled(
            "  Model management  →  search & download from Hugging Face",
            row_style(SettingsRow::ModelManagement),
        )),
        Line::raw(""),
        Line::from(Span::styled(
            format!("  Diarization:    {diarize_state}"),
            row_style(SettingsRow::Diarize),
        )),
        Line::raw(""),
        Line::from(Span::styled(
            format!("  Split mode:     {}", app.config.split_mode.label()),
            row_style(SettingsRow::SplitMode),
        )),
        Line::raw(""),
        language_line,
        Line::raw(""),
        Line::from(Span::styled(
            if app.settings.language_input.is_some() {
                "  Type an ISO 639-1 code (en, es, de, fr…) or auto, Enter to save"
            } else {
                "  ↑↓ select · Enter change · Esc close  (saved to ~/.config/transcribe-stt)"
            },
            Style::default().fg(DIM),
        )),
    ];

    frame.render_widget(Paragraph::new(lines).block(block), area);
}
