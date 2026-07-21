use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;

use crate::app::{App, SettingsRow};
use crate::export::ExportFormat;
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
        let editing =
            app.settings.language_input.is_some() || app.settings.speakers_input.is_some();
        if app.settings.selected == row && !editing {
            highlight_style()
        } else {
            Style::default()
        }
    };

    // Cheap on purpose (rendered per frame): resolving the strategy is
    // pure; the download checks live in the status line via Enter/d.
    let diarize_state = {
        use crate::diarize::DiarizeStrategy;
        let strategy = app.config.diarize;
        match strategy {
            DiarizeStrategy::Auto => {
                let method = strategy
                    .resolve(app.library.selected.as_ref().map(|m| m.name.as_str()));
                format!("Auto → {}", method.label())
            }
            _ => strategy.label().to_string(),
        }
    };

    let speakers_line: Line = if let Some(input) = &app.settings.speakers_input {
        Line::from(vec![
            Span::raw("  Speakers:       "),
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
                "  Speakers:       {}",
                match app.config.diarize_speakers {
                    Some(n) => format!("exactly {n}"),
                    None => "auto-detect".into(),
                }
            ),
            row_style(SettingsRow::DiarizeSpeakers),
        ))
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
            format!(
                "  Export formats: {}",
                app.config
                    .export_formats
                    .iter()
                    .map(|f| f.key())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            row_style(SettingsRow::ExportFormats),
        )),
        Line::raw(""),
        Line::from(Span::styled(
            "  Model management  →  installed models · download · delete",
            row_style(SettingsRow::ModelManagement),
        )),
        Line::raw(""),
        Line::from(Span::styled(
            format!("  Diarization:    {diarize_state}"),
            row_style(SettingsRow::Diarize),
        )),
        Line::raw(""),
        speakers_line,
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
            } else if app.settings.speakers_input.is_some() {
                "  Known speaker count (1–26) pins the clustering; empty = auto-detect"
            } else {
                "  ↑↓ select · Enter change · Esc close  (saved to ~/.config/transcribe-stt)"
            },
            Style::default().fg(DIM),
        )),
    ];

    frame.render_widget(Paragraph::new(lines).block(block), area);
}

/// Checkbox dialog over the settings modal: which formats every
/// transcription exports. At least one always stays selected.
pub(super) fn draw_export_formats(frame: &mut Frame, app: &App) {
    let area = centered_rect(52, 40, frame.area());
    frame.render_widget(Clear, area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT))
        .title(" export formats ");

    let cursor = app.settings.formats_cursor.unwrap_or(0);
    let mut lines = vec![Line::raw("")];
    for (i, format) in ExportFormat::ALL.into_iter().enumerate() {
        let checked = app.config.export_formats.contains(&format);
        let mark = if checked { "[x]" } else { "[ ]" };
        let style = if i == cursor {
            highlight_style()
        } else if checked {
            Style::default()
        } else {
            Style::default().fg(DIM)
        };
        lines.push(Line::from(Span::styled(
            format!("  {mark} {}", format.label()),
            style,
        )));
    }
    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        "  ↑↓ select · Space/Enter toggle · Esc done · min. one stays on",
        Style::default().fg(DIM),
    )));

    frame.render_widget(Paragraph::new(lines).block(block), area);
}
