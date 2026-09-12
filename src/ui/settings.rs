use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;

use crate::app::{App, SettingsRow};
use crate::config::{language_label, INPUT_LANGUAGES};
use crate::export::ExportFormat;
use crate::ui::layout::{centered_rect, centered_rect_rows};
use crate::ui::theme::{highlight_style, ACCENT, DIM};

pub(super) fn draw_settings(frame: &mut Frame, app: &App) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT))
        .title(" settings ");

    let model = app
        .library
        .selected
        .as_ref()
        .map(|m| format!("{} ({})", m.display_name(), m.size_human()))
        .unwrap_or_else(|| "none".into());

    let row_style = |row: SettingsRow| {
        let editing = app.settings.hf_token_input.is_some();
        if app.settings.selected == row && !editing {
            highlight_style()
        } else {
            Style::default()
        }
    };

    // The token is a credential: show only enough of it to recognize
    // which one is stored, never the whole value.
    let hf_token_line: Line = if let Some(input) = &app.settings.hf_token_input {
        Line::from(vec![
            Span::raw("  HF token:       "),
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
                "  HF token:       {}",
                match &app.config.hf_token {
                    Some(token) => masked_token(token),
                    None => "not set (needed for gated models, e.g. pyannote)".into(),
                }
            ),
            row_style(SettingsRow::HfToken),
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
            format!(
                "  Input language: {}",
                language_label(app.config.language.as_deref())
            ),
            row_style(SettingsRow::Language),
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
            format!("  Split mode:     {}", app.config.split_mode.label()),
            row_style(SettingsRow::SplitMode),
        )),
        Line::raw(""),
        hf_token_line,
        Line::raw(""),
        Line::from(Span::styled(
            if app.settings.hf_token_input.is_some() {
                "  Paste your hf.co token (Enter saves, empty clears) — used for gated models"
            } else {
                "  ↑↓ select · Enter change · Esc close  (saved to ~/.config/transcribe-stt)"
            },
            Style::default().fg(DIM),
        )),
    ];

    // Size the modal to its content: a fixed percentage height used to
    // clip the bottom rows invisibly on short terminals. If even the
    // full content cannot fit, drop the blank separator rows first —
    // every setting stays reachable.
    let mut lines = lines;
    if lines.len() as u16 + 2 > frame.area().height {
        lines.retain(|l| l.width() != 0);
    }
    let area = centered_rect_rows(70, lines.len() as u16 + 2, frame.area());
    frame.render_widget(Clear, area);
    frame.render_widget(Paragraph::new(lines).block(block), area);
}

/// A stored token shown as `hf_…wxyz`: enough to recognize which token
/// is set without ever rendering the credential itself.
fn masked_token(token: &str) -> String {
    let tail: String = token
        .chars()
        .rev()
        .take(4)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("set (…{tail})")
}

pub(super) fn draw_input_languages(frame: &mut Frame, app: &App) {
    let cursor = app.settings.language_cursor.unwrap_or(0);
    let selected = app.config.language.as_deref().unwrap_or("auto");
    let mut lines = Vec::new();
    for (i, (code, label)) in INPUT_LANGUAGES.iter().enumerate() {
        let mark = if *code == selected { "●" } else { "○" };
        let description = if *code == "auto" {
            " (detect automatically)"
        } else {
            ""
        };
        lines.push(Line::from(Span::styled(
            format!("  {mark} {label}{description}"),
            if i == cursor {
                highlight_style()
            } else {
                Style::default()
            },
        )));
    }
    lines.push(Line::raw(""));
    for text in [
        "  Used for new file and microphone transcriptions.",
        "  Skips language detection where supported.",
        "  English-only models stay English; Voxtral uses Auto.",
        "  ↑↓ select · Enter save · Esc cancel",
    ] {
        lines.push(Line::from(Span::styled(text, Style::default().fg(DIM))));
    }
    let area = centered_rect_rows(76, lines.len() as u16 + 2, frame.area());
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT))
        .title(" input language ");
    frame.render_widget(Clear, area);
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
