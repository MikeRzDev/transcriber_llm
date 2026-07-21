use ratatui::layout::{Alignment, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::app::App;
use crate::ui::theme::{ACCENT, DIM};

pub(super) fn draw_header(frame: &mut Frame, area: Rect, app: &App) {
    let model = app
        .library
        .selected
        .as_ref()
        .map(|m| format!("{} ({})", m.name, m.size_human()))
        .unwrap_or_else(|| "no model — press m".into());
    let line = Line::from(vec![
        Span::styled(
            " transcribe-stt ",
            Style::default().fg(Color::Black).bg(ACCENT),
        ),
        Span::raw("  "),
        Span::styled(model, Style::default().fg(ACCENT)),
        Span::styled("  •  Metal GPU", Style::default().fg(DIM)),
        if app.config.diarize != crate::diarize::DiarizeStrategy::Off {
            let speakers = match app.resolved_diarize_method() {
                // tinydiarize's two-speaker assumption is fixed
                crate::diarize::DiarizeMethod::Tdrz => "2".into(),
                _ => app
                    .config
                    .diarize_speakers
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "auto".into()),
            };
            Span::styled(
                format!(
                    "  •  diarize: {}, speakers: {speakers}",
                    app.config.diarize.key()
                ),
                Style::default().fg(Color::Magenta),
            )
        } else {
            Span::raw("")
        },
    ]);
    frame.render_widget(Paragraph::new(line), area);

    // Live process resource usage, right-aligned over the same line
    let stats_style = if app.busy() {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default().fg(DIM)
    };
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            format!("{} ", app.stats.label()),
            stats_style,
        )))
        .alignment(Alignment::Right),
        area,
    );
}
