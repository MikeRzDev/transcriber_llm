use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;

use crate::app::App;
use crate::ui::layout::centered_rect;
use crate::ui::theme::{highlight_style, DIM};

pub(super) fn draw_move_prompt(frame: &mut Frame, app: &App) {
    let Some(prompt) = &app.settings.move_prompt else {
        return;
    };
    let area = centered_rect(64, 30, frame.area());
    frame.render_widget(Clear, area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow))
        .title(" move models? ");

    let option = |label: &str, selected: bool| {
        if selected {
            Span::styled(format!("[ {label} ]"), highlight_style())
        } else {
            Span::styled(format!("[ {label} ]"), Style::default().fg(DIM))
        }
    };

    let lines = vec![
        Line::raw(""),
        Line::raw(format!(
            "  The previous folder still holds {} model(s).",
            prompt.count
        )),
        Line::raw("  Move them to the new models folder?"),
        Line::raw(""),
        Line::from(vec![
            Span::styled("  from: ", Style::default().fg(DIM)),
            Span::raw(prompt.from.display().to_string()),
        ]),
        Line::from(vec![
            Span::styled("  to:   ", Style::default().fg(DIM)),
            Span::raw(prompt.to.display().to_string()),
        ]),
        Line::raw(""),
        Line::from(vec![
            Span::raw("  "),
            option("Yes, move them", prompt.yes_selected),
            Span::raw("   "),
            option("No, leave them", !prompt.yes_selected),
        ]),
        Line::raw(""),
        Line::from(Span::styled(
            "  ←→ choose · Enter confirm · y / n shortcuts · Esc leaves them",
            Style::default().fg(DIM),
        )),
    ];

    frame.render_widget(Paragraph::new(lines).block(block), area);
}
