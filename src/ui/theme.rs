//! Shared visual vocabulary: accent colors, the spinner, and the one
//! highlight style every list and modal uses.

use ratatui::style::{Color, Modifier, Style};

pub const ACCENT: Color = Color::Cyan;
pub const DIM: Color = Color::DarkGray;

const SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// The spinner frame for this render tick.
pub fn spinner_frame(tick: usize) -> char {
    SPINNER[tick % SPINNER.len()]
}

/// Inverse-accent highlight used for every selected row.
pub fn highlight_style() -> Style {
    Style::default()
        .bg(ACCENT)
        .fg(Color::Black)
        .add_modifier(Modifier::BOLD)
}

/// One consistent color per diarized speaker index, shared by the
/// transcript pane and the naming dialog; cycles past the palette.
pub fn speaker_color(speaker: u8) -> Color {
    const PALETTE: [Color; 6] = [
        Color::Cyan,
        Color::Magenta,
        Color::Green,
        Color::Yellow,
        Color::Blue,
        Color::LightRed,
    ];
    PALETTE[speaker as usize % PALETTE.len()]
}
