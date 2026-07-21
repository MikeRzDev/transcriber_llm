//! Machine-memory budget checks: whether a model's runtime footprint
//! plausibly fits this Mac's unified memory. Used to badge oversized
//! models in the library and to warn before a download that could never
//! run here (`metal::malloc` aborts the job once weights + decode
//! buffers exceed the process's Metal working-set limit).

use std::sync::OnceLock;

/// Weights → runtime footprint envelope: inference needs the weights
/// plus KV/compute buffers, ~1.5× the file size in practice.
const OVERHEAD_NUM: u64 = 3;
const OVERHEAD_DEN: u64 = 2;

/// Metal caps a process's GPU working set well below physical RAM;
/// ~70% of unified memory is what a model can realistically claim.
const BUDGET_NUM: u64 = 7;
const BUDGET_DEN: u64 = 10;

/// Physical memory in bytes (`sysctl hw.memsize`), cached. 0 = unknown
/// (non-macOS or sysctl failure) — callers treat unknown as "fits" so
/// the check only ever blocks confident misfits.
pub fn total_ram_bytes() -> u64 {
    static RAM: OnceLock<u64> = OnceLock::new();
    *RAM.get_or_init(|| {
        std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0)
    })
}

/// Whether a model with `weights_bytes` of weights plausibly runs on
/// this machine. Unknown RAM or unknown size passes.
pub fn fits(weights_bytes: u64) -> bool {
    fits_in(weights_bytes, total_ram_bytes())
}

fn fits_in(weights_bytes: u64, ram: u64) -> bool {
    ram == 0
        || weights_bytes == 0
        || weights_bytes / OVERHEAD_DEN * OVERHEAD_NUM <= ram / BUDGET_DEN * BUDGET_NUM
}

/// The badge/refusal line for a model that fails [`fits`].
pub fn misfit_note(weights_bytes: u64) -> String {
    format!(
        "needs ~{} of unified memory · this Mac has {}",
        crate::format::human_size(weights_bytes / OVERHEAD_DEN * OVERHEAD_NUM),
        crate::format::human_size(total_ram_bytes()),
    )
}

/// Approximate bytes from a curated-catalog size label ("3.1 GB",
/// "488 MB"). Unrecognized labels give 0 — size unknown, passes `fits`.
pub fn parse_human_size(s: &str) -> u64 {
    let s = s.trim();
    let (number, unit) = match s.find(|c: char| c.is_ascii_alphabetic()) {
        Some(i) => (s[..i].trim(), s[i..].trim()),
        None => return 0,
    };
    let Ok(value) = number.parse::<f64>() else {
        return 0;
    };
    let scale = match unit.to_ascii_uppercase().as_str() {
        "GB" => 1e9,
        "MB" => 1e6,
        "KB" => 1e3,
        "B" => 1.0,
        _ => return 0,
    };
    (value * scale) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1_000_000_000;

    #[test]
    fn small_models_fit_everywhere() {
        assert!(fits_in(75_000_000, 8 * GB)); // tiny on an 8 GB Mac
        assert!(fits_in(3 * GB, 16 * GB)); // large-v3 on 16 GB
    }

    #[test]
    fn oversized_weights_are_rejected() {
        // 3.1 GB large-v3 needs ~4.7 GB, over an 8 GB Mac's ~5.6 GB
        // budget only once weights push past ~3.7 GB
        assert!(!fits_in(4 * GB, 8 * GB));
        assert!(!fits_in(30 * GB, 36 * GB));
    }

    #[test]
    fn unknowns_always_pass() {
        assert!(fits_in(0, 8 * GB));
        assert!(fits_in(100 * GB, 0));
    }

    #[test]
    fn parses_catalog_size_labels() {
        assert_eq!(parse_human_size("3.1 GB"), 3_100_000_000);
        assert_eq!(parse_human_size("488 MB"), 488_000_000);
        assert_eq!(parse_human_size("142M"), 0); // unknown unit form
        assert_eq!(parse_human_size(""), 0);
    }

    #[test]
    fn misfit_note_round_trip() {
        // parse → note math stays in human-size units
        let bytes = parse_human_size("10.0 GB");
        assert!(misfit_note(bytes).starts_with("needs ~15.0 GB"));
    }
}
