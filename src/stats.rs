use std::time::{Duration, Instant};

use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

const REFRESH_EVERY: Duration = Duration::from_millis(1000);

/// Live resource usage of this process (whisper worker threads included),
/// refreshed at most once per second from the render loop.
pub struct ProcStats {
    system: System,
    pid: Pid,
    last_refresh: Instant,
    pub mem_bytes: u64,
    /// Raw sysinfo value: percent of a single core, so it can exceed 100
    /// on multi-core work; label() normalizes it to 0–100% of the machine
    pub cpu_percent: f32,
    pub core_count: usize,
}

impl ProcStats {
    pub fn new() -> Self {
        let mut stats = Self {
            system: System::new(),
            pid: Pid::from_u32(std::process::id()),
            // Trick refresh_if_due into running on the first call
            last_refresh: Instant::now() - REFRESH_EVERY,
            mem_bytes: 0,
            cpu_percent: 0.0,
            core_count: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
        };
        stats.refresh_if_due();
        stats
    }

    pub fn refresh_if_due(&mut self) {
        if self.last_refresh.elapsed() < REFRESH_EVERY {
            return;
        }
        self.last_refresh = Instant::now();
        self.system.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[self.pid]),
            true,
            ProcessRefreshKind::nothing().with_memory().with_cpu(),
        );
        if let Some(process) = self.system.process(self.pid) {
            self.mem_bytes = process.memory();
            self.cpu_percent = process.cpu_usage();
        }
    }

    pub fn label(&self) -> String {
        format_stats(self.mem_bytes, self.cpu_percent, self.core_count)
    }
}

fn format_stats(mem_bytes: u64, cpu_percent: f32, core_count: usize) -> String {
    let mem = crate::format::human_size(mem_bytes);
    let machine_pct = (cpu_percent / core_count.max(1) as f32).clamp(0.0, 100.0);
    format!("RAM {mem} · CPU {machine_pct:.0}%")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_formatting() {
        // per-core percentages normalize to a single 0–100% machine value
        assert_eq!(
            format_stats(3_400_000_000, 312.4, 14),
            "RAM 3.4 GB · CPU 22%"
        );
        assert_eq!(format_stats(52_000_000, 1.6, 8), "RAM 52 MB · CPU 0%");
        // full blast on every core caps at 100%, even if sysinfo overshoots
        assert_eq!(format_stats(52_000_000, 810.0, 8), "RAM 52 MB · CPU 100%");
    }

    #[test]
    fn reads_own_process() {
        let stats = ProcStats::new();
        // The test binary certainly uses some memory
        assert!(stats.mem_bytes > 1_000_000, "mem: {}", stats.mem_bytes);
        assert!(stats.core_count >= 1);
    }
}
