use std::time::{Duration, Instant};

use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

const REFRESH_EVERY: Duration = Duration::from_millis(1000);

/// Live resource usage of this process tree — whisper worker threads,
/// plus subprocess engines (the MLX Python child holds the model
/// weights, so counting only our own PID would hide most of the RAM a
/// job uses). Refreshed at most once per second from the render loop.
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
        // All processes, not just our PID: descendants (the MLX/pip
        // children) can only be found by walking parent links.
        self.system.refresh_processes_specifics(
            ProcessesToUpdate::All,
            true,
            ProcessRefreshKind::nothing().with_memory().with_cpu(),
        );
        (self.mem_bytes, self.cpu_percent) = tally_tree(&self.system, self.pid);
    }

    pub fn label(&self) -> String {
        format_stats(self.mem_bytes, self.cpu_percent, self.core_count)
    }
}

impl Default for ProcStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Does `pid`'s parent chain reach `root`? Depth-capped in case of a
/// stale/cyclic parent link in the snapshot.
fn is_descendant(system: &System, pid: Pid, root: Pid) -> bool {
    let mut current = pid;
    for _ in 0..32 {
        let Some(parent) = system.process(current).and_then(|p| p.parent()) else {
            return false;
        };
        if parent == root {
            return true;
        }
        current = parent;
    }
    false
}

/// Memory and CPU of `root` plus every live descendant.
fn tally_tree(system: &System, root: Pid) -> (u64, f32) {
    let mut mem = 0u64;
    let mut cpu = 0.0f32;
    if let Some(process) = system.process(root) {
        mem = process.memory();
        cpu = process.cpu_usage();
    }
    for (pid, process) in system.processes() {
        if *pid != root && is_descendant(system, *pid, root) {
            mem += process.memory();
            cpu += process.cpu_usage();
        }
    }
    (mem, cpu)
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

    #[test]
    fn child_processes_count_toward_the_tally() {
        let mut child = std::process::Command::new("sleep")
            .arg("10")
            .stdin(std::process::Stdio::null())
            .spawn()
            .unwrap();

        let mut system = System::new();
        system.refresh_processes_specifics(
            ProcessesToUpdate::All,
            true,
            ProcessRefreshKind::nothing().with_memory().with_cpu(),
        );
        let root = Pid::from_u32(std::process::id());
        let child_pid = Pid::from_u32(child.id());

        assert!(
            is_descendant(&system, child_pid, root),
            "spawned child must be found via parent links"
        );
        let (tree_mem, _) = tally_tree(&system, root);
        let own_mem = system.process(root).map(|p| p.memory()).unwrap_or(0);
        assert!(tree_mem >= own_mem);

        let _ = child.kill();
        let _ = child.wait();
    }
}
