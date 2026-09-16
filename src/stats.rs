use std::sync::{
    mpsc::{sync_channel, SyncSender},
    Arc, Mutex,
};
use std::time::{Duration, Instant};

use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

const REFRESH_EVERY: Duration = Duration::from_millis(1000);

/// Live resource usage of this process tree — whisper worker threads,
/// plus subprocess engines (the MLX Python child holds the model
/// weights, so counting only our own PID would hide most of the RAM a
/// job uses). Refreshed at most once per second from the render loop.
pub struct ProcStats {
    requests: SyncSender<()>,
    latest: Arc<Mutex<Option<(u64, f32)>>>,
    last_refresh: Instant,
    pub mem_bytes: u64,
    /// Raw sysinfo value: percent of a single core, so it can exceed 100
    /// on multi-core work; label() normalizes it to 0–100% of the machine
    pub cpu_percent: f32,
    pub core_count: usize,
}

impl ProcStats {
    pub fn new() -> Self {
        let mut system = System::new();
        let pid = Pid::from_u32(std::process::id());
        Self::with_sampler(move || {
            system.refresh_processes_specifics(
                ProcessesToUpdate::All,
                true,
                ProcessRefreshKind::nothing().with_memory().with_cpu(),
            );
            tally_tree(&system, pid)
        })
    }

    fn with_sampler(mut sample: impl FnMut() -> (u64, f32) + Send + 'static) -> Self {
        let (requests, receiver) = sync_channel(1);
        let latest = Arc::new(Mutex::new(None));
        let worker_latest = Arc::downgrade(&latest);
        std::thread::spawn(move || {
            while receiver.recv().is_ok() {
                if worker_latest.strong_count() == 0 {
                    break;
                }
                let result = sample();
                let Some(latest) = worker_latest.upgrade() else {
                    break;
                };
                if let Ok(mut slot) = latest.lock() {
                    *slot = Some(result);
                };
            }
        });
        let mut stats = Self {
            requests,
            latest,
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
        // Both operations are nonblocking: a slow OS process query must never
        // delay a streamed word or even contend on its result mutex in the UI.
        if let Ok(mut latest) = self.latest.try_lock() {
            if let Some((memory, cpu)) = latest.take() {
                self.mem_bytes = memory;
                self.cpu_percent = cpu;
            }
        }
        if self.last_refresh.elapsed() >= REFRESH_EVERY {
            let _ = self.requests.try_send(());
            self.last_refresh = Instant::now();
        }
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
        let mut stats = ProcStats::new();
        let deadline = Instant::now() + Duration::from_secs(2);
        while stats.mem_bytes == 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
            stats.refresh_if_due();
        }
        // The test binary certainly uses some memory
        assert!(stats.mem_bytes > 1_000_000, "mem: {}", stats.mem_bytes);
        assert!(stats.core_count >= 1);
    }

    #[test]
    fn slow_sampling_and_a_locked_result_do_not_block_ui_refresh() {
        let (release, waiting) = std::sync::mpsc::channel::<()>();
        let (entered, started) = std::sync::mpsc::channel();
        let stats = ProcStats::with_sampler(move || {
            let _ = entered.send(());
            let _ = waiting.recv();
            (1234567, 12.0)
        });
        started.recv_timeout(Duration::from_secs(1)).unwrap();
        let latest = stats.latest.clone();
        let guard = latest.lock().unwrap();
        let (done, result) = std::sync::mpsc::channel();
        let ui = std::thread::spawn(move || {
            let mut stats = stats;
            for _ in 0..10 {
                stats.last_refresh = Instant::now() - REFRESH_EVERY;
                stats.refresh_if_due();
            }
            let _ = done.send(stats);
        });
        let returned = result.recv_timeout(Duration::from_secs(1));
        drop(guard);
        drop(release);
        ui.join().unwrap();
        let mut stats = returned.expect("UI refresh blocked on the sampler or its mutex");
        let deadline = Instant::now() + Duration::from_secs(1);
        while stats.mem_bytes == 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
            stats.refresh_if_due();
        }
        assert_eq!(stats.mem_bytes, 1234567);
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
