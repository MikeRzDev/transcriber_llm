use anyhow::Result;
use clap::Parser;

fn main() {
    let code = match real_main() {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("Error: {e:#}");
            1
        }
    };
    use std::io::Write;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    // ggml registers Metal teardown in atexit handlers that can hang or
    // assert on macOS. Everything that needs cleanup (terminal state,
    // worker thread, whisper context) is already shut down explicitly,
    // so skip the C++ static destructors entirely.
    unsafe { libc::_exit(code) }
}

fn real_main() -> Result<()> {
    // Parsed at the process edge: clap prints help/usage errors and exits
    // before any whisper/Metal state exists.
    let args = transcribe_stt::cli::Args::parse();
    transcribe_stt::run(args)
}
