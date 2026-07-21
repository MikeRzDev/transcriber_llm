#!/usr/bin/env bash
# Check transcribe-stt's dependencies in priority order (Xcode CLT → Homebrew →
# Rust → cmake → ffmpeg), install anything missing, then compile and install
# the transcribe-stt binary. Models are downloaded from inside the app
# (s → Model management). Already-installed dependencies are skipped;
# safe to re-run.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: transcribe-stt requires macOS (Metal); this is $(uname -s)" >&2
  exit 1
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "warning: $(uname -m) detected — Metal acceleration needs an Apple Silicon Mac" >&2
fi

# 1. Xcode command line tools — compilers; everything below depends on them
if xcode-select -p >/dev/null 2>&1; then
  echo "✓ Xcode command line tools"
else
  echo "→ installing Xcode command line tools (accept the GUI prompt)…"
  xcode-select --install
  echo "re-run this script once that installation finishes."
  exit 1
fi

# 2. Homebrew — installs the rest
if command -v brew >/dev/null 2>&1; then
  echo "✓ Homebrew"
else
  echo "→ installing Homebrew…"
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  eval "$(/opt/homebrew/bin/brew shellenv)"
fi

# 3. Rust — existing but outdated toolchains fail the build, so check the
# version too (sysinfo needs rustc 1.95+; assumes rustc stays on major 1)
MIN_RUST_MINOR=95
rust_current() {
  command -v rustc >/dev/null 2>&1 || return 1
  [ "$(rustc --version | awk '{print $2}' | cut -d. -f2)" -ge "$MIN_RUST_MINOR" ]
}
if rust_current; then
  echo "✓ rust ($(rustc --version | awk '{print $2}'))"
elif command -v rustup >/dev/null 2>&1; then
  echo "→ rust older than 1.${MIN_RUST_MINOR} — running rustup update…"
  rustup update stable
elif command -v rustc >/dev/null 2>&1; then
  echo "→ rust older than 1.${MIN_RUST_MINOR} — upgrading via brew…"
  brew upgrade rust
else
  echo "→ installing rust…"
  brew install rust
fi

# 4–5. brew packages
brew_dep() {
  local formula="$1" cmd="$2" why="$3"
  if command -v "$cmd" >/dev/null 2>&1; then
    echo "✓ $formula"
  else
    echo "→ installing $formula — ${why}…"
    brew install "$formula"
  fi
}
brew_dep cmake cmake "builds whisper.cpp"
brew_dep ffmpeg ffmpeg "video files and exotic audio codecs"

# 6. compile and install the binary (into ~/.cargo/bin, reusing any
# existing release artifacts in ./target)
echo "→ building and installing transcribe-stt…"
cargo install --path "$ROOT" --target-dir "$ROOT/target"

echo
if command -v transcribe-stt >/dev/null 2>&1; then
  echo "installed: $(command -v transcribe-stt)"
  echo "run 'transcribe-stt', then press s → Model management to download a model"
else
  echo "installed to \$CARGO_HOME/bin (usually ~/.cargo/bin) — add it to your PATH:"
  echo '  export PATH="$HOME/.cargo/bin:$PATH"'
fi
