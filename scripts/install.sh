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

# 6. MLX runtime — mlx-audio in an app-managed venv powers directory
# (MLX) models: Parakeet, Qwen3-ASR, Canary, Whisper-MLX, … The app also
# auto-installs this on first MLX use; doing it here front-loads the wait.
if [[ "$(uname -m)" == "arm64" ]]; then
  MLX_VENV="$HOME/Library/Application Support/transcribe-stt/mlx-venv"
  probe_mlx() {
    "$MLX_VENV/bin/python3" -c \
      "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('mlx_audio') else 1)" \
      2>/dev/null
  }
  if probe_mlx; then
    echo "✓ mlx-audio runtime"
  else
    BASE_PY="$(command -v python3 || true)"
    py_ok() {
      [[ -n "$BASE_PY" ]] && "$BASE_PY" -c \
        "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null
    }
    if ! py_ok; then
      echo "→ installing python — runs the MLX engine (mlx-audio)…"
      brew install python
      BASE_PY="$(brew --prefix)/bin/python3"
    fi
    echo "→ installing mlx-audio runtime into $MLX_VENV…"
    "$BASE_PY" -m venv "$MLX_VENV"
    "$MLX_VENV/bin/python3" -m pip install --quiet --upgrade pip
    "$MLX_VENV/bin/python3" -m pip install --quiet mlx-audio
    probe_mlx && echo "✓ mlx-audio runtime"
  fi
fi

# 7. compile and install the binary (into ~/.cargo/bin, reusing any
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
