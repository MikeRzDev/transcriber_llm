#!/bin/sh
# Build an optimized executable and stage it in the project's binary/ folder.
# Can be invoked from any working directory.
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$project_dir"

if ! command -v cargo >/dev/null 2>&1; then
    echo "error: Cargo is required. Install Rust or run scripts/install.sh first." >&2
    exit 1
fi

# Also support the isolated environment used for this checkout's MLX tests.
if [ -z "${CMAKE:-}" ] && ! command -v cmake >/dev/null 2>&1; then
    if [ -x "$project_dir/.venv-realtime/bin/cmake" ]; then
        export CMAKE="$project_dir/.venv-realtime/bin/cmake"
    else
        echo "error: CMake is required to build whisper.cpp. Install it or set CMAKE." >&2
        exit 1
    fi
fi

echo "Building transcribe-stt (release)…"
cargo build --release --locked --bin transcribe-stt --target-dir "$project_dir/target"

mkdir -p "$project_dir/binary"
staged_binary=$(mktemp "$project_dir/binary/.transcribe-stt.XXXXXX")
trap 'rm -f "$staged_binary"' EXIT
trap 'exit 1' HUP INT TERM
cp "$project_dir/target/release/transcribe-stt" "$staged_binary"
chmod 755 "$staged_binary"
mv -f "$staged_binary" "$project_dir/binary/transcribe-stt"

echo "Built: $project_dir/binary/transcribe-stt"
