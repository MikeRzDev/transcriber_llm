#!/bin/sh
# Launch from this checkout, optionally using an external model library.
# Usage: scripts/run-realtime.sh [models-directory] [extra CLI options...]
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_dir"
if [ -x "$project_dir/.venv-realtime/bin/python3" ]; then
    export TRANSCRIBE_STT_MLX_VENV="$project_dir/.venv-realtime"
fi
if ! command -v cmake >/dev/null 2>&1 && [ -x "$project_dir/.venv-realtime/bin/cmake" ]; then
    export CMAKE="$project_dir/.venv-realtime/bin/cmake"
fi
if [ "$#" -gt 0 ] && [ -d "$1" ]; then
    live_models_dir=$1
    shift
    exec cargo run -- --realtime --models-dir "$live_models_dir" "$@"
fi
exec cargo run -- --realtime "$@"
