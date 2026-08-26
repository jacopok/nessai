#!/usr/bin/env bash
# Launch the interactive Optuna dashboard for the replay-optimisation study.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
uv run --extra replay optuna-dashboard sqlite:///nessai_replay.sqlite3 "$@"
