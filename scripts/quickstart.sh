#!/bin/zsh
# One-time setup + health check for a fresh Mac (M1 today, M5 later).
set -e
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo "install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
[ -d .venv ] || uv venv
uv pip install -e .
source .venv/bin/activate
awake selftest
awake bench
awake data shakespeare
echo
echo "Next: awake run configs/scr_cbp.toml --steps 1000000   (about 5 minutes; then: awake report scr_cbp --open)"
