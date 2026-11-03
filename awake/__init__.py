"""awake: an always-on learner for Apple Silicon.

One stream, one example at a time, weights updated on every step, a sleep phase
that consolidates and measures retention, checkpoints that let it run for months.
"""
from __future__ import annotations
import os
from pathlib import Path

__version__ = "0.1.0"
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = Path(os.environ.get("AWAKE_RUNS", REPO_ROOT / "runs"))
CACHE_DIR = Path(os.environ.get("AWAKE_CACHE", Path.home() / ".cache" / "awake"))
