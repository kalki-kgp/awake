from __future__ import annotations
from dataclasses import dataclass


@dataclass
class LoopState:
    step: int = 0
    wall_s: float = 0.0
    sleeps: int = 0
    wake_pos: int = 0
    last_ckpt_step: int = 0
