from __future__ import annotations
from .base import Stream
from .scr import SCRStream
from .pmnist import PMNISTStream
from .text import TextStream

KINDS = {"scr": SCRStream, "pmnist": PMNISTStream, "text": TextStream}


def build_stream(cfg_stream, seed: int) -> Stream:
    kind = cfg_stream["kind"]
    if kind not in KINDS:
        raise ValueError(f"unknown stream kind {kind!r}; choose from {sorted(KINDS)}")
    kw = {k: v for k, v in cfg_stream.items() if k != "kind"}
    return KINDS[kind](seed=seed, **kw)
