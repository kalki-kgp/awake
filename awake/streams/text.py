"""A byte stream from a file or a folder of files. Each example is a chunk of
seq_len bytes and the next-byte targets; the learner carries its hidden state from
chunk to chunk, so this is a genuinely sequential stream (batch size must be 1)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .base import Stream
from .. import CACHE_DIR

DEFAULT_TEXT = CACHE_DIR / "text" / "tinyshakespeare.txt"


class TextStream(Stream):
    name = "text"
    loss = "ce"
    sequential = True
    out_dim = 256
    x_dtype = np.int32
    y_dtype = np.int32

    def __init__(self, seed=0, path=None, seq_len=64, retention_bytes=65536, on_end="loop"):
        p = Path(path).expanduser() if path else DEFAULT_TEXT
        if not p.exists():
            raise FileNotFoundError(f"text stream source {p} not found; run `awake data shakespeare` "
                                    f"or set [stream] path to any text file or folder")
        files = sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p]
        files = [f for f in files if f.stat().st_size > seq_len + 1]
        if not files:
            raise FileNotFoundError(f"no usable text files under {p}")
        self.files = [str(f) for f in files]
        self.seq_len = int(seq_len)
        self.x_shape = self.y_shape = (self.seq_len,)
        self.retention_bytes = int(retention_bytes)
        self.on_end = on_end
        self.file_idx, self.pos, self.epoch = 0, 0, 0
        self._reset_pending = True
        self._load()
        self._ret = None

    def _load(self) -> None:
        self.buf = np.fromfile(self.files[self.file_idx], dtype=np.uint8)

    def next(self):
        L = self.seq_len
        while self.pos + L + 1 > len(self.buf):
            self.file_idx += 1
            self.pos = 0
            self._reset_pending = True
            if self.file_idx >= len(self.files):
                if self.on_end == "stop":
                    raise StopIteration
                self.file_idx = 0
                self.epoch += 1
            self._load()
        chunk = self.buf[self.pos: self.pos + L + 1].astype(np.int32)
        self.pos += L
        reset, self._reset_pending = self._reset_pending, False
        return chunk[:-1], chunk[1:], reset

    def retention(self):
        if self._ret is None:
            buf = np.fromfile(self.files[0], dtype=np.uint8)[: self.retention_bytes].astype(np.int32)
            n = (len(buf) - 1) // self.seq_len
            idx = np.arange(n)[:, None] * self.seq_len + np.arange(self.seq_len)[None, :]
            self._ret = (buf[idx], buf[idx + 1])
        return self._ret

    @property
    def task(self) -> int:
        return self.epoch

    def state(self) -> dict:
        return {"file_idx": self.file_idx, "pos": self.pos, "epoch": self.epoch,
                "reset_pending": self._reset_pending}

    def load_state(self, d: dict) -> None:
        self.file_idx = int(d["file_idx"]); self.pos = int(d["pos"]); self.epoch = int(d["epoch"])
        self._reset_pending = bool(d.get("reset_pending", True))
        self._load()

    def describe(self) -> str:
        total = sum(Path(f).stat().st_size for f in self.files)
        return f"text({len(self.files)} file(s), {total/1e6:.1f} MB, seq_len={self.seq_len})"
