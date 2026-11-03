"""Online Permuted MNIST: each task applies a fresh random pixel permutation; every
image is seen once, in a random order. Retention is measured on task 0's permutation."""
from __future__ import annotations
import gzip
import shutil
import urllib.request
from pathlib import Path
import numpy as np
from .base import Stream
from .. import CACHE_DIR

MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://raw.githubusercontent.com/fgnt/mnist/master/",
]
FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}


def _fetch(fname: str, cache: Path) -> Path:
    dest = cache / fname
    if dest.exists():
        return dest
    last = None
    for base in MIRRORS:
        try:
            tmp = dest.with_suffix(".part")
            with urllib.request.urlopen(base + fname, timeout=60) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            tmp.rename(dest)
            return dest
        except Exception as e:  # try the next mirror
            last = e
    raise RuntimeError(f"could not download {fname} from any mirror: {last}")


def load_mnist(cache_dir=None, download=True) -> dict[str, np.ndarray]:
    cache = Path(cache_dir) if cache_dir else CACHE_DIR / "mnist"
    npz = cache / "mnist.npz"
    if npz.exists():
        d = np.load(npz)
        return {k: d[k] for k in d.files}
    if not download:
        raise FileNotFoundError(f"MNIST not cached at {npz}; run `awake data mnist`")
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, fname in FILES.items():
        with gzip.open(_fetch(fname, cache)) as f:
            data = f.read()
        if key.endswith("_x"):
            n = int.from_bytes(data[4:8], "big")
            out[key] = np.frombuffer(data, np.uint8, offset=16).reshape(n, 784).copy()
        else:
            out[key] = np.frombuffer(data, np.uint8, offset=8).copy()
    np.savez_compressed(npz, **out)
    return out


class PMNISTStream(Stream):
    name = "pmnist"
    loss = "ce"
    sequential = False
    x_shape = (784,)
    y_shape = ()
    out_dim = 10
    y_dtype = np.int32

    def __init__(self, seed=0, samples_per_task=10_000, retention_samples=2000, cache_dir=None):
        d = load_mnist(cache_dir, download=False)
        self.X, self.Y = d["train_x"], d["train_y"].astype(np.int32)
        self.Xt, self.Yt = d["test_x"], d["test_y"].astype(np.int32)
        self.seed = int(seed)
        self.spt = min(int(samples_per_task), len(self.X))
        self.retention_samples = min(int(retention_samples), len(self.Xt))
        self._task = 0
        self.pos = 0
        self._setup_task()
        self._ret = None

    def _task_rng(self, task: int) -> np.random.Generator:
        return np.random.default_rng(self.seed * 1_000_003 + task + 17)

    def _setup_task(self) -> None:
        r = self._task_rng(self._task)
        self.perm = r.permutation(784)
        self.order = r.permutation(len(self.X))[: self.spt]

    def next(self):
        if self.pos >= self.spt:
            self._task += 1
            self.pos = 0
            self._setup_task()
        i = self.order[self.pos]
        self.pos += 1
        x = self.X[i][self.perm].astype(np.float32) / 255.0
        return x, self.Y[i], False

    def retention(self):
        if self._ret is None:
            perm0 = self._task_rng(0).permutation(784)
            n = self.retention_samples
            self._ret = (self.Xt[:n][:, perm0].astype(np.float32) / 255.0, self.Yt[:n])
        return self._ret

    @property
    def task(self) -> int:
        return self._task

    def state(self) -> dict:
        return {"task": self._task, "pos": self.pos}

    def load_state(self, d: dict) -> None:
        self._task = int(d["task"]); self.pos = int(d["pos"])
        self._setup_task()

    def describe(self) -> str:
        return f"pmnist(samples_per_task={self.spt})"
