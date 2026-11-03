from __future__ import annotations
import shutil
import urllib.request
from . import CACHE_DIR
from .streams.pmnist import load_mnist

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def fetch_shakespeare() -> None:
    dest = CACHE_DIR / "text" / "tinyshakespeare.txt"
    if dest.exists():
        print(f"already have {dest} ({dest.stat().st_size / 1e6:.2f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(SHAKESPEARE_URL, timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.rename(dest)
    print(f"downloaded {dest} ({dest.stat().st_size / 1e6:.2f} MB)")


def fetch_mnist() -> None:
    d = load_mnist(download=True)
    print(f"MNIST cached under {CACHE_DIR / 'mnist'}: train {d['train_x'].shape}, test {d['test_x'].shape}")
