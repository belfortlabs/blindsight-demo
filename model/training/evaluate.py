"""
Evaluate a checkpoint on the TEST split (never trained on), with a breakdown.

  uv run evaluate.py                                   # runs/baseline/best.pt
  uv run evaluate.py runs/baseline/last.pt                # any other checkpoint

Breakdown lines: recall per class; 'person far (rejected)' are label-0 rows that contain a
person the model must NOT fire on; lighting rows use Wake Vision's lighting flags.

Next to the checkpoint it writes evaluate.json (the checkpoint, the configuration, the test
split size and every breakdown line; committed as the run's record, not read by code).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dataset"))   # wake_vision importable

from torch.utils.data import DataLoader  # noqa: E402

from wake_vision import IMG_SIZE, MemmapDS, cache_dir  # noqa: E402
from common import accuracy_breakdown, pick_device, predict_correct, print_breakdown  # noqa: E402
from personnet import CHANNELS, STRIDES, WEIGHTS, load  # noqa: E402

WORKERS = min(4, max(1, (os.cpu_count() or 4) - 2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", nargs="?", default=None, help="checkpoint to score (default: runs/baseline/best.pt)")
    a = p.parse_args()
    ckpt = Path(a.ckpt or WEIGHTS)

    device = pick_device()
    model = load(ckpt).to(device)
    ds = MemmapDS(cache_dir(), "test")
    rows = accuracy_breakdown(predict_correct(model, DataLoader(ds, batch_size=512, num_workers=WORKERS), device), ds)
    print_breakdown(rows, f"[eval] {ckpt} on test ({len(ds):,} rows; label 1 = near person)")
    json.dump(dict(checkpoint=ckpt.name, img_size=IMG_SIZE, channels=CHANNELS, strides=STRIDES,
                   split="test", rows=len(ds), accuracy=rows[0][1],
                   breakdown=[dict(name=n, accuracy=acc, n=k) for n, acc, k in rows]),
              open(ckpt.parent / "evaluate.json", "w"), indent=2)
    print(f"[eval] -> {ckpt.parent / 'evaluate.json'}", flush=True)
