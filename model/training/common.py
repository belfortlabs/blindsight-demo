"""
What train.py and evaluate.py share: the accelerator, and scoring a model on a cached split.

  pick_device()                        MPS on Apple silicon, CUDA on an NVIDIA GPU, else CPU
  predict_correct(model, dl, device)   one bool per dataset row: argmax == label
  accuracy_breakdown(correct, ds)      overall, recall per class, far-person rejection, lighting
  print_breakdown(rows, title)

The data side (the cache, MemmapDS, the label names) is ../../dataset/wake_vision.py; this file
only knows how to run a model over it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dataset"))   # wake_vision importable

import torch  # noqa: E402

from wake_vision import CLASS_NAMES, FLAGS, LIGHTING, MemmapDS, to_input  # noqa: E402


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def predict_correct(model, dl, device) -> torch.Tensor:
    """Bool tensor, one entry per dataset row, True where argmax == label."""
    model.eval()
    correct = torch.zeros(len(dl.dataset), dtype=torch.bool)
    for xb, yb, idx in dl:
        correct[idx] = model(to_input(xb, device)).argmax(1).cpu() == yb
    return correct


def accuracy_breakdown(correct: torch.Tensor, ds: MemmapDS):
    """[(name, acc, n)]: overall, recall per class, per distance reason, per lighting flag.
    Rows with label 0 that contain a far person: that line is the rejection rate."""
    y, d = torch.from_numpy(ds.Y), torch.from_numpy(ds.DIST)
    rows = [("overall", correct.float().mean().item(), len(correct))]
    for c in (1, 0):
        rows.append((f"{CLASS_NAMES[c]} (recall)", correct[y == c].float().mean().item(), int((y == c).sum())))
    for k, name in ((-1, "no person"), (2, "person far (rejected)")):
        m = d == k
        if m.any():
            rows.append((name, correct[m].float().mean().item(), int(m.sum())))
    for j, f in enumerate(FLAGS):
        if f not in LIGHTING:
            continue
        m = torch.from_numpy(ds.FL[:, j] == 1)
        if m.any():
            rows.append((f, correct[m].float().mean().item(), int(m.sum())))
    return rows


def print_breakdown(rows, title=""):
    if title:
        print(title)
    for name, acc, n in rows:
        print(f"  {name:30} {acc * 100:6.2f}%   (n={n:,})")
